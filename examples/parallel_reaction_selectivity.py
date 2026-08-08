"""Optimize temperature for two competing Arrhenius batch reactions.

Feed A reacts irreversibly through the parallel network A -> B and A -> C,
where B is the desired product.  The unwanted channel has the larger
activation energy, so aggressive heating improves conversion but erodes
selectivity.  A bounded heating/cooling ramp reaches a prescribed conversion
by the batch deadline and returns the reactor to its charging temperature at
zero ramp rate.

For ``i`` in ``{B, C}``, ``k_i(T) = A_i*exp(-E_i/(R*T))`` and
``c_A_dot = -(k_B + k_C)*c_A``, ``c_B_dot = k_B*c_A``, and
``c_C_dot = k_C*c_A``.  At fixed terminal conversion, minimizing ``c_C`` alone
would maximize desired-product selectivity. The implemented objective adds a
small temperature-ramp regularizer, so it makes that trade-off explicit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import cumulative_trapezoid, trapezoid

from pockit.lobatto import System, linear_guess
from pockit.optimizer import ipopt

if __package__:
    from ._plotting import (
        COLORS,
        configure_matplotlib,
        parse_plot_arguments,
        require_finite,
        save_or_show,
        style_axes,
    )
else:
    from _plotting import (
        COLORS,
        configure_matplotlib,
        parse_plot_arguments,
        require_finite,
        save_or_show,
        style_axes,
    )


GAS_CONSTANT = 8.314  # J/(mol K)
DESIRED_PREEXPONENTIAL = 2.0e8  # 1/min
DESIRED_ACTIVATION_ENERGY = 60_000.0  # J/mol
UNWANTED_PREEXPONENTIAL = 5.0e10  # 1/min
UNWANTED_ACTIVATION_ENERGY = 75_000.0  # J/mol
MIN_TEMPERATURE = 298.15  # K
MAX_TEMPERATURE = 343.15  # K
TEMPERATURE_SPAN = MAX_TEMPERATURE - MIN_TEMPERATURE
MAX_TEMPERATURE_RAMP = 2.0  # K/min
MAX_TEMPERATURE_RAMP_ACCELERATION = 0.35  # K/min^2
HORIZON = 90.0  # min
TARGET_REACTANT = 0.08  # mol/L
INITIAL_REACTANT = 1.0  # mol/L
RAMP_WEIGHT = 2.0e-4  # mol/(L min), after normalizing the ramp
RAMP_ACCELERATION_RELATIVE_WEIGHT = 0.15
DENSE_CHECK_POINTS = 6_001


def _rate_constant(temperature, preexponential, activation_energy):
    return preexponential * sp.exp(-activation_energy / (GAS_CONSTANT * temperature))


def _rate_constants_array(temperature: np.ndarray):
    desired = DESIRED_PREEXPONENTIAL * np.exp(
        -DESIRED_ACTIVATION_ENERGY / (GAS_CONSTANT * temperature)
    )
    unwanted = UNWANTED_PREEXPONENTIAL * np.exp(
        -UNWANTED_ACTIVATION_ENERGY / (GAS_CONSTANT * temperature)
    )
    return desired, unwanted


def build_problem():
    """Return the selective parallel-reaction batch problem."""
    system = System(["desired_final", "unwanted_final"])
    desired_final, unwanted_final = system.s
    phase = system.new_phase(
        [
            "reactant_concentration",
            "desired_concentration",
            "unwanted_concentration",
            "scaled_temperature",
            "temperature_ramp",
        ],
        ["temperature_ramp_acceleration"],
    )
    reactant, desired, unwanted, scaled_temperature, temperature_ramp = phase.x
    (temperature_ramp_acceleration,) = phase.u
    temperature = MIN_TEMPERATURE + TEMPERATURE_SPAN * scaled_temperature
    desired_rate_constant = _rate_constant(
        temperature, DESIRED_PREEXPONENTIAL, DESIRED_ACTIVATION_ENERGY
    )
    unwanted_rate_constant = _rate_constant(
        temperature, UNWANTED_PREEXPONENTIAL, UNWANTED_ACTIVATION_ENERGY
    )
    desired_rate = desired_rate_constant * reactant
    unwanted_rate = unwanted_rate_constant * reactant

    phase.set_dynamics(
        [
            -desired_rate - unwanted_rate,
            desired_rate,
            unwanted_rate,
            temperature_ramp / TEMPERATURE_SPAN,
            temperature_ramp_acceleration,
        ]
    )
    phase.set_integral(
        [
            (temperature_ramp / MAX_TEMPERATURE_RAMP) ** 2
            + RAMP_ACCELERATION_RELATIVE_WEIGHT
            * (temperature_ramp_acceleration / MAX_TEMPERATURE_RAMP_ACCELERATION) ** 2
        ]
    )
    phase.set_phase_constraint(
        [
            reactant,
            desired,
            unwanted,
            scaled_temperature,
            temperature_ramp,
            temperature_ramp_acceleration,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            -MAX_TEMPERATURE_RAMP,
            -MAX_TEMPERATURE_RAMP_ACCELERATION,
        ],
        [
            INITIAL_REACTANT,
            INITIAL_REACTANT,
            INITIAL_REACTANT,
            1.0,
            MAX_TEMPERATURE_RAMP,
            MAX_TEMPERATURE_RAMP_ACCELERATION,
        ],
    )
    phase.set_boundary_condition(
        [INITIAL_REACTANT, 0.0, 0.0, 0.0, 0.0],
        [TARGET_REACTANT, desired_final, unwanted_final, 0.0, 0.0],
        0.0,
        HORIZON,
    )
    # Piecewise-linear temperature ramps preserve actuator bounds between
    # nodes and resolve the smooth Arrhenius kinetics over the full batch.
    phase.set_discretization(200, 2)

    system.set_phase([phase])
    system.set_objective(unwanted_final + RAMP_WEIGHT * phase.I[0])
    system.set_system_constraint(
        [desired_final, unwanted_final], [0.0, 0.0], [1.0, 1.0]
    )
    return system, phase


def initial_guess(phase):
    """Return a smooth heat-hold-cool profile and mass-conserving guess."""
    guess = linear_guess(phase, 0.0)
    state_time = guess.t_x
    scaled_temperature = 0.46 * np.sin(np.pi * state_time / HORIZON) ** 2
    temperature = MIN_TEMPERATURE + TEMPERATURE_SPAN * scaled_temperature
    desired_k, unwanted_k = _rate_constants_array(temperature)
    cumulative_rate = np.concatenate(
        ([0.0], cumulative_trapezoid(desired_k + unwanted_k, state_time))
    )
    reactant = np.exp(-cumulative_rate)
    desired = np.concatenate(
        ([0.0], cumulative_trapezoid(desired_k * reactant, state_time))
    )
    unwanted = 1.0 - reactant - desired
    guess.x[0] = reactant
    guess.x[1] = desired
    guess.x[2] = unwanted
    guess.x[3] = scaled_temperature
    guess.x[4] = (
        0.46
        * TEMPERATURE_SPAN
        * np.pi
        / HORIZON
        * np.sin(2.0 * np.pi * state_time / HORIZON)
    )

    control_time = guess.t_u
    guess.u[0] = (
        0.46
        * TEMPERATURE_SPAN
        * 2.0
        * np.pi**2
        / HORIZON**2
        * np.cos(2.0 * np.pi * control_time / HORIZON)
    )
    desired_final = max(1.0 - reactant[-1] - unwanted[-1], 0.0)
    return [guess, [desired_final, max(unwanted[-1], 0.0)]]


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    phase_solution, static = solution
    time = np.linspace(phase_solution.t_0, phase_solution.t_f, DENSE_CHECK_POINTS)
    states = np.vstack(
        [phase_solution.V_x(time) @ component for component in phase_solution.x]
    )
    temperature = MIN_TEMPERATURE + TEMPERATURE_SPAN * states[3]
    temperature_ramp = states[4]
    temperature_ramp_acceleration = phase_solution.V_u(time) @ phase_solution.u[0]
    return (
        time,
        states,
        temperature,
        temperature_ramp,
        temperature_ramp_acceleration,
        np.asarray(static),
    )


def solve_problem(system, guess):
    """Solve the batch and verify analytical conversion and species balances."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1800, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    (
        time,
        states,
        temperature,
        temperature_ramp,
        temperature_ramp_acceleration,
        static,
    ) = _dense_solution(solution)
    reactant, desired, unwanted, scaled_temperature, _ramp_state = states
    desired_k, unwanted_k = _rate_constants_array(temperature)
    cumulative_total_rate = np.concatenate(
        ([0.0], cumulative_trapezoid(desired_k + unwanted_k, time))
    )
    analytical_reactant = INITIAL_REACTANT * np.exp(-cumulative_total_rate)
    require_finite(
        time=time,
        states=states,
        temperature=temperature,
        temperature_ramp=temperature_ramp,
        temperature_ramp_acceleration=temperature_ramp_acceleration,
        static_values=static,
        objective=info["obj_val"],
        desired_rate_constant=desired_k,
        unwanted_rate_constant=unwanted_k,
        analytical_reactant=analytical_reactant,
    )
    analytical_error = float(np.max(np.abs(reactant - analytical_reactant)))
    desired_balance = abs(
        float(desired[-1] - desired[0] - trapezoid(desired_k * reactant, time))
    )
    unwanted_balance = abs(
        float(unwanted[-1] - unwanted[0] - trapezoid(unwanted_k * reactant, time))
    )
    conservation_error = float(
        np.max(np.abs(reactant + desired + unwanted - INITIAL_REACTANT))
    )
    static_endpoint_error = max(
        abs(float(desired[-1] - static[0])),
        abs(float(unwanted[-1] - static[1])),
    )
    prescribed_endpoint_error = max(
        abs(float(reactant[0] - INITIAL_REACTANT)),
        abs(float(reactant[-1] - TARGET_REACTANT)),
        abs(float(temperature[0] - MIN_TEMPERATURE)),
        abs(float(temperature[-1] - MIN_TEMPERATURE)),
        abs(float(temperature_ramp[0])),
        abs(float(temperature_ramp[-1])),
    )
    path_violation = max(
        float(np.max(-states[:3])),
        float(np.max(states[:3] - INITIAL_REACTANT)),
        float(np.max(-scaled_temperature)),
        float(np.max(scaled_temperature - 1.0)),
        float(np.max(np.abs(temperature_ramp) - MAX_TEMPERATURE_RAMP)),
        float(
            np.max(
                np.abs(temperature_ramp_acceleration)
                - MAX_TEMPERATURE_RAMP_ACCELERATION
            )
        ),
        0.0,
    )
    temperature_balance = abs(
        float(temperature[-1] - temperature[0] - trapezoid(temperature_ramp, time))
    )
    if (
        path_violation > 3e-6
        or static_endpoint_error > 2e-7
        or prescribed_endpoint_error > 2e-7
    ):
        raise RuntimeError("the batch solution violates a bound or endpoint")
    if analytical_error > 3e-5 or conservation_error > 2e-5:
        raise RuntimeError("the batch solution failed its analytical mass check")
    if max(desired_balance, unwanted_balance) > 2e-5:
        raise RuntimeError("the dense reaction-rate balance is inconsistent")
    if temperature_balance > 2e-5:
        raise RuntimeError("the dense reactor-temperature balance is inconsistent")

    converted = INITIAL_REACTANT - reactant[-1]
    selectivity = desired[-1] / unwanted[-1]
    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"final desired-product yield: {desired[-1] / INITIAL_REACTANT:.6f}")
    print(f"desired/unwanted selectivity: {selectivity:.6f}")
    print(f"conversion: {converted / INITIAL_REACTANT:.6f}")
    print(f"peak temperature: {np.max(temperature):.6f} K")
    print(f"maximum analytical reactant error: {analytical_error:.3e} mol/L")
    print(f"maximum species-conservation error: {conservation_error:.3e} mol/L")
    print(
        "maximum dense reaction-rate balance error: "
        f"{max(desired_balance, unwanted_balance):.3e} mol/L"
    )
    print(f"dense temperature-balance error: {temperature_balance:.3e} K")
    reported_path_violation = max(0.0, path_violation)
    print(f"maximum dense path-bound violation: {reported_path_violation:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot concentrations, temperature, its ramp, and both kinetic rates."""
    configure_matplotlib()
    (
        time,
        states,
        temperature,
        temperature_ramp,
        _temperature_ramp_acceleration,
        _static,
    ) = _dense_solution(solution)
    reactant, desired, unwanted, _scaled_temperature, _ramp_state = states
    desired_k, unwanted_k = _rate_constants_array(temperature)

    fig, axes = plt.subplots(
        3, 1, figsize=(8.4, 7.8), sharex=True, layout="constrained"
    )
    axes[0].plot(time, reactant, color=COLORS["black"], label="Reactant A")
    axes[0].plot(time, desired, color=COLORS["green"], label="Desired B")
    axes[0].plot(time, unwanted, color=COLORS["vermillion"], label="Unwanted C")
    axes[0].set_ylabel("Concentration [mol/L]")
    axes[0].legend(ncol=3)

    axes[1].plot(time, temperature, color=COLORS["orange"], label="Reactor temperature")
    temperature_ramp_axis = axes[1].twinx()
    temperature_ramp_axis.plot(
        time,
        temperature_ramp,
        color=COLORS["purple"],
        alpha=0.8,
        label="Temperature ramp",
    )
    axes[1].set_ylabel("Temperature [K]")
    temperature_ramp_axis.set_ylabel("Ramp [K/min]")
    lines = axes[1].lines + temperature_ramp_axis.lines
    axes[1].legend(lines, [line.get_label() for line in lines], ncol=2)

    axes[2].plot(time, desired_k, color=COLORS["green"], label=r"Desired $k_B$")
    axes[2].plot(time, unwanted_k, color=COLORS["vermillion"], label=r"Unwanted $k_C$")
    axes[2].set_ylabel("Rate constant [1/min]")
    axes[2].set_xlabel("Batch time [min]")
    axes[2].legend(ncol=2)

    style_axes(axes)
    temperature_ramp_axis.grid(False)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "parallel_reaction_selectivity_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
