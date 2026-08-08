"""Optimize a catalyst-switching batch reactor with two process phases.

Phase 1 converts feed A into intermediate B. After one catalyst change,
phase 2 converts B into product P. The switch time and temperature-intensity
profiles are optimized together, with concentrations continuous at the switch.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pockit.lobatto import System, linear_guess
from pockit.optimizer import ipopt

if __package__:
    from ._plotting import (
        configure_matplotlib,
        parse_plot_arguments,
        require_finite,
        save_or_show,
        style_axes,
    )
else:
    from _plotting import (
        configure_matplotlib,
        parse_plot_arguments,
        require_finite,
        save_or_show,
        style_axes,
    )

BATCH_TIME = 2.0
YIELD = 0.92
EFFORT_WEIGHT = 0.15
DENSE_CHECK_POINTS = 2_001


def build_problem():
    """Return the two-phase reactor system and its phases."""
    names = [
        "a_switch",
        "b_switch",
        "p_switch",
        "a_final",
        "b_final",
        "p_final",
        "t_switch",
    ]
    system = System(names)
    a_s, b_s, p_s, a_f, b_f, p_f, t_s = system.s

    phase_1 = system.new_phase(["a_1", "b_1", "p_1"], ["temperature_1"])
    a_1, _, _ = phase_1.x
    (temperature_1,) = phase_1.u
    rate_1 = (0.20 + 1.80 * temperature_1) * a_1
    phase_1.set_dynamics([-rate_1, rate_1, 0.0])
    phase_1.set_integral([temperature_1**2])
    phase_1.set_phase_constraint([temperature_1], [0.0], [1.0])
    phase_1.set_boundary_condition([1.0, 0.0, 0.0], [a_s, b_s, p_s], 0.0, t_s)
    # Linear Lobatto interpolants preserve the bounded, monotone chemistry
    # between nodes without increasing the number of decision points.
    phase_1.set_discretization(24, 2)

    phase_2 = system.new_phase(["a_2", "b_2", "p_2"], ["temperature_2"])
    _, b_2, _ = phase_2.x
    (temperature_2,) = phase_2.u
    rate_2 = (0.10 + 2.40 * temperature_2) * b_2
    phase_2.set_dynamics([0.0, -rate_2, YIELD * rate_2])
    phase_2.set_integral([temperature_2**2])
    phase_2.set_phase_constraint([temperature_2], [0.0], [1.0])
    phase_2.set_boundary_condition([a_s, b_s, p_s], [a_f, b_f, p_f], t_s, BATCH_TIME)
    phase_2.set_discretization(24, 2)

    system.set_phase([phase_1, phase_2])
    system.set_objective(-p_f + EFFORT_WEIGHT * (phase_1.I[0] + phase_2.I[0]))
    system.set_system_constraint(
        [a_s, b_s, p_s, a_f, b_f, p_f, t_s],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.80],
    )
    return system, (phase_1, phase_2)


def initial_guess(phases):
    """Return phase guesses and shared switch/final values."""
    phase_1, phase_2 = phases
    switch = np.array([0.40, 0.60, 0.00])
    final = np.array([0.40, 0.08, 0.48])
    switch_time = 0.80

    guess_1 = linear_guess(phase_1, 0.0)
    tau_1 = guess_1.t_x / guess_1.t_f
    for i, (start, end) in enumerate(zip([1.0, 0.0, 0.0], switch)):
        guess_1.x[i] = start + (end - start) * tau_1
    guess_1.u[0] = 0.60
    guess_1.t_f = switch_time

    guess_2 = linear_guess(phase_2, 0.0)
    guess_2.t_0 = switch_time
    guess_2.t_f = BATCH_TIME
    tau_2 = (guess_2.t_x - switch_time) / (BATCH_TIME - switch_time)
    for i, (start, end) in enumerate(zip(switch, final)):
        guess_2.x[i] = start + (end - start) * tau_2
    guess_2.u[0] = 0.60

    static_guess = [*switch, *final, switch_time]
    return [guess_1, guess_2, static_guess]


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def solve_problem(system, guess):
    """Solve for the catalyst switch and both temperature profiles."""
    result, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1200, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)
    phase_1, phase_2, static = result

    dense_phases = []
    for phase in (phase_1, phase_2):
        time = np.linspace(phase.t_0, phase.t_f, DENSE_CHECK_POINTS)
        concentration = np.vstack(
            [phase.V_x(time) @ component for component in phase.x]
        )
        control = np.vstack([phase.V_u(time) @ component for component in phase.u])
        dense_phases.append((time, concentration, control))
    require_finite(
        phase_1_time=dense_phases[0][0],
        phase_1_states=dense_phases[0][1],
        phase_1_controls=dense_phases[0][2],
        phase_2_time=dense_phases[1][0],
        phase_2_states=dense_phases[1][1],
        phase_2_controls=dense_phases[1][2],
        static_values=static,
        objective=info["obj_val"],
    )

    continuity_error = max(abs(phase_1.x[i][-1] - phase_2.x[i][0]) for i in range(3))
    if continuity_error > 2e-6:
        raise RuntimeError("phase concentrations are not continuous")

    time_averages = []
    path_violation = 0.0
    for phase, (time, concentration, control) in zip((phase_1, phase_2), dense_phases):
        temperature = control[0]
        time_average = np.sum((temperature[:-1] + temperature[1:]) * np.diff(time)) / (
            2.0 * (phase.t_f - phase.t_0)
        )
        time_averages.append(float(time_average))
        path_violation = max(
            path_violation,
            float(np.max(-concentration)),
            float(np.max(concentration - 1.0)),
            float(np.max(-temperature)),
            float(np.max(temperature - 1.0)),
        )
    if path_violation > 2e-6:
        raise RuntimeError(f"dense path-bound violation: {path_violation:.3e}")

    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"catalyst switch time: {static[6]:.6f}")
    print(f"final product fraction: {static[5]:.6f}")
    print(f"phase 1 time-average temperature intensity: {time_averages[0]:.6f}")
    print(f"phase 2 time-average temperature intensity: {time_averages[1]:.6f}")
    print(f"maximum dense path-bound violation: {max(path_violation, 0.0):.3e}")
    return phase_1, phase_2, static


def plot_solution(result, *, save: str | Path | None = None, show: bool = True):
    """Plot concentrations and temperature intensity across both phases."""
    phase_1, phase_2, static = result
    configure_matplotlib()
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 6.2), sharex=True, layout="constrained"
    )
    colors = ["#0072B2", "#D55E00", "#009E73"]
    labels = ["Feed A", "Intermediate B", "Product P"]
    for i, (color, label) in enumerate(zip(colors, labels)):
        axes[0].plot(phase_1.t_x, phase_1.x[i], color=color, label=label)
        axes[0].plot(phase_2.t_x, phase_2.x[i], color=color)
    axes[0].set_ylabel("Concentration fraction")
    axes[0].legend(ncol=3)

    axes[1].plot(phase_1.t_u, phase_1.u[0], color="#CC79A7", label="Phase 1")
    axes[1].plot(phase_2.t_u, phase_2.u[0], color="#E69F00", label="Phase 2")
    axes[1].set_ylabel("Temperature intensity")
    axes[1].set_xlabel("Batch time [h]")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].legend()
    for axis in axes:
        axis.axvline(static[6], color="#555555", linestyle="--", linewidth=1.2)
    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "multiphase_batch_reactor_solution.png"
    )
    system, phases = build_problem()
    guess = initial_guess(phases)
    result = solve_problem(system, guess)
    plot_solution(result, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
