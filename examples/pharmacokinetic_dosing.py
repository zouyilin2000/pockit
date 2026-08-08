"""Design a loading-to-maintenance intravenous infusion in a PK/PD model.

The two-compartment pharmacokinetic model includes an effect-site delay. During
a 22-hour loading phase, the infusion pump rate and slew rate are states while
pump jerk is the control, so the optimized regimen is smooth. A second two-hour
phase holds the analytically derived maintenance rate and verifies that every
PK/PD state remains at equilibrium. This educational model is not a clinical
dosing recommendation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

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


HORIZON = 24.0  # h
MAINTENANCE_DURATION = 2.0  # h
LOADING_END = HORIZON - MAINTENANCE_DURATION
CENTRAL_VOLUME = 20.0  # L
CLEARANCE = 3.0  # L/h
TRANSFER_TO_PERIPHERAL = 0.25  # 1/h
TRANSFER_TO_CENTRAL = 0.15  # 1/h
EFFECT_EQUILIBRATION = 0.50  # 1/h
TARGET_EFFECT_CONCENTRATION = 2.0  # mg/L
MAX_CENTRAL_CONCENTRATION = 6.0  # mg/L
MAX_EFFECT_CONCENTRATION = 4.0  # mg/L
MAX_INFUSION_RATE = 50.0  # mg/h
MAX_INFUSION_SLEW = 20.0  # mg/h^2
MAX_INFUSION_JERK = 20.0  # mg/h^3
INFUSION_WEIGHT = 0.02
SLEW_WEIGHT = 0.002
JERK_WEIGHT = 0.0002
STEADY_CENTRAL_AMOUNT = CENTRAL_VOLUME * TARGET_EFFECT_CONCENTRATION
STEADY_PERIPHERAL_AMOUNT = (
    TRANSFER_TO_PERIPHERAL / TRANSFER_TO_CENTRAL * STEADY_CENTRAL_AMOUNT
)
MAINTENANCE_INFUSION_RATE = CLEARANCE * TARGET_EFFECT_CONCENTRATION
DENSE_CHECK_POINTS = 4_001


def build_problem():
    """Return the loading and steady-maintenance PK/PD phases."""
    system = System(0)
    loading = system.new_phase(
        [
            "central_amount",
            "peripheral_amount",
            "effect_concentration",
            "infusion_rate",
            "infusion_slew_rate",
        ],
        ["infusion_jerk"],
    )
    central, peripheral, effect, infusion, infusion_slew = loading.x
    (infusion_jerk,) = loading.u

    elimination_rate = CLEARANCE / CENTRAL_VOLUME
    central_concentration = central / CENTRAL_VOLUME
    loading.set_dynamics(
        [
            infusion
            - (elimination_rate + TRANSFER_TO_PERIPHERAL) * central
            + TRANSFER_TO_CENTRAL * peripheral,
            TRANSFER_TO_PERIPHERAL * central - TRANSFER_TO_CENTRAL * peripheral,
            EFFECT_EQUILIBRATION * (central_concentration - effect),
            infusion_slew,
            infusion_jerk,
        ]
    )
    tracking_error = (effect / TARGET_EFFECT_CONCENTRATION - 1.0) ** 2
    infusion_effort = INFUSION_WEIGHT * (infusion / MAX_INFUSION_RATE) ** 2
    slew_effort = SLEW_WEIGHT * (infusion_slew / MAX_INFUSION_SLEW) ** 2
    jerk_effort = JERK_WEIGHT * (infusion_jerk / MAX_INFUSION_JERK) ** 2
    loading.set_integral([tracking_error + infusion_effort + slew_effort + jerk_effort])
    loading.set_phase_constraint(
        [
            central,
            peripheral,
            effect,
            central_concentration,
            infusion,
            infusion_slew,
            infusion_jerk,
        ],
        [
            0.0,
            0.0,
            0.0,
            -np.inf,
            0.0,
            -MAX_INFUSION_SLEW,
            -MAX_INFUSION_JERK,
        ],
        [
            np.inf,
            200.0,
            MAX_EFFECT_CONCENTRATION,
            MAX_CENTRAL_CONCENTRATION,
            MAX_INFUSION_RATE,
            MAX_INFUSION_SLEW,
            MAX_INFUSION_JERK,
        ],
    )
    loading.set_boundary_condition(
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [
            STEADY_CENTRAL_AMOUNT,
            STEADY_PERIPHERAL_AMOUNT,
            TARGET_EFFECT_CONCENTRATION,
            MAINTENANCE_INFUSION_RATE,
            0.0,
        ],
        0.0,
        LOADING_END,
    )
    loading.set_discretization(88, 2)

    maintenance = system.new_phase(
        ["central_amount", "peripheral_amount", "effect_concentration"], 0
    )
    central_hold, peripheral_hold, effect_hold = maintenance.x
    central_concentration_hold = central_hold / CENTRAL_VOLUME
    maintenance.set_dynamics(
        [
            MAINTENANCE_INFUSION_RATE
            - (elimination_rate + TRANSFER_TO_PERIPHERAL) * central_hold
            + TRANSFER_TO_CENTRAL * peripheral_hold,
            TRANSFER_TO_PERIPHERAL * central_hold
            - TRANSFER_TO_CENTRAL * peripheral_hold,
            EFFECT_EQUILIBRATION * (central_concentration_hold - effect_hold),
        ]
    )
    maintenance.set_integral(
        [
            (effect_hold / TARGET_EFFECT_CONCENTRATION - 1.0) ** 2
            + INFUSION_WEIGHT * (MAINTENANCE_INFUSION_RATE / MAX_INFUSION_RATE) ** 2
        ]
    )
    maintenance.set_phase_constraint(
        [central_hold, peripheral_hold, effect_hold, central_concentration_hold],
        [0.0, 0.0, 0.0, -np.inf],
        [np.inf, 200.0, MAX_EFFECT_CONCENTRATION, MAX_CENTRAL_CONCENTRATION],
    )
    maintenance_state = [
        STEADY_CENTRAL_AMOUNT,
        STEADY_PERIPHERAL_AMOUNT,
        TARGET_EFFECT_CONCENTRATION,
    ]
    maintenance.set_boundary_condition(
        maintenance_state, maintenance_state, LOADING_END, HORIZON
    )
    maintenance.set_discretization(8, 3)

    system.set_phase([loading, maintenance])
    system.set_objective(loading.I[0] + maintenance.I[0])
    return system, (loading, maintenance)


def initial_guess(phases):
    """Construct a smooth loading ramp followed by the steady regimen."""
    loading, maintenance = phases
    loading_guess = linear_guess(loading, 0.0)
    normalized_time = loading_guess.t_x / LOADING_END
    smooth_rise = 3.0 * normalized_time**2 - 2.0 * normalized_time**3
    loading_guess.x[0] = STEADY_CENTRAL_AMOUNT * smooth_rise
    loading_guess.x[1] = STEADY_PERIPHERAL_AMOUNT * smooth_rise
    loading_guess.x[2] = TARGET_EFFECT_CONCENTRATION * smooth_rise
    loading_guess.x[3] = MAINTENANCE_INFUSION_RATE * smooth_rise
    loading_guess.x[4] = (
        MAINTENANCE_INFUSION_RATE
        * (6.0 * normalized_time - 6.0 * normalized_time**2)
        / LOADING_END
    )
    normalized_control_time = loading_guess.t_u / LOADING_END
    loading_guess.u[0] = (
        MAINTENANCE_INFUSION_RATE
        * (6.0 - 12.0 * normalized_control_time)
        / LOADING_END**2
    )
    maintenance_guess = linear_guess(maintenance, 0.0)
    maintenance_guess.x[0] = np.full_like(maintenance_guess.t_x, STEADY_CENTRAL_AMOUNT)
    maintenance_guess.x[1] = np.full_like(
        maintenance_guess.t_x, STEADY_PERIPHERAL_AMOUNT
    )
    maintenance_guess.x[2] = np.full_like(
        maintenance_guess.t_x, TARGET_EFFECT_CONCENTRATION
    )
    return [loading_guess, maintenance_guess]


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    loading, maintenance = solution
    loading_count = round((DENSE_CHECK_POINTS - 1) * LOADING_END / HORIZON) + 1
    maintenance_count = DENSE_CHECK_POINTS - loading_count + 1
    loading_time = np.linspace(loading.t_0, loading.t_f, loading_count)
    maintenance_time = np.linspace(maintenance.t_0, maintenance.t_f, maintenance_count)
    loading_states = np.vstack(
        [loading.V_x(loading_time) @ component for component in loading.x]
    )
    maintenance_pk_states = np.vstack(
        [maintenance.V_x(maintenance_time) @ component for component in maintenance.x]
    )
    maintenance_states = np.vstack(
        [
            maintenance_pk_states,
            np.full(maintenance_count, MAINTENANCE_INFUSION_RATE),
            np.zeros(maintenance_count),
        ]
    )
    loading_jerk = loading.V_u(loading_time) @ loading.u[0]
    time = np.concatenate([loading_time, maintenance_time[1:]])
    states = np.hstack([loading_states, maintenance_states[:, 1:]])
    infusion_jerk = np.concatenate([loading_jerk, np.zeros(maintenance_count - 1)])
    phase_histories = (
        (loading_time, loading_states, loading_jerk[np.newaxis, :]),
        (
            maintenance_time,
            maintenance_pk_states,
            np.empty((0, maintenance_count)),
        ),
    )
    return time, states, infusion_jerk, phase_histories


def solve_problem(system, guess):
    """Optimize the infusion and verify exposure and actuator limits."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1500, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, states, infusion_jerk, phase_histories = _dense_solution(solution)
    infusion = states[3]
    infusion_slew = states[4]
    central_concentration = states[0] / CENTRAL_VOLUME
    require_finite(
        loading_time=phase_histories[0][0],
        loading_states=phase_histories[0][1],
        loading_controls=phase_histories[0][2],
        maintenance_time=phase_histories[1][0],
        maintenance_states=phase_histories[1][1],
        maintenance_controls=phase_histories[1][2],
        time=time,
        states=states,
        infusion_jerk=infusion_jerk,
        central_concentration=central_concentration,
        objective=info["obj_val"],
    )
    path_violation = max(
        float(np.max(-states[:4])),
        float(np.max(states[1] - 200.0)),
        float(np.max(central_concentration - MAX_CENTRAL_CONCENTRATION)),
        float(np.max(states[2] - MAX_EFFECT_CONCENTRATION)),
        float(np.max(infusion - MAX_INFUSION_RATE)),
        float(np.max(np.abs(infusion_slew) - MAX_INFUSION_SLEW)),
        float(np.max(np.abs(infusion_jerk) - MAX_INFUSION_JERK)),
        0.0,
    )
    terminal_target = np.array(
        [
            STEADY_CENTRAL_AMOUNT,
            STEADY_PERIPHERAL_AMOUNT,
            TARGET_EFFECT_CONCENTRATION,
            MAINTENANCE_INFUSION_RATE,
            0.0,
        ]
    )
    terminal_error = float(np.max(np.abs(states[:, -1] - terminal_target)))
    maintenance_mask = time >= LOADING_END
    maintenance_deviation = float(
        np.max(np.abs(states[:, maintenance_mask] - terminal_target[:, np.newaxis]))
    )
    terminal_derivative = np.array(
        [
            infusion[-1]
            - (CLEARANCE / CENTRAL_VOLUME + TRANSFER_TO_PERIPHERAL) * states[0, -1]
            + TRANSFER_TO_CENTRAL * states[1, -1],
            TRANSFER_TO_PERIPHERAL * states[0, -1]
            - TRANSFER_TO_CENTRAL * states[1, -1],
            EFFECT_EQUILIBRATION * (central_concentration[-1] - states[2, -1]),
            infusion_slew[-1],
            infusion_jerk[-1],
        ]
    )
    equilibrium_residual = float(np.max(np.abs(terminal_derivative)))
    if (
        path_violation > 2e-6
        or terminal_error > 2e-6
        or maintenance_deviation > 2e-6
        or equilibrium_residual > 2e-6
    ):
        raise RuntimeError("the dense trajectory or terminal target is infeasible")

    total_dose = trapezoid(infusion, time)
    effect_rmse = float(
        np.sqrt(np.mean((states[2] - TARGET_EFFECT_CONCENTRATION) ** 2))
    )
    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"total administered dose: {total_dose:.6f} mg")
    print(f"effect-site tracking RMSE: {effect_rmse:.6f} mg/L")
    print(f"peak central concentration: {np.max(central_concentration):.6f} mg/L")
    print(f"maximum terminal-state error: {terminal_error:.3e}")
    print(f"maximum maintenance-phase deviation: {maintenance_deviation:.3e}")
    print(f"terminal PK/pump-equilibrium residual: {equilibrium_residual:.3e}")
    print(f"maximum dense path-bound violation: {path_violation:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot exposure, compartment amounts, and the continuous pump regimen."""
    configure_matplotlib()
    time, states, infusion_jerk, _phase_histories = _dense_solution(solution)
    infusion = states[3]
    infusion_slew = states[4]
    central_concentration = states[0] / CENTRAL_VOLUME

    fig, axes = plt.subplots(
        4, 1, figsize=(8.2, 9.2), sharex=True, layout="constrained"
    )
    axes[0].plot(
        time,
        central_concentration,
        color=COLORS["blue"],
        label="Central concentration",
    )
    axes[0].plot(
        time,
        states[2],
        color=COLORS["orange"],
        label="Effect-site concentration",
    )
    axes[0].axhline(
        MAX_CENTRAL_CONCENTRATION,
        color=COLORS["vermillion"],
        linestyle="--",
        label="Central exposure limit",
    )
    axes[0].axhline(
        TARGET_EFFECT_CONCENTRATION,
        color=COLORS["black"],
        linestyle=":",
        label="Effect target",
    )
    axes[0].set_ylabel("Concentration [mg/L]")
    axes[0].legend(ncol=2)

    axes[1].plot(time, states[0], color=COLORS["blue"], label="Central")
    axes[1].plot(time, states[1], color=COLORS["green"], label="Peripheral")
    axes[1].set_ylabel("Drug amount [mg]")
    axes[1].legend(ncol=2)

    axes[2].plot(time, infusion, color=COLORS["purple"], label="Infusion rate")
    axes[2].axhline(
        MAINTENANCE_INFUSION_RATE,
        color=COLORS["black"],
        linestyle=":",
        label="Maintenance rate",
    )
    axes[3].plot(
        time,
        infusion_slew,
        color=COLORS["green"],
        label="Slew rate",
    )
    axes[2].set_ylabel("Infusion rate [mg/h]")
    axes[2].set_ylim(-1.0, MAX_INFUSION_RATE + 3.0)
    axes[2].legend(loc="upper right")
    jerk_axis = axes[3].twinx()
    jerk_axis.plot(
        time,
        infusion_jerk,
        color=COLORS["vermillion"],
        alpha=0.72,
        label="Jerk",
    )
    axes[3].set_ylabel("Slew rate [mg/h$^2$]")
    jerk_axis.set_ylabel("Jerk [mg/h$^3$]")
    axes[3].set_xlabel("Time [h]")
    axes[3].legend(loc="upper right")
    jerk_axis.legend(loc="lower right")

    for axis in axes:
        axis.axvline(LOADING_END, color=COLORS["black"], linestyle="--", linewidth=1.0)

    style_axes([*axes, jerk_axis])
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "pharmacokinetic_dosing_solution.png"
    )
    system, phases = build_problem()
    guess = initial_guess(phases)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
