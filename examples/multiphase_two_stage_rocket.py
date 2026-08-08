"""Optimize a vertical two-stage rocket ascent with a mass reset.

Altitude, velocity, and separation time connect the powered stages. Vehicle
mass drops instantaneously when the empty first-stage hardware is discarded,
so the example demonstrates both continuous phase links and a discrete reset.
All quantities are nondimensional to keep the numerical scales well balanced.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pockit.optimizer import ipopt
from pockit.radau import System, linear_guess

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

GRAVITY = 1.0
INITIAL_MASS = 1.0
FIRST_STAGE_PROPELLANT_MASS = 0.06
STAGE_DROP_MASS = 0.20
SECOND_STAGE_PROPELLANT_MASS = 0.12
FIRST_STAGE_BURNOUT_MASS = INITIAL_MASS - FIRST_STAGE_PROPELLANT_MASS
SECOND_STAGE_INITIAL_MASS = FIRST_STAGE_BURNOUT_MASS - STAGE_DROP_MASS
SECOND_STAGE_DRY_MASS = SECOND_STAGE_INITIAL_MASS - SECOND_STAGE_PROPELLANT_MASS
TARGET_ALTITUDE = 2.0
DENSE_CHECK_POINTS = 2_001


def _configure_stage(
    phase,
    *,
    thrust: float,
    mass_flow: float,
    mass_bounds: tuple[float, float],
    boundaries,
    times,
):
    altitude, velocity, mass = phase.x
    (throttle,) = phase.u
    phase.set_dynamics(
        [velocity, thrust * throttle / mass - GRAVITY, -mass_flow * throttle]
    )
    phase.set_integral([throttle**2])
    phase.set_phase_constraint(
        [throttle, altitude, velocity, mass],
        [0.0, 0.0, 0.0, mass_bounds[0]],
        [1.0, TARGET_ALTITUDE, 3.0, mass_bounds[1]],
        [True, False, False, False],
    )
    phase.set_boundary_condition(*boundaries, *times)
    # Piecewise-constant Radau controls make the nodal throttle bounds valid
    # throughout every interval, including coast arcs.
    phase.set_discretization(96, 1)


def build_problem():
    """Return the linked two-stage rocket system and phases."""
    system = System(
        [
            "h_separation",
            "v_separation",
            "m_before_drop",
            "m_after_drop",
            "t_separation",
            "m_final",
            "t_final",
        ]
    )
    h_s, v_s, m_before, m_after, t_s, m_f, t_f = system.s

    first_stage = system.new_phase(
        ["altitude_1", "velocity_1", "mass_1"], ["throttle_1"]
    )
    _configure_stage(
        first_stage,
        thrust=2.40,
        mass_flow=0.080,
        mass_bounds=(FIRST_STAGE_BURNOUT_MASS, INITIAL_MASS),
        boundaries=([0.0, 0.0, INITIAL_MASS], [h_s, v_s, m_before]),
        times=(0.0, t_s),
    )

    second_stage = system.new_phase(
        ["altitude_2", "velocity_2", "mass_2"], ["throttle_2"]
    )
    _configure_stage(
        second_stage,
        thrust=1.60,
        mass_flow=0.045,
        mass_bounds=(SECOND_STAGE_DRY_MASS, SECOND_STAGE_INITIAL_MASS),
        boundaries=([h_s, v_s, m_after], [TARGET_ALTITUDE, 0.0, m_f]),
        times=(t_s, t_f),
    )

    system.set_phase([first_stage, second_stage])
    system.set_objective(t_f + 0.04 * (first_stage.I[0] + second_stage.I[0]))
    system.set_system_constraint(
        [
            h_s,
            v_s,
            m_before,
            m_after,
            m_after - m_before,
            t_s,
            m_f,
            t_f - t_s,
            t_f,
        ],
        [
            0.10,
            0.05,
            FIRST_STAGE_BURNOUT_MASS,
            SECOND_STAGE_DRY_MASS,
            -STAGE_DROP_MASS,
            0.20,
            SECOND_STAGE_DRY_MASS,
            0.40,
            1.00,
        ],
        [
            1.80,
            2.50,
            FIRST_STAGE_BURNOUT_MASS,
            SECOND_STAGE_INITIAL_MASS,
            -STAGE_DROP_MASS,
            3.00,
            SECOND_STAGE_INITIAL_MASS,
            5.00,
            7.00,
        ],
    )
    return system, (first_stage, second_stage)


def initial_guess(phases):
    """Construct a dynamically plausible initial ascent guess."""
    first_stage, second_stage = phases
    separation = np.array([0.40, 0.80, FIRST_STAGE_BURNOUT_MASS])
    mass_after = SECOND_STAGE_INITIAL_MASS
    final_mass = 0.68
    separation_time = 1.10
    final_time = 3.60

    guess_1 = linear_guess(first_stage, 0.0)
    guess_1.t_f = separation_time
    tau_1 = guess_1.t_x / separation_time
    guess_1.x[0] = separation[0] * tau_1**2
    guess_1.x[1] = separation[1] * tau_1
    guess_1.x[2] = INITIAL_MASS + (separation[2] - INITIAL_MASS) * tau_1
    guess_1.u[0] = 0.75

    guess_2 = linear_guess(second_stage, 0.0)
    guess_2.t_0 = separation_time
    guess_2.t_f = final_time
    tau_2 = (guess_2.t_x - separation_time) / (final_time - separation_time)
    guess_2.x[0] = separation[0] + (TARGET_ALTITUDE - separation[0]) * tau_2
    guess_2.x[1] = separation[1] * (1 - tau_2) + 0.55 * np.sin(np.pi * tau_2)
    guess_2.x[2] = mass_after + (final_mass - mass_after) * tau_2
    tau_2_u = (guess_2.t_u - separation_time) / (final_time - separation_time)
    guess_2.u[0] = np.where(tau_2_u < 0.6, 0.9, 0.0)

    static_guess = [
        separation[0],
        separation[1],
        separation[2],
        mass_after,
        separation_time,
        final_mass,
        final_time,
    ]
    return [guess_1, guess_2, static_guess]


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def solve_problem(system, guess):
    """Solve the staged ascent and validate the separation reset."""
    result, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-8, "max_iter": 1800, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)
    first_stage, second_stage, static = result

    dense_ranges = []
    for stage in (first_stage, second_stage):
        time = np.linspace(stage.t_0, stage.t_f, DENSE_CHECK_POINTS)
        state = np.vstack([stage.V_x(time) @ component for component in stage.x])
        control = np.vstack([stage.V_u(time) @ component for component in stage.u])
        dense_ranges.append((time, state, control))
    require_finite(
        first_stage_time=dense_ranges[0][0],
        first_stage_states=dense_ranges[0][1],
        first_stage_controls=dense_ranges[0][2],
        second_stage_time=dense_ranges[1][0],
        second_stage_states=dense_ranges[1][1],
        second_stage_controls=dense_ranges[1][2],
        static_values=static,
        objective=info["obj_val"],
    )

    altitude_gap = abs(first_stage.x[0][-1] - second_stage.x[0][0])
    velocity_gap = abs(first_stage.x[1][-1] - second_stage.x[1][0])
    mass_drop = first_stage.x[2][-1] - second_stage.x[2][0]
    if max(altitude_gap, velocity_gap) > 2e-6:
        raise RuntimeError("continuous states do not match at separation")
    if abs(mass_drop - STAGE_DROP_MASS) > 2e-6:
        raise RuntimeError("stage-separation mass reset is incorrect")
    if abs(first_stage.x[2][-1] - FIRST_STAGE_BURNOUT_MASS) > 2e-6:
        raise RuntimeError("first-stage propellant was not depleted")
    if second_stage.x[2][-1] < SECOND_STAGE_DRY_MASS - 2e-6:
        raise RuntimeError("second-stage dry-mass bound was violated")
    if max(first_stage.u[0]) < 0.1 or max(second_stage.u[0]) < 0.1:
        raise RuntimeError("both powered stages must contribute to the ascent")

    path_violation = 0.0
    for (_time, state, control), mass_bounds in zip(
        dense_ranges,
        (
            (FIRST_STAGE_BURNOUT_MASS, INITIAL_MASS),
            (SECOND_STAGE_DRY_MASS, SECOND_STAGE_INITIAL_MASS),
        ),
    ):
        throttle = control[0]
        path_violation = max(
            path_violation,
            float(np.max(-throttle)),
            float(np.max(throttle - 1.0)),
            float(np.max(-state[0])),
            float(np.max(state[0] - TARGET_ALTITUDE)),
            float(np.max(-state[1])),
            float(np.max(state[1] - 3.0)),
            float(np.max(mass_bounds[0] - state[2])),
            float(np.max(state[2] - mass_bounds[1])),
        )
    if path_violation > 2e-6:
        raise RuntimeError(f"dense path-bound violation: {path_violation:.3e}")

    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"separation time: {static[4]:.6f}")
    print(f"final time: {static[6]:.6f}")
    print(f"separation mass drop: {mass_drop:.6f}")
    print(f"maximum dense path-bound violation: {max(path_violation, 0.0):.3e}")
    print(
        f"second-stage propellant used: {second_stage.x[2][0] - second_stage.x[2][-1]:.6f}"
    )
    return first_stage, second_stage, static


def plot_solution(result, *, save: str | Path | None = None, show: bool = True):
    """Plot state histories and throttle on a shared time axis."""
    first_stage, second_stage, static = result
    configure_matplotlib()
    fig, axes = plt.subplots(
        4, 1, figsize=(8.0, 8.8), sharex=True, layout="constrained"
    )
    colors = ["#0072B2", "#D55E00", "#009E73"]
    labels = ["Altitude", "Velocity", "Mass"]
    for i, (color, label) in enumerate(zip(colors, labels)):
        axes[i].plot(first_stage.t_x, first_stage.x[i], color=color)
        axes[i].plot(second_stage.t_x, second_stage.x[i], color=color)
        axes[i].set_ylabel(label)
    axes[2].plot(
        [static[4], static[4]],
        [first_stage.x[2][-1], second_stage.x[2][0]],
        color="#555555",
        linestyle=":",
        marker="o",
        markersize=3,
    )
    for stage in (first_stage, second_stage):
        throttle_time = np.append(stage.t_u, stage.t_f)
        throttle = np.append(stage.u[0], stage.u[0][-1])
        axes[3].step(throttle_time, throttle, where="post", color="#CC79A7")
    axes[3].set_ylabel("Throttle")
    axes[3].set_xlabel("Nondimensional time")
    for axis in axes:
        axis.axvline(static[4], color="#555555", linestyle="--", linewidth=1.2)
    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "multiphase_two_stage_rocket_solution.png"
    )
    system, phases = build_problem()
    guess = initial_guess(phases)
    result = solve_problem(system, guess)
    plot_solution(result, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
