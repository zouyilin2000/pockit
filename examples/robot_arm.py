"""Solve a minimum-time, variable-inertia robot-arm teaching benchmark.

This is a decoupled equivalent-inertia model, not a complete rigid-body
manipulator model. It intentionally omits inertia-rate momentum terms such as
``I_dot * qdot`` and the Coriolis and coupling terms of full manipulator
dynamics. The sliding coordinate still demonstrates how changing equivalent
inertia affects a bounded minimum-time maneuver.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

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


ARM_LENGTH = 5.0
INITIAL_PIVOT_POSITION = 4.5
INITIAL_AZIMUTH = 0.0
TARGET_AZIMUTH = 2.0 * np.pi / 3.0
FIXED_POLAR_ANGLE = np.pi / 4.0
POLAR_SINGULARITY_MARGIN = np.deg2rad(10.0)
MAX_ACTUATION = 1.0
NUM_MESH_INTERVALS = 160
CONTROL_POLYNOMIAL_POINTS = 2
DENSE_CONTROL_CHECK_POINTS = 10_001


def _control_history(solution, time):
    """Interpolate every control onto the requested physical times."""
    interpolation = solution.V_u(np.asarray(time, dtype=float).copy())
    return np.vstack(
        [
            np.asarray(interpolation @ np.asarray(control)).reshape(-1)
            for control in solution.u
        ]
    )


def build_problem():
    """Build the variable-inertia minimum-time arm problem."""
    system = System(0)
    phase = system.new_phase(
        [
            "pivot_position",
            "pivot_speed",
            "azimuth",
            "azimuth_rate",
            "polar_angle",
            "polar_angle_rate",
        ],
        ["pivot_force", "azimuth_torque", "polar_torque"],
    )
    (
        pivot_position,
        pivot_speed,
        _,
        azimuth_rate,
        polar_angle,
        polar_angle_rate,
    ) = phase.x
    pivot_force, azimuth_torque, polar_torque = phase.u

    polar_inertia = ((ARM_LENGTH - pivot_position) ** 3 + pivot_position**3) / 3.0
    azimuth_inertia = polar_inertia * sp.sin(polar_angle) ** 2

    phase.set_dynamics(
        [
            pivot_speed,
            pivot_force / ARM_LENGTH,
            azimuth_rate,
            azimuth_torque / azimuth_inertia,
            polar_angle_rate,
            polar_torque / polar_inertia,
        ]
    )
    phase.set_integral([1.0])
    phase.set_phase_constraint(
        [
            pivot_position,
            polar_angle,
            pivot_force,
            azimuth_torque,
            polar_torque,
        ],
        [0.0, POLAR_SINGULARITY_MARGIN, *([-MAX_ACTUATION] * 3)],
        [ARM_LENGTH, np.pi - POLAR_SINGULARITY_MARGIN, *([MAX_ACTUATION] * 3)],
        [False, False, True, True, True],
    )
    phase.set_boundary_condition(
        [
            INITIAL_PIVOT_POSITION,
            0.0,
            INITIAL_AZIMUTH,
            0.0,
            FIXED_POLAR_ANGLE,
            0.0,
        ],
        [
            INITIAL_PIVOT_POSITION,
            0.0,
            TARGET_AZIMUTH,
            0.0,
            FIXED_POLAR_ANGLE,
            0.0,
        ],
        0.0,
        None,
    )
    # Two Lobatto points make every control segment linear. Bounds imposed at
    # its endpoints therefore hold over the complete continuous-time segment.
    phase.set_discretization(NUM_MESH_INTERVALS, CONTROL_POLYNOMIAL_POINTS)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct a transfer guess with the expected torque reversal."""
    guess = linear_guess(phase, 0.0)
    guess.t_f = 10.5

    tau_u = guess.t_u / guess.t_f
    guess.u[0] = np.where(
        tau_u < 0.25,
        -0.7,
        np.where(tau_u > 0.75, 0.7, 0.0),
    )
    guess.u[1] = np.where(tau_u < 0.5, 0.8, -0.8)
    guess.u[2] = 0.0
    return guess


def solve_problem(system, guess):
    """Solve the arm benchmark and validate its boundary conditions."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={
            "tol": 1.0e-9,
            "max_iter": 2000,
            "print_level": 0,
            "sb": "yes",
            "bound_relax_factor": 0.0,
        },
    )
    status = int(info["status"])
    status_message = info["status_msg"]
    if isinstance(status_message, bytes):
        status_message = status_message.decode()
    if status not in (0, 1):
        raise RuntimeError(f"Ipopt failed ({status}): {status_message}")

    dense_time = np.linspace(solution.t_0, solution.t_f, DENSE_CONTROL_CHECK_POINTS)
    state_interpolation = solution.V_x(dense_time.copy())
    dense_state = np.vstack(
        [state_interpolation @ component for component in solution.x]
    )
    dense_control = _control_history(solution, dense_time)
    require_finite(
        dense_time=dense_time,
        dense_state=dense_state,
        dense_control=dense_control,
        final_time=solution.t_f,
        objective=info["obj_val"],
    )

    expected_final_state = np.array(
        [
            INITIAL_PIVOT_POSITION,
            0.0,
            TARGET_AZIMUTH,
            0.0,
            FIXED_POLAR_ANGLE,
            0.0,
        ]
    )
    actual_final_state = np.array([state[-1] for state in solution.x])
    if not np.allclose(actual_final_state, expected_final_state, rtol=0.0, atol=2.0e-7):
        endpoint_error = np.max(np.abs(actual_final_state - expected_final_state))
        raise RuntimeError(f"terminal-state error {endpoint_error:.3e} exceeds 2.0e-7")
    if not 8.0 < solution.t_f < 11.0:
        raise RuntimeError(f"unexpected final time: {solution.t_f:.9f} s")

    minimum_control = float(np.min(dense_control))
    maximum_control = float(np.max(dense_control))
    domain_violation = max(
        float(np.max(-dense_state[0])),
        float(np.max(dense_state[0] - ARM_LENGTH)),
        float(np.max(POLAR_SINGULARITY_MARGIN - dense_state[4])),
        float(np.max(dense_state[4] - (np.pi - POLAR_SINGULARITY_MARGIN))),
        0.0,
    )
    if minimum_control < -MAX_ACTUATION or maximum_control > MAX_ACTUATION:
        raise RuntimeError(
            "continuous actuator bound violated: "
            f"[{minimum_control:.12f}, {maximum_control:.12f}]"
        )
    if domain_violation > 2.0e-7:
        raise RuntimeError(f"physical-domain violation: {domain_violation:.3e}")

    print(f"Ipopt status: {status_message}")
    print(f"Minimum time: {solution.t_f:.9f} s")
    print(f"Smallest pivot position: {np.min(dense_state[0]):.6f} m")
    print(f"Smallest polar angle: {np.rad2deg(np.min(dense_state[4])):.6f} deg")
    print(f"Maximum physical-domain violation: {domain_violation:.3e}")
    print(
        f"Dense control range ({DENSE_CONTROL_CHECK_POINTS} points): "
        f"[{minimum_control:.9f}, {maximum_control:.9f}]"
    )
    return solution


def plot_solution(solution, *, save=None, show=True):
    """Plot the optimized coordinates, rates, and bounded actuators."""
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex="col")

    axes[0, 0].plot(
        solution.t_x,
        solution.x[0],
        color=COLORS["blue"],
        label="Pivot position",
    )
    axes[0, 0].axhline(
        ARM_LENGTH / 2.0,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.2,
        label="Minimum-inertia position",
    )
    axes[0, 0].set_ylabel("Position [m]")
    axes[0, 0].set_title("Variable-inertia robot arm")
    axes[0, 0].legend()

    axes[1, 0].plot(
        solution.t_x,
        np.rad2deg(solution.x[2]),
        color=COLORS["orange"],
        label="Azimuth",
    )
    axes[1, 0].plot(
        solution.t_x,
        np.rad2deg(solution.x[4]),
        color=COLORS["green"],
        label="Polar angle",
    )
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Angle [deg]")
    axes[1, 0].legend()

    rate_labels = ("Pivot speed", "Azimuth rate", "Polar-angle rate")
    rate_colors = (COLORS["blue"], COLORS["orange"], COLORS["green"])
    for state_index, label, color in zip((1, 3, 5), rate_labels, rate_colors):
        axes[0, 1].plot(
            solution.t_x,
            solution.x[state_index],
            color=color,
            label=label,
        )
    axes[0, 1].set_ylabel("Generalized rate")
    axes[0, 1].set_title("State and actuator histories")
    axes[0, 1].legend()

    control_time = np.linspace(solution.t_0, solution.t_f, 2001)
    control_history = _control_history(solution, control_time)
    control_labels = ("Pivot force", "Azimuth torque", "Polar torque")
    for control, label, color in zip(control_history, control_labels, rate_colors):
        axes[1, 1].plot(control_time, control, color=color, label=label)
    axes[1, 1].axhline(
        MAX_ACTUATION, color=COLORS["black"], linestyle="--", linewidth=1.0
    )
    axes[1, 1].axhline(
        -MAX_ACTUATION, color=COLORS["black"], linestyle="--", linewidth=1.0
    )
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Actuation [-]")
    axes[1, 1].set_ylim(-1.12, 1.12)
    axes[1, 1].legend()

    style_axes(axes)
    fig.tight_layout()
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(__doc__.splitlines()[0], "robot_arm_solution.png")
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
