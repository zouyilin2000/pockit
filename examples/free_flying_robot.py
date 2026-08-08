"""Plan a fuel-efficient planar maneuver for a free-flying robot.

The robot has two opposed-thruster modules. Each module produces signed force
along the body axis by firing one of two one-sided jets, while the difference
between module forces produces a yaw moment. The task transfers position and
attitude in a fixed time while minimizing total propellant flow.
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


MOMENT_ARM_A = 0.2
MOMENT_ARM_B = 0.2
MAX_JET_COMMAND = 1.0
HORIZON = 12.0
NUM_MESH_INTERVALS = 160
CONTROL_POLYNOMIAL_POINTS = 2
DENSE_CONTROL_CHECK_POINTS = 10_001
INITIAL_STATE = np.array([-10.0, -10.0, 0.0, 0.0, np.pi / 2.0, 0.0])
TARGET_STATE = np.zeros(6)


def _quintic_profile(tau):
    """Return a quintic rest-to-rest profile and its first two derivatives."""
    value = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    rate = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    acceleration = 60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3
    return value, rate, acceleration


def _control_history(solution, time):
    """Interpolate every jet command onto the requested physical times."""
    interpolation = solution.V_u(np.asarray(time, dtype=float).copy())
    return np.vstack(
        [
            np.asarray(interpolation @ np.asarray(control)).reshape(-1)
            for control in solution.u
        ]
    )


def build_problem():
    """Build the fixed-time, minimum-fuel free-flyer problem."""
    system = System(0)
    phase = system.new_phase(
        ["x", "y", "velocity_x", "velocity_y", "heading", "yaw_rate"],
        [
            "module_a_positive",
            "module_a_negative",
            "module_b_positive",
            "module_b_negative",
        ],
    )
    _, _, velocity_x, velocity_y, heading, yaw_rate = phase.x
    (
        module_a_positive,
        module_a_negative,
        module_b_positive,
        module_b_negative,
    ) = phase.u

    thrust_a = module_a_positive - module_a_negative
    thrust_b = module_b_positive - module_b_negative
    total_thrust = thrust_a + thrust_b
    yaw_moment = MOMENT_ARM_A * thrust_a - MOMENT_ARM_B * thrust_b

    phase.set_dynamics(
        [
            velocity_x,
            velocity_y,
            total_thrust * sp.cos(heading),
            total_thrust * sp.sin(heading),
            yaw_rate,
            yaw_moment,
        ]
    )
    phase.set_integral([sum(phase.u)])
    phase.set_phase_constraint(
        list(phase.u),
        [0.0] * 4,
        [MAX_JET_COMMAND] * 4,
        True,
    )
    phase.set_boundary_condition(
        INITIAL_STATE.tolist(), TARGET_STATE.tolist(), 0.0, HORIZON
    )
    # Two Lobatto points make every control segment linear. Bounds imposed at
    # its endpoints therefore hold over the complete continuous-time segment.
    phase.set_discretization(NUM_MESH_INTERVALS, CONTROL_POLYNOMIAL_POINTS)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct a smooth rest-to-rest translation and attitude guess."""
    guess = linear_guess(phase, 0.0)

    tau_x = guess.t_x / HORIZON
    progress_x, rate_x, _ = _quintic_profile(tau_x)
    displacement = TARGET_STATE[:2] - INITIAL_STATE[:2]
    guess.x[0] = INITIAL_STATE[0] + displacement[0] * progress_x
    guess.x[1] = INITIAL_STATE[1] + displacement[1] * progress_x
    guess.x[2] = displacement[0] * rate_x / HORIZON
    guess.x[3] = displacement[1] * rate_x / HORIZON
    guess.x[4] = INITIAL_STATE[4] * (1.0 - progress_x)
    guess.x[5] = -INITIAL_STATE[4] * rate_x / HORIZON

    tau_u = guess.t_u / HORIZON
    progress_u, _, acceleration_u = _quintic_profile(tau_u)
    heading_u = INITIAL_STATE[4] * (1.0 - progress_u)
    acceleration_xy = displacement[0] * acceleration_u / HORIZON**2
    axial_force = acceleration_xy * (np.cos(heading_u) + np.sin(heading_u))
    yaw_moment = -INITIAL_STATE[4] * acceleration_u / HORIZON**2
    thrust_a = (axial_force + yaw_moment / MOMENT_ARM_A) / 2.0
    thrust_b = (axial_force - yaw_moment / MOMENT_ARM_B) / 2.0
    guess.u[0] = np.clip(thrust_a, 0.0, MAX_JET_COMMAND)
    guess.u[1] = np.clip(-thrust_a, 0.0, MAX_JET_COMMAND)
    guess.u[2] = np.clip(thrust_b, 0.0, MAX_JET_COMMAND)
    guess.u[3] = np.clip(-thrust_b, 0.0, MAX_JET_COMMAND)
    return guess


def solve_problem(system, guess):
    """Solve the maneuver and validate the endpoint and actuator limits."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={
            "tol": 1.0e-9,
            "max_iter": 3000,
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
        objective=info["obj_val"],
    )

    terminal_state = np.array([state[-1] for state in solution.x])
    if not np.allclose(terminal_state, TARGET_STATE, rtol=0.0, atol=2.0e-7):
        endpoint_error = np.max(np.abs(terminal_state - TARGET_STATE))
        raise RuntimeError(f"terminal-state error {endpoint_error:.3e} exceeds 2.0e-7")

    minimum_command = float(np.min(dense_control))
    maximum_command = float(np.max(dense_control))
    if minimum_command < 0.0 or maximum_command > MAX_JET_COMMAND:
        raise RuntimeError(
            "continuous jet-command bound violated: "
            f"[{minimum_command:.12f}, {maximum_command:.12f}]"
        )

    print(f"Ipopt status: {status_message}")
    print(f"Minimum propellant proxy: {float(info['obj_val']):.9f}")
    print(
        f"Dense command range ({DENSE_CONTROL_CHECK_POINTS} points): "
        f"[{minimum_command:.9f}, {maximum_command:.9f}]"
    )
    return solution


def plot_solution(solution, *, save=None, show=True):
    """Plot the planar maneuver, states, and individual jet commands."""
    configure_matplotlib()
    fig = plt.figure(figsize=(10.0, 7.2))
    grid = fig.add_gridspec(3, 2, width_ratios=(1.15, 1.0))
    path_axis = fig.add_subplot(grid[:, 0])
    velocity_axis = fig.add_subplot(grid[0, 1])
    attitude_axis = fig.add_subplot(grid[1, 1], sharex=velocity_axis)
    command_axis = fig.add_subplot(grid[2, 1], sharex=velocity_axis)

    path_axis.plot(
        solution.x[0],
        solution.x[1],
        color=COLORS["blue"],
        label="Center-of-mass path",
    )
    pose_indices = np.linspace(0, len(solution.t_x) - 1, 7, dtype=int)
    pose_indices = np.unique(pose_indices)
    for index in pose_indices:
        x_position = solution.x[0][index]
        y_position = solution.x[1][index]
        heading = solution.x[4][index]
        body_axis = 0.55 * np.array([np.cos(heading), np.sin(heading)])
        normal_axis = 0.32 * np.array([-np.sin(heading), np.cos(heading)])
        corners = np.array(
            [
                body_axis + normal_axis,
                body_axis - normal_axis,
                -body_axis - normal_axis,
                -body_axis + normal_axis,
                body_axis + normal_axis,
            ]
        )
        alpha = 0.35 if index != pose_indices[-1] else 0.9
        path_axis.plot(
            x_position + corners[:, 0],
            y_position + corners[:, 1],
            color=COLORS["orange"],
            linewidth=1.2,
            alpha=alpha,
        )
    path_axis.scatter(
        [INITIAL_STATE[0], TARGET_STATE[0]],
        [INITIAL_STATE[1], TARGET_STATE[1]],
        color=[COLORS["green"], COLORS["vermillion"]],
        zorder=3,
        label="Endpoints",
    )
    path_axis.set_xlabel("Position x [m]")
    path_axis.set_ylabel("Position y [m]")
    path_axis.set_title("Planar free-flyer maneuver")
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.legend()

    velocity_axis.plot(
        solution.t_x,
        solution.x[2],
        color=COLORS["blue"],
        label=r"$v_x$",
    )
    velocity_axis.plot(
        solution.t_x,
        solution.x[3],
        color=COLORS["green"],
        label=r"$v_y$",
    )
    velocity_axis.set_ylabel("Velocity [m/s]")
    velocity_axis.set_title("State and actuator histories")
    velocity_axis.legend()

    attitude_axis.plot(
        solution.t_x,
        np.rad2deg(solution.x[4]),
        color=COLORS["orange"],
        label="Heading",
    )
    attitude_axis.plot(
        solution.t_x,
        np.rad2deg(solution.x[5]),
        color=COLORS["purple"],
        label="Yaw rate",
    )
    attitude_axis.set_ylabel("Angle / rate [deg, deg/s]")
    attitude_axis.legend()

    control_time = np.linspace(solution.t_0, solution.t_f, 2001)
    control_history = _control_history(solution, control_time)
    command_labels = ("A+", "A-", "B+", "B-")
    command_colors = (
        COLORS["blue"],
        COLORS["sky_blue"],
        COLORS["orange"],
        COLORS["vermillion"],
    )
    for control, label, color in zip(control_history, command_labels, command_colors):
        command_axis.plot(control_time, control, color=color, label=label)
    command_axis.set_xlabel("Time [s]")
    command_axis.set_ylabel("Jet command [-]")
    command_axis.set_ylim(-0.04, 1.08)
    command_axis.legend(ncol=4)

    style_axes([path_axis, velocity_axis, attitude_axis, command_axis])
    fig.tight_layout()
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "free_flying_robot_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
