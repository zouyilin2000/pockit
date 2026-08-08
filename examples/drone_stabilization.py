"""Recover a six-degree-of-freedom quadrotor to a hover equilibrium.

A Crazyflie-class vehicle starts away from the origin with translational and
angular motion. Four normalized squared rotor speeds are actuator states, and
their bounded rates are the controls. The commands start and end at the level-
hover value, so the recovered terminal state is paired with a balanced wrench.
Attitude uses three modified Rodrigues parameters (MRPs), avoiding a redundant
quaternion state and unit-norm constraint. A path bound keeps the motion well
inside the local MRP chart.
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


MASS = 0.033  # kg
GRAVITY = 9.81  # m/s^2
INERTIA = np.diag([1.395e-5, 1.395e-5, 2.173e-5])  # kg m^2
THRUST_COEFFICIENT = 2.3e-8  # N / (rad/s)^2
DRAG_MOMENT_COEFFICIENT = 7.8e-10  # N m / (rad/s)^2
ROTOR_POSITIONS = np.array(
    [
        [0.028, 0.028, 0.0],
        [0.028, -0.028, 0.0],
        [-0.028, -0.028, 0.0],
        [-0.028, 0.028, 0.0],
    ]
)  # m, expressed in the body frame
ROTOR_REACTION_YAW_SIGNS = np.array([1.0, -1.0, 1.0, -1.0])
MAX_ROTOR_SPEED = 2_500.0  # rad/s
HOVER_ROTOR_SPEED = np.sqrt(MASS * GRAVITY / (4.0 * THRUST_COEFFICIENT))
MAX_ROTOR_COMMAND = (MAX_ROTOR_SPEED / HOVER_ROTOR_SPEED) ** 2
MAX_ROTOR_COMMAND_RATE = 4.0  # 1/s, squared-speed-ratio slew limit
ROTOR_HOVER_THRUST = MASS * GRAVITY / 4.0
MAX_MRP_NORM = np.tan(np.deg2rad(150.0) / 4.0)
HORIZON = 5.0  # s
DENSE_CHECK_POINTS = 4_001

# Reference scales make every objective term dimensionless.
POSITION_ERROR_SCALE = 1.0  # m
VELOCITY_ERROR_SCALE = 1.0  # m/s
ATTITUDE_ERROR_SCALE = 1.0  # rad; applied to the local 4*p angle proxy
ANGULAR_RATE_ERROR_SCALE = 1.0  # rad/s

INITIAL_POSITION = np.array([1.5, -1.0, 1.8])  # m
INITIAL_VELOCITY = np.array([0.6, -0.4, 0.2])  # m/s
INITIAL_MRP = np.array([0.15, -0.10, 0.08])
INITIAL_ANGULAR_RATE = np.array([0.45, -0.35, 0.25])  # rad/s
INITIAL_STATE = np.concatenate(
    [INITIAL_POSITION, INITIAL_VELOCITY, INITIAL_MRP, INITIAL_ANGULAR_RATE]
)


def _mrp_rotation_symbolic(mrp: sp.Matrix) -> sp.Matrix:
    """Return the body-to-inertial rotation matrix for an MRP vector."""
    p1, p2, p3 = mrp
    norm_squared = mrp.dot(mrp)
    denominator = 1.0 + norm_squared
    q0 = (1.0 - norm_squared) / denominator
    q1 = 2.0 * p1 / denominator
    q2 = 2.0 * p2 / denominator
    q3 = 2.0 * p3 / denominator
    return sp.Matrix(
        [
            [
                q0**2 + q1**2 - q2**2 - q3**2,
                2.0 * (q1 * q2 - q0 * q3),
                2.0 * (q1 * q3 + q0 * q2),
            ],
            [
                2.0 * (q1 * q2 + q0 * q3),
                q0**2 - q1**2 + q2**2 - q3**2,
                2.0 * (q2 * q3 - q0 * q1),
            ],
            [
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q2 * q3 + q0 * q1),
                q0**2 - q1**2 - q2**2 + q3**2,
            ],
        ]
    )


def _mrp_rotation_numeric(mrp: np.ndarray) -> np.ndarray:
    """Return the numeric body-to-inertial rotation matrix for an MRP vector."""
    p1, p2, p3 = np.asarray(mrp, dtype=float)
    norm_squared = p1**2 + p2**2 + p3**2
    denominator = 1.0 + norm_squared
    q0 = (1.0 - norm_squared) / denominator
    q1, q2, q3 = 2.0 * np.array([p1, p2, p3]) / denominator
    return np.array(
        [
            [
                q0**2 + q1**2 - q2**2 - q3**2,
                2.0 * (q1 * q2 - q0 * q3),
                2.0 * (q1 * q3 + q0 * q2),
            ],
            [
                2.0 * (q1 * q2 + q0 * q3),
                q0**2 - q1**2 + q2**2 - q3**2,
                2.0 * (q2 * q3 - q0 * q1),
            ],
            [
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q2 * q3 + q0 * q1),
                q0**2 - q1**2 - q2**2 + q3**2,
            ],
        ]
    )


def _mrp_kinematics(mrp: sp.Matrix, angular_rate: sp.Matrix) -> sp.Matrix:
    """Return MRP rate for body angular velocity expressed in the body frame."""
    p1, p2, p3 = mrp
    wx, wy, wz = angular_rate
    norm_squared = mrp.dot(mrp)
    return 0.25 * sp.Matrix(
        [
            (1.0 - norm_squared + 2.0 * p1**2) * wx
            + 2.0 * (p1 * p2 - p3) * wy
            + 2.0 * (p1 * p3 + p2) * wz,
            2.0 * (p1 * p2 + p3) * wx
            + (1.0 - norm_squared + 2.0 * p2**2) * wy
            + 2.0 * (p2 * p3 - p1) * wz,
            2.0 * (p1 * p3 - p2) * wx
            + 2.0 * (p2 * p3 + p1) * wy
            + (1.0 - norm_squared + 2.0 * p3**2) * wz,
        ]
    )


def build_problem(quick: bool = False):
    """Build the rate-limited finite-horizon hover-recovery problem."""
    rigid_body_state_names = [
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "mrp_1",
        "mrp_2",
        "mrp_3",
        "angular_rate_x",
        "angular_rate_y",
        "angular_rate_z",
    ]
    rotor_state_names = [f"rotor_{index}_speed_squared_ratio" for index in range(1, 5)]
    system = System([f"final_{name}" for name in rigid_body_state_names], fastmath=True)
    phase = system.new_phase(
        [*rigid_body_state_names, *rotor_state_names],
        [f"rotor_{index}_command_rate" for index in range(1, 5)],
    )

    position = sp.Matrix(phase.x[:3])
    velocity = sp.Matrix(phase.x[3:6])
    mrp = sp.Matrix(phase.x[6:9])
    angular_rate = sp.Matrix(phase.x[9:12])
    rotor_command = sp.Matrix(phase.x[12:16])
    rotor_command_rate = sp.Matrix(phase.u)
    rotation = _mrp_rotation_symbolic(mrp)

    total_force_body = sp.zeros(3, 1)
    total_moment_body = sp.zeros(3, 1)
    body_z = sp.Matrix([0.0, 0.0, 1.0])
    for index in range(4):
        rotor_thrust = ROTOR_HOVER_THRUST * rotor_command[index]
        rotor_force = rotor_thrust * body_z
        rotor_drag_moment = (
            ROTOR_REACTION_YAW_SIGNS[index]
            * DRAG_MOMENT_COEFFICIENT
            / THRUST_COEFFICIENT
            * rotor_thrust
            * body_z
        )
        arm = sp.Matrix(ROTOR_POSITIONS[index])
        total_force_body += rotor_force
        total_moment_body += arm.cross(rotor_force) + rotor_drag_moment

    inertia = sp.Matrix(INERTIA)
    position_rate = velocity
    velocity_rate = rotation * total_force_body / MASS - sp.Matrix([0.0, 0.0, GRAVITY])
    mrp_rate = _mrp_kinematics(mrp, angular_rate)
    angular_acceleration = inertia.inv() * (
        total_moment_body - angular_rate.cross(inertia * angular_rate)
    )
    phase.set_dynamics(
        [
            *position_rate,
            *velocity_rate,
            *mrp_rate,
            *angular_acceleration,
            *rotor_command_rate,
        ]
    )

    normalized_position = position / POSITION_ERROR_SCALE
    normalized_velocity = velocity / VELOCITY_ERROR_SCALE
    normalized_attitude_error = 4.0 * mrp / ATTITUDE_ERROR_SCALE
    normalized_angular_rate = angular_rate / ANGULAR_RATE_ERROR_SCALE
    control_deviation = sum((command - 1.0) ** 2 for command in rotor_command)
    normalized_command_rate = rotor_command_rate / MAX_ROTOR_COMMAND_RATE
    dimensionless_running_cost = (
        5.0 * normalized_position.dot(normalized_position)
        + 1.5 * normalized_velocity.dot(normalized_velocity)
        + 8.0 * normalized_attitude_error.dot(normalized_attitude_error)
        + 0.5 * normalized_angular_rate.dot(normalized_angular_rate)
        + 0.06 * control_deviation
        + 0.02 * normalized_command_rate.dot(normalized_command_rate)
    )
    phase.set_integral([dimensionless_running_cost])
    phase.set_phase_constraint(
        [*rotor_command, *rotor_command_rate, mrp.dot(mrp)],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            *([-MAX_ROTOR_COMMAND_RATE] * 4),
            0.0,
        ],
        [
            MAX_ROTOR_COMMAND,
            MAX_ROTOR_COMMAND,
            MAX_ROTOR_COMMAND,
            MAX_ROTOR_COMMAND,
            *([MAX_ROTOR_COMMAND_RATE] * 4),
            MAX_MRP_NORM**2,
        ],
    )
    hover_command = [1.0] * 4
    phase.set_boundary_condition(
        [*INITIAL_STATE, *hover_command],
        [*system.s, *hover_command],
        0.0,
        HORIZON,
    )
    phase.set_discretization(28 if quick else 44, 2)
    system.set_phase([phase])

    final_position = sp.Matrix(system.s[:3])
    final_velocity = sp.Matrix(system.s[3:6])
    final_mrp = sp.Matrix(system.s[6:9])
    final_angular_rate = sp.Matrix(system.s[9:12])
    terminal_cost = (
        800.0 * final_position.dot(final_position) / POSITION_ERROR_SCALE**2
        + 250.0 * final_velocity.dot(final_velocity) / VELOCITY_ERROR_SCALE**2
        + 500.0 * 16.0 * final_mrp.dot(final_mrp) / ATTITUDE_ERROR_SCALE**2
        + 120.0
        * final_angular_rate.dot(final_angular_rate)
        / ANGULAR_RATE_ERROR_SCALE**2
    )
    system.set_objective(phase.I[0] / HORIZON + terminal_cost)
    return system, phase


def initial_guess(phase):
    """Construct a smooth rigid-body decay with balanced rotor states."""
    phase_guess = linear_guess(phase, 1.0)
    fraction = phase_guess.t_x / HORIZON
    progress = 10.0 * fraction**3 - 15.0 * fraction**4 + 6.0 * fraction**5
    for index, initial_value in enumerate(INITIAL_STATE):
        phase_guess.x[index] = initial_value * (1.0 - progress)
    for index in range(12, 16):
        phase_guess.x[index][:] = 1.0
    for control in phase_guess.u:
        control[:] = 0.0
    return [phase_guess, np.zeros(12)]


def _dense_history(solution, count: int = DENSE_CHECK_POINTS):
    """Interpolate states and controls onto a uniform physical-time grid."""
    time = np.linspace(solution.t_0, solution.t_f, count)
    state_interpolation = solution.V_x(time.copy())
    control_interpolation = solution.V_u(time.copy())
    state = np.vstack([state_interpolation @ component for component in solution.x])
    control = np.vstack([control_interpolation @ component for component in solution.u])
    return time, state, control


def _principal_angles(mrp_history: np.ndarray) -> np.ndarray:
    """Return principal rotation angles for MRP columns inside the local chart."""
    return 4.0 * np.arctan(np.linalg.norm(mrp_history, axis=0))


def _rigid_body_accelerations(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return inertial linear and body angular acceleration for one state."""
    mrp = np.asarray(state[6:9], dtype=float)
    angular_rate = np.asarray(state[9:12], dtype=float)
    rotor_command = np.asarray(state[12:16], dtype=float)
    body_force = np.array([0.0, 0.0, ROTOR_HOVER_THRUST * rotor_command.sum()])
    body_moment = np.zeros(3)
    for index, command in enumerate(rotor_command):
        rotor_force = np.array([0.0, 0.0, ROTOR_HOVER_THRUST * command])
        rotor_drag_moment = np.array(
            [
                0.0,
                0.0,
                ROTOR_REACTION_YAW_SIGNS[index]
                * DRAG_MOMENT_COEFFICIENT
                / THRUST_COEFFICIENT
                * ROTOR_HOVER_THRUST
                * command,
            ]
        )
        body_moment += np.cross(ROTOR_POSITIONS[index], rotor_force) + rotor_drag_moment
    linear_acceleration = _mrp_rotation_numeric(mrp) @ body_force / MASS - np.array(
        [0.0, 0.0, GRAVITY]
    )
    angular_acceleration = np.linalg.solve(
        INERTIA,
        body_moment - np.cross(angular_rate, INERTIA @ angular_rate),
    )
    return linear_acceleration, angular_acceleration


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def solve_problem(system, guess, quick: bool = False):
    """Solve and verify actuator, chart, recovery, and terminal-hover limits."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={
            "tol": 2.0e-7 if quick else 2.0e-9,
            "acceptable_tol": 1.0e-6 if quick else 1.0e-8,
            "max_iter": 1_500,
            "mu_strategy": "adaptive",
            "print_level": 0,
            "sb": "yes",
            "bound_relax_factor": 0.0,
        },
    )
    status = int(info["status"])
    message = _status_message(info)
    if status not in (0, 1):
        raise RuntimeError(f"Ipopt failed ({status}): {message}")

    phase_solution, final_state = solution
    final_state = np.asarray(final_state, dtype=float)
    endpoint_error = float(
        np.max(
            np.abs(
                np.asarray([phase_solution.x[index][-1] for index in range(12)])
                - final_state
            )
        )
    )
    _, dense_state, dense_control = _dense_history(phase_solution)
    require_finite(
        final_state=final_state,
        state=dense_state,
        control=dense_control,
        objective=info["obj_val"],
    )
    dense_rotor_command = dense_state[12:16]
    rotor_speed = HOVER_ROTOR_SPEED * np.sqrt(np.maximum(dense_rotor_command, 0.0))
    chart_margin = MAX_MRP_NORM - np.linalg.norm(dense_state[6:9], axis=0)
    command_violation = max(
        float(np.max(-dense_rotor_command)),
        float(np.max(dense_rotor_command - MAX_ROTOR_COMMAND)),
        0.0,
    )
    command_rate_violation = max(
        float(np.max(np.abs(dense_control) - MAX_ROTOR_COMMAND_RATE)), 0.0
    )
    terminal_command_error = float(np.max(np.abs(dense_rotor_command[:, -1] - 1.0)))
    terminal_linear_acceleration, terminal_angular_acceleration = (
        _rigid_body_accelerations(dense_state[:, -1])
    )
    terminal_linear_acceleration_norm = float(
        np.linalg.norm(terminal_linear_acceleration)
    )
    terminal_angular_acceleration_norm = float(
        np.linalg.norm(terminal_angular_acceleration)
    )
    final_position_error = float(np.linalg.norm(final_state[:3]))
    final_velocity_error = float(np.linalg.norm(final_state[3:6]))
    final_attitude_error = float(_principal_angles(final_state[6:9, np.newaxis])[0])
    final_rate_error = float(np.linalg.norm(final_state[9:12]))

    recovery_limits = (0.08, 0.08, np.deg2rad(3.0), 0.08)
    recovery_errors = (
        final_position_error,
        final_velocity_error,
        final_attitude_error,
        final_rate_error,
    )
    if endpoint_error > 3.0e-7:
        raise RuntimeError(
            f"endpoint consistency error is too large: {endpoint_error:.3e}"
        )
    if command_violation > 2.0e-7:
        raise RuntimeError(f"dense rotor-command violation: {command_violation:.3e}")
    if command_rate_violation > 2.0e-7:
        raise RuntimeError(
            f"dense rotor-command-rate violation: {command_rate_violation:.3e}"
        )
    if terminal_command_error > 3.0e-7:
        raise RuntimeError(
            f"terminal hover-command error is too large: {terminal_command_error:.3e}"
        )
    if float(np.min(chart_margin)) < -2.0e-7:
        raise RuntimeError("the trajectory left the bounded local MRP chart")
    if any(error > limit for error, limit in zip(recovery_errors, recovery_limits)):
        raise RuntimeError(
            "terminal recovery target was not reached: "
            f"position={final_position_error:.3e} m, "
            f"velocity={final_velocity_error:.3e} m/s, "
            f"attitude={np.rad2deg(final_attitude_error):.3e} deg, "
            f"rate={final_rate_error:.3e} rad/s"
        )
    if terminal_linear_acceleration_norm > 0.02:
        raise RuntimeError(
            "terminal linear acceleration is inconsistent with hover: "
            f"{terminal_linear_acceleration_norm:.3e} m/s^2"
        )
    if terminal_angular_acceleration_norm > 0.02:
        raise RuntimeError(
            "terminal angular acceleration is inconsistent with hover: "
            f"{terminal_angular_acceleration_norm:.3e} rad/s^2"
        )

    print(f"Ipopt status: {message}")
    print(f"Objective: {float(info['obj_val']):.8f}")
    print(f"Terminal position error: {final_position_error:.6f} m")
    print(f"Terminal velocity error: {final_velocity_error:.6f} m/s")
    print(f"Terminal attitude error: {np.rad2deg(final_attitude_error):.6f} deg")
    print(f"Terminal angular-rate error: {final_rate_error:.6f} rad/s")
    print(f"Terminal rotor-command error: {terminal_command_error:.3e}")
    print(
        "Terminal linear/angular acceleration: "
        f"{terminal_linear_acceleration_norm:.3e} m/s^2 / "
        f"{terminal_angular_acceleration_norm:.3e} rad/s^2"
    )
    print(
        "Dense rotor-speed range: "
        f"[{np.min(rotor_speed):.2f}, {np.max(rotor_speed):.2f}] rad/s"
    )
    print(
        "Peak total thrust: "
        f"{ROTOR_HOVER_THRUST * np.max(np.sum(dense_rotor_command, axis=0)):.4f} N"
    )
    print(f"Peak rotor-command rate: {np.max(np.abs(dense_control)):.4f} 1/s")
    print(f"Minimum local-chart margin: {np.min(chart_margin):.6f}")
    return solution


def plot_solution(solution, *, save=None, show=True):
    """Plot the recovery path, state errors, rotor speeds, and command rates."""
    configure_matplotlib()
    phase_solution, _ = solution
    time, state, control = _dense_history(phase_solution, 2_001)
    rotor_speed = HOVER_ROTOR_SPEED * np.sqrt(np.maximum(state[12:16], 0.0))
    attitude_angle = np.rad2deg(_principal_angles(state[6:9]))

    fig = plt.figure(figsize=(10.4, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    path_axis = fig.add_subplot(grid[0, 0], projection="3d")
    state_axis = fig.add_subplot(grid[0, 1])
    rotor_axis = fig.add_subplot(grid[1, 0])
    rate_axis = fig.add_subplot(grid[1, 1], sharex=state_axis)

    path_axis.plot(
        state[0], state[1], state[2], color=COLORS["blue"], label="Recovery path"
    )
    path_axis.scatter(
        *INITIAL_POSITION,
        color=COLORS["vermillion"],
        marker="o",
        s=34,
        label="Initial state",
    )
    path_axis.scatter(
        state[0, -1],
        state[1, -1],
        state[2, -1],
        color=COLORS["green"],
        marker="^",
        s=38,
        label="Recovered hover",
    )
    path_axis.set_xlabel("Position x [m]")
    path_axis.set_ylabel("Position y [m]")
    path_axis.set_zlabel("Position z [m]", labelpad=8)
    coordinate_midpoint = 0.5 * (np.max(state[:3], axis=1) + np.min(state[:3], axis=1))
    coordinate_span = max(float(np.max(np.ptp(state[:3], axis=1))), 1.0e-6)
    coordinate_half_span = 0.52 * coordinate_span
    path_axis.set_xlim(
        coordinate_midpoint[0] - coordinate_half_span,
        coordinate_midpoint[0] + coordinate_half_span,
    )
    path_axis.set_ylim(
        coordinate_midpoint[1] - coordinate_half_span,
        coordinate_midpoint[1] + coordinate_half_span,
    )
    path_axis.set_zlim(
        coordinate_midpoint[2] - coordinate_half_span,
        coordinate_midpoint[2] + coordinate_half_span,
    )
    path_axis.set_box_aspect((1.0, 1.0, 1.0))
    path_axis.view_init(elev=24, azim=45)
    path_axis.set_title("6-DoF quadrotor recovery")
    path_axis.legend(loc="upper left")

    state_axis.plot(
        time,
        np.linalg.norm(state[:3], axis=0) / POSITION_ERROR_SCALE,
        color=COLORS["blue"],
        label="Normalized position",
    )
    state_axis.plot(
        time,
        np.linalg.norm(state[3:6], axis=0) / VELOCITY_ERROR_SCALE,
        color=COLORS["green"],
        label="Normalized velocity",
    )
    attitude_axis = state_axis.twinx()
    attitude_axis.spines["right"].set_visible(True)
    attitude_axis.plot(time, attitude_angle, color=COLORS["orange"], label="Attitude")
    state_axis.set_ylabel("Normalized error [-]")
    attitude_axis.set_ylabel("Principal attitude error [deg]")
    state_axis.set_title("Recovery errors")
    state_handles, state_labels = state_axis.get_legend_handles_labels()
    attitude_handles, attitude_labels = attitude_axis.get_legend_handles_labels()
    state_axis.legend(state_handles + attitude_handles, state_labels + attitude_labels)

    rotor_colors = (
        COLORS["blue"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["purple"],
    )
    for index, color in enumerate(rotor_colors):
        rotor_axis.plot(
            time, rotor_speed[index], color=color, label=f"Rotor {index + 1}"
        )
        rate_axis.plot(time, control[index], color=color, label=f"Rotor {index + 1}")
    rotor_axis.axhline(
        HOVER_ROTOR_SPEED,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.1,
        label="Hover speed",
    )
    rotor_axis.set_xlabel("Time [s]")
    rotor_axis.set_ylabel("Rotor speed [rad/s]")
    rotor_axis.set_title("Rate-limited rotor states")
    rotor_axis.legend(ncol=3, fontsize=8)

    rate_axis.axhline(
        MAX_ROTOR_COMMAND_RATE,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.0,
        label="Rate bounds",
    )
    rate_axis.axhline(
        -MAX_ROTOR_COMMAND_RATE,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.0,
    )
    rate_axis.set_xlabel("Time [s]")
    rate_axis.set_ylabel("Squared-speed-ratio rate [1/s]")
    rate_axis.set_title("Actuator slew controls")
    rate_axis.legend(ncol=3, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.90))

    style_axes([path_axis, state_axis, attitude_axis, rotor_axis, rate_axis])
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "drone_stabilization_solution.png", quick=True
    )
    initial_rotation = _mrp_rotation_numeric(INITIAL_MRP)
    np.testing.assert_allclose(
        initial_rotation.T @ initial_rotation, np.eye(3), rtol=0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(np.linalg.det(initial_rotation), 1.0, atol=2.0e-14)
    hover_acceleration = _rigid_body_accelerations(
        np.concatenate([np.zeros(12), np.ones(4)])
    )
    np.testing.assert_allclose(hover_acceleration[0], np.zeros(3), atol=2.0e-14)
    np.testing.assert_allclose(hover_acceleration[1], np.zeros(3), atol=2.0e-14)

    system, phase = build_problem(quick=args.quick)
    solution = solve_problem(system, initial_guess(phase), quick=args.quick)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
