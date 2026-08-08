"""Demonstrate acceleration-level OSC/WBC on a planar humanoid upper body.

This is a teaching model with identity joint-space acceleration dynamics. It
shows an exact task hierarchy: the right hand follows a primary Cartesian
trajectory, while the left hand and torso use the remaining null space. It is
not production whole-body control for a floating-base robot: contact forces,
rigid-body inertia, torque limits, and balance constraints are intentionally
outside the scope of this compact example. The hand tasks are expressed in a
stabilized shoulder frame so the torso and opposite arm form a clear null
space for teaching purposes.
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


TORSO_LENGTH = 0.60
UPPER_ARM_LENGTH = 0.38
FOREARM_LENGTH = 0.30
HORIZON = 2.5
PRIMARY_KP = 36.0
PRIMARY_KD = 12.0
LEFT_POSITION_WEIGHT = 120.0
LEFT_VELOCITY_WEIGHT = 3.0
TORSO_WEIGHT = 20.0
JOINT_VELOCITY_WEIGHT = 0.03
NULL_ACCELERATION_WEIGHT = 0.01
MAX_JOINT_SPEED = 3.0
MAX_NULL_ACCELERATION = 10.0
JOINT_LOWER_BOUNDS = np.array([-0.55, -1.8, 0.35, -2.2, -2.2])
JOINT_UPPER_BOUNDS = np.array([0.55, 1.2, 1.8, 2.2, 2.2])
DENSE_CHECK_POINTS = 4_001


def _joint_positions(joint_angles):
    """Return pelvis, shoulder, elbow, and hand positions for plotting."""
    torso, right_shoulder, right_elbow, left_shoulder, left_elbow = np.asarray(
        joint_angles, dtype=float
    )
    shoulder = np.array([0.0, TORSO_LENGTH])
    pelvis = shoulder - TORSO_LENGTH * np.array([np.sin(torso), np.cos(torso)])

    right_upper_angle = right_shoulder
    right_forearm_angle = right_upper_angle + right_elbow
    right_elbow_position = shoulder + UPPER_ARM_LENGTH * np.array(
        [np.cos(right_upper_angle), np.sin(right_upper_angle)]
    )
    right_hand = right_elbow_position + FOREARM_LENGTH * np.array(
        [np.cos(right_forearm_angle), np.sin(right_forearm_angle)]
    )

    left_upper_angle = left_shoulder
    left_forearm_angle = left_upper_angle + left_elbow
    left_elbow_position = shoulder + UPPER_ARM_LENGTH * np.array(
        [-np.cos(left_upper_angle), np.sin(left_upper_angle)]
    )
    left_hand = left_elbow_position + FOREARM_LENGTH * np.array(
        [-np.cos(left_forearm_angle), np.sin(left_forearm_angle)]
    )
    return {
        "pelvis": pelvis,
        "shoulder": shoulder,
        "right_elbow": right_elbow_position,
        "right_hand": right_hand,
        "left_elbow": left_elbow_position,
        "left_hand": left_hand,
    }


INITIAL_JOINT_ANGLES = np.array([0.25, -0.45, 0.95, 0.60, -1.10])
LEFT_TARGET_CONFIGURATION = np.array([0.0, -0.70, 1.10, -0.45, 1.00])
RIGHT_INITIAL_POSITION = _joint_positions(INITIAL_JOINT_ANGLES)["right_hand"]
RIGHT_DISPLACEMENT = np.array([0.04, 0.08])
LEFT_TARGET_POSITION = _joint_positions(LEFT_TARGET_CONFIGURATION)["left_hand"]


def _symbolic_hand_positions(joint_angles):
    _, right_shoulder, right_elbow, left_shoulder, left_elbow = joint_angles
    shoulder = sp.Matrix([0.0, TORSO_LENGTH])

    right_upper_angle = right_shoulder
    right_forearm_angle = right_upper_angle + right_elbow
    right_hand = shoulder + sp.Matrix(
        [
            UPPER_ARM_LENGTH * sp.cos(right_upper_angle)
            + FOREARM_LENGTH * sp.cos(right_forearm_angle),
            UPPER_ARM_LENGTH * sp.sin(right_upper_angle)
            + FOREARM_LENGTH * sp.sin(right_forearm_angle),
        ]
    )

    left_upper_angle = left_shoulder
    left_forearm_angle = left_upper_angle + left_elbow
    left_hand = shoulder + sp.Matrix(
        [
            -UPPER_ARM_LENGTH * sp.cos(left_upper_angle)
            - FOREARM_LENGTH * sp.cos(left_forearm_angle),
            UPPER_ARM_LENGTH * sp.sin(left_upper_angle)
            + FOREARM_LENGTH * sp.sin(left_forearm_angle),
        ]
    )
    return right_hand, left_hand


def _right_jacobian_numeric(joint_angles):
    """Evaluate the analytical right-hand Jacobian."""
    _, right_shoulder, right_elbow, _, _ = np.asarray(joint_angles, dtype=float)
    upper_angle = right_shoulder
    forearm_angle = upper_angle + right_elbow
    jacobian = np.zeros((2, 5))
    jacobian[:, 1] = [
        -UPPER_ARM_LENGTH * np.sin(upper_angle)
        - FOREARM_LENGTH * np.sin(forearm_angle),
        UPPER_ARM_LENGTH * np.cos(upper_angle) + FOREARM_LENGTH * np.cos(forearm_angle),
    ]
    jacobian[:, 2] = [
        -FOREARM_LENGTH * np.sin(forearm_angle),
        FOREARM_LENGTH * np.cos(forearm_angle),
    ]
    return jacobian


def _quintic_reference(time):
    """Evaluate the primary right-hand rest-to-rest reference position."""
    tau = np.asarray(time, dtype=float) / HORIZON
    progress = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    return RIGHT_INITIAL_POSITION[:, np.newaxis] + np.outer(
        RIGHT_DISPLACEMENT, np.atleast_1d(progress)
    )


def build_problem():
    """Build the hierarchical acceleration-level OSC/WBC problem."""
    system = System(0)
    phase = system.new_phase(
        [
            "torso_angle",
            "right_shoulder_angle",
            "right_elbow_angle",
            "left_shoulder_angle",
            "left_elbow_angle",
            "torso_rate",
            "right_shoulder_rate",
            "right_elbow_rate",
            "left_shoulder_rate",
            "left_elbow_rate",
        ],
        [
            "null_torso_acceleration",
            "null_right_shoulder_acceleration",
            "null_right_elbow_acceleration",
            "null_left_shoulder_acceleration",
            "null_left_elbow_acceleration",
        ],
    )
    joint_angles = sp.Matrix(phase.x[:5])
    joint_velocities = sp.Matrix(phase.x[5:])
    null_acceleration = sp.Matrix(phase.u)

    right_hand, left_hand = _symbolic_hand_positions(joint_angles)
    right_jacobian = right_hand.jacobian(joint_angles)
    left_jacobian = left_hand.jacobian(joint_angles)
    right_jacobian_dot = sp.zeros(2, 5)
    for joint_index in range(5):
        right_jacobian_dot += (
            right_jacobian.diff(joint_angles[joint_index])
            * joint_velocities[joint_index]
        )

    # J# = J.T * (J * J.T)^-1. The closed form below is exact for the
    # nonsingular two-link right arm and keeps automatic differentiation
    # compact.
    arm_jacobian = right_jacobian[:, 1:3]
    arm_determinant = (
        arm_jacobian[0, 0] * arm_jacobian[1, 1]
        - arm_jacobian[0, 1] * arm_jacobian[1, 0]
    )
    arm_inverse = (
        sp.Matrix(
            [
                [arm_jacobian[1, 1], -arm_jacobian[0, 1]],
                [-arm_jacobian[1, 0], arm_jacobian[0, 0]],
            ]
        )
        / arm_determinant
    )
    right_pseudoinverse = sp.zeros(5, 2)
    right_pseudoinverse[1:3, :] = arm_inverse
    # N = I - J# * J, simplified exactly for this stabilized shoulder model.
    null_projector = sp.diag(1.0, 0.0, 0.0, 1.0, 1.0)

    tau = phase.t / HORIZON
    progress = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    progress_rate = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / HORIZON
    progress_acceleration = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / HORIZON**2
    desired_position = (
        sp.Matrix(RIGHT_INITIAL_POSITION) + sp.Matrix(RIGHT_DISPLACEMENT) * progress
    )
    desired_velocity = sp.Matrix(RIGHT_DISPLACEMENT) * progress_rate
    desired_acceleration = sp.Matrix(RIGHT_DISPLACEMENT) * progress_acceleration

    right_velocity = right_jacobian * joint_velocities
    acceleration_reference = (
        desired_acceleration
        + PRIMARY_KP * (desired_position - right_hand)
        + PRIMARY_KD * (desired_velocity - right_velocity)
    )
    primary_acceleration = right_pseudoinverse * (
        acceleration_reference - right_jacobian_dot * joint_velocities
    )
    # qdd = J# * (a_ref - Jdot * qdot) + N * z.
    joint_acceleration = primary_acceleration + null_projector * null_acceleration

    phase.set_dynamics([*joint_velocities, *joint_acceleration])

    left_error = left_hand - sp.Matrix(LEFT_TARGET_POSITION)
    left_velocity = left_jacobian * joint_velocities
    running_cost = (
        LEFT_POSITION_WEIGHT * left_error.dot(left_error)
        + LEFT_VELOCITY_WEIGHT * left_velocity.dot(left_velocity)
        + TORSO_WEIGHT * joint_angles[0] ** 2
        + JOINT_VELOCITY_WEIGHT * joint_velocities.dot(joint_velocities)
        + NULL_ACCELERATION_WEIGHT * null_acceleration.dot(null_acceleration)
    )
    phase.set_integral([running_cost])

    phase.set_phase_constraint(
        [*joint_angles, *joint_velocities, *null_acceleration],
        [
            *JOINT_LOWER_BOUNDS,
            *([-MAX_JOINT_SPEED] * 5),
            *([-MAX_NULL_ACCELERATION] * 5),
        ],
        [
            *JOINT_UPPER_BOUNDS,
            *([MAX_JOINT_SPEED] * 5),
            *([MAX_NULL_ACCELERATION] * 5),
        ],
    )
    phase.set_boundary_condition(
        [*INITIAL_JOINT_ANGLES, *np.zeros(5)],
        [None] * 10,
        0.0,
        HORIZON,
    )
    phase.set_discretization(10, 4)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct a smooth joint-space guess toward the secondary targets."""
    guess = linear_guess(phase, 0.0)
    final_guess = np.array([0.03, -0.36, 0.92, -0.45, 1.00])
    displacement = final_guess - INITIAL_JOINT_ANGLES

    tau_x = guess.t_x / HORIZON
    progress_x = 10.0 * tau_x**3 - 15.0 * tau_x**4 + 6.0 * tau_x**5
    rate_x = (30.0 * tau_x**2 - 60.0 * tau_x**3 + 30.0 * tau_x**4) / HORIZON
    for joint_index in range(5):
        guess.x[joint_index] = (
            INITIAL_JOINT_ANGLES[joint_index] + displacement[joint_index] * progress_x
        )
        guess.x[5 + joint_index] = displacement[joint_index] * rate_x

    tau_u = guess.t_u / HORIZON
    acceleration_u = (60.0 * tau_u - 180.0 * tau_u**2 + 120.0 * tau_u**3) / HORIZON**2
    for joint_index in range(5):
        guess.u[joint_index] = displacement[joint_index] * acceleration_u
    return guess


def _dense_history(solution, count: int = DENSE_CHECK_POINTS):
    """Interpolate all states and controls onto a uniform physical-time grid."""
    time = np.linspace(solution.t_0, solution.t_f, count)
    state_interpolation = solution.V_x(time.copy())
    control_interpolation = solution.V_u(time.copy())
    state = np.vstack([state_interpolation @ component for component in solution.x])
    control = np.vstack([control_interpolation @ component for component in solution.u])
    return time, state, control


def _solution_diagnostics(time, state, desired_right=None):
    """Evaluate hand-task and hierarchy diagnostics for supplied histories."""
    joint_history = np.asarray(state[:5], dtype=float)
    right_history = []
    left_history = []
    null_leakage = []
    for column in range(joint_history.shape[1]):
        joint_angles = joint_history[:, column]
        positions = _joint_positions(joint_angles)
        right_history.append(positions["right_hand"])
        left_history.append(positions["left_hand"])

        jacobian = _right_jacobian_numeric(joint_angles)
        pseudoinverse = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T)
        null_projector = np.eye(5) - pseudoinverse @ jacobian
        null_leakage.append(np.linalg.norm(jacobian @ null_projector))

    right_history = np.asarray(right_history).T
    left_history = np.asarray(left_history).T
    if desired_right is None:
        desired_right = _quintic_reference(time)
    right_error = np.linalg.norm(right_history - desired_right, axis=0)
    left_error = np.linalg.norm(
        left_history - LEFT_TARGET_POSITION[:, np.newaxis], axis=0
    )
    return {
        "right_hand": right_history,
        "left_hand": left_history,
        "desired_right": desired_right,
        "right_error": right_error,
        "left_error": left_error,
        "null_leakage": np.asarray(null_leakage),
    }


def solve_problem(system, guess):
    """Solve and numerically verify the primary/null-space hierarchy."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={
            "tol": 2.0e-8,
            "acceptable_tol": 1.0e-7,
            "max_iter": 2500,
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

    dense_time, dense_state, dense_control = _dense_history(solution)
    desired_right = _quintic_reference(dense_time)
    require_finite(
        time=dense_time,
        state=dense_state,
        control=dense_control,
        objective=info["obj_val"],
        right_hand_reference=desired_right,
    )
    diagnostics = _solution_diagnostics(dense_time, dense_state, desired_right)
    require_finite(**diagnostics)
    maximum_leakage = float(np.max(diagnostics["null_leakage"]))
    maximum_primary_error = float(np.max(diagnostics["right_error"]))
    initial_left_error = float(diagnostics["left_error"][0])
    final_left_error = float(diagnostics["left_error"][-1])
    bound_violation = max(
        float(np.max(JOINT_LOWER_BOUNDS[:, np.newaxis] - dense_state[:5])),
        float(np.max(dense_state[:5] - JOINT_UPPER_BOUNDS[:, np.newaxis])),
        float(np.max(np.abs(dense_state[5:]) - MAX_JOINT_SPEED)),
        float(np.max(np.abs(dense_control) - MAX_NULL_ACCELERATION)),
        0.0,
    )
    if maximum_leakage >= 1.0e-10:
        raise RuntimeError(f"null-space leakage is too large: {maximum_leakage:.3e}")
    if maximum_primary_error >= 2.0e-3:
        raise RuntimeError(
            f"right-hand tracking error is too large: {maximum_primary_error:.3e} m"
        )
    if final_left_error >= 0.65 * initial_left_error:
        raise RuntimeError(
            "the secondary hand task did not reduce its error sufficiently: "
            f"{initial_left_error:.3e} m -> {final_left_error:.3e} m"
        )
    if bound_violation > 2.0e-7:
        raise RuntimeError(f"dense path-bound violation: {bound_violation:.3e}")

    print(f"Ipopt status: {status_message}")
    print(f"Maximum ||J N||: {maximum_leakage:.3e}")
    print(f"Maximum right-hand tracking error: {maximum_primary_error:.3e} m")
    print(
        "Left-hand target error: "
        f"{initial_left_error:.4f} m -> {final_left_error:.4f} m"
    )
    print(f"Maximum dense path-bound violation: {bound_violation:.3e}")
    return solution


def plot_solution(solution, *, save=None, show=True):
    """Plot arm poses, hand trajectories, and hierarchy diagnostics."""
    configure_matplotlib()
    dense_time, dense_state, _ = _dense_history(solution)
    diagnostics = _solution_diagnostics(dense_time, dense_state)
    fig = plt.figure(figsize=(10.2, 6.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0))
    pose_axis = fig.add_subplot(grid[:, 0])
    tracking_axis = fig.add_subplot(grid[0, 1])
    hierarchy_axis = fig.add_subplot(grid[1, 1], sharex=tracking_axis)

    pose_indices = np.unique(np.linspace(0, dense_time.size - 1, 6, dtype=int))
    for index in pose_indices:
        joint_angles = dense_state[:5, index]
        positions = _joint_positions(joint_angles)
        alpha = 0.28 if index != pose_indices[-1] else 0.95
        pose_axis.plot(
            [positions["pelvis"][0], positions["shoulder"][0]],
            [positions["pelvis"][1], positions["shoulder"][1]],
            color=COLORS["black"],
            linewidth=2.2,
            alpha=alpha,
        )
        pose_axis.plot(
            [
                positions["shoulder"][0],
                positions["right_elbow"][0],
                positions["right_hand"][0],
            ],
            [
                positions["shoulder"][1],
                positions["right_elbow"][1],
                positions["right_hand"][1],
            ],
            color=COLORS["blue"],
            marker="o",
            markersize=2.8,
            linewidth=1.5,
            alpha=alpha,
        )
        pose_axis.plot(
            [
                positions["shoulder"][0],
                positions["left_elbow"][0],
                positions["left_hand"][0],
            ],
            [
                positions["shoulder"][1],
                positions["left_elbow"][1],
                positions["left_hand"][1],
            ],
            color=COLORS["orange"],
            marker="o",
            markersize=2.8,
            linewidth=1.5,
            alpha=alpha,
        )

    pose_axis.plot(
        diagnostics["desired_right"][0],
        diagnostics["desired_right"][1],
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.4,
        label="Right-hand reference",
    )
    pose_axis.plot(
        diagnostics["right_hand"][0],
        diagnostics["right_hand"][1],
        color=COLORS["blue"],
        label="Right hand",
    )
    pose_axis.plot(
        diagnostics["left_hand"][0],
        diagnostics["left_hand"][1],
        color=COLORS["orange"],
        label="Left hand",
    )
    pose_axis.scatter(
        [LEFT_TARGET_POSITION[0]],
        [LEFT_TARGET_POSITION[1]],
        color=COLORS["vermillion"],
        marker="x",
        s=55,
        label="Left-hand target",
        zorder=4,
    )
    pose_axis.set_xlabel("Horizontal position [m]")
    pose_axis.set_ylabel("Vertical position [m]")
    pose_axis.set_title("Planar dual-arm task hierarchy")
    pose_axis.set_aspect("equal", adjustable="box")
    pose_axis.legend(loc="lower left")

    tracking_axis.plot(
        dense_time,
        diagnostics["right_error"],
        color=COLORS["blue"],
        label="Right hand (primary)",
    )
    tracking_axis.plot(
        dense_time,
        diagnostics["left_error"],
        color=COLORS["orange"],
        label="Left hand (secondary)",
    )
    tracking_axis.set_ylabel("Position error [m]")
    tracking_axis.set_title("Task errors")
    tracking_axis.set_yscale("symlog", linthresh=1.0e-9)
    tracking_axis.legend()

    hierarchy_axis.plot(
        dense_time,
        np.rad2deg(np.abs(dense_state[0])),
        color=COLORS["green"],
        label="Torso posture error",
    )
    hierarchy_axis.set_xlabel("Time [s]")
    hierarchy_axis.set_ylabel("Torso posture error [deg]")
    hierarchy_axis.set_title("Null-space verification")

    leakage_axis = hierarchy_axis.twinx()
    leakage_axis.scatter(
        dense_time,
        diagnostics["null_leakage"],
        color=COLORS["purple"],
        s=3.0,
        alpha=0.35,
        linewidths=0.0,
        label=r"$\|J N\|_F$",
    )
    leakage_axis.set_ylabel(r"Null-space leakage $\|J N\|_F$")
    hierarchy_axis.legend(loc="upper right")
    leakage_axis.legend(loc="lower right")

    style_axes([pose_axis, tracking_axis, hierarchy_axis, leakage_axis])
    fig.tight_layout()
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "humanoid_whole_body_control_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
