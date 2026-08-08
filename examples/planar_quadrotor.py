"""Plan a planar quadrotor trajectory around a circular obstacle.

The vehicle moves in the horizontal-vertical plane with one pitch angle and
one angular velocity. Total thrust acts along the body vertical axis, while a
body torque controls pitch; no redundant quaternion variables or unit-norm
constraints are required.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.patches import Circle

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


MASS = 1.20
PITCH_INERTIA = 0.025
GRAVITY = 9.81
HORIZON = 5.0
START = np.array([0.0, 0.0])
TARGET = np.array([5.0, 0.0])
OBSTACLE_CENTER = np.array([2.5, 0.80])
OBSTACLE_RADIUS = 0.80
VEHICLE_CLEARANCE = 0.12
SAFE_RADIUS = OBSTACLE_RADIUS + VEHICLE_CLEARANCE
MAX_THRUST = 2.2 * MASS * GRAVITY
MAX_TORQUE = 0.25
MAX_PITCH = np.deg2rad(65.0)
MAX_PITCH_RATE = 3.0
GROUND_GUARD_HEIGHT = 0.03
DENSE_CHECK_POINTS = 4_001


def build_problem(quick: bool = False):
    """Build the fixed-time planar flight problem with obstacle avoidance."""
    system = System(0, fastmath=True)
    phase = system.new_phase(
        ["x", "z", "velocity_x", "velocity_z", "pitch", "pitch_rate"],
        ["thrust", "torque"],
    )
    x, z, velocity_x, velocity_z, pitch, pitch_rate = phase.x
    thrust, torque = phase.u

    # Pitch is positive counterclockwise. Body thrust is R(pitch) @ [0, T],
    # hence a negative pitch produces positive horizontal acceleration.
    phase.set_dynamics(
        [
            velocity_x,
            velocity_z,
            -thrust * sp.sin(pitch) / MASS,
            thrust * sp.cos(pitch) / MASS - GRAVITY,
            pitch_rate,
            torque / PITCH_INERTIA,
        ]
    )
    hover_thrust = MASS * GRAVITY
    phase.set_integral(
        [
            0.025 * ((thrust - hover_thrust) / hover_thrust) ** 2
            + 0.012 * (torque / MAX_TORQUE) ** 2
            + 0.002 * pitch_rate**2
            + 0.004 * z**2
        ]
    )
    obstacle_distance_squared = (x - OBSTACLE_CENTER[0]) ** 2 + (
        z - OBSTACLE_CENTER[1]
    ) ** 2
    normalized_time = phase.t / HORIZON
    ground_guard = (
        16.0 * GROUND_GUARD_HEIGHT * normalized_time**2 * (1.0 - normalized_time) ** 2
    )
    # A small transcription guard covers interpolation between constrained
    # nodes. Reported clearance is still measured from SAFE_RADIUS below.
    enforced_radius = SAFE_RADIUS + (0.012 if quick else 0.004)
    phase.set_phase_constraint(
        [
            x,
            z,
            pitch,
            pitch_rate,
            thrust,
            torque,
            obstacle_distance_squared,
            z - ground_guard,
        ],
        [
            -0.20,
            0.0,
            -MAX_PITCH,
            -MAX_PITCH_RATE,
            0.0,
            -MAX_TORQUE,
            enforced_radius**2,
            0.0,
        ],
        [
            5.20,
            3.0,
            MAX_PITCH,
            MAX_PITCH_RATE,
            MAX_THRUST,
            MAX_TORQUE,
            np.inf,
            np.inf,
        ],
    )
    phase.set_boundary_condition(
        [START[0], START[1], 0.0, 0.0, 0.0, 0.0],
        [TARGET[0], TARGET[1], 0.0, 0.0, 0.0, 0.0],
        0.0,
        HORIZON,
    )
    phase.set_discretization(8 if quick else 14, 4 if quick else 6)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def _reference_profiles(time: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return a smooth obstacle-clearing reference state at the supplied times."""
    fraction = np.asarray(time) / HORIZON
    progress = 10.0 * fraction**3 - 15.0 * fraction**4 + 6.0 * fraction**5
    progress_rate = (
        30.0 * fraction**2 - 60.0 * fraction**3 + 30.0 * fraction**4
    ) / HORIZON
    progress_acceleration = (
        60.0 * fraction - 180.0 * fraction**2 + 120.0 * fraction**3
    ) / HORIZON**2

    arch_height = 1.82
    x = START[0] + (TARGET[0] - START[0]) * progress
    velocity_x = (TARGET[0] - START[0]) * progress_rate
    acceleration_x = (TARGET[0] - START[0]) * progress_acceleration
    z = arch_height * np.sin(np.pi * fraction) ** 2
    velocity_z = arch_height * np.pi * np.sin(2.0 * np.pi * fraction) / HORIZON
    acceleration_z = (
        2.0 * arch_height * np.pi**2 * np.cos(2.0 * np.pi * fraction) / HORIZON**2
    )

    vertical_specific_force = acceleration_z + GRAVITY
    pitch = -np.arctan2(acceleration_x, vertical_specific_force)
    thrust = MASS * np.hypot(acceleration_x, vertical_specific_force)
    return x, z, velocity_x, velocity_z, pitch, thrust


def initial_guess(phase):
    """Construct a smooth arch that clears the obstacle and nearly obeys dynamics."""
    guess = linear_guess(phase, 0.0)
    x, z, velocity_x, velocity_z, pitch, _ = _reference_profiles(guess.t_x)
    guess.x[0] = x
    guess.x[1] = z
    guess.x[2] = velocity_x
    guess.x[3] = velocity_z
    guess.x[4] = pitch
    guess.x[5] = np.gradient(pitch, guess.t_x, edge_order=2)

    _, _, _, _, pitch_u, thrust_u = _reference_profiles(guess.t_u)
    pitch_rate_u = np.gradient(pitch_u, guess.t_u, edge_order=2)
    torque_u = PITCH_INERTIA * np.gradient(pitch_rate_u, guess.t_u, edge_order=2)
    guess.u[0] = np.clip(thrust_u, 0.0, MAX_THRUST)
    guess.u[1] = np.clip(torque_u, -MAX_TORQUE, MAX_TORQUE)
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_history(
    solution, count: int = DENSE_CHECK_POINTS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate all states and controls onto a uniform physical-time grid."""
    time = np.linspace(solution.t_0, solution.t_f, count)
    state_interpolation = solution.V_x(time.copy())
    control_interpolation = solution.V_u(time.copy())
    state = np.vstack([state_interpolation @ component for component in solution.x])
    control = np.vstack([control_interpolation @ component for component in solution.u])
    return time, state, control


def solve_problem(system, guess, quick: bool = False):
    """Solve the flight problem and check endpoints, clearance, and actuators."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={
            "tol": 1.0e-7 if quick else 1.0e-9,
            "acceptable_tol": 1.0e-6 if quick else 1.0e-8,
            "max_iter": 1_200,
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

    dense_time, dense_state, dense_control = _dense_history(solution)
    require_finite(
        dense_time=dense_time,
        dense_state=dense_state,
        dense_control=dense_control,
        objective=info["obj_val"],
    )
    clearance = (
        np.hypot(
            dense_state[0] - OBSTACLE_CENTER[0],
            dense_state[1] - OBSTACLE_CENTER[1],
        )
        - SAFE_RADIUS
    )
    minimum_clearance = float(np.min(clearance))
    expected_endpoints = np.array(
        [
            [START[0], TARGET[0]],
            [START[1], TARGET[1]],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    actual_endpoints = np.array(dense_state[:, [0, -1]])
    endpoint_error = float(np.max(np.abs(actual_endpoints - expected_endpoints)))
    path_violation = max(
        float(np.max(-0.20 - dense_state[0])),
        float(np.max(dense_state[0] - 5.20)),
        float(np.max(-dense_state[1])),
        float(np.max(dense_state[1] - 3.0)),
        float(np.max(np.abs(dense_state[4]) - MAX_PITCH)),
        float(np.max(np.abs(dense_state[5]) - MAX_PITCH_RATE)),
        float(np.max(-dense_control[0])),
        float(np.max(dense_control[0] - MAX_THRUST)),
        float(np.max(np.abs(dense_control[1]) - MAX_TORQUE)),
        0.0,
    )

    if endpoint_error > 2.0e-7:
        raise RuntimeError(f"endpoint error is too large: {endpoint_error:.3e}")
    if minimum_clearance < (-2.0e-3 if quick else -2.0e-4):
        raise RuntimeError(
            f"dense obstacle clearance is negative: {minimum_clearance:.3e} m"
        )
    if path_violation > 2.0e-7:
        raise RuntimeError(f"dense path-bound violation: {path_violation:.3e}")

    print(f"Ipopt status: {message}")
    print(f"Objective: {float(info['obj_val']):.8f}")
    print(f"Minimum obstacle clearance: {minimum_clearance:.6f} m")
    print(f"Maximum endpoint error: {endpoint_error:.3e}")
    print(f"Maximum dense path-bound violation: {path_violation:.3e}")
    print(f"Peak dense thrust: {np.max(dense_control[0]):.6f} N")
    print(f"Peak dense absolute torque: {np.max(np.abs(dense_control[1])):.6f} N m")
    return solution


def plot_solution(solution, *, save=None, show=True):
    """Plot the path, attitude, actuator histories, and obstacle clearance."""
    configure_matplotlib()
    fig = plt.figure(figsize=(10.0, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0))
    path_axis = fig.add_subplot(grid[:, 0])
    state_axis = fig.add_subplot(grid[0, 1])
    control_axis = fig.add_subplot(grid[1, 1], sharex=state_axis)

    dense_time, dense_state, dense_control = _dense_history(solution)
    path_axis.add_patch(
        Circle(
            OBSTACLE_CENTER,
            OBSTACLE_RADIUS,
            facecolor=COLORS["vermillion"],
            edgecolor="none",
            alpha=0.25,
            label="Obstacle",
        )
    )
    path_axis.add_patch(
        Circle(
            OBSTACLE_CENTER,
            SAFE_RADIUS,
            facecolor="none",
            edgecolor=COLORS["vermillion"],
            linestyle="--",
            linewidth=1.4,
            label="Required clearance",
        )
    )
    path_axis.plot(
        dense_state[0], dense_state[1], color=COLORS["blue"], label="Optimal path"
    )
    path_axis.scatter(*START, color=COLORS["green"], zorder=4, label="Start")
    path_axis.scatter(*TARGET, color=COLORS["purple"], zorder=4, label="Target")

    attitude_indices = np.linspace(0, dense_time.size - 1, 9, dtype=int)
    body_half_length = 0.16
    for index in attitude_indices:
        direction = body_half_length * np.array(
            [np.cos(dense_state[4, index]), np.sin(dense_state[4, index])]
        )
        center = dense_state[:2, index]
        path_axis.plot(
            [center[0] - direction[0], center[0] + direction[0]],
            [center[1] - direction[1], center[1] + direction[1]],
            color=COLORS["black"],
            linewidth=1.0,
            alpha=0.65,
        )
    path_axis.set_xlabel("Horizontal position x [m]")
    path_axis.set_ylabel("Altitude z [m]")
    path_axis.set_title("Planar obstacle-avoidance trajectory")
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_xlim(-0.25, 5.25)
    path_axis.set_ylim(-0.10, 2.45)
    path_axis.legend(fontsize=8, ncol=2, loc="upper center")

    state_axis.plot(dense_time, dense_state[2], color=COLORS["blue"], label=r"$v_x$")
    state_axis.plot(dense_time, dense_state[3], color=COLORS["green"], label=r"$v_z$")
    pitch_axis = state_axis.twinx()
    pitch_axis.spines["right"].set_visible(True)
    pitch_axis.plot(
        dense_time,
        np.rad2deg(dense_state[4]),
        color=COLORS["orange"],
        label=r"Pitch $\theta$",
    )
    state_axis.set_ylabel("Velocity [m/s]")
    pitch_axis.set_ylabel("Pitch [deg]")
    state_axis.set_title("Velocity and attitude")
    state_handles, state_labels = state_axis.get_legend_handles_labels()
    pitch_handles, pitch_labels = pitch_axis.get_legend_handles_labels()
    state_axis.legend(
        state_handles + pitch_handles, state_labels + pitch_labels, fontsize=8, ncol=3
    )

    control_axis.plot(
        dense_time, dense_control[0], color=COLORS["purple"], label="Thrust"
    )
    torque_axis = control_axis.twinx()
    torque_axis.spines["right"].set_visible(True)
    torque_axis.plot(
        dense_time,
        dense_control[1],
        color=COLORS["vermillion"],
        label="Torque",
    )
    control_axis.axhline(
        MASS * GRAVITY,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.0,
        label="Hover thrust",
    )
    control_axis.set_xlabel("Time [s]")
    control_axis.set_ylabel("Total thrust [N]")
    torque_axis.set_ylabel("Pitch torque [N m]")
    control_axis.set_title("Total thrust and pitch torque")
    thrust_handles, thrust_labels = control_axis.get_legend_handles_labels()
    torque_handles, torque_labels = torque_axis.get_legend_handles_labels()
    control_axis.legend(
        thrust_handles + torque_handles, thrust_labels + torque_labels, fontsize=8
    )

    style_axes([path_axis, state_axis, control_axis])
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "planar_quadrotor_solution.png", quick=True
    )
    system, phase = build_problem(quick=args.quick)
    solution = solve_problem(system, initial_guess(phase), quick=args.quick)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
