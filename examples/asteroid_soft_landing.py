"""Plan a minimum-effort soft landing in asteroid-fixed coordinates.

The spacecraft descends from a nearby body-fixed point to a landing site on
the equator of asteroid (101955) Bennu with a 2 mm/s normal contact speed.  The
planar dynamics include spherical gravity plus Coriolis and centrifugal
acceleration in Bennu's uniformly rotating frame.  Selected 2019 estimates of
Bennu's mean radius, gravitational parameter, and sidereal rotation period set
the physical scales; the spacecraft acceleration limit is an explicit
mission-design assumption rather than asteroid data.

The guidance-level model assumes perfect tracking of a planar thrust-
acceleration command.  It neglects Bennu's irregular gravity, solar radiation
pressure, navigation uncertainty, attitude dynamics, plume interaction, and
terrain.  It is therefore a transparent near-body trajectory-optimization
example, not flight-qualified landing guidance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, trapezoid

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


# Rounded early-mission 2019 OSIRIS-REx estimates for Bennu's bulk properties.
BENNU_MEAN_RADIUS = 245.03  # m
BENNU_MU = 4.892  # m^3/s^2
BENNU_ROTATION_PERIOD = 4.296057 * 3600.0  # s
BENNU_ROTATION_RATE = 2.0 * np.pi / BENNU_ROTATION_PERIOD  # rad/s

# Mission-design quantities for a guidance-level low-thrust spacecraft.
MAX_THRUST_ACCELERATION = 2.0e-4  # m/s^2
FLIGHT_TIME = 2.5 * 3600.0  # s
CONTACT_SPEED = 2.0e-3  # m/s, inward normal velocity at first surface contact
MIN_GLIDE_SLOPE_ANGLE = np.deg2rad(20.0)  # rad above the local tangent plane
TRANSCRIPTION_GLIDE_SLOPE_ANGLE = np.deg2rad(21.0)  # 1 deg interpolation guard
INITIAL_POSITION = np.array([2.20, -0.50]) * BENNU_MEAN_RADIUS  # m
INITIAL_VELOCITY = np.zeros(2)  # m/s in the asteroid-fixed frame
TARGET_POSITION = np.array([BENNU_MEAN_RADIUS, 0.0])  # m
TARGET_VELOCITY = np.array([-CONTACT_SPEED, 0.0])  # m/s, body-fixed frame

# Natural scales make gravity order one in the nonlinear program.
TIME_SCALE = np.sqrt(BENNU_MEAN_RADIUS**3 / BENNU_MU)  # s
VELOCITY_SCALE = BENNU_MEAN_RADIUS / TIME_SCALE  # m/s
ACCELERATION_SCALE = BENNU_MU / BENNU_MEAN_RADIUS**2  # m/s^2
SCALED_ROTATION_RATE = BENNU_ROTATION_RATE * TIME_SCALE
SCALED_MAX_ACCELERATION = MAX_THRUST_ACCELERATION / ACCELERATION_SCALE
SCALED_FLIGHT_TIME = FLIGHT_TIME / TIME_SCALE
DENSE_CHECK_POINTS = 4_001


def _hermite_trajectory(time: np.ndarray):
    """Return endpoint-matching scaled position, velocity, and acceleration."""
    fraction = np.asarray(time, dtype=float) / SCALED_FLIGHT_TIME
    initial_position = INITIAL_POSITION / BENNU_MEAN_RADIUS
    final_position = TARGET_POSITION / BENNU_MEAN_RADIUS
    initial_velocity = INITIAL_VELOCITY / VELOCITY_SCALE
    final_velocity = TARGET_VELOCITY / VELOCITY_SCALE

    h00 = 2.0 * fraction**3 - 3.0 * fraction**2 + 1.0
    h10 = fraction**3 - 2.0 * fraction**2 + fraction
    h01 = -2.0 * fraction**3 + 3.0 * fraction**2
    h11 = fraction**3 - fraction**2
    position = (
        np.outer(initial_position, h00)
        + np.outer(SCALED_FLIGHT_TIME * initial_velocity, h10)
        + np.outer(final_position, h01)
        + np.outer(SCALED_FLIGHT_TIME * final_velocity, h11)
    )

    dh00 = 6.0 * fraction**2 - 6.0 * fraction
    dh10 = 3.0 * fraction**2 - 4.0 * fraction + 1.0
    dh01 = -6.0 * fraction**2 + 6.0 * fraction
    dh11 = 3.0 * fraction**2 - 2.0 * fraction
    velocity = (
        np.outer(initial_position, dh00)
        + np.outer(SCALED_FLIGHT_TIME * initial_velocity, dh10)
        + np.outer(final_position, dh01)
        + np.outer(SCALED_FLIGHT_TIME * final_velocity, dh11)
    ) / SCALED_FLIGHT_TIME

    ddh00 = 12.0 * fraction - 6.0
    ddh10 = 6.0 * fraction - 4.0
    ddh01 = -12.0 * fraction + 6.0
    ddh11 = 6.0 * fraction - 2.0
    acceleration = (
        np.outer(initial_position, ddh00)
        + np.outer(SCALED_FLIGHT_TIME * initial_velocity, ddh10)
        + np.outer(final_position, ddh01)
        + np.outer(SCALED_FLIGHT_TIME * final_velocity, ddh11)
    ) / SCALED_FLIGHT_TIME**2
    return position, velocity, acceleration


def _scaled_dynamics(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Evaluate the nondimensional rotating-frame point-mass dynamics."""
    position = np.asarray(state[:2], dtype=float)
    velocity = np.asarray(state[2:], dtype=float)
    radius = np.linalg.norm(position)
    omega = SCALED_ROTATION_RATE
    coriolis = np.array([2.0 * omega * velocity[1], -2.0 * omega * velocity[0]])
    acceleration = -position / radius**3 + coriolis + omega**2 * position + control
    return np.concatenate((velocity, acceleration))


def build_problem(quick: bool = False):
    """Build the fixed-time, minimum-squared-acceleration landing problem."""
    system = System(0, fastmath=True)
    phase = system.new_phase(
        ["scaled_x", "scaled_y", "scaled_velocity_x", "scaled_velocity_y"],
        ["scaled_acceleration_x", "scaled_acceleration_y"],
    )
    x_position, y_position, x_velocity, y_velocity = phase.x
    x_control, y_control = phase.u
    radius_squared = x_position**2 + y_position**2
    radius_cubed = radius_squared**1.5
    omega = SCALED_ROTATION_RATE

    phase.set_dynamics(
        [
            x_velocity,
            y_velocity,
            -x_position / radius_cubed
            + 2.0 * omega * y_velocity
            + omega**2 * x_position
            + x_control,
            -y_position / radius_cubed
            - 2.0 * omega * x_velocity
            + omega**2 * y_position
            + y_control,
        ]
    )
    control_squared = x_control**2 + y_control**2
    scaled_altitude_above_tangent = x_position - 1.0
    glide_factor = np.tan(TRANSCRIPTION_GLIDE_SLOPE_ANGLE)
    phase.set_integral([control_squared])
    phase.set_phase_constraint(
        [
            radius_squared,
            scaled_altitude_above_tangent - glide_factor * y_position,
            scaled_altitude_above_tangent + glide_factor * y_position,
            control_squared,
        ],
        [1.0, 0.0, 0.0, 0.0],
        [np.inf, np.inf, np.inf, SCALED_MAX_ACCELERATION**2],
    )
    initial_state = [
        *(INITIAL_POSITION / BENNU_MEAN_RADIUS),
        *(INITIAL_VELOCITY / VELOCITY_SCALE),
    ]
    target_state = [
        *(TARGET_POSITION / BENNU_MEAN_RADIUS),
        *(TARGET_VELOCITY / VELOCITY_SCALE),
    ]
    phase.set_boundary_condition(initial_state, target_state, 0.0, SCALED_FLIGHT_TIME)
    phase.set_discretization(24 if quick else 48, 3)
    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct an endpoint-matching path and inverse-dynamics control guess."""
    guess = linear_guess(phase, 0.0)
    position_x, velocity_x, _ = _hermite_trajectory(guess.t_x)
    guess.x[0] = position_x[0]
    guess.x[1] = position_x[1]
    guess.x[2] = velocity_x[0]
    guess.x[3] = velocity_x[1]

    position_u, velocity_u, commanded_acceleration = _hermite_trajectory(guess.t_u)
    for index in range(guess.t_u.size):
        radius = np.linalg.norm(position_u[:, index])
        omega = SCALED_ROTATION_RATE
        coriolis = np.array(
            [
                2.0 * omega * velocity_u[1, index],
                -2.0 * omega * velocity_u[0, index],
            ]
        )
        control = (
            commanded_acceleration[:, index]
            + position_u[:, index] / radius**3
            - coriolis
            - omega**2 * position_u[:, index]
        )
        magnitude = np.linalg.norm(control)
        if magnitude > 0.98 * SCALED_MAX_ACCELERATION:
            control *= 0.98 * SCALED_MAX_ACCELERATION / magnitude
        guess.u[0][index] = control[0]
        guess.u[1][index] = control[1]
    return guess


def _dense_history(solution, count: int = DENSE_CHECK_POINTS):
    """Reconstruct scaled states and controls on a uniform time grid."""
    scaled_time = np.linspace(solution.t_0, solution.t_f, count)
    state_matrix = solution.V_x(scaled_time.copy())
    control_matrix = solution.V_u(scaled_time.copy())
    state = np.vstack([state_matrix @ component for component in solution.x])
    control = np.vstack([control_matrix @ component for component in solution.u])
    return scaled_time, state, control


def _physical_history(scaled_time: np.ndarray, state: np.ndarray, control: np.ndarray):
    """Convert a reconstructed solution from natural units to SI units."""
    return {
        "time": TIME_SCALE * scaled_time,
        "position": BENNU_MEAN_RADIUS * state[:2],
        "velocity": VELOCITY_SCALE * state[2:],
        "acceleration": ACCELERATION_SCALE * control,
    }


def _integrate_controls(solution, evaluation_time: np.ndarray) -> np.ndarray:
    """Independently integrate the optimized controls with DOP853."""
    control_times = np.asarray(solution.t_u, dtype=float)
    control_values = np.vstack(solution.u)
    if (control_times.size - 1) % 2:
        raise RuntimeError("expected a three-point Lobatto control grid")
    interval_boundaries = control_times[::2]

    def control_at(time: float) -> np.ndarray:
        interval = int(np.searchsorted(interval_boundaries, time, side="right") - 1)
        interval = int(np.clip(interval, 0, interval_boundaries.size - 2))
        start = 2 * interval
        nodes = control_times[start : start + 3]
        values = control_values[:, start : start + 3]
        weights = np.array(
            [
                (time - nodes[1])
                * (time - nodes[2])
                / ((nodes[0] - nodes[1]) * (nodes[0] - nodes[2])),
                (time - nodes[0])
                * (time - nodes[2])
                / ((nodes[1] - nodes[0]) * (nodes[1] - nodes[2])),
                (time - nodes[0])
                * (time - nodes[1])
                / ((nodes[2] - nodes[0]) * (nodes[2] - nodes[1])),
            ]
        )
        return values @ weights

    interval_width = np.diff(interval_boundaries)
    comparison_time = np.sort(
        np.concatenate(
            (
                interval_boundaries[:-1] + 0.25 * interval_width,
                interval_boundaries[:-1] + 0.75 * interval_width,
            )
        )
    )
    pockit_matrix = solution.V_u(comparison_time.copy())
    pockit_control = np.vstack([pockit_matrix @ component for component in solution.u])
    local_control = np.column_stack([control_at(time) for time in comparison_time])
    if np.max(np.abs(pockit_control - local_control)) > 2.0e-12:
        raise RuntimeError("local control reconstruction disagrees with pockit")

    def right_hand_side(time, state):
        return _scaled_dynamics(state, control_at(time))

    initial_state = np.array(
        [
            *(INITIAL_POSITION / BENNU_MEAN_RADIUS),
            *(INITIAL_VELOCITY / VELOCITY_SCALE),
        ]
    )
    integration = solve_ivp(
        right_hand_side,
        (float(evaluation_time[0]), float(evaluation_time[-1])),
        initial_state,
        method="DOP853",
        t_eval=evaluation_time,
        rtol=2.0e-11,
        atol=2.0e-13,
        max_step=SCALED_FLIGHT_TIME / 500.0,
    )
    if not integration.success:
        raise RuntimeError(f"independent integration failed: {integration.message}")
    return integration.y


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def solve_problem(system, guess, quick: bool = False):
    """Solve the landing and independently verify dynamics and constraints."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={
            "tol": 2.0e-8 if quick else 2.0e-10,
            "acceptable_tol": 2.0e-7 if quick else 2.0e-9,
            "max_iter": 2_000,
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

    scaled_time, state, control = _dense_history(solution)
    history = _physical_history(scaled_time, state, control)
    integrated_state = _integrate_controls(solution, scaled_time)
    require_finite(
        state=state,
        control=control,
        integrated_state=integrated_state,
        objective=info["obj_val"],
    )
    radius = np.linalg.norm(history["position"], axis=0)
    altitude = radius - BENNU_MEAN_RADIUS
    tangent_altitude = history["position"][0] - BENNU_MEAN_RADIUS
    glide_margin = tangent_altitude - np.tan(MIN_GLIDE_SLOPE_ANGLE) * np.abs(
        history["position"][1]
    )
    acceleration_magnitude = np.linalg.norm(history["acceleration"], axis=0)
    terminal_position_error = float(
        np.linalg.norm(history["position"][:, -1] - TARGET_POSITION)
    )
    terminal_velocity_error = float(
        np.linalg.norm(history["velocity"][:, -1] - TARGET_VELOCITY)
    )
    surface_violation = max(float(-np.min(altitude)), 0.0)
    glide_violation = max(float(-np.min(glide_margin)), 0.0)
    thrust_violation = max(
        float(np.max(acceleration_magnitude) - MAX_THRUST_ACCELERATION), 0.0
    )
    integration_mismatch = float(np.max(np.abs(integrated_state - state)))
    integrated_terminal_position_error = float(
        np.linalg.norm(BENNU_MEAN_RADIUS * integrated_state[:2, -1] - TARGET_POSITION)
    )
    integrated_terminal_velocity_error = float(
        np.linalg.norm(VELOCITY_SCALE * integrated_state[2:, -1] - TARGET_VELOCITY)
    )
    integrated_position = BENNU_MEAN_RADIUS * integrated_state[:2]
    integrated_altitude = (
        np.linalg.norm(integrated_position, axis=0) - BENNU_MEAN_RADIUS
    )
    integrated_glide_margin = (
        integrated_position[0]
        - BENNU_MEAN_RADIUS
        - np.tan(MIN_GLIDE_SLOPE_ANGLE) * np.abs(integrated_position[1])
    )
    integrated_surface_violation = max(float(-np.min(integrated_altitude)), 0.0)
    integrated_glide_violation = max(float(-np.min(integrated_glide_margin)), 0.0)
    delta_velocity = float(trapezoid(acceleration_magnitude, x=history["time"]))
    physical_effort = ACCELERATION_SCALE**2 * TIME_SCALE * float(info["obj_val"])

    endpoint_position_tolerance = 2.0e-5  # m
    endpoint_velocity_tolerance = 2.0e-8  # m/s
    mismatch_tolerance = 3.0e-4 if quick else 7.0e-5
    integrated_position_tolerance = 1.0e-2 if quick else 1.0e-3  # m
    integrated_velocity_tolerance = 2.0e-6 if quick else 2.0e-7  # m/s
    if terminal_position_error > endpoint_position_tolerance or (
        terminal_velocity_error > endpoint_velocity_tolerance
    ):
        raise RuntimeError(
            "collocation endpoint error is too large: "
            f"position={terminal_position_error:.3e} m, "
            f"velocity={terminal_velocity_error:.3e} m/s"
        )
    glide_tolerance = 1.0e-3 if quick else 2.0e-4
    if (
        surface_violation > 2.0e-4
        or glide_violation > glide_tolerance
        or (thrust_violation > 2.0e-9)
    ):
        raise RuntimeError(
            "dense path validation failed: "
            f"surface={surface_violation:.3e} m, "
            f"glide-slope={glide_violation:.3e} m, "
            f"acceleration={thrust_violation:.3e} m/s^2"
        )
    if integration_mismatch > mismatch_tolerance:
        raise RuntimeError(
            f"independent-integration mismatch {integration_mismatch:.3e} exceeds "
            f"{mismatch_tolerance:.1e} scaled units"
        )
    integrated_path_tolerance = 5.0e-3 if quick else 5.0e-4
    if max(integrated_surface_violation, integrated_glide_violation) > (
        integrated_path_tolerance
    ):
        raise RuntimeError(
            "independently integrated path violates the guarded approach: "
            f"surface={integrated_surface_violation:.3e} m, "
            f"glide-slope={integrated_glide_violation:.3e} m"
        )
    if integrated_terminal_position_error > integrated_position_tolerance or (
        integrated_terminal_velocity_error > integrated_velocity_tolerance
    ):
        raise RuntimeError(
            "independently integrated endpoint error is too large: "
            f"position={integrated_terminal_position_error:.3e} m, "
            f"velocity={integrated_terminal_velocity_error:.3e} m/s"
        )

    print(f"Ipopt status: {message}")
    print(f"Flight time: {history['time'][-1] / 3600.0:.3f} h")
    print(f"Squared-acceleration effort: {physical_effort:.9e} m^2/s^3")
    print(f"Integrated acceleration (delta-v proxy): {delta_velocity:.6f} m/s")
    print(
        "Peak commanded acceleration: "
        f"{1.0e3 * np.max(acceleration_magnitude):.6f} mm/s^2"
    )
    print(f"Minimum altitude: {np.min(altitude):.6f} m")
    print(f"Minimum glide-slope margin: {np.min(glide_margin):.6f} m")
    print(
        "Independent terminal error: "
        f"{integrated_terminal_position_error:.6f} m, "
        f"{integrated_terminal_velocity_error:.3e} m/s"
    )
    print(
        "Independent minimum margins: "
        f"altitude={np.min(integrated_altitude):.6f} m, "
        f"glide-slope={np.min(integrated_glide_margin):.6f} m"
    )
    print(f"Independent-integration mismatch: {integration_mismatch:.3e} scaled")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot the body-fixed landing path, altitude, velocity, and command."""
    configure_matplotlib()
    scaled_time, state, control = _dense_history(solution, 2_001)
    history = _physical_history(scaled_time, state, control)
    time_hours = history["time"] / 3600.0
    position = history["position"]
    radius = np.linalg.norm(position, axis=0)
    altitude = radius - BENNU_MEAN_RADIUS
    acceleration = 1.0e3 * history["acceleration"]
    acceleration_magnitude = np.linalg.norm(acceleration, axis=0)

    fig = plt.figure(figsize=(10.6, 7.4), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=(1.25, 1.0))
    path_axis = fig.add_subplot(grid[:, 0])
    altitude_axis = fig.add_subplot(grid[0, 1])
    velocity_axis = fig.add_subplot(grid[1, 1], sharex=altitude_axis)
    command_axis = fig.add_subplot(grid[2, 1], sharex=altitude_axis)

    asteroid = plt.Circle(
        (0.0, 0.0),
        BENNU_MEAN_RADIUS,
        facecolor="#D9D9D9",
        edgecolor=COLORS["black"],
        linewidth=1.0,
        label="Spherical Bennu model",
    )
    path_axis.add_patch(asteroid)
    corridor_cross_range = np.linspace(-0.62, 0.62, 151) * BENNU_MEAN_RADIUS
    corridor_x = BENNU_MEAN_RADIUS + np.tan(MIN_GLIDE_SLOPE_ANGLE) * np.abs(
        corridor_cross_range
    )
    path_axis.plot(
        corridor_x,
        corridor_cross_range,
        color=COLORS["purple"],
        linestyle="--",
        linewidth=1.2,
        label=r"$20^\circ$ glide corridor",
    )
    path_axis.plot(
        position[0],
        position[1],
        color=COLORS["blue"],
        linewidth=2.4,
        label="Optimized approach",
    )
    path_axis.scatter(
        *position[:, 0], color=COLORS["vermillion"], s=42, zorder=3, label="Start"
    )
    path_axis.scatter(
        *TARGET_POSITION,
        color=COLORS["green"],
        marker="^",
        s=52,
        zorder=3,
        label="Landing site",
    )
    path_axis.axhline(0.0, color="#A0A0A0", linewidth=0.7, alpha=0.5)
    path_axis.axvline(0.0, color="#A0A0A0", linewidth=0.7, alpha=0.5)
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_xlabel("Body-fixed $x$ [m]")
    path_axis.set_ylabel("Body-fixed $y$ [m]")
    path_axis.set_title("Bennu soft-landing trajectory")
    path_axis.legend(loc="lower left")

    altitude_axis.plot(time_hours, altitude, color=COLORS["blue"])
    altitude_axis.axhline(0.0, color=COLORS["black"], linewidth=1.0)
    altitude_axis.set_ylabel("Altitude [m]")
    altitude_axis.set_title("Surface clearance")

    velocity_axis.plot(
        time_hours,
        history["velocity"][0],
        color=COLORS["blue"],
        label=r"$v_x$",
    )
    velocity_axis.plot(
        time_hours,
        history["velocity"][1],
        color=COLORS["orange"],
        label=r"$v_y$",
    )
    velocity_axis.set_ylabel("Velocity [m/s]")
    velocity_axis.legend(ncol=2)

    command_axis.plot(time_hours, acceleration[0], color=COLORS["blue"], label=r"$a_x$")
    command_axis.plot(
        time_hours, acceleration[1], color=COLORS["orange"], label=r"$a_y$"
    )
    command_axis.plot(
        time_hours,
        acceleration_magnitude,
        color=COLORS["purple"],
        linestyle="--",
        label=r"$\|\mathbf{a}_T\|$",
    )
    command_axis.axhline(
        1.0e3 * MAX_THRUST_ACCELERATION,
        color=COLORS["black"],
        linewidth=1.0,
        linestyle=":",
        label="Limit",
    )
    command_axis.set_ylabel(r"Command [mm/s$^2$]")
    command_axis.set_xlabel("Time [h]")
    command_axis.legend(ncol=2)

    style_axes([path_axis, altitude_axis, velocity_axis, command_axis])
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "asteroid-soft-landing.png", quick=True
    )
    system, phase = build_problem(args.quick)
    guess = initial_guess(phase)
    solution = solve_problem(system, guess, args.quick)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
