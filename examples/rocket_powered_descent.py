"""Plan a fuel-efficient six-degree-of-freedom powered descent on Mars.

The lander must remove horizontal error, reduce its descent rate, and finish
upright while respecting thrust, gimbal, tilt, angular-rate, glide-slope, and
dry-mass limits. Position, velocity, mass, and thrust are scaled for numerical
conditioning; all declared vehicle data and reported results use SI units.
Attitude uses a bounded local modified-Rodrigues-parameter chart rather than a
redundant quaternion state.
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


# Numerical scales. Time and angular rate remain in seconds and radians/second.
LENGTH_SCALE = 100.0  # m
VELOCITY_SCALE = 10.0  # m/s
MASS_SCALE = 2_000.0  # kg
FORCE_SCALE = 2_000.0  # N

MARS_GRAVITY = 3.711  # m/s^2
EXHAUST_VELOCITY = 2_250.0  # m/s
WET_MASS = 2_000.0  # kg
DRY_MASS = 1_400.0  # kg
MIN_THRUST = 4_000.0  # N
MAX_THRUST = 30_000.0  # N
MIN_THRUST_INTERPOLATION_GUARD = 10.0  # N
MAX_GIMBAL_ANGLE = np.deg2rad(15.0)
MAX_TILT_ANGLE = np.deg2rad(25.0)
MAX_GLIDE_CONE_ANGLE = np.deg2rad(30.0)
MAX_ANGULAR_RATE = 0.70  # rad/s
MAX_MRP_NORM = np.tan(np.deg2rad(120.0) / 4.0)
INERTIA = np.diag([4_000.0, 2_500.0, 2_500.0])  # kg m^2
ENGINE_POSITION_BODY = np.array([-1.5, 0.0, 0.0])  # m from center of mass
HORIZON = 16.0  # s
DENSE_CHECK_POINTS = 4_001

# Coordinates are [up, east, north]. The body x axis points through the engine.
INITIAL_POSITION = np.array([100.0, 30.0, -15.0])  # m
INITIAL_VELOCITY = np.array([-12.0, -4.0, 2.0])  # m/s
INITIAL_MRP = np.array([0.0, 0.04, -0.03])
INITIAL_ANGULAR_RATE = np.zeros(3)  # rad/s
TARGET_POSITION = np.zeros(3)  # m
TARGET_VELOCITY = np.array([-0.5, 0.0, 0.0])  # m/s at contact
TARGET_MRP = np.zeros(3)
TARGET_ANGULAR_RATE = np.zeros(3)  # rad/s


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
    """Build the fixed-time, minimum-propellant powered-descent problem."""
    system = System(0, fastmath=True)
    phase = system.new_phase(
        [
            "scaled_mass",
            "scaled_position_up",
            "scaled_position_east",
            "scaled_position_north",
            "scaled_velocity_up",
            "scaled_velocity_east",
            "scaled_velocity_north",
            "mrp_1",
            "mrp_2",
            "mrp_3",
            "angular_rate_x",
            "angular_rate_y",
            "angular_rate_z",
        ],
        [
            "scaled_thrust_body_x",
            "scaled_thrust_body_y",
            "scaled_thrust_body_z",
        ],
    )

    scaled_mass = phase.x[0]
    scaled_position = sp.Matrix(phase.x[1:4])
    scaled_velocity = sp.Matrix(phase.x[4:7])
    mrp = sp.Matrix(phase.x[7:10])
    angular_rate = sp.Matrix(phase.x[10:13])
    scaled_thrust_body = sp.Matrix(phase.u)

    rotation = _mrp_rotation_symbolic(mrp)
    physical_mass = MASS_SCALE * scaled_mass
    physical_thrust_body = FORCE_SCALE * scaled_thrust_body
    physical_thrust_inertial = rotation * physical_thrust_body
    thrust_magnitude_scaled = sp.sqrt(scaled_thrust_body.dot(scaled_thrust_body))
    thrust_magnitude_physical = FORCE_SCALE * thrust_magnitude_scaled

    scaled_mass_rate = -thrust_magnitude_physical / (EXHAUST_VELOCITY * MASS_SCALE)
    scaled_position_rate = VELOCITY_SCALE / LENGTH_SCALE * scaled_velocity
    scaled_velocity_rate = physical_thrust_inertial / (
        physical_mass * VELOCITY_SCALE
    ) + sp.Matrix([-MARS_GRAVITY / VELOCITY_SCALE, 0.0, 0.0])
    mrp_rate = _mrp_kinematics(mrp, angular_rate)
    inertia = sp.Matrix(INERTIA)
    engine_position = sp.Matrix(ENGINE_POSITION_BODY)
    physical_moment_body = engine_position.cross(physical_thrust_body)
    angular_acceleration = inertia.inv() * (
        physical_moment_body - angular_rate.cross(inertia * angular_rate)
    )
    phase.set_dynamics(
        [
            scaled_mass_rate,
            *scaled_position_rate,
            *scaled_velocity_rate,
            *mrp_rate,
            *angular_acceleration,
        ]
    )

    glide_margin_squared = (
        np.tan(MAX_GLIDE_CONE_ANGLE) ** 2 * scaled_position[0] ** 2
        - scaled_position[1] ** 2
        - scaled_position[2] ** 2
    )
    gimbal_margin = (
        scaled_thrust_body[0] - np.cos(MAX_GIMBAL_ANGLE) * thrust_magnitude_scaled
    )
    tilt_cosine = rotation[0, 0]
    angular_rate_squared = angular_rate.dot(angular_rate)
    mrp_norm_squared = mrp.dot(mrp)
    phase.set_phase_constraint(
        [
            scaled_mass,
            thrust_magnitude_scaled,
            gimbal_margin,
            glide_margin_squared,
            tilt_cosine,
            angular_rate_squared,
            scaled_position[0],
            mrp_norm_squared,
        ],
        [
            DRY_MASS / MASS_SCALE,
            (MIN_THRUST + MIN_THRUST_INTERPOLATION_GUARD) / FORCE_SCALE,
            0.0,
            0.0,
            np.cos(MAX_TILT_ANGLE),
            0.0,
            0.0,
            0.0,
        ],
        [
            WET_MASS / MASS_SCALE,
            MAX_THRUST / FORCE_SCALE,
            np.inf,
            np.inf,
            1.0,
            MAX_ANGULAR_RATE**2,
            np.inf,
            MAX_MRP_NORM**2,
        ],
    )
    phase.set_integral([thrust_magnitude_scaled])

    initial_state = [
        WET_MASS / MASS_SCALE,
        *(INITIAL_POSITION / LENGTH_SCALE),
        *(INITIAL_VELOCITY / VELOCITY_SCALE),
        *INITIAL_MRP,
        *INITIAL_ANGULAR_RATE,
    ]
    target_state = [
        None,
        *(TARGET_POSITION / LENGTH_SCALE),
        *(TARGET_VELOCITY / VELOCITY_SCALE),
        *TARGET_MRP,
        *TARGET_ANGULAR_RATE,
    ]
    phase.set_boundary_condition(initial_state, target_state, 0.0, HORIZON)
    phase.set_discretization(24 if quick else 40, 2)
    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def _hermite_translation(time: np.ndarray):
    """Return a cubic endpoint-matching position, velocity, and acceleration."""
    time = np.asarray(time, dtype=float)
    fraction = time / HORIZON
    position_initial = INITIAL_POSITION / LENGTH_SCALE
    position_final = TARGET_POSITION / LENGTH_SCALE
    position_rate_initial = INITIAL_VELOCITY / LENGTH_SCALE
    position_rate_final = TARGET_VELOCITY / LENGTH_SCALE

    h00 = 2.0 * fraction**3 - 3.0 * fraction**2 + 1.0
    h10 = fraction**3 - 2.0 * fraction**2 + fraction
    h01 = -2.0 * fraction**3 + 3.0 * fraction**2
    h11 = fraction**3 - fraction**2
    position = (
        np.outer(position_initial, h00)
        + np.outer(HORIZON * position_rate_initial, h10)
        + np.outer(position_final, h01)
        + np.outer(HORIZON * position_rate_final, h11)
    )

    dh00 = 6.0 * fraction**2 - 6.0 * fraction
    dh10 = 3.0 * fraction**2 - 4.0 * fraction + 1.0
    dh01 = -6.0 * fraction**2 + 6.0 * fraction
    dh11 = 3.0 * fraction**2 - 2.0 * fraction
    position_rate = (
        np.outer(position_initial, dh00)
        + np.outer(HORIZON * position_rate_initial, dh10)
        + np.outer(position_final, dh01)
        + np.outer(HORIZON * position_rate_final, dh11)
    ) / HORIZON
    velocity = LENGTH_SCALE / VELOCITY_SCALE * position_rate

    ddh00 = 12.0 * fraction - 6.0
    ddh10 = 6.0 * fraction - 4.0
    ddh01 = -12.0 * fraction + 6.0
    ddh11 = 6.0 * fraction - 2.0
    position_acceleration = (
        np.outer(position_initial, ddh00)
        + np.outer(HORIZON * position_rate_initial, ddh10)
        + np.outer(position_final, ddh01)
        + np.outer(HORIZON * position_rate_final, ddh11)
    ) / HORIZON**2
    physical_acceleration = LENGTH_SCALE * position_acceleration
    return position, velocity, physical_acceleration


def _smooth_progress(time: np.ndarray) -> np.ndarray:
    fraction = np.asarray(time, dtype=float) / HORIZON
    return 10.0 * fraction**3 - 15.0 * fraction**4 + 6.0 * fraction**5


def initial_guess(phase):
    """Construct an endpoint-matching descent with force-balanced thrust."""
    guess = linear_guess(phase, 0.0)
    position, velocity, _ = _hermite_translation(guess.t_x)
    progress_x = _smooth_progress(guess.t_x)
    estimated_final_mass = WET_MASS - 80.0

    guess.x[0] = (
        WET_MASS + (estimated_final_mass - WET_MASS) * guess.t_x / HORIZON
    ) / MASS_SCALE
    for index in range(3):
        guess.x[1 + index] = position[index]
        guess.x[4 + index] = velocity[index]
        guess.x[7 + index] = INITIAL_MRP[index] * (1.0 - progress_x)
        guess.x[10 + index] = 0.0

    _, _, physical_acceleration = _hermite_translation(guess.t_u)
    progress_u = _smooth_progress(guess.t_u)
    mass_u = WET_MASS + (estimated_final_mass - WET_MASS) * guess.t_u / HORIZON
    for column in range(guess.t_u.size):
        thrust_inertial = mass_u[column] * (
            physical_acceleration[:, column] + np.array([MARS_GRAVITY, 0.0, 0.0])
        )
        mrp = INITIAL_MRP * (1.0 - progress_u[column])
        thrust_body = _mrp_rotation_numeric(mrp).T @ thrust_inertial
        magnitude = np.linalg.norm(thrust_body)
        guarded_minimum_thrust = MIN_THRUST + MIN_THRUST_INTERPOLATION_GUARD
        if magnitude < guarded_minimum_thrust:
            thrust_body *= guarded_minimum_thrust / magnitude
        elif magnitude > MAX_THRUST:
            thrust_body *= MAX_THRUST / magnitude
        scaled_thrust = thrust_body / FORCE_SCALE
        for component in range(3):
            guess.u[component][column] = scaled_thrust[component]
    return guess


def _dense_history(solution, count: int = DENSE_CHECK_POINTS):
    """Interpolate states and controls onto a uniform physical-time grid."""
    time = np.linspace(solution.t_0, solution.t_f, count)
    state_interpolation = solution.V_x(time.copy())
    control_interpolation = solution.V_u(time.copy())
    state = np.vstack([state_interpolation @ component for component in solution.x])
    control = np.vstack([control_interpolation @ component for component in solution.u])
    return time, state, control


def _physical_history(state: np.ndarray, control: np.ndarray):
    """Convert scaled optimization histories to SI quantities."""
    return {
        "mass": MASS_SCALE * state[0],
        "position": LENGTH_SCALE * state[1:4],
        "velocity": VELOCITY_SCALE * state[4:7],
        "mrp": state[7:10],
        "angular_rate": state[10:13],
        "thrust_body": FORCE_SCALE * control,
    }


def _attitude_diagnostics(mrp: np.ndarray):
    """Return principal angle and body-x tilt for each MRP column."""
    principal_angle = 4.0 * np.arctan(np.linalg.norm(mrp, axis=0))
    tilt = np.empty(mrp.shape[1])
    for index in range(mrp.shape[1]):
        rotation = _mrp_rotation_numeric(mrp[:, index])
        tilt[index] = np.arccos(np.clip(rotation[0, 0], -1.0, 1.0))
    return principal_angle, tilt


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def solve_problem(system, guess, quick: bool = False):
    """Solve the descent and verify endpoints and dense physical constraints."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={
            "tol": 3.0e-7 if quick else 2.0e-9,
            "acceptable_tol": 2.0e-6 if quick else 1.0e-8,
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

    time, scaled_state, scaled_control = _dense_history(solution)
    history = _physical_history(scaled_state, scaled_control)
    thrust_magnitude = np.linalg.norm(history["thrust_body"], axis=0)
    gimbal_angle = np.arctan2(
        np.linalg.norm(history["thrust_body"][1:], axis=0),
        history["thrust_body"][0],
    )
    principal_angle, tilt_angle = _attitude_diagnostics(history["mrp"])
    require_finite(
        time=time,
        scaled_state=scaled_state,
        scaled_control=scaled_control,
        mass=history["mass"],
        position=history["position"],
        velocity=history["velocity"],
        mrp=history["mrp"],
        angular_rate=history["angular_rate"],
        thrust_body=history["thrust_body"],
        thrust_magnitude=thrust_magnitude,
        gimbal_angle=gimbal_angle,
        principal_angle=principal_angle,
        tilt_angle=tilt_angle,
        objective=info["obj_val"],
    )
    horizontal_distance = np.linalg.norm(history["position"][1:], axis=0)
    glide_margin = (
        np.tan(MAX_GLIDE_CONE_ANGLE) * history["position"][0] - horizontal_distance
    )
    angular_speed = np.linalg.norm(history["angular_rate"], axis=0)
    chart_margin = MAX_MRP_NORM - np.linalg.norm(history["mrp"], axis=0)

    final_position_error = float(
        np.max(np.abs(history["position"][:, -1] - TARGET_POSITION))
    )
    final_velocity_error = float(
        np.max(np.abs(history["velocity"][:, -1] - TARGET_VELOCITY))
    )
    final_attitude_error = float(principal_angle[-1])
    final_rate_error = float(angular_speed[-1])
    dry_mass_violation = max(DRY_MASS - float(np.min(history["mass"])), 0.0)
    thrust_violation = max(
        MIN_THRUST - float(np.min(thrust_magnitude)),
        float(np.max(thrust_magnitude)) - MAX_THRUST,
        0.0,
    )
    path_violation = max(
        float(np.max(-history["position"][0])),
        float(np.max(-glide_margin)),
        float(np.max(gimbal_angle - MAX_GIMBAL_ANGLE)),
        float(np.max(tilt_angle - MAX_TILT_ANGLE)),
        float(np.max(angular_speed - MAX_ANGULAR_RATE)),
        float(np.max(-chart_margin)),
        0.0,
    )
    propellant_used = WET_MASS - float(history["mass"][-1])
    propellant_from_objective = FORCE_SCALE * float(info["obj_val"]) / EXHAUST_VELOCITY

    endpoint_tolerance = 4.0e-4 if quick else 5.0e-5
    if max(final_position_error, final_velocity_error) > endpoint_tolerance:
        raise RuntimeError(
            "landing endpoint error is too large: "
            f"position={final_position_error:.3e} m, "
            f"velocity={final_velocity_error:.3e} m/s"
        )
    if max(final_attitude_error, final_rate_error) > 4.0e-6:
        raise RuntimeError(
            "terminal attitude state is inconsistent: "
            f"angle={final_attitude_error:.3e} rad, rate={final_rate_error:.3e} rad/s"
        )
    if dry_mass_violation > 2.0e-3:
        raise RuntimeError(f"dry-mass violation: {dry_mass_violation:.3e} kg")
    if thrust_violation > 5.0e-3:
        raise RuntimeError(f"dense thrust-bound violation: {thrust_violation:.3e} N")
    if path_violation > (2.0e-3 if quick else 5.0e-4):
        raise RuntimeError(f"dense path-constraint violation: {path_violation:.3e}")
    if abs(propellant_used - propellant_from_objective) > 3.0e-3:
        raise RuntimeError("objective and integrated mass depletion are inconsistent")

    print(f"Ipopt status: {message}")
    print(f"Flight time: {time[-1]:.3f} s")
    print(f"Propellant used: {propellant_used:.3f} kg")
    print(f"Contact velocity: {history['velocity'][0, -1]:.6f} m/s")
    print(
        "Dense thrust range: "
        f"[{np.min(thrust_magnitude):.2f}, {np.max(thrust_magnitude):.2f}] N"
    )
    print(f"Peak gimbal angle: {np.rad2deg(np.max(gimbal_angle)):.4f} deg")
    print(f"Peak tilt angle: {np.rad2deg(np.max(tilt_angle)):.4f} deg")
    print(f"Minimum glide-cone margin: {np.min(glide_margin):.6f} m")
    return solution


def plot_solution(solution, *, save=None, show=True):
    """Plot the landing path, kinematics, attitude, thrust, and mass."""
    configure_matplotlib()
    time, scaled_state, scaled_control = _dense_history(solution, 2_001)
    history = _physical_history(scaled_state, scaled_control)
    thrust_magnitude = np.linalg.norm(history["thrust_body"], axis=0)
    gimbal_angle = np.arctan2(
        np.linalg.norm(history["thrust_body"][1:], axis=0),
        history["thrust_body"][0],
    )
    _, tilt_angle = _attitude_diagnostics(history["mrp"])

    fig = plt.figure(figsize=(10.6, 8.0), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=(1.25, 1.0))
    path_axis = fig.add_subplot(grid[:, 0], projection="3d")
    velocity_axis = fig.add_subplot(grid[0, 1])
    attitude_axis = fig.add_subplot(grid[1, 1], sharex=velocity_axis)
    propulsion_axis = fig.add_subplot(grid[2, 1], sharex=velocity_axis)

    position = history["position"]
    path_axis.plot(
        position[1],
        position[2],
        position[0],
        color=COLORS["blue"],
        label="Optimal descent",
    )
    path_axis.scatter(
        position[1, 0],
        position[2, 0],
        position[0, 0],
        color=COLORS["vermillion"],
        marker="o",
        s=36,
        label="Ignition",
    )
    path_axis.scatter(
        0.0,
        0.0,
        0.0,
        color=COLORS["green"],
        marker="^",
        s=42,
        label="Touchdown",
    )
    path_axis.set_xlabel("East position [m]")
    path_axis.set_ylabel("North position [m]")
    path_axis.set_zlabel("Altitude [m]", labelpad=8)
    path_axis.view_init(elev=24, azim=45)
    path_axis.set_title("Mars powered-descent trajectory")
    path_axis.legend(loc="upper left")

    velocity_labels = ("Up", "East", "North")
    velocity_colors = (COLORS["blue"], COLORS["orange"], COLORS["green"])
    for component, label, color in zip(
        history["velocity"], velocity_labels, velocity_colors
    ):
        velocity_axis.plot(time, component, color=color, label=label)
    velocity_axis.set_ylabel("Velocity [m/s]")
    velocity_axis.set_title("Landing kinematics")
    velocity_axis.legend(ncol=3)

    attitude_axis.plot(
        time,
        np.rad2deg(tilt_angle),
        color=COLORS["purple"],
        label="Vehicle tilt",
    )
    attitude_axis.plot(
        time,
        np.rad2deg(gimbal_angle),
        color=COLORS["orange"],
        label="Engine gimbal",
    )
    attitude_axis.axhline(
        np.rad2deg(MAX_TILT_ANGLE),
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.0,
        label="Tilt limit",
    )
    attitude_axis.set_ylabel("Angle [deg]")
    attitude_axis.set_title("Attitude and thrust direction")
    attitude_axis.legend(ncol=3, fontsize=8)

    propulsion_axis.plot(time, thrust_magnitude, color=COLORS["blue"], label="Thrust")
    propulsion_axis.set_xlabel("Time [s]")
    propulsion_axis.set_ylabel("Thrust [N]")
    propulsion_axis.set_title("Propulsion and remaining mass")
    mass_axis = propulsion_axis.twinx()
    mass_axis.spines["right"].set_visible(True)
    mass_axis.plot(time, history["mass"], color=COLORS["green"], label="Mass")
    mass_axis.set_ylabel("Mass [kg]")
    thrust_handles, thrust_labels = propulsion_axis.get_legend_handles_labels()
    mass_handles, mass_labels = mass_axis.get_legend_handles_labels()
    propulsion_axis.legend(thrust_handles + mass_handles, thrust_labels + mass_labels)

    style_axes([path_axis, velocity_axis, attitude_axis, propulsion_axis, mass_axis])
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "rocket_powered_descent_solution.png", quick=True
    )
    initial_rotation = _mrp_rotation_numeric(INITIAL_MRP)
    np.testing.assert_allclose(
        initial_rotation.T @ initial_rotation, np.eye(3), rtol=0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(np.linalg.det(initial_rotation), 1.0, atol=2.0e-14)

    system, phase = build_problem(quick=args.quick)
    solution = solve_problem(system, initial_guess(phase), quick=args.quick)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
