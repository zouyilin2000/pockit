"""Minimize propellant for a low-thrust Earth-to-Venus transfer.

The spacecraft follows heliocentric modified equinoctial-element dynamics in
AU, Julian years, and kilograms. A nonnegative throttle ``rho`` bounds the
radial, transverse, and normal thrust commands with a smooth quadratic cone.
The final mesh uses piecewise-constant controls, so this convex cone remains
feasible between all constrained nodes. Using ``rho`` directly in mass flow and
the objective avoids the nondifferentiable Euclidean norm at zero thrust.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from pockit.optimizer import ipopt
from pockit.radau import System, linear_guess

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


ASTRONOMICAL_UNIT_M = 149_597_870_700.0
JULIAN_YEAR_S = 365.25 * 86_400.0
DAY_IN_YEARS = 1.0 / 365.25
MU_SUN = 1.327_124_400_18e20 * JULIAN_YEAR_S**2 / ASTRONOMICAL_UNIT_M**3
STANDARD_GRAVITY = 9.80665 * JULIAN_YEAR_S**2 / ASTRONOMICAL_UNIT_M
MAX_THRUST = 0.33 * JULIAN_YEAR_S**2 / ASTRONOMICAL_UNIT_M
SPECIFIC_IMPULSE = 3_800.0 / JULIAN_YEAR_S
INITIAL_MASS = 1_500.0
MINIMUM_MASS = 500.0
TRANSFER_TIME = 1_000.0 * DAY_IN_YEARS
TARGET_REVOLUTIONS = 3
FULL_THRUST_MASS_FLOW = MAX_THRUST / (SPECIFIC_IMPULSE * STANDARD_GRAVITY)
DENSE_CHECK_POINTS = 10_001
SEMILATUS_RECTUM_BOUNDS = (0.20, 2.00)
MINIMUM_RADIUS_DENOMINATOR = 0.10
THRUST_DIRECTION_LIMIT = 0.9999

# Heliocentric ecliptic J2000 states at departure and arrival. The positive
# departure x-coordinate is consistent with Earth's heliocentric longitude on
# 2005-10-07; the opposite sign sometimes quoted for this benchmark is a typo.
EARTH_POSITION = np.array([9.708322e-1, 2.375844e-1, -1.671055e-6])
EARTH_VELOCITY = np.array([-1.598191, 6.081958, 9.443368e-5])
VENUS_POSITION = np.array([-3.277178e-1, 6.389172e-1, 2.765929e-2])
VENUS_VELOCITY = np.array([-6.598211, -3.412933, 3.340902e-1])


def _equinoctial_basis(h: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the two inertial basis vectors associated with MEE ``h`` and ``k``."""
    scale = 1.0 / (1.0 + h * h + k * k)
    basis_f = scale * np.array([1.0 + h * h - k * k, 2.0 * h * k, -2.0 * k])
    basis_g = scale * np.array([2.0 * h * k, 1.0 - h * h + k * k, 2.0 * h])
    return basis_f, basis_g


def cartesian_to_mee(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Convert an inertial Cartesian state to Walker modified equinoctial elements."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    radius = np.linalg.norm(position)
    angular_momentum = np.cross(position, velocity)
    momentum_norm = np.linalg.norm(angular_momentum)
    if radius <= 0.0 or momentum_norm <= 0.0:
        raise ValueError("position and angular momentum must be nonzero")

    denominator = momentum_norm + angular_momentum[2]
    if denominator <= 1.0e-14 * momentum_norm:
        raise ValueError("this MEE chart is singular for a retrograde equatorial orbit")
    h = -angular_momentum[1] / denominator
    k = angular_momentum[0] / denominator
    basis_f, basis_g = _equinoctial_basis(h, k)

    eccentricity_vector = (
        np.cross(velocity, angular_momentum) / MU_SUN - position / radius
    )
    f = float(np.dot(eccentricity_vector, basis_f))
    g = float(np.dot(eccentricity_vector, basis_g))
    longitude = float(
        np.mod(
            np.arctan2(np.dot(position, basis_g), np.dot(position, basis_f)),
            2.0 * np.pi,
        )
    )
    semilatus_rectum = momentum_norm**2 / MU_SUN
    return np.array([semilatus_rectum, f, g, h, k, longitude])


def mee_to_cartesian(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Walker modified equinoctial elements to an inertial state."""
    p, f, g, h, k, longitude = np.asarray(elements, dtype=float)
    if p <= 0.0:
        raise ValueError("semilatus rectum must be positive")
    basis_f, basis_g = _equinoctial_basis(h, k)
    cosine = np.cos(longitude)
    sine = np.sin(longitude)
    w = 1.0 + f * cosine + g * sine
    if w <= 0.0:
        raise ValueError("MEE radius denominator must be positive")
    position = p / w * (cosine * basis_f + sine * basis_g)
    velocity = np.sqrt(MU_SUN / p) * (-(sine + g) * basis_f + (cosine + f) * basis_g)
    return position, velocity


INITIAL_MEE = cartesian_to_mee(EARTH_POSITION, EARTH_VELOCITY)
TARGET_MEE = cartesian_to_mee(VENUS_POSITION, VENUS_VELOCITY)
TARGET_MEE[5] += TARGET_REVOLUTIONS * 2.0 * np.pi


def build_problem(quick: bool = False):
    """Build the fixed-time, minimum-propellant Earth-to-Venus transfer."""
    system = System(0, fastmath=True)
    phase = system.new_phase(
        ["p", "f", "g", "h", "k", "longitude", "mass"],
        ["direction_radial", "direction_transverse", "direction_normal", "rho"],
    )
    p, f, g, h, k, longitude, mass = phase.x
    direction_radial, direction_transverse, direction_normal, rho = phase.u
    u_radial = rho * direction_radial
    u_transverse = rho * direction_transverse
    u_normal = rho * direction_normal

    cosine = sp.cos(longitude)
    sine = sp.sin(longitude)
    w = 1.0 + f * cosine + g * sine
    s_squared = 1.0 + h**2 + k**2
    normal_coupling = h * sine - k * cosine
    scale = sp.sqrt(p / MU_SUN)
    acceleration_scale = MAX_THRUST / mass
    acceleration_r = acceleration_scale * u_radial
    acceleration_t = acceleration_scale * u_transverse
    acceleration_n = acceleration_scale * u_normal

    phase.set_dynamics(
        [
            2.0 * p * scale * acceleration_t / w,
            scale
            * (
                sine * acceleration_r
                + ((w + 1.0) * cosine + f) * acceleration_t / w
                - normal_coupling * g * acceleration_n / w
            ),
            scale
            * (
                -cosine * acceleration_r
                + ((w + 1.0) * sine + g) * acceleration_t / w
                + normal_coupling * f * acceleration_n / w
            ),
            scale * s_squared * cosine * acceleration_n / (2.0 * w),
            scale * s_squared * sine * acceleration_n / (2.0 * w),
            sp.sqrt(MU_SUN * p) * (w / p) ** 2
            + scale * normal_coupling * acceleration_n / w,
            -FULL_THRUST_MASS_FLOW * rho,
        ]
    )
    phase.set_integral([FULL_THRUST_MASS_FLOW * rho])

    direction_squared = (
        direction_radial**2 + direction_transverse**2 + direction_normal**2
    )
    phase.set_phase_constraint(
        [
            p,
            mass,
            w,
            rho,
            direction_radial,
            direction_transverse,
            direction_normal,
            direction_squared,
        ],
        [
            SEMILATUS_RECTUM_BOUNDS[0],
            MINIMUM_MASS,
            MINIMUM_RADIUS_DENOMINATOR,
            0.0,
            -THRUST_DIRECTION_LIMIT,
            -THRUST_DIRECTION_LIMIT,
            -THRUST_DIRECTION_LIMIT,
            0.0,
        ],
        [
            SEMILATUS_RECTUM_BOUNDS[1],
            INITIAL_MASS,
            np.inf,
            1.0,
            THRUST_DIRECTION_LIMIT,
            THRUST_DIRECTION_LIMIT,
            THRUST_DIRECTION_LIMIT,
            THRUST_DIRECTION_LIMIT**2,
        ],
        [False, False, False, True, False, False, False, False],
    )
    phase.set_boundary_condition(
        [*INITIAL_MEE, INITIAL_MASS],
        [*TARGET_MEE, None],
        0.0,
        TRANSFER_TIME,
    )
    # A compact high-order mesh supplies a reliable warm start. The returned
    # solution is adapted to a piecewise-constant-control mesh below.
    phase.set_discretization(12, 3)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct a feasible-scale inward-spiral guess for the optimizer."""
    guess = linear_guess(phase, 0.0)
    transfer_fraction_x = guess.t_x / TRANSFER_TIME
    guess.x[6] = INITIAL_MASS - 360.0 * transfer_fraction_x
    guess.u[0] = 0.0
    guess.u[1] = -0.84
    guess.u[2] = 0.03
    guess.u[3] = 0.50
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _solve_once(system, guess, *, fine: bool):
    """Run one Ipopt stage with tolerances appropriate to its mesh."""
    return ipopt.solve(
        system,
        guess,
        optimizer_options={
            "tol": 1.0e-8 if fine else 3.0e-7,
            "acceptable_tol": 1.0e-7 if fine else 3.0e-6,
            "constr_viol_tol": 1.0e-9 if fine else 1.0e-7,
            "acceptable_constr_viol_tol": 1.0e-8 if fine else 1.0e-6,
            "max_iter": 2_500 if fine else 1_600,
            "mu_strategy": "adaptive",
            "bound_relax_factor": 0.0,
            "print_level": 0,
            "sb": "yes",
        },
    )


def solve_problem(system, phase, guess, quick: bool = False):
    """Solve the transfer and report endpoint, fuel, and thrust-cone checks."""
    solution, info = _solve_once(system, guess, fine=False)
    if int(info["status"]) not in (0, 1):
        raise RuntimeError(
            f"Ipopt coarse stage failed ({info['status']}): {_status_message(info)}"
        )

    phase.set_discretization(120, 1)
    system.update()
    solution = solution.adapt(phase)
    solution, info = _solve_once(system, solution, fine=quick)

    if not quick:
        if int(info["status"]) not in (0, 1):
            raise RuntimeError(
                "Ipopt low-order stage failed "
                f"({info['status']}): {_status_message(info)}"
            )
        phase.set_discretization(320, 1)
        system.update()
        solution = solution.adapt(phase)
        solution, info = _solve_once(system, solution, fine=True)
    status = int(info["status"])
    message = _status_message(info)
    if status not in (0, 1):
        raise RuntimeError(f"Ipopt failed ({status}): {message}")

    dense_time, dense_states, dense_controls, dense_raw_controls = _dense_history(
        solution
    )
    require_finite(
        dense_time=dense_time,
        dense_states=dense_states,
        dense_controls=dense_controls,
        dense_raw_controls=dense_raw_controls,
        objective=info["obj_val"],
    )
    terminal_error = float(np.max(np.abs(dense_states[:6, -1] - TARGET_MEE)))
    direction_norm = np.linalg.norm(dense_controls[:3], axis=0)
    rho = dense_controls[3]
    cone_margin = rho - direction_norm
    quadratic_cone_margin = rho**2 - direction_norm**2
    minimum_cone_margin = float(np.min(cone_margin))
    minimum_quadratic_cone_margin = float(np.min(quadratic_cone_margin))
    dense_direction_norm = np.linalg.norm(dense_raw_controls[:3], axis=0)
    propellant = INITIAL_MASS - float(solution.x[6][-1])
    objective = float(info["obj_val"])

    feasibility_tolerance = 3.0e-6 if quick else 5.0e-7
    terminal_tolerance = 2.0e-5 if quick else 2.0e-7
    if terminal_error > terminal_tolerance:
        raise RuntimeError(f"terminal MEE error is too large: {terminal_error:.3e}")
    if float(np.min(rho)) < -feasibility_tolerance:
        raise RuntimeError(f"dense throttle is negative: {np.min(rho):.3e}")
    if float(np.max(rho)) > 1.0 + feasibility_tolerance:
        raise RuntimeError(f"dense throttle exceeds one: {np.max(rho):.3e}")
    if minimum_cone_margin < -feasibility_tolerance:
        raise RuntimeError(f"dense thrust cone is violated: {minimum_cone_margin:.3e}")
    if float(np.max(dense_direction_norm)) > (
        THRUST_DIRECTION_LIMIT + feasibility_tolerance
    ):
        raise RuntimeError(
            "dense direction norm exceeds its protected unit ball: "
            f"{np.max(dense_direction_norm):.9f}"
        )

    p_dense = dense_states[0]
    f_dense = dense_states[1]
    g_dense = dense_states[2]
    longitude_dense = dense_states[5]
    mass_dense = dense_states[6]
    w_dense = (
        1.0 + f_dense * np.cos(longitude_dense) + g_dense * np.sin(longitude_dense)
    )
    state_margins = {
        "semilatus rectum lower": float(np.min(p_dense - SEMILATUS_RECTUM_BOUNDS[0])),
        "semilatus rectum upper": float(np.min(SEMILATUS_RECTUM_BOUNDS[1] - p_dense)),
        "mass lower": float(np.min(mass_dense - MINIMUM_MASS)),
        "mass upper": float(np.min(INITIAL_MASS - mass_dense)),
        "radius denominator": float(np.min(w_dense - MINIMUM_RADIUS_DENOMINATOR)),
    }
    for name, margin in state_margins.items():
        if margin < -feasibility_tolerance:
            raise RuntimeError(f"dense {name} bound is violated: {margin:.3e}")

    mass_increase = float(np.max(np.diff(mass_dense)))
    mass_rate = solution.D_x(dense_time) @ solution.x[6]
    maximum_mass_rate = float(np.max(mass_rate))
    if mass_increase > feasibility_tolerance:
        raise RuntimeError(
            f"dense mass is not monotone: maximum increase {mass_increase:.3e} kg"
        )
    if maximum_mass_rate > feasibility_tolerance / DAY_IN_YEARS:
        raise RuntimeError(
            "dense mass derivative is positive: "
            f"{maximum_mass_rate * DAY_IN_YEARS:.3e} kg/day"
        )
    if abs(propellant - objective) > (2.0e-3 if quick else 2.0e-5):
        raise RuntimeError("fuel integral and terminal mass loss are inconsistent")

    print(f"Ipopt status: {message}")
    print(f"Propellant used: {propellant:.6f} kg")
    print(f"Final mass: {solution.x[6][-1]:.6f} kg")
    print(f"Maximum terminal MEE error: {terminal_error:.3e}")
    print(f"Dense validation points: {dense_time.size}")
    print(f"Dense throttle range: [{np.min(rho):.9f}, {np.max(rho):.9f}]")
    print(f"Minimum dense cone margin (rho - ||u||): {minimum_cone_margin:.3e}")
    print(f"Minimum dense quadratic-cone margin: {minimum_quadratic_cone_margin:.3e}")
    print(f"Maximum dense direction norm: {np.max(dense_direction_norm):.9f}")
    print(
        "Maximum dense mass increase: "
        f"{mass_increase:.3e} kg per {1_000.0 / (dense_time.size - 1):.1f}-day step"
    )
    print(
        f"Maximum dense mass derivative: {maximum_mass_rate * DAY_IN_YEARS:.3e} kg/day"
    )
    print(
        "Dense semilatus-rectum range: "
        f"[{np.min(p_dense):.9f}, {np.max(p_dense):.9f}] AU"
    )
    print(f"Dense mass range: [{np.min(mass_dense):.6f}, {np.max(mass_dense):.6f}] kg")
    print(f"Minimum dense radius denominator w: {np.min(w_dense):.9f}")
    print(f"Minimum dense state-bound margin: {min(state_margins.values()):.3e}")
    return solution


def _dense_history(
    solution, count: int = DENSE_CHECK_POINTS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate every state and control on a common dense time grid."""
    time = np.linspace(solution.t_0, solution.t_f, count)
    value_state = solution.V_x(time)
    value_control = solution.V_u(time)
    states = np.vstack([value_state @ solution.x[index] for index in range(7)])
    raw_controls = np.vstack([value_control @ solution.u[index] for index in range(4)])
    rho = raw_controls[3]
    controls = np.vstack([rho * raw_controls[index] for index in range(3)] + [rho])
    return time, states, controls, raw_controls


def _cartesian_history(solution, count: int = 1_201) -> np.ndarray:
    _, states, _, _ = _dense_history(solution, count)
    return np.array(
        [mee_to_cartesian(states[:6, index])[0] for index in range(states.shape[1])]
    )


def plot_solution(solution, *, save=None, show=True):
    """Plot the heliocentric path, thrust commands, mass, and orbital elements."""
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), layout="constrained")
    trajectory = _cartesian_history(solution)

    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    earth_radius = np.linalg.norm(EARTH_POSITION)
    venus_radius = np.linalg.norm(VENUS_POSITION)
    axes[0, 0].plot(
        earth_radius * np.cos(angle),
        earth_radius * np.sin(angle),
        color=COLORS["sky_blue"],
        linestyle="--",
        linewidth=1.0,
        label="Earth orbit radius",
    )
    axes[0, 0].plot(
        venus_radius * np.cos(angle),
        venus_radius * np.sin(angle),
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.0,
        label="Venus orbit radius",
    )
    axes[0, 0].plot(
        trajectory[:, 0], trajectory[:, 1], color=COLORS["blue"], label="Spacecraft"
    )
    axes[0, 0].scatter(
        0.0,
        0.0,
        color=COLORS["yellow"],
        edgecolor=COLORS["black"],
        zorder=4,
        label="Sun",
    )
    axes[0, 0].scatter(
        *EARTH_POSITION[:2], color=COLORS["green"], zorder=4, label="Departure"
    )
    axes[0, 0].scatter(
        *VENUS_POSITION[:2], color=COLORS["vermillion"], zorder=4, label="Arrival"
    )
    axes[0, 0].set_xlabel("Heliocentric x [AU]")
    axes[0, 0].set_ylabel("Heliocentric y [AU]")
    axes[0, 0].set_title("Earth-to-Venus trajectory")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].legend(fontsize=8, ncol=2)

    plot_time, plot_states, plot_controls, _ = _dense_history(solution, 2_001)
    days_u = plot_time / DAY_IN_YEARS
    labels = (r"$u_r$", r"$u_t$", r"$u_n$")
    colors = (COLORS["blue"], COLORS["orange"], COLORS["green"])
    for index, (label, color) in enumerate(zip(labels, colors)):
        axes[0, 1].plot(days_u, plot_controls[index], color=color, label=label)
    axes[0, 1].plot(
        days_u,
        plot_controls[3],
        color=COLORS["black"],
        linestyle="--",
        label=r"Throttle $\rho$",
    )
    axes[0, 1].set_ylabel("Normalized command [-]")
    axes[0, 1].set_title("RTN thrust commands")
    axes[0, 1].legend(ncol=2)

    days_x = plot_time / DAY_IN_YEARS
    axes[1, 0].plot(days_x, plot_states[6], color=COLORS["green"], label="Mass")
    axes[1, 0].set_xlabel("Time since departure [days]")
    axes[1, 0].set_ylabel("Spacecraft mass [kg]")
    axes[1, 0].set_title("Propellant consumption")
    axes[1, 0].legend()

    eccentricity = np.sqrt(plot_states[1] ** 2 + plot_states[2] ** 2)
    axes[1, 1].plot(
        days_x,
        plot_states[0],
        color=COLORS["blue"],
        label=r"Semilatus rectum $p$ [AU]",
    )
    axes[1, 1].plot(
        days_x, eccentricity, color=COLORS["purple"], label=r"Eccentricity $e$ [-]"
    )
    axes[1, 1].set_xlabel("Time since departure [days]")
    axes[1, 1].set_ylabel("Element value")
    axes[1, 1].set_title("Orbit-shape evolution")
    axes[1, 1].legend()

    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "orbit_transfer_solution.png", quick=True
    )

    for position, velocity, elements in (
        (EARTH_POSITION, EARTH_VELOCITY, INITIAL_MEE),
        (VENUS_POSITION, VENUS_VELOCITY, TARGET_MEE),
    ):
        reconstructed_position, reconstructed_velocity = mee_to_cartesian(elements)
        if not np.allclose(
            reconstructed_position, position, atol=2.0e-12, rtol=2.0e-12
        ):
            raise RuntimeError("Cartesian/MEE position round trip failed")
        if not np.allclose(
            reconstructed_velocity, velocity, atol=2.0e-11, rtol=2.0e-12
        ):
            raise RuntimeError("Cartesian/MEE velocity round trip failed")

    system, phase = build_problem(quick=args.quick)
    solution = solve_problem(system, phase, initial_guess(phase), quick=args.quick)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
