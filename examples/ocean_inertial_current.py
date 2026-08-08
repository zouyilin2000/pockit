"""Generate an ocean mixed-layer inertial current with minimum wind stress.

A homogeneous Northern Hemisphere mixed layer starts at rest. The optimizer
chooses two horizontal surface-stress components that create a prescribed
eastward current after one day while Coriolis acceleration rotates the flow
and linear drag dissipates momentum. The result is checked against the exact
minimum-energy solution from the finite-horizon controllability Gramian.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp, trapezoid

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


SECONDS_PER_HOUR = 3_600.0
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s
LATITUDE = np.deg2rad(45.0)
CORIOLIS_PARAMETER = (
    2.0 * EARTH_ROTATION_RATE * np.sin(LATITUDE) * SECONDS_PER_HOUR
)  # rad/h
MIXED_LAYER_DENSITY = 1_025.0  # kg/m^3
MIXED_LAYER_DEPTH = 50.0  # m
STRESS_ACCELERATION_GAIN = SECONDS_PER_HOUR / (
    MIXED_LAYER_DENSITY * MIXED_LAYER_DEPTH
)  # (m/s)/h per (N/m^2)
DAMPING_TIMESCALE = 48.0  # h
LINEAR_DAMPING_RATE = 1.0 / DAMPING_TIMESCALE  # 1/h
HORIZON = 24.0  # h
TARGET_CURRENT = np.array([0.20, 0.0])  # east, north [m/s]
MAX_CURRENT_COMPONENT = 0.35  # m/s
MAX_WIND_STRESS_COMPONENT = 0.30  # N/m^2
WIND_STRESS_SCALE = 0.25  # N/m^2
DENSE_CHECK_POINTS = 4_001


def _gramian_scalar() -> float:
    """Return the isotropic finite-horizon controllability Gramian entry."""
    decay_integral = -np.expm1(-2.0 * LINEAR_DAMPING_RATE * HORIZON) / (
        2.0 * LINEAR_DAMPING_RATE
    )
    return STRESS_ACCELERATION_GAIN**2 * decay_integral


def _analytical_reference(
    time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact minimum-energy current and stress at supplied times."""
    time = np.asarray(time, dtype=float)
    remaining = HORIZON - time
    gramian = _gramian_scalar()

    angle_to_go = CORIOLIS_PARAMETER * remaining
    cosine_to_go = np.cos(angle_to_go)
    sine_to_go = np.sin(angle_to_go)
    stress_gain = (
        STRESS_ACCELERATION_GAIN * np.exp(-LINEAR_DAMPING_RATE * remaining) / gramian
    )
    stress = stress_gain * np.vstack(
        [
            cosine_to_go * TARGET_CURRENT[0] - sine_to_go * TARGET_CURRENT[1],
            sine_to_go * TARGET_CURRENT[0] + cosine_to_go * TARGET_CURRENT[1],
        ]
    )

    accumulated_gramian = (
        STRESS_ACCELERATION_GAIN**2
        * np.exp(-LINEAR_DAMPING_RATE * remaining)
        * (-np.expm1(-2.0 * LINEAR_DAMPING_RATE * time))
        / (2.0 * LINEAR_DAMPING_RATE)
    )
    state_gain = accumulated_gramian / gramian
    state_angle = CORIOLIS_PARAMETER * (time - HORIZON)
    state_cosine = np.cos(state_angle)
    state_sine = np.sin(state_angle)
    current = state_gain * np.vstack(
        [
            state_cosine * TARGET_CURRENT[0] + state_sine * TARGET_CURRENT[1],
            -state_sine * TARGET_CURRENT[0] + state_cosine * TARGET_CURRENT[1],
        ]
    )
    return current, stress


def _analytical_stress_energy() -> float:
    """Return the exact integral of squared stress in (N/m^2)^2 h."""
    return float(TARGET_CURRENT @ TARGET_CURRENT / _gramian_scalar())


def build_problem(quick: bool = False):
    """Build the minimum-stress inertial-current forcing problem."""
    system = System(0)
    phase = system.new_phase(
        ["eastward_current", "northward_current"],
        ["eastward_wind_stress", "northward_wind_stress"],
    )
    eastward_current, northward_current = phase.x
    eastward_stress, northward_stress = phase.u

    phase.set_dynamics(
        [
            CORIOLIS_PARAMETER * northward_current
            - LINEAR_DAMPING_RATE * eastward_current
            + STRESS_ACCELERATION_GAIN * eastward_stress,
            -CORIOLIS_PARAMETER * eastward_current
            - LINEAR_DAMPING_RATE * northward_current
            + STRESS_ACCELERATION_GAIN * northward_stress,
        ]
    )
    phase.set_integral(
        [
            (eastward_stress / WIND_STRESS_SCALE) ** 2
            + (northward_stress / WIND_STRESS_SCALE) ** 2
        ]
    )
    phase.set_phase_constraint(
        [
            eastward_current,
            northward_current,
            eastward_stress,
            northward_stress,
        ],
        [
            -MAX_CURRENT_COMPONENT,
            -MAX_CURRENT_COMPONENT,
            -MAX_WIND_STRESS_COMPONENT,
            -MAX_WIND_STRESS_COMPONENT,
        ],
        [
            MAX_CURRENT_COMPONENT,
            MAX_CURRENT_COMPONENT,
            MAX_WIND_STRESS_COMPONENT,
            MAX_WIND_STRESS_COMPONENT,
        ],
    )
    phase.set_boundary_condition(
        [0.0, 0.0],
        TARGET_CURRENT.tolist(),
        0.0,
        HORIZON,
    )
    phase.set_discretization(32 if quick else 48, 4)

    system.set_phase([phase])
    system.set_objective(phase.I[0] / HORIZON)
    return system, phase


def initial_guess(phase):
    """Use the exact unconstrained optimum as a feasible initial guess."""
    guess = linear_guess(phase, 0.0)
    current, _ = _analytical_reference(guess.t_x)
    _, stress = _analytical_reference(guess.t_u)
    for index in range(2):
        guess.x[index] = current[index]
        guess.u[index] = stress[index]
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    current = np.vstack([solution.V_x(time) @ component for component in solution.x])
    stress = np.vstack([solution.V_u(time) @ component for component in solution.u])
    return time, current, stress


def _forward_response(time: np.ndarray, stress: np.ndarray) -> np.ndarray:
    """Independently integrate the slab-ocean dynamics for sampled stress."""

    def dynamics(current_time, current):
        eastward_stress = np.interp(current_time, time, stress[0])
        northward_stress = np.interp(current_time, time, stress[1])
        return [
            CORIOLIS_PARAMETER * current[1]
            - LINEAR_DAMPING_RATE * current[0]
            + STRESS_ACCELERATION_GAIN * eastward_stress,
            -CORIOLIS_PARAMETER * current[0]
            - LINEAR_DAMPING_RATE * current[1]
            + STRESS_ACCELERATION_GAIN * northward_stress,
        ]

    result = solve_ivp(
        dynamics,
        (time[0], time[-1]),
        [0.0, 0.0],
        t_eval=time,
        rtol=2.0e-11,
        atol=2.0e-13,
        method="DOP853",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y


def solve_problem(system, guess, quick: bool = False):
    """Solve and verify the current against analytical and forward references."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={
            "tol": 2.0e-8 if quick else 1.0e-10,
            "acceptable_tol": 2.0e-7 if quick else 1.0e-9,
            "max_iter": 1_000,
            "print_level": 0,
            "sb": "yes",
            "bound_relax_factor": 0.0,
        },
    )
    status = int(info["status"])
    status_message = _status_message(info)
    if status not in (0, 1):
        raise RuntimeError(f"Ipopt failed ({status}): {status_message}")

    time, current, stress = _dense_solution(solution)
    reference_current, reference_stress = _analytical_reference(time)
    reintegrated_current = _forward_response(time, stress)
    require_finite(
        current=current,
        stress=stress,
        reference_current=reference_current,
        reference_stress=reference_stress,
        reintegrated_current=reintegrated_current,
        objective=info["obj_val"],
    )
    current_error = float(np.max(np.abs(current - reference_current)))
    stress_error = float(np.max(np.abs(stress - reference_stress)))
    forward_error = float(np.max(np.abs(reintegrated_current - current)))
    endpoint_error = float(np.max(np.abs(current[:, -1] - TARGET_CURRENT)))
    path_violation = max(
        float(np.max(np.abs(current) - MAX_CURRENT_COMPONENT)),
        float(np.max(np.abs(stress) - MAX_WIND_STRESS_COMPONENT)),
        0.0,
    )

    stress_energy = float(trapezoid(np.sum(stress**2, axis=0), time))
    reference_energy = _analytical_stress_energy()
    dense_objective = stress_energy / (HORIZON * WIND_STRESS_SCALE**2)
    objective_error = max(
        abs(float(info["obj_val"]) - dense_objective),
        abs(dense_objective - reference_energy / (HORIZON * WIND_STRESS_SCALE**2)),
    )

    current_tolerance = 1.0e-6 if quick else 2.0e-7
    stress_tolerance = 5.0e-5 if quick else 1.5e-5
    forward_tolerance = 2.0e-6 if quick else 5.0e-7
    if current_error > current_tolerance or stress_error > stress_tolerance:
        raise RuntimeError(
            "the collocation solution does not match the Gramian reference: "
            f"current={current_error:.3e}, stress={stress_error:.3e}"
        )
    if forward_error > forward_tolerance:
        raise RuntimeError(
            f"independent forward-integration error is too large: {forward_error:.3e}"
        )
    if endpoint_error > 2.0e-7 or path_violation > 2.0e-7:
        raise RuntimeError("the dense inertial-current trajectory is infeasible")
    if objective_error > (2.0e-8 if quick else 5.0e-9):
        raise RuntimeError(
            f"the numerical and analytical objectives differ: {objective_error:.3e}"
        )

    current_speed = np.linalg.norm(current, axis=0)
    stress_magnitude = np.linalg.norm(stress, axis=0)
    print(f"status: {status_message}")
    print(f"normalized objective: {float(info['obj_val']):.10f}")
    print(f"stress energy: {stress_energy:.10f} (N/m^2)^2 h")
    print(f"peak current speed: {np.max(current_speed):.8f} m/s")
    print(f"peak wind-stress magnitude: {np.max(stress_magnitude):.8f} N/m^2")
    print(f"maximum analytical current error: {current_error:.3e} m/s")
    print(f"maximum analytical stress error: {stress_error:.3e} N/m^2")
    print(f"maximum forward-integration error: {forward_error:.3e} m/s")
    print(f"maximum dense path-bound violation: {path_violation:.3e}")
    return solution


def plot_solution(
    solution,
    *,
    save: str | Path | None = None,
    show: bool = True,
):
    """Plot current, stress, hodograph, and accumulated forcing effort."""
    configure_matplotlib()
    time, current, stress = _dense_solution(solution)
    reference_current, reference_stress = _analytical_reference(time)
    cumulative_objective = cumulative_trapezoid(
        np.sum(stress**2, axis=0) / (HORIZON * WIND_STRESS_SCALE**2),
        time,
        initial=0.0,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), layout="constrained")
    current_axis, hodograph_axis, stress_axis, effort_axis = axes.reshape(-1)

    component_labels = ("Eastward", "Northward")
    component_colors = (COLORS["blue"], COLORS["orange"])
    for index, (label, color) in enumerate(zip(component_labels, component_colors)):
        current_axis.plot(
            time,
            current[index],
            color=color,
            label=label,
        )
        current_axis.plot(
            time,
            reference_current[index],
            color=color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            label="Analytical reference" if index == 0 else None,
        )
    current_axis.set_ylabel("Mixed-layer current [m/s]")
    current_axis.set_xlabel("Time [h]")
    current_axis.set_title("Current components")
    current_axis.legend(ncol=3, fontsize=8)

    hodograph_axis.plot(
        current[0],
        current[1],
        color=COLORS["blue"],
        label="Pockit trajectory",
    )
    hodograph_axis.plot(
        reference_current[0],
        reference_current[1],
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.2,
        label="Gramian reference",
    )
    hodograph_axis.scatter(
        current[0, 0],
        current[1, 0],
        color=COLORS["vermillion"],
        s=32,
        label="Initial",
        zorder=3,
    )
    hodograph_axis.scatter(
        *TARGET_CURRENT,
        color=COLORS["green"],
        marker="^",
        s=42,
        label="Target",
        zorder=3,
    )
    hodograph_axis.set_xlabel("Eastward current [m/s]")
    hodograph_axis.set_ylabel("Northward current [m/s]")
    hodograph_axis.set_title("Inertial-current hodograph")
    hodograph_axis.set_aspect("equal", adjustable="datalim")
    hodograph_axis.legend(fontsize=8)

    stress_colors = (COLORS["green"], COLORS["vermillion"])
    for index, (label, color) in enumerate(zip(component_labels, stress_colors)):
        stress_axis.plot(
            time,
            stress[index],
            color=color,
            label=label,
        )
        stress_axis.plot(
            time,
            reference_stress[index],
            color=color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            label="Analytical reference" if index == 0 else None,
        )
    stress_axis.axhline(
        MAX_WIND_STRESS_COMPONENT,
        color=COLORS["black"],
        linestyle=":",
        linewidth=1.0,
        label="Component limit",
    )
    stress_axis.axhline(
        -MAX_WIND_STRESS_COMPONENT,
        color=COLORS["black"],
        linestyle=":",
        linewidth=1.0,
    )
    stress_axis.set_ylabel("Surface stress [N/m^2]")
    stress_axis.set_xlabel("Time [h]")
    stress_axis.set_title("Minimum-energy wind stress")
    stress_axis.set_ylim(-0.34, 0.38)
    stress_axis.legend(ncol=4, fontsize=7)

    effort_axis.plot(
        time,
        cumulative_objective,
        color=COLORS["purple"],
        label="Accumulated objective",
    )
    effort_axis.scatter(
        time[-1],
        cumulative_objective[-1],
        color=COLORS["purple"],
        s=30,
        zorder=3,
    )
    effort_axis.set_xlabel("Time [h]")
    effort_axis.set_ylabel("Normalized accumulated effort [-]")
    effort_axis.set_title("Forcing-effort accumulation")
    effort_axis.legend()

    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(
        __doc__.splitlines()[0],
        "ocean_inertial_current_solution.png",
        quick=True,
    )
    system, phase = build_problem(quick=args.quick)
    solution = solve_problem(
        system,
        initial_guess(phase),
        quick=args.quick,
    )
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
