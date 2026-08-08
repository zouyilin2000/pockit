"""Schedule root-zone irrigation under weather and soil-water constraints.

The root-zone water balance includes forecast rainfall, moisture-limited crop
evapotranspiration, and nonlinear deep percolation.  Irrigation rate is a
state driven by a bounded valve-ramp command, which produces continuous
schedules.  A hard reserve protects the crop, while a soft water-stress term is
used as a yield-loss proxy when trading crop protection against applied water.

The balance is ``S_dot = rain + irrigation - ET_p*f(S) - drainage(S)``.
The smooth extraction factor ``f`` is zero at the wilting point and normalized
to one at root-zone capacity; drainage rises as the sixth power of relative
storage, approximating rapid percolation near field capacity.

The calculation assumes a perfect deterministic weather forecast and a
spatially uniform root zone. It omits runoff, application losses, and discrete
irrigation windows, so it is a scheduling example rather than a field design.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import trapezoid

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


HORIZON = 14.0  # days
ROOT_ZONE_CAPACITY = 110.0  # mm plant-available water above wilting point
MIN_STORAGE = 12.0  # mm hard crop-protection reserve
INITIAL_STORAGE = 58.0  # mm
MAX_IRRIGATION_RATE = 10.0  # mm/day
MAX_IRRIGATION_RAMP = 8.0  # mm/day^2
STRESS_HALF_SATURATION = 24.0  # mm
DEEP_PERCOLATION_RATE = 2.2  # mm/day at root-zone capacity
IRRIGATION_WEIGHT = 0.030  # 1/mm
STRESS_WEIGHT = 1.0  # 1/day
RAMP_WEIGHT = 2.0e-4  # day^3/mm^2
DENSE_CHECK_POINTS = 6_001


def rainfall(time):
    """Return the forecast effective rainfall in mm/day."""
    return 9.0 * sp.exp(-(((time - 3.8) / 0.55) ** 2)) + 4.0 * sp.exp(
        -(((time - 10.4) / 0.75) ** 2)
    )


def potential_evapotranspiration(time):
    """Return the crop potential evapotranspiration in mm/day."""
    return 5.2 + 0.8 * sp.sin(2.0 * sp.pi * (time - 1.0) / 7.0)


def _rainfall_array(time: np.ndarray) -> np.ndarray:
    return 9.0 * np.exp(-(((time - 3.8) / 0.55) ** 2)) + 4.0 * np.exp(
        -(((time - 10.4) / 0.75) ** 2)
    )


def _potential_evapotranspiration_array(time: np.ndarray) -> np.ndarray:
    return 5.2 + 0.8 * np.sin(2.0 * np.pi * (time - 1.0) / 7.0)


def build_problem():
    """Return the root-zone irrigation scheduling problem."""
    system = System(0)
    phase = system.new_phase(
        ["scaled_root_zone_storage", "irrigation_rate"], ["irrigation_ramp"]
    )
    scaled_storage, irrigation_rate = phase.x
    (irrigation_ramp,) = phase.u
    storage = ROOT_ZONE_CAPACITY * scaled_storage
    stress_factor = (
        storage
        * (ROOT_ZONE_CAPACITY + STRESS_HALF_SATURATION)
        / (ROOT_ZONE_CAPACITY * (storage + STRESS_HALF_SATURATION))
    )
    actual_evapotranspiration = potential_evapotranspiration(phase.t) * stress_factor
    deep_percolation = DEEP_PERCOLATION_RATE * scaled_storage**6
    water_balance = (
        rainfall(phase.t)
        + irrigation_rate
        - actual_evapotranspiration
        - deep_percolation
    )

    phase.set_dynamics([water_balance / ROOT_ZONE_CAPACITY, irrigation_ramp])
    phase.set_integral(
        [
            IRRIGATION_WEIGHT * irrigation_rate
            + STRESS_WEIGHT * (1.0 - stress_factor) ** 2
            + RAMP_WEIGHT * irrigation_ramp**2
        ]
    )
    phase.set_phase_constraint(
        [scaled_storage, irrigation_rate, irrigation_ramp],
        [MIN_STORAGE / ROOT_ZONE_CAPACITY, 0.0, -MAX_IRRIGATION_RAMP],
        [1.0, MAX_IRRIGATION_RATE, MAX_IRRIGATION_RAMP],
    )
    initial_scaled_storage = INITIAL_STORAGE / ROOT_ZONE_CAPACITY
    phase.set_boundary_condition(
        [initial_scaled_storage, 0.0],
        [initial_scaled_storage, 0.0],
        0.0,
        HORIZON,
    )
    # Linear interpolation preserves the nonnegative irrigation-rate bound
    # between nodes while the daily weather forcing remains well resolved.
    phase.set_discretization(140, 2)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Return a cyclic-storage guess with a smoothly tapered irrigation rate."""
    guess = linear_guess(phase, 0.0)
    guess.x[0] = np.full_like(guess.t_x, INITIAL_STORAGE / ROOT_ZONE_CAPACITY)
    dense_time = np.linspace(0.0, HORIZON, 20_001)
    potential_et = _potential_evapotranspiration_array(dense_time)
    stress_factor = (
        INITIAL_STORAGE
        * (ROOT_ZONE_CAPACITY + STRESS_HALF_SATURATION)
        / (ROOT_ZONE_CAPACITY * (INITIAL_STORAGE + STRESS_HALF_SATURATION))
    )
    percolation = DEEP_PERCOLATION_RATE * (INITIAL_STORAGE / ROOT_ZONE_CAPACITY) ** 6
    required_irrigation = (
        trapezoid(
            potential_et * stress_factor + percolation - _rainfall_array(dense_time),
            dense_time,
        )
        / HORIZON
    )
    guess.x[1] = np.full_like(guess.t_x, max(required_irrigation, 0.0))
    guess.u[0] = np.zeros_like(guess.t_u)
    # Boundary values are fixed at zero; tapering the interior rate makes the
    # initial guess closer to that actuator condition without changing scale.
    guess.x[1] *= np.sin(np.pi * guess.t_x / HORIZON) ** 0.25
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    scaled_storage = solution.V_x(time) @ solution.x[0]
    storage = ROOT_ZONE_CAPACITY * scaled_storage
    irrigation_rate = solution.V_x(time) @ solution.x[1]
    irrigation_ramp = solution.V_u(time) @ solution.u[0]
    return time, storage, irrigation_rate, irrigation_ramp


def solve_problem(system, guess):
    """Solve the schedule and verify dense bounds and the water balance."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1600, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, storage, irrigation_rate, irrigation_ramp = _dense_solution(solution)
    rainfall_values = _rainfall_array(time)
    potential_et = _potential_evapotranspiration_array(time)
    stress_factor = (
        storage
        * (ROOT_ZONE_CAPACITY + STRESS_HALF_SATURATION)
        / (ROOT_ZONE_CAPACITY * (storage + STRESS_HALF_SATURATION))
    )
    actual_et = potential_et * stress_factor
    deep_percolation = DEEP_PERCOLATION_RATE * (storage / ROOT_ZONE_CAPACITY) ** 6
    require_finite(
        time=time,
        storage=storage,
        irrigation_rate=irrigation_rate,
        irrigation_ramp=irrigation_ramp,
        rainfall=rainfall_values,
        potential_evapotranspiration=potential_et,
        stress_factor=stress_factor,
        actual_evapotranspiration=actual_et,
        deep_percolation=deep_percolation,
        objective=info["obj_val"],
    )
    path_violation = max(
        float(np.max(MIN_STORAGE - storage)),
        float(np.max(storage - ROOT_ZONE_CAPACITY)),
        float(np.max(-irrigation_rate)),
        float(np.max(irrigation_rate - MAX_IRRIGATION_RATE)),
        float(np.max(np.abs(irrigation_ramp) - MAX_IRRIGATION_RAMP)),
        0.0,
    )
    water_balance_error = abs(
        float(
            storage[-1]
            - storage[0]
            - trapezoid(
                rainfall_values + irrigation_rate - actual_et - deep_percolation,
                time,
            )
        )
    )
    actuator_balance_error = abs(
        float(
            irrigation_rate[-1] - irrigation_rate[0] - trapezoid(irrigation_ramp, time)
        )
    )
    endpoint_error = max(
        abs(float(storage[0] - INITIAL_STORAGE)),
        abs(float(storage[-1] - INITIAL_STORAGE)),
        abs(float(irrigation_rate[0])),
        abs(float(irrigation_rate[-1])),
    )
    if path_violation > 3e-6:
        raise RuntimeError("the irrigation schedule violates a dense path bound")
    if endpoint_error > 2e-6:
        raise RuntimeError("the irrigation schedule violates a cyclic endpoint")
    if max(water_balance_error, actuator_balance_error) > 3e-4:
        raise RuntimeError("the irrigation schedule failed its integral balance check")

    total_irrigation = trapezoid(irrigation_rate, time)
    total_rainfall = trapezoid(rainfall_values, time)
    total_actual_et = trapezoid(actual_et, time)
    total_percolation = trapezoid(deep_percolation, time)
    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"total irrigation: {total_irrigation:.6f} mm")
    print(f"effective rainfall: {total_rainfall:.6f} mm")
    print(f"actual evapotranspiration: {total_actual_et:.6f} mm")
    print(f"deep percolation: {total_percolation:.6f} mm")
    print(f"minimum root-zone storage: {np.min(storage):.6f} mm")
    print(f"dense water-balance error: {water_balance_error:.3e} mm")
    print(f"dense actuator-balance error: {actuator_balance_error:.3e} mm/day")
    reported_path_violation = max(0.0, path_violation)
    print(f"maximum dense path-bound violation: {reported_path_violation:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot soil-water storage, irrigation, weather fluxes, and crop stress."""
    configure_matplotlib()
    time, storage, irrigation_rate, _irrigation_ramp = _dense_solution(solution)
    rainfall_values = _rainfall_array(time)
    potential_et = _potential_evapotranspiration_array(time)
    stress_factor = (
        storage
        * (ROOT_ZONE_CAPACITY + STRESS_HALF_SATURATION)
        / (ROOT_ZONE_CAPACITY * (storage + STRESS_HALF_SATURATION))
    )
    actual_et = potential_et * stress_factor
    deep_percolation = DEEP_PERCOLATION_RATE * (storage / ROOT_ZONE_CAPACITY) ** 6

    fig, axes = plt.subplots(
        3, 1, figsize=(8.4, 7.8), sharex=True, layout="constrained"
    )
    axes[0].axhspan(
        MIN_STORAGE,
        ROOT_ZONE_CAPACITY,
        color=COLORS["green"],
        alpha=0.10,
        label="Admissible root-zone storage",
    )
    axes[0].plot(time, storage, color=COLORS["blue"], label="Available water")
    axes[0].axhline(MIN_STORAGE, color=COLORS["black"], linestyle="--")
    axes[0].set_ylabel("Storage [mm]")
    axes[0].legend()

    axes[1].plot(time, irrigation_rate, color=COLORS["blue"], label="Irrigation")
    axes[1].plot(time, rainfall_values, color=COLORS["sky_blue"], label="Rainfall")
    axes[1].plot(
        time,
        potential_et,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.4,
        label="Potential ET",
    )
    axes[1].plot(time, actual_et, color=COLORS["orange"], label="Actual ET")
    axes[1].plot(time, deep_percolation, color=COLORS["purple"], label="Percolation")
    axes[1].set_ylabel("Water flux [mm/day]")
    axes[1].legend(ncol=3)

    axes[2].plot(
        time,
        1.0 - stress_factor,
        color=COLORS["vermillion"],
        label="Water-stress fraction",
    )
    axes[2].fill_between(
        time, 0.0, 1.0 - stress_factor, color=COLORS["vermillion"], alpha=0.12
    )
    axes[2].set_ylabel("Water-stress fraction")
    axes[2].set_xlabel("Time [days]")
    axes[2].set_ylim(0.0, max(0.25, 1.1 * float(np.max(1.0 - stress_factor))))
    axes[2].legend()

    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "irrigation_scheduling_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
