"""Schedule two-zone HVAC cooling under comfort and electricity constraints.

The example uses a resistance-capacitance thermal model for two neighbouring
office zones. Cooling can be shifted toward lower-price hours by pre-cooling,
while hard temperature bounds keep both zones comfortable over a 12-hour
occupied period. Resetting both terminal temperatures to their initial values
is a finite-horizon bookkeeping condition that prevents borrowing free thermal
storage beyond the scheduling window; it is not a 24-hour periodic heat balance.
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


HORIZON = 12.0  # h
THERMAL_CAPACITANCE = np.array([4.0, 3.2])  # kWh/K
OUTDOOR_CONDUCTANCE = np.array([0.55, 0.42])  # kW/K
INTERZONE_CONDUCTANCE = 0.18  # kW/K
INTERNAL_GAIN = np.array([1.20, 0.80])  # kW
COEFFICIENT_OF_PERFORMANCE = 3.2
MAX_COOLING_POWER = 4.0  # kW electrical, per zone
MIN_TEMPERATURE = 22.5  # deg C
MAX_TEMPERATURE = 25.0  # deg C
INITIAL_TEMPERATURE = np.array([24.0, 24.5])  # deg C
CONTROL_REGULARIZATION = 0.004  # currency / (kW^2 h)
DENSE_CHECK_POINTS = 4_001


def outdoor_temperature(time):
    """Return the smooth daytime outdoor-temperature profile in deg C."""
    return 30.0 + 4.0 * sp.sin(sp.pi * time / HORIZON)


def electricity_price(time):
    """Return the time-of-use electricity price in currency units per kWh."""
    return 0.12 + 0.18 * sp.exp(-(((time - 8.0) / 2.0) ** 2))


def _outdoor_temperature_array(time: np.ndarray) -> np.ndarray:
    return 30.0 + 4.0 * np.sin(np.pi * time / HORIZON)


def _electricity_price_array(time: np.ndarray) -> np.ndarray:
    return 0.12 + 0.18 * np.exp(-(((time - 8.0) / 2.0) ** 2))


def build_problem():
    """Return the configured two-zone building energy-management problem."""
    system = System(0)
    phase = system.new_phase(
        ["zone_1_temperature", "zone_2_temperature"],
        ["zone_1_cooling_power", "zone_2_cooling_power"],
    )
    temperature_1, temperature_2 = phase.x
    power_1, power_2 = phase.u

    outdoor = outdoor_temperature(phase.t)
    heat_flow_1 = (
        OUTDOOR_CONDUCTANCE[0] * (outdoor - temperature_1)
        + INTERZONE_CONDUCTANCE * (temperature_2 - temperature_1)
        + INTERNAL_GAIN[0]
        - COEFFICIENT_OF_PERFORMANCE * power_1
    )
    heat_flow_2 = (
        OUTDOOR_CONDUCTANCE[1] * (outdoor - temperature_2)
        + INTERZONE_CONDUCTANCE * (temperature_1 - temperature_2)
        + INTERNAL_GAIN[1]
        - COEFFICIENT_OF_PERFORMANCE * power_2
    )
    phase.set_dynamics(
        [
            heat_flow_1 / THERMAL_CAPACITANCE[0],
            heat_flow_2 / THERMAL_CAPACITANCE[1],
        ]
    )

    energy_cost = electricity_price(phase.t) * (power_1 + power_2)
    regularization = CONTROL_REGULARIZATION * (power_1**2 + power_2**2)
    phase.set_integral([energy_cost + regularization])
    phase.set_phase_constraint(
        [temperature_1, temperature_2, power_1, power_2],
        [MIN_TEMPERATURE, MIN_TEMPERATURE, 0.0, 0.0],
        [
            MAX_TEMPERATURE,
            MAX_TEMPERATURE,
            MAX_COOLING_POWER,
            MAX_COOLING_POWER,
        ],
    )
    phase.set_boundary_condition(
        INITIAL_TEMPERATURE.tolist(), INITIAL_TEMPERATURE.tolist(), 0.0, HORIZON
    )
    # Piecewise-linear states make nodal comfort bounds valid between nodes.
    # The fine mesh still resolves the smooth weather and price profiles.
    phase.set_discretization(72, 2)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct a physically scaled temperature and cooling-power guess."""
    guess = linear_guess(phase, 0.0)
    normalized_time = guess.t_x / HORIZON
    guess.x[0] = INITIAL_TEMPERATURE[0] + 0.7 * normalized_time
    guess.x[1] = INITIAL_TEMPERATURE[1] + 0.4 * normalized_time
    guess.u[0] = np.full_like(guess.t_u, 1.8)
    guess.u[1] = np.full_like(guess.t_u, 1.3)
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    temperatures = np.vstack(
        [solution.V_x(time) @ component for component in solution.x]
    )
    powers = np.vstack([solution.V_u(time) @ component for component in solution.u])
    return time, temperatures, powers


def solve_problem(system, guess):
    """Optimize the cooling schedule and verify dense-time feasibility."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1200, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, temperatures, powers = _dense_solution(solution)
    price = _electricity_price_array(time)
    require_finite(
        time=time,
        temperatures=temperatures,
        cooling_powers=powers,
        electricity_price=price,
        objective=info["obj_val"],
    )
    path_violation = max(
        float(np.max(MIN_TEMPERATURE - temperatures)),
        float(np.max(temperatures - MAX_TEMPERATURE)),
        float(np.max(-powers)),
        float(np.max(powers - MAX_COOLING_POWER)),
        0.0,
    )
    if path_violation > 2e-6:
        raise RuntimeError(f"dense path-bound violation: {path_violation:.3e}")

    energy = trapezoid(np.sum(powers, axis=0), time)
    energy_charge = trapezoid(price * np.sum(powers, axis=0), time)
    peak_power = float(np.max(np.sum(powers, axis=0)))
    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"cooling electricity: {energy:.6f} kWh")
    print(f"time-of-use energy charge: {energy_charge:.6f}")
    print(f"peak total electrical power: {peak_power:.6f} kW")
    print(f"maximum dense path-bound violation: {path_violation:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot temperatures, cooling powers, weather, and electricity price."""
    configure_matplotlib()
    time, temperatures, powers = _dense_solution(solution)
    outdoor = _outdoor_temperature_array(time)
    price = _electricity_price_array(time)

    fig, axes = plt.subplots(
        3, 1, figsize=(8.4, 8.0), sharex=True, layout="constrained"
    )
    axes[0].axhspan(
        MIN_TEMPERATURE,
        MAX_TEMPERATURE,
        color=COLORS["green"],
        alpha=0.12,
        label="Comfort band",
    )
    axes[0].plot(time, temperatures[0], color=COLORS["blue"], label="Zone 1")
    axes[0].plot(time, temperatures[1], color=COLORS["orange"], label="Zone 2")
    axes[0].plot(
        time,
        outdoor,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.4,
        label="Outdoor",
    )
    axes[0].set_ylabel("Temperature [deg C]")
    axes[0].legend(ncol=2)

    axes[1].plot(time, powers[0], color=COLORS["blue"], label="Zone 1")
    axes[1].plot(time, powers[1], color=COLORS["orange"], label="Zone 2")
    axes[1].set_ylabel("HVAC electric power [kW]")
    axes[1].set_ylim(-0.1, MAX_COOLING_POWER + 0.25)
    axes[1].legend(ncol=2)

    axes[2].plot(time, price, color=COLORS["vermillion"])
    axes[2].fill_between(time, 0.0, price, color=COLORS["vermillion"], alpha=0.12)
    axes[2].set_ylabel("Price [currency/kWh]")
    axes[2].set_xlabel("Time [h]")
    axes[2].set_ylim(0.0, 1.1 * float(np.max(price)))

    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "building_hvac_control_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
