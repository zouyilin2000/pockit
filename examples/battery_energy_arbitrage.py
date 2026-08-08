"""Schedule a grid-connected battery against a time-of-use electricity price.

The battery shifts energy from low-price to high-price hours while respecting
state-of-charge, charge-power, discharge-power, efficiency, and terminal-energy
constraints. A small quadratic power penalty is a smooth surrogate for cycling
wear.

Separate nonnegative charge and discharge controls are an exact relaxation for
the present nonnegative prices, lossy storage, and nondecreasing wear penalty:
simultaneous charging and discharging is strictly dominated under these
assumptions.
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


HORIZON = 24.0  # h
ENERGY_CAPACITY = 100.0  # kWh
MIN_ENERGY = 0.10 * ENERGY_CAPACITY  # kWh
MAX_ENERGY = 0.90 * ENERGY_CAPACITY  # kWh
INITIAL_ENERGY = 50.0  # kWh
MAX_POWER = 20.0  # kW
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.94
DEGRADATION_WEIGHT = 0.003  # currency / (kW^2 h)
DENSE_CHECK_POINTS = 4_001


def electricity_price(time):
    """Return a smooth day-ahead price profile in currency units per kWh."""
    morning = 0.08 * sp.exp(-(((time - 8.0) / 2.0) ** 2))
    evening = 0.22 * sp.exp(-(((time - 19.0) / 2.4) ** 2))
    solar_discount = 0.06 * sp.exp(-(((time - 13.0) / 2.5) ** 2))
    return 0.12 + morning + evening - solar_discount


def _electricity_price_array(time: np.ndarray) -> np.ndarray:
    morning = 0.08 * np.exp(-(((time - 8.0) / 2.0) ** 2))
    evening = 0.22 * np.exp(-(((time - 19.0) / 2.4) ** 2))
    solar_discount = 0.06 * np.exp(-(((time - 13.0) / 2.5) ** 2))
    return 0.12 + morning + evening - solar_discount


def build_problem():
    """Return the configured battery-arbitrage problem."""
    system = System(0)
    phase = system.new_phase(["stored_energy"], ["charge_power", "discharge_power"])
    (energy,) = phase.x
    charge_power, discharge_power = phase.u

    phase.set_dynamics(
        [CHARGE_EFFICIENCY * charge_power - discharge_power / DISCHARGE_EFFICIENCY]
    )
    grid_cost = electricity_price(phase.t) * (charge_power - discharge_power)
    degradation = DEGRADATION_WEIGHT * (charge_power**2 + discharge_power**2)
    phase.set_integral([grid_cost + degradation])
    phase.set_phase_constraint(
        [energy, charge_power, discharge_power],
        [MIN_ENERGY, 0.0, 0.0],
        [MAX_ENERGY, MAX_POWER, MAX_POWER],
    )
    phase.set_boundary_condition([INITIAL_ENERGY], [INITIAL_ENERGY], 0.0, HORIZON)
    phase.set_discretization(96, 2)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Return a feasible no-action schedule with constant stored energy."""
    guess = linear_guess(phase, 0.0)
    guess.x[0] = np.full_like(guess.t_x, INITIAL_ENERGY)
    guess.u[0] = np.zeros_like(guess.t_u)
    guess.u[1] = np.zeros_like(guess.t_u)
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    energy = solution.V_x(time) @ solution.x[0]
    charge = solution.V_u(time) @ solution.u[0]
    discharge = solution.V_u(time) @ solution.u[1]
    return time, energy, charge, discharge


def solve_problem(system, guess):
    """Solve the dispatch problem and verify its dense-time bounds."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1200, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, energy, charge, discharge = _dense_solution(solution)
    price = _electricity_price_array(time)
    require_finite(
        time=time,
        energy=energy,
        charge_power=charge,
        discharge_power=discharge,
        electricity_price=price,
        objective=info["obj_val"],
    )
    path_violation = max(
        float(np.max(MIN_ENERGY - energy)),
        float(np.max(energy - MAX_ENERGY)),
        float(np.max(-charge)),
        float(np.max(charge - MAX_POWER)),
        float(np.max(-discharge)),
        float(np.max(discharge - MAX_POWER)),
        0.0,
    )
    if path_violation > 2e-6:
        raise RuntimeError(f"dense path-bound violation: {path_violation:.3e}")

    grid_cash_flow = trapezoid(price * (charge - discharge), time)
    grid_side_throughput = trapezoid(charge + discharge, time)
    simultaneous_power = float(np.max(np.minimum(charge, discharge)))
    terminal_error = abs(float(energy[-1] - INITIAL_ENERGY))
    if terminal_error > 2e-6:
        raise RuntimeError(f"terminal-energy error: {terminal_error:.3e}")
    if simultaneous_power > 2e-5:
        raise RuntimeError(
            "simultaneous charge/discharge exceeds tolerance: "
            f"{simultaneous_power:.3e} kW"
        )

    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"net energy-market cost: {grid_cash_flow:.6f}")
    print(f"grid-side energy throughput: {grid_side_throughput:.6f} kWh")
    print(f"maximum simultaneous charge/discharge: {simultaneous_power:.3e} kW")
    print(f"maximum dense path-bound violation: {path_violation:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot stored energy, battery power, and the electricity price."""
    configure_matplotlib()
    time, energy, charge, discharge = _dense_solution(solution)
    price = _electricity_price_array(time)

    fig, axes = plt.subplots(
        3, 1, figsize=(8.4, 7.6), sharex=True, layout="constrained"
    )
    axes[0].axhspan(
        MIN_ENERGY,
        MAX_ENERGY,
        color=COLORS["green"],
        alpha=0.10,
        label="Allowed range",
    )
    axes[0].plot(time, energy, color=COLORS["blue"], label="Stored energy")
    axes[0].set_ylabel("Energy [kWh]")
    axes[0].legend()

    axes[1].plot(time, charge, color=COLORS["green"], label="Charge")
    axes[1].plot(time, -discharge, color=COLORS["vermillion"], label="Discharge")
    axes[1].axhline(0.0, color=COLORS["black"], linewidth=1.0)
    axes[1].set_ylabel("Grid power [kW]")
    axes[1].legend(ncol=2)

    axes[2].plot(time, price, color=COLORS["purple"])
    axes[2].fill_between(time, 0.0, price, color=COLORS["purple"], alpha=0.12)
    axes[2].set_ylabel("Price [currency/kWh]")
    axes[2].set_xlabel("Time [h]")
    axes[2].set_ylim(0.0, 1.1 * float(np.max(price)))

    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "battery_energy_arbitrage_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
