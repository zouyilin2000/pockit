"""Solve the classical Almgren-Chriss optimal trade-execution problem.

A trader liquidates a fixed inventory over one trading day. Temporary market
impact penalizes rapid selling, while inventory risk penalizes waiting. The
continuous-time model has a closed-form solution used for numerical validation.
Inventory, trading speed, and both objective weights are normalized, so the
reported cost is a dimensionless benchmark rather than a monetary estimate. The
sell-rate cap is chosen to remain inactive and is checked against the solution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


HORIZON = 1.0  # trading day
ORDER_SIZE = 100_000.0  # shares
TEMPORARY_IMPACT = 0.05
INVENTORY_RISK = 0.20
MAX_SELL_RATE = 3.0  # initial-inventory fractions per day
KAPPA = np.sqrt(INVENTORY_RISK / TEMPORARY_IMPACT)
DENSE_CHECK_POINTS = 4_001


def exact_inventory(time: np.ndarray) -> np.ndarray:
    """Return the normalized closed-form optimal inventory."""
    return np.sinh(KAPPA * (HORIZON - time)) / np.sinh(KAPPA * HORIZON)


def exact_sell_rate(time: np.ndarray) -> np.ndarray:
    """Return the normalized closed-form optimal sell rate."""
    return KAPPA * np.cosh(KAPPA * (HORIZON - time)) / np.sinh(KAPPA * HORIZON)


def build_problem():
    """Return the normalized Almgren-Chriss liquidation problem."""
    system = System(0)
    phase = system.new_phase(["inventory_fraction"], ["sell_rate"])
    (inventory,) = phase.x
    (sell_rate,) = phase.u

    phase.set_dynamics([-sell_rate])
    phase.set_integral(
        [TEMPORARY_IMPACT * sell_rate**2 + INVENTORY_RISK * inventory**2]
    )
    phase.set_phase_constraint([inventory, sell_rate], [0.0, 0.0], [1.0, MAX_SELL_RATE])
    phase.set_boundary_condition([1.0], [0.0], 0.0, HORIZON)
    phase.set_discretization(48, 4)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Use the analytical liquidation path as the solver initial guess."""
    guess = linear_guess(phase, 0.0)
    guess.x[0] = exact_inventory(guess.t_x)
    guess.u[0] = exact_sell_rate(guess.t_u)
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    inventory = solution.V_x(time) @ solution.x[0]
    sell_rate = solution.V_u(time) @ solution.u[0]
    return time, inventory, sell_rate


def solve_problem(system, guess):
    """Solve the execution problem and compare with the closed-form path."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-10, "max_iter": 800, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, inventory, sell_rate = _dense_solution(solution)
    reference_inventory = exact_inventory(time)
    reference_sell_rate = exact_sell_rate(time)
    require_finite(
        time=time,
        inventory=inventory,
        sell_rate=sell_rate,
        reference_inventory=reference_inventory,
        reference_sell_rate=reference_sell_rate,
        objective=info["obj_val"],
    )
    inventory_error = float(np.max(np.abs(inventory - reference_inventory)))
    rate_error = float(np.max(np.abs(sell_rate - reference_sell_rate)))
    path_violation = max(
        float(np.max(-inventory)),
        float(np.max(inventory - 1.0)),
        float(np.max(-sell_rate)),
        float(np.max(sell_rate - MAX_SELL_RATE)),
        0.0,
    )
    if max(inventory_error, rate_error) > 2e-6 or path_violation > 2e-7:
        raise RuntimeError("the execution trajectory failed analytical validation")

    exact_cost = trapezoid(
        TEMPORARY_IMPACT * reference_sell_rate**2
        + INVENTORY_RISK * reference_inventory**2,
        time,
    )
    require_finite(exact_cost=exact_cost)
    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.10f}")
    print(f"analytical objective: {exact_cost:.10f}")
    print(f"maximum inventory error: {inventory_error:.3e}")
    print(f"maximum sell-rate error: {rate_error:.3e}")
    print(f"initial sell rate: {sell_rate[0] * ORDER_SIZE:.3f} shares/day")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot inventory, trading speed, and objective-density components."""
    configure_matplotlib()
    time, inventory, sell_rate = _dense_solution(solution)
    impact_cost = TEMPORARY_IMPACT * sell_rate**2
    risk_cost = INVENTORY_RISK * inventory**2

    fig, axes = plt.subplots(
        3, 1, figsize=(8.2, 7.6), sharex=True, layout="constrained"
    )
    axes[0].plot(time, inventory * ORDER_SIZE, color=COLORS["blue"])
    axes[0].set_ylabel("Inventory [shares]")

    axes[1].plot(time, sell_rate * ORDER_SIZE, color=COLORS["vermillion"])
    axes[1].set_ylabel("Sell rate [shares/day]")

    axes[2].plot(time, impact_cost, color=COLORS["orange"], label="Market impact")
    axes[2].plot(time, risk_cost, color=COLORS["purple"], label="Inventory risk")
    axes[2].set_ylabel("Cost density")
    axes[2].set_xlabel("Time [trading day]")
    axes[2].legend(ncol=2)

    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "optimal_trade_execution_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
