"""Plan a renewable fishery under ecological and economic constraints.

Biomass follows logistic growth while the harvest rate changes subject to a
bounded adjustment rate.  The manager maximizes discounted dockside value net
of a convex catch cost and harvest-adjustment cost, never allowing the stock
below an ecological floor.  Matching initial and terminal biomass and harvest
removes the usual finite-horizon incentive to liquidate the resource at the
deadline; it does not turn the discounted solution into an infinite-horizon
periodic policy.

The stock equation is ``B_dot = r*B*(1 - B/K) - H`` and the harvest schedule
obeys ``H_dot = u``.  With state-cyclic endpoints, integrating the first
equation requires cumulative catch to equal cumulative natural production.
This aggregate deterministic model omits age structure, recruitment
uncertainty, and price feedback.
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


CARRYING_CAPACITY = 1_000.0  # kt
INTRINSIC_GROWTH_RATE = 0.32  # 1/year
INITIAL_BIOMASS = 0.68 * CARRYING_CAPACITY  # kt
MIN_BIOMASS = 0.42 * CARRYING_CAPACITY  # kt
MAX_HARVEST = 105.0  # kt/year
INITIAL_HARVEST = (
    INTRINSIC_GROWTH_RATE
    * INITIAL_BIOMASS
    * (1.0 - INITIAL_BIOMASS / CARRYING_CAPACITY)
)
MAX_HARVEST_ADJUSTMENT = 40.0  # kt/year^2
HORIZON = 18.0  # years
DISCOUNT_RATE = 0.035  # 1/year
DOCKSIDE_PRICE = 1.0  # normalized value/kt
CATCH_COST = 2.2  # normalized value year/kt
ADJUSTMENT_COST = 0.025  # normalized value * year^3 / kt^2
DENSE_CHECK_POINTS = 6_001


def build_problem():
    """Return the finite-horizon bioeconomic fishery problem."""
    system = System(0)
    phase = system.new_phase(["biomass", "harvest"], ["harvest_adjustment"])
    biomass, harvest = phase.x
    (harvest_adjustment,) = phase.u

    natural_growth = (
        INTRINSIC_GROWTH_RATE * biomass * (1.0 - biomass / CARRYING_CAPACITY)
    )
    phase.set_dynamics([natural_growth - harvest, harvest_adjustment])
    gross_value = DOCKSIDE_PRICE * harvest
    operating_cost = CATCH_COST * harvest**2 / biomass
    adjustment_cost = ADJUSTMENT_COST * harvest_adjustment**2
    discounted_profit = sp.exp(-DISCOUNT_RATE * phase.t) * (
        gross_value - operating_cost - adjustment_cost
    )
    phase.set_integral([-discounted_profit])
    phase.set_phase_constraint(
        [biomass, harvest, harvest_adjustment],
        [MIN_BIOMASS, 0.0, -MAX_HARVEST_ADJUSTMENT],
        [CARRYING_CAPACITY, MAX_HARVEST, MAX_HARVEST_ADJUSTMENT],
    )
    phase.set_boundary_condition(
        [INITIAL_BIOMASS, INITIAL_HARVEST],
        [INITIAL_BIOMASS, INITIAL_HARVEST],
        0.0,
        HORIZON,
    )
    phase.set_discretization(90, 3)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Return the stationary sustainable-harvest policy as a feasible-scale guess."""
    guess = linear_guess(phase, 0.0)
    guess.x[0] = np.full_like(guess.t_x, INITIAL_BIOMASS)
    guess.x[1] = np.full_like(guess.t_x, INITIAL_HARVEST)
    guess.u[0] = np.zeros_like(guess.t_u)
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    biomass = solution.V_x(time) @ solution.x[0]
    harvest = solution.V_x(time) @ solution.x[1]
    harvest_adjustment = solution.V_u(time) @ solution.u[0]
    return time, biomass, harvest, harvest_adjustment


def solve_problem(system, guess):
    """Solve the policy and verify renewable-stock and harvest-rate balances."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1600, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, biomass, harvest, harvest_adjustment = _dense_solution(solution)
    growth = INTRINSIC_GROWTH_RATE * biomass * (1.0 - biomass / CARRYING_CAPACITY)
    require_finite(
        time=time,
        biomass=biomass,
        harvest=harvest,
        harvest_adjustment=harvest_adjustment,
        natural_growth=growth,
        objective=info["obj_val"],
    )
    path_violation = max(
        float(np.max(MIN_BIOMASS - biomass)),
        float(np.max(biomass - CARRYING_CAPACITY)),
        float(np.max(-harvest)),
        float(np.max(harvest - MAX_HARVEST)),
        float(np.max(np.abs(harvest_adjustment) - MAX_HARVEST_ADJUSTMENT)),
        0.0,
    )
    biomass_balance = abs(
        float(biomass[-1] - biomass[0] - trapezoid(growth - harvest, time))
    )
    harvest_rate_balance = abs(
        float(harvest[-1] - harvest[0] - trapezoid(harvest_adjustment, time))
    )
    renewable_balance = abs(float(trapezoid(growth - harvest, time)))
    endpoint_error = max(
        abs(float(biomass[0] - INITIAL_BIOMASS)),
        abs(float(biomass[-1] - INITIAL_BIOMASS)),
        abs(float(harvest[0] - INITIAL_HARVEST)),
        abs(float(harvest[-1] - INITIAL_HARVEST)),
    )
    if path_violation > 3e-6:
        raise RuntimeError("the fishery policy violates a dense path bound")
    if endpoint_error > 2e-6:
        raise RuntimeError("the fishery policy violates a cyclic endpoint")
    if max(biomass_balance, harvest_rate_balance, renewable_balance) > 3e-4:
        raise RuntimeError("the fishery policy failed its integral balance check")

    discounted_profit = trapezoid(
        np.exp(-DISCOUNT_RATE * time)
        * (
            DOCKSIDE_PRICE * harvest
            - CATCH_COST * harvest**2 / biomass
            - ADJUSTMENT_COST * harvest_adjustment**2
        ),
        time,
    )
    require_finite(discounted_profit=discounted_profit)
    objective_error = abs(float(discounted_profit + info["obj_val"]))
    cumulative_harvest = trapezoid(harvest, time)
    if objective_error > 5e-4:
        raise RuntimeError(
            f"dense economic-objective error is too large: {objective_error:.3e}"
        )
    print(f"status: {status_message}")
    print(f"discounted net value: {discounted_profit:.8f}")
    print(f"minimum biomass: {np.min(biomass):.6f} kt")
    print(f"peak harvest: {np.max(harvest):.6f} kt/year")
    print(f"cumulative harvest: {cumulative_harvest:.6f} kt")
    print(f"renewable-stock balance error: {renewable_balance:.3e} kt")
    print(f"dense economic-objective error: {objective_error:.3e}")
    print(f"maximum dense path-bound violation: {path_violation:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot stock, natural growth, harvest, and discounted annual profit."""
    configure_matplotlib()
    time, biomass, harvest, harvest_adjustment = _dense_solution(solution)
    growth = INTRINSIC_GROWTH_RATE * biomass * (1.0 - biomass / CARRYING_CAPACITY)
    discounted_profit_rate = np.exp(-DISCOUNT_RATE * time) * (
        DOCKSIDE_PRICE * harvest
        - CATCH_COST * harvest**2 / biomass
        - ADJUSTMENT_COST * harvest_adjustment**2
    )

    fig, axes = plt.subplots(
        3, 1, figsize=(8.4, 7.8), sharex=True, layout="constrained"
    )
    axes[0].axhspan(
        0.0,
        MIN_BIOMASS,
        color=COLORS["vermillion"],
        alpha=0.10,
        label="Below ecological floor",
    )
    axes[0].plot(time, biomass, color=COLORS["blue"], label="Biomass")
    axes[0].axhline(MIN_BIOMASS, color=COLORS["black"], linestyle="--")
    axes[0].set_ylim(
        MIN_BIOMASS - 35.0,
        max(INITIAL_BIOMASS, float(np.max(biomass))) + 35.0,
    )
    axes[0].set_ylabel("Biomass [kt]")
    axes[0].legend()

    axes[1].plot(time, growth, color=COLORS["green"], label="Natural growth")
    axes[1].plot(time, harvest, color=COLORS["orange"], label="Harvest")
    axes[1].set_ylabel("Flow [kt/year]")
    axes[1].legend(ncol=2)

    axes[2].plot(
        time,
        discounted_profit_rate,
        color=COLORS["purple"],
        label="Discounted net value rate",
    )
    adjustment_axis = axes[2].twinx()
    adjustment_axis.plot(
        time,
        harvest_adjustment,
        color=COLORS["vermillion"],
        label="Harvest adjustment",
    )
    axes[2].set_ylabel("Discounted value rate")
    adjustment_axis.set_ylabel("Harvest adjustment [kt/year^2]")
    axes[2].set_xlabel("Time [years]")
    lines = axes[2].lines + adjustment_axis.lines
    axes[2].legend(lines, [line.get_label() for line in lines], ncol=2)

    style_axes(axes)
    adjustment_axis.grid(False)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "bioeconomic_fishery_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
