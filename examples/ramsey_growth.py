"""Compute a finite-horizon transition in the Ramsey growth model.

The social planner allocates output between consumption and investment. The
economy starts below its modified-golden-rule steady state and must reach that
capital stock while maximizing discounted logarithmic utility. Terminal
consumption is free, so this is a finite-horizon capital transition rather than
complete convergence to a steady path. Utility uses ``log(c / c_ss)``; changing
that normalization shifts the reported welfare by a constant.
"""

from __future__ import annotations

from pathlib import Path

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


PRODUCTIVITY = 1.0
CAPITAL_SHARE = 0.33
DEPRECIATION = 0.06  # 1/year
DISCOUNT_RATE = 0.04  # 1/year
HORIZON = 30.0  # years
STEADY_CAPITAL = (CAPITAL_SHARE * PRODUCTIVITY / (DISCOUNT_RATE + DEPRECIATION)) ** (
    1.0 / (1.0 - CAPITAL_SHARE)
)
STEADY_CONSUMPTION = (
    PRODUCTIVITY * STEADY_CAPITAL**CAPITAL_SHARE - DEPRECIATION * STEADY_CAPITAL
)
INITIAL_CAPITAL = 0.55 * STEADY_CAPITAL
MIN_CAPITAL = 0.20 * STEADY_CAPITAL
MAX_CAPITAL = 1.20 * STEADY_CAPITAL
MIN_CONSUMPTION = 0.05 * STEADY_CONSUMPTION
MAX_CONSUMPTION = 1.80 * STEADY_CONSUMPTION
DENSE_CHECK_POINTS = 4_001


def build_problem():
    """Return the finite-horizon Ramsey planner problem."""
    system = System(0)
    phase = system.new_phase(["capital"], ["consumption"])
    (capital,) = phase.x
    (consumption,) = phase.u

    output = PRODUCTIVITY * capital**CAPITAL_SHARE
    phase.set_dynamics([output - DEPRECIATION * capital - consumption])
    discounted_utility = sp.exp(-DISCOUNT_RATE * phase.t) * sp.log(
        consumption / STEADY_CONSUMPTION
    )
    phase.set_integral([-discounted_utility])
    phase.set_phase_constraint(
        [capital, consumption],
        [MIN_CAPITAL, MIN_CONSUMPTION],
        [MAX_CAPITAL, MAX_CONSUMPTION],
    )
    phase.set_boundary_condition([INITIAL_CAPITAL], [STEADY_CAPITAL], 0.0, HORIZON)
    phase.set_discretization(90, 2)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct a positive transition guess near the resource balance."""
    guess = linear_guess(phase, STEADY_CONSUMPTION)
    normalized_time = guess.t_x / HORIZON
    guess.x[0] = INITIAL_CAPITAL + (STEADY_CAPITAL - INITIAL_CAPITAL) * normalized_time
    capital_at_control = guess.V_x(guess.t_u) @ guess.x[0]
    required_investment = (STEADY_CAPITAL - INITIAL_CAPITAL) / HORIZON
    guess.u[0] = (
        PRODUCTIVITY * capital_at_control**CAPITAL_SHARE
        - DEPRECIATION * capital_at_control
        - required_investment
    )
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    capital = solution.V_x(time) @ solution.x[0]
    consumption = solution.V_u(time) @ solution.u[0]
    return time, capital, consumption


def solve_problem(system, guess):
    """Solve the planner problem and check feasibility and economic residuals."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1500, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, capital, consumption = _dense_solution(solution)
    require_finite(
        time=time,
        capital=capital,
        consumption=consumption,
        objective=info["obj_val"],
    )
    path_violation = max(
        float(np.max(MIN_CAPITAL - capital)),
        float(np.max(capital - MAX_CAPITAL)),
        float(np.max(MIN_CONSUMPTION - consumption)),
        float(np.max(consumption - MAX_CONSUMPTION)),
        0.0,
    )
    terminal_error = abs(float(capital[-1] - STEADY_CAPITAL))
    if path_violation > 2e-6 or terminal_error > 2e-6:
        raise RuntimeError(
            "the computed transition violates a bound or terminal condition"
        )

    # On an interior log-utility arc, the Euler equation is
    # d(log c)/dt = MPK - depreciation - discount rate.
    midpoint_time = 0.5 * (solution.t_u[:-1] + solution.t_u[1:])
    midpoint_consumption = 0.5 * (solution.u[0][:-1] + solution.u[0][1:])
    consumption_rate = np.diff(solution.u[0]) / np.diff(solution.t_u)
    midpoint_capital = solution.V_x(midpoint_time) @ solution.x[0]
    euler_residual = consumption_rate / midpoint_consumption - (
        CAPITAL_SHARE * PRODUCTIVITY * midpoint_capital ** (CAPITAL_SHARE - 1.0)
        - DEPRECIATION
        - DISCOUNT_RATE
    )
    require_finite(euler_residual=euler_residual)
    interior = (
        (midpoint_consumption > MIN_CONSUMPTION + 1e-3)
        & (midpoint_consumption < MAX_CONSUMPTION - 1e-3)
        & (midpoint_time > 1.0)
        & (midpoint_time < HORIZON - 1.0)
    )
    if not np.any(interior):
        raise RuntimeError("no interior intervals are available for the Euler check")
    euler_rms = float(np.sqrt(np.mean(euler_residual[interior] ** 2)))
    if euler_rms > 5e-4:
        raise RuntimeError(f"Euler-equation RMS residual: {euler_rms:.3e}")

    print(f"status: {status_message}")
    print(f"discounted welfare: {-info['obj_val']:.8f}")
    print(f"steady-state capital: {STEADY_CAPITAL:.6f}")
    print(f"steady-state consumption benchmark: {STEADY_CONSUMPTION:.6f}")
    print(f"finite-horizon terminal consumption: {consumption[-1]:.6f}")
    print(f"interior Euler-equation RMS residual: {euler_rms:.3e} 1/year")
    print(f"maximum dense path-bound violation: {path_violation:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot capital, consumption, output, and the net marginal return."""
    configure_matplotlib()
    time, capital, consumption = _dense_solution(solution)
    output = PRODUCTIVITY * capital**CAPITAL_SHARE
    net_marginal_return = (
        CAPITAL_SHARE * PRODUCTIVITY * capital ** (CAPITAL_SHARE - 1.0) - DEPRECIATION
    )

    fig, axes = plt.subplots(
        3, 1, figsize=(8.2, 7.8), sharex=True, layout="constrained"
    )
    axes[0].plot(time, capital, color=COLORS["blue"], label="Capital")
    axes[0].axhline(
        STEADY_CAPITAL,
        color=COLORS["black"],
        linestyle="--",
        label="Modified golden rule",
    )
    axes[0].set_ylabel("Capital per worker")
    axes[0].legend(ncol=2)

    axes[1].plot(time, consumption, color=COLORS["green"], label="Consumption")
    axes[1].plot(time, output, color=COLORS["orange"], label="Output")
    axes[1].set_ylabel("Flow per worker/year")
    axes[1].legend(ncol=2)

    axes[2].plot(time, net_marginal_return, color=COLORS["purple"])
    axes[2].axhline(
        DISCOUNT_RATE,
        color=COLORS["black"],
        linestyle="--",
        label="Discount rate",
    )
    axes[2].set_ylabel("Net return [1/year]")
    axes[2].set_xlabel("Time [years]")
    axes[2].legend()

    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(__doc__.splitlines()[0], "ramsey_growth_solution.png")
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
