"""Schedule reservoir releases through a deterministic flood hydrograph.

The operator draws down storage before peak inflow, limits downstream release,
and returns the reservoir to its initial storage after the event. The example
uses a standard lumped water-balance model with explicit physical units.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import cumulative_trapezoid, trapezoid

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


HORIZON = 10.0  # days
INITIAL_STORAGE = 40.0  # million m^3
MIN_STORAGE = 20.0  # million m^3
MAX_STORAGE = 80.0  # million m^3
MAX_RELEASE = 15.0  # million m^3/day
REFERENCE_RELEASE = 5.0  # million m^3/day
STORAGE_WEIGHT = 1.0
RELEASE_WEIGHT = 0.025
DENSE_CHECK_POINTS = 4_001


def inflow(time):
    """Return the forecast flood hydrograph in million m^3/day."""
    return 3.0 + 18.0 * sp.exp(-(((time - 4.0) / 1.4) ** 2))


def _inflow_array(time: np.ndarray) -> np.ndarray:
    return 3.0 + 18.0 * np.exp(-(((time - 4.0) / 1.4) ** 2))


def build_problem():
    """Return the reservoir release-scheduling problem."""
    system = System(0)
    phase = system.new_phase(["storage"], ["release"])
    (storage,) = phase.x
    (release,) = phase.u

    phase.set_dynamics([inflow(phase.t) - release])
    storage_penalty = STORAGE_WEIGHT * ((storage - INITIAL_STORAGE) / 40.0) ** 2
    release_penalty = RELEASE_WEIGHT * ((release - REFERENCE_RELEASE) / 10.0) ** 2
    phase.set_integral([storage_penalty + release_penalty])
    phase.set_phase_constraint(
        [storage, release], [MIN_STORAGE, 0.0], [MAX_STORAGE, MAX_RELEASE]
    )
    phase.set_boundary_condition([INITIAL_STORAGE], [INITIAL_STORAGE], 0.0, HORIZON)
    phase.set_discretization(100, 2)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Return a constant-storage guess using the mean forecast inflow."""
    guess = linear_guess(phase, 0.0)
    guess.x[0] = np.full_like(guess.t_x, INITIAL_STORAGE)
    dense_time = np.linspace(0.0, HORIZON, 10_001)
    mean_inflow = trapezoid(_inflow_array(dense_time), dense_time) / HORIZON
    guess.u[0] = np.full_like(guess.t_u, mean_inflow)
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    storage = solution.V_x(time) @ solution.x[0]
    release = solution.V_u(time) @ solution.u[0]
    return time, storage, release


def solve_problem(system, guess):
    """Optimize reservoir releases and verify water balance and bounds."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1200, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, storage, release = _dense_solution(solution)
    inflow_values = _inflow_array(time)
    require_finite(
        time=time,
        storage=storage,
        release=release,
        inflow=inflow_values,
        objective=info["obj_val"],
    )
    path_violation = max(
        float(np.max(MIN_STORAGE - storage)),
        float(np.max(storage - MAX_STORAGE)),
        float(np.max(-release)),
        float(np.max(release - MAX_RELEASE)),
        0.0,
    )
    cumulative_net_inflow = cumulative_trapezoid(
        inflow_values - release, time, initial=0.0
    )
    balance_residual = storage - storage[0] - cumulative_net_inflow
    maximum_balance_error = float(np.max(np.abs(balance_residual)))
    balance_tolerance = 5e-4 * (MAX_STORAGE - MIN_STORAGE)
    if path_violation > 2e-6 or maximum_balance_error > balance_tolerance:
        raise RuntimeError("the reservoir schedule failed dense validation")

    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"minimum storage: {np.min(storage):.6f} million m^3")
    print(f"maximum storage: {np.max(storage):.6f} million m^3")
    print(f"peak release: {np.max(release):.6f} million m^3/day")
    print(
        "maximum cumulative water-balance residual: "
        f"{maximum_balance_error:.3e} million m^3"
    )
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot storage, flow rates, and cumulative inflow and release volumes."""
    configure_matplotlib()
    time, storage, release = _dense_solution(solution)
    inflow_values = _inflow_array(time)
    cumulative_inflow = cumulative_trapezoid(inflow_values, time, initial=0.0)
    cumulative_release = cumulative_trapezoid(release, time, initial=0.0)

    fig, axes = plt.subplots(
        3, 1, figsize=(8.2, 7.8), sharex=True, layout="constrained"
    )
    axes[0].axhspan(
        MIN_STORAGE,
        MAX_STORAGE,
        color=COLORS["green"],
        alpha=0.10,
        label="Operating range",
    )
    axes[0].plot(time, storage, color=COLORS["blue"], label="Storage")
    axes[0].set_ylabel("Storage [million m^3]")
    axes[0].legend()

    axes[1].plot(time, inflow_values, color=COLORS["orange"], label="Inflow")
    axes[1].plot(time, release, color=COLORS["blue"], label="Release")
    axes[1].axhline(
        MAX_RELEASE,
        color=COLORS["black"],
        linestyle="--",
        label="Release limit",
    )
    axes[1].set_ylabel("Flow [million m^3/day]")
    axes[1].legend(ncol=2)

    axes[2].plot(time, cumulative_inflow, color=COLORS["orange"], label="Inflow")
    axes[2].plot(time, cumulative_release, color=COLORS["blue"], label="Release")
    axes[2].set_ylabel("Cumulative volume [million m^3]")
    axes[2].set_xlabel("Time [days]")
    axes[2].legend(ncol=2)

    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "reservoir_flood_control_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
