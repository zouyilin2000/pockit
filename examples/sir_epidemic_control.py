"""Plan a bounded non-pharmaceutical intervention for an SIR epidemic.

The control ``u`` is the fractional reduction in potentially infectious
contacts. The problem minimizes infection burden and intervention effort
while enforcing a hard health-care capacity limit on the infected fraction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pockit.lobatto import System, linear_guess
from pockit.optimizer import ipopt

if __package__:
    from ._plotting import (
        configure_matplotlib,
        parse_plot_arguments,
        require_finite,
        save_or_show,
        style_axes,
    )
else:
    from _plotting import (
        configure_matplotlib,
        parse_plot_arguments,
        require_finite,
        save_or_show,
        style_axes,
    )

BETA = 0.32
GAMMA = 0.10
INTERVENTION_MAX = 0.80
CAPACITY = 0.08
HORIZON = 100.0
EFFORT_WEIGHT = 0.05
DENSE_CHECK_POINTS = 4_001


def build_problem():
    """Return the configured SIR optimal-control system and phase."""
    system = System(0)
    phase = system.new_phase(["susceptible", "infected", "removed"], ["intervention"])
    susceptible, infected, _ = phase.x
    (intervention,) = phase.u

    incidence = BETA * (1 - intervention) * susceptible * infected
    phase.set_dynamics([-incidence, incidence - GAMMA * infected, GAMMA * infected])
    phase.set_integral([infected**2 + EFFORT_WEIGHT * intervention**2])
    phase.set_phase_constraint(
        [intervention, infected],
        [0.0, 0.0],
        [INTERVENTION_MAX, CAPACITY],
    )
    phase.set_boundary_condition([0.99, 0.01, 0.0], [None, None, None], 0.0, HORIZON)
    # Piecewise-linear interpolation makes the nodal capacity and intervention
    # limits valid between mesh points as well.
    phase.set_discretization(80, 2)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct a smooth, feasible-scale initial guess."""
    guess = linear_guess(phase, 0.0)
    tau_x = guess.t_x / HORIZON
    guess.x[0] = 0.99 - 0.45 * tau_x
    guess.x[1] = 0.01 + 0.025 * np.sin(np.pi * tau_x) ** 2
    guess.x[2] = 1.0 - guess.x[0] - guess.x[1]
    guess.u[0] = np.full_like(guess.t_u, 0.45)
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def solve_problem(system, guess):
    """Solve the intervention-planning problem with Ipopt."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-8, "max_iter": 1000, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    population = np.vstack([solution.V_x(time) @ component for component in solution.x])
    intervention = solution.V_u(time) @ solution.u[0]
    require_finite(
        time=time,
        population=population,
        intervention=intervention,
        objective=info["obj_val"],
    )
    peak_infected = float(np.max(population[1]))
    path_violation = max(
        float(np.max(-population[1])),
        float(np.max(population[1] - CAPACITY)),
        float(np.max(-intervention)),
        float(np.max(intervention - INTERVENTION_MAX)),
        0.0,
    )
    conservation_error = float(np.max(np.abs(np.sum(population, axis=0) - 1.0)))
    if path_violation > 2e-6:
        raise RuntimeError("the computed trajectory violates the capacity limit")
    if conservation_error > 2e-6:
        raise RuntimeError(
            f"population conservation error is too large: {conservation_error:.3e}"
        )
    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"peak infected fraction: {peak_infected:.6f}")
    print(f"final removed fraction: {solution.x[2][-1]:.6f}")
    print(f"maximum dense path-bound violation: {path_violation:.3e}")
    print(f"maximum population conservation error: {conservation_error:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot population fractions and the optimized intervention."""
    configure_matplotlib()
    fig, axes = plt.subplots(
        3, 1, figsize=(8.0, 7.5), sharex=True, layout="constrained"
    )

    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    population = np.vstack([solution.V_x(time) @ component for component in solution.x])
    intervention = solution.V_u(time) @ solution.u[0]

    axes[0].plot(time, population[0], label="Susceptible", color="#0072B2")
    axes[0].plot(time, population[2], label="Removed", color="#009E73")
    axes[0].set_ylabel("Population fraction")
    axes[0].legend(ncol=2)

    axes[1].plot(time, population[1], label="Infected", color="#D55E00")
    axes[1].axhline(CAPACITY, color="#555555", linestyle="--", label="Capacity")
    axes[1].set_ylabel("Infected fraction")
    axes[1].set_ylim(-0.003, CAPACITY + 0.012)
    axes[1].legend(ncol=2)

    axes[2].plot(time, intervention, color="#CC79A7")
    axes[2].set_ylabel("Contact reduction")
    axes[2].set_xlabel("Time [days]")
    axes[2].set_ylim(-0.02, INTERVENTION_MAX + 0.05)
    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "sir_epidemic_control_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
