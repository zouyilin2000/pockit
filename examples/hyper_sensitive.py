"""Solve a long-horizon hyper-sensitive optimal control problem.

The nondimensional scalar system has narrow boundary layers near both ends of
a 10,000-unit model-time horizon and an almost steady interior arc. This makes
it a useful test of a transcription's ability to resolve widely separated
time scales.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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


INITIAL_STATE = 1.5
TERMINAL_STATE = 1.0
HORIZON = 10_000.0
BOUNDARY_WINDOW = 20.0
MAX_MESH_UPDATES = 20


def build_problem(quick=False):
    """Build the standard long-horizon hyper-sensitive control problem."""
    system = System(0)
    phase = system.new_phase(["state"], ["control"])
    (state,) = phase.x
    (control,) = phase.u

    phase.set_dynamics([-(state**3) + control])
    phase.set_integral([(state**2 + control**2) / 2.0])
    phase.set_boundary_condition([INITIAL_STATE], [TERMINAL_STATE], 0.0, HORIZON)
    if quick:
        phase.set_discretization(6, 8)
    else:
        phase.set_discretization(10, 10)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct a slowly varying state guess that approximately obeys dynamics."""
    guess = linear_guess(phase, 0.0)
    state_at_control_nodes = np.interp(guess.t_u, guess.t_x, guess.x[0])
    state_slope = (TERMINAL_STATE - INITIAL_STATE) / HORIZON
    guess.u[0] = state_at_control_nodes**3 + state_slope
    return guess


def solve_problem(system, guess, quick=False):
    """Solve the problem, updating the mesh only while its error requires it."""
    optimizer_options = {"print_level": 0, "sb": "yes"}
    solution, info = ipopt.solve(system, guess, optimizer_options=optimizer_options)
    status = int(info["status"])
    status_message = info["status_msg"]
    if isinstance(status_message, bytes):
        status_message = status_message.decode()
    if status not in (0, 1):
        raise RuntimeError(f"Ipopt failed ({status}): {status_message}")
    require_finite(
        state_time=solution.t_x,
        control_time=solution.t_u,
        states=np.vstack(solution.x),
        controls=np.vstack(solution.u),
        objective=info["obj_val"],
    )

    tolerance = 1.0e-3 if quick else 1.0e-8
    max_mesh_updates = 3 if quick else MAX_MESH_UPDATES
    mesh_updates = 0
    for mesh_updates in range(max_mesh_updates + 1):
        if system.check(
            solution,
            absolute_tolerance_continuous=tolerance,
            relative_tolerance_continuous=tolerance,
        ):
            break
        if mesh_updates == max_mesh_updates:
            if quick:
                print("Quick mode stopped before the default mesh-error target.")
                break
            raise RuntimeError("Mesh error tolerance was not reached")
        solution = system.refine(
            solution,
            absolute_tolerance_continuous=tolerance,
            relative_tolerance_continuous=tolerance,
            num_point_min=6 if quick else 10,
            num_point_max=12 if quick else 20,
            mesh_length_min=1.0e-6 if quick else 1.0e-8,
        )
        solution, info = ipopt.solve(
            system, solution, optimizer_options=optimizer_options
        )
        status = int(info["status"])
        status_message = info["status_msg"]
        if isinstance(status_message, bytes):
            status_message = status_message.decode()
        if status not in (0, 1):
            raise RuntimeError(f"Ipopt failed ({status}): {status_message}")
        require_finite(
            state_time=solution.t_x,
            control_time=solution.t_u,
            states=np.vstack(solution.x),
            controls=np.vstack(solution.u),
            objective=info["obj_val"],
        )

    np.testing.assert_allclose(
        [solution.x[0][0], solution.x[0][-1]],
        [INITIAL_STATE, TERMINAL_STATE],
        rtol=0.0,
        atol=1.0e-7,
    )

    print(f"Ipopt status: {status_message}")
    print(f"Objective: {float(info['obj_val']):.12f}")
    print(f"Mesh updates: {mesh_updates}")
    return solution


def plot_solution(solution, *, save=None, show=True):
    """Plot the full horizon and detailed views of both boundary layers."""
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))

    axes[0, 0].plot(
        solution.t_x,
        solution.x[0],
        color=COLORS["blue"],
        label="State",
    )
    axes[0, 0].set_xlabel(r"Model time $t$")
    axes[0, 0].set_ylabel(r"State $x$ [-]")
    axes[0, 0].set_title("State over the full horizon")
    axes[0, 0].legend()

    axes[0, 1].plot(
        solution.t_u,
        solution.u[0],
        color=COLORS["orange"],
        label="Control",
    )
    axes[0, 1].set_xlabel(r"Model time $t$")
    axes[0, 1].set_ylabel(r"Control $u$ [-]")
    axes[0, 1].set_title("Control over the full horizon")
    axes[0, 1].legend()

    initial_time = np.linspace(0.0, BOUNDARY_WINDOW, 400)
    initial_state = solution.V_x(initial_time) @ solution.x[0]
    initial_control = solution.V_u(initial_time) @ solution.u[0]
    axes[1, 0].plot(initial_time, initial_state, color=COLORS["blue"], label="State")
    axes[1, 0].plot(
        initial_time,
        initial_control,
        color=COLORS["orange"],
        label="Control",
    )
    axes[1, 0].set_xlabel(r"Model time $t$")
    axes[1, 0].set_ylabel("Value [-]")
    axes[1, 0].set_title("Initial boundary layer")
    axes[1, 0].legend()

    terminal_time = np.linspace(HORIZON - BOUNDARY_WINDOW, HORIZON, 400)
    terminal_state = solution.V_x(terminal_time) @ solution.x[0]
    terminal_control = solution.V_u(terminal_time) @ solution.u[0]
    axes[1, 1].plot(terminal_time, terminal_state, color=COLORS["blue"], label="State")
    axes[1, 1].plot(
        terminal_time,
        terminal_control,
        color=COLORS["orange"],
        label="Control",
    )
    axes[1, 1].set_xlabel(r"Model time $t$")
    axes[1, 1].set_ylabel("Value [-]")
    axes[1, 1].set_title("Terminal boundary layer")
    axes[1, 1].legend()

    style_axes(axes)
    fig.tight_layout()
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "hyper_sensitive_solution.png", quick=True
    )
    system, phase = build_problem(quick=args.quick)
    guess = initial_guess(phase)
    solution = solve_problem(system, guess, quick=args.quick)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
