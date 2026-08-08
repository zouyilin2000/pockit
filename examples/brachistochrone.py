"""Solve the classical minimum-time brachistochrone problem.

A bead starts from rest and slides without friction under uniform gravity.
Horizontal displacement is positive to the right, vertical displacement is
positive downward, and the path angle is measured from the downward vertical.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.optimize import brentq

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


GRAVITY = 9.81
TARGET_X = 2.0
TARGET_Y = 2.0


def build_problem():
    """Build the frictionless brachistochrone optimal control problem."""
    system = System(0)
    phase = system.new_phase(["x", "y", "speed"], ["path_angle"])
    _, _, speed = phase.x
    (path_angle,) = phase.u

    phase.set_dynamics(
        [
            speed * sp.sin(path_angle),
            speed * sp.cos(path_angle),
            GRAVITY * sp.cos(path_angle),
        ]
    )
    phase.set_integral([1.0])
    phase.set_phase_constraint([speed, path_angle], [0.0, 0.0], [np.inf, np.pi / 2.0])
    phase.set_boundary_condition([0.0, 0.0, 0.0], [TARGET_X, TARGET_Y, None], 0.0, None)
    phase.set_discretization(10, 8)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct a descending straight-line guess with energy-consistent speed."""
    guess = linear_guess(phase, 0.0)
    guess.t_f = 1.0
    tau = guess.t_x / guess.t_f
    guess.x[0] = TARGET_X * tau
    guess.x[1] = TARGET_Y * tau
    guess.x[2] = np.sqrt(2.0 * GRAVITY * guess.x[1])
    guess.u[0] = np.arctan2(TARGET_X, TARGET_Y)
    return guess


def solve_problem(system, guess):
    """Solve the problem and compare its travel time with the cycloid solution."""
    solution, info = ipopt.solve(
        system, guess, optimizer_options={"print_level": 0, "sb": "yes"}
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
        final_time=solution.t_f,
        objective=info["obj_val"],
    )

    terminal_parameter = brentq(
        lambda value: (
            (value - np.sin(value)) / (1.0 - np.cos(value)) - TARGET_X / TARGET_Y
        ),
        1.0e-6,
        2.0 * np.pi - 1.0e-6,
    )
    cycloid_scale = TARGET_Y / (1.0 - np.cos(terminal_parameter))
    analytical_time = np.sqrt(cycloid_scale / GRAVITY) * terminal_parameter
    require_finite(
        terminal_parameter=terminal_parameter,
        cycloid_scale=cycloid_scale,
        analytical_time=analytical_time,
    )
    np.testing.assert_allclose(solution.t_f, analytical_time, rtol=2.0e-4, atol=2.0e-6)

    print(f"Ipopt status: {status_message}")
    print(f"Minimum time: {solution.t_f:.12f} s")
    print(f"Cycloid reference: {analytical_time:.12f} s")
    return solution


def plot_solution(solution, *, save=None, show=True):
    """Plot the optimal curve, speed, and path angle."""
    configure_matplotlib()
    fig = plt.figure(figsize=(9.0, 4.8))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1.0))
    trajectory_axis = fig.add_subplot(grid[:, 0])
    speed_axis = fig.add_subplot(grid[0, 1])
    angle_axis = fig.add_subplot(grid[1, 1], sharex=speed_axis)

    terminal_parameter = brentq(
        lambda value: (
            (value - np.sin(value)) / (1.0 - np.cos(value)) - TARGET_X / TARGET_Y
        ),
        1.0e-6,
        2.0 * np.pi - 1.0e-6,
    )
    cycloid_scale = TARGET_Y / (1.0 - np.cos(terminal_parameter))
    parameter = np.linspace(0.0, terminal_parameter, 400)
    cycloid_x = cycloid_scale * (parameter - np.sin(parameter))
    cycloid_y = cycloid_scale * (1.0 - np.cos(parameter))

    trajectory_axis.plot(
        cycloid_x,
        cycloid_y,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.4,
        label="Analytical cycloid",
    )
    trajectory_axis.plot(
        solution.x[0],
        solution.x[1],
        color=COLORS["blue"],
        label="Pockit",
    )
    trajectory_axis.scatter(
        [0.0, TARGET_X],
        [0.0, TARGET_Y],
        color=[COLORS["green"], COLORS["vermillion"]],
        zorder=3,
    )
    trajectory_axis.set_xlabel(r"Horizontal displacement $x$ [m]")
    trajectory_axis.set_ylabel(r"Downward displacement $y$ [m]")
    trajectory_axis.set_title("Minimum-time path")
    trajectory_axis.set_aspect("equal", adjustable="box")
    trajectory_axis.invert_yaxis()
    trajectory_axis.legend()

    speed_axis.plot(
        solution.t_x,
        solution.x[2],
        color=COLORS["green"],
        label="Speed",
    )
    speed_axis.set_ylabel(r"Speed $v$ [m/s]")
    speed_axis.set_title("State and control histories")
    speed_axis.legend()

    angle_axis.plot(
        solution.t_u,
        np.rad2deg(solution.u[0]),
        color=COLORS["orange"],
        label="Path angle",
    )
    angle_axis.set_xlabel(r"Time $t$ [s]")
    angle_axis.set_ylabel(r"Angle $\theta$ [deg]")
    angle_axis.legend()

    style_axes([trajectory_axis, speed_axis, angle_axis])
    fig.tight_layout()
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(__doc__.splitlines()[0], "brachistochrone_solution.png")
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
