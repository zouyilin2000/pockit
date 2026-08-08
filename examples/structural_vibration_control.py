"""Suppress earthquake-driven structural vibration with an active actuator.

A single-mode building model is excited by a known synthetic ground-acceleration
record. The optimized open-loop force trades modal displacement against actuator
effort while returning the structure to rest after the shaking. This idealized
feedforward study omits actuator dynamics and does not represent a causal feedback
controller. The trajectory is independently forward-integrated and compared with
the passive response.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, trapezoid

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


HORIZON = 20.0  # s
MODAL_MASS = 250_000.0  # kg
NATURAL_FREQUENCY = 2.0 * np.pi * 1.2  # rad/s
DAMPING_RATIO = 0.03
MAX_ACTUATOR_FORCE = 1_000_000.0  # N
MAX_DISPLACEMENT = 0.08  # m
MAX_RELATIVE_SPEED = 1.0  # m/s
GRAVITY = 9.80665  # m/s^2
DENSE_CHECK_POINTS = 4_001


def ground_acceleration(time):
    """Return the analytic ground-acceleration record in m/s^2."""
    primary = (
        0.35
        * GRAVITY
        * sp.exp(-(((time - 5.0) / 2.2) ** 2))
        * sp.sin(2.0 * sp.pi * 1.1 * time)
    )
    aftershock = (
        0.12
        * GRAVITY
        * sp.exp(-(((time - 9.0) / 3.0) ** 2))
        * sp.sin(2.0 * sp.pi * 2.4 * time + 0.4)
    )
    return primary + aftershock


def _ground_acceleration_array(time: np.ndarray) -> np.ndarray:
    primary = (
        0.35
        * GRAVITY
        * np.exp(-(((time - 5.0) / 2.2) ** 2))
        * np.sin(2.0 * np.pi * 1.1 * time)
    )
    aftershock = (
        0.12
        * GRAVITY
        * np.exp(-(((time - 9.0) / 3.0) ** 2))
        * np.sin(2.0 * np.pi * 2.4 * time + 0.4)
    )
    return primary + aftershock


def build_problem():
    """Return the active structural-vibration control problem."""
    system = System(0)
    phase = system.new_phase(
        ["relative_displacement", "relative_velocity"],
        ["normalized_actuator_force"],
    )
    displacement, velocity = phase.x
    (normalized_force,) = phase.u

    acceleration = (
        -2.0 * DAMPING_RATIO * NATURAL_FREQUENCY * velocity
        - NATURAL_FREQUENCY**2 * displacement
        + MAX_ACTUATOR_FORCE / MODAL_MASS * normalized_force
        - ground_acceleration(phase.t)
    )
    phase.set_dynamics([velocity, acceleration])
    phase.set_integral(
        [
            (displacement / MAX_DISPLACEMENT) ** 2
            + 0.05 * (velocity / (NATURAL_FREQUENCY * MAX_DISPLACEMENT)) ** 2
            + 0.02 * normalized_force**2
        ]
    )
    phase.set_phase_constraint(
        [displacement, velocity, normalized_force],
        [-MAX_DISPLACEMENT, -MAX_RELATIVE_SPEED, -1.0],
        [MAX_DISPLACEMENT, MAX_RELATIVE_SPEED, 1.0],
    )
    phase.set_boundary_condition([0.0, 0.0], [0.0, 0.0], 0.0, HORIZON)
    phase.set_discretization(160, 4)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Return a feasible at-rest guess that cancels ground acceleration."""
    guess = linear_guess(phase, 0.0)
    guess.u[0] = MODAL_MASS / MAX_ACTUATOR_FORCE * _ground_acceleration_array(guess.t_u)
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    states = np.vstack([solution.V_x(time) @ component for component in solution.x])
    normalized_force = solution.V_u(time) @ solution.u[0]
    return time, states, normalized_force


def _forward_response(time, normalized_force):
    """Integrate the physical oscillator for a supplied sampled control."""

    def dynamics(current_time, state):
        force = np.interp(current_time, time, normalized_force)
        acceleration = (
            -2.0 * DAMPING_RATIO * NATURAL_FREQUENCY * state[1]
            - NATURAL_FREQUENCY**2 * state[0]
            + MAX_ACTUATOR_FORCE / MODAL_MASS * force
            - _ground_acceleration_array(np.array([current_time]))[0]
        )
        return state[1], acceleration

    result = solve_ivp(
        dynamics,
        (time[0], time[-1]),
        (0.0, 0.0),
        t_eval=time,
        rtol=2e-10,
        atol=2e-12,
        method="DOP853",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y


def solve_problem(system, guess):
    """Optimize the actuator history and verify it by forward integration."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1500, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, states, normalized_force = _dense_solution(solution)
    reintegrated = _forward_response(time, normalized_force)
    passive = _forward_response(time, np.zeros_like(time))
    require_finite(
        time=time,
        states=states,
        normalized_force=normalized_force,
        reintegrated_states=reintegrated,
        passive_states=passive,
        objective=info["obj_val"],
    )
    reintegration_error = np.max(np.abs(reintegrated - states), axis=1)
    path_violation = max(
        float(np.max(np.abs(states[0]) - MAX_DISPLACEMENT)),
        float(np.max(np.abs(states[1]) - MAX_RELATIVE_SPEED)),
        float(np.max(np.abs(normalized_force) - 1.0)),
        0.0,
    )
    terminal_error = float(np.max(np.abs(states[:, -1])))
    controlled_peak = float(np.max(np.abs(states[0])))
    passive_peak = float(np.max(np.abs(passive[0])))
    if reintegration_error[0] > 1e-4 or reintegration_error[1] > 1e-3:
        raise RuntimeError("the collocation trajectory failed forward integration")
    if path_violation > 2e-6 or terminal_error > 2e-6:
        raise RuntimeError("the dense structural trajectory is infeasible")
    if controlled_peak >= 0.8 * passive_peak:
        raise RuntimeError("active control did not materially reduce peak displacement")

    actuator_energy = trapezoid((MAX_ACTUATOR_FORCE * normalized_force) ** 2, time)
    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"passive peak displacement: {1e3 * passive_peak:.4f} mm")
    print(f"controlled peak displacement: {1e3 * controlled_peak:.4f} mm")
    print(
        f"peak actuator force: {np.max(np.abs(normalized_force)) * 1e-3 * MAX_ACTUATOR_FORCE:.3f} kN"
    )
    print(f"integral squared actuator force: {actuator_energy:.6e} N^2 s")
    print(
        "maximum forward-integration error: "
        f"{reintegration_error[0]:.3e} m, {reintegration_error[1]:.3e} m/s"
    )
    print(f"maximum dense path-bound violation: {path_violation:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot excitation, passive/controlled motion, and actuator demand."""
    configure_matplotlib()
    time, states, normalized_force = _dense_solution(solution)
    passive = _forward_response(time, np.zeros_like(time))
    acceleration = _ground_acceleration_array(time)

    fig, axes = plt.subplots(
        4, 1, figsize=(8.2, 9.2), sharex=True, layout="constrained"
    )
    axes[0].plot(time, acceleration / GRAVITY, color=COLORS["orange"])
    axes[0].fill_between(
        time, 0.0, acceleration / GRAVITY, color=COLORS["orange"], alpha=0.12
    )
    axes[0].set_ylabel("Ground acceleration [g]")

    axes[1].plot(time, 1e3 * passive[0], color=COLORS["black"], alpha=0.72)
    axes[1].axhline(1e3 * MAX_DISPLACEMENT, color=COLORS["vermillion"], linestyle="--")
    axes[1].axhline(-1e3 * MAX_DISPLACEMENT, color=COLORS["vermillion"], linestyle="--")
    axes[1].set_ylabel("Passive displacement [mm]")

    controlled_displacement = 1e3 * states[0]
    controlled_limit = max(1.25 * float(np.max(np.abs(controlled_displacement))), 1.0)
    axes[2].plot(time, controlled_displacement, color=COLORS["blue"])
    axes[2].axhline(0.0, color=COLORS["black"], linewidth=0.9)
    axes[2].set_ylim(-controlled_limit, controlled_limit)
    axes[2].set_ylabel("Controlled displacement [mm]")

    axes[3].plot(
        time,
        1e-3 * MAX_ACTUATOR_FORCE * normalized_force,
        color=COLORS["purple"],
    )
    axes[3].set_ylabel("Actuator force [kN]")
    axes[3].set_xlabel("Time [s]")

    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "structural_vibration_control_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
