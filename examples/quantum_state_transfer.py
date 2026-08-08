"""Transfer a resonantly driven two-level quantum state on the Bloch sphere.

The bounded Rabi controls rotate the Bloch vector from the north pole to the
positive x-axis in fixed time while minimizing normalized Rabi-control fluence,
not a calibrated energy in joules. The rotating-frame model assumes exact
resonance, the rotating-wave approximation, zero detuning, and no decoherence.
Under those assumptions it has a known geodesic validation target.
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


HORIZON = 2.0
MAX_RABI_RATE = 2.0  # rad / unit time
EXPECTED_RABI_RATE = np.pi / (2.0 * HORIZON)
EXPECTED_OBJECTIVE = np.pi**2 / (4.0 * HORIZON)
DENSE_CHECK_POINTS = 4_001


def build_problem():
    """Return the minimum-fluence Bloch-equation control problem."""
    system = System(0)
    phase = system.new_phase(["bloch_x", "bloch_y", "bloch_z"], ["rabi_x", "rabi_y"])
    bloch_x, bloch_y, bloch_z = phase.x
    rabi_x, rabi_y = phase.u

    phase.set_dynamics(
        [
            rabi_y * bloch_z,
            -rabi_x * bloch_z,
            rabi_x * bloch_y - rabi_y * bloch_x,
        ]
    )
    phase.set_integral([rabi_x**2 + rabi_y**2])
    phase.set_phase_constraint(
        [rabi_x, rabi_y],
        [-MAX_RABI_RATE, -MAX_RABI_RATE],
        [MAX_RABI_RATE, MAX_RABI_RATE],
    )
    phase.set_boundary_condition([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 0.0, HORIZON)
    phase.set_discretization(24, 4)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Return the exact resonant great-circle rotation as an initial guess."""
    guess = linear_guess(phase, 0.0)
    angle_x = 0.5 * np.pi * guess.t_x / HORIZON
    guess.x[0] = np.sin(angle_x)
    guess.x[1] = np.zeros_like(guess.t_x)
    guess.x[2] = np.cos(angle_x)
    guess.u[0] = np.zeros_like(guess.t_u)
    guess.u[1] = np.full_like(guess.t_u, EXPECTED_RABI_RATE)
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    bloch = np.vstack([solution.V_x(time) @ component for component in solution.x])
    controls = np.vstack([solution.V_u(time) @ component for component in solution.u])
    return time, bloch, controls


def solve_problem(system, guess):
    """Solve the quantum transfer and compare it with the analytical pulse."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-10, "max_iter": 1000, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, bloch, controls = _dense_solution(solution)
    angle = 0.5 * np.pi * time / HORIZON
    exact_bloch = np.vstack([np.sin(angle), np.zeros_like(angle), np.cos(angle)])
    require_finite(
        time=time,
        bloch=bloch,
        controls=controls,
        exact_bloch=exact_bloch,
        objective=info["obj_val"],
    )
    norm_error = float(np.max(np.abs(np.sum(bloch**2, axis=0) - 1.0)))
    control_violation = max(float(np.max(np.abs(controls) - MAX_RABI_RATE)), 0.0)
    objective_error = abs(float(info["obj_val"] - EXPECTED_OBJECTIVE))
    pulse_area = trapezoid(controls[1], time)
    terminal_fidelity = 0.5 * (1.0 + bloch[0, -1])
    trajectory_error = float(np.max(np.abs(bloch - exact_bloch)))
    require_finite(
        norm_error=norm_error,
        control_violation=control_violation,
        objective_error=objective_error,
        pulse_area=pulse_area,
        terminal_fidelity=terminal_fidelity,
        trajectory_error=trajectory_error,
    )
    if norm_error > 2e-6 or control_violation > 2e-7:
        raise RuntimeError("the dense Bloch trajectory failed validation")
    if objective_error > 2e-5:
        raise RuntimeError("pulse energy does not match the analytical optimum")
    if (
        abs(float(pulse_area - 0.5 * np.pi)) > 2e-6
        or 1.0 - terminal_fidelity > 2e-8
        or trajectory_error > 2e-6
    ):
        raise RuntimeError("the pulse sign or Bloch trajectory is incorrect")

    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.10f}")
    print(f"analytical objective: {EXPECTED_OBJECTIVE:.10f}")
    print(f"y-axis pulse area: {pulse_area:.10f} rad")
    print(f"terminal state fidelity: {terminal_fidelity:.10f}")
    print(f"maximum analytical trajectory error: {trajectory_error:.3e}")
    print(f"maximum Bloch-norm error: {norm_error:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot Bloch coordinates, Rabi controls, and the Bloch-sphere path."""
    configure_matplotlib()
    time, bloch, controls = _dense_solution(solution)

    fig = plt.figure(figsize=(10.0, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])
    state_axis = fig.add_subplot(grid[0, 0])
    control_axis = fig.add_subplot(grid[1, 0], sharex=state_axis)
    sphere_axis = fig.add_subplot(grid[:, 1], projection="3d")

    state_axis.plot(time, bloch[0], color=COLORS["blue"], label=r"$r_x$")
    state_axis.plot(time, bloch[1], color=COLORS["orange"], label=r"$r_y$")
    state_axis.plot(time, bloch[2], color=COLORS["green"], label=r"$r_z$")
    state_axis.set_ylabel("Bloch coordinate")
    state_axis.legend(ncol=3)

    control_axis.plot(time, controls[0], color=COLORS["blue"], label=r"$\Omega_x$")
    control_axis.plot(time, controls[1], color=COLORS["purple"], label=r"$\Omega_y$")
    control_axis.set_ylabel("Rabi rate [rad/time]")
    control_axis.set_xlabel("Time")
    control_axis.legend(ncol=2)

    azimuth = np.linspace(0.0, 2.0 * np.pi, 48)
    polar = np.linspace(0.0, np.pi, 24)
    sphere_x = np.outer(np.cos(azimuth), np.sin(polar))
    sphere_y = np.outer(np.sin(azimuth), np.sin(polar))
    sphere_z = np.outer(np.ones_like(azimuth), np.cos(polar))
    sphere_axis.plot_wireframe(
        sphere_x,
        sphere_y,
        sphere_z,
        color="#B8B8B8",
        linewidth=0.45,
        alpha=0.35,
        rstride=4,
        cstride=4,
    )
    sphere_axis.plot(
        bloch[0], bloch[1], bloch[2], color=COLORS["vermillion"], linewidth=2.5
    )
    sphere_axis.scatter(*bloch[:, 0], color=COLORS["green"], s=36, label="Initial")
    sphere_axis.scatter(*bloch[:, -1], color=COLORS["blue"], s=36, label="Target")
    sphere_axis.set(
        xlabel=r"$r_x$", ylabel=r"$r_y$", zlabel=r"$r_z$", box_aspect=(1, 1, 1)
    )
    sphere_axis.set_title("Bloch-sphere trajectory")
    sphere_axis.legend(loc="upper left")

    style_axes([state_axis, control_axis])
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "quantum_state_transfer_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
