"""Find a minimum-energy stimulus for a FitzHugh-Nagumo neuron.

The FitzHugh-Nagumo equations retain the fast excitation and slow recovery
mechanisms of neuronal spiking while remaining two-dimensional. Here a bounded
injected current drives a resting neuron through a prescribed voltage on its
rising trajectory at a fixed deadline. This endpoint crossing is not a steady
depolarized state or a validation of the complete post-deadline action
potential. An independent forward integration checks the original nonlinear
dynamics.

All quantities are nondimensional. The ideal one-sided current omits electrode
dynamics and charge-balancing requirements, so it is not a clinical stimulation
waveform.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


HORIZON = 12.0  # nondimensional time
RECOVERY_OFFSET = 0.7
RECOVERY_GAIN = 0.8
TIME_SCALE_RATIO = 0.08
RESTING_VOLTAGE = -1.1994080352440348
RESTING_RECOVERY = (RESTING_VOLTAGE + RECOVERY_OFFSET) / RECOVERY_GAIN
TARGET_VOLTAGE = 1.0
MIN_VOLTAGE = -2.5
MAX_VOLTAGE = 2.5
MIN_RECOVERY = -1.0
MAX_RECOVERY = 2.0
MAX_CURRENT = 2.0
DENSE_CHECK_POINTS = 4_001


def build_problem():
    """Return the minimum-energy neuronal-stimulation problem."""
    system = System(0)
    phase = system.new_phase(
        ["membrane_voltage", "recovery_variable"], ["injected_current"]
    )
    voltage, recovery = phase.x
    (current,) = phase.u

    phase.set_dynamics(
        [
            voltage - voltage**3 / 3.0 - recovery + current,
            TIME_SCALE_RATIO * (voltage + RECOVERY_OFFSET - RECOVERY_GAIN * recovery),
        ]
    )
    phase.set_integral([(current / MAX_CURRENT) ** 2])
    phase.set_phase_constraint(
        [voltage, recovery, current],
        [MIN_VOLTAGE, MIN_RECOVERY, 0.0],
        [MAX_VOLTAGE, MAX_RECOVERY, MAX_CURRENT],
    )
    phase.set_boundary_condition(
        [RESTING_VOLTAGE, RESTING_RECOVERY],
        [TARGET_VOLTAGE, None],
        0.0,
        HORIZON,
    )
    phase.set_discretization(80, 4)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct a smooth depolarization guess with moderate current."""
    guess = linear_guess(phase, 0.35)
    tau = guess.t_x / HORIZON
    progress = 3.0 * tau**2 - 2.0 * tau**3
    guess.x[0] = RESTING_VOLTAGE + (TARGET_VOLTAGE - RESTING_VOLTAGE) * progress
    guess.x[1] = RESTING_RECOVERY + (0.2 - RESTING_RECOVERY) * progress
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    states = np.vstack([solution.V_x(time) @ component for component in solution.x])
    current = solution.V_u(time) @ solution.u[0]
    return time, states, current


def _forward_response(time, current):
    """Integrate the FitzHugh-Nagumo equations for a sampled current."""

    def dynamics(current_time, state):
        stimulus = np.interp(current_time, time, current)
        return (
            state[0] - state[0] ** 3 / 3.0 - state[1] + stimulus,
            TIME_SCALE_RATIO * (state[0] + RECOVERY_OFFSET - RECOVERY_GAIN * state[1]),
        )

    result = solve_ivp(
        dynamics,
        (time[0], time[-1]),
        (RESTING_VOLTAGE, RESTING_RECOVERY),
        t_eval=time,
        rtol=2e-10,
        atol=2e-12,
        method="DOP853",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y


def solve_problem(system, guess):
    """Optimize the stimulus and verify the nonlinear forward trajectory."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1800, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, states, current = _dense_solution(solution)
    reintegrated = _forward_response(time, current)
    require_finite(
        time=time,
        states=states,
        current=current,
        reintegrated_states=reintegrated,
        objective=info["obj_val"],
    )
    reintegration_error = float(np.max(np.abs(reintegrated - states)))
    terminal_error = abs(float(states[0, -1] - TARGET_VOLTAGE))
    forward_terminal_error = abs(float(reintegrated[0, -1] - TARGET_VOLTAGE))
    forward_terminal_voltage_rate = float(
        reintegrated[0, -1]
        - reintegrated[0, -1] ** 3 / 3.0
        - reintegrated[1, -1]
        + current[-1]
    )
    require_finite(forward_terminal_voltage_rate=forward_terminal_voltage_rate)
    path_violation = max(
        float(np.max(MIN_VOLTAGE - states[0])),
        float(np.max(states[0] - MAX_VOLTAGE)),
        float(np.max(MIN_RECOVERY - states[1])),
        float(np.max(states[1] - MAX_RECOVERY)),
        float(np.max(-current)),
        float(np.max(current - MAX_CURRENT)),
        0.0,
    )
    if reintegration_error > 2e-3 or forward_terminal_error > 2e-3:
        raise RuntimeError("the optimized stimulus failed forward integration")
    if forward_terminal_voltage_rate <= 0.0:
        raise RuntimeError("the terminal voltage target is not crossed while rising")
    if terminal_error > 2e-6 or path_violation > 2e-6:
        raise RuntimeError("the dense neuronal trajectory is infeasible")

    charge = trapezoid(current, time)
    stimulus_energy = trapezoid(current**2, time)
    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"injected charge: {charge:.8f} nondimensional")
    print(f"stimulus energy: {stimulus_energy:.8f} nondimensional")
    print(f"peak current: {np.max(current):.8f}")
    print(f"forward terminal-voltage error: {forward_terminal_error:.3e}")
    print(f"forward terminal voltage rate: {forward_terminal_voltage_rate:.6f}")
    print(f"maximum forward-integration error: {reintegration_error:.3e}")
    print(f"maximum dense path-bound violation: {path_violation:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot voltage, stimulus current, and the nonlinear phase portrait."""
    configure_matplotlib()
    time, states, current = _dense_solution(solution)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), layout="constrained")
    axes[0].plot(time, states[0], color=COLORS["blue"], label="Voltage")
    axes[0].axhline(
        TARGET_VOLTAGE,
        color=COLORS["vermillion"],
        linestyle="--",
        label="Terminal crossing target",
    )
    axes[0].set_xlabel("Time [nondimensional]")
    axes[0].set_ylabel("Membrane voltage")
    axes[0].legend()

    axes[1].plot(time, current, color=COLORS["purple"])
    axes[1].fill_between(time, 0.0, current, color=COLORS["purple"], alpha=0.12)
    axes[1].set_xlabel("Time [nondimensional]")
    axes[1].set_ylabel("Injected current")
    axes[1].set_ylim(-0.05, MAX_CURRENT + 0.1)

    voltage_grid = np.linspace(MIN_VOLTAGE, 2.0, 500)
    voltage_nullcline = voltage_grid - voltage_grid**3 / 3.0
    recovery_nullcline = (voltage_grid + RECOVERY_OFFSET) / RECOVERY_GAIN
    axes[2].plot(
        voltage_grid,
        voltage_nullcline,
        color=COLORS["orange"],
        linestyle="--",
        label=r"$\dot v=0$, $I=0$",
    )
    axes[2].plot(
        voltage_grid,
        recovery_nullcline,
        color=COLORS["green"],
        linestyle=":",
        label=r"$\dot w=0$",
    )
    axes[2].plot(states[0], states[1], color=COLORS["blue"], label="Stimulated path")
    axes[2].scatter(
        [RESTING_VOLTAGE, states[0, -1]],
        [RESTING_RECOVERY, states[1, -1]],
        color=[COLORS["black"], COLORS["vermillion"]],
        s=28,
        zorder=3,
    )
    axes[2].set_xlabel("Membrane voltage")
    axes[2].set_ylabel("Recovery variable")
    axes[2].set_xlim(MIN_VOLTAGE, 2.0)
    axes[2].set_ylim(MIN_RECOVERY, 1.5)
    axes[2].legend(fontsize=8)

    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "neural_stimulation_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
