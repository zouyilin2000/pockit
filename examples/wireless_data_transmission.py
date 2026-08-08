"""Minimize transmit energy over a deterministic time-varying wireless link.

A transmitter must deliver a fixed data payload before a deadline. Channel
quality improves and then fades as a receiver passes the access point, so the
energy-optimal power schedule follows the classical water-filling condition.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import cumulative_trapezoid, trapezoid
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


HORIZON = 10.0  # s
PAYLOAD = 18.0  # Mbit
BANDWIDTH = 1.0  # MHz, so B*log2(1+SNR) is in Mbit/s
MAX_POWER = 5.0  # W
DENSE_CHECK_POINTS = 4_001


def channel_quality(time):
    """Return received SNR per watt for the deterministic channel."""
    return 0.65 + 3.2 * sp.exp(-(((time - 6.0) / 1.8) ** 2))


def _channel_quality_array(time: np.ndarray) -> np.ndarray:
    return 0.65 + 3.2 * np.exp(-(((time - 6.0) / 1.8) ** 2))


def build_problem():
    """Return the deadline-constrained wireless-transmission problem."""
    system = System(0)
    phase = system.new_phase(["remaining_data"], ["transmit_power"])
    (remaining_data,) = phase.x
    (power,) = phase.u

    rate = BANDWIDTH * sp.log(1.0 + channel_quality(phase.t) * power) / sp.log(2.0)
    phase.set_dynamics([-rate])
    phase.set_integral([power])
    phase.set_phase_constraint(
        [remaining_data, power], [0.0, 0.0], [PAYLOAD, MAX_POWER]
    )
    phase.set_boundary_condition([PAYLOAD], [0.0], 0.0, HORIZON)
    # Linear interpolants preserve the data-queue and power bounds between nodes.
    phase.set_discretization(100, 2)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Return a constant-power, linearly decreasing payload guess."""
    guess = linear_guess(phase, 2.0)
    guess.x[0] = PAYLOAD * (1.0 - guess.t_x / HORIZON)
    guess.u[0] = np.full_like(guess.t_u, 2.0)
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    remaining = solution.V_x(time) @ solution.x[0]
    power = solution.V_u(time) @ solution.u[0]
    return time, remaining, power


def _water_filling_reference(time: np.ndarray, quality: np.ndarray):
    """Return the continuous water level and its clipped power schedule."""

    def power_for_level(water_level: float) -> np.ndarray:
        return np.clip(water_level - 1.0 / quality, 0.0, MAX_POWER)

    def payload_residual(water_level: float) -> float:
        reference_power = power_for_level(water_level)
        reference_rate = BANDWIDTH * np.log2(1.0 + quality * reference_power)
        return float(trapezoid(reference_rate, time) - PAYLOAD)

    lower_level = float(np.min(1.0 / quality))
    upper_level = float(MAX_POWER + np.max(1.0 / quality))
    if payload_residual(upper_level) < 0.0:
        raise RuntimeError("the requested payload exceeds the channel capacity")
    water_level = brentq(
        payload_residual,
        lower_level,
        upper_level,
        xtol=1e-13,
        rtol=1e-13,
    )
    return water_level, power_for_level(water_level)


def solve_problem(system, guess):
    """Solve the transmission problem and verify water filling and delivery."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1200, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, remaining, power = _dense_solution(solution)
    quality = _channel_quality_array(time)
    rate = BANDWIDTH * np.log2(1.0 + quality * power)
    cumulative_delivered = cumulative_trapezoid(rate, time, initial=0.0)
    require_finite(
        time=time,
        remaining_data=remaining,
        transmit_power=power,
        channel_quality=quality,
        transmission_rate=rate,
        cumulative_delivered=cumulative_delivered,
        objective=info["obj_val"],
    )
    delivered_by_quadrature = float(cumulative_delivered[-1])
    delivery_error = abs(delivered_by_quadrature - PAYLOAD)
    queue_balance_error = float(
        np.max(np.abs(remaining - (PAYLOAD - cumulative_delivered)))
    )
    path_violation = max(
        0.0,
        float(np.max(-remaining)),
        float(np.max(remaining - PAYLOAD)),
        float(np.max(-power)),
        float(np.max(power - MAX_POWER)),
    )
    water_level, reference_power = _water_filling_reference(time, quality)
    reference_rate = BANDWIDTH * np.log2(1.0 + quality * reference_power)
    require_finite(
        water_level=water_level,
        reference_power=reference_power,
        reference_rate=reference_rate,
    )
    reference_payload_error = abs(float(trapezoid(reference_rate, time) - PAYLOAD))
    power_error = float(np.max(np.abs(power - reference_power)))
    energy = trapezoid(power, time)
    reference_energy = trapezoid(reference_power, time)
    energy_error = abs(float(energy - reference_energy))
    if path_violation > 2e-6 or delivery_error > 3e-3 or queue_balance_error > 3e-3:
        raise RuntimeError("the dense transmission schedule failed validation")
    if reference_payload_error > 1e-8 or power_error > 2e-3 or energy_error > 2e-3:
        raise RuntimeError("the power schedule does not match clipped water filling")

    print(f"status: {status_message}")
    print(f"transmit energy: {energy:.8f} J")
    print(f"delivered payload: {delivered_by_quadrature:.8f} Mbit")
    print(f"analytical water level: {water_level:.8f} W")
    print(f"maximum analytical power error: {power_error:.3e} W")
    print(f"maximum cumulative queue-balance error: {queue_balance_error:.3e} Mbit")
    print(f"maximum dense path-bound violation: {path_violation:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot the remaining payload, transmit power, and channel quality."""
    configure_matplotlib()
    time, remaining, power = _dense_solution(solution)
    quality = _channel_quality_array(time)

    fig, axes = plt.subplots(
        3, 1, figsize=(8.2, 7.6), sharex=True, layout="constrained"
    )
    axes[0].plot(time, remaining, color=COLORS["blue"])
    axes[0].set_ylabel("Remaining data [Mbit]")

    axes[1].plot(time, power, color=COLORS["vermillion"])
    axes[1].set_ylabel("Transmit power [W]")
    axes[1].set_ylim(-0.1, MAX_POWER + 0.25)

    axes[2].plot(time, quality, color=COLORS["green"])
    axes[2].fill_between(time, 0.0, quality, color=COLORS["green"], alpha=0.12)
    axes[2].set_ylabel("SNR per watt [1/W]")
    axes[2].set_xlabel("Time [s]")

    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "wireless_data_transmission_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
