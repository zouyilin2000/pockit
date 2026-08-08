"""Compute an energy-aware speed profile across two road segments.

The urban and arterial segments have different speed limits and grades. Their
states meet at an optimized crossing time and speed, which makes the road
change a genuine multi-phase optimal-control problem.
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

DISTANCE_SWITCH = 1.0
DISTANCE_FINAL = 3.0
URBAN_SPEED_LIMIT = 0.60
ARTERIAL_SPEED_LIMIT = 1.50
DENSE_CHECK_POINTS = 2_001


def _configure_segment(phase, *, grade: float, speed_limit: float, boundaries, times):
    _, speed, _ = phase.x
    (acceleration,) = phase.u
    drag = 0.06 * speed**2
    phase.set_dynamics(
        [
            speed,
            acceleration - drag - grade,
            0.08 * speed + 0.04 * speed**3 + 0.03 * acceleration**2,
        ]
    )
    phase.set_phase_constraint(
        [speed, acceleration],
        [0.0, -1.20],
        [speed_limit, 1.00],
    )
    phase.set_boundary_condition(*boundaries, *times)
    # Piecewise-linear Lobatto interpolation carries the nodal speed and
    # acceleration bounds across each complete mesh interval.
    phase.set_discretization(40, 2)


def build_problem():
    """Return the two-road-segment system and its phases."""
    system = System(["v_switch", "e_switch", "t_switch", "e_final", "t_final"])
    v_s, e_s, t_s, e_f, t_f = system.s

    urban = system.new_phase(["position_u", "speed_u", "energy_u"], ["acceleration_u"])
    _configure_segment(
        urban,
        grade=0.025,
        speed_limit=URBAN_SPEED_LIMIT,
        boundaries=([0.0, 0.0, 0.0], [DISTANCE_SWITCH, v_s, e_s]),
        times=(0.0, t_s),
    )

    arterial = system.new_phase(
        ["position_a", "speed_a", "energy_a"], ["acceleration_a"]
    )
    _configure_segment(
        arterial,
        grade=-0.010,
        speed_limit=ARTERIAL_SPEED_LIMIT,
        boundaries=([DISTANCE_SWITCH, v_s, e_s], [DISTANCE_FINAL, 0.0, e_f]),
        times=(t_s, t_f),
    )

    system.set_phase([urban, arterial])
    system.set_objective(e_f + 0.15 * t_f)
    system.set_system_constraint(
        [v_s, e_s, t_s, e_f, t_f - t_s, t_f],
        [0.05, 0.0, 1.70, 0.0, 1.50, 3.20],
        [URBAN_SPEED_LIMIT, 5.0, 5.00, 8.0, 5.00, 9.00],
    )
    return system, (urban, arterial)


def initial_guess(phases):
    """Return physically scaled guesses for both road segments."""
    urban, arterial = phases
    v_switch = 0.55
    t_switch = 2.70
    t_final = 5.20
    e_switch = 0.20
    e_final = 0.55

    guess_u = linear_guess(urban, 0.0)
    guess_u.t_f = t_switch
    tau_u = guess_u.t_x / t_switch
    guess_u.x[0] = DISTANCE_SWITCH * tau_u
    guess_u.x[1] = v_switch * np.sin(0.5 * np.pi * tau_u)
    guess_u.x[2] = e_switch * tau_u
    guess_u.u[0] = 0.25

    guess_a = linear_guess(arterial, 0.0)
    guess_a.t_0 = t_switch
    guess_a.t_f = t_final
    tau_a = (guess_a.t_x - t_switch) / (t_final - t_switch)
    guess_a.x[0] = DISTANCE_SWITCH + (DISTANCE_FINAL - DISTANCE_SWITCH) * tau_a
    guess_a.x[1] = v_switch * (1 - tau_a) + 0.95 * np.sin(np.pi * tau_a)
    guess_a.x[2] = e_switch + (e_final - e_switch) * tau_a
    guess_a.u[0] = 0.05

    static_guess = [v_switch, e_switch, t_switch, e_final, t_final]
    return [guess_u, guess_a, static_guess]


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def solve_problem(system, guess):
    """Solve the linked urban and arterial eco-driving problem."""
    result, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-8, "max_iter": 1500, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)
    urban, arterial, static = result

    dense_segments = []
    for segment in (urban, arterial):
        time = np.linspace(segment.t_0, segment.t_f, DENSE_CHECK_POINTS)
        state = np.vstack([segment.V_x(time) @ component for component in segment.x])
        control = np.vstack([segment.V_u(time) @ component for component in segment.u])
        dense_segments.append((time, state, control))
    require_finite(
        urban_time=dense_segments[0][0],
        urban_states=dense_segments[0][1],
        urban_controls=dense_segments[0][2],
        arterial_time=dense_segments[1][0],
        arterial_states=dense_segments[1][1],
        arterial_controls=dense_segments[1][2],
        static_values=static,
        objective=info["obj_val"],
    )

    if abs(urban.x[1][-1] - arterial.x[1][0]) > 2e-6:
        raise RuntimeError("speed is not continuous at the road transition")

    path_violation = 0.0
    dense_speed_maxima = []
    for (_time, state, control), speed_limit in zip(
        dense_segments,
        (URBAN_SPEED_LIMIT, ARTERIAL_SPEED_LIMIT),
    ):
        speed = state[1]
        acceleration = control[0]
        dense_speed_maxima.append(float(np.max(speed)))
        path_violation = max(
            path_violation,
            float(np.max(-speed)),
            float(np.max(speed - speed_limit)),
            float(np.max(-1.20 - acceleration)),
            float(np.max(acceleration - 1.00)),
        )
    if path_violation > 2e-6:
        raise RuntimeError(f"dense path-bound violation: {path_violation:.3e}")

    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"transition time: {static[2]:.6f} min")
    print(f"transition speed: {60 * static[0]:.3f} km/h")
    print(f"trip time: {static[4]:.6f} min")
    print(
        "dense speed maxima: "
        f"{60 * dense_speed_maxima[0]:.3f}, "
        f"{60 * dense_speed_maxima[1]:.3f} km/h"
    )
    print(f"maximum dense path-bound violation: {max(path_violation, 0.0):.3e}")
    return urban, arterial, static


def plot_solution(result, *, save: str | Path | None = None, show: bool = True):
    """Plot speed, acceleration, and accumulated energy proxy."""
    urban, arterial, static = result
    configure_matplotlib()
    fig, axes = plt.subplots(
        3, 1, figsize=(8.0, 7.5), sharex=True, layout="constrained"
    )

    urban_time = np.linspace(urban.t_0, urban.t_f, DENSE_CHECK_POINTS)
    arterial_time = np.linspace(arterial.t_0, arterial.t_f, DENSE_CHECK_POINTS)
    urban_speed = urban.V_x(urban_time) @ urban.x[1]
    arterial_speed = arterial.V_x(arterial_time) @ arterial.x[1]
    urban_acceleration = urban.V_u(urban_time) @ urban.u[0]
    arterial_acceleration = arterial.V_u(arterial_time) @ arterial.u[0]

    axes[0].plot(urban_time, 60 * urban_speed, color="#0072B2")
    axes[0].plot(arterial_time, 60 * arterial_speed, color="#0072B2")
    axes[0].hlines(
        60 * URBAN_SPEED_LIMIT, 0.0, static[2], color="#777777", linestyle="--"
    )
    axes[0].hlines(
        60 * ARTERIAL_SPEED_LIMIT, static[2], static[4], color="#777777", linestyle="--"
    )
    axes[0].set_ylabel("Speed [km/h]")

    axes[1].plot(urban_time, urban_acceleration, color="#D55E00")
    axes[1].plot(arterial_time, arterial_acceleration, color="#D55E00")
    axes[1].axhline(0.0, color="#555555", linewidth=0.9)
    axes[1].set_ylabel("Command [km/min^2]")

    axes[2].plot(urban.t_x, urban.x[2], color="#009E73")
    axes[2].plot(arterial.t_x, arterial.x[2], color="#009E73")
    axes[2].set_ylabel("Energy proxy")
    axes[2].set_xlabel("Time [min]")
    for axis in axes:
        axis.axvline(static[2], color="#555555", linestyle="--", linewidth=1.2)
    style_axes(axes)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "multiphase_electric_vehicle_solution.png"
    )
    system, phases = build_problem()
    guess = initial_guess(phases)
    result = solve_problem(system, guess)
    plot_solution(result, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
