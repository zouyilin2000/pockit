"""Point a flexible space telescope while suppressing structural vibration.

Large observatories cannot be treated as perfectly rigid during a slew: an
attitude acceleration excites low-frequency appendage and optical-bench
modes.  This rest-to-rest maneuver uses a reaction-wheel torque state and a
bounded torque-slew command.  The terminal rigid-body rate, wheel torque, and
one lightly damped flexible mode are all driven to zero before observation.

The model is ``J * angle_ddot = torque`` and
``mode_ddot + 2*zeta*w*mode_dot + w**2*mode = -b*angle_ddot``, with
``torque_dot`` as the control.  A minimum-jerk rigid slew provides an
independently integrated, unshaped reference for the residual-vibration test.
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


HORIZON = 30.0  # s
TARGET_ANGLE = np.deg2rad(20.0)
SPACECRAFT_INERTIA = 1_800.0  # kg m^2
MODE_FREQUENCY = 0.32  # Hz
MODE_ANGULAR_FREQUENCY = 2.0 * np.pi * MODE_FREQUENCY  # rad/s
MODE_DAMPING_RATIO = 0.008
MODE_COUPLING = 0.18  # equivalent line-of-sight rad per rigid-body rad
MAX_TORQUE = 12.0  # N m
MAX_TORQUE_SLEW = 2.5  # N m/s
MAX_ANGULAR_RATE = np.deg2rad(1.8)  # rad/s
MAX_MODE_DISPLACEMENT = 8.0e-4  # equivalent line-of-sight rad
MODE_SCALE = 1.0e-4  # rad
TORQUE_SLEW_WEIGHT = 0.02
MODE_WEIGHT = 0.8
DENSE_CHECK_POINTS = 6_001


def build_problem():
    """Return the flexible-telescope rest-to-rest pointing problem."""
    system = System(0)
    phase = system.new_phase(
        [
            "pointing_angle",
            "angular_rate",
            "mode_displacement",
            "mode_velocity",
            "wheel_torque",
        ],
        ["torque_slew"],
    )
    angle, angular_rate, mode, mode_rate, torque = phase.x
    (torque_slew,) = phase.u

    angular_acceleration = torque / SPACECRAFT_INERTIA
    phase.set_dynamics(
        [
            angular_rate,
            angular_acceleration,
            mode_rate,
            -2.0 * MODE_DAMPING_RATIO * MODE_ANGULAR_FREQUENCY * mode_rate
            - MODE_ANGULAR_FREQUENCY**2 * mode
            - MODE_COUPLING * angular_acceleration,
            torque_slew,
        ]
    )
    phase.set_integral(
        [
            (torque / MAX_TORQUE) ** 2
            + TORQUE_SLEW_WEIGHT * (torque_slew / MAX_TORQUE_SLEW) ** 2
            + MODE_WEIGHT
            * (
                (mode / MODE_SCALE) ** 2
                + (mode_rate / (MODE_ANGULAR_FREQUENCY * MODE_SCALE)) ** 2
            )
        ]
    )
    phase.set_phase_constraint(
        [angle, angular_rate, mode, mode_rate, torque, torque_slew],
        [
            -0.02,
            -MAX_ANGULAR_RATE,
            -MAX_MODE_DISPLACEMENT,
            -MODE_ANGULAR_FREQUENCY * MAX_MODE_DISPLACEMENT,
            -MAX_TORQUE,
            -MAX_TORQUE_SLEW,
        ],
        [
            TARGET_ANGLE + 0.02,
            MAX_ANGULAR_RATE,
            MAX_MODE_DISPLACEMENT,
            MODE_ANGULAR_FREQUENCY * MAX_MODE_DISPLACEMENT,
            MAX_TORQUE,
            MAX_TORQUE_SLEW,
        ],
    )
    phase.set_boundary_condition(
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [TARGET_ANGLE, 0.0, 0.0, 0.0, 0.0],
        0.0,
        HORIZON,
    )
    # Piecewise-linear controls make the actuator bounds valid between nodes;
    # the dense mesh still resolves the 0.32 Hz flexible mode.
    phase.set_discretization(300, 2)

    system.set_phase([phase])
    system.set_objective(phase.I[0] / HORIZON)
    return system, phase


def initial_guess(phase):
    """Return a minimum-jerk rigid slew with an initially quiet mode."""
    guess = linear_guess(phase, 0.0)
    scaled_state_time = guess.t_x / HORIZON
    blend = (
        10.0 * scaled_state_time**3
        - 15.0 * scaled_state_time**4
        + 6.0 * scaled_state_time**5
    )
    blend_rate = (
        30.0 * scaled_state_time**2
        - 60.0 * scaled_state_time**3
        + 30.0 * scaled_state_time**4
    ) / HORIZON
    blend_acceleration = (
        60.0 * scaled_state_time
        - 180.0 * scaled_state_time**2
        + 120.0 * scaled_state_time**3
    ) / HORIZON**2
    guess.x[0] = TARGET_ANGLE * blend
    guess.x[1] = TARGET_ANGLE * blend_rate
    guess.x[2] = np.zeros_like(guess.t_x)
    guess.x[3] = np.zeros_like(guess.t_x)
    guess.x[4] = SPACECRAFT_INERTIA * TARGET_ANGLE * blend_acceleration

    scaled_control_time = guess.t_u / HORIZON
    blend_jerk = (
        60.0 - 360.0 * scaled_control_time + 360.0 * scaled_control_time**2
    ) / HORIZON**3
    guess.u[0] = SPACECRAFT_INERTIA * TARGET_ANGLE * blend_jerk
    return guess


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    time = np.linspace(solution.t_0, solution.t_f, DENSE_CHECK_POINTS)
    states = np.vstack([solution.V_x(time) @ component for component in solution.x])
    torque_slew = solution.V_u(time) @ solution.u[0]
    return time, states, torque_slew


def _minimum_jerk_mode_response(time: np.ndarray) -> np.ndarray:
    """Integrate the flexible response to an unshaped minimum-jerk slew."""

    def mode_dynamics(current_time, mode_state):
        scaled_time = current_time / HORIZON
        angular_acceleration = (
            TARGET_ANGLE
            * (60.0 * scaled_time - 180.0 * scaled_time**2 + 120.0 * scaled_time**3)
            / HORIZON**2
        )
        return [
            mode_state[1],
            -2.0 * MODE_DAMPING_RATIO * MODE_ANGULAR_FREQUENCY * mode_state[1]
            - MODE_ANGULAR_FREQUENCY**2 * mode_state[0]
            - MODE_COUPLING * angular_acceleration,
        ]

    reference = solve_ivp(
        mode_dynamics,
        (0.0, HORIZON),
        (0.0, 0.0),
        t_eval=time,
        rtol=1e-10,
        atol=1e-13,
    )
    if not reference.success:
        raise RuntimeError("minimum-jerk reference integration failed")
    return reference.y


def _forward_response(time: np.ndarray, torque_slew: np.ndarray) -> np.ndarray:
    """Integrate the coupled telescope dynamics for a sampled slew command."""

    def dynamics(current_time, state):
        command = np.interp(current_time, time, torque_slew)
        angular_acceleration = state[4] / SPACECRAFT_INERTIA
        return [
            state[1],
            angular_acceleration,
            state[3],
            -2.0 * MODE_DAMPING_RATIO * MODE_ANGULAR_FREQUENCY * state[3]
            - MODE_ANGULAR_FREQUENCY**2 * state[2]
            - MODE_COUPLING * angular_acceleration,
            command,
        ]

    response = solve_ivp(
        dynamics,
        (time[0], time[-1]),
        np.zeros(5),
        t_eval=time,
        rtol=2e-10,
        atol=2e-12,
        method="DOP853",
    )
    if not response.success:
        raise RuntimeError(
            f"optimized-slew forward integration failed: {response.message}"
        )
    return response.y


def solve_problem(system, guess):
    """Solve the slew and verify endpoint, bound, and integral dynamics."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-9, "max_iter": 1600, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    time, states, torque_slew = _dense_solution(solution)
    angle, angular_rate, mode, mode_rate, torque = states
    reintegrated = _forward_response(time, torque_slew)
    reference_mode = _minimum_jerk_mode_response(time)
    require_finite(
        time=time,
        states=states,
        torque_slew=torque_slew,
        objective=info["obj_val"],
        reintegrated_states=reintegrated,
        minimum_jerk_reference=reference_mode,
    )
    reintegration_error = np.max(np.abs(reintegrated - states), axis=1)
    error_scales = np.array(
        [
            TARGET_ANGLE,
            MAX_ANGULAR_RATE,
            MAX_MODE_DISPLACEMENT,
            MODE_ANGULAR_FREQUENCY * MAX_MODE_DISPLACEMENT,
            MAX_TORQUE,
        ]
    )
    scaled_reintegration_error = float(np.max(reintegration_error / error_scales))
    endpoint_target = np.array([TARGET_ANGLE, 0.0, 0.0, 0.0, 0.0])
    endpoint_error = float(np.max(np.abs(states[:, -1] - endpoint_target)))
    forward_endpoint_error = float(
        np.max(np.abs(reintegrated[:, -1] - endpoint_target) / error_scales)
    )
    path_violation = max(
        float(np.max(-0.02 - angle)),
        float(np.max(angle - TARGET_ANGLE - 0.02)),
        float(np.max(np.abs(angular_rate) - MAX_ANGULAR_RATE)),
        float(np.max(np.abs(mode) - MAX_MODE_DISPLACEMENT)),
        float(
            np.max(np.abs(mode_rate) - MODE_ANGULAR_FREQUENCY * MAX_MODE_DISPLACEMENT)
        ),
        float(np.max(np.abs(torque) - MAX_TORQUE)),
        float(np.max(np.abs(torque_slew) - MAX_TORQUE_SLEW)),
        0.0,
    )
    rigid_rate_balance = abs(
        float(
            angular_rate[-1]
            - angular_rate[0]
            - trapezoid(torque, time) / SPACECRAFT_INERTIA
        )
    )
    angle_balance = abs(float(angle[-1] - angle[0] - trapezoid(angular_rate, time)))
    torque_balance = abs(float(torque[-1] - torque[0] - trapezoid(torque_slew, time)))
    mode_displacement_balance = abs(
        float(mode[-1] - mode[0] - trapezoid(mode_rate, time))
    )
    mode_acceleration = (
        -2.0 * MODE_DAMPING_RATIO * MODE_ANGULAR_FREQUENCY * mode_rate
        - MODE_ANGULAR_FREQUENCY**2 * mode
        - MODE_COUPLING * torque / SPACECRAFT_INERTIA
    )
    mode_velocity_balance = abs(
        float(mode_rate[-1] - mode_rate[0] - trapezoid(mode_acceleration, time))
    )
    maximum_balance_error = max(
        rigid_rate_balance,
        angle_balance,
        torque_balance,
        mode_displacement_balance,
        mode_velocity_balance,
    )
    reference_residual = float(
        np.hypot(
            reference_mode[0, -1],
            reference_mode[1, -1] / MODE_ANGULAR_FREQUENCY,
        )
    )
    optimized_residual = float(
        np.hypot(mode[-1], mode_rate[-1] / MODE_ANGULAR_FREQUENCY)
    )
    forward_residual = float(
        np.hypot(
            reintegrated[2, -1],
            reintegrated[3, -1] / MODE_ANGULAR_FREQUENCY,
        )
    )
    if endpoint_error > 2e-7 or path_violation > 3e-6:
        raise RuntimeError("the telescope slew violates an endpoint or path bound")
    if maximum_balance_error > 2e-5:
        raise RuntimeError(
            f"dense integral-dynamics error is too large: {maximum_balance_error:.3e}"
        )
    if scaled_reintegration_error > 3e-3 or forward_endpoint_error > 3e-3:
        raise RuntimeError("the shaped slew failed independent forward integration")
    if optimized_residual > 1e-4 * reference_residual:
        raise RuntimeError("the collocation endpoint has residual flexible motion")
    if forward_residual > 0.05 * reference_residual:
        raise RuntimeError("the forward-integrated slew did not suppress vibration")

    peak_mode_arcsec = np.rad2deg(float(np.max(np.abs(mode)))) * 3_600.0
    print(f"status: {status_message}")
    print(f"objective: {info['obj_val']:.8f}")
    print(f"peak angular rate: {np.rad2deg(np.max(np.abs(angular_rate))):.6f} deg/s")
    print(f"peak wheel torque: {np.max(np.abs(torque)):.6f} N m")
    print(f"peak flexible line-of-sight error: {peak_mode_arcsec:.6f} arcsec")
    print(
        "unshaped/forward-integrated terminal modal amplitude: "
        f"{np.rad2deg(reference_residual) * 3_600.0:.6f} / "
        f"{np.rad2deg(forward_residual) * 3_600.0:.3e} arcsec"
    )
    print(f"maximum scaled forward-integration error: {scaled_reintegration_error:.3e}")
    print(f"scaled forward endpoint error: {forward_endpoint_error:.3e}")
    print(f"maximum dense path-bound violation: {path_violation:.3e}")
    print(f"maximum integral-dynamics error: {maximum_balance_error:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot pointing motion, flexible response, and reaction-wheel commands."""
    configure_matplotlib()
    time, states, torque_slew = _dense_solution(solution)
    angle, angular_rate, mode, _mode_rate, torque = states
    reference_mode = _minimum_jerk_mode_response(time)

    fig, axes = plt.subplots(
        2, 2, figsize=(10.0, 7.2), sharex=True, layout="constrained"
    )
    angle_axis, rate_axis, mode_axis, actuator_axis = axes.reshape(-1)
    angle_axis.plot(
        time, np.rad2deg(angle), color=COLORS["blue"], label="Pointing angle"
    )
    angle_axis.axhline(
        np.rad2deg(TARGET_ANGLE),
        color=COLORS["black"],
        linestyle="--",
        label="Target",
    )
    angle_axis.set_ylabel("Pointing angle [deg]")
    angle_axis.legend(ncol=2)

    rate_axis.plot(
        time,
        np.rad2deg(angular_rate),
        color=COLORS["orange"],
        label="Angular rate",
    )
    rate_axis.set_ylabel("Angular rate [deg/s]")
    rate_axis.legend()

    mode_axis.plot(
        time,
        np.rad2deg(reference_mode[0]) * 3_600.0,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.4,
        label="Unshaped reference",
    )
    mode_axis.plot(
        time,
        np.rad2deg(mode) * 3_600.0,
        color=COLORS["vermillion"],
        label="Optimized mode",
    )
    mode_axis.axhline(0.0, color=COLORS["black"], linewidth=1.0)
    mode_axis.set_ylabel("Mode displacement [arcsec]")
    mode_axis.set_xlabel("Time [s]")
    mode_axis.legend(loc="upper left")

    actuator_axis.plot(time, torque, color=COLORS["blue"], label="Wheel torque")
    slew_axis = actuator_axis.twinx()
    slew_axis.plot(time, torque_slew, color=COLORS["purple"], label="Torque slew")
    actuator_axis.set_ylabel("Wheel torque [N m]")
    slew_axis.set_ylabel("Torque slew [N m/s]")
    actuator_axis.set_xlabel("Time [s]")
    lines = actuator_axis.lines + slew_axis.lines
    actuator_axis.legend(lines, [line.get_label() for line in lines], ncol=2)

    style_axes(axes)
    slew_axis.grid(False)
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "flexible_telescope_slew_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
