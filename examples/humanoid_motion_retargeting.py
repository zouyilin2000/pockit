"""Retarget a human squat-and-reach motion to a contact-constrained humanoid.

The human reference is scaled to the robot's leg length, but geometric scaling
alone does not enforce Newton-Euler dynamics. This example optimizes a planar
centroidal trajectory and both foot wrenches together. The retargeted motion
tracks the scaled reference while respecting unilateral contact, Coulomb
friction, center-of-pressure, wrench, and wrench-rate limits.

This compact model is useful for contact-aware motion retargeting and preview
control. It is not a full rigid-body model: joint torque, self-collision,
reachability, and three-dimensional contact geometry still require a downstream
whole-body optimization stage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

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


MASS = 62.0  # kg
PITCH_INERTIA = 8.0  # kg m^2
GRAVITY = 9.81  # m/s^2
HORIZON = 2.4  # s
HUMAN_LEG_LENGTH = 0.96  # m
ROBOT_LEG_LENGTH = 0.82  # m
NOMINAL_COM_HEIGHT = 0.84  # m
HUMAN_FORWARD_AMPLITUDE = 0.085  # m
HUMAN_SQUAT_DEPTH = 0.19  # m
PITCH_AMPLITUDE = 0.13  # rad

FOOT_CENTER = 0.14  # foot centers are at +/- this coordinate [m]
FOOT_HALF_LENGTH = 0.10  # m
FRICTION_COEFFICIENT = 0.55
MAX_NORMAL_FORCE = 0.90 * MASS * GRAVITY
MAX_WRENCH_RATE = np.array([900.0, 2_000.0, 220.0] * 2)
STATE_LOWER = np.array(
    [
        -0.20,
        0.60,
        -0.55,
        -0.70,
        -0.28,
        -0.80,
        -MAX_NORMAL_FORCE,
        0.0,
        -FOOT_HALF_LENGTH * MAX_NORMAL_FORCE,
        -MAX_NORMAL_FORCE,
        0.0,
        -FOOT_HALF_LENGTH * MAX_NORMAL_FORCE,
    ]
)
STATE_UPPER = np.array(
    [
        0.20,
        0.94,
        0.55,
        0.70,
        0.28,
        0.80,
        MAX_NORMAL_FORCE,
        MAX_NORMAL_FORCE,
        FOOT_HALF_LENGTH * MAX_NORMAL_FORCE,
        MAX_NORMAL_FORCE,
        MAX_NORMAL_FORCE,
        FOOT_HALF_LENGTH * MAX_NORMAL_FORCE,
    ]
)
DENSE_CHECK_POINTS = 4_001

POSITION_WEIGHT = np.array([2_400.0, 3_200.0])
VELOCITY_WEIGHT = np.array([80.0, 100.0])
PITCH_WEIGHT = 500.0
PITCH_RATE_WEIGHT = 35.0
WRENCH_WEIGHT = 0.025
WRENCH_RATE_WEIGHT = 0.004


def _bump(time):
    """Return a C2 rest-to-rest motion primitive and two derivatives."""
    tau = np.asarray(time, dtype=float) / HORIZON
    value = 64.0 * tau**3 * (1.0 - tau) ** 3
    rate = (
        64.0 * (3.0 * tau**2 - 12.0 * tau**3 + 15.0 * tau**4 - 6.0 * tau**5) / HORIZON
    )
    acceleration = (
        64.0 * (6.0 * tau - 36.0 * tau**2 + 60.0 * tau**3 - 30.0 * tau**4) / HORIZON**2
    )
    return value, rate, acceleration


def _reference(time):
    """Evaluate the geometrically scaled human reference."""
    scale = ROBOT_LEG_LENGTH / HUMAN_LEG_LENGTH
    bump, bump_rate, bump_acceleration = _bump(time)
    bump = np.atleast_1d(bump)
    bump_rate = np.atleast_1d(bump_rate)
    bump_acceleration = np.atleast_1d(bump_acceleration)
    position = np.vstack(
        [
            scale * HUMAN_FORWARD_AMPLITUDE * bump,
            NOMINAL_COM_HEIGHT - scale * HUMAN_SQUAT_DEPTH * bump,
        ]
    )
    velocity = np.vstack(
        [
            scale * HUMAN_FORWARD_AMPLITUDE * bump_rate,
            -scale * HUMAN_SQUAT_DEPTH * bump_rate,
        ]
    )
    acceleration = np.vstack(
        [
            scale * HUMAN_FORWARD_AMPLITUDE * bump_acceleration,
            -scale * HUMAN_SQUAT_DEPTH * bump_acceleration,
        ]
    )
    return (
        position,
        velocity,
        acceleration,
        PITCH_AMPLITUDE * bump,
        PITCH_AMPLITUDE * bump_rate,
        PITCH_AMPLITUDE * bump_acceleration,
    )


def _symbolic_reference(time):
    """Return the scaled reference as symbolic expressions."""
    tau = time / HORIZON
    bump = 64.0 * tau**3 * (1.0 - tau) ** 3
    scale = ROBOT_LEG_LENGTH / HUMAN_LEG_LENGTH
    position = sp.Matrix(
        [
            scale * HUMAN_FORWARD_AMPLITUDE * bump,
            NOMINAL_COM_HEIGHT - scale * HUMAN_SQUAT_DEPTH * bump,
        ]
    )
    pitch = PITCH_AMPLITUDE * bump
    return position, position.diff(time), pitch, sp.diff(pitch, time)


def _reference_wrenches(time):
    """Construct a dynamically consistent zero-foot-moment wrench guess."""
    position, _, acceleration, _, _, pitch_acceleration = _reference(time)
    total_fx = MASS * acceleration[0]
    total_fz = MASS * (GRAVITY + acceleration[1])
    desired_moment = PITCH_INERTIA * pitch_acceleration

    # For feet at +/- a and zero foot moments:
    # a(fz_R-fz_L) - cx(fz_L+fz_R) + cz(fx_L+fx_R) = I theta_ddot.
    normal_difference = (
        desired_moment + position[0] * total_fz - position[1] * total_fx
    ) / FOOT_CENTER
    return np.vstack(
        [
            0.5 * total_fx,
            0.5 * (total_fz - normal_difference),
            np.zeros_like(total_fx),
            0.5 * total_fx,
            0.5 * (total_fz + normal_difference),
            np.zeros_like(total_fx),
        ]
    )


def build_problem(quick: bool = False):
    """Build the centroidal-dynamics motion-retargeting problem."""
    system = System(0)
    phase = system.new_phase(
        [
            "com_x",
            "com_z",
            "com_vx",
            "com_vz",
            "pitch",
            "pitch_rate",
            "left_fx",
            "left_fz",
            "left_moment",
            "right_fx",
            "right_fz",
            "right_moment",
        ],
        [
            "left_fx_rate",
            "left_fz_rate",
            "left_moment_rate",
            "right_fx_rate",
            "right_fz_rate",
            "right_moment_rate",
        ],
    )
    (
        com_x,
        com_z,
        velocity_x,
        velocity_z,
        pitch,
        pitch_rate,
        left_fx,
        left_fz,
        left_moment,
        right_fx,
        right_fz,
        right_moment,
    ) = phase.x
    wrench_rate = sp.Matrix(phase.u)

    total_fx = left_fx + right_fx
    total_fz = left_fz + right_fz
    centroidal_moment = (
        (-FOOT_CENTER - com_x) * left_fz
        + com_z * left_fx
        + left_moment
        + (FOOT_CENTER - com_x) * right_fz
        + com_z * right_fx
        + right_moment
    )
    phase.set_dynamics(
        [
            velocity_x,
            velocity_z,
            total_fx / MASS,
            total_fz / MASS - GRAVITY,
            pitch_rate,
            centroidal_moment / PITCH_INERTIA,
            *wrench_rate,
        ]
    )

    reference_position, reference_velocity, reference_pitch, reference_pitch_rate = (
        _symbolic_reference(phase.t)
    )
    position_error = sp.Matrix([com_x, com_z]) - reference_position
    velocity_error = sp.Matrix([velocity_x, velocity_z]) - reference_velocity
    force_scale = MASS * GRAVITY
    moment_scale = force_scale * FOOT_HALF_LENGTH
    normalized_wrench = sp.Matrix(
        [
            left_fx / force_scale,
            (left_fz - 0.5 * force_scale) / force_scale,
            left_moment / moment_scale,
            right_fx / force_scale,
            (right_fz - 0.5 * force_scale) / force_scale,
            right_moment / moment_scale,
        ]
    )
    normalized_rate = sp.Matrix(
        [phase.u[index] / MAX_WRENCH_RATE[index] for index in range(6)]
    )
    phase.set_integral(
        [
            POSITION_WEIGHT[0] * position_error[0] ** 2
            + POSITION_WEIGHT[1] * position_error[1] ** 2
            + VELOCITY_WEIGHT[0] * velocity_error[0] ** 2
            + VELOCITY_WEIGHT[1] * velocity_error[1] ** 2
            + PITCH_WEIGHT * (pitch - reference_pitch) ** 2
            + PITCH_RATE_WEIGHT * (pitch_rate - reference_pitch_rate) ** 2
            + WRENCH_WEIGHT * normalized_wrench.dot(normalized_wrench)
            + WRENCH_RATE_WEIGHT * normalized_rate.dot(normalized_rate)
        ]
    )

    contact_margins = [
        FRICTION_COEFFICIENT * left_fz + left_fx,
        FRICTION_COEFFICIENT * left_fz - left_fx,
        FOOT_HALF_LENGTH * left_fz + left_moment,
        FOOT_HALF_LENGTH * left_fz - left_moment,
        FRICTION_COEFFICIENT * right_fz + right_fx,
        FRICTION_COEFFICIENT * right_fz - right_fx,
        FOOT_HALF_LENGTH * right_fz + right_moment,
        FOOT_HALF_LENGTH * right_fz - right_moment,
    ]
    phase.set_phase_constraint(
        [*phase.x, *phase.u, *contact_margins],
        [*STATE_LOWER, *(-MAX_WRENCH_RATE), *([0.0] * 8)],
        [*STATE_UPPER, *MAX_WRENCH_RATE, *([np.inf] * 8)],
    )

    static_wrenches = [
        0.0,
        0.5 * MASS * GRAVITY,
        0.0,
        0.0,
        0.5 * MASS * GRAVITY,
        0.0,
    ]
    boundary_state = [
        0.0,
        NOMINAL_COM_HEIGHT,
        0.0,
        0.0,
        0.0,
        0.0,
        *static_wrenches,
    ]
    phase.set_boundary_condition(boundary_state, boundary_state, 0.0, HORIZON)
    phase.set_discretization(8 if quick else 10, 3 if quick else 4)
    system.set_phase([phase]).set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct an approximately balanced reference-following initial guess."""
    guess = linear_guess(phase, 0.0)
    position, velocity, _, pitch, pitch_rate, _ = _reference(guess.t_x)
    histories = [
        position[0],
        position[1],
        velocity[0],
        velocity[1],
        pitch,
        pitch_rate,
        *_reference_wrenches(guess.t_x),
    ]
    for index, history in enumerate(histories):
        guess.x[index] = history

    step = 1.0e-4
    upper_time = np.minimum(guess.t_u + step, HORIZON)
    lower_time = np.maximum(guess.t_u - step, 0.0)
    rate = (_reference_wrenches(upper_time) - _reference_wrenches(lower_time)) / (
        upper_time - lower_time
    )
    for index, history in enumerate(rate):
        guess.u[index] = history
    return guess


def _dense_history(solution, count: int = DENSE_CHECK_POINTS):
    """Interpolate all trajectories on a uniform physical-time grid."""
    time = np.linspace(solution.t_0, solution.t_f, count)
    state_operator = solution.V_x(time.copy())
    control_operator = solution.V_u(time.copy())
    state = np.vstack([state_operator @ component for component in solution.x])
    control = np.vstack([control_operator @ component for component in solution.u])
    return time, state, control


def _forward_integrate(time, state, control):
    """Integrate the continuous dynamics independently of the transcription."""
    rate_interpolator = CubicSpline(time, control, axis=1)

    def dynamics(current_time, current_state):
        com_x, com_z = current_state[:2]
        velocity_x, velocity_z, _, pitch_rate = current_state[2:6]
        left_fx, left_fz, left_moment, right_fx, right_fz, right_moment = current_state[
            6:
        ]
        moment = (
            (-FOOT_CENTER - com_x) * left_fz
            + com_z * left_fx
            + left_moment
            + (FOOT_CENTER - com_x) * right_fz
            + com_z * right_fx
            + right_moment
        )
        return np.concatenate(
            [
                [
                    velocity_x,
                    velocity_z,
                    (left_fx + right_fx) / MASS,
                    (left_fz + right_fz) / MASS - GRAVITY,
                    pitch_rate,
                    moment / PITCH_INERTIA,
                ],
                np.asarray(rate_interpolator(current_time), dtype=float),
            ]
        )

    result = solve_ivp(
        dynamics,
        (time[0], time[-1]),
        state[:, 0],
        t_eval=time,
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    if not result.success:
        raise RuntimeError(f"independent integration failed: {result.message}")
    return result.y


def _diagnostics(
    time,
    state,
    control,
    *,
    reference_position=None,
    reference_pitch=None,
    integrated=None,
):
    """Evaluate tracking, contact, and independent-integration diagnostics."""
    if reference_position is None or reference_pitch is None:
        reference_position, _, _, reference_pitch, _, _ = _reference(time)
    left_fx, left_fz, left_moment = state[6:9]
    right_fx, right_fz, right_moment = state[9:12]
    if integrated is None:
        integrated = _forward_integrate(time, state, control)
    scales = np.array(
        [
            *([1.0] * 6),
            MASS * GRAVITY,
            MASS * GRAVITY,
            MASS * GRAVITY * FOOT_HALF_LENGTH,
            MASS * GRAVITY,
            MASS * GRAVITY,
            MASS * GRAVITY * FOOT_HALF_LENGTH,
        ]
    )
    return {
        "reference_position": reference_position,
        "reference_pitch": reference_pitch,
        "friction_margin": np.vstack(
            [
                FRICTION_COEFFICIENT * left_fz - np.abs(left_fx),
                FRICTION_COEFFICIENT * right_fz - np.abs(right_fx),
            ]
        ),
        "cop_margin": np.vstack(
            [
                FOOT_HALF_LENGTH * left_fz - np.abs(left_moment),
                FOOT_HALF_LENGTH * right_fz - np.abs(right_moment),
            ]
        ),
        "position_rms": np.sqrt(np.mean((state[:2] - reference_position) ** 2, axis=1)),
        "pitch_rms": float(np.sqrt(np.mean((state[4] - reference_pitch) ** 2))),
        "integration_error": float(
            np.max(np.abs(integrated - state) / scales[:, np.newaxis])
        ),
    }


def solve_problem(system, guess, quick: bool = False):
    """Solve the retargeting problem and verify its continuous trajectory."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={
            "tol": 3.0e-8,
            "acceptable_tol": 2.0e-7,
            "max_iter": 2_000,
            "print_level": 0,
            "sb": "yes",
            "bound_relax_factor": 0.0,
        },
    )
    status = int(info["status"])
    status_message = info["status_msg"]
    if isinstance(status_message, bytes):
        status_message = status_message.decode()
    if status not in (0, 1):
        raise RuntimeError(f"Ipopt failed ({status}): {status_message}")

    time, state, control = _dense_history(solution)
    reference_position, _, _, reference_pitch, _, _ = _reference(time)
    integrated = _forward_integrate(time, state, control)
    require_finite(
        time=time,
        state=state,
        control=control,
        objective=info["obj_val"],
        reference_position=reference_position,
        reference_pitch=reference_pitch,
        integrated_state=integrated,
    )
    diagnostics = _diagnostics(
        time,
        state,
        control,
        reference_position=reference_position,
        reference_pitch=reference_pitch,
        integrated=integrated,
    )
    require_finite(**diagnostics)
    minimum_normal = float(np.min(state[[7, 10]]))
    maximum_normal = float(np.max(state[[7, 10]]))
    minimum_friction = float(np.min(diagnostics["friction_margin"]))
    minimum_cop = float(np.min(diagnostics["cop_margin"]))
    maximum_state_bound_violation = max(
        float(np.max(STATE_LOWER[:, np.newaxis] - state)),
        float(np.max(state - STATE_UPPER[:, np.newaxis])),
        0.0,
    )
    maximum_rate_violation = max(
        float(np.max(np.abs(control) - MAX_WRENCH_RATE[:, np.newaxis])), 0.0
    )
    if minimum_normal < -2.0e-5 or maximum_normal > MAX_NORMAL_FORCE + 2.0e-4:
        raise RuntimeError("dense normal-force bounds are violated")
    if minimum_friction < -2.0e-4:
        raise RuntimeError(f"friction-cone violation: {-minimum_friction:.3e} N")
    if minimum_cop < -2.0e-4:
        raise RuntimeError(f"center-of-pressure violation: {-minimum_cop:.3e} N m")
    if maximum_rate_violation > 2.0e-3:
        raise RuntimeError(f"wrench-rate violation: {maximum_rate_violation:.3e}")
    if maximum_state_bound_violation > 2.0e-4:
        raise RuntimeError(
            f"dense state-bound violation: {maximum_state_bound_violation:.3e}"
        )
    integration_tolerance = 5.0e-2 if quick else 7.0e-3
    if diagnostics["integration_error"] > integration_tolerance:
        raise RuntimeError(
            "independent integration mismatch: "
            f"{diagnostics['integration_error']:.3e} scaled"
        )

    print(f"Ipopt status: {status_message}")
    print(
        "RMS CoM tracking error: "
        f"x={1e3 * diagnostics['position_rms'][0]:.3f} mm, "
        f"z={1e3 * diagnostics['position_rms'][1]:.3f} mm"
    )
    print(f"RMS pitch tracking error: {np.rad2deg(diagnostics['pitch_rms']):.4f} deg")
    print(f"Minimum friction margin: {minimum_friction:.3f} N")
    print(f"Minimum CoP margin: {minimum_cop:.3f} N m")
    print(f"Maximum dense state-bound violation: {maximum_state_bound_violation:.3e}")
    print(
        "Maximum independent-integration mismatch: "
        f"{diagnostics['integration_error']:.3e} scaled"
    )
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot the motion, tracking errors, contact forces, and constraint usage."""
    configure_matplotlib()
    time, state, control = _dense_history(solution)
    diagnostics = _diagnostics(time, state, control)
    reference_position = diagnostics["reference_position"]
    reference_pitch = diagnostics["reference_pitch"]
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 6.8))
    motion_axis, tracking_axis, force_axis, contact_axis = axes.ravel()

    motion_axis.plot(
        state[0],
        state[1],
        color=COLORS["blue"],
        linewidth=2.8,
        label="Retargeted robot CoM",
    )
    motion_axis.plot(
        reference_position[0],
        reference_position[1],
        color=COLORS["black"],
        linewidth=1.5,
        linestyle="--",
        label="Scaled human reference",
    )
    for center in (-FOOT_CENTER, FOOT_CENTER):
        motion_axis.plot(
            [center - FOOT_HALF_LENGTH, center + FOOT_HALF_LENGTH],
            [0.0, 0.0],
            color=COLORS["orange"],
            linewidth=5.0,
        )
    for index in np.unique(np.linspace(0, time.size - 1, 7, dtype=int)):
        direction = np.array([np.sin(state[4, index]), np.cos(state[4, index])])
        endpoints = state[:2, index, np.newaxis] + np.outer(direction, [-0.22, 0.22])
        motion_axis.plot(
            endpoints[0],
            endpoints[1],
            color=COLORS["green"],
            alpha=0.30,
            linewidth=2.0,
        )
    motion_axis.set(
        xlabel="Sagittal position [m]",
        ylabel="Height [m]",
        title="Contact-aware centroidal motion",
        xlim=(-0.26, 0.24),
        ylim=(-0.03, 1.12),
    )
    motion_axis.set_aspect("equal", adjustable="box")
    handles, labels = motion_axis.get_legend_handles_labels()
    motion_axis.legend(handles[::-1], labels[::-1], loc="lower right")

    tracking_axis.plot(
        time,
        1e3 * (state[0] - reference_position[0]),
        color=COLORS["blue"],
        label="CoM x error",
    )
    tracking_axis.plot(
        time,
        1e3 * (state[1] - reference_position[1]),
        color=COLORS["orange"],
        label="CoM z error",
    )
    pitch_axis = tracking_axis.twinx()
    pitch_axis.plot(
        time,
        np.rad2deg(state[4] - reference_pitch),
        color=COLORS["green"],
        label="Pitch error",
    )
    tracking_axis.set(
        xlabel="Time [s]",
        ylabel="CoM tracking error [mm]",
        title="Reference tracking",
    )
    pitch_axis.set_ylabel("Pitch tracking error [deg]")
    tracking_axis.legend(loc="upper left")
    pitch_axis.legend(loc="upper right")

    force_axis.plot(time, state[7], color=COLORS["blue"], label="Left normal")
    force_axis.plot(time, state[10], color=COLORS["orange"], label="Right normal")
    tangential_axis = force_axis.twinx()
    tangential_axis.plot(
        time,
        state[6] + state[9],
        color=COLORS["green"],
        linestyle="--",
        label="Total tangential",
    )
    force_axis.set(
        xlabel="Time [s]",
        ylabel="Normal force [N]",
        title="Optimized ground-reaction forces",
    )
    tangential_axis.set_ylabel("Tangential force [N]")
    force_axis.legend(loc="upper left")
    tangential_axis.legend(loc="upper right")

    friction_usage = np.vstack(
        [
            np.abs(state[6]) / (FRICTION_COEFFICIENT * state[7]),
            np.abs(state[9]) / (FRICTION_COEFFICIENT * state[10]),
        ]
    )
    cop_usage = np.vstack(
        [
            np.abs(state[8]) / (FOOT_HALF_LENGTH * state[7]),
            np.abs(state[11]) / (FOOT_HALF_LENGTH * state[10]),
        ]
    )
    contact_axis.plot(
        time, friction_usage[0], color=COLORS["blue"], label="Left friction"
    )
    contact_axis.plot(
        time, friction_usage[1], color=COLORS["orange"], label="Right friction"
    )
    contact_axis.plot(
        time,
        cop_usage[0],
        color=COLORS["purple"],
        linestyle="--",
        label="Left CoP",
    )
    contact_axis.plot(
        time,
        cop_usage[1],
        color=COLORS["green"],
        linestyle="--",
        label="Right CoP",
    )
    contact_axis.axhline(1.0, color=COLORS["black"], linewidth=1.0, linestyle=":")
    contact_axis.set(
        xlabel="Time [s]",
        ylabel="Constraint utilization",
        title="Friction and center-of-pressure usage",
        ylim=(0.0, 1.08),
    )
    contact_axis.legend(ncol=2, loc="upper right", bbox_to_anchor=(1.0, 0.88))

    style_axes(
        [
            motion_axis,
            tracking_axis,
            pitch_axis,
            force_axis,
            tangential_axis,
            contact_axis,
        ]
    )
    figure.tight_layout()
    save_or_show(figure, save, show)
    return figure


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "humanoid_motion_retargeting_solution.png", quick=True
    )
    system, phase = build_problem(quick=args.quick)
    guess = initial_guess(phase)
    solution = solve_problem(system, guess, quick=args.quick)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
