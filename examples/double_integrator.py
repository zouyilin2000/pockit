"""Solve the minimum-time transfer of a bounded double integrator.

A unit mass starts one meter from the target at rest. Its acceleration is
bounded to one meter per second squared, so the analytical optimum accelerates
toward the target for one second and brakes for one second.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from pockit.optimizer import ipopt
from pockit.radau import System, linear_guess

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


INITIAL_POSITION = 1.0
TARGET_POSITION = 0.0
MAX_ACCELERATION = 1.0
COLLOCATION_ACCELERATION_LIMIT = MAX_ACCELERATION - 1.0e-7
ANALYTICAL_FINAL_TIME = 2.0
DENSE_CHECK_POINTS = 10_001


def build_problem():
    """Build the minimum-time double-integrator optimal control problem."""
    system = System(0)
    phase = system.new_phase(["position", "velocity"], ["acceleration"])
    _, velocity = phase.x
    (acceleration,) = phase.u

    phase.set_dynamics([velocity, acceleration])
    phase.set_integral([1.0])
    phase.set_phase_constraint(
        [acceleration],
        [-COLLOCATION_ACCELERATION_LIMIT],
        [COLLOCATION_ACCELERATION_LIMIT],
        True,
    )
    phase.set_boundary_condition(
        [INITIAL_POSITION, 0.0], [TARGET_POSITION, 0.0], 0.0, None
    )
    phase.set_discretization([0.0, 0.5, 1.0], [6, 6])

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    return system, phase


def initial_guess(phase):
    """Construct a smooth state guess and a two-arc acceleration guess."""
    guess = linear_guess(phase, 0.0)
    guess.t_f = 2.2

    tau_x = guess.t_x / guess.t_f
    guess.x[0] = 1.0 - 3.0 * tau_x**2 + 2.0 * tau_x**3
    guess.x[1] = (-6.0 * tau_x + 6.0 * tau_x**2) / guess.t_f
    guess.u[0] = np.where(
        guess.t_u < guess.t_f / 2.0,
        -COLLOCATION_ACCELERATION_LIMIT,
        COLLOCATION_ACCELERATION_LIMIT,
    )
    return guess


def _dense_history(solution, count: int = DENSE_CHECK_POINTS):
    """Interpolate the state and control onto a uniform physical-time grid."""
    time = np.linspace(solution.t_0, solution.t_f, count)
    state_interpolation = solution.V_x(time.copy())
    control_interpolation = solution.V_u(time.copy())
    state = np.vstack([state_interpolation @ component for component in solution.x])
    control = np.asarray(control_interpolation @ solution.u[0]).reshape(-1)
    return time, state, control


def solve_problem(system, guess):
    """Solve the transfer and check it against the analytical final time."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={
            "tol": 1.0e-10,
            "acceptable_tol": 1.0e-9,
            "max_iter": 1_000,
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

    require_finite(
        state_time=solution.t_x,
        control_time=solution.t_u,
        states=np.vstack(solution.x),
        controls=np.vstack(solution.u),
        final_time=solution.t_f,
        objective=info["obj_val"],
    )

    np.testing.assert_allclose(
        solution.t_f, ANALYTICAL_FINAL_TIME, rtol=0.0, atol=2.0e-5
    )
    np.testing.assert_allclose(
        [solution.x[0][-1], solution.x[1][-1]],
        [TARGET_POSITION, 0.0],
        rtol=0.0,
        atol=1.0e-7,
    )

    switching_time = solution.t_f / 2.0
    if not np.all(solution.u[0][solution.t_u < switching_time] < 0.0):
        raise RuntimeError("the acceleration before the switch must be negative")
    if not np.all(solution.u[0][solution.t_u >= switching_time] > 0.0):
        raise RuntimeError("the acceleration after the switch must be positive")

    dense_time, dense_state, dense_acceleration = _dense_history(solution)
    require_finite(
        dense_time=dense_time,
        dense_state=dense_state,
        dense_acceleration=dense_acceleration,
    )
    control_violation = max(
        float(np.max(np.abs(dense_acceleration)) - MAX_ACCELERATION), 0.0
    )
    if control_violation > 1.0e-10:
        raise RuntimeError(
            f"dense acceleration-bound violation: {control_violation:.3e}"
        )

    print(f"Ipopt status: {status_message}")
    print(f"Minimum time: {solution.t_f:.12f} s")
    print(f"Estimated switching time: {switching_time:.12f} s")
    print(
        f"Dense acceleration range: [{np.min(dense_acceleration):.12f}, "
        f"{np.max(dense_acceleration):.12f}] m/s^2"
    )
    return solution


def plot_solution(solution, *, save=None, show=True):
    """Plot the numerical trajectory together with the analytical solution."""
    configure_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True)

    reference_time = np.linspace(0.0, ANALYTICAL_FINAL_TIME, 401)
    reference_position = np.where(
        reference_time <= 1.0,
        1.0 - 0.5 * reference_time**2,
        0.5 * (ANALYTICAL_FINAL_TIME - reference_time) ** 2,
    )
    reference_velocity = np.where(
        reference_time <= 1.0,
        -reference_time,
        reference_time - ANALYTICAL_FINAL_TIME,
    )

    axes[0].plot(
        reference_time,
        reference_position,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.4,
        label="Analytical",
    )
    axes[0].plot(
        solution.t_x,
        solution.x[0],
        color=COLORS["blue"],
        label="pockit",
    )
    axes[0].set_ylabel(r"Position $x$ [m]")
    axes[0].set_title("Minimum-time double-integrator transfer")
    axes[0].legend()

    axes[1].plot(
        reference_time,
        reference_velocity,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.4,
        label="Analytical",
    )
    axes[1].plot(
        solution.t_x,
        solution.x[1],
        color=COLORS["green"],
        label="pockit",
    )
    axes[1].set_ylabel(r"Velocity $v$ [m/s]")
    axes[1].legend()

    dense_time, _, dense_acceleration = _dense_history(solution)
    axes[2].plot(
        dense_time,
        dense_acceleration,
        color=COLORS["orange"],
        label="Acceleration",
    )
    axes[2].axhline(0.0, color=COLORS["black"], linewidth=0.8)
    axes[2].set_xlabel(r"Time $t$ [s]")
    axes[2].set_ylabel(r"Acceleration $u$ [m/s$^2$]")
    axes[2].set_ylim(-1.15, 1.15)
    axes[2].legend()

    style_axes(axes)
    fig.tight_layout()
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(
        __doc__.splitlines()[0], "double_integrator_solution.png"
    )
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
