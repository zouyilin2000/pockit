"""Solve a finite-horizon scalar linear-quadratic regulator problem.

The controller regulates a unit initial state over a fixed, nondimensional
time horizon. The running cost penalizes state error and control effort, while
a terminal penalty leaves the final state free but discourages residual error.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from pockit.lobatto import System, constant_guess
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


DYNAMICS_A = -1.0
DYNAMICS_B = 1.0
STATE_WEIGHT = 1.0
CONTROL_WEIGHT = 0.1
TERMINAL_WEIGHT = 1.0
INITIAL_STATE = 1.0
HORIZON = 1.0
DENSE_CHECK_POINTS = 2_001


def build_problem():
    """Build the scalar LQR transcription and return its system and phase."""
    system = System(["x_final"])
    (x_final,) = system.s
    phase = system.new_phase(["x"], ["u"])
    (x,) = phase.x
    (u,) = phase.u

    phase.set_dynamics([DYNAMICS_A * x + DYNAMICS_B * u])
    phase.set_integral([STATE_WEIGHT * x**2 + CONTROL_WEIGHT * u**2])
    phase.set_boundary_condition([INITIAL_STATE], [x_final], 0.0, HORIZON)
    phase.set_discretization(8, 6)

    system.set_phase([phase])
    system.set_objective(phase.I[0] + TERMINAL_WEIGHT * x_final**2 / 2.0)
    return system, phase


def initial_guess(phase):
    """Return a phase guess and a consistent guess for the terminal state."""
    phase_guess = constant_guess(phase, 0.0)
    static_guess = np.array([0.0])
    return [phase_guess, static_guess]


def _riccati_reference(time):
    """Return the continuous-time optimum from the scalar Riccati equation."""
    riccati = solve_ivp(
        lambda _, value: (
            -2.0 * STATE_WEIGHT
            - 2.0 * DYNAMICS_A * value
            + DYNAMICS_B**2 * value**2 / (2.0 * CONTROL_WEIGHT)
        ),
        (HORIZON, 0.0),
        [TERMINAL_WEIGHT],
        rtol=1.0e-12,
        atol=1.0e-14,
        dense_output=True,
    )
    if not riccati.success:
        raise RuntimeError(riccati.message)

    def closed_loop_dynamics(t, state):
        gain = DYNAMICS_B * riccati.sol(t)[0] / (2.0 * CONTROL_WEIGHT)
        return [(DYNAMICS_A - DYNAMICS_B * gain) * state[0]]

    state_solution = solve_ivp(
        closed_loop_dynamics,
        (0.0, HORIZON),
        [INITIAL_STATE],
        t_eval=time,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    if not state_solution.success:
        raise RuntimeError(state_solution.message)
    state = state_solution.y[0]
    control = -DYNAMICS_B * riccati.sol(time)[0] * state / (2.0 * CONTROL_WEIGHT)
    objective = float(riccati.sol(0.0)[0] * INITIAL_STATE**2 / 2.0)
    return state, control, objective


def solve_problem(system, guess):
    """Solve the LQR problem and compare it with the Riccati solution."""
    solution, info = ipopt.solve(
        system, guess, optimizer_options={"print_level": 0, "sb": "yes"}
    )
    status = int(info["status"])
    status_message = info["status_msg"]
    if isinstance(status_message, bytes):
        status_message = status_message.decode()
    if status not in (0, 1):
        raise RuntimeError(f"Ipopt failed ({status}): {status_message}")

    phase_solution, static_solution = solution
    static_values = np.asarray(static_solution, dtype=float)
    time = np.linspace(0.0, HORIZON, DENSE_CHECK_POINTS)
    state = phase_solution.V_x(time) @ phase_solution.x[0]
    control = phase_solution.V_u(time) @ phase_solution.u[0]
    state_reference, control_reference, reference_objective = _riccati_reference(time)
    require_finite(
        time=time,
        state=state,
        control=control,
        static_values=static_values,
        objective=info["obj_val"],
        reference_state=state_reference,
        reference_control=control_reference,
        reference_objective=reference_objective,
    )
    terminal_state = float(static_values[0])
    np.testing.assert_allclose(
        phase_solution.x[0][-1], terminal_state, rtol=0.0, atol=1.0e-7
    )
    state_error = float(np.max(np.abs(state - state_reference)))
    control_error = float(np.max(np.abs(control - control_reference)))
    objective_error = abs(float(info["obj_val"]) - reference_objective)
    if state_error > 2.0e-7 or control_error > 2.0e-6 or objective_error > 2.0e-7:
        raise RuntimeError(
            "the collocation solution does not match the Riccati reference: "
            f"state={state_error:.3e}, control={control_error:.3e}, "
            f"objective={objective_error:.3e}"
        )

    print(f"Ipopt status: {status_message}")
    print(f"Objective: {float(info['obj_val']):.12f}")
    print(f"Terminal state: {terminal_state:.12f}")
    print(f"Maximum Riccati state error: {state_error:.3e}")
    print(f"Maximum Riccati control error: {control_error:.3e}")
    print(f"Riccati objective error: {objective_error:.3e}")
    return solution


def plot_solution(solution, *, save=None, show=True):
    """Plot the optimal state and control histories."""
    configure_matplotlib()
    phase_solution, _ = solution
    time = np.linspace(0.0, HORIZON, DENSE_CHECK_POINTS)
    reference_state, reference_control, _ = _riccati_reference(time)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)

    axes[0].plot(
        phase_solution.t_x,
        phase_solution.x[0],
        color=COLORS["blue"],
        label="State",
    )
    axes[0].plot(
        time,
        reference_state,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.2,
        label="Riccati reference",
    )
    axes[0].set_ylabel(r"State $x$ [-]")
    axes[0].set_title("Finite-horizon linear-quadratic regulator")
    axes[0].legend()

    axes[1].plot(
        phase_solution.t_u,
        phase_solution.u[0],
        color=COLORS["orange"],
        label="Control",
    )
    axes[1].plot(
        time,
        reference_control,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.2,
        label="Riccati reference",
    )
    axes[1].set_xlabel(r"Time $t$ [-]")
    axes[1].set_ylabel(r"Control $u$ [-]")
    axes[1].legend()

    style_axes(axes)
    fig.tight_layout()
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    """Run the example from the command line."""
    args = parse_plot_arguments(__doc__.splitlines()[0], "lqr_solution.png")
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
