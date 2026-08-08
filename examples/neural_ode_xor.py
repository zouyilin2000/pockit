"""Train an augmented Neural ODE to classify the XOR data set.

The four dimensionless inputs are the corners of ``{-1, +1}^2``.  Samples
with different signs have label ``+1`` and samples with equal signs have
label ``-1``.  No affine classifier can attain a positive margin on these
points: adding the two positive-class inequalities requires the intercept to
be positive, while adding the two negative-class inequalities requires it to
be negative.

For sample ``i``, the augmented state is ``(x1_i, x2_i, h_i)``.  The input
coordinates remain fixed and a leaky continuous-depth layer evolves the logit

    dh_i/dt = c0(t) + c_left(t) tanh(k (x1_i + x2_i + 1))
              + c_right(t) tanh(k (x1_i + x2_i - 1)) - decay h_i.

This is a one-hidden-layer tanh vector field whose hidden directions and
thresholds encode the symmetry of XOR.  Its three trainable output weights
are shared by all four samples.  The extra state lifts the linearly
inseparable inputs out of their plane; the fixed readout is simply
``sign(h_i(1))``.  Training minimizes integrated squared weight magnitude
subject to a hard signed terminal margin.  Depth, states, weights, and the
objective are all dimensionless.

The restricted architecture is deliberate: it is an authentic
continuous-depth supervised-learning problem, yet its minimum-energy solution
can also be derived analytically.  That reference and an independent forward
integration make the example suitable for validating the transcription. The
fixed features encode XOR symmetry and only their shared output weights are
trained, so this is not a general-purpose architecture or a generalization test.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

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


INPUTS = np.array(
    [
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ]
)
LABELS = np.array([-1.0, 1.0, 1.0, -1.0])
DEPTH = 1.0
FEATURE_SLOPE = 2.0
LATENT_DECAY = 0.4
CLASSIFICATION_MARGIN = 1.0
MAX_LOGIT = 2.0
MAX_WEIGHT = 3.0
DENSE_CHECK_POINTS = 4_001


def _feature_matrix(inputs: np.ndarray = INPUTS) -> np.ndarray:
    """Evaluate the fixed tanh hidden features for each input sample."""
    input_sum = np.sum(np.asarray(inputs, dtype=float), axis=1)
    return np.column_stack(
        [
            np.ones(input_sum.size),
            np.tanh(FEATURE_SLOPE * (input_sum + 1.0)),
            np.tanh(FEATURE_SLOPE * (input_sum - 1.0)),
        ]
    )


def _kernel_energy() -> float:
    """Return the squared L2 norm of the terminal leakage kernel."""
    return (1.0 - np.exp(-2.0 * LATENT_DECAY * DEPTH)) / (2.0 * LATENT_DECAY)


def _analytical_solution(time: np.ndarray):
    """Return the minimum-energy weights and latent trajectories."""
    time = np.asarray(time, dtype=float)
    terminal_logits = CLASSIFICATION_MARGIN * LABELS
    integrated_weights = np.linalg.lstsq(
        _feature_matrix(), terminal_logits, rcond=None
    )[0]
    terminal_kernel = np.exp(-LATENT_DECAY * (DEPTH - time))
    weights = np.outer(integrated_weights, terminal_kernel / _kernel_energy())
    latent_factor = (
        np.exp(-LATENT_DECAY * (DEPTH - time)) - np.exp(-LATENT_DECAY * (DEPTH + time))
    ) / (1.0 - np.exp(-2.0 * LATENT_DECAY * DEPTH))
    logits = np.outer(terminal_logits, latent_factor)
    objective = float(np.dot(integrated_weights, integrated_weights) / _kernel_energy())
    return logits, weights, objective


def build_problem():
    """Return the shared-weight augmented Neural ODE training problem."""
    terminal_names = [f"terminal_logit_{index}" for index in range(len(INPUTS))]
    system = System(terminal_names)
    phase = system.new_phase(
        [
            name
            for index in range(len(INPUTS))
            for name in (f"x1_{index}", f"x2_{index}", f"logit_{index}")
        ],
        ["bias_weight", "left_feature_weight", "right_feature_weight"],
    )
    bias_weight, left_weight, right_weight = phase.u

    dynamics = []
    phase_expressions = []
    phase_lower_bounds = []
    phase_upper_bounds = []
    initial_state = []
    terminal_state = []
    for sample_index, (input_point, terminal_logit) in enumerate(
        zip(INPUTS, system.s, strict=True)
    ):
        x1, x2, logit = phase.x[3 * sample_index : 3 * sample_index + 3]
        input_sum = x1 + x2
        feature_left = sp.tanh(FEATURE_SLOPE * (input_sum + 1.0))
        feature_right = sp.tanh(FEATURE_SLOPE * (input_sum - 1.0))
        dynamics.extend(
            [
                0.0,
                0.0,
                bias_weight
                + left_weight * feature_left
                + right_weight * feature_right
                - LATENT_DECAY * logit,
            ]
        )
        phase_expressions.extend([x1, x2, logit])
        phase_lower_bounds.extend([-1.05, -1.05, -MAX_LOGIT])
        phase_upper_bounds.extend([1.05, 1.05, MAX_LOGIT])
        initial_state.extend([*input_point, 0.0])
        terminal_state.extend([None, None, terminal_logit])

    phase.set_dynamics(dynamics)
    phase.set_integral([bias_weight**2 + left_weight**2 + right_weight**2])
    phase.set_phase_constraint(
        [*phase_expressions, *phase.u],
        [*phase_lower_bounds, *([-MAX_WEIGHT] * 3)],
        [*phase_upper_bounds, *([MAX_WEIGHT] * 3)],
    )
    phase.set_boundary_condition(initial_state, terminal_state, 0.0, DEPTH)
    phase.set_discretization(32, 4)

    system.set_phase([phase])
    system.set_objective(phase.I[0])
    signed_terminal_logits = [
        label * terminal_logit
        for label, terminal_logit in zip(LABELS, system.s, strict=True)
    ]
    system.set_system_constraint(
        signed_terminal_logits,
        [CLASSIFICATION_MARGIN] * len(INPUTS),
        [MAX_LOGIT] * len(INPUTS),
    )
    return system, phase


def initial_guess(phase):
    """Use the analytical minimum-energy classifier as the initial guess."""
    guess = linear_guess(phase, 0.0)
    analytical_logits, _, _ = _analytical_solution(guess.t_x)
    _, control_weights, _ = _analytical_solution(guess.t_u)
    for sample_index, input_point in enumerate(INPUTS):
        guess.x[3 * sample_index] = np.full_like(guess.t_x, input_point[0])
        guess.x[3 * sample_index + 1] = np.full_like(guess.t_x, input_point[1])
        guess.x[3 * sample_index + 2] = analytical_logits[sample_index]
    for weight_index in range(3):
        guess.u[weight_index] = control_weights[weight_index]
    return [guess, CLASSIFICATION_MARGIN * LABELS]


def _status_message(info) -> str:
    message = info["status_msg"]
    return message.decode() if isinstance(message, bytes) else str(message)


def _dense_solution(solution):
    phase_solution, static = solution
    depth = np.linspace(phase_solution.t_0, phase_solution.t_f, DENSE_CHECK_POINTS)
    states = np.vstack(
        [phase_solution.V_x(depth) @ component for component in phase_solution.x]
    )
    inputs = np.stack([states[0::3], states[1::3]], axis=2)
    logits = states[2::3]
    weights = np.vstack(
        [phase_solution.V_u(depth) @ component for component in phase_solution.u]
    )
    return depth, inputs, logits, weights, np.asarray(static, dtype=float)


def _forward_integrate(phase_solution, depth: np.ndarray) -> np.ndarray:
    """Integrate the learned shared vector field independently of collocation."""
    features = _feature_matrix()

    def latent_dynamics(current_depth, logits):
        interpolation = phase_solution.V_u(np.array([current_depth]))
        weights = np.array(
            [(interpolation @ component).item() for component in phase_solution.u]
        )
        return features @ weights - LATENT_DECAY * logits

    integration = solve_ivp(
        latent_dynamics,
        (0.0, DEPTH),
        np.zeros(len(INPUTS)),
        t_eval=depth,
        rtol=2e-11,
        atol=2e-12,
        method="DOP853",
    )
    if not integration.success:
        raise RuntimeError(f"forward integration failed: {integration.message}")
    return integration.y


def solve_problem(system, guess):
    """Train the classifier and validate it on a dense depth grid."""
    solution, info = ipopt.solve(
        system,
        guess,
        optimizer_options={"tol": 1e-10, "max_iter": 1000, "print_level": 0},
    )
    status_message = _status_message(info)
    if info["status"] not in (0, 1):
        raise RuntimeError(status_message)

    phase_solution = solution[0]
    depth, inputs, logits, weights, terminal_parameters = _dense_solution(solution)
    analytical_logits, analytical_weights, analytical_objective = _analytical_solution(
        depth
    )
    reintegrated_logits = _forward_integrate(phase_solution, depth)
    require_finite(
        depth=depth,
        inputs=inputs,
        logits=logits,
        weights=weights,
        terminal_parameters=terminal_parameters,
        objective=info["obj_val"],
        analytical_logits=analytical_logits,
        analytical_weights=analytical_weights,
        analytical_objective=analytical_objective,
        reintegrated_logits=reintegrated_logits,
    )

    signed_margins = LABELS * logits[:, -1]
    minimum_margin = float(np.min(signed_margins))
    classification_accuracy = float(np.mean(np.sign(logits[:, -1]) == LABELS))
    terminal_parameter_error = float(
        np.max(np.abs(logits[:, -1] - terminal_parameters))
    )
    input_drift = float(np.max(np.abs(inputs - INPUTS[:, np.newaxis, :])))
    control_violation = max(float(np.max(np.abs(weights) - MAX_WEIGHT)), 0.0)
    logit_violation = max(float(np.max(np.abs(logits) - MAX_LOGIT)), 0.0)
    forward_error = float(np.max(np.abs(logits - reintegrated_logits)))
    analytical_logit_error = float(np.max(np.abs(logits - analytical_logits)))
    analytical_weight_error = float(np.max(np.abs(weights - analytical_weights)))
    objective_error = abs(float(info["obj_val"] - analytical_objective))

    if classification_accuracy != 1.0 or minimum_margin < CLASSIFICATION_MARGIN - 2e-7:
        raise RuntimeError("the learned terminal logits do not separate XOR")
    if max(terminal_parameter_error, input_drift) > 2e-7:
        raise RuntimeError("the augmented state violates an endpoint condition")
    if max(control_violation, logit_violation) > 2e-7:
        raise RuntimeError("the dense trajectory violates a state or weight bound")
    if forward_error > 3e-7:
        raise RuntimeError("the learned vector field failed forward reintegration")
    if max(analytical_logit_error, analytical_weight_error) > 3e-6:
        raise RuntimeError("the learned trajectory differs from the analytical optimum")
    if objective_error > 2e-6:
        raise RuntimeError("the training loss differs from the analytical optimum")

    print(f"status: {status_message}")
    print(f"training objective: {info['obj_val']:.10f}")
    print(f"analytical objective: {analytical_objective:.10f}")
    print(f"classification accuracy: {classification_accuracy:.0%}")
    print(f"minimum signed margin: {minimum_margin:.10f}")
    print(f"maximum weight magnitude: {np.max(np.abs(weights)):.10f}")
    print(f"maximum forward-reintegration error: {forward_error:.3e}")
    print(f"maximum analytical logit error: {analytical_logit_error:.3e}")
    return solution


def plot_solution(solution, *, save: str | Path | None = None, show: bool = True):
    """Plot the raw data, latent logits, learned weights, and augmented lift."""
    configure_matplotlib()
    depth, inputs, logits, weights, _ = _dense_solution(solution)

    fig = plt.figure(figsize=(10.4, 8.0), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    input_axis = fig.add_subplot(grid[0, 0])
    logit_axis = fig.add_subplot(grid[0, 1])
    weight_axis = fig.add_subplot(grid[1, 0])
    lift_axis = fig.add_subplot(grid[1, 1], projection="3d")

    class_styles = {
        -1.0: (COLORS["vermillion"], "X", "Equal signs (-1)"),
        1.0: (COLORS["blue"], "o", "Different signs (+1)"),
    }
    for label, (color, marker, legend_label) in class_styles.items():
        mask = LABELS == label
        input_axis.scatter(
            INPUTS[mask, 0],
            INPUTS[mask, 1],
            color=color,
            marker=marker,
            s=72,
            linewidths=2.0,
            label=legend_label,
        )
    input_axis.set(
        xlabel=r"Input $x_1$",
        ylabel=r"Input $x_2$",
        title="Linearly inseparable input plane",
        xlim=(-1.35, 1.35),
        ylim=(-1.35, 1.35),
        xticks=[-1, 0, 1],
        yticks=[-1, 0, 1],
        aspect="equal",
    )
    input_axis.legend(loc="upper center", ncol=1)

    line_styles = ["-", "-", "--", "--"]
    for sample_index, (point, label) in enumerate(zip(INPUTS, LABELS, strict=True)):
        color = COLORS["blue"] if label > 0.0 else COLORS["vermillion"]
        point_label = f"({point[0]:+.0f}, {point[1]:+.0f})"
        logit_axis.plot(
            depth,
            logits[sample_index],
            color=color,
            linestyle=line_styles[sample_index],
            label=point_label,
        )
    logit_axis.axhline(
        CLASSIFICATION_MARGIN,
        color=COLORS["black"],
        linestyle=":",
        linewidth=1.2,
    )
    logit_axis.axhline(
        -CLASSIFICATION_MARGIN,
        color=COLORS["black"],
        linestyle=":",
        linewidth=1.2,
    )
    logit_axis.set(
        xlabel="Continuous depth",
        ylabel="Augmented logit",
        title="Shared-flow classification",
    )
    logit_axis.legend(ncol=2)

    weight_labels = [r"Bias $c_0$", r"Left feature $c_L$", r"Right feature $c_R$"]
    weight_colors = [COLORS["black"], COLORS["green"], COLORS["purple"]]
    for weight, label, color in zip(weights, weight_labels, weight_colors, strict=True):
        weight_axis.plot(depth, weight, color=color, label=label)
    weight_axis.set(
        xlabel="Continuous depth",
        ylabel="Shared weight",
        title="Minimum-energy trainable weights",
    )
    weight_axis.legend(ncol=1)

    for sample_index, label in enumerate(LABELS):
        color = COLORS["blue"] if label > 0.0 else COLORS["vermillion"]
        lift_axis.plot(
            inputs[sample_index, :, 0],
            inputs[sample_index, :, 1],
            logits[sample_index],
            color=color,
            linewidth=2.5,
        )
        lift_axis.scatter(
            inputs[sample_index, -1, 0],
            inputs[sample_index, -1, 1],
            logits[sample_index, -1],
            color=color,
            marker="o" if label > 0.0 else "X",
            s=42,
        )
    lift_axis.plot_surface(
        np.array([[-1.25, 1.25], [-1.25, 1.25]]),
        np.array([[-1.25, -1.25], [1.25, 1.25]]),
        np.zeros((2, 2)),
        color="#B8B8B8",
        alpha=0.18,
        linewidth=0,
    )
    lift_axis.set(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        zlabel="Logit",
        title="Augmented-state lift",
        xlim=(-1.3, 1.3),
        ylim=(-1.3, 1.3),
        zlim=(-1.2, 1.2),
        box_aspect=(1.0, 1.0, 0.9),
    )
    lift_axis.view_init(elev=24, azim=-55)

    style_axes([input_axis, logit_axis, weight_axis])
    fig.suptitle("Augmented Neural ODE training on XOR", fontsize=14, fontweight="bold")
    save_or_show(fig, save, show)
    return fig


def main() -> None:
    args = parse_plot_arguments(__doc__.splitlines()[0], "neural_ode_xor_solution.png")
    system, phase = build_problem()
    guess = initial_guess(phase)
    solution = solve_problem(system, guess)
    plot_solution(solution, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
