# Copyright (c) 2024 Yilin Zou
import numpy as np
import pytest

from pockit.lobatto import System as LobattoSystem
from pockit.radau import System as RadauSystem
from pockit.radau import constant_guess


@pytest.mark.parametrize("system_class", [RadauSystem, LobattoSystem])
def test_static_only_system(system_class):
    system = system_class(1)
    system.set_objective(system.s[0] ** 2)
    x = np.array([2.0], dtype=np.float64)

    assert system.objective(x) == pytest.approx(4.0)
    assert np.allclose(system.gradient(x), [4.0])
    assert system.constraints(x).shape == (0,)


def test_phase_check_uses_discontinuous_tolerance():
    system = RadauSystem(0)
    phase = system.new_phase(1, 1)
    phase.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1)
    phase.set_phase_constraint(
        [phase.u[0]], [0], [1], bang_bang_control=True
    ).set_discretization(1, 3)
    variable = constant_guess(phase, 0)
    variable.u[0] = 0.9995

    assert phase.check(variable, tolerance_discontinuous=1.0e-3)


def test_reconfiguring_boundary_condition_clears_old_derivatives():
    system = RadauSystem(1)
    phase = system.new_phase(1, 0)
    phase.set_dynamics([0]).set_boundary_condition(
        [system.s[0] ** 2], [None], 0, 1
    ).set_discretization(1, 3)

    phase.set_boundary_condition([None], [None], 0, 1)
    node = phase._node_state_front[0]
    assert not node.args
    assert not len(node.g_i)
    assert not len(node.h_i_row)
    assert len(node.G_i) == 1

    phase.set_boundary_condition([0], [None], 0, 1)
    assert not node.args
    assert not len(node.g_i)
    assert not len(node.h_i_row)
    assert not len(node.G_i)
    assert not len(node.H_i_row)

    system.set_phase([phase]).set_objective(0)
    x = np.concatenate([constant_guess(phase, 0).data, [2.0]])
    row, col = system.jacobianstructure()
    jacobian = np.zeros((len(system.c_lb), system.L), dtype=np.float64)
    np.add.at(jacobian, (row, col), system.jacobian(x.copy()))

    eps = 1.0e-6
    finite_difference = np.empty_like(jacobian)
    for i in range(system.L):
        delta = np.zeros(system.L, dtype=np.float64)
        delta[i] = eps
        finite_difference[:, i] = (
            system.constraints(x.copy() + delta)
            - system.constraints(x.copy() - delta)
        ) / (2 * eps)
    assert np.allclose(jacobian, finite_difference)


@pytest.mark.parametrize(
    ("system_class", "minimum_num_point"),
    [(RadauSystem, 1), (LobattoSystem, 2)],
)
def test_discretization_validation_is_atomic(system_class, minimum_num_point):
    system = system_class(0)
    phase = system.new_phase(1, 0)
    phase.set_discretization(1, minimum_num_point)

    phase.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1)
    phase.set_discretization(1, max(minimum_num_point, 3))
    mesh_before = phase._mesh.copy()
    num_point_before = phase._num_point.copy()
    discretization_before = phase._object_discretization

    invalid_values = [
        (0, 3),
        ([0], [3]),
        ([0, 0], [3]),
        ([1, 0], [3]),
        ([0, np.inf], [3]),
        ([0, 0.5, 1], [3]),
        ([0, 1], [minimum_num_point - 1]),
        ([0, 1], [2.5]),
    ]
    for mesh, num_point in invalid_values:
        with pytest.raises(ValueError):
            phase.set_discretization(mesh, num_point)
        assert phase.ok
        assert np.array_equal(phase._mesh, mesh_before)
        assert np.array_equal(phase._num_point, num_point_before)
        assert phase._object_discretization is discretization_before
