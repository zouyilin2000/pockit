import numpy as np

from pockit.radau import System


def test_variable_bound():
    s = System(4)
    p = s.new_phase(2, 2)
    p.set_dynamics([0, 0]).set_boundary_condition(
        [0, 0], [s.s[0], 0], None, s.s[2]
    ).set_discretization([0, 0.2, 1], [3, 4]).set_phase_constraint(
        [p.x[0], p.u[1], p.t, p.s[3]], [2, 4, 6, 8], [3, np.inf, 7, 9]
    )
    s.set_phase([p]).set_objective(0).set_system_constraint([s.s[1]], [0], [1])

    lb = [2] * 8 + [-np.inf] * 8 + [-np.inf] * 7 + [4] * 7 + [6] * 2 + [2, 0, 6, 8]
    ub = [3] * 8 + [np.inf] * 8 + [np.inf] * 7 + [np.inf] * 7 + [7] * 2 + [3, 1, 7, 9]

    assert np.allclose(lb, s.v_lb)
    assert np.allclose(ub, s.v_ub)


def test_constraint_bound():
    s = System(2)
    p = s.new_phase(2, 2)
    p.set_dynamics([0, 0]).set_boundary_condition(
        [0, 0], [s.s[0], 0], None, 1
    ).set_discretization([0, 0.2, 1], [3, 4]).set_phase_constraint(
        [p.x[0], p.u[1], p.x[0] + p.u[1]], [2, 4, -1], [3, np.inf, 1]
    )
    p_2 = s.new_phase(1, 1)
    p_2.set_dynamics([0]).set_discretization(4, 4).set_boundary_condition(
        [0], [s.s[0] * 0.1], None, 3 * s.s[1]
    ).set_phase_constraint([p_2.x[0], p_2.t], [0, 1], [0, 2])
    s.set_phase([p, p_2]).set_objective(0).set_system_constraint(
        [s.s[1], s.s[0] + s.s[1]], [0, -2], [1, 2]
    )

    lb = [-2, 0, 1] + [0] * 7 * 2 + [-1] * 7 + [0] * 16
    ub = [2, 0, 2] + [0] * 7 * 2 + [1] * 7 + [0] * 16

    assert np.allclose(lb, s.c_lb)
    assert np.allclose(ub, s.c_ub)
