# Copyright (c) 2024 Yilin Zou
import numpy as np
import pytest

from pockit.radau import System, constant_guess


class TestCheckRadau:
    s = System(1)
    p = s.new_phase(1, 1)
    p.set_dynamics([p.u[0]])
    p.set_boundary_condition([None], [None], None, None)
    p.set_phase_constraint([p.u[0] + p.s[0]], [0.0], [2.0], [True])
    p.set_discretization([0, 0.1, 1], [2, 3])
    s.set_phase([p])
    s.set_objective(s.s[0])

    def test_check_discontinuous(self):
        v = constant_guess(self.p, 0.0)
        assert self.s.check_discontinuous([v, [2.0]])
        assert self.s.check_discontinuous([v, [2.01]])
        assert not self.s.check_discontinuous([v, [1.99]])

        v.u[0] = np.array([-1, -1, 1, 1, 1], dtype=np.float64)
        assert self.s.check_discontinuous([v, [1.0]])
        assert not self.s.check_discontinuous([v, [1.01]])

        v.u[0] = np.array([0, 0.01, 2, 2, 2], dtype=np.float64)
        assert not self.s.check_discontinuous([v, [0.0]])

        with pytest.raises(ValueError):
            self.s.check_discontinuous(v)

    def test_check_continuous(self):
        v = constant_guess(self.p, 1.0)
        v.x[0] = v.t_x
        assert self.s.check_continuous([v, [0.0]])

        v.u[0] = v.t_u * 2
        v.x[0] = v.t_x**2
        assert self.s.check_continuous([v, [0.0]])

        v.u[0][0] += 0.01
        assert not self.s.check_continuous([v, [0.0]])

        v.u[0] = v.t_u * 1.99
        assert not self.s.check_continuous([v, [0.0]])
