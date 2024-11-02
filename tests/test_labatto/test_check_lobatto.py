# Copyright (c) 2024 Yilin Zou
import pytest

from pockit.lobatto import System, constant_guess


class TestCheckLobatto:
    s = System(1)
    p = s.new_phase(1, 1)
    p.set_dynamics([p.u[0]])
    p.set_boundary_condition([None], [None], None, None)
    p.set_phase_constraint([p.u[0] + p.s[0]], [0.0], [2.0], [True])
    p.set_discretization([0, 0.1, 1], [3, 4])
    s.set_phase([p])
    s.set_objective(s.s[0])

    def test_check_discontinuous(self):
        v = constant_guess(self.p, 0.0)
        with pytest.raises(NotImplementedError):
            self.s.check_discontinuous([v, [2.0]])

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
