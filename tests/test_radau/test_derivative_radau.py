# Copyright (c) 2024 Yilin Zou
import numpy as np
import sympy as sp

from pockit.radau import System


# we assume the objective & constraints functions are correct and test derivatives


class TestDerivativeRadau:
    s = System(2)
    p = s.new_phase(1, 1)
    p.set_dynamics([p.x[0] * sp.cos(s.s[0]) / p.u[0] + p.t**2])
    p.set_boundary_condition([0], [sp.cos(s.s[0] * 0.1)], None, 3 * sp.sin(s.s[1]))
    p.set_integral(
        [
            sp.cos(p.x[0]) * p.u[0]
            + 2 * p.x[0] * sp.cos(s.s[0])
            + 3 * sp.cos(p.x[0]) * p.t
            + 4 * p.u[0] * sp.cos(s.s[0])
            + 5 * sp.cos(p.u[0]) * p.t
            + 6 * s.s[1] * sp.cos(p.t),
            6 * sp.cos(p.x[0]) * p.u[0]
            + 5 * p.x[0] * sp.cos(s.s[0])
            + 4 * sp.cos(p.x[0]) * p.t
            + 3 * p.u[0] * sp.cos(s.s[0])
            + 2 * sp.cos(p.u[0]) * p.t
            + s.s[1] * sp.cos(p.t),
        ]
    )
    p.set_phase_constraint(
        [p.t - p.x[0] * p.u[0] * s.s[0] * s.s[1], p.x[0]], [0, 0], [0, 1]
    )
    p.set_discretization([0, 0.2, 1], [3, 4])
    s.set_phase([p])
    s.set_objective((p.I[0] + p.I[1] + s.s[0]) ** 2)
    s.set_system_constraint([(s.s[0] + 1) ** 2, s.s[1] / 2 * p.I[0]], [0, 0], [0, 0])
    assert p.L == 17
    x = np.arange(19, dtype=np.float64) / 10 + 1
    assert len(s.constraints(x)) == 17

    def test_gradient(self):
        sym = self.s.gradient(self.x)

        fd = np.zeros(19)
        eps = 1e-6
        for i in range(19):
            self.x[i] += eps
            fp = self.s.objective(self.x)
            self.x[i] -= 2 * eps
            fm = self.s.objective(self.x)
            self.x[i] += eps
            fd[i] = (fp - fm) / (2 * eps)

        for i in range(19):
            if not np.allclose(sym[i], fd[i]):
                print(i, sym[i], fd[i])
        assert np.allclose(sym, fd)

    def test_jacobian(self):
        fd = np.zeros((17, 19))
        eps = 1e-6
        for i in range(19):
            self.x[i] += eps
            fp = self.s.constraints(self.x)
            self.x[i] -= 2 * eps
            fm = self.s.constraints(self.x)
            self.x[i] += eps
            fd[:, i] = (fp - fm) / (2 * eps)

        sym = np.zeros((17, 19))
        data = self.s.jacobian(self.x)
        row, col = self.s.jacobianstructure()
        for r_, c_, d in zip(row, col, data):
            sym[r_, c_] += d

        for r_ in range(17):
            for c_ in range(19):
                if not np.allclose(sym[r_, c_], fd[r_, c_]):
                    print(r_, c_, sym[r_, c_], fd[r_, c_])
        assert np.allclose(sym, fd)

    def test_hessian_objective(self):
        fd = np.zeros((19, 19))
        eps = 2e-3
        for i in range(19):
            for j in range(i + 1):
                self.x[i] += eps
                self.x[j] += eps
                fpp = self.s.objective(self.x)
                self.x[i] -= 2 * eps
                fmp = self.s.objective(self.x)
                self.x[j] -= 2 * eps
                fmm = self.s.objective(self.x)
                self.x[i] += 2 * eps
                fpm = self.s.objective(self.x)
                self.x[i] -= eps
                self.x[j] += eps
                fd[i, j] = (fpp - fpm - fmp + fmm) / eps / eps / 4

        sym = np.zeros((19, 19))
        data = self.s.hessian_o(self.x)
        row, col = self.s.hessianstructure_o()
        for r_, c_, d in zip(row, col, data):
            sym[r_, c_] += d

        for i in range(19):
            for j in range(19):
                if not np.allclose(sym[i, j], fd[i, j], atol=1e-4, rtol=1e-4):
                    print(i, j, sym[i, j], fd[i, j])
        assert np.allclose(sym, fd, atol=1e-4, rtol=1e-4)

    def test_hessian_constraints(self):
        for c_ in range(17):
            fd = np.zeros((19, 19))
            eps = 2e-3
            for i in range(19):
                for j in range(i + 1):
                    self.x[i] += eps
                    self.x[j] += eps
                    fpp = self.s.constraints(self.x)[c_]
                    self.x[i] -= 2 * eps
                    fmp = self.s.constraints(self.x)[c_]
                    self.x[j] -= 2 * eps
                    fmm = self.s.constraints(self.x)[c_]
                    self.x[i] += 2 * eps
                    fpm = self.s.constraints(self.x)[c_]
                    self.x[i] -= eps
                    self.x[j] += eps
                    fd[i, j] = (fpp - fpm - fmp + fmm) / eps / eps / 4

            sym = np.zeros((19, 19))
            fct_c = np.zeros(17, dtype=np.float64)
            fct_c[c_] = 1.0
            data = self.s.hessian(self.x, fct_c, 0.0)
            row, col = self.s.hessianstructure()
            for r_, c_, d in zip(row, col, data):
                sym[r_, c_] += d
            for i in range(19):
                for j in range(19):
                    if not np.allclose(sym[i, j], fd[i, j], atol=1e-4, rtol=1e-4):
                        print(c_, i, j, sym[i, j], fd[i, j])
            assert np.allclose(sym, fd, atol=1e-4, rtol=1e-4)
