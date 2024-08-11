from typing import Iterable, Self
from abc import ABC, abstractmethod

import sympy as sp

from .autoupdate import AutoUpdate
from .easyderiv import *
from .fastfunc import FastFunc
from .phasebase import PhaseBase, BcType
from .vectypes import *
from .variablebase import VariableBase


@nb.njit("int32[:](int32[:], int32, int32)")
def _translate_index(index, l_p, r_s):
    index_2 = np.empty_like(index)
    for i in range(len(index)):
        if index[i] >= 0:
            index_2[i] = index[i] + l_p
        else:
            index_2[i] = index[i] + r_s
    return index_2


@nb.njit
def _translate_value(
    V_front,
    V_middle,
    V_back,
    w_m,
    index_front,
    index_middle_0,
    index_middle_1,
    index_back,
):
    V = ListJit.empty_list(nb.float64[:])
    if index_front:
        for V_ in V_front:
            V.append(V_ * w_m[0])
    for V_ in V_middle:
        V.append(V_ * w_m[index_middle_0:index_middle_1])
    if index_back:
        for V_ in V_back:
            V.append(V_ * w_m[-1])
    return V


class SystemBase(ABC):
    """A system is the higher level objective of a multiple-phase optimal
    control problem."""

    def __init__(
        self,
        static_parameter: int | list[str],
        simplify: bool = False,
        parallel: bool = False,
        fastmath: bool = False,
    ) -> None:
        r"""Initialize a system with given static parameters.

        If ``static_parameter`` is an integer, the names are generated automatically as :math:`s_0, s_1, \dots, s_{n-1}`.

        If ``simplify`` is ``True``, every symbolic expression will be simplified (by :func:`sympy.simplify`) before
        being compiled. This will slow down the speed of compilation.

        If ``parallel`` is ``True``, the ``parallel`` flag will be passed to the Numba JIT compiler,
        which will generate parallel code for multicore CPUs.
        This will slow down the speed of compilation and sometimes the speed of execution.

        If ``fastmath`` is ``True``, the ``fastmath`` flag will be passed to the Numba JIT compiler,
        see [Numba](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#fastmath)
        and [LLVM](https://llvm.org/docs/LangRef.html#fast-math-flags) documentations for details.

        Args:
            static_parameter: Number of static parameters or list of static parameter names.
            simplify: Whether to use Sympy to simplify :class:`sympy.Expr` before compilation.
            parallel: Whether to use Numba ``parallel`` mode.
            fastmath: Whether to use Numba ``fastmath`` mode.
        """
        if isinstance(static_parameter, int):
            self._num_static_parameter = static_parameter
            self._name_static_parameter = [f"s_{i}" for i in range(static_parameter)]
        elif isinstance(static_parameter, list):
            self._name_static_parameter = static_parameter
            self._num_static_parameter = len(static_parameter)
        else:
            raise ValueError("static_parameter must be int or list of str")

        self._symbol_static_parameter = [
            sp.Symbol(name) for name in self._name_static_parameter
        ]
        self._symbols = self._symbol_static_parameter

        self._identifier_phase = 0

        self._simplify = simplify
        self._parallel = parallel
        self._fastmath = fastmath
        self._compile_parameters = simplify, parallel, fastmath

        # phase: 0
        # objective: 1
        # system_constraint: 2
        # lr_phase: 3
        # symbols: 4
        # system_constraint_extra: 5
        # system_constraint_all: 6
        # node_basic: 7
        # structure_objective: 8
        # structure_system_constraint: 9
        # index_objective: 10
        # index_system_constraint: 11
        # index_constraint: 12
        self._auto_update = AutoUpdate(
            13,
            [
                self._update_lr_phase,
                self._update_symbols,
                self._update_system_constraint_extra,
                self._update_system_constraint_all,
                self._update_node_basic,
                self._update_structure_objective,
                self._update_structure_system_constraint,
                self._update_index_objective,
                self._update_index_system_constraint,
                self._update_index_constraint,
                self._update_index_all,
                self._update_bound,
            ],
        )
        self._auto_update.set_dependency(0, [0])  # _update_lr_phase
        self._auto_update.set_dependency(1, [0])  # _update_symbols
        self._auto_update.set_dependency(2, [0])  # _update_system_constraint_extra
        self._auto_update.set_dependency(3, [2, 5])  # _update_system_constraint_all
        self._auto_update.set_dependency(4, [0, 3])  # _update_node_basic
        self._auto_update.set_dependency(5, [0, 1, 7])  # _update_structure_objective
        self._auto_update.set_dependency(
            6, [0, 6, 7]
        )  # _update_structure_system_constraint
        self._auto_update.set_dependency(7, [8])  # _update_index_objective
        self._auto_update.set_dependency(8, [9])  # _update_index_system_constraint
        self._auto_update.set_dependency(9, [0, 11])  # _update_index_constraint
        self._auto_update.set_dependency(10, [10, 12])  # _update_index_all
        self._auto_update.set_dependency(11, [0, 6])  # _update_bound

        self._phase_set = False
        self._objective_set = False  # user must set an objective function
        self._system_constraint_set = False
        self.set_phase([])  # no phase is a valid system and is set by default
        self.set_system_constraint(
            [], np.array([]), np.array([])
        )  # no system constraint by default

    def new_phase(self, state: int | list[str], control: int | list[str]) -> PhaseBase:
        """Create a new phase for the given system.

        This method is recommended to create a phase rather than using ``Phase``'s constructor directly,
        because it automatically set the ``symbol_static_parameter``, ``identifier`` and flags of the phase.

        Args:
            state: Number of state variables or list of state variable names.
            control: Number of control variables or list of control variable names.

        Returns:
            A new phase, with ``symbol_static_parameter``, ``identifier`` and flags automatically set.
        """
        self._identifier_phase += 1
        return self._class_phase(
            self._identifier_phase - 1,
            state,
            control,
            self._symbol_static_parameter,
            *self._compile_parameters,
        )

    def set_phase(self, phase: list[PhaseBase]) -> Self:
        """Set the phases of the system.

        Args:
            phase: List of ``Phase`` s of the system.
        """
        for i, p in enumerate(phase):
            if not p.ok:
                raise ValueError(
                    f"Dynamics, boundary conditions, "
                    f"or discretization scheme of phase {i} are not fully set"
                )
        self._phase = phase
        self._num_phase = len(phase)

        self._auto_update.update(0)
        self._phase_set = True
        return self

    def set_objective(self, objective: float | sp.Expr) -> Self:
        """Set the objective of the system.

        Args:
            objective: Objective of the system composed with I and s.
        """
        self._expr_objective = sp.sympify(objective)
        self._func_objective = FastFunc(
            self._expr_objective,
            self._symbols,
            *self._compile_parameters,
        )

        self._auto_update.update(1)
        self._objective_set = True
        return self

    def set_system_constraint(
        self,
        system_constraint: list[sp.Expr],
        lower_bound: Iterable[float],
        upper_bound: Iterable[float],
    ) -> Self:
        """Set the system constraints of the system.

        For equality constraints, set the corresponding entry of ``lower_bounds`` and ``upper_bounds`` to the same value.
        For one-sided inequality constraints, set the corresponding entry of ``lower_bounds`` or ``upper_bounds``
        to ``-inf`` or ``inf``.

        Args:
            system_constraint: List of system constraints composed with I and s.
            lower_bound: Iterable of lower bounds of system constraints.
            upper_bound: Iterable of upper bounds of system constraints.
        """
        lower_bound = list(lower_bound)
        upper_bound = list(upper_bound)
        if not len(system_constraint) == len(lower_bound) == len(upper_bound):
            raise ValueError(
                "system_constraint, lower_bound and upper_bound must have the same length"
            )
        self._system_constraint_user = system_constraint
        self._system_constraint_user_lower_bound = lower_bound
        self._system_constraint_user_upper_bound = upper_bound
        self._auto_update.update(2)
        self._system_constraint_set = True
        return self

    def update(self) -> None:
        """Update the system after changing any phase of the system."""
        self._auto_update.update(0)

    def _update_lr_phase(self) -> None:
        if self._num_phase == 0:
            self.l_p = np.array([], dtype=np.int32)
            self.r_p = np.array([], dtype=np.int32)
            self.l_i = np.array([], dtype=np.int32)
            self.r_i = np.array([], dtype=np.int32)
            self.l_s = 0
            self.r_s = self._num_static_parameter
        else:
            l_p = [0]
            r_p = []
            l_i = [0]
            r_i = []
            for p in self.p:
                r_p.append(l_p[-1] + p.L)
                l_p.append(r_p[-1])
                r_i.append(l_i[-1] + p.n_I)
                l_i.append(r_i[-1])
            self.l_p = np.array(l_p[:-1], dtype=np.int32)
            self.r_p = np.array(r_p, dtype=np.int32)
            self.l_i = np.array(l_i[:-1], dtype=np.int32)
            self.r_i = np.array(r_i, dtype=np.int32)
            self.l_s = r_p[-1]
            self.r_s = self.l_s + self._num_static_parameter
        self._auto_update.update(3)

    def _update_symbols(self) -> None:
        self._symbols = []
        for i, p in enumerate(self.p):
            self._symbols += p.I
        self._symbols += self._symbol_static_parameter.copy()
        self._num_symbol = len(self._symbols)
        self._auto_update.update(4)

    def _update_system_constraint_extra(self) -> None:
        system_constraint = []
        lower_bound = []
        upper_bound = []
        for p in self.p:
            for i, lb, ub in p._variable_bounds_phase:
                if i < p.n_x and p.info_bc_0[i].t == BcType.FUNC:
                    system_constraint.append(p.bc_0[i])
                    lower_bound.append(lb)
                    upper_bound.append(ub)
                if i < p.n_x and p.info_bc_f[i].t == BcType.FUNC:
                    system_constraint.append(p.bc_f[i])
                    lower_bound.append(lb)
                    upper_bound.append(ub)
            for lb, ub in p._time_bounds_phase:
                if p.info_t_0.t == BcType.FUNC:
                    system_constraint.append(p.t_0)
                    lower_bound.append(lb)
                    upper_bound.append(ub)
                if p.info_t_f.t == BcType.FUNC:
                    system_constraint.append(p.t_f)
                    lower_bound.append(lb)
                    upper_bound.append(ub)
        self._system_constraint_extra = system_constraint
        self._system_constraint_extra_lower_bound = lower_bound
        self._system_constraint_extra_upper_bound = upper_bound
        self._auto_update.update(5)

    def _update_system_constraint_all(self):
        system_constraint = self._system_constraint_user + self._system_constraint_extra
        lower_bound = (
            self._system_constraint_user_lower_bound
            + self._system_constraint_extra_lower_bound
        )
        upper_bound = (
            self._system_constraint_user_upper_bound
            + self._system_constraint_extra_upper_bound
        )

        self._static_parameter_bounds_system = []
        self._expr_system_constraint = []
        lower_bound_system_constraint = []
        upper_bound_system_constraint = []
        for c, lb, ub in zip(system_constraint, lower_bound, upper_bound):
            if c.is_symbol and c in self.s:
                self._static_parameter_bounds_system.append((self.s.index(c), lb, ub))
            else:
                self._expr_system_constraint.append(sp.sympify(c))
                lower_bound_system_constraint.append(lb)
                upper_bound_system_constraint.append(ub)

        self._func_system_constraint = [
            FastFunc(c, self._symbols, *self._compile_parameters)
            for c in self._expr_system_constraint
        ]
        self._num_system_constraint = len(self._expr_system_constraint)
        self._lower_bound_system_constraint = np.array(lower_bound_system_constraint)
        self._upper_bound_system_constraint = np.array(upper_bound_system_constraint)

        self._auto_update.update(6)

    def _update_node_basic(self) -> None:
        self._node_static_parameter = [
            Node() for _ in range(self._num_static_parameter)
        ]
        for i in range(self._num_static_parameter):
            self._node_static_parameter[i].set_G_i(
                [np.array([self.l_s + i], dtype=np.int32)]
            )
            self._node_static_parameter[i].set_G([np.array([1.0], dtype=np.float64)])

        self._node_integral = []
        for p_, p in enumerate(self.p):
            for i in range(p.n_I):
                node = Node()
                if p.index_mstage.f:
                    for G_i_ in p._node_integral_front[i].G_i:
                        node.G_i.append(_translate_index(G_i_, self.l_p[p_], self.r_s))
                    for row_ in p._node_integral_front[i].H_i_row:
                        node.H_i_row.append(
                            _translate_index(row_, self.l_p[p_], self.r_s)
                        )
                    for col_ in p._node_integral_front[i].H_i_col:
                        node.H_i_col.append(
                            _translate_index(col_, self.l_p[p_], self.r_s)
                        )
                for G_i_ in p._node_integral_middle[i].G_i:
                    node.G_i.append(_translate_index(G_i_, self.l_p[p_], self.r_s))
                for row_ in p._node_integral_middle[i].H_i_row:
                    node.H_i_row.append(_translate_index(row_, self.l_p[p_], self.r_s))
                for col_ in p._node_integral_middle[i].H_i_col:
                    node.H_i_col.append(_translate_index(col_, self.l_p[p_], self.r_s))
                if p.index_mstage.b:
                    for G_i_ in p._node_integral_back[i].G_i:
                        node.G_i.append(_translate_index(G_i_, self.l_p[p_], self.r_s))
                    for row_ in p._node_integral_back[i].H_i_row:
                        node.H_i_row.append(
                            _translate_index(row_, self.l_p[p_], self.r_s)
                        )
                    for col_ in p._node_integral_back[i].H_i_col:
                        node.H_i_col.append(
                            _translate_index(col_, self.l_p[p_], self.r_s)
                        )
                self._node_integral.append(node)

        self._node_basic = self._node_integral + self._node_static_parameter
        self._auto_update.update(7)

    def _update_structure_objective(self) -> None:
        free_symbols_objective = self._expr_objective.free_symbols
        which_objective = [np.zeros(p.n_I, dtype=bool) for p in self.p]
        for p_, p in enumerate(self.p):
            for i_, i in enumerate(p.I):
                if i in free_symbols_objective:
                    which_objective[p_][i_] = True
        self._which_objective = which_objective

        self._node_objective = Node()
        self._node_objective.args = self._node_basic
        self._node_objective.g_i = self._func_objective.G_index
        self._node_objective.h_i_row = self._func_objective.H_index_row
        self._node_objective.h_i_col = self._func_objective.H_index_col

        forward_gradient_i([self._node_objective])
        forward_hessian_system_i([self._node_objective])
        self._auto_update.update(8)

    def _update_structure_system_constraint(self) -> None:
        which_system_constraint = [np.zeros(p.n_I, dtype=bool) for p in self.p]
        for c_, c in enumerate(self._expr_system_constraint):
            free_symbols_system_constraint = c.free_symbols
            for p_, p in enumerate(self.p):
                for i_, i in enumerate(p.I):
                    if i in free_symbols_system_constraint:
                        which_system_constraint[p_][i_] = True
        self._which_system_constraint = which_system_constraint

        self._node_system_constraint = []
        for c_, c in enumerate(self._func_system_constraint):
            node = Node()
            node.args = self._node_basic
            node.g_i = c.G_index
            node.h_i_row = c.H_index_row
            node.h_i_col = c.H_index_col
            self._node_system_constraint.append(node)

        forward_gradient_i(self._node_system_constraint)
        forward_hessian_system_i(self._node_system_constraint)
        self._auto_update.update(9)

    def _update_index_objective(self) -> None:
        grad_col = self._node_objective.G_i

        hess_row = self._node_objective.H_i_row
        hess_col = self._node_objective.H_i_col

        self._grad_objective_col = (
            np.concatenate(grad_col) if grad_col else np.array([], dtype=np.int32)
        )
        self._hess_objective_row = (
            np.concatenate(hess_row) if hess_row else np.array([], dtype=np.int32)
        )
        self._hess_objective_col = (
            np.concatenate(hess_col) if hess_col else np.array([], dtype=np.int32)
        )
        self._auto_update.update(10)

    def _update_index_system_constraint(self) -> None:
        jac_row = []
        jac_col = []
        for c_, node in enumerate(self._node_system_constraint):
            for g_i in node.G_i:
                jac_row.append(np.full(len(g_i), c_, dtype=np.int32))
                jac_col.append(g_i)

        hess_row = []
        hess_col = []
        for c_, node in enumerate(self._node_system_constraint):
            hess_row.extend(node.H_i_row)
            hess_col.extend(node.H_i_col)

        self._jac_system_constraint_row = (
            np.concatenate(jac_row) if jac_row else np.array([], dtype=np.int32)
        )
        self._jac_system_constraint_col = (
            np.concatenate(jac_col) if jac_col else np.array([], dtype=np.int32)
        )
        self._hess_system_constraint_row = (
            np.concatenate(hess_row) if hess_row else np.array([], dtype=np.int32)
        )
        self._hess_system_constraint_col = (
            np.concatenate(hess_col) if hess_col else np.array([], dtype=np.int32)
        )
        self._auto_update.update(11)

    def _update_index_constraint(self) -> None:
        jac_row = [self._jac_system_constraint_row]
        jac_col = [self._jac_system_constraint_col]
        c_ = self._num_system_constraint
        for p_, p in enumerate(self.p):
            jac_row.append(c_ + p._jac_dynamic_constraint_row)
            jac_col.append(
                _translate_index(p._jac_dynamic_constraint_col, self.l_p[p_], self.r_s)
            )
            c_ += p.r_d[-1]
            jac_row.append(c_ + p._jac_phase_constraint_row)
            jac_col.append(
                _translate_index(p._jac_phase_constraint_col, self.l_p[p_], self.r_s)
            )
            c_ += p.n_c * p.L_m
        self._jac_constraint_row = (
            np.concatenate(jac_row) if jac_row else np.array([], dtype=np.int32)
        )
        self._jac_constraint_col = (
            np.concatenate(jac_col) if jac_col else np.array([], dtype=np.int32)
        )

        hess_row = [self._hess_system_constraint_row]
        hess_col = [self._hess_system_constraint_col]
        for p_, p in enumerate(self.p):
            hess_row.append(
                _translate_index(p._hess_dynamic_constraint_row, self.l_p[p_], self.r_s)
            )
            hess_col.append(
                _translate_index(p._hess_dynamic_constraint_col, self.l_p[p_], self.r_s)
            )
            hess_row.append(
                _translate_index(p._hess_phase_constraint_row, self.l_p[p_], self.r_s)
            )
            hess_col.append(
                _translate_index(p._hess_phase_constraint_col, self.l_p[p_], self.r_s)
            )
        self._hess_constraint_row = (
            np.concatenate(hess_row) if hess_row else np.array([], dtype=np.int32)
        )
        self._hess_constraint_col = (
            np.concatenate(hess_col) if hess_col else np.array([], dtype=np.int32)
        )
        self._auto_update.update(12)

    def _update_index_all(self) -> None:
        self._hess_all_row = np.concatenate(
            [self._hess_objective_row, self._hess_constraint_row]
        )
        self._hess_all_col = np.concatenate(
            [self._hess_objective_col, self._hess_constraint_col]
        )

    def _update_bound(self) -> None:
        lower_bound_static_parameter = np.full(
            self._num_static_parameter, -np.inf, dtype=np.float64
        )
        upper_bound_static_parameter = np.full(
            self._num_static_parameter, np.inf, dtype=np.float64
        )
        for p in self.p:
            for i, lb, ub in p.s_b:
                lower_bound_static_parameter[i] = np.maximum(
                    lower_bound_static_parameter[i], lb
                )
                upper_bound_static_parameter[i] = np.minimum(
                    upper_bound_static_parameter[i], ub
                )
        for i, lb, ub in self._static_parameter_bounds_system:
            lower_bound_static_parameter[i] = np.maximum(
                lower_bound_static_parameter[i], lb
            )
            upper_bound_static_parameter[i] = np.minimum(
                upper_bound_static_parameter[i], ub
            )
        self._lower_bound_variable = np.concatenate(
            [p.v_lb for p in self.p] + [np.array(lower_bound_static_parameter)]
        )
        self._upper_bound_variable = np.concatenate(
            [p.v_ub for p in self.p] + [np.array(upper_bound_static_parameter)]
        )

        lower_bound_constraint = [self._lower_bound_system_constraint]
        upper_bound_constraint = [self._upper_bound_system_constraint]
        for p in self.p:
            lower_bound_constraint.append(np.zeros(p.r_d[-1], dtype=np.float64))
            upper_bound_constraint.append(np.zeros(p.r_d[-1], dtype=np.float64))
            lower_bound_constraint.append(np.repeat(p.c_lb, p.L_m))
            upper_bound_constraint.append(np.repeat(p.c_ub, p.L_m))
        self._lower_bound_constraint = np.concatenate(lower_bound_constraint)
        self._upper_bound_constraint = np.concatenate(upper_bound_constraint)

    def _value_basic(self, which, x) -> VecFloat:
        v = np.empty(self._num_symbol, dtype=np.float64)
        for p_, p in enumerate(self.p):
            v[self.l_i[p_] : self.r_i[p_]] = p._value_integral(
                which[p_], x[self.l_p[p_] : self.r_p[p_]], x[self.l_s : self.r_s]
            )
        v[self.r_i[-1] :] = x[self.l_s : self.r_s]
        return v

    def objective(self, x: VecFloat) -> np.float64:
        """The objective function of the discretized optimization problem."""
        vb = self._value_basic(self._which_objective, x)
        return self._func_objective.F(vb, 1)[0]

    def _system_constraint(self, x: VecFloat) -> VecFloat:
        vb = self._value_basic(self._which_system_constraint, x)
        return np.array(
            [f_c.F(vb, 1)[0] for f_c in self._func_system_constraint], dtype=np.float64
        )

    def constraints(self, x: VecFloat) -> VecFloat:
        """Constraint functions of the discretized optimization problem."""
        cons = []
        cons.append(self._system_constraint(x))

        s = x[self.l_s : self.r_s]
        for p_, p in enumerate(self.p):
            x_ = x[self.l_p[p_] : self.r_p[p_]]
            cons.append(p._value_dynamic_constraint(x_, s))
            cons.append(p._value_phase_constraint(x_, s))

        return np.concatenate(cons) if cons else np.array([], dtype=np.float64)

    def _grad_basic(self, which, x):
        s = x[self.l_s : self.r_s]
        n_ = 0
        for p_, p in enumerate(self.p):
            x_ = x[self.l_p[p_] : self.r_p[p_]]
            p._grad_basic(x_, s)
            p._grad_integral(which[p_], x_, s)
            for i_ in range(p.n_I):
                if which[p_][i_]:
                    self._node_integral[n_ + i_].G = _translate_value(
                        p._node_integral_front[i_].G,
                        p._node_integral_middle[i_].G,
                        p._node_integral_back[i_].G,
                        p.w_m,
                        p.index_mstage.f,
                        p.index_mstage.l_m,
                        p.index_mstage.r_m,
                        p.index_mstage.b,
                    )
            n_ += p.n_I

    def gradient(self, x: VecFloat) -> VecFloat:
        """Gradient of the objective function of the discretized optimization
        problem."""
        self._grad_basic(self._which_objective, x)
        vb = self._value_basic(self._which_objective, x)
        self._node_objective.g = self._func_objective.G(vb, 1)
        forward_gradient_v([self._node_objective])

        grad = np.zeros(self.L, dtype=np.float64)
        for i, v in zip(self._node_objective.G_i, self._node_objective.G):
            np.add.at(grad, i, v)
        return grad

    def _jac_system_constraint(self, x: VecFloat) -> VecFloat:
        """Jacobian of the system constraints of the discretized optimization
        problem."""
        self._grad_basic(self._which_system_constraint, x)
        vb = self._value_basic(self._which_system_constraint, x)
        for i in range(self._num_system_constraint):
            self._node_system_constraint[i].g = self._func_system_constraint[i].G(vb, 1)
        forward_gradient_v(self._node_system_constraint)

        jac = [G_ for node in self._node_system_constraint for G_ in node.G]
        return np.concatenate(jac) if jac else np.array([], dtype=np.float64)

    def jacobianstructure(self) -> tuple[VecInt, VecInt]:
        """Coordinates of the Jacobian of the constraint functions of the
        discretized optimization problem."""
        return self._jac_constraint_row, self._jac_constraint_col

    def jacobian(self, x: VecFloat) -> VecFloat:
        """Jacobian of the constraint functions of the discretized optimization
        problem.

        Args:
            x: Vector of optimization variables.

        Returns:
            A plain 1D array, with coordinates given by :meth:`jacobianstructure`.
        """
        jac = [self._jac_system_constraint(x)]

        s = x[self.l_s : self.r_s]
        for p_, p in enumerate(self.p):
            x_ = x[self.l_p[p_] : self.r_p[p_]]
            jac.append(p._grad_dynamic_constraint(x_, s))
            jac.append(p._grad_phase_constraint(x_, s))
        return np.concatenate(jac) if jac else np.array([], dtype=np.float64)

    def _hess_basic(self, which, x):
        s = x[self.l_s : self.r_s]
        n_ = 0
        for p_, p in enumerate(self.p):
            x_ = x[self.l_p[p_] : self.r_p[p_]]
            p._hess_basic(x_, s)
            p._hess_integral(which[p_], x_, s)
            for i_ in range(p.n_I):
                if which[p_][i_]:
                    self._node_integral[n_ + i_].G = _translate_value(
                        p._node_integral_front[i_].G,
                        p._node_integral_middle[i_].G,
                        p._node_integral_back[i_].G,
                        p.w_m,
                        p.index_mstage.f,
                        p.index_mstage.l_m,
                        p.index_mstage.r_m,
                        p.index_mstage.b,
                    )
                    self._node_integral[n_ + i_].H = _translate_value(
                        p._node_integral_front[i_].H,
                        p._node_integral_middle[i_].H,
                        p._node_integral_back[i_].H,
                        p.w_m,
                        p.index_mstage.f,
                        p.index_mstage.l_m,
                        p.index_mstage.r_m,
                        p.index_mstage.b,
                    )
            n_ += p.n_I

    def hessianstructure_o(self) -> tuple[VecInt, VecInt]:
        """Coordinates of the Hessian of the objective function of the
        discretized optimization problem.

        Only includes entries in the lower triangle of the Hessian
        matrix.
        """
        return self._hess_objective_row, self._hess_objective_col

    def hessian_o(self, x: VecFloat) -> VecFloat:
        """Hessian of the objective function of the discretized optimization
        problem.

        Args:
            x: Vector of optimization variables.

        Returns:
            A plain 1D array, with coordinates given by :meth:`hessianstructure_o`.
            Only includes entries in the lower triangle of the Hessian matrix.
        """
        self._hess_basic(self._which_objective, x)
        vb = self._value_basic(self._which_objective, x)
        self._node_objective.g = self._func_objective.G(vb, 1)
        self._node_objective.h = self._func_objective.H(vb, 1)
        forward_gradient_v([self._node_objective])
        forward_hessian_system_v([self._node_objective])
        return (
            np.concatenate(self._node_objective.H)
            if self._node_objective.H
            else np.array([], dtype=np.float64)
        )

    def hessianstructure_c(self) -> tuple[VecInt, VecInt]:
        """Coordinates of the Hessian of the constraint functions of the
        discretized optimization problem.

        Only includes entries in the lower triangle of the Hessian
        matrix.
        """
        return self._hess_constraint_row, self._hess_constraint_col

    def _hess_system_constraint(self, x: VecFloat, fct_c: VecFloat) -> VecFloat:
        self._hess_basic(self._which_system_constraint, x)
        vb = self._value_basic(self._which_system_constraint, x)
        for i in range(self._num_system_constraint):
            self._node_system_constraint[i].g = self._func_system_constraint[i].G(vb, 1)
            self._node_system_constraint[i].h = self._func_system_constraint[i].H(vb, 1)
        forward_gradient_v(self._node_system_constraint)
        forward_hessian_system_v(self._node_system_constraint)

        hess = []
        for i in range(self._num_system_constraint):
            hess_ = (
                np.concatenate(self._node_system_constraint[i].H)
                if self._node_system_constraint[i].H
                else np.array([], dtype=np.float64)
            )
            hess.append(hess_ * fct_c[i])
        return np.concatenate(hess) if hess else np.array([], dtype=np.float64)

    def hessian_c(self, x: VecFloat, fct_c: VecFloat) -> VecFloat:
        """Sum of Hessian of the constraint functions of the discretized
        optimization problem with factor ``fct_c``.

        Args:
            x: Vector of optimization variables.
            fct_c: Factors (Lagrange multipliers) for the constraints.

        Returns:
            A plain 1D array, with coordinates given by :meth:`hessianstructure_c`.
            Only includes entries in the lower triangle of the Hessian matrix.
        """
        hess = [self._hess_system_constraint(x, fct_c[: self._num_system_constraint])]

        s = x[self.l_s : self.r_s]
        f_ = self._num_system_constraint
        for p_, p in enumerate(self.p):
            x_ = x[self.l_p[p_] : self.r_p[p_]]
            hess.append(p._hess_dynamic_constraint(x_, s, fct_c[f_ : f_ + p.r_d[-1]]))
            f_ += p.r_d[-1]
            hess.append(p._hess_phase_constraint(x_, s, fct_c[f_ : f_ + p.n_c * p.L_m]))
            f_ += p.n_c * p.L_m

        return np.concatenate(hess) if hess else np.array([], dtype=np.float64)

    def hessianstructure(self) -> tuple[VecInt, VecInt]:
        """Coordinates of the Hessian of the Lagrangian of the discretized
        optimization problem.

        Only includes entries in the lower triangle of the Hessian
        matrix.
        """
        return self._hess_all_row, self._hess_all_col

    def hessian(self, x: VecFloat, fct_c: VecFloat, fct_o: float) -> VecFloat:
        """Hessian of the Lagrangian of the discretized optimization problem
        with factors ``fct_c`` for constraints and ``fct_o`` for the objective.

        Args:
            x: Vector of optimization variables.
            fct_c: Factors (Lagrange multipliers) for the constraints.
            fct_o: Factor (Lagrange multiplier) for the objective.

        Returns:
            A plain 1D array, with coordinates given by :meth:`hessianstructure`.
            Only includes entries in the lower triangle of the Hessian matrix.
        """
        hessian_o = self.hessian_o(x) * fct_o
        hessian_c = self.hessian_c(x, fct_c)
        return np.concatenate([hessian_o, hessian_c])

    def check_continuous(
        self,
        value: VariableBase | list[VariableBase | Iterable[float]],
        absolute_tolerance_continuous: float = 1.0e-8,
        relative_tolerance_continuous: float = 1.0e-8,
        tolerance_mesh: float = 1.0e-4,
    ) -> bool:
        """Check the continuous error.

        Args:
            value: The variable to be checked. If the system has only one phase and no static variables, ``value`` can
                be a single `Variable` object. Otherwise, ``value`` should be a list of
                ``Variable`` objects, one for each ``Phase``, followed by an array
                as values of static variables.
            absolute_tolerance_continuous: Absolute tolerance for continuous error.
            relative_tolerance_continuous: Relative tolerance for continuous error.
            tolerance_mesh: Skip the check if the mesh width is smaller than this value.

        Returns:
            ``True`` if the error is within the tolerance, ``False`` otherwise.
        """
        if not self.ok:
            raise ValueError("system is not fully configured")

        value_is_variable = isinstance(value, VariableBase)
        if value_is_variable:
            value = [value]

        if not self._num_static_parameter and len(value) != self._num_phase:
            raise ValueError("len(value) must be equal to the number of phases")
        elif self._num_static_parameter and len(value) != self._num_phase + 1:
            raise ValueError(
                "len(value) must be equal to the number of phases + 1 (for static variables)"
            )

        ok = []
        if not self._num_static_parameter:
            for p_, v_ in zip(self.p, value):
                ok.append(
                    p_.check_continuous(
                        v_,
                        absolute_tolerance_continuous=absolute_tolerance_continuous,
                        relative_tolerance_continuous=relative_tolerance_continuous,
                        tolerance_mesh=tolerance_mesh,
                    )
                )
        else:
            for p_, v_ in zip(self.p, value[:-1]):
                ok.append(
                    p_.check_continuous(
                        v_,
                        np.array(list(value[-1]), dtype=np.float64),
                        absolute_tolerance_continuous=absolute_tolerance_continuous,
                        relative_tolerance_continuous=relative_tolerance_continuous,
                        tolerance_mesh=tolerance_mesh,
                    )
                )

        return np.all(ok)

    def check_discontinuous(
        self,
        value: VariableBase | list[VariableBase | Iterable[float]],
        tolerance_discontinuous: float = 1.0e-3,
        tolerance_mesh: float = 1.0e-4,
    ) -> bool:
        """Check the discontinuous error.

        Args:
            value: The variable to be checked. If the system has only one phase and no static variables, ``value`` can
                be a single `Variable` object. Otherwise, ``value`` should be a list of
                ``Variable`` objects, one for each ``Phase``, followed by an array
                as values of static variables.
            tolerance_discontinuous: In each subinterval, after scaling to ``[0, 1]``, the bang-bang control functions
                should either be less than ``tolerance_discontinuous`` or greater than ``1 - tolerance_discontinuous``
                simultaneously.
            tolerance_mesh: Skip the check if the mesh width is smaller than this value.

        Returns:
            ``True`` if the error is within the tolerance, ``False`` otherwise.
        """
        if not self.ok:
            raise ValueError("system is not fully configured")

        value_is_variable = isinstance(value, VariableBase)
        if value_is_variable:
            value = [value]

        if not self._num_static_parameter and len(value) != self._num_phase:
            raise ValueError("len(value) must be equal to the number of phases")
        elif self._num_static_parameter and len(value) != self._num_phase + 1:
            raise ValueError(
                "len(value) must be equal to the number of phases + 1 (for static variables)"
            )

        ok = []
        if not self._num_static_parameter:
            for p_, v_ in zip(self.p, value):
                ok.append(
                    p_.check_discontinuous(
                        v_,
                        tolerance_discontinuous=tolerance_discontinuous,
                        tolerance_mesh=tolerance_mesh,
                    )
                )
        else:
            for p_, v_ in zip(self.p, value[:-1]):
                ok.append(
                    p_.check_discontinuous(
                        v_,
                        np.array(list(value[-1]), dtype=np.float64),
                        tolerance_discontinuous=tolerance_discontinuous,
                        tolerance_mesh=tolerance_mesh,
                    )
                )

        return np.all(ok)

    def check(
        self,
        value: VariableBase | list[VariableBase | Iterable[float]],
        absolute_tolerance_continuous: float = 1.0e-8,
        relative_tolerance_continuous: float = 1.0e-8,
        tolerance_discontinuous: float = 1.0e-3,
        tolerance_mesh: float = 1.0e-4,
    ) -> bool:
        """Check the continuous and discontinuous error.

        Args:
            value: The variable to be checked. If the system has only one phase and no static variables, ``value`` can
                be a single `Variable` object. Otherwise, ``value`` should be a list of
                ``Variable`` objects, one for each ``Phase``, followed by an array
                as values of static variables.
            absolute_tolerance_continuous: Absolute tolerance for continuous error.
            relative_tolerance_continuous: Relative tolerance for continuous error.
            tolerance_discontinuous: In each subinterval, after scaling to ``[0, 1]``, the bang-bang control functions
                should either be less than ``tolerance_discontinuous`` or greater than ``1 - tolerance_discontinuous``
                simultaneously.
            tolerance_mesh: Skip the check if the mesh width is smaller than this value.

        Returns:
            ``True`` if the error is within the tolerance, ``False`` otherwise.
        """
        return self.check_continuous(
            value,
            absolute_tolerance_continuous=absolute_tolerance_continuous,
            relative_tolerance_continuous=relative_tolerance_continuous,
            tolerance_mesh=tolerance_mesh,
        ) and self.check_discontinuous(
            value,
            tolerance_discontinuous=tolerance_discontinuous,
            tolerance_mesh=tolerance_mesh,
        )

    def refine_continuous(
        self,
        value: VariableBase | list[VariableBase | Iterable[float]],
        absolute_tolerance_continuous: float = 1.0e-8,
        relative_tolerance_continuous: float = 1.0e-8,
        num_point_min: int = 6,
        num_point_max: int = 12,
        mesh_length_min: float = 1.0e-3,
        mesh_length_max: float = 1,
    ) -> VariableBase | list[VariableBase | Iterable[float]]:
        """Adjust the mesh and the number of interpolation points to match the
        continuous error tolerance.

        Args:
            value: The variable to be checked. If the system has only one phase and no static variables, ``value`` can
                be a single `Variable` object. Otherwise, ``value`` should be a list of
                ``Variable`` objects, one for each ``Phase``, followed by an array
                as values of static variables.
            absolute_tolerance_continuous: Absolute tolerance for continuous error.
            relative_tolerance_continuous: Relative tolerance for continuous error.
            num_point_min: Minimum number of interpolation points.
            num_point_max: Maximum number of interpolation points.
            mesh_length_min: Minimum mesh length.
            mesh_length_max: Maximum mesh length.

        Returns:
            The ``Variable`` s interpolated to the new discretization scheme.
        """
        if not self.ok:
            raise ValueError("system is not fully configured")

        if self.check_continuous(
            value,
            absolute_tolerance_continuous=absolute_tolerance_continuous,
            relative_tolerance_continuous=relative_tolerance_continuous,
            tolerance_mesh=mesh_length_min,
        ):
            return value

        value_is_variable = isinstance(value, VariableBase)
        if value_is_variable:
            value = [value]

        value_adapt = []
        if not self._num_static_parameter:
            for p_, v_ in zip(self.p, value):
                p_.refine_continuous(
                    v_,
                    absolute_tolerance_continuous=absolute_tolerance_continuous,
                    relative_tolerance_continuous=relative_tolerance_continuous,
                    num_point_min=num_point_min,
                    num_point_max=num_point_max,
                    mesh_length_min=mesh_length_min,
                    mesh_length_max=mesh_length_max,
                )
                value_adapt.append(v_.adapt(p_))
        else:
            for p_, v_ in zip(self.p, value[:-1]):
                p_.refine_continuous(
                    v_,
                    np.array(list(value[-1]), dtype=np.float64),
                    absolute_tolerance_continuous=absolute_tolerance_continuous,
                    relative_tolerance_continuous=relative_tolerance_continuous,
                    num_point_min=num_point_min,
                    num_point_max=num_point_max,
                    mesh_length_min=mesh_length_min,
                    mesh_length_max=mesh_length_max,
                )
                value_adapt.append(v_.adapt(p_))

        self.update()

        if value_is_variable:
            return value_adapt[0]
        elif len(value_adapt) == len(value):
            return value_adapt
        else:
            return value_adapt + [value[-1]]

    def refine_discontinuous(
        self,
        value: VariableBase | list[VariableBase | Iterable[float]],
        tolerance_discontinuous: float = 1.0e-3,
        num_point_min: int = 6,
        num_point_max: int = 12,
        mesh_length_min: float = 1.0e-3,
        mesh_length_max: float = 1,
    ) -> VariableBase | list[VariableBase | Iterable[float]]:
        """Adjust the mesh and the number of interpolation points to match the
        discontinuous error tolerance.

        Args:
            value: The variable to be checked. If the system has only one phase and no static variables, ``value`` can
                be a single `Variable` object. Otherwise, ``value`` should be a list of
                ``Variable`` objects, one for each ``Phase``, followed by an array
                as values of static variables.
            tolerance_discontinuous: In each subinterval, after scaling to ``[0, 1]``, the bang-bang control functions
                should either be less than ``tolerance_discontinuous`` or greater than ``1 - tolerance_discontinuous``
                simultaneously.
            num_point_min: Minimum number of interpolation points.
            num_point_max: Maximum number of interpolation points.
            mesh_length_min: Minimum mesh length.
            mesh_length_max: Maximum mesh length.

        Returns:
            The ``Variable`` s interpolated to the new discretization scheme.
        """
        if not self.ok:
            raise ValueError("system is not fully configured")

        if self.check_discontinuous(
            value,
            tolerance_discontinuous=tolerance_discontinuous,
            tolerance_mesh=mesh_length_min,
        ):
            return value

        value_is_variable = isinstance(value, VariableBase)
        if value_is_variable:
            value = [value]

        value_adapt = []
        if not self._num_static_parameter:
            for p_, v_ in zip(self.p, value):
                p_.refine_discontinuous(
                    v_,
                    tolerance_discontinuous=tolerance_discontinuous,
                    num_point_min=num_point_min,
                    num_point_max=num_point_max,
                    mesh_length_min=mesh_length_min,
                    mesh_length_max=mesh_length_max,
                )
                value_adapt.append(v_.adapt(p_))
        else:
            for p_, v_ in zip(self.p, value[:-1]):
                p_.refine_discontinuous(
                    v_,
                    np.array(list(value[-1]), dtype=np.float64),
                    tolerance_discontinuous=tolerance_discontinuous,
                    num_point_min=num_point_min,
                    num_point_max=num_point_max,
                    mesh_length_min=mesh_length_min,
                    mesh_length_max=mesh_length_max,
                )
                value_adapt.append(v_.adapt(p_))

        self.update()

        if value_is_variable:
            return value_adapt[0]
        elif len(value_adapt) == len(value):
            return value_adapt
        else:
            return value_adapt + [value[-1]]

    def refine(
        self,
        value: VariableBase | list[VariableBase | Iterable[float]],
        absolute_tolerance_continuous: float = 1.0e-8,
        relative_tolerance_continuous: float = 1.0e-8,
        tolerance_discontinuous: float = 1.0e-3,
        num_point_min: int = 6,
        num_point_max: int = 12,
        mesh_length_min: float = 1.0e-3,
        mesh_length_max: float = 1,
    ) -> VariableBase | list[VariableBase | Iterable[float]]:
        """Adjust the mesh and the number of interpolation points to match the
        error tolerances.

        If the discontinuous error is not within the tolerance, refine for discontinuous error.
        Otherwise, if the continuous error is not within the tolerance, refine for continuous error.
        At most one of the continuous or discontinuous refinements will be performed.

        Args:
            value: The variable to be checked. If the system has only one phase and no static variables, ``value`` can
                be a single `Variable` object. Otherwise, ``value`` should be a list of
                ``Variable`` objects, one for each ``Phase``, followed by an array
                as values of static variables.
            absolute_tolerance_continuous: Absolute tolerance for continuous error.
            relative_tolerance_continuous: Relative tolerance for continuous error.
            tolerance_discontinuous: In each subinterval, after scaling to ``[0, 1]``, the bang-bang control functions
                should either be less than ``tolerance_discontinuous`` or greater than ``1 - tolerance_discontinuous``
                simultaneously.
            num_point_min: Minimum number of interpolation points.
            num_point_max: Maximum number of interpolation points.
            mesh_length_min: Minimum mesh length.
            mesh_length_max: Maximum mesh length.

        Returns:
            The ``Variable`` s interpolated to the new discretization scheme.
        """
        if not self.ok:
            raise ValueError("system is not fully configured")

        if self.check(
            value,
            absolute_tolerance_continuous=absolute_tolerance_continuous,
            relative_tolerance_continuous=relative_tolerance_continuous,
            tolerance_discontinuous=tolerance_discontinuous,
            tolerance_mesh=mesh_length_min,
        ):
            return value

        value_is_variable = isinstance(value, VariableBase)
        if value_is_variable:
            value = [value]

        value_adapt = []
        if not self._num_static_parameter:
            for p_, v_ in zip(self.p, value):
                p_.refine(
                    v_,
                    absolute_tolerance_continuous=absolute_tolerance_continuous,
                    relative_tolerance_continuous=relative_tolerance_continuous,
                    tolerance_discontinuous=tolerance_discontinuous,
                    num_point_min=num_point_min,
                    num_point_max=num_point_max,
                    mesh_length_min=mesh_length_min,
                    mesh_length_max=mesh_length_max,
                )
                value_adapt.append(v_.adapt(p_))
        else:
            for p_, v_ in zip(self.p, value[:-1]):
                p_.refine(
                    v_,
                    np.array(list(value[-1]), dtype=np.float64),
                    absolute_tolerance_continuous=absolute_tolerance_continuous,
                    relative_tolerance_continuous=relative_tolerance_continuous,
                    tolerance_discontinuous=tolerance_discontinuous,
                    num_point_min=num_point_min,
                    num_point_max=num_point_max,
                    mesh_length_min=mesh_length_min,
                    mesh_length_max=mesh_length_max,
                )
                value_adapt.append(v_.adapt(p_))

        self.update()

        if value_is_variable:
            return value_adapt[0]
        elif len(value_adapt) == len(value):
            return value_adapt
        else:
            return value_adapt + [value[-1]]

    @property
    @abstractmethod
    def _class_phase(self) -> type[PhaseBase]:
        pass

    @property
    def n_s(self) -> int:
        """Number of static parameters."""
        return self._num_static_parameter

    @property
    def s(self) -> list[sp.Symbol]:
        """:class:`sympy.Symbol` s of static parameters."""
        return self._symbol_static_parameter

    @property
    def n_p(self) -> int:
        """Number of phases."""
        return self._num_phase

    @property
    def p(self) -> list[PhaseBase]:
        """:class:`pockit.phase.Phase` s of system."""
        return self._phase

    @property
    def F_o(self) -> FastFunc:
        """:class:`pockit.base.fastfunc.FastFunc` s of the objective
        function."""
        return self._func_objective

    @property
    def n_c(self) -> int:
        """Number of system constraints."""
        return self._num_system_constraint

    @property
    def F_c(self) -> list[FastFunc]:
        """:class:`pockit.base.fastfunc.FastFunc` s of system constraints."""
        return self._func_system_constraint

    @property
    def v_lb(self) -> VecFloat:
        """Lower bounds of variables."""
        return self._lower_bound_variable

    @property
    def v_ub(self) -> VecFloat:
        """Upper bounds of variables."""
        return self._upper_bound_variable

    @property
    def c_lb(self) -> VecFloat:
        """Lower bounds of constraints."""
        return self._lower_bound_constraint

    @property
    def c_ub(self) -> VecFloat:
        """Upper bounds of constraints."""
        return self._upper_bound_constraint

    @property
    def N(self) -> int:
        """Number of phases."""
        return self._num_phase

    @property
    def L(self) -> int:
        """Number of optimization variables of the discretized optimization
        problem."""
        return self.r_s

    @property
    def ok(self) -> bool:
        """Whether the system is fully configured."""
        return self._phase_set and self._objective_set and self._system_constraint_set
