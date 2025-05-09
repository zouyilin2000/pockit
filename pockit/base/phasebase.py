# Copyright (c) 2024 Yilin Zou
import os
import itertools as it
from collections import namedtuple
from enum import Enum
from typing import Callable, Optional, Self
from abc import ABC, abstractmethod
import scipy.sparse
import sympy as sp

from .autoupdate import AutoUpdate
from .easyderiv import *
from .fastfunc import FastFunc, ensure_directory
from .vectypes import *
from .discretizationbase import *


class BcType(Enum):
    """Enum class to represent the type of boundary conditions."""

    FREE = 0
    """Free boundary condition."""
    FIXED = 1
    """Fixed boundary condition."""
    FUNC = 2
    """Boundary condition as a function of static parameters."""


class BcInfo(namedtuple("BcInfo", ["t", "v"])):
    """Named tuple to store boundary condition information."""

    t: BcType
    """Type of the boundary condition."""
    v: None | float | FastFunc
    """Value of the boundary condition."""


class PhaseBase(ABC):
    """A phase is a lower level objective of a multiple-phase optimal control
    problem."""

    def __init__(
        self,
        # class_discretization: type[DiscretizationBase],
        identifier: int,
        state: int | list[str],
        control: int | list[str],
        symbol_static_parameter: list[sp.Symbol],
        simplify: bool = False,
        parallel: bool = False,
        fastmath: bool = False,
    ) -> None:
        r"""Initialize a phase with given state, control, and static variables.

        States and controls can be given as the number of variables or the list of variable names.
        If names are given, they are used as the names of the variables.
        Otherwise, the names are generated automatically as :math:`x_0, x_1, \dots, x_{n - 1}`.

        Static variables should be identical to those defined in the system level.

        The ``identifier`` should be unique for each phase in a given system to avoid possible name conflict.

        It is recommended to use the :meth:`System.new_phase` method of the ``System``
        object to create phases instead of manually using this method.

        If ``simplify`` is ``True``, every symbolic expression will be simplified (by :func:`sympy.simplify`) before
        being compiled. This will slow down the speed of compilation.

        If ``parallel`` is ``True``, the ``parallel`` flag will be passed to the Numba JIT compiler,
        which will generate parallel code for multicore CPUs.
        This will slow down the speed of compilation and sometimes the speed of execution.

        If ``fastmath`` is ``True``, the ``fastmath`` flag will be passed to the Numba JIT compiler,
        see [Numba](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#fastmath)
        and [LLVM](https://llvm.org/docs/LangRef.html#fast-math-flags) documentations for details.

        Args:
            state: Number of state variables or list of state variable names.
            control: Number of control variables or list of control variable names.
            symbol_static_parameter: list of static parameters,
                should be identical to those in the ``System`` object.
            identifier: Unique identifier of the phase.
            simplify: Whether to use Sympy to simplify :class:`sympy.Expr` before compilation.
            parallel: Whether to use Numba ``parallel`` mode.
            fastmath: Whether to use Numba ``fastmath`` mode.
        """
        self._identifier = identifier
        # self._class_discretization = class_discretization

        if isinstance(state, int):
            self._num_state = state
            self._name_state = [f"x_{i}^{{({identifier})}}" for i in range(state)]
        elif isinstance(state, list):
            if "t" in state:
                raise ValueError(
                    'Symbol "t" is reserved for time. Use a different name for state variables'
                )
            self._name_state = [s + f"^{{({identifier})}}" for s in state]
            self._num_state = len(state)
        else:
            raise ValueError("state must be int or list of str")

        if isinstance(control, int):
            self._num_control = control
            self._name_control = [f"u_{i}^{{({identifier})}}" for i in range(control)]
        elif isinstance(control, list):
            if "t" in control:
                raise ValueError(
                    'Symbol "t" is reserved for time. Use a different name for control variables'
                )
            self._name_control = [c + f"^{{({identifier})}}" for c in control]
            self._num_control = len(control)
        else:
            raise ValueError("control must be int or list of str")

        self._num_variable = self._num_state + self._num_control

        self._num_static_parameter = len(symbol_static_parameter)
        self._name_static_parameter = [p.name for p in symbol_static_parameter]
        self._symbol_static_parameter = symbol_static_parameter

        self._symbol_state = [sp.Symbol(name) for name in self._name_state]
        self._symbol_control = [sp.Symbol(name) for name in self._name_control]
        self._symbol_time = sp.Symbol(f"t^{{({identifier})}}")
        self._symbols = (
            self._symbol_state
            + self._symbol_control
            + [self._symbol_time]
            + symbol_static_parameter
        )

        self._node_state_front = [Node() for _ in range(self._num_state)]
        self._node_state_middle = [Node() for _ in range(self._num_state)]
        self._node_state_back = [Node() for _ in range(self._num_state)]
        self._node_control_front = [Node() for _ in range(self._num_control)]
        self._node_control_middle = [Node() for _ in range(self._num_control)]
        self._node_control_back = [Node() for _ in range(self._num_control)]
        self._node_time_front = Node()
        self._node_time_back = Node()
        self._node_time_middle = Node()
        self._node_time_delta = Node()
        self._node_static_parameter = [
            Node() for _ in range(self._num_static_parameter)
        ]
        self._node_static_parameter_middle = [
            Node() for _ in range(self._num_static_parameter)
        ]
        self._node_basic = sum(
            [
                self._node_static_parameter,
                self._node_state_middle,
                self._node_control_front,
                self._node_control_middle,
                self._node_control_back,
                self._node_static_parameter_middle,
                self._node_state_front,
                self._node_state_back,
                [
                    self._node_time_front,
                    self._node_time_back,
                    self._node_time_middle,
                    self._node_time_delta,
                ],
            ],
            [],
        )
        self._node_front = sum(
            [
                self._node_state_front,
                self._node_control_front,
                [self._node_time_front],
                self._node_static_parameter,
            ],
            [],
        )
        self._node_middle = sum(
            [
                self._node_state_middle,
                self._node_control_middle,
                [self._node_time_middle],
                self._node_static_parameter_middle,
            ],
            [],
        )
        self._node_back = sum(
            [
                self._node_state_back,
                self._node_control_back,
                [self._node_time_back],
                self._node_static_parameter,
            ],
            [],
        )
        for i in range(self._num_static_parameter):
            self._node_static_parameter[i].set_G_i(
                [np.array([-self._num_static_parameter + i], dtype=np.int32)]
            )
            self._node_static_parameter[i].set_G([np.array([1.0], dtype=np.float64)])
        self._node_time_delta.args = [self._node_time_front, self._node_time_back]
        self._node_time_delta.g_i = np.array([0, 1], dtype=np.int32)
        self._node_time_delta.g = np.array([[-1.0], [1.0]], dtype=np.float64)

        self._simplify = simplify
        self._parallel = parallel
        self._fastmath = fastmath
        self._compile_parameters = simplify, parallel, fastmath

        # dynamics: 0
        # integral: 1
        # phase_constraint: 2
        # boundary_condition: 3
        # discretization: 4
        # node_basic: 5
        # node_integral: 6
        # node_dynamics: 7
        # node_phase_constraint: 8
        self._auto_update = AutoUpdate(
            9,
            [
                self._update_bound_variable,
                self._update_node_basic,
                self._update_node_integral,
                self._update_node_dynamics,
                self._update_node_phase_constraint,
                self._update_index_dynamic_constraint,
                self._update_index_phase_constraint,
                self._update_discontinuous_check_status,
            ],
        )
        self._auto_update.set_dependency(0, [2, 4])  # _update_bound_variable
        self._auto_update.set_dependency(1, [3, 4])  # _update_node_basic
        self._auto_update.set_dependency(2, [1, 5])  # _update_node_integral
        self._auto_update.set_dependency(3, [0, 5])  # _update_node_dynamics
        self._auto_update.set_dependency(4, [2, 5])  # _update_node_phase_constraint
        self._auto_update.set_dependency(5, [5, 7])  # _update_index_dynamic_constraint
        self._auto_update.set_dependency(6, [5, 8])  # _update_index_phase_constraint
        self._auto_update.set_dependency(
            7, [0, 1, 2, 3, 4]
        )  # _update_discontinuous_check_status

        self._dynamics_set = False
        self._boundary_condition_set = False
        self._discretization_set = False
        self._integral_set = False
        self._phase_constraint_set = False

        self.set_integral([])  # no integral by default
        self.set_phase_constraint([], [], [])  # no phase constraint by default

    def set_dynamics(
        self, dynamics: list[float | sp.Expr], *, cache: Optional[str] = None
    ) -> Self:
        """Set the dynamics of the phase.

        Args:
            dynamics: List of time derivatives of states composed with x, u, t, and s.
            cache: Path to the directory to store the compiled functions.
        """
        if len(dynamics) != self.n_x:
            raise ValueError(
                "the number of dynamics must be equal to the number of state variables"
            )

        if cache is None:
            cache_dynamic = it.repeat(None)
        else:
            ensure_directory(cache)
            cache_dynamic = (
                os.path.join(cache, f"dynamic_{i}.py") for i in range(self.n_x)
            )

        self._expr_dynamics = [sp.sympify(d) for d in dynamics]
        self._func_dynamics = [
            FastFunc(d, self._symbols, *self._compile_parameters, cache=c)
            for d, c in zip(self._expr_dynamics, cache_dynamic)
        ]

        self._auto_update.update(0)
        self._dynamics_set = True
        return self

    def set_integral(
        self, integral: list[float | sp.Expr], *, cache: Optional[str] = None
    ) -> Self:
        r"""Set the integrals of the phase.

        Symbols :math:`I_0, I_1, \dots, I_{n - 1}` will be automatically generated and set as a list as the attribute
        :attr:`I` to represent corresponding integrals.

        Args:
            integral: List of integrals to be concerned composed with x, u, t, and s.
            cache: Path to the directory to store the compiled functions.
        """
        self._num_integral = len(integral)
        if cache is None:
            cache_integral = it.repeat(None)
        else:
            ensure_directory(cache)
            cache_integral = (
                os.path.join(cache, f"integral_{i}.py") for i in range(self.n_I)
            )

        self._expr_integral = [sp.sympify(i) for i in integral]
        self._func_integral = [
            FastFunc(i, self._symbols, *self._compile_parameters, cache=c)
            for i, c in zip(self._expr_integral, cache_integral)
        ]

        self._symbol_integral = [
            sp.Symbol(f"I_{i}^{{({self._identifier})}}")
            for i in range(self._num_integral)
        ]

        self._auto_update.update(1)
        self._integral_set = True
        return self

    def set_phase_constraint(
        self,
        phase_constraint: list[sp.Expr],
        lower_bound: list[float],
        upper_bound: list[float],
        bang_bang_control: bool | list[bool] = False,
        *,
        cache: Optional[str] = None,
    ) -> Self:
        """Set phase constraints of the system, which is enforced in the entire
        time interval of the phase.

        For equality constraints, set the corresponding entry of ``lower_bounds`` and ``upper_bounds`` to the same value.
        For one-sided inequality constraints, set the corresponding entry of ``lower_bounds`` or ``upper_bounds``
        to ``-inf`` or ``inf``.
        If the problem to be solved is a bang-bang control problem, set ``bang_bang_control`` as a list of bools indicating
        whether the corresponding phase constraint is a bang-bang constraint.
        If all phase constraints are bang-bang constraints, ``bang_bang_control`` can also be set to ``True`` directly.

        Args:
            phase_constraint: List of phase constraints composed with x, u, t, and s
            lower_bound: List of lower bounds of phase constraints
            upper_bound: List of upper bounds of phase constraints
            bang_bang_control: List of bools indicating whether the corresponding phase constraint is a bang-bang constraint.
                Alternatively, set ``bang_bang_control`` as a single bool to apply to all phase constraints.
            cache: Path to the directory to store the compiled functions.
        """
        phase_constraint = list(phase_constraint)
        lower_bound = list(lower_bound)
        upper_bound = list(upper_bound)
        if not len(phase_constraint) == len(lower_bound) == len(upper_bound):
            raise ValueError(
                "phase_constraint, lower_bound and upper_bound must have the same length"
            )

        self._variable_bounds_phase = []
        self._static_parameter_bounds_phase = []
        self._time_bounds_phase = []

        self._expr_phase_constraint = []
        lower_bound_phase_constraint = []
        upper_bound_phase_constraint = []
        for c, lb, ub in zip(phase_constraint, lower_bound, upper_bound):
            if c.is_symbol:
                i = self._symbols.index(c)
                if i < self._num_variable:
                    self._variable_bounds_phase.append((i, lb, ub))
                elif i == self._num_variable:
                    self._time_bounds_phase.append((lb, ub))
                else:
                    self._static_parameter_bounds_phase.append(
                        (i - self._num_variable - 1, lb, ub)
                    )
            else:
                self._expr_phase_constraint.append(sp.sympify(c))
                lower_bound_phase_constraint.append(lb)
                upper_bound_phase_constraint.append(ub)

        self._num_phase_constraint = len(self._expr_phase_constraint)
        if cache is None:
            cache_phase_constraint = it.repeat(None)
        else:
            ensure_directory(cache)
            cache_phase_constraint = (
                os.path.join(cache, f"phase_constraint_{i}.py") for i in range(self.n_c)
            )
        self._func_phase_constraint = [
            FastFunc(pc, self._symbols, *self._compile_parameters, cache=c)
            for pc, c in zip(self._expr_phase_constraint, cache_phase_constraint)
        ]

        self._lower_bound_phase_constraint = np.array(
            lower_bound_phase_constraint, dtype=np.float64
        )
        self._upper_bound_phase_constraint = np.array(
            upper_bound_phase_constraint, dtype=np.float64
        )

        # scale to [0, 1]
        if isinstance(bang_bang_control, bool):
            bang_bang_control = it.repeat(bang_bang_control)
        self._func_bang_bang_control = []
        for expr, lb, rb, bb in zip(
            phase_constraint, lower_bound, upper_bound, bang_bang_control
        ):
            if bb:
                if np.isinf(lb) or np.isinf(rb):
                    raise ValueError(
                        "lower_bound and upper_bound must be finite for bang-bang control constraint"
                    )
                if rb <= lb + 1e-10:
                    raise ValueError(
                        "lower_bound must be strictly less than upper_bound for bang-bang control constraint"
                    )
                self._func_bang_bang_control.append(
                    FastFunc(
                        (expr - lb) / (rb - lb),
                        self._symbols,
                        *self._compile_parameters,
                    )
                )

        self._num_bang_bang = len(self._func_bang_bang_control)

        self._auto_update.update(2)
        self._phase_constraint_set = True

        return self

    def _parse_boundary_condition(
        self, bc: None | float | sp.Expr, *, cache: Optional[str]
    ) -> BcInfo:
        if bc is None:
            return BcInfo(BcType.FREE, None)
        elif isinstance(bc, float):
            return BcInfo(BcType.FIXED, bc)
        elif isinstance(bc, sp.Expr):
            return BcInfo(
                BcType.FUNC,
                FastFunc(
                    bc,
                    self._symbol_static_parameter,
                    *self._compile_parameters,
                    cache=cache,
                ),
            )
        else:
            raise ValueError("boundary condition must be None, number or sp.Expr")

    def set_boundary_condition(
        self,
        initial_value: list[None | float | sp.Expr],
        terminal_value: list[None | float | sp.Expr],
        initial_time: None | float | sp.Expr,
        terminal_time: None | float | sp.Expr,
        *,
        cache: Optional[str] = None,
    ) -> Self:
        """Set the boundary condition and initial/terminal time of the phase.

        ``initial_time``, ``terminal_time``, and each element of ``initial_value`` and ``terminal_value`` can be set as
        ``None`` for free, a floating number for fixed, or :class:`sympy.Expr` of static parameters.

        Args:
            initial_value: List of initial values of states.
            terminal_value: List of terminal values of states.
            initial_time: Initial time.
            terminal_time: Terminal time.
            cache: Path to the directory to store the compiled functions.
        """
        if not len(initial_value) == len(terminal_value) == self.n_x:
            raise ValueError(
                "initial_value, terminal_value must have the same length as number of state variables"
            )
        for i in range(self.n_x):
            if isinstance(initial_value[i], int):
                initial_value[i] = float(initial_value[i])
            if isinstance(terminal_value[i], int):
                terminal_value[i] = float(terminal_value[i])
        if isinstance(initial_time, int):
            initial_time = float(initial_time)
        if isinstance(terminal_time, int):
            terminal_time = float(terminal_time)

        self._initial_value = initial_value
        self._terminal_value = terminal_value
        self._initial_time = initial_time
        self._terminal_time = terminal_time

        if cache is None:
            self.info_bc_0 = [
                self._parse_boundary_condition(bc, cache=None) for bc in initial_value
            ]
            self.info_bc_f = [
                self._parse_boundary_condition(bc, cache=None) for bc in terminal_value
            ]
            self.info_t_0 = self._parse_boundary_condition(initial_time, cache=None)
            self.info_t_f = self._parse_boundary_condition(terminal_time, cache=None)
        else:
            ensure_directory(cache)
            self.info_bc_0 = [
                self._parse_boundary_condition(
                    bc, cache=os.path.join(cache, f"boundary_condition_0_{i}.py")
                )
                for i, bc in enumerate(initial_value)
            ]
            self.info_bc_f = [
                self._parse_boundary_condition(
                    bc, cache=os.path.join(cache, f"boundary_condition_f_{i}.py")
                )
                for i, bc in enumerate(terminal_value)
            ]
            self.info_t_0 = self._parse_boundary_condition(
                initial_time, cache=os.path.join(cache, "boundary_condition_t_0.py")
            )
            self.info_t_f = self._parse_boundary_condition(
                terminal_time, cache=os.path.join(cache, "boundary_condition_t_f.py")
            )

        self._auto_update.update(3)
        self._boundary_condition_set = True
        return self

    def set_discretization(
        self, mesh: int | Iterable[float], num_point: int | Iterable[int]
    ):
        """Set the discretization scheme of the system.

        ``mesh`` will be rescaled when needed. If it is set to an integer, a uniform mesh will be used.
        If set with an array, it will be used as mesh directly after scaling.

        ``num_point`` decides the number of interpolation/integration points in each subinterval.
        Same number of points will be used in all subintervals if set with an integer.

        Args:
            mesh: Number of mesh or mesh points.
            num_point: Number of interpolation and integration points in each subinterval.
        """
        if isinstance(mesh, int):  # uniform mesh
            self._mesh = np.linspace(0, 1, mesh + 1, endpoint=True)
        else:  # scale to [0, 1]
            mesh = np.array(list(mesh), dtype=np.float64)
            self._mesh = (mesh - mesh[0]) / (mesh[-1] - mesh[0])
        self._num_interval = len(self._mesh) - 1

        if isinstance(num_point, int):
            self._num_point = np.full(self._num_interval, num_point, dtype=np.int32)
        else:
            num_point = np.array(list(num_point), dtype=np.int32)
            self._num_point = num_point

        if len(self._num_point) != self._num_interval:
            raise ValueError(
                "num_point must have the same length as mesh intervals (= len(mesh) - 1)"
            )

        self._object_discretization = self._class_discretization(
            self._mesh, self._num_point, self.n_x, self.n_u
        )

        for i in range(self.n_x):
            self._node_state_middle[i].l = self.index_state.L_m
            self._node_state_middle[i].set_G_i(
                [
                    np.arange(
                        self.l_v[i] + self.index_state.l_m,
                        self.l_v[i] + self.index_state.r_m,
                        dtype=np.int32,
                    )
                ]
            )
            self._node_state_middle[i].set_G(
                [np.full(self.index_state.L_m, 1.0, dtype=np.float64)]
            )
        for i in range(self.n_u):
            self._node_control_middle[i].l = self.index_control.L_m
            self._node_control_middle[i].set_G_i(
                [
                    np.arange(
                        self.l_v[self.n_x + i] + self.index_control.l_m,
                        self.l_v[self.n_x + i] + self.index_control.r_m,
                        dtype=np.int32,
                    )
                ]
            )
            self._node_control_middle[i].set_G(
                [np.full(self.index_control.L_m, 1.0, dtype=np.float64)]
            )
        self._node_time_middle.l = self.index_mstage.L_m
        self._node_time_middle.args = [self._node_time_front, self._node_time_back]
        self._node_time_middle.g_i = np.array([0, 1], dtype=np.int32)
        self._node_time_middle.g = np.array(
            [
                1.0 - self.t_m[self.index_mstage.m],
                self.t_m[self.index_mstage.m],
            ],
            dtype=np.float64,
        )
        for i in range(self._num_static_parameter):
            self._node_static_parameter_middle[i].l = self.index_mstage.L_m
            self._node_static_parameter_middle[i].args = [
                self._node_static_parameter[i]
            ]
            self._node_static_parameter_middle[i].g_i = np.array([0], dtype=np.int32)
            self._node_static_parameter_middle[i].g = np.array(
                [np.full(self.index_mstage.L_m, 1.0, dtype=np.float64)]
            )

        self._auto_update.update(4)
        self._discretization_set = True
        return self

    def _update_bound_variable(self) -> None:
        """Update lower and upper bound of variables.

        Should be called after the discretization scheme, phase
        constraints, and boundary conditions are set.
        """
        self._lower_bound_variable = np.full(self.L, -np.inf, dtype=np.float64)
        self._upper_bound_variable = np.full(self.L, np.inf, dtype=np.float64)
        for i, lb, ub in self._variable_bounds_phase:
            self._lower_bound_variable[self.l_v[i] : self.r_v[i]] = np.maximum(
                self._lower_bound_variable[self.l_v[i] : self.r_v[i]], lb
            )
            self._upper_bound_variable[self.l_v[i] : self.r_v[i]] = np.minimum(
                self._upper_bound_variable[self.l_v[i] : self.r_v[i]], ub
            )
        for lb, ub in self._time_bounds_phase:
            self._lower_bound_variable[-2] = np.maximum(
                self._lower_bound_variable[-2], lb
            )
            self._lower_bound_variable[-1] = np.maximum(
                self._lower_bound_variable[-1], lb
            )
            self._upper_bound_variable[-2] = np.minimum(
                self._upper_bound_variable[-2], ub
            )
            self._upper_bound_variable[-1] = np.minimum(
                self._upper_bound_variable[-1], ub
            )

    def _update_node_basic(self):
        for i in range(self.n_x):
            if self.info_bc_0[i].t == BcType.FREE:
                self._node_state_front[i].set_G_i(
                    [np.array([self.l_v[i]], dtype=np.int32)]
                )
                self._node_state_front[i].set_G([np.array([1.0], dtype=np.float64)])
            elif self.info_bc_0[i].t == BcType.FUNC:
                self._node_state_front[i].args = self._node_static_parameter
                self._node_state_front[i].g_i = self.info_bc_0[i].v.G_index
                self._node_state_front[i].h_i_row = self.info_bc_0[i].v.H_index_row
                self._node_state_front[i].h_i_col = self.info_bc_0[i].v.H_index_col
            if self.info_bc_f[i].t == BcType.FREE:
                self._node_state_back[i].set_G_i(
                    [np.array([self.r_v[i] - 1], dtype=np.int32)]
                )
                self._node_state_back[i].set_G([np.array([1.0], dtype=np.float64)])
            elif self.info_bc_f[i].t == BcType.FUNC:
                self._node_state_back[i].args = self._node_static_parameter
                self._node_state_back[i].g_i = self.info_bc_f[i].v.G_index
                self._node_state_back[i].h_i_row = self.info_bc_f[i].v.H_index_row
                self._node_state_back[i].h_i_col = self.info_bc_f[i].v.H_index_col
        if self.index_mstage.f:
            for i in range(self._num_control):
                self._node_control_front[i].set_G_i(
                    [np.array([self.l_v[self.n_x + i]], dtype=np.int32)]
                )
                self._node_control_front[i].set_G([np.array([1.0], dtype=np.float64)])
        if self.index_mstage.b:
            for i in range(self._num_control):
                self._node_control_back[i].set_G_i(
                    [np.array([self.r_v[self.n_x + i] - 1], dtype=np.int32)]
                )
                self._node_control_back[i].set_G([np.array([1.0], dtype=np.float64)])
        if self.info_t_0.t == BcType.FREE:
            self._node_time_front.set_G_i([np.array([self.L - 2], dtype=np.int32)])
            self._node_time_front.set_G([np.array([1.0], dtype=np.float64)])
        elif self.info_t_0.t == BcType.FUNC:
            self._node_time_front.args = self._node_static_parameter
            self._node_time_front.g_i = self.info_t_0.v.G_index
            self._node_time_front.h_i_row = self.info_t_0.v.H_index_row
            self._node_time_front.h_i_col = self.info_t_0.v.H_index_col
        if self.info_t_f.t == BcType.FREE:
            self._node_time_back.set_G_i([np.array([self.L - 1], dtype=np.int32)])
            self._node_time_back.set_G([np.array([1.0], dtype=np.float64)])
        elif self.info_t_f.t == BcType.FUNC:
            self._node_time_back.args = self._node_static_parameter
            self._node_time_back.g_i = self.info_t_f.v.G_index
            self._node_time_back.h_i_row = self.info_t_f.v.H_index_row
            self._node_time_back.h_i_col = self.info_t_f.v.H_index_col

        forward_gradient_i(self._node_basic)
        forward_hessian_phase_i(self._node_basic)
        self._auto_update.update(5)

    def _update_node_function(
        self, funcs: list[FastFunc]
    ) -> tuple[list[Node], list[Node], list[Node]]:
        l_f = len(funcs)
        nodes_front = [Node() for _ in range(l_f)]
        nodes_middle = [Node() for _ in range(l_f)]
        nodes_back = [Node() for _ in range(l_f)]
        for i in range(len(funcs)):
            nodes_front[i].args = self._node_front
            nodes_front[i].g_i = funcs[i].G_index
            nodes_front[i].h_i_row = funcs[i].H_index_row
            nodes_front[i].h_i_col = funcs[i].H_index_col
            nodes_middle[i].l = self.index_mstage.L_m
            nodes_middle[i].args = self._node_middle
            nodes_middle[i].g_i = funcs[i].G_index
            nodes_middle[i].h_i_row = funcs[i].H_index_row
            nodes_middle[i].h_i_col = funcs[i].H_index_col
            nodes_back[i].args = self._node_back
            nodes_back[i].g_i = funcs[i].G_index
            nodes_back[i].h_i_row = funcs[i].H_index_row
            nodes_back[i].h_i_col = funcs[i].H_index_col
        forward_gradient_i(nodes_front + nodes_middle + nodes_back)
        forward_hessian_phase_i(nodes_front + nodes_middle + nodes_back)
        return nodes_front, nodes_middle, nodes_back

    def _update_node_scale(self, node: list[Node]) -> list[Node]:
        node_scaled = [Node() for _ in range(len(node))]
        for n_unscaled, n_scaled in zip(node, node_scaled):
            n_scaled.l = n_unscaled.l
            n_scaled.args = [n_unscaled, self._node_time_delta]
            n_scaled.g_i = np.array([0, 1], dtype=np.int32)
            n_scaled.h_i_row = np.array([1], dtype=np.int32)
            n_scaled.h_i_col = np.array([0], dtype=np.int32)
            n_scaled.h = np.array([[1.0]], dtype=np.float64)
        forward_gradient_i(node_scaled)
        forward_hessian_phase_i(node_scaled)
        return node_scaled

    def _update_node_integral(self) -> None:
        (
            self._node_integral_unscaled_front,
            self._node_integral_unscaled_middle,
            self._node_integral_unscaled_back,
        ) = self._update_node_function(self._func_integral)
        self._node_integral_front = self._update_node_scale(
            self._node_integral_unscaled_front
        )
        self._node_integral_middle = self._update_node_scale(
            self._node_integral_unscaled_middle
        )
        self._node_integral_back = self._update_node_scale(
            self._node_integral_unscaled_back
        )
        self._node_integral = (
            self._node_integral_unscaled_front
            + self._node_integral_unscaled_middle
            + self._node_integral_unscaled_back
            + self._node_integral_front
            + self._node_integral_middle
            + self._node_integral_back
        )
        self._auto_update.update(6)

    def _update_node_dynamics(self) -> None:
        (
            self._node_dynamics_unscaled_front,
            self._node_dynamics_unscaled_middle,
            self._node_dynamics_unscaled_back,
        ) = self._update_node_function(self._func_dynamics)
        self._node_dynamics_front = self._update_node_scale(
            self._node_dynamics_unscaled_front
        )
        self._node_dynamics_middle = self._update_node_scale(
            self._node_dynamics_unscaled_middle
        )
        self._node_dynamics_back = self._update_node_scale(
            self._node_dynamics_unscaled_back
        )
        self._node_dynamics = (
            self._node_dynamics_unscaled_front
            + self._node_dynamics_unscaled_middle
            + self._node_dynamics_unscaled_back
            + self._node_dynamics_front
            + self._node_dynamics_middle
            + self._node_dynamics_back
        )
        self._auto_update.update(7)

    def _update_node_phase_constraint(self) -> None:
        (
            self._node_phase_constraint_front,
            self._node_phase_constraint_middle,
            self._node_phase_constraint_back,
        ) = self._update_node_function(self._func_phase_constraint)
        self._node_phase_constraint = (
            self._node_phase_constraint_front
            + self._node_phase_constraint_middle
            + self._node_phase_constraint_back
        )
        self._auto_update.update(8)

    def _update_discontinuous_check_status(self) -> None:
        self._discontinuous_check_passed = False

    @staticmethod
    def _value_boundary_condition(info, x: VecFloat, s: VecFloat):
        if info.t == BcType.FREE:
            return x
        elif info.t == BcType.FIXED:
            return info.v
        else:
            return info.v.F(s, 1)[0]

    def _value_basic(self, x: VecFloat, s: VecFloat):
        for i, bc_info in enumerate(self.info_bc_0):
            x[self.l_v[i]] = self._value_boundary_condition(bc_info, x[self.l_v[i]], s)
        for i, bc_info in enumerate(self.info_bc_f):
            x[self.r_v[i] - 1] = self._value_boundary_condition(
                bc_info, x[self.r_v[i] - 1], s
            )
        x[-2] = self._value_boundary_condition(self.info_t_0, x[-2], s)
        x[-1] = self._value_boundary_condition(self.info_t_f, x[-1], s)
        s_ = np.repeat(s, self.L_m)
        mt = (x[-1] + x[-2]) / 2
        dt = x[-1] - x[-2]
        t_ = (self.t_m - 0.5) * dt + mt
        return np.concatenate([self.f_v2m(x[:-2]), t_, s_]), dt

    def _update_index_dynamic_constraint(self):
        jac_T_row = []
        jac_T_col = []
        for i in range(self.n_x):
            if self._node_state_front[i].G_i:
                G_i_front = np.concatenate(self._node_state_front[i].G_i)
                jac_T_row.append(
                    self.l_d[i] + np.repeat(self.T_v_coo.f.row, len(G_i_front))
                )
                jac_T_col.append(np.tile(G_i_front, len(self.T_v_coo.f)))

            jac_T_row.append(self.l_d[i] + self.T_v_coo.m.row)
            jac_T_col.append(self.l_v[i] + self.T_v_coo.m.col)

            if self._node_state_back[i].G_i:
                G_i_back = np.concatenate(self._node_state_back[i].G_i)
                jac_T_row.append(
                    self.l_d[i] + np.repeat(self.T_v_coo.b.row, len(G_i_back))
                )
                jac_T_col.append(np.tile(G_i_back, len(self.T_v_coo.b)))

        jac_I_row = []
        jac_I_col = []
        for i in range(self.n_x):
            if self.index_mstage.f and self._node_dynamics_front[i].G_i:
                G_i_front = np.concatenate(self._node_dynamics_front[i].G_i)
                jac_I_row.append(
                    self.l_d[i] + np.repeat(self.I_m_coo.f.row, len(G_i_front))
                )
                jac_I_col.append(np.tile(G_i_front, len(self.I_m_coo.f)))

            for G_i_ in self._node_dynamics_middle[i].G_i:
                jac_I_row.append(self.l_d[i] + self.I_m_coo.m.row)
                jac_I_col.append(G_i_[self.I_m_coo.m.col - self.index_mstage.l_m])

            if self.index_mstage.b and self._node_dynamics_back[i].G_i:
                G_i_back = np.concatenate(self._node_dynamics_back[i].G_i)
                jac_I_row.append(
                    self.l_d[i] + np.repeat(self.I_m_coo.b.row, len(G_i_back))
                )
                jac_I_col.append(np.tile(G_i_back, len(self.I_m_coo.b)))

        self._jac_dynamic_constraint_row = np.concatenate(jac_T_row + jac_I_row)
        self._jac_dynamic_constraint_col = np.concatenate(jac_T_col + jac_I_col)

        hess_T_row = []
        hess_T_col = []
        for i in range(self.n_x):
            if self._node_state_front[i].H_i_row:
                H_i_row_front = np.concatenate(self._node_state_front[i].H_i_row)
                H_i_col_front = np.concatenate(self._node_state_front[i].H_i_col)
                hess_T_row.append(np.tile(H_i_row_front, len(self.T_v_coo.f)))
                hess_T_col.append(np.tile(H_i_col_front, len(self.T_v_coo.f)))

            if self._node_state_back[i].H_i_row:
                H_i_row_back = np.concatenate(self._node_state_back[i].H_i_row)
                H_i_col_back = np.concatenate(self._node_state_back[i].H_i_col)
                hess_T_row.append(np.tile(H_i_row_back, len(self.T_v_coo.b)))
                hess_T_col.append(np.tile(H_i_col_back, len(self.T_v_coo.b)))

        hess_I_row = []
        hess_I_col = []
        for i in range(self.n_x):
            if self.index_mstage.f and self._node_dynamics_front[i].H_i_row:
                H_i_row_front = np.concatenate(self._node_dynamics_front[i].H_i_row)
                H_i_col_front = np.concatenate(self._node_dynamics_front[i].H_i_col)
                hess_I_row.append(np.tile(H_i_row_front, len(self.I_m_coo.f)))
                hess_I_col.append(np.tile(H_i_col_front, len(self.I_m_coo.f)))

            for H_i_row_, H_i_col_ in zip(
                self._node_dynamics_middle[i].H_i_row,
                self._node_dynamics_middle[i].H_i_col,
            ):
                hess_I_row.append(H_i_row_[self.I_m_coo.m.col - self.index_mstage.l_m])
                hess_I_col.append(H_i_col_[self.I_m_coo.m.col - self.index_mstage.l_m])

            if self.index_mstage.b and self._node_dynamics_back[i].H_i_row:
                H_i_row_back = np.concatenate(self._node_dynamics_back[i].H_i_row)
                H_i_col_back = np.concatenate(self._node_dynamics_back[i].H_i_col)
                hess_I_row.append(np.tile(H_i_row_back, len(self.I_m_coo.b)))
                hess_I_col.append(np.tile(H_i_col_back, len(self.I_m_coo.b)))

        hess_row = hess_T_row + hess_I_row
        hess_col = hess_T_col + hess_I_col
        self._hess_dynamic_constraint_row = (
            np.concatenate(hess_row) if hess_row else np.array([], dtype=np.int32)
        )
        self._hess_dynamic_constraint_col = (
            np.concatenate(hess_col) if hess_col else np.array([], dtype=np.int32)
        )

    def _update_index_phase_constraint(self):
        jac_row = []
        jac_col = []
        r_ = 0
        for i in range(self._num_phase_constraint):
            if self.index_mstage.f:
                for G_i_ in self._node_phase_constraint_front[i].G_i:
                    jac_row.append(np.array([r_], dtype=np.int32))
                    jac_col.append(G_i_)
            for G_i_ in self._node_phase_constraint_middle[i].G_i:
                jac_row.append(
                    np.arange(
                        r_ + self.index_mstage.l_m,
                        r_ + self.index_mstage.r_m,
                        dtype=np.int32,
                    )
                )
                jac_col.append(G_i_)
            if self.index_mstage.b:
                for G_i_ in self._node_phase_constraint_back[i].G_i:
                    jac_row.append(np.array([r_ + self.L_m - 1], dtype=np.int32))
                    jac_col.append(G_i_)
            r_ += self.L_m
        self._jac_phase_constraint_row = (
            np.concatenate(jac_row) if jac_row else np.array([], dtype=np.int32)
        )
        self._jac_phase_constraint_col = (
            np.concatenate(jac_col) if jac_col else np.array([], dtype=np.int32)
        )

        hess_row = []
        hess_col = []
        for i in range(self._num_phase_constraint):
            if self.index_mstage.f:
                hess_row.extend(self._node_phase_constraint_front[i].H_i_row)
                hess_col.extend(self._node_phase_constraint_front[i].H_i_col)
            hess_row.extend(self._node_phase_constraint_middle[i].H_i_row)
            hess_col.extend(self._node_phase_constraint_middle[i].H_i_col)
            if self.index_mstage.b:
                hess_row.extend(self._node_phase_constraint_back[i].H_i_row)
                hess_col.extend(self._node_phase_constraint_back[i].H_i_col)
        self._hess_phase_constraint_row = (
            np.concatenate(hess_row, dtype=np.int32)
            if hess_row
            else np.array([], dtype=np.int32)
        )
        self._hess_phase_constraint_col = (
            np.concatenate(hess_col, dtype=np.int32)
            if hess_col
            else np.array([], dtype=np.int32)
        )

    def _value_integral(self, which, x: VecFloat, s: VecFloat):
        vb, dt = self._value_basic(x, s)
        vi = [
            self.F_I[i].F(vb, self.L_m) if flag else None
            for i, flag in enumerate(which)
        ]
        return np.array(
            [vi_.dot(self.w_m) * dt if vi_ is not None else 0.0 for vi_ in vi],
            dtype=np.float64,
        )

    def _value_dynamic_constraint(self, x: VecFloat, s: VecFloat):
        vb, dt = self._value_basic(x, s)
        Tx = [self.T_v.dot(x[self.l_v[i] : self.r_v[i]]) for i in range(self.n_x)]
        If = [self.I_m.dot(f_d.F(vb, self.L_m)) * dt for f_d in self.F_d]
        return np.concatenate([tx_ - if_ for tx_, if_ in zip(Tx, If)], dtype=np.float64)

    def _value_phase_constraint(self, x: VecFloat, s: VecFloat):
        vb, _ = self._value_basic(x, s)
        vpc = [f_c.F(vb, self.L_m) for f_c in self.F_c]
        return (
            np.concatenate(vpc, dtype=np.float64)
            if vpc
            else np.array([], dtype=np.float64)
        )

    def _grad_basic(self, x: VecFloat, s: VecFloat):
        for i, bc_info in enumerate(self.info_bc_0):
            if bc_info.t == BcType.FUNC:
                self._node_state_front[i].g = bc_info.v.G(s, 1)
        for i, bc_info in enumerate(self.info_bc_f):
            if bc_info.t == BcType.FUNC:
                self._node_state_back[i].g = bc_info.v.G(s, 1)
        if self.info_t_0.t == BcType.FUNC:
            self._node_time_front.g = self.info_t_0.v.G(s, 1)
        if self.info_t_f.t == BcType.FUNC:
            self._node_time_back.g = self.info_t_f.v.G(s, 1)
        forward_gradient_v(self._node_basic)

    def _grad_integral(self, which, x: VecFloat, s: VecFloat):
        vb, dt = self._value_basic(x, s)
        for i, flag in enumerate(which):
            if flag:
                f = self.F_I[i].F(vb, self.L_m)
                g = self.F_I[i].G(vb, self.L_m)
                if self.index_mstage.f:
                    integral_unscaled_front_V = np.array([f[0]], dtype=np.float64)
                    self._node_integral_unscaled_front[i].g = g[:, :1]
                    self._node_integral_front[i].g = np.array(
                        [
                            np.full_like(integral_unscaled_front_V, dt),
                            integral_unscaled_front_V,
                        ]
                    )
                integral_unscaled_middle_V = f[self.index_mstage.m]
                self._node_integral_unscaled_middle[i].g = g[:, self.index_mstage.m]
                self._node_integral_middle[i].g = np.array(
                    [
                        np.full_like(integral_unscaled_middle_V, dt),
                        integral_unscaled_middle_V,
                    ]
                )
                if self.index_mstage.b:
                    integral_unscaled_back_V = np.array([f[-1]], dtype=np.float64)
                    self._node_integral_unscaled_back[i].g = g[:, -1:]
                    self._node_integral_back[i].g = np.array(
                        [
                            np.full_like(integral_unscaled_back_V, dt),
                            integral_unscaled_back_V,
                        ]
                    )
        forward_gradient_v(self._node_integral)

    def _grad_dynamic_constraint(self, x: VecFloat, s: VecFloat):
        jac_T = []
        for i in range(self.n_x):
            if self._node_state_front[i].G:
                G_front = np.concatenate(self._node_state_front[i].G)
                jac_T.append(np.kron(self.T_v_coo.f.data, G_front))

            jac_T.append(self.T_v_coo.m.data)

            if self._node_state_back[i].G:
                G_back = np.concatenate(self._node_state_back[i].G)
                jac_T.append(np.kron(self.T_v_coo.b.data, G_back))

        vb, dt = self._value_basic(x, s)
        for i in range(self.n_x):
            f = self.F_d[i].F(vb, self.L_m)
            g = self.F_d[i].G(vb, self.L_m)
            if self.index_mstage.f:
                dynamics_unscaled_front_V = np.array([f[0]], dtype=np.float64)
                self._node_dynamics_unscaled_front[i].g = g[:, :1]
                self._node_dynamics_front[i].g = np.array(
                    [
                        np.full_like(dynamics_unscaled_front_V, dt),
                        dynamics_unscaled_front_V,
                    ]
                )
            dynamics_unscaled_middle_V = f[self.index_mstage.m]
            self._node_dynamics_unscaled_middle[i].g = g[:, self.index_mstage.m]
            self._node_dynamics_middle[i].g = np.array(
                [
                    np.full_like(dynamics_unscaled_middle_V, dt),
                    dynamics_unscaled_middle_V,
                ]
            )
            if self.index_mstage.b:
                dynamics_unscaled_back_V = np.array([f[-1]], dtype=np.float64)
                self._node_dynamics_unscaled_back[i].g = g[:, -1:]
                self._node_dynamics_back[i].g = np.array(
                    [
                        np.full_like(dynamics_unscaled_back_V, dt),
                        dynamics_unscaled_back_V,
                    ]
                )
        forward_gradient_v(self._node_dynamics)

        jac_I = []
        for i in range(self.n_x):
            if self.index_mstage.f and self._node_dynamics_front[i].G:
                G_front = np.concatenate(self._node_dynamics_front[i].G)
                jac_I.append(-np.kron(self.I_m_coo.f.data, G_front))
            for G_ in self._node_dynamics_middle[i].G:
                jac_I.append(
                    -self.I_m_coo.m.data
                    * G_[self.I_m_coo.m.col - self.index_mstage.l_m]
                )
            if self.index_mstage.b and self._node_dynamics_back[i].G:
                G_back = np.concatenate(self._node_dynamics_back[i].G)
                jac_I.append(-np.kron(self.I_m_coo.b.data, G_back))
        return np.concatenate(jac_T + jac_I, dtype=np.float64)

    def _grad_phase_constraint(self, x: VecFloat, s: VecFloat):
        vb, _ = self._value_basic(x, s)
        for i, f in enumerate(self._func_phase_constraint):
            g = f.G(vb, self.L_m)
            if self.index_mstage.f:
                self._node_phase_constraint_front[i].g = g[:, :1]
            self._node_phase_constraint_middle[i].g = g[:, self.index_mstage.m]
            if self.index_mstage.b:
                self._node_phase_constraint_back[i].g = g[:, -1:]
        forward_gradient_v(self._node_phase_constraint)

        jac = []
        for i in range(self._num_phase_constraint):
            if self.index_mstage.f:
                jac.extend(self._node_phase_constraint_front[i].G)
            jac.extend(self._node_phase_constraint_middle[i].G)
            if self.index_mstage.b:
                jac.extend(self._node_phase_constraint_back[i].G)
        return (
            np.concatenate(jac, dtype=np.float64)
            if jac
            else np.array([], dtype=np.float64)
        )

    def _hess_basic(self, x: VecFloat, s: VecFloat):
        for i, bc_info in enumerate(self.info_bc_0):
            if bc_info.t == BcType.FUNC:
                self._node_state_front[i].g = bc_info.v.G(s, 1)
                self._node_state_front[i].h = bc_info.v.H(s, 1)
        for i, bc_info in enumerate(self.info_bc_f):
            if bc_info.t == BcType.FUNC:
                self._node_state_back[i].g = bc_info.v.G(s, 1)
                self._node_state_back[i].h = bc_info.v.H(s, 1)
        if self.info_t_0.t == BcType.FUNC:
            self._node_time_front.g = self.info_t_0.v.G(s, 1)
            self._node_time_front.h = self.info_t_0.v.H(s, 1)
        if self.info_t_f.t == BcType.FUNC:
            self._node_time_back.g = self.info_t_f.v.G(s, 1)
            self._node_time_back.h = self.info_t_f.v.H(s, 1)
        forward_gradient_v(self._node_basic)
        forward_hessian_phase_v(self._node_basic)

    def _hess_integral(self, which, x: VecFloat, s: VecFloat):
        vb, dt = self._value_basic(x, s)
        for i, flag in enumerate(which):
            if flag:
                f = self.F_I[i].F(vb, self.L_m)
                g = self.F_I[i].G(vb, self.L_m)
                h = self.F_I[i].H(vb, self.L_m)
                if self.index_mstage.f:
                    integral_unscaled_front_V = np.array([f[0]], dtype=np.float64)
                    self._node_integral_unscaled_front[i].g = g[:, :1]
                    self._node_integral_front[i].g = np.array(
                        [
                            np.full_like(integral_unscaled_front_V, dt),
                            integral_unscaled_front_V,
                        ]
                    )
                    self._node_integral_unscaled_front[i].h = h[:, :1]
                integral_unscaled_middle_V = f[self.index_mstage.m]
                self._node_integral_unscaled_middle[i].g = g[:, self.index_mstage.m]
                self._node_integral_middle[i].g = np.array(
                    [
                        np.full_like(integral_unscaled_middle_V, dt),
                        integral_unscaled_middle_V,
                    ]
                )
                self._node_integral_unscaled_middle[i].h = h[:, self.index_mstage.m]
                if self.index_mstage.b:
                    integral_unscaled_back_V = np.array([f[-1]], dtype=np.float64)
                    self._node_integral_unscaled_back[i].g = g[:, -1:]
                    self._node_integral_back[i].g = np.array(
                        [
                            np.full_like(integral_unscaled_back_V, dt),
                            integral_unscaled_back_V,
                        ]
                    )
                    self._node_integral_unscaled_back[i].h = h[:, -1:]
        forward_gradient_v(self._node_integral)
        forward_hessian_phase_v(self._node_integral)

    def _hess_dynamic_constraint(self, x: VecFloat, s: VecFloat, fct: VecFloat):
        hess_T = []
        for i in range(self.n_x):
            if self._node_state_front[i].H:
                H_front = np.concatenate(self._node_state_front[i].H)
                hess_T.append(
                    np.kron(
                        self.T_v_coo.f.data * fct[self.l_d[i] + self.T_v_coo.f.row],
                        H_front,
                    )
                )

            if self._node_state_back[i].H:
                H_back = np.concatenate(self._node_state_back[i].H)
                hess_T.append(
                    np.kron(
                        self.T_v_coo.b.data * fct[self.l_d[i] + self.T_v_coo.b.row],
                        H_back,
                    )
                )

        vb, dt = self._value_basic(x, s)
        for i in range(self.n_x):
            f = self.F_d[i].F(vb, self.L_m)
            g = self.F_d[i].G(vb, self.L_m)
            h = self.F_d[i].H(vb, self.L_m)
            if self.index_mstage.f:
                dynamics_unscaled_front_V = np.array([f[0]], dtype=np.float64)
                self._node_dynamics_unscaled_front[i].g = g[:, :1]
                self._node_dynamics_front[i].g = np.array(
                    [
                        np.full_like(dynamics_unscaled_front_V, dt),
                        dynamics_unscaled_front_V,
                    ]
                )
                self._node_dynamics_unscaled_front[i].h = h[:, :1]
            dynamics_unscaled_middle_V = f[self.index_mstage.m]
            self._node_dynamics_unscaled_middle[i].g = g[:, self.index_mstage.m]
            self._node_dynamics_middle[i].g = np.array(
                [
                    np.full_like(dynamics_unscaled_middle_V, dt),
                    dynamics_unscaled_middle_V,
                ]
            )
            self._node_dynamics_unscaled_middle[i].h = h[:, self.index_mstage.m]
            if self.index_mstage.b:
                dynamics_unscaled_back_V = np.array([f[-1]], dtype=np.float64)
                self._node_dynamics_unscaled_back[i].g = g[:, -1:]
                self._node_dynamics_back[i].g = np.array(
                    [
                        np.full_like(dynamics_unscaled_back_V, dt),
                        dynamics_unscaled_back_V,
                    ]
                )
                self._node_dynamics_unscaled_back[i].h = h[:, -1:]
        forward_gradient_v(self._node_dynamics)
        forward_hessian_phase_v(self._node_dynamics)

        hess_I = []
        for i in range(self.n_x):
            if self.index_mstage.f and self._node_dynamics_front[i].H:
                H_front = np.concatenate(self._node_dynamics_front[i].H)
                hess_I.append(
                    -np.kron(
                        self.I_m_coo.f.data * fct[self.l_d[i] + self.I_m_coo.f.row],
                        H_front,
                    )
                )

            for H_ in self._node_dynamics_middle[i].H:
                hess_I.append(
                    -self.I_m_coo.m.data
                    * fct[self.l_d[i] + self.I_m_coo.m.row]
                    * H_[self.I_m_coo.m.col - self.index_mstage.l_m]
                )

            if self.index_mstage.b and self._node_dynamics_back[i].H:
                H_back = np.concatenate(self._node_dynamics_back[i].H)
                hess_I.append(
                    -np.kron(
                        self.I_m_coo.b.data * fct[self.l_d[i] + self.I_m_coo.b.row],
                        H_back,
                    )
                )

        hess = hess_T + hess_I
        return (
            np.concatenate(hess, dtype=np.float64)
            if hess
            else np.array([], dtype=np.float64)
        )

    def _hess_phase_constraint(self, x: VecFloat, s: VecFloat, fct: VecFloat):
        vb, _ = self._value_basic(x, s)
        for i, f in enumerate(self._func_phase_constraint):
            g = f.G(vb, self.L_m)
            h = f.H(vb, self.L_m)
            if self.index_mstage.f:
                self._node_phase_constraint_front[i].g = g[:, :1]
                self._node_phase_constraint_front[i].h = h[:, :1]
            self._node_phase_constraint_middle[i].g = g[:, self.index_mstage.m]
            self._node_phase_constraint_middle[i].h = h[:, self.index_mstage.m]
            if self.index_mstage.b:
                self._node_phase_constraint_back[i].g = g[:, -1:]
                self._node_phase_constraint_back[i].h = h[:, -1:]
        forward_gradient_v(self._node_phase_constraint)
        forward_hessian_phase_v(self._node_phase_constraint)

        hess = []
        f_ = 0
        for i in range(self._num_phase_constraint):
            if self.index_mstage.f:
                for H_ in self._node_phase_constraint_front[i].H:
                    hess.append(H_ * fct[f_])
            for H_ in self._node_phase_constraint_middle[i].H:
                hess.append(
                    H_ * fct[f_ + self.index_mstage.l_m : f_ + self.index_mstage.r_m]
                )
            if self.index_mstage.b:
                for H_ in self._node_phase_constraint_back[i].H:
                    hess.append(H_ * fct[f_ + self.L_m - 1])
            f_ += self.L_m
        return (
            np.concatenate(hess, dtype=np.float64)
            if hess
            else np.array([], dtype=np.float64)
        )

    def _value_basic_aug(self, x: VecFloat, s: VecFloat) -> tuple[VecFloat, float]:
        for i, bc_info in enumerate(self.info_bc_0):
            x[self.l_v[i]] = self._value_boundary_condition(bc_info, x[self.l_v[i]], s)
        for i, bc_info in enumerate(self.info_bc_f):
            x[self.r_v[i] - 1] = self._value_boundary_condition(
                bc_info, x[self.r_v[i] - 1], s
            )
        x[-2] = self._value_boundary_condition(self.info_t_0, x[-2], s)
        x[-1] = self._value_boundary_condition(self.info_t_f, x[-1], s)
        s_aug = np.repeat(s, self.L_m_aug)
        mt = (x[-1] + x[-2]) / 2
        dt = x[-1] - x[-2]
        t_aug = (self.t_m_aug - 0.5) * dt + mt
        xu_aug = self.V_xu_aug.dot(x[: self.L_xu])
        return np.concatenate([xu_aug, t_aug, s_aug], dtype=np.float64), dt

    def _error_estimation_data_continuous(
        self, x: VecFloat, s: VecFloat
    ) -> tuple[VecFloat, VecFloat]:
        vb_aug, dt = self._value_basic_aug(x, s)
        T_x_aug = self.T_x_aug.dot(x[: self.L_x]).reshape(self.n_x, -1)
        I_f_aug = (
            np.array(
                [self.I_m_aug.dot(f_d.F(vb_aug, self.L_m_aug)) for f_d in self.F_d]
            )
            * dt
        )
        return T_x_aug, I_f_aug

    def _error_estimation_data_discontinuous(
        self, x: VecFloat, s: VecFloat
    ) -> VecFloat:
        vb, _ = self._value_basic(x, s)
        return np.array([f_bb.F(vb, self.L_m) for f_bb in self.F_b], dtype=np.float64)

    def _error_check_interval_continuous(
        self, T_x_aug, I_f_aug, atol, rtol, mtol
    ) -> VecBool:
        ec_c = np.ones(self._num_interval, dtype=bool)
        for i_n in range(self._num_interval):
            if self._mesh[i_n + 1] - self._mesh[i_n] < mtol:
                continue
            l = self.l_m_aug[i_n]
            r = self.r_m_aug[i_n]
            ec_c[i_n] = np.allclose(
                T_x_aug[:, l:r], I_f_aug[:, l:r], atol=atol, rtol=rtol
            )
        return ec_c

    def _error_check_interval_discontinuous(self, f_bb, dtol, mtol) -> VecBool:
        ec_dc = np.ones(self._num_interval, dtype=bool)
        for i_n in range(self._num_interval):
            if self._mesh[i_n + 1] - self._mesh[i_n] < mtol:
                continue
            for i_b in range(self.n_b):
                f_bb_ = f_bb[i_b, self.l_m[i_n] : self.r_m[i_n]]
                near_0 = np.all(f_bb_ < dtol)
                near_1 = np.all(f_bb_ > 1 - dtol)
                ok = near_0 or near_1
                ec_dc[i_n] = ec_dc[i_n] and ok
        return ec_dc

    def check_continuous(
        self,
        variable,
        static_parameter: Optional[Iterable[float]] = None,
        absolute_tolerance_continuous: float = 1e-8,
        relative_tolerance_continuous: float = 1e-8,
        tolerance_mesh: float = 1e-4,
    ) -> bool:
        """Check the continuous error.

        Args:
            variable: Variable to be checked.
            static_parameter: Static parameter to be checked. Set to ``None`` if the phase has no static parameters.
            absolute_tolerance_continuous: Absolute tolerance for continuous error.
            relative_tolerance_continuous: Relative tolerance for continuous error.
            tolerance_mesh: Skip the check if the mesh width is smaller than this value.

        Returns:
            ``True`` if the error is within the tolerance, ``False`` otherwise.
        """
        if self._num_static_parameter and static_parameter is None:
            raise ValueError(
                "phase has static parameters, but the value of static parameters is not given"
            )
        if static_parameter is None:
            static_parameter = []
        x = variable.data
        s = np.array(static_parameter, dtype=np.float64)
        T_x_aug, I_f_aug = self._error_estimation_data_continuous(x, s)
        ec_c = self._error_check_interval_continuous(
            T_x_aug,
            I_f_aug,
            absolute_tolerance_continuous,
            relative_tolerance_continuous,
            tolerance_mesh,
        )
        return bool(np.all(ec_c))

    def check_discontinuous(
        self,
        variable,
        static_parameter: Optional[Iterable[float]] = None,
        tolerance_discontinuous: float = 1e-3,
        tolerance_mesh: float = 1e-4,
    ) -> bool:
        """Check the discontinuous error.

        Args:
            variable: Variable to be checked.
            static_parameter: Static parameter to be checked. Set to ``None`` if the phase has no static parameters.
            tolerance_discontinuous: In each subinterval, after scaling to ``[0, 1]``, the bang-bang control functions
                should either be less than ``tolerance_discontinuous`` or greater than ``1 - tolerance_discontinuous``
                simultaneously.
            tolerance_mesh: Skip the check if the mesh width is smaller than this value.

        Returns:
            ``True`` if the error is within the tolerance, ``False`` otherwise.
        """
        if self._num_static_parameter and static_parameter is None:
            raise ValueError(
                "phase has static parameters, but the value of static parameters is not given"
            )
        if static_parameter is None:
            static_parameter = []
        x = variable.data
        s = np.array(static_parameter, dtype=np.float64)
        f_bb = self._error_estimation_data_discontinuous(x, s)
        ec_dc = self._error_check_interval_discontinuous(
            f_bb, tolerance_discontinuous, tolerance_mesh
        )
        discontinuous_check_passed = bool(np.all(ec_dc))
        if discontinuous_check_passed:
            self._discontinuous_check_passed = True
        return discontinuous_check_passed

    def check(
        self,
        variable,
        static_parameter: Optional[Iterable[float]] = None,
        absolute_tolerance_continuous: float = 1e-8,
        relative_tolerance_continuous: float = 1e-8,
        tolerance_discontinuous: float = 1e-3,
        tolerance_mesh: float = 1e-4,
    ) -> bool:
        """Check the continuous and discontinuous error.

        Args:
            variable: Variable to be checked.
            static_parameter: Static parameter to be checked. Set to ``None`` if the phase has no static parameters.
            absolute_tolerance_continuous: Absolute tolerance for continuous error.
            relative_tolerance_continuous: Relative tolerance for continuous error.
            tolerance_discontinuous: In each subinterval, after scaling to ``[0, 1]``, the bang-bang control functions
                should either be less than ``tolerance_discontinuous`` or greater than ``1 - tolerance_discontinuous``
                simultaneously.
            tolerance_mesh: Skip the check if the mesh width is smaller than this value.

        Returns:
            ``True`` if the error is within the tolerance, ``False`` otherwise.
        """
        if self._discontinuous_check_passed:
            return self.check_continuous(
                variable,
                static_parameter,
                absolute_tolerance_continuous,
                relative_tolerance_continuous,
                tolerance_mesh,
            )
        else:
            return self.check_discontinuous(
                variable,
                static_parameter,
                relative_tolerance_continuous,
                tolerance_mesh,
            ) and self.check_continuous(
                variable,
                static_parameter,
                absolute_tolerance_continuous,
                relative_tolerance_continuous,
                tolerance_mesh,
            )

    def refine_continuous(
        self,
        variable,
        static_parameter: Optional[Iterable[float]] = None,
        absolute_tolerance_continuous: float = 1e-8,
        relative_tolerance_continuous: float = 1e-8,
        num_point_min: int = 6,
        num_point_max: int = 12,
        mesh_length_min: float = 1e-3,
        mesh_length_max: float = 1.0,
    ) -> None:
        """Adjust the mesh and the number of interpolation points to match the
        continuous error tolerance.

        Args:
            variable: Variable of the previous iteration.
            static_parameter: Static parameter of the previous iteration.
                Set to ``None`` if the phase has no static parameters.
            absolute_tolerance_continuous: Absolute tolerance for continuous error.
            relative_tolerance_continuous: Relative tolerance for continuous error.
            num_point_min: Minimum number of interpolation points.
            num_point_max: Maximum number of interpolation points.
            mesh_length_min: Minimum mesh length.
            mesh_length_max: Maximum mesh length.
        """
        if self.check_continuous(
            variable,
            static_parameter,
            absolute_tolerance_continuous,
            relative_tolerance_continuous,
            mesh_length_min,
        ):
            return
        if static_parameter is None:
            static_parameter = []
        x = variable.data
        s = np.array(static_parameter, dtype=np.float64)
        T_x_aug, I_f_aug = self._error_estimation_data_continuous(x, s)
        ec_c = self._error_check_interval_continuous(
            T_x_aug,
            I_f_aug,
            absolute_tolerance_continuous,
            relative_tolerance_continuous,
            mesh_length_min,
        )

        mesh_new = []
        num_point_new = []
        for i_n in range(self._num_interval):
            if ec_c[i_n]:
                mesh_new.append(self._mesh[i_n])
                num_point_new.append(self._num_point[i_n])
                continue
            T_x_aug_n = T_x_aug[:, self.l_m_aug[i_n] : self.r_m_aug[i_n]]
            I_f_aug_n = I_f_aug[:, self.l_m_aug[i_n] : self.r_m_aug[i_n]]
            absolute_error = np.abs(T_x_aug_n - I_f_aug_n)
            I_f_aug_max = np.max(np.abs(I_f_aug_n), axis=1).reshape(-1, 1)
            relative_error = absolute_error / (1.0 + I_f_aug_max)
            relative_error_max = np.max(relative_error)
            num_new_point = max(
                int(
                    np.ceil(
                        np.log(relative_error_max / relative_tolerance_continuous)
                        / np.log(self._num_point[i_n])
                    )
                ),
                1,
            )
            if self._num_point[i_n] + num_new_point <= num_point_max:
                mesh_new.append(self._mesh[i_n])
                num_point_new.append(self._num_point[i_n] + num_new_point)
            else:
                mesh_d = self._mesh[i_n + 1] - self._mesh[i_n]
                num_interval_min = int(np.ceil(mesh_d / mesh_length_max))
                num_interval_max = max(
                    int(np.floor(mesh_d / mesh_length_min)), 1
                )  # at least one interval whatever
                num_interval = max(
                    int(
                        np.ceil((self._num_point[i_n] + num_new_point) / num_point_min)
                    ),
                    2,
                )
                num_interval = min(num_interval, num_interval_max)
                num_interval = max(num_interval, num_interval_min)
                mesh_new_ = np.linspace(
                    self._mesh[i_n], self._mesh[i_n + 1], num_interval, endpoint=False
                )
                for mesh_ in mesh_new_:
                    mesh_new.append(mesh_)
                    num_point_new.append(num_point_min)
        mesh_new.append(1.0)

        discontinuous_check_passed = self._discontinuous_check_passed
        self.set_discretization(mesh_new, num_point_new)
        self._discontinuous_check_passed = discontinuous_check_passed

    def refine_discontinuous(
        self,
        variable,
        static_parameter: Optional[Iterable[float]] = None,
        tolerance_discontinuous: float = 1e-3,
        num_point_min: int = 6,
        num_point_max: int = 12,
        mesh_length_min: float = 1e-3,
        mesh_length_max: float = 1.0,
    ) -> None:
        """Adjust the mesh and the number of interpolation points to match the
        discontinuous error tolerance.

        Args:
            variable: Variable of the previous iteration.
            static_parameter: Static parameter of the previous iteration.
                Set to ``None`` if the phase has no static parameters.
            tolerance_discontinuous: In each subinterval, after scaling to ``[0, 1]``, the bang-bang control functions
                should either be less than ``tolerance_discontinuous`` or greater than ``1 - tolerance_discontinuous``
                simultaneously.
            num_point_min: Minimum number of interpolation points.
            num_point_max: Maximum number of interpolation points.
            mesh_length_min: Minimum mesh length.
            mesh_length_max: Maximum mesh length.
        """
        if self.check_discontinuous(
            variable, static_parameter, tolerance_discontinuous, mesh_length_min
        ):
            return

        # fixed parameters
        shock_threshold = 0.1
        factor = 1

        if static_parameter is None:
            static_parameter = []
        x = variable.data
        s = np.array(static_parameter, dtype=np.float64)
        f_bb = self._error_estimation_data_discontinuous(x, s)

        f_bb_mean = np.empty(
            (self._num_bang_bang, self._num_interval), dtype=np.float64
        )
        for m_ in range(self._num_interval):
            w = self.w_aug[m_]
            for bb_ in range(self._num_bang_bang):
                f_bb_mean[bb_, m_] = w.dot(f_bb[bb_][self.l_m[m_] : self.r_m[m_]]) / 2

        mesh_new = []
        no_shock = set()
        to_delete = set()
        for p_ in range(1, self._num_interval):
            for bb_ in range(self._num_bang_bang):
                if (
                    np.abs(f_bb_mean[bb_, p_ - 1] - f_bb_mean[bb_, p_])
                    > shock_threshold
                ):
                    break
            else:
                no_shock.add(p_)

        ok = np.zeros((self._num_bang_bang, self._num_interval), dtype=np.bool_)
        index_mid = self._num_interval // 2

        for bb_ in range(self._num_bang_bang):
            f_bb_ = f_bb[bb_]
            for m_ in range(index_mid):
                mesh_l = self._mesh[m_]
                mesh_r = self._mesh[m_ + 1]
                mesh_m = (mesh_l + mesh_r) / 2
                mesh_d = mesh_r - mesh_l
                if np.any(f_bb_[self.l_m[m_] : self.r_m[m_]] < 0.5) and np.any(
                    f_bb_[self.l_m[m_] : self.r_m[m_]] > 0.5
                ):
                    roots = (
                        _find_root_discontinuous(
                            f_bb_[self.l_m[m_] : self.r_m[m_]] - 0.5, self.P
                        )
                        * mesh_d
                        / 2
                        + mesh_m
                    )
                    for root in roots:
                        if root < mesh_l + mesh_length_min:
                            if m_ not in to_delete:
                                to_delete.add(m_)
                                mesh_new.append(root)
                                ok[bb_, m_] = True
                        elif root > mesh_r - mesh_length_min:
                            if m_ + 1 not in to_delete:
                                to_delete.add(m_ + 1)
                                mesh_new.append(root)
                                ok[bb_, m_] = True
                        else:
                            mesh_new.append(root)
                            ok[bb_, m_] = True
                elif np.all(
                    f_bb_[self.l_m[m_] : self.r_m[m_]] < tolerance_discontinuous
                ) or np.all(
                    f_bb_[self.l_m[m_] : self.r_m[m_]] > 1 - tolerance_discontinuous
                ):
                    ok[bb_, m_] = True
            for m_ in reversed(range(index_mid, self._num_interval)):
                mesh_l = self._mesh[m_]
                mesh_r = self._mesh[m_ + 1]
                mesh_m = (mesh_l + mesh_r) / 2
                mesh_d = mesh_r - mesh_l
                if np.any(f_bb_[self.l_m[m_] : self.r_m[m_]] < 0.5) and np.any(
                    f_bb_[self.l_m[m_] : self.r_m[m_]] > 0.5
                ):
                    roots = (
                        _find_root_discontinuous(
                            f_bb_[self.l_m[m_] : self.r_m[m_]] - 0.5, self.P
                        )
                        * mesh_d
                        / 2
                        + mesh_m
                    )
                    for root in reversed(roots):
                        if root > mesh_r - mesh_length_min:
                            if m_ + 1 not in to_delete:
                                to_delete.add(m_ + 1)
                                mesh_new.append(root)
                                ok[bb_, m_] = True
                        elif root < mesh_l + mesh_length_min:
                            if m_ not in to_delete:
                                to_delete.add(m_)
                                mesh_new.append(root)
                                ok[bb_, m_] = True
                        else:
                            mesh_new.append(root)
                            ok[bb_, m_] = True
                elif np.all(
                    f_bb_[self.l_m[m_] : self.r_m[m_]] < tolerance_discontinuous
                ) or np.all(
                    f_bb_[self.l_m[m_] : self.r_m[m_]] > 1 - tolerance_discontinuous
                ):
                    ok[bb_, m_] = True

        for bb_ in range(self._num_bang_bang):
            f_bb_ = f_bb[bb_]
            for m_ in range(index_mid):
                if not ok[bb_, m_]:
                    mesh_l = self._mesh[m_]
                    mesh_r = self._mesh[m_ + 1]
                    mesh_m = (mesh_l + mesh_r) / 2
                    mesh_d = mesh_r - mesh_l

                    r_i = np.abs(f_bb_mean[bb_, m_] - round(f_bb_mean[bb_, m_]))
                    r_s = r_i * factor

                    f_lr = f_bb_[self.l_m[m_]]
                    f_rl = f_bb_[self.r_m[m_] - 1]
                    f_ll = None if m_ == 0 else f_bb_[self.l_m[m_] - 1]
                    f_rr = None if m_ == self._num_interval - 1 else f_bb_[self.r_m[m_]]

                    ok_l, ok_r = _check_boundary_discontinuous(
                        f_ll, f_lr, f_rl, f_rr, tolerance_discontinuous
                    )

                    if not ok_l:
                        if m_ not in to_delete:
                            mesh_new.append(mesh_l + r_s * mesh_d)
                            to_delete.add(m_)
                        else:
                            index_r = m_ + 1
                            while index_r in no_shock:
                                index_r += 1
                            if (
                                index_r not in to_delete
                                and index_r < self._num_interval
                            ):
                                mesh_r_2 = self._mesh[index_r]
                                mesh_new.append(mesh_r_2 - r_s * mesh_d)
                                to_delete.add(index_r)
                    if not ok_r:
                        if m_ + 1 not in to_delete:
                            mesh_new.append(mesh_r - r_s * mesh_d)
                            to_delete.add(m_ + 1)
                        else:
                            index_l = m_
                            while index_l in no_shock:
                                index_l -= 1
                            if index_l not in to_delete and index_l > 0:
                                mesh_l_2 = self._mesh[index_l]
                                mesh_new.append(mesh_l_2 + r_s * mesh_d)
                                to_delete.add(index_l)
            for m_ in reversed(range(index_mid, self._num_interval)):
                if not ok[bb_, m_]:
                    mesh_l = self._mesh[m_]
                    mesh_r = self._mesh[m_ + 1]
                    mesh_m = (mesh_l + mesh_r) / 2
                    mesh_d = mesh_r - mesh_l

                    r_i = np.abs(f_bb_mean[bb_, m_] - round(f_bb_mean[bb_, m_]))
                    r_s = r_i * factor

                    f_lr = f_bb_[self.l_m[m_]]
                    f_rl = f_bb_[self.r_m[m_] - 1]
                    f_ll = None if m_ == 0 else f_bb_[self.l_m[m_] - 1]
                    f_rr = None if m_ == self._num_interval - 1 else f_bb_[self.r_m[m_]]

                    ok_l, ok_r = _check_boundary_discontinuous(
                        f_ll, f_lr, f_rl, f_rr, tolerance_discontinuous
                    )

                    if not ok_r:
                        if m_ + 1 not in to_delete:
                            mesh_new.append(mesh_r - r_s * mesh_d)
                            to_delete.add(m_ + 1)
                        else:
                            index_l = m_
                            while index_l in no_shock:
                                index_l -= 1
                            if index_l not in to_delete and index_l > 0:
                                mesh_l_2 = self._mesh[index_l]
                                mesh_new.append(mesh_l_2 + r_s * mesh_d)
                                to_delete.add(index_l)
                    if not ok_l:
                        if m_ not in to_delete:
                            mesh_new.append(mesh_l + r_s * mesh_d)
                            to_delete.add(m_)
                        else:
                            index_r = m_ + 1
                            while index_r in no_shock:
                                index_r += 1
                            if (
                                index_r not in to_delete
                                and index_r < self._num_interval
                            ):
                                mesh_r_2 = self._mesh[index_r]
                                mesh_new.append(mesh_r_2 - r_s * mesh_d)
                                to_delete.add(index_r)

        for p_ in range(1, self._num_interval):
            if p_ not in to_delete and p_ not in no_shock:
                mesh_new.append(self._mesh[p_])

        mesh_new_2 = _mesh_gen_discontinuous(
            mesh_new, self._mesh[1:-1], mesh_length_min, mesh_length_max
        )
        num_points_new = []
        for i in range(len(mesh_new_2) - 1):
            if mesh_new_2[i + 1] - mesh_new_2[i] < min(1e-2, mesh_length_min * 10):
                num_points_new.append(num_point_min)
            else:
                num_points_new.append(num_point_max)

        self.set_discretization(mesh_new_2, num_points_new)

    def refine(
        self,
        variable,
        static_parameter: Optional[Iterable[float]] = None,
        absolute_tolerance_continuous: float = 1e-8,
        relative_tolerance_continuous: float = 1e-8,
        tolerance_discontinuous: float = 1e-3,
        num_point_min: int = 6,
        num_point_max: int = 12,
        mesh_length_min: float = 1e-3,
        mesh_length_max: float = 1.0,
    ) -> None:
        """Adjust the mesh and the number of interpolation points to match the
        error tolerances.

        If the discontinuous error is not within the tolerance, refine for discontinuous error.
        Otherwise, if the continuous error is not within the tolerance, refine for continuous error.
        At most one of the continuous or discontinuous refinements will be performed.

        Args:
            variable: Variable of the previous iteration.
            static_parameter: Static parameter of the previous iteration.
                Set to ``None`` if the phase has no static parameters.
            absolute_tolerance_continuous: Absolute tolerance for continuous error.
            relative_tolerance_continuous: Relative tolerance for continuous error.
            tolerance_discontinuous: In each subinterval, after scaling to ``[0, 1]``, the bang-bang control functions
                should either be less than ``tolerance_discontinuous`` or greater than ``1 - tolerance_discontinuous``
                simultaneously.
            num_point_min: Minimum number of interpolation points.
            num_point_max: Maximum number of interpolation points.
            mesh_length_min: Minimum mesh length.
            mesh_length_max: Maximum mesh length.
        """
        if not self._discontinuous_check_passed and not self.check_discontinuous(
            variable, static_parameter, tolerance_discontinuous, mesh_length_min
        ):
            self.refine_discontinuous(
                variable,
                static_parameter,
                tolerance_discontinuous,
                num_point_min,
                num_point_max,
                mesh_length_min,
                mesh_length_max,
            )
        elif not self.check_continuous(
            variable,
            static_parameter,
            absolute_tolerance_continuous,
            relative_tolerance_continuous,
            mesh_length_min,
        ):
            self.refine_continuous(
                variable,
                static_parameter,
                absolute_tolerance_continuous,
                relative_tolerance_continuous,
                num_point_min,
                num_point_max,
                mesh_length_min,
                mesh_length_max,
            )

    @property
    @abstractmethod
    def _class_discretization(self) -> type[DiscretizationBase]:
        pass

    @property
    def n_x(self) -> int:
        """Number of state variables."""
        return self._num_state

    @property
    def x(self) -> list[sp.Symbol]:
        """:class:`sympy.Symbol` s of state variables."""
        return self._symbol_state

    @property
    def n_u(self) -> int:
        """Number of control variables."""
        return self._num_control

    @property
    def u(self) -> list[sp.Symbol]:
        """:class:`sympy.Symbol` s of control variables."""
        return self._symbol_control

    @property
    def n(self) -> int:
        """Number of state and control variables."""
        return self._num_variable

    @property
    def n_s(self) -> int:
        """Number of static parameters."""
        return self._num_static_parameter

    @property
    def s(self) -> list[sp.Symbol]:
        """:class:`sympy.Symbol` s of static parameters."""
        return self._symbol_static_parameter

    @property
    def t(self) -> sp.Symbol:
        """The :class:`sympy.Symbol` representing the time."""
        return self._symbol_time

    @property
    def F_d(self) -> list[FastFunc]:
        """:class:`pockit.base.fastfunc.FastFunc` s of dynamics."""
        return self._func_dynamics

    @property
    def n_d(self) -> int:
        """Number of dynamics."""
        return self._num_state

    @property
    def F_I(self) -> list[FastFunc]:
        """:class:`pockit.base.fastfunc.FastFunc` s of integrals."""
        return self._func_integral

    @property
    def n_I(self) -> int:
        """Number of integrals."""
        return self._num_integral

    @property
    def I(self) -> list[sp.Symbol]:
        """:class:`sympy.Symbol` s of integrals."""
        return self._symbol_integral

    @property
    def F_c(self) -> list[FastFunc]:
        """:class:`pockit.base.fastfunc.FastFunc` s of phase constraints."""
        return self._func_phase_constraint

    @property
    def n_c(self) -> int:
        """Number of phase constraints."""
        return self._num_phase_constraint

    @property
    def v_lb(self) -> VecFloat:
        """Lower bounds of optimization variables."""
        return self._lower_bound_variable

    @property
    def v_ub(self) -> VecFloat:
        """Upper bounds of optimization variables."""
        return self._upper_bound_variable

    @property
    def c_lb(self) -> VecFloat:
        """Lower bounds of optimization constraints."""
        return self._lower_bound_phase_constraint

    @property
    def c_ub(self) -> VecFloat:
        """Upper bounds of optimization constraints."""
        return self._upper_bound_phase_constraint

    @property
    def s_b(self) -> list[tuple[int, float, float]]:
        """Bounds of static parameters."""
        return self._static_parameter_bounds_phase

    @property
    def bc_0(self) -> list[None | float | sp.Expr]:
        """Initial boundary conditions."""
        return self._initial_value

    @property
    def bc_f(self) -> list[None | float | sp.Expr]:
        """Terminal boundary conditions."""
        return self._terminal_value

    @property
    def F_b(self) -> list[FastFunc]:
        """:class:`pockit.base.fastfunc.FastFunc` s of bang-bang constraints.

        (scaled to ``[0, 1]``.)
        """
        return self._func_bang_bang_control

    @property
    def n_b(self) -> int:
        """Number of bang-bang constraints."""
        return self._num_bang_bang

    @property
    def t_0(self) -> None | float | sp.Expr:
        """Initial time."""
        return self._initial_time

    @property
    def t_f(self) -> None | float | sp.Expr:
        """Terminal time."""
        return self._terminal_time

    @property
    def N(self) -> int:
        """Number of subintervals."""
        return self._num_interval

    @property
    def ok(self) -> bool:
        """Whether the phase is fully configured."""
        return (
            self._dynamics_set
            and self._boundary_condition_set
            and self._discretization_set
        )

    @property
    def index_state(self) -> IndexNode:
        """Index partition of the state variables."""
        return self._object_discretization.index_state

    @property
    def index_control(self) -> IndexNode:
        """Index partition of the control variables."""
        return self._object_discretization.index_control

    @property
    def index_mstage(self) -> IndexNode:
        """Index partition of the middle-stage variables."""
        return self._object_discretization.index_mstage

    @property
    def l_v(self) -> VecInt:
        """The left index of the state and control variables."""
        return self._object_discretization.l_v

    @property
    def r_v(self) -> VecInt:
        """The right index of the state and control variables (exclusive)."""
        return self._object_discretization.r_v

    @property
    def t_m(self) -> VecFloat:
        """Position of all interpolation nodes in the middle stage, scaled to
        ``[0, 1]``."""
        return self._object_discretization.t_m

    @property
    def l_m(self) -> VecInt:
        """The left index of subintervals in the middle stage."""
        return self._object_discretization.l_m

    @property
    def r_m(self) -> VecInt:
        """The right index of subintervals in the middle stage (exclusive)."""
        return self._object_discretization.r_m

    @property
    def L_m(self) -> int:
        """The number of interpolation points in the middle stage."""
        return self._object_discretization.L_m

    @property
    def w_m(self) -> VecFloat:
        """The integration weights of the middle stage."""
        return self._object_discretization.w_m

    @property
    def f_v2m(self) -> Callable[[VecFloat], VecFloat]:
        """Function turns the state and control variables into the middle
        stage."""
        return self._object_discretization.f_v2m

    @property
    def T_v(self) -> scipy.sparse.csr_array:
        """Translation matrix of a state variable to eliminate the integration
        coefficient."""
        return self._object_discretization.T_v

    @property
    def T_v_coo(self) -> CooMatrixNode:
        """Translation matrix of a state variable in COO format partitioned by
        the index."""
        return self._object_discretization.T_v_coo

    @property
    def I_m(self) -> scipy.sparse.csr_array:
        """Integration matrix of a middle-stage variable."""
        return self._object_discretization.I_m

    @property
    def I_m_coo(self) -> CooMatrixNode:
        """Integration matrix of a middle-stage variable in COO format
        partitioned by the index."""
        return self._object_discretization.I_m_coo

    @property
    def l_d(self) -> VecInt:
        """The left index of the dynamic constraints of each state variable."""
        return self._object_discretization.l_d

    @property
    def r_d(self) -> VecInt:
        """The right index of the dynamic constraints of each state variable
        (exclusive)."""
        return self._object_discretization.r_d

    @property
    def t_m_aug(self) -> VecFloat:
        """Position of all interpolation nodes in the middle stage, scaled to
        ``[0, 1]`` with one additional interpolation point in each
        subinterval."""
        return self._object_discretization.t_m_aug

    @property
    def l_m_aug(self) -> VecInt:
        """The left index of subintervals in the middle stage, with one
        additional interpolation point in each subinterval."""
        return self._object_discretization.l_m_aug

    @property
    def r_m_aug(self) -> VecInt:
        """The right index of subintervals in the middle stage (exclusive),
        with one additional interpolation point in each subinterval."""
        return self._object_discretization.r_m_aug

    @property
    def L_m_aug(self) -> int:
        """The number of interpolation points in the middle stage, with one
        additional interpolation point in each subinterval."""
        return self._object_discretization.L_m_aug

    @property
    def w_aug(self) -> list[VecFloat]:
        """The integration weights of each subinterval in the middle stage,
        with one additional interpolation point in each subinterval."""
        return self._object_discretization.w_aug

    @property
    def P(self) -> Callable[[int], VecFloat]:
        """Function to compute the polynomial matrix given the number of
        interpolation points."""
        return self._object_discretization.P

    @property
    def V_xu_aug(self) -> scipy.sparse.csr_array:
        """The value matrix translates all state and control variables to the
        middle stage with one additional interpolation point in each
        subinterval."""
        return self._object_discretization.V_xu_aug

    @property
    def T_x_aug(self) -> scipy.sparse.csr_array:
        """The translation matrix of all state variables, with one additional
        interpolation point in each subinterval."""
        return self._object_discretization.T_x_aug

    @property
    def I_m_aug(self) -> scipy.sparse.csr_array:
        """The integration matrix of a middle-stage variable, with one
        additional interpolation point in each subinterval."""
        return self._object_discretization.I_m_aug

    @property
    def t_x(self) -> VecFloat:
        """Position of all interpolation nodes of a state variable, scaled to
        ``[0, 1]``."""
        return self._object_discretization.t_x

    @property
    def t_u(self) -> VecFloat:
        """Position of all interpolation nodes of a control variable, scaled to
        ``[0, 1]``."""
        return self._object_discretization.t_u

    @property
    def l_x(self) -> VecInt:
        """The left index of subintervals of a state variable."""
        return self._object_discretization.l_x

    @property
    def r_x(self) -> VecInt:
        """The right index of subintervals of a state variable (exclusive)."""
        return self._object_discretization.r_x

    @property
    def l_u(self) -> VecInt:
        """The left index of subintervals of a control variable."""
        return self._object_discretization.l_u

    @property
    def r_u(self) -> VecInt:
        """The right index of subintervals of a control variable
        (exclusive)."""
        return self._object_discretization.r_u

    @property
    def L_x(self) -> int:
        """Length of all state variables."""
        return self.r_v[self.n_x - 1]

    @property
    def L_xu(self) -> int:
        """Length of all state and control variables."""
        return self.r_v[-1]

    @property
    def L(self) -> int:
        """Length of all state, control, and time variables."""
        return self.r_v[-1] + 2


def _find_root_discontinuous(y: VecFloat, func_P) -> VecFloat:
    n = len(y)
    mat_P = func_P(n)
    coef = mat_P @ y
    roots = np.roots(coef)
    filtered = []
    for root in roots:
        if np.isreal(root) and -1.0 < root < 1.0:
            filtered.append(root.real)
    filtered.sort()
    return np.array(filtered, dtype=np.float64)


def _check_boundary_discontinuous(f_ll, f_lr, f_rl, f_rr, tolerance_discontinuous):
    def _classify(f, tolerance_discontinuous):
        if f is None:
            return -100
        elif f < tolerance_discontinuous:
            return 0
        elif f > 1 - tolerance_discontinuous:
            return 1
        else:
            return 10

    c_ll = _classify(f_ll, tolerance_discontinuous)
    c_lr = _classify(f_lr, tolerance_discontinuous)
    c_rl = _classify(f_rl, tolerance_discontinuous)
    c_rr = _classify(f_rr, tolerance_discontinuous)

    c_l = c_ll + c_lr
    c_r = c_rl + c_rr

    ok_l = c_l <= 2
    ok_r = c_r <= 2

    return ok_l, ok_r


def _mesh_gen_discontinuous(mesh_new, mesh_old, mesh_length_min, mesh_length_max):
    mesh_new.sort()
    mesh = [0.0]
    for mesh_ in mesh_new:
        if mesh_length_min < mesh_ < 1 - mesh_length_min:
            mesh.append(mesh_)
    mesh.append(1.0)

    mesh_clean = [0.0]
    for i in range(len(mesh) - 1):
        mesh_ = mesh[i + 1]
        if mesh_ - mesh_clean[-1] < mesh_length_min:  # mesh too dense
            if mesh_clean[-1] in mesh_old:
                mesh_clean[-1] = mesh_
            elif mesh_ in mesh_old:
                pass
            else:
                mesh_clean[-1] = (mesh_ + mesh_clean[-1]) / 2
            continue
        if mesh_ - mesh_clean[-1] > mesh_length_max:  # mesh too sparse
            mesh_last = mesh_clean[-1]
            n_split = int(np.ceil((mesh_ - mesh_last) / mesh_length_max))
            for j in range(n_split):
                mesh_clean.append(mesh_last + (mesh_ - mesh_last) * (j + 1) / n_split)
            continue
        mesh_clean.append(mesh_)
    return np.array(mesh_clean, dtype=np.float64)
