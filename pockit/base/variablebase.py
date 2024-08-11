from abc import ABC, abstractmethod
from typing import Self, Type

import scipy.interpolate

from .phasebase import PhaseBase, BcType, lr_c
from .vectypes import *


def V_interpolation(x_old: VecFloat, x_new: VecFloat) -> VecFloat:
    """Value matrix with interpolation nodes ``x_old`` and evaluation nodes
    ``x_new``."""
    if not len(x_new):  # empty
        return np.array([], dtype=np.float64).reshape(0, len(x_old))
    if np.array_equal(
        x_new, x_old
    ):  # speed up for discontinuous adaptation with no change
        return np.eye(len(x_old), dtype=np.float64)
    if np.array_equal(
        x_new, x_old[1:]
    ):  # speed up for continuous adaptation with no change
        return np.hstack(
            [
                np.zeros((len(x_new), 1), dtype=np.float64),
                np.eye(len(x_new), dtype=np.float64),
            ]
        )
    # scale to [0, 1]
    x_new = (x_new - x_old[0]) / (x_old[-1] - x_old[0])
    x_old = (x_old - x_old[0]) / (x_old[-1] - x_old[0])
    V = []
    for i in range(len(x_old)):
        y = np.zeros_like(x_old)
        y[i] = 1
        poly = scipy.interpolate.lagrange(x_old, y)
        V.append(np.polyval(poly, x_new))
    return np.array(V, dtype=np.float64).T


def D_interpolation(x_old: VecFloat, x_new: VecFloat) -> VecFloat:
    """Derivative matrix with interpolation nodes ``x_old`` and evaluation
    nodes ``x_new``."""
    if not len(x_new):
        return np.array([], dtype=np.float64).reshape(0, len(x_old))
    # scale to [0, 1]
    width = x_old[-1] - x_old[0]
    x_new = (x_new - x_old[0]) / width
    x_old = (x_old - x_old[0]) / width

    D = []
    for i in range(len(x_old)):
        y = np.zeros_like(x_old)
        y[i] = 1
        poly = scipy.interpolate.lagrange(x_old, y)
        deriv_poly = np.polyder(poly)
        D.append(np.polyval(deriv_poly, x_new))
    return np.array(D, dtype=np.float64).T / width


class BatchIndexArray:
    """Utility class for firstly indexing a batch of elements and then further
    indexing the value."""

    def __init__(self, data: VecFloat, l_index: VecInt, r_index: VecInt) -> None:
        """
        Args:
            data: The underlying data array.
            l_index: The left indices of each batch.
            r_index: The right indices of each batch (exclusive).
        """
        if not len(l_index) == len(r_index):
            raise ValueError("l_index and r_index must have the same length")
        self._data = data
        self._l_index = l_index
        self._r_index = r_index
        self._n = len(l_index)

    def __getitem__(self, i: int) -> VecFloat:
        return self._data[self._l_index[i] : self._r_index[i]]

    def __setitem__(self, i: int, value: VecFloat) -> None:
        self._data[self._l_index[i] : self._r_index[i]] = value

    def __len__(self) -> int:
        return self._n


class VariableBase(ABC):
    """Optimization variable for a discretized phase.

    ``Variable`` objects provide two kinds of interfaces:
    - Plain 1D array for passing to the solver;
    - Methods for quickly accessing specific variables for users to set and extract corresponding values.

    Besides, ``Variable`` provides methods to generate interpolation matrices for mesh adaption and plotting.

    Generally, users need not create ``Variable`` objects directly.
    A better way is to use the func:`constant_guess` and func:`linear_guess` functions to generate a starting point and possibly adjust it manually.
    """

    def __init__(self, phase: PhaseBase, data: VecFloat) -> None:
        """
        Args:
            phase: The ``Phase`` object to create the ``Variable`` for.
            data: The underlying data array.
        """
        self._data = data
        self._l_v = phase.l_v
        self._r_v = phase.r_v
        self._n_x = phase.n_x
        self._n_u = phase.n_u
        self._n = phase.n
        self._array_state = BatchIndexArray(
            data, self._l_v[: self._n_x], self._r_v[: self._n_x]
        )
        self._array_control = BatchIndexArray(
            data, self._l_v[self._n_x :], self._r_v[self._n_x :]
        )

        self._mesh = phase._mesh
        self._num_point = phase._num_point
        self._N = phase.N

        self._t_x = phase.t_x
        self._t_u = phase.t_u
        self._l_x = phase.l_x
        self._r_x = phase.r_x
        self._l_u = phase.l_u
        self._r_u = phase.r_u

    @staticmethod
    def _almost_equal(a, b) -> bool:
        return np.isclose(a, b, rtol=1e-8, atol=1e-8)

    def _interval_partition(self, t: VecFloat) -> list[list[np.float64]]:
        interval_partition = [[] for _ in range(self._N)]
        n_old = 0
        for i, t_ in enumerate(t):
            while self._mesh[n_old + 1] < t_ and not self._almost_equal(
                self._mesh[n_old + 1], t_
            ):
                n_old += 1
            if (
                self._almost_equal(self._mesh[n_old + 1], t_)
                and i > 0
                and self._almost_equal(t[i - 1], t_)
                and n_old + 1 < self._N
            ):
                n_old += 1
            interval_partition[n_old].append(t_)
        return interval_partition

    def _guard_t(self, t: VecFloat) -> VecFloat:
        for i in range(len(t) - 1):
            if not np.isclose(t[i], t[i + 1]) and t[i] > t[i + 1]:
                raise ValueError("t is not in ascending order")
        if t[0] < self.t_0:
            if np.isclose(t[0], self.t_0, rtol=0, atol=1e-8):
                t[0] = self.t_0
            else:
                raise ValueError("t[0] must be equal or greater than t_0")
        if t[-1] > self.t_f:
            if np.isclose(t[-1], self.t_f, rtol=0, atol=1e-8):
                t[-1] = self.t_f
            else:
                raise ValueError("t[-1] must be equal or smaller than t_f")
        return (t - self.t_0) / (self.t_f - self.t_0)

    @staticmethod
    def _assemble_c(num_point, V_interval) -> scipy.sparse.csr_array:
        data = []
        row = []
        col = []
        l_r = 0
        l_s, r_s = lr_c(num_point)
        for i, (l_c, n) in enumerate(zip(l_s, num_point)):
            if not len(V_interval[i]):
                continue
            data.extend(V_interval[i].flatten())
            l_r_ = V_interval[i].shape[0]
            row.extend(l_r + np.repeat(np.arange(l_r_), n))
            col.extend(l_c + np.tile(np.arange(n), l_r_))
            l_r += l_r_
        L_row = l_r
        L_col = r_s[-1]
        M_coo = scipy.sparse.coo_array((data, (row, col)), shape=(L_row, L_col))
        M_coo.sum_duplicates()
        M_coo.eliminate_zeros()
        return M_coo.tocsr()

    @staticmethod
    def _assemble_nc(V_interval) -> scipy.sparse.csr_array:
        return scipy.sparse.block_diag(V_interval, format="csr")

    @abstractmethod
    def _assemble_x(self, V_interval) -> scipy.sparse.csr_array:
        pass

    @abstractmethod
    def _assemble_u(self, V_interval) -> scipy.sparse.csr_array:
        pass

    def V_x(self, t: VecFloat) -> scipy.sparse.csr_array:
        """Return the value interpolation matrix for the state variables at the
        output time nodes ``t``.

        Args:
            t: Time points for output.

        Returns:
            The interpolation matrix in the compressed sparse row format.

        Examples:
            Plot the first state variable at the output time nodes ``t_out``:

            >>> t_out = np.linspace(t_0, t_f, 100)
            >>> V_x = v.V_x(t_out)
            >>> x_out_0 = V_x @ v.x[0]
            >>> plt.plot(t_out, x_out_0)
        """
        t = self._guard_t(t)
        interval_info = self._interval_partition(t)
        V_interval = [
            V_interpolation(self._t_x[self._l_x[i] : self._r_x[i]], np.array(t_))
            for i, t_ in enumerate(interval_info)
        ]
        return self._assemble_x(V_interval)

    def V_u(self, t: VecFloat) -> scipy.sparse.csr_array:
        """Return the value interpolation matrix for the control variables at
        the output time nodes ``t``.

        Args:
            t: Time points for output.

        Returns:
            The interpolation matrix in the compressed sparse row format.

        Examples:
            Plot the first control variable at the output time nodes ``t_out``:

            >>> t_out = np.linspace(t_0, t_f, 100)
            >>> V_u = v.V_u(t_out)
            >>> u_out_0 = V_u @ v.u[0]
            >>> plt.plot(t_out, u_out_0)
        """
        t = self._guard_t(t)
        interval_info = self._interval_partition(t)
        V_interval = [
            V_interpolation(self._t_u[self._l_u[i] : self._r_u[i]], np.array(t_))
            for i, t_ in enumerate(interval_info)
        ]
        return self._assemble_u(V_interval)

    def D_x(self, t: VecFloat) -> scipy.sparse.csr_array:
        """Return the derivative interpolation matrix for the state variables
        at the output time nodes ``t``.

        Args:
            t: Time points for output.

        Returns:
            The derivative matrix in the compressed sparse row format.

        Examples:
            Plot the derivative of the first state variable at the output time nodes ``t_out``:

            >>> t_out = np.linspace(t_0, t_f, 100)
            >>> D_x = v.D_x(t_out)
            >>> dx_out_0 = D_x @ v.x[0]
            >>> plt.plot(t_out, dx_out_0)
        """
        t = self._guard_t(t)
        interval_info = self._interval_partition(t)
        D_interval = [
            D_interpolation(self._t_x[self._l_x[i] : self._r_x[i]], np.array(t_))
            for i, t_ in enumerate(interval_info)
        ]
        return self._assemble_x(D_interval)

    def D_u(self, t: VecFloat) -> scipy.sparse.csr_array:
        """Return the derivative interpolation matrix for the control variables
        at the output time nodes ``t``.

        Args:
            t: Time points for output.

        Returns:
            The derivative matrix in the compressed sparse row format.

        Examples:
            Plot the derivative of the first control variable at the output time nodes ``t_out``:

            >>> t_out = np.linspace(t_0, t_f, 100)
            >>> D_u = v.D_u(t_out)
            >>> du_out_0 = D_u @ v.u[0]
            >>> plt.plot(t_out, du_out_0)
        """
        t = self._guard_t(t)
        interval_info = self._interval_partition(t)
        D_interval = [
            D_interpolation(self._t_u[self._l_u[i] : self._r_u[i]], np.array(t_))
            for i, t_ in enumerate(interval_info)
        ]
        return self._assemble_u(D_interval)

    @property
    def x(self) -> BatchIndexArray:
        """The state variables of the variable that could be further
        indexed."""
        return self._array_state

    @property
    def u(self) -> BatchIndexArray:
        """The control variables of the variable that could be further
        indexed."""
        return self._array_control

    @property
    def t_0(self) -> float:
        """The initial time of the variable."""
        return self._data[-2]

    @t_0.setter
    def t_0(self, value: float) -> None:
        self._data[-2] = value

    @property
    def t_f(self) -> float:
        """The terminal time of the variable."""
        return self._data[-1]

    @t_f.setter
    def t_f(self, value: float) -> None:
        self._data[-1] = value

    @property
    def data(self) -> VecFloat:
        """The underlying data array.

        Typically used to pass to the solver.
        """
        return self._data

    @property
    def t_x(self) -> VecFloat:
        """The time interpolation nodes of the state variables."""
        return self._t_x * (self.t_f - self.t_0) + self.t_0

    @property
    def t_u(self) -> VecFloat:
        """The time interpolation nodes of the control variables."""
        return self._t_u * (self.t_f - self.t_0) + self.t_0

    def adapt(self, phase: PhaseBase) -> Self:
        """Adapt the ``Variable`` to a ``Phase`` with a different mesh and
        interpolation degree.

        Return a new ``Variable`` object without changing the current one.

        Args:
            phase: The ``Phase`` with a different mesh and interpolation degree to adapt to.

        Returns:
            A new ``Variable`` object adapted with values interpolated from the current one and
            compatible with the discretization scheme of the new ``Phase``.
        """
        V_x = self.V_x(phase.t_x * (self.t_f - self.t_0) + self.t_0)
        V_u = self.V_u(phase.t_u * (self.t_f - self.t_0) + self.t_0)

        data_new = np.empty(phase.L)
        for n_ in range(phase.n_x):
            data_new[phase.l_v[n_] : phase.r_v[n_]] = V_x @ self.x[n_]
        for n_ in range(phase.n_u):
            data_new[phase.l_v[phase.n_x + n_] : phase.r_v[phase.n_x + n_]] = (
                V_u @ self.u[n_]
            )
        data_new[-2] = self._data[-2]
        data_new[-1] = self._data[-1]
        return type(self)(phase, data_new)


def constant_guess_base(
    Variable: Type[VariableBase], phase: PhaseBase, value: float = 1.0
) -> VariableBase:
    """Return a ``Variable`` with constant guesses for a ``Phase``.

    Fixed boundary conditions are set to the corresponding values, while the other variables are set to ``value``.
    The function could be used as a starting point to obtain the desired dimensions and interpolation nodes, and then the guesses could be manually adjusted further.

    Args:
        phase: The ``Phase`` to guess for.
        value: The constant value to guess.

    Returns:
        A ``Variable`` with constant guesses for the given ``Phase``.
    """
    if not phase.ok:
        raise ValueError("phase is not fully configured")
    value = float(value)
    v = Variable(phase, np.full(phase.L, value, dtype=np.float64))
    for i in range(phase.n_x):
        if phase.info_bc_0[i].t == BcType.FIXED:
            v.x[i][0] = phase.bc_0[i]
        if phase.info_bc_f[i].t == BcType.FIXED:
            v.x[i][-1] = phase.bc_f[i]
    if phase.info_t_0.t == BcType.FIXED:
        v.t_0 = phase.t_0
    else:
        v.t_0 -= 0.5
    if phase.info_t_f.t == BcType.FIXED:
        v.t_f = phase.t_f
    else:
        v.t_f += 0.5
    return v


def linear_guess_base(
    Variable: Type[VariableBase], phase: PhaseBase, default: float = 1.0
) -> VariableBase:
    """Return a ``Variable`` with linear guesses for a ``Phase``.

    Fixed boundary conditions are set to the corresponding values; all other boundary conditions are assumed to be ``default``. Then, linear interpolation is used to set variables in the middle.
    The function could be used as a starting point to obtain the desired dimensions and interpolation nodes, and then the guesses could be manually adjusted further.

    Args:
        phase: The ``Phase`` to guess for.
        default: The default value to guess.

    Returns:
        A ``Variable`` with linear guesses for the given ``Phase``.
    """
    if not phase.ok:
        raise ValueError("phase is not fully configured")
    default = float(default)
    v = Variable(phase, np.full(phase.L, default, dtype=np.float64))
    for i in range(phase.n_x):
        if (
            phase.info_bc_0[i].t == BcType.FIXED
            and phase.info_bc_f[i].t == BcType.FIXED
        ):
            v.x[i] = v._t_x * (phase.bc_f[i] - phase.bc_0[i]) + phase.bc_0[i]
        elif phase.info_bc_0[i].t == BcType.FIXED:
            v.x[i] = phase.bc_0[i]
        elif phase.info_bc_f[i].t == BcType.FIXED:
            v.x[i] = phase.bc_f[i]
    if phase.info_t_0.t == BcType.FIXED:
        v.t_0 = phase.t_0
    else:
        v.t_0 -= 0.5
    if phase.info_t_f.t == BcType.FIXED:
        v.t_f = phase.t_f
    else:
        v.t_f += 0.5
    return v
