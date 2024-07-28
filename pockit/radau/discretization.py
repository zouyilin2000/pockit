import functools
from typing import Callable

import numba as nb
import numpy as np
import scipy.interpolate
import scipy.sparse
import scipy.special

from pockit.base.vectypes import *
from pockit.base.discretizationbase import *


def lr_s(num_point: VecInt) -> tuple[VecInt, VecInt]:
    """Compute the left and right index of each subinterval. The left index is
    ``l_s[i]`` and the right index is ``r_s[i]``, which represents the range of
    subinterval ``i`` in a half-open interval ``[l_s[i], r_s[i])``.

    Args:
        num_point: Number of interpolation points of each interval.

    Returns:
        Left and right index ``l_s, r_s`` for all subintervals.
    """
    return lr_nc(num_point)


def lr_m(num_point: VecInt) -> tuple[VecInt, VecInt]:
    """Compute the left and right index of each subinterval in the middle
    stage. The left index is ``l_m[i]`` and the right index is ``r_m[i]``,
    which represents the range of subinterval ``i`` in a half-open interval
    ``[l_m[i], r_m[i])``.

    Args:
        num_point: Number of interpolation points of each interval.

    Returns:
        Left and right index ``l_m, r_m`` for all subintervals in the middle stage.
    """
    return lr_nc(num_point)


def lr_v(num_point: VecInt, num_state: int, num_control: int) -> tuple[VecInt, VecInt]:
    """Compute the left and right index of each variable. The left index is
    ``l_v[i]`` and the right index is ``r_v[i]``, which represents the range of
    variable ``i`` in a half-open interval ``[l_v[i], r_v[i])``.

    Args:
        num_point: Number of interpolation points of each subinterval.
        num_state: Number of state variables.
        num_control: Number of control variables.

    Returns:
        Left and right index ``l_v, r_v`` of each state and control variable.
    """
    L_u = np.sum(num_point)
    L_x = L_u + 1
    l_v = [0]
    r_v = []
    for _ in range(num_state):
        r_v.append(l_v[-1] + L_x)
        l_v.append(r_v[-1])
    for _ in range(num_control):
        r_v.append(l_v[-1] + L_u)
        l_v.append(r_v[-1])
    return np.array(l_v[:-1], dtype=int), np.array(r_v, dtype=int)


def lr_d(num_point: VecInt, num_state: int, num_control: int) -> tuple[VecInt, VecInt]:
    """Compute the left and right index of dynamical constraints for each
    variable.

    Return the left and right index of each variable.
    The left index of variable ``i`` is ``l_d[i]``, and the right index is ``r_d[i] - 1``.

    Args:
        num_point: Number of interpolation points of each subinterval.
        num_state: Number of state variables.
        num_control: Number of control variables.

    Returns:
        Left and right index ``l_d, r_d`` of each variable.
    """
    L_d = np.sum(num_point)
    return lr_nc(np.full(num_state, L_d, dtype=np.int32))


@functools.lru_cache
def xw_lgr(num_point: int) -> tuple[VecFloat, VecFloat]:
    """Compute the Legendre-Gauss-Radau nodes and Gaussian quadrature weights.

    The interval is assumed to be ``[-1, 1]``.

    Args:
        num_point: Number of interpolation points.

    Returns:
        Interpolation nodes and integration weights of the Legendre-Gauss-Radau scheme.
    """
    if num_point <= 0:
        raise ValueError("Number of interpolation points must be at least 1.")

    p_j = scipy.special.jacobi(num_point - 1, 0, 1)
    x = [-1.0]
    roots = np.roots(p_j)
    for root in roots:
        x.append(root.real)
    x.sort()
    x = np.array(x, dtype=float)

    p_l = scipy.special.legendre(num_point)
    w = (1 - x) / (num_point * np.polyval(p_l, x)) ** 2
    return np.array(x, dtype=float), np.array(w, dtype=float)


def xw_s(mesh: VecFloat, num_point: VecInt) -> tuple[VecFloat, VecFloat]:
    """Compute the interpolation nodes and integration weights of each
    variable.

    The nodes and weights of each subinterval are scaled and joined together.

    Args:
        mesh: Mesh points. I.e., boundaries of each subinterval.
        num_point: Number of interpolation points of each subinterval.

    Returns:
        Interpolation nodes and integration weights of each variable.
    """
    l_s, r_s = lr_s(num_point)
    L_s = r_s[-1]  # length
    x = np.zeros(L_s)
    w = np.zeros(L_s)
    width = np.diff(mesh)  # length of each interval
    mid = (mesh[1:] + mesh[:-1]) / 2  # mid-point of each interval
    for l, r, n, d, m in zip(l_s, r_s, num_point, width, mid):
        x_, w_ = xw_lgr(n)
        x[l:r] = x_ * d / 2 + m
        w[l:r] = w_ * d / 2
    return x, w


def v2m(num_point: VecInt, n_x: int, n_u: int) -> Callable[[VecFloat], VecFloat]:
    """Return a closure that converts all variables to the middle stage.

    Args:
        num_point: Number of interpolation points of each subinterval.
        n_x: Number of state variables.
        n_u: Number of control variables.

    Returns:
        Function that converts variables to the middle stage.
    """
    l_v, r_v = lr_v(num_point, n_x, n_u)
    L_s = np.sum(num_point)

    @nb.njit
    def v2m_(v: VecFloat) -> VecFloat:
        m = np.empty(L_s * (n_x + n_u), dtype=np.float64)
        for i in range(n_x):
            m[i * L_s : (i + 1) * L_s] = v[l_v[i] : r_v[i] - 1]
        for i in range(n_u):
            m[(n_x + i) * L_s : (n_x + i + 1) * L_s] = v[l_v[n_x + i] : r_v[n_x + i]]
        return m

    return v2m_


@functools.lru_cache
def T_lgr(num_point: int) -> VecFloat:
    """Compute the translation matrix of the Legendre-Gauss-Radau nodes. The
    translation matrix is used to eliminate the constant of integrations.

    Args:
        num_point: Number of interpolation points.

    Returns:
        Translation matrix of the Legendre-Gauss-Radau nodes.
    """
    c = np.full((num_point, 1), -1, dtype=np.float64)
    e = np.eye(num_point, dtype=np.float64)
    return np.hstack((e, c))


def I_lgr(num_point: int) -> VecFloat:
    """Compute the integration matrix of the Legendre-Gauss-Radau nodes.

    Args:
        num_point: Number of interpolation points.

    Returns:
        Integration matrix of the Legendre-Gauss-Radau nodes.
    """
    x, _ = xw_lgr(num_point)
    x_1 = np.concatenate((x, [1]), dtype=np.float64)
    I = []
    for i in range(num_point):
        y = np.zeros_like(x)
        y[i] = 1
        poly = scipy.interpolate.lagrange(x, y)
        poly_int = np.polyint(poly)
        val = np.polyval(poly_int, x_1)
        I.append(val[:-1] - val[-1])
    return np.array(I, dtype=float).T


def T_v(mesh: VecFloat, num_point: VecInt) -> scipy.sparse.csr_array:
    """Compute the translation matrix of each state variable.

    Each subinterval's derivative matrices are placed diagonally.
    The derivative matrix is stored in the compressed sparse row format.

    Args:
        mesh: Mesh points. I.e., boundaries of each subinterval.
        num_point: Number of interpolation points of each subinterval.

    Returns:
        Derivative matrix of a state variable in the compressed sparse row format.
    """
    data = []
    row = []
    col = []
    l_row, r_row = lr_nc(num_point)
    l_col, r_col = lr_c(num_point + 1)
    L_row = r_row[-1]
    L_col = r_col[-1]
    for l_r, l_c, n in zip(l_row, l_col, num_point):
        T = T_lgr(n)
        data.extend(T.flatten())
        row.extend(l_r + np.repeat(np.arange(n), n + 1))
        col.extend(l_c + np.tile(np.arange(n + 1), n))
    T_coo = scipy.sparse.coo_array((data, (row, col)), shape=(L_row, L_col))
    T_coo.eliminate_zeros()
    return T_coo.tocsr()


def I_m(mesh: VecFloat, num_point: VecInt) -> scipy.sparse.csr_array:
    """Compute the intetgration matrix of variable in the middle stage.

    Each subinterval's intetgration matrices are placed diagonally.
    The intetgration matrix is stored in the compressed sparse row format.

    Args:
        mesh: Mesh points. I.e., boundaries of each subinterval.
        num_point: Number of interpolation points of each subinterval.

    Returns:
        Intetgration matrix of a variable in the middle stage in the compressed sparse row format.
    """
    data = []
    row = []
    col = []
    l_row, r_row = lr_nc(num_point)
    l_col, r_col = lr_nc(num_point)
    L_row = r_row[-1]
    L_col = r_col[-1]
    width = np.diff(mesh)  # length of each interval
    for l_r, l_c, n, d in zip(l_row, l_col, num_point, width):
        I = I_lgr(n) * d / 2
        data.extend(I.flatten())
        row.extend(l_r + np.repeat(np.arange(n), n))
        col.extend(l_c + np.tile(np.arange(n), n))
    I_coo = scipy.sparse.coo_array((data, (row, col)), shape=(L_row, L_col))
    I_coo.eliminate_zeros()
    return I_coo.tocsr()


@functools.lru_cache
def P_lgr(num_point: int) -> VecFloat:
    """Compute the coefficients matrix of interpolation polynomials of the
    Legendre-Gauss-Radau nodes.

    The coefficients matrix permits fast computation of the interpolation
    polynomials given arbitrary values at the
    interpolation nodes. I.e., ``coefficients of interpolation polynomial = P @ x``.

    Args:
        num_point: Number of interpolation points.

    Returns:
        Coefficients matrix of interpolation polynomials of the Legendre-Gauss-Radau nodes.
    """
    x, _ = xw_lgr(num_point)
    P = []
    for i in range(num_point):
        y = np.zeros(num_point)
        y[i] = 1
        poly = scipy.interpolate.lagrange(x, y)
        P.append(poly.coef)
    return np.array(P, dtype=float).T


def V_lgr_x_aug(num_point: int) -> VecFloat:
    """Compute the value matrix of the state variables for error check.

    The number of interpolation points is augmented by one.

    Args:
        num_point: Number of input interpolation points.

    Returns:
        Value matrix of the state variables for error check.
    """
    x, _ = xw_lgr(num_point)
    x_1 = np.concatenate((x, [1]), dtype=np.float64)
    x_aug, _ = xw_lgr(num_point + 1)
    V = []
    for i in range(num_point + 1):
        y = np.zeros_like(x_1)
        y[i] = 1
        poly = scipy.interpolate.lagrange(x_1, y)
        V.append(np.polyval(poly, x_aug))
    return np.array(V, dtype=float).T


def V_lgr_u_aug(num_point: int) -> VecFloat:
    """Compute the value matrix of the control variables for error check.

    The number of interpolation points is augmented by one.

    Args:
        num_point: Number of input interpolation points.

    Returns:
        Value matrix of the control variables for error check.
    """
    x, _ = xw_lgr(num_point)
    x_aug, _ = xw_lgr(num_point + 1)
    V = []
    for i in range(num_point):
        y = np.zeros_like(x)
        y[i] = 1
        poly = scipy.interpolate.lagrange(x, y)
        V.append(np.polyval(poly, x_aug))
    return np.array(V, dtype=float).T


def T_lgr_aug(num_point: int) -> VecFloat:
    """Compute the translation matrix of the state variable for error check.

    The number of interpolation points is augmented by one.

    Args:
        num_point: Number of input interpolation points.

    Returns:
        Translation matrix of the state variable for error check.
    """
    x, _ = xw_lgr(num_point)
    x_1 = np.concatenate((x, [1]), dtype=np.float64)
    x_aug, _ = xw_lgr(num_point + 1)
    x_aug_1 = np.concatenate((x_aug, [1]), dtype=np.float64)
    T = []
    for i in range(num_point + 1):
        y = np.zeros_like(x_1)
        y[i] = 1
        poly = scipy.interpolate.lagrange(x_1, y)
        val = np.polyval(poly, x_aug_1)
        T.append(val[:-1] - val[-1])
    return np.array(T, dtype=float).T


def V_s_x_aug(mesh: VecFloat, num_point: VecInt) -> scipy.sparse.csr_array:
    """Compute the value matrix of each state variable.

    Each subinterval's value matrices are placed diagonally.
    The value matrix is stored in the compressed sparse row format.
    The number of interpolation points is augmented by one for each subinterval.

    Args:
        mesh: Mesh points. I.e., boundaries of each subinterval.
        num_point: Number of interpolation points of each subinterval.

    Returns:
        Value matrix of a state variable for error check in the compressed sparse row format.
    """
    data = []
    row = []
    col = []
    l_row, r_row = lr_nc(num_point + 1)
    l_col, r_col = lr_c(num_point + 1)
    L_row = r_row[-1]
    L_col = r_col[-1]
    for l_r, l_c, n in zip(l_row, l_col, num_point):
        V = V_lgr_x_aug(n)
        data.extend(V.flatten())
        row.extend(l_r + np.repeat(np.arange(n + 1), n + 1))
        col.extend(l_c + np.tile(np.arange(n + 1), n + 1))
    V_coo = scipy.sparse.coo_array((data, (row, col)), shape=(L_row, L_col))
    V_coo.sum_duplicates()
    V_coo.eliminate_zeros()
    return V_coo.tocsr()


def V_s_u_aug(mesh: VecFloat, num_point: VecInt) -> scipy.sparse.csr_array:
    """Compute the value matrix of each control variable.

    Each subinterval's value matrices are placed diagonally.
    The value matrix is stored in the compressed sparse row format.
    The number of interpolation points is augmented by one for each subinterval.

    Args:
        mesh: Mesh points. I.e., boundaries of each subinterval.
        num_point: Number of interpolation points of each subinterval.

    Returns:
        Value matrix of a control variable for error check in the compressed sparse row format.
    """
    return scipy.sparse.block_diag([V_lgr_u_aug(n) for n in num_point]).tocsr()


def T_s_aug(mesh: VecFloat, num_point: VecInt) -> scipy.sparse.csr_array:
    """Compute the translation matrix of each variable for error check.

    Each subinterval's translation matrices are placed diagonally.
    The translation matrix is stored in the compressed sparse row format.
    The number of interpolation points is augmented by one for each subinterval.

    Args:
        mesh: Mesh points. I.e., boundaries of each subinterval.
        num_point: Number of interpolation points of each subinterval.

    Returns:
        Translation matrix of each variable for error check in the compressed sparse row format.
    """
    data = []
    row = []
    col = []
    l_row, r_row = lr_s(num_point + 1)
    l_col, r_col = lr_c(num_point + 1)
    L_row = r_row[-1]
    L_col = r_col[-1]
    for l_r, l_c, n in zip(l_row, l_col, num_point):
        T = T_lgr_aug(n)
        data.extend(T.flatten())
        row.extend(l_r + np.repeat(np.arange(n + 1), n + 1))
        col.extend(l_c + np.tile(np.arange(n + 1), n + 1))
    T_coo = scipy.sparse.coo_array((data, (row, col)), shape=(L_row, L_col))
    T_coo.sum_duplicates()
    T_coo.eliminate_zeros()
    return T_coo.tocsr()


def V_xu_aug(
    mesh: VecFloat, num_point: VecInt, n_x: int, n_u: int
) -> scipy.sparse.csr_array:
    """Compute the value matrix of all state and control variables for error
    check.

    The number of interpolation points is augmented by one for each subinterval.
    The value matrices of all state and control variables are placed diagonally.
    The result is stored in the compressed sparse row format.

    Args:
        mesh: Mesh points. I.e., boundaries of each subinterval.
        num_point: Number of interpolation points of each subinterval.
        n_x: Number of state variables.
        n_u: Number of control variables

    Returns:
        Value matrix of all variables for error check in the compressed sparse row format.
    """
    V_s_x_ = V_s_x_aug(mesh, num_point)
    V_s_u_ = V_s_u_aug(mesh, num_point)
    diag = []
    for _ in range(n_x):
        diag.append(V_s_x_)
    for _ in range(n_u):
        diag.append(V_s_u_)
    return scipy.sparse.block_diag(diag).tocsr()


def T_x_aug(mesh: VecFloat, num_point: VecInt, n_x: int) -> scipy.sparse.csr_array:
    """Compute the translation matrix of all state variables for error check.

    The number of interpolation points is augmented by one for each subinterval.
    The translation matrices of all state variables are placed diagonally.
    The result is stored in the compressed sparse row format.

    Args:
        mesh: Mesh points. I.e., boundaries of each subinterval.
        num_point: Number of interpolation points of each subinterval.
        n_x: Number of state variables.

    Returns:
        Translation matrix of all state variables for error check in the compressed sparse row format.
    """
    T_s_ = T_s_aug(mesh, num_point)
    diag = [T_s_ for _ in range(n_x)]
    return scipy.sparse.block_diag(diag).tocsr()


class Discretization(DiscretizationBase):
    def __init__(self, mesh: VecFloat, num_point: VecInt, n_x: int, n_u: int):
        super().__init__(mesh, num_point, n_x, n_u)

        self._l_v, self._r_v = lr_v(num_point, self.n_x, self.n_u)
        self._t_m, self._w_m = xw_s(mesh, num_point)
        self._l_m, self._r_m = lr_m(num_point)

        self._f_v2m = v2m(num_point, self.n_x, self.n_u)
        self._T_v = T_v(mesh, num_point)

        self._I_m = I_m(mesh, num_point)

        self._l_d, self._r_d = lr_d(num_point, self.n_x, self.n_u)

        self._index_state = IndexNode(0, (1, self.L_m), self.L_m)
        self._index_control = IndexNode(self.L_m, (1, self.L_m), None)
        self._index_mstage = IndexNode(0, (1, self.L_m), None)
        self._T_v_coo = CooMatrixNode(self._T_v, self._index_state)
        self._I_m_coo = CooMatrixNode(self._I_m, self._index_mstage)

        # data for error estimation and refining mesh
        self._l_m_aug, self._r_m_aug = lr_m(num_point + 1)
        self._t_m_aug, _ = xw_s(mesh, num_point + 1)
        self._w_aug = [xw_lgr(n)[1] for n in num_point]
        self._P = P_lgr
        self._V_xu_aug = V_xu_aug(mesh, num_point, self.n_x, self.n_u)
        self._T_x_aug = T_x_aug(mesh, num_point, self.n_x)
        self._I_m_aug = I_m(mesh, num_point + 1)

        # data for Vriable
        self._t_x = np.concatenate([self.t_m, [1]], dtype=np.float64)
        self._l_x, self._r_x = lr_nc(num_point)
        self._r_x[-1] += 1
        self._l_u, self._r_u = lr_nc(num_point)

    @property
    def index_state(self) -> IndexNode:
        return self._index_state

    @property
    def index_control(self) -> IndexNode:
        return self._index_control

    @property
    def index_mstage(self) -> IndexNode:
        return self._index_mstage

    @property
    def l_v(self) -> VecInt:
        return self._l_v

    @property
    def r_v(self) -> VecInt:
        return self._r_v

    @property
    def t_m(self) -> VecFloat:
        return self._t_m

    @property
    def w_m(self) -> VecFloat:
        return self._w_m

    @property
    def l_m(self) -> VecInt:
        return self._l_m

    @property
    def r_m(self) -> VecInt:
        return self._r_m

    @property
    def L_m(self) -> int:
        return self.r_m[-1]

    @property
    def f_v2m(self) -> Callable[[VecFloat], VecFloat]:
        return self._f_v2m

    @property
    def T_v(self) -> scipy.sparse.csr_array:
        return self._T_v

    @property
    def T_v_coo(self) -> CooMatrixNode:
        return self._T_v_coo

    @property
    def I_m(self) -> scipy.sparse.csr_array:
        return self._I_m

    @property
    def I_m_coo(self) -> CooMatrixNode:
        return self._I_m_coo

    @property
    def l_d(self) -> VecInt:
        return self._l_d

    @property
    def r_d(self) -> VecInt:
        return self._r_d

    @property
    def t_m_aug(self) -> VecFloat:
        return self._t_m_aug

    @property
    def l_m_aug(self) -> VecInt:
        return self._l_m_aug

    @property
    def r_m_aug(self) -> VecInt:
        return self._r_m_aug

    @property
    def L_m_aug(self) -> int:
        return self.r_m_aug[-1]

    @property
    def w_aug(self) -> list[VecFloat]:
        return self._w_aug

    @property
    def P(self) -> Callable[[int], VecFloat]:
        return self._P

    @property
    def V_xu_aug(self) -> scipy.sparse.csr_array:
        return self._V_xu_aug

    @property
    def T_x_aug(self) -> scipy.sparse.csr_array:
        return self._T_x_aug

    @property
    def I_m_aug(self) -> scipy.sparse.csr_array:
        return self._I_m_aug

    @property
    def t_x(self) -> VecFloat:
        return self._t_x

    @property
    def t_u(self) -> VecFloat:
        return self._t_m

    @property
    def l_x(self) -> VecInt:
        return self._l_x

    @property
    def r_x(self) -> VecInt:
        return self._r_x

    @property
    def l_u(self) -> VecInt:
        return self._l_u

    @property
    def r_u(self) -> VecInt:
        return self._r_u
