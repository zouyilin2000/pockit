# Copyright (c) 2024 Yilin Zou
from abc import ABC, abstractmethod
from typing import Callable, Optional
import scipy.sparse

from .vectypes import *


def lr_c(num_point: VecInt) -> tuple[VecInt, VecInt]:
    """Compute the left and right indices if the border variables are shared.
    The left index is ``l_c[i]`` and the right index is ``r_c[i]``, which
    represents the range of interval ``i`` in a half-open interval ``[l_c[i],
    r_c[i])``.

    Args:
        num_point: Number of interpolation points of each interval.

    Returns:
        Left and right index ``l_c, r_c`` for shared border variables.
    """
    l = np.concatenate(([0], np.cumsum(num_point[:-1] - 1)))
    return l, l + num_point


def lr_nc(num_point: VecInt) -> tuple[VecInt, VecInt]:
    """Compute the left and right indices if separate variables represent the
    common borders. The left index is ``l_nc[i]`` and the right index is
    ``r_nc[i]``, which represents the range of interval ``i`` in a half-open
    interval ``[l_nc[i], r_nc[i])``.

    Args:
        num_point: Number of interpolation points of each interval.

    Returns:
        Left and right index ``l_nc, r_nc`` for separate border variables.
    """
    return np.concatenate(([0], np.cumsum(num_point[:-1]))), np.cumsum(num_point)


class CooMatrix:
    """Hold the data of a sparse matrix in COO format."""

    row: VecInt
    col: VecInt
    data: VecFloat

    def __init__(self, row: VecInt, col: VecInt, data: VecFloat) -> None:
        self.row = row
        self.col = col
        self.data = data

    def __len__(self) -> int:
        return len(self.row)


class IndexNode:
    """In the discretization scheme, the state and control variables are
    represented by values at discrete time points. The scheme requires
    additional information about whether the vector contains the variable's
    initial and terminal values to handle boundary conditions.

    The data structure is applied to state, control, and middle-stage
    variables.
    """

    _front: Optional[int]
    _middle: tuple[int, int]
    _back: Optional[int]

    def __init__(
        self, front: Optional[int], middle: tuple[int, int], back: Optional[int]
    ) -> None:
        """
        Args:
            front: The index of the front variable or ``None`` if the vector does not contain the initial value.
            middle: The range of the middle variables.
            back: The index of the back variable or ``None`` if the vector does not contain the terminal value.
        """
        self._front = front
        self._middle = middle
        self._back = back
        self._length_middle = middle[1] - middle[0]

    @property
    def f(self) -> bool:
        """Whether the vector contains the front variable."""
        return self._front is not None

    @property
    def m(self) -> slice:
        """The slice of the middle variables."""
        return slice(*self._middle)

    @property
    def b(self) -> bool:
        """Whether the vector contains the back variable."""
        return self._back is not None

    @property
    def L_m(self) -> int:
        """The length of the middle variables."""
        return self._length_middle

    @property
    def l_m(self) -> int:
        """The left index of the middle variables."""
        return self._middle[0]

    @property
    def r_m(self) -> int:
        """The right index of the middle variables (exclusive)."""
        return self._middle[1]


class CooMatrixNode:
    """Classify the COO matrix into front, middle, and back parts according to
    the column index and the index partition.

    The data structure is used to generate the Jacobian and Hessian
    matrices of the discretized problem.
    """

    _front: CooMatrix
    _middle: CooMatrix
    _back: CooMatrix

    def __init__(self, csr_matrix: scipy.sparse.csr_array, index: IndexNode) -> None:
        """
        Args:
            csr_matrix: The CSR matrix to be classified.
            index: The index partition of the vector.
        """
        coo_matrix = csr_matrix.tocoo()
        coo_matrix.eliminate_zeros()
        row_front = []
        col_front = []
        data_front = []
        row_middle = []
        col_middle = []
        data_middle = []
        row_back = []
        col_back = []
        data_back = []
        for r_, c_, d_ in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            if c_ == index._front:
                row_front.append(r_)
                col_front.append(c_)
                data_front.append(d_)
            elif c_ == index._back:
                row_back.append(r_)
                col_back.append(c_)
                data_back.append(d_)
            else:
                row_middle.append(r_)
                col_middle.append(c_)
                data_middle.append(d_)
        self._front = CooMatrix(
            np.array(row_front, dtype=np.int32),
            np.array(col_front, dtype=np.int32),
            np.array(data_front, dtype=np.float64),
        )
        self._middle = CooMatrix(
            np.array(row_middle, dtype=np.int32),
            np.array(col_middle, dtype=np.int32),
            np.array(data_middle, dtype=np.float64),
        )
        self._back = CooMatrix(
            np.array(row_back, dtype=np.int32),
            np.array(col_back, dtype=np.int32),
            np.array(data_back, dtype=np.float64),
        )

    @property
    def f(self) -> CooMatrix:
        """The front part of the COO matrix."""
        return self._front

    @property
    def m(self) -> CooMatrix:
        """The middle part of the COO matrix."""
        return self._middle

    @property
    def b(self) -> CooMatrix:
        """The back part of the COO matrix."""
        return self._back


class DiscretizationBase(ABC):
    """Essential information for a discretization scheme.

    The class contains the information to discretize the continuous-time
    optimal control problem into a discrete-time problem and generate
    the derivative matrices.
    """

    mesh: VecFloat
    """The mesh points of the time interval, scaled to ``[0, 1]``."""
    num_point: VecInt
    """The number of interpolation points in each subinterval."""
    n_x: int
    """The number of state variables."""
    n_u: int
    """The number of control variables."""

    def __init__(self, mesh: VecFloat, num_point: VecInt, n_x: int, n_u: int) -> None:
        """
        Args:
            mesh: The mesh points of the time interval.
            num_point: The number of interpolation points in each subinterval.
            n_x: The number of state variables.
            n_u: The number of control variables.
        """
        self.mesh = mesh
        self.num_point = num_point
        self.n_x = n_x
        self.n_u = n_u

    @property
    @abstractmethod
    def index_state(self) -> IndexNode:
        """Index partition of the state variables."""
        pass

    @property
    @abstractmethod
    def index_control(self) -> IndexNode:
        """Index partition of the control variables."""
        pass

    @property
    @abstractmethod
    def index_mstage(self) -> IndexNode:
        """Index partition of the middle-stage variables."""
        pass

    @property
    @abstractmethod
    def l_v(self) -> VecInt:
        """The left index of the state and control variables."""
        pass

    @property
    @abstractmethod
    def r_v(self) -> VecInt:
        """The right index of the state and control variables (exclusive)."""
        pass

    @property
    @abstractmethod
    def t_m(self) -> VecFloat:
        """Position of all interpolation nodes in the middle stage, scaled to
        ``[0, 1]``."""
        pass

    @property
    @abstractmethod
    def l_m(self) -> VecInt:
        """The left index of subintervals in the middle stage."""
        pass

    @property
    @abstractmethod
    def r_m(self) -> VecInt:
        """The right index of subintervals in the middle stage (exclusive)."""
        pass

    @property
    @abstractmethod
    def L_m(self) -> int:
        """The number of interpolation points in the middle stage."""
        pass

    @property
    @abstractmethod
    def w_m(self) -> VecFloat:
        """The integration weights of the middle stage."""
        pass

    @property
    @abstractmethod
    def f_v2m(self) -> Callable[[VecFloat], VecFloat]:
        """Function turns the state and control variables into the middle
        stage."""
        pass

    @property
    @abstractmethod
    def T_v(self) -> scipy.sparse.csr_array:
        """Translation matrix of a state variable to eliminate the integration
        coefficient."""
        pass

    @property
    @abstractmethod
    def T_v_coo(self) -> CooMatrixNode:
        """Translation matrix of a state variable in COO format partitioned by
        the index."""
        pass

    @property
    @abstractmethod
    def I_m(self) -> scipy.sparse.csr_array:
        """Integration matrix of a middle-stage variable."""
        pass

    @property
    @abstractmethod
    def I_m_coo(self) -> CooMatrixNode:
        """Integration matrix of a middle-stage variable in COO format
        partitioned by the index."""
        pass

    @property
    @abstractmethod
    def l_d(self) -> VecInt:
        """The left index of the dynamic constraints of each state variable."""
        pass

    @property
    @abstractmethod
    def r_d(self) -> VecInt:
        """The right index of the dynamic constraints of each state variable
        (exclusive)."""
        pass

    @property
    @abstractmethod
    def t_m_aug(self) -> VecFloat:
        """Position of all interpolation nodes in the middle stage, scaled to
        ``[0, 1]`` with one additional interpolation point in each
        subinterval."""
        pass

    @property
    @abstractmethod
    def l_m_aug(self) -> VecInt:
        """The left index of subintervals in the middle stage, with one
        additional interpolation point in each subinterval."""
        pass

    @property
    @abstractmethod
    def r_m_aug(self) -> VecInt:
        """The right index of subintervals in the middle stage (exclusive),
        with one additional interpolation point in each subinterval."""
        pass

    @property
    @abstractmethod
    def L_m_aug(self) -> int:
        """The number of interpolation points in the middle stage, with one
        additional interpolation point in each subinterval."""
        pass

    @property
    @abstractmethod
    def w_aug(self) -> list[VecFloat]:
        """The integration weights of each subinterval in the middle stage,
        with one additional interpolation point in each subinterval."""
        pass

    @property
    @abstractmethod
    def P(self) -> Callable[[int], VecFloat]:
        """Function to compute the polynomial matrix given the number of
        interpolation points."""
        pass

    @property
    @abstractmethod
    def V_xu_aug(self) -> scipy.sparse.csr_array:
        """The value matrix translates all state and control variables to the
        middle stage with one additional interpolation point in each
        subinterval."""
        pass

    @property
    @abstractmethod
    def T_x_aug(self) -> scipy.sparse.csr_array:
        """The translation matrix of all state variables, with one additional
        interpolation point in each subinterval."""
        pass

    @property
    @abstractmethod
    def I_m_aug(self) -> scipy.sparse.csr_array:
        """The integration matrix of a middle-stage variable, with one
        additional interpolation point in each subinterval."""
        pass

    @property
    @abstractmethod
    def t_x(self) -> VecFloat:
        """Position of all interpolation nodes of a state variable, scaled to
        ``[0, 1]``."""
        pass

    @property
    @abstractmethod
    def t_u(self) -> VecFloat:
        """Position of all interpolation nodes of a control variable, scaled to
        ``[0, 1]``."""
        pass

    @property
    @abstractmethod
    def l_x(self) -> VecInt:
        """The left index of subintervals of a state variable."""
        pass

    @property
    @abstractmethod
    def r_x(self) -> VecInt:
        """The right index of subintervals of a state variable (exclusive)."""
        pass

    @property
    @abstractmethod
    def l_u(self) -> VecInt:
        """The left index of subintervals of a control variable."""
        pass

    @property
    @abstractmethod
    def r_u(self) -> VecInt:
        """The right index of subintervals of a control variable
        (exclusive)."""
        pass
