import numpy as np
from numpy.typing import NDArray
from numba.typed import List

VecFloat = NDArray[np.float64]
VecInt = NDArray[np.int32]
VecBool = NDArray[bool]
ListJit = List