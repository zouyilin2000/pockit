# Copyright (c) 2024 Yilin Zou
from functools import partial
from typing import Callable

from pockit.lobatto.phase import Phase
from pockit.base.variablebase import *


class Variable(VariableBase):
    def __init__(self, phase: Phase, data: VecFloat) -> None:
        super().__init__(phase, data)

    def _assemble_x(self, V_interval) -> scipy.sparse.csr_array:
        return self._assemble_c(self._num_point, V_interval)

    def _assemble_u(self, V_interval) -> scipy.sparse.csr_array:
        return self._assemble_c(self._num_point, V_interval)


constant_guess: Callable[[Phase, float], Variable] = partial(
    constant_guess_base, Variable
)
"""Return a ``Variable`` with constant guesses for a ``Phase``.

Fixed boundary conditions are set to the corresponding values, while the other variables are set to ``value``.
The function could be used as a starting point to obtain the desired dimensions and interpolation nodes, and then the guesses could be manually adjusted further.

Args:
    phase: The ``Phase`` to guess for.
    value: The constant value to guess.

Returns:
    A ``Variable`` with constant guesses for the given ``Phase``.
"""

linear_guess: Callable[[Phase, float], Variable] = partial(linear_guess_base, Variable)
"""Return a ``Variable`` with linear guesses for a ``Phase``.

Fixed boundary conditions are set to the corresponding values; all other boundary conditions are assumed to be ``default``. Then, linear interpolation is used to set variables in the middle.
The function could be used as a starting point to obtain the desired dimensions and interpolation nodes, and then the guesses could be manually adjusted further.

Args:
    phase: The ``Phase`` to guess for.
    default: The default value to guess.

Returns:
    A ``Variable`` with linear guesses for the given ``Phase``.
"""
