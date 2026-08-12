# Copyright (c) 2024 Yilin Zou
from functools import partial
from typing import Callable

from pockit.radau.phase import Phase
from pockit.base.variablebase import *


class Variable(VariableBase):
    def __init__(self, phase: Phase, data: VecFloat) -> None:
        super().__init__(phase, data)

    def _assemble_x(self, V_interval) -> scipy.sparse.csr_array:
        return self._assemble_c(self._num_point + 1, V_interval)

    def _assemble_u(self, V_interval) -> scipy.sparse.csr_array:
        return self._assemble_nc(V_interval)


constant_guess: Callable[[Phase, float], Variable] = partial(
    constant_guess_base, Variable
)
"""Return a ``Variable`` initialized with constant values for a ``Phase``.

Fixed boundary values are preserved, and all other variables are set to
``value``. The returned object can be adjusted before it is passed to a solver.

Args:
    phase: The ``Phase`` to guess for.
    value: The constant value to guess.

Returns:
    A ``Variable`` with constant guesses for the given ``Phase``.
"""

linear_guess: Callable[[Phase, float], Variable] = partial(linear_guess_base, Variable)
"""Return a ``Variable`` initialized with linear state values for a ``Phase``.

Fixed boundary values are preserved. Missing boundary values are replaced by
``default``, and state values between the boundaries are interpolated linearly.
The returned object can be adjusted before it is passed to a solver.

Args:
    phase: The ``Phase`` to guess for.
    default: The default value to guess.

Returns:
    A ``Variable`` with linear guesses for the given ``Phase``.
"""
