# Copyright (c) 2024 Yilin Zou
"""Submodule for Legendre-Gauss-Radau pseudospectral methods.

The Radau phase is suitable for problems with continuous or
discontinuous state and control variables.
"""

from .system import System
from .phase import Phase
from .variable import Variable, constant_guess, linear_guess

__all__ = [
    "Phase",
    "System",
    "Variable",
    "constant_guess",
    "linear_guess",
]
