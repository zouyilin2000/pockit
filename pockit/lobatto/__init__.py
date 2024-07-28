"""Submodule for Legendre-Gauss-Lobatto pseudo-spectral methods.

The Lobatto phase is suitable for problems with continuous state and
control variables.
"""

from .phase import Phase
from .system import System
from .variable import Variable, constant_guess, linear_guess

__all__ = [
    "Phase",
    "System",
    "Variable",
    "constant_guess",
    "linear_guess",
]
