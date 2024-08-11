import sympy as sp

from pockit.base.phasebase import *
from pockit.radau.discretization import *


class Phase(PhaseBase):
    def __init__(
        self,
        identifier: int,
        state: int | list[str],
        control: int | list[str],
        symbol_static_parameter: list[sp.Symbol],
        simplify: bool = False,
        parallel: bool = False,
        fastmath: bool = False,
    ) -> None:
        super().__init__(
            identifier,
            state,
            control,
            symbol_static_parameter,
            simplify,
            parallel,
            fastmath,
        )

    @property
    def _class_discretization(self) -> type[DiscretizationBase]:
        return Discretization
