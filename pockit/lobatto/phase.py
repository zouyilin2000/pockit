# Copyright (c) 2024 Yilin Zou
from typing import Iterable
import sympy as sp

from pockit.base.phasebase import *
from pockit.lobatto.discretization import *


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

    def check_discontinuous(
        self,
        variable,
        static_parameter: Optional[Iterable[float]] = None,
        tolerance_discontinuous: float = 1e-3,
        tolerance_mesh: float = 1e-4,
    ) -> bool:
        """Lobatto nodes cannot approximate discontinuous functions precisely.

        Use radau nodes instead.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "Lobatto nodes cannot approximate discontinuous functions precisely."
        )

    def check(
        self,
        variable,
        static_parameter: Optional[Iterable[float]] = None,
        absolute_tolerance_continuous: float = 1e-8,
        relative_tolerance_continuous: float = 1e-8,
        tolerance_discontinuous: float = 1e-3,
        tolerance_mesh: float = 1e-4,
    ) -> bool:
        """Check the continuous error. Same as :meth:`check_continuous`.

        Args:
            variable: Variable to be checked.
            static_parameter: Static parameter to be checked. Set to ``None`` if the phase has no static parameters.
            absolute_tolerance_continuous: Absolute tolerance for continuous error.
            relative_tolerance_continuous: Relative tolerance for continuous error.
            tolerance_discontinuous: In each subinterval, after scaling to :math:`[0, 1]`, the bang-bang control functions
                should either be less than ``tolerance_discontinuous`` or greater than ``1 - tolerance_discontinuous``
                simultaneously.
            tolerance_mesh: Skip the check if the mesh width is smaller than this value.

        Returns:
            ``True`` if the error is within the tolerance, ``False`` otherwise.
        """
        return self.check_continuous(
            variable,
            static_parameter,
            absolute_tolerance_continuous,
            relative_tolerance_continuous,
            tolerance_mesh,
        )

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
        """Lobatto nodes cannot approximate discontinuous functions precisely.

        Use radau nodes instead.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "Lobatto nodes cannot approximate discontinuous functions precisely."
        )

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
        continuous error tolerance. Same as :meth:`refine_continuous`.

        Args:
            variable: Variable of the previous iteration.
            static_parameter: Static parameter of the previous iteration.
                Set to ``None`` if the phase has no static parameters.
            absolute_tolerance_continuous: Absolute tolerance for continuous error.
            relative_tolerance_continuous: Relative tolerance for continuous error.
            tolerance_discontinuous: In each subinterval, after scaling to :math:`[0, 1]`, the bang-bang control functions
                should either be less than ``tolerance_discontinuous`` or greater than ``1 - tolerance_discontinuous``
                simultaneously.
            num_point_min: Minimum number of interpolation points.
            num_point_max: Maximum number of interpolation points.
            mesh_length_min: Minimum mesh length.
            mesh_length_max: Maximum mesh length.
        """
        if not self.check_continuous(
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
