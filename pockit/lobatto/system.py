# Copyright (c) 2024 Yilin Zou
from typing import Iterable

from pockit.base.systembase import SystemBase, PhaseBase
from pockit.lobatto.phase import Phase
from pockit.lobatto.variable import Variable


class System(SystemBase):
    def __init__(
        self,
        static_parameter: int | list[str],
        simplify: bool = False,
        fastmath: bool = False,
    ) -> None:
        super().__init__(static_parameter, simplify, fastmath)

    @property
    def _class_phase(self) -> type[PhaseBase]:
        return Phase

    def check_discontinuous(
        self,
        value: Variable | list[Variable | Iterable[float]],
        tolerance_discontinuous: float = 1.0e-3,
        tolerance_mesh: float = 1.0e-4,
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
        value: Variable | list[Variable | Iterable[float]],
        absolute_tolerance_continuous: float = 1.0e-8,
        relative_tolerance_continuous: float = 1.0e-8,
        tolerance_discontinuous: float = 1.0e-3,
        tolerance_mesh: float = 1.0e-4,
    ) -> bool:
        """Check the continuous error. Same as :meth:`check_continuous`.

        Args:
            value: The variable to be checked. If the system has only one phase and no static variables, ``value`` can
                be a single `Variable` object. Otherwise, ``value`` should be a list of
                ``Variable`` objects, one for each ``Phase``, followed by an array
                as values of static variables.
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
            value,
            absolute_tolerance_continuous=absolute_tolerance_continuous,
            relative_tolerance_continuous=relative_tolerance_continuous,
            tolerance_mesh=tolerance_mesh,
        )

    def refine_discontinuous(
        self,
        value: Variable | list[Variable | Iterable[float]],
        tolerance_discontinuous: float = 1.0e-3,
        num_point_min: int = 6,
        num_point_max: int = 12,
        mesh_length_min: float = 1.0e-3,
        mesh_length_max: float = 1,
    ):
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
        value: Variable | list[Variable | Iterable[float]],
        absolute_tolerance_continuous: float = 1.0e-8,
        relative_tolerance_continuous: float = 1.0e-8,
        tolerance_discontinuous: float = 1.0e-3,
        num_point_min: int = 6,
        num_point_max: int = 12,
        mesh_length_min: float = 1.0e-3,
        mesh_length_max: float = 1,
    ):
        """Adjust the mesh and the number of interpolation points to match the
        continuous error tolerance. Same as :meth:`refine_continuous`.

        Args:
            value: The variable to be checked. If the system has only one phase and no static variables, ``value`` can
                be a single `Variable` object. Otherwise, ``value`` should be a list of
                ``Variable`` objects, one for each ``Phase``, followed by an array
                as values of static variables.
            absolute_tolerance_continuous: Absolute tolerance for continuous error.
            relative_tolerance_continuous: Relative tolerance for continuous error.
            tolerance_discontinuous: In each subinterval, after scaling to :math:`[0, 1]`, the bang-bang control functions
                should either be less than ``tolerance_discontinuous`` or greater than ``1 - tolerance_discontinuous``
                simultaneously.
            num_point_min: Minimum number of interpolation points.
            num_point_max: Maximum number of interpolation points.
            mesh_length_min: Minimum mesh length.
            mesh_length_max: Maximum mesh length.

        Returns:
            The ``Variable`` s interpolated to the new discretization scheme.
        """
        return self.refine_continuous(
            value,
            absolute_tolerance_continuous=absolute_tolerance_continuous,
            relative_tolerance_continuous=relative_tolerance_continuous,
            num_point_min=num_point_min,
            num_point_max=num_point_max,
            mesh_length_min=mesh_length_min,
            mesh_length_max=mesh_length_max,
        )
