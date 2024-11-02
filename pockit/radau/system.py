# Copyright (c) 2024 Yilin Zou
from pockit.base.systembase import SystemBase, PhaseBase
from pockit.radau.phase import Phase


class System(SystemBase):
    def __init__(
        self,
        static_parameter: int | list[str],
        simplify: bool = False,
        parallel: bool = False,
        fastmath: bool = False,
    ) -> None:
        super().__init__(static_parameter, simplify, parallel, fastmath)

    @property
    def _class_phase(self) -> type[PhaseBase]:
        return Phase
