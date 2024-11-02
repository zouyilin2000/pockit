# Copyright (c) 2024 Yilin Zou
from typing import Callable


class AutoUpdate:
    """Utility class for managing dependencies between functions."""

    n_in: int
    """Number of sources."""
    f_target: list[Callable[[], None]]
    """List of target functions."""
    dependent_table: list[list[int]]
    """Table of dependencies."""
    todo: list[bool]
    """List of targets that need to be updated."""

    def __init__(self, n_source: int, f_target: list[Callable[[], None]]) -> None:
        """
        Args:
            n_source: Number of sources.
            f_target: List of target functions
        """
        self.n_in = n_source
        self.f_target = f_target
        self.dependent_table = [[-1] * n_source for _ in f_target]
        self.todo = [False for _ in f_target]

    def set_dependency(self, i_target: int, i_sources: list[int]) -> None:
        """Set dependency of the target function with index ``i_target`` on the
        sources with indices ``i_sources``."""
        for i_source in i_sources:
            self.dependent_table[i_target][i_source] = 0

    def update(self, i_source: int) -> None:
        """Call the target functions that depend on the source with index
        ``i_source``."""
        for i_target, dt in enumerate(self.dependent_table):
            if dt[i_source] == 0:
                dt[i_source] = 1
            elif dt[i_source] == -1:
                continue
            if all(dt):
                self.todo[i_target] = True

        while True:
            for i_target, td in enumerate(self.todo):
                if td:
                    self.todo[i_target] = False
                    self.f_target[i_target]()
                    break
            else:
                break

    def update_all(self) -> None:
        """Update all targets."""
        [f() for f in self.f_target]
