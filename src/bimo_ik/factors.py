"""The experimental cell matrix.

An intentionally unbalanced factorial:

  algorithm {PSO, GA} x memory {cold, injection, persistence} x landscape
  {static, adaptive}, where persistence exists ONLY for PSO — 10 cells.

The imbalance is by design (persistence has no principled GA analog) and the
statistics module analyzes it with planned comparisons, never a full-crossed
ANOVA over cells that do not exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .algorithms.base import MemoryMode


class AlgorithmKind(str, Enum):
    PSO = "pso"
    GA = "ga"


class LandscapeMode(str, Enum):
    STATIC = "static"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class Cell:
    algorithm: AlgorithmKind
    memory: MemoryMode
    landscape: LandscapeMode

    def __post_init__(self):
        # normalize string inputs to enums (keeps configs terse)
        object.__setattr__(self, "algorithm", AlgorithmKind(self.algorithm))
        object.__setattr__(self, "memory", MemoryMode(self.memory))
        object.__setattr__(self, "landscape", LandscapeMode(self.landscape))
        if (
            self.algorithm is AlgorithmKind.GA
            and self.memory is MemoryMode.PERSISTENCE
        ):
            raise ValueError("persistence is PSO-exclusive by design")

    @property
    def cell_id(self) -> str:
        return f"{self.algorithm.value}-{self.memory.value}-{self.landscape.value}"


def full_matrix() -> tuple[Cell, ...]:
    """All 10 cells of the study."""
    cells = []
    for landscape in LandscapeMode:
        for memory in (MemoryMode.COLD, MemoryMode.INJECTION, MemoryMode.PERSISTENCE):
            cells.append(Cell(AlgorithmKind.PSO, memory, landscape))
        for memory in (MemoryMode.COLD, MemoryMode.INJECTION):
            cells.append(Cell(AlgorithmKind.GA, memory, landscape))
    return tuple(cells)
