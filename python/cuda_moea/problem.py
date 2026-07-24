from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._specs import PythonProblem


@dataclass
class DTLZProblem:
    dimension: int = 12
    objectives: int = 3
    constraint_activation_ratio: float = 0.0
    _kind: str = "DTLZ2"

    def _spec(self) -> dict[str, Any]:
        return {"type": "DTLZProblem", "kind": self._kind,
                "dimension": self.dimension, "objectives": self.objectives,
                "constraint_activation_ratio": self.constraint_activation_ratio}


def _problem_class(name: str):
    class BuiltinProblem(DTLZProblem):
        def __init__(self, dimension: int = 12, objectives: int = 3,
                     constraint_activation_ratio: float = 0.0) -> None:
            super().__init__(dimension, objectives,
                             constraint_activation_ratio, name)
    BuiltinProblem.__name__ = name
    BuiltinProblem.__qualname__ = name
    return BuiltinProblem


DTLZ1 = _problem_class("DTLZ1")
DTLZ2 = _problem_class("DTLZ2")
DTLZ3 = _problem_class("DTLZ3")
DTLZ4 = _problem_class("DTLZ4")
DTLZ5 = _problem_class("DTLZ5")
DTLZ6 = _problem_class("DTLZ6")
DTLZ7 = _problem_class("DTLZ7")
ConvexDTLZ2 = _problem_class("ConvexDTLZ2")
C1DTLZ1 = _problem_class("C1DTLZ1")
C1DTLZ3 = _problem_class("C1DTLZ3")
C2DTLZ2 = _problem_class("C2DTLZ2")
C2ConvexDTLZ2 = _problem_class("C2ConvexDTLZ2")
C3DTLZ1 = _problem_class("C3DTLZ1")
C3DTLZ4 = _problem_class("C3DTLZ4")
CSDP = _problem_class("CSDP")

__all__ = ["PythonProblem", "DTLZProblem", "DTLZ1", "DTLZ2", "DTLZ3",
           "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7", "ConvexDTLZ2",
           "C1DTLZ1", "C1DTLZ3", "C2DTLZ2", "C2ConvexDTLZ2",
           "C3DTLZ1", "C3DTLZ4", "CSDP"]
