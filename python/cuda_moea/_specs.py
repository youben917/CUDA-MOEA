from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


class _Spec:
    def _spec(self) -> dict[str, Any]:
        data = asdict(self) if hasattr(self, "__dataclass_fields__") else {}
        data["type"] = type(self).__name__
        return data


@dataclass
class CudaConfig:
    device_id: int = 0
    evaluation_pool_ratio: float = 0.25
    execution_pool_ratio: float = 0.75
    memory_pool_policy: int = 1
    seed: int = 2887
    enable_warmup: bool = True


@dataclass
class SBX(_Spec):
    eta_initial: float = 30.0
    eta_final: float = 30.0
    probability: float = 1.0
    variable_copy_probability: float = 0.0


@dataclass
class PolynomialMutation(_Spec):
    eta_initial: float = 20.0
    eta_final: float = 20.0
    probability: float = 1.0


@dataclass
class NoMutation(_Spec):
    pass


@dataclass
class RandomMating(_Spec):
    pass


@dataclass
class TournamentMating(_Spec):
    pass


@dataclass
class DasDennisDirections(_Spec):
    partitions: int = 0


@dataclass
class AdaptiveRVEADirections(_Spec):
    frequency: float = 0.1


@dataclass
class UserDefinedDirections(_Spec):
    values: Any = None
    normalize: bool = True

    def _spec(self) -> dict[str, Any]:
        return {"type": "UserDefinedDirections", "values": self.values,
                "normalize": self.normalize}


@dataclass
class NSGA3EnvironmentSelector(_Spec):
    sparse_ratio: float = 0.5
    cv_bins: int = 0
    cv_clip_upper: float = 0.0
    cv_log_alpha: float = 0.0
    feasibility_epsilon: float = 0.0


@dataclass
class RVEAEnvironmentSelector(_Spec):
    alpha: float = 2.0


class PythonProblem(_Spec):
    def __init__(self, dimension: int, objectives: int,
                 lower_bounds: float | Sequence[float] = 0.0,
                 upper_bounds: float | Sequence[float] = 1.0,
                 constraints: int = 0, name: str | None = None) -> None:
        self.dimension = dimension
        self.objectives = objectives
        self.lower_bounds = _expand(lower_bounds, dimension)
        self.upper_bounds = _expand(upper_bounds, dimension)
        self.constraints = constraints
        self.name = name or type(self).__name__

    def _spec(self) -> dict[str, Any]:
        return {"type": "PythonProblem", "object": self,
                "dimension": self.dimension, "objectives": self.objectives,
                "lower_bounds": self.lower_bounds,
                "upper_bounds": self.upper_bounds,
                "constraints": self.constraints, "name": self.name}

    def evaluate(self, variables: Any, context: dict[str, Any]) -> Any:
        raise NotImplementedError


class PythonMating(_Spec):
    def _spec(self) -> dict[str, Any]:
        return {"type": "PythonMating", "object": self}
    def select(self, parents: dict[str, Any], state: dict[str, Any],
               context: dict[str, Any]) -> Any:
        raise NotImplementedError


class PythonCrossover(_Spec):
    def _spec(self) -> dict[str, Any]:
        return {"type": "PythonCrossover", "object": self}
    def apply(self, parents: dict[str, Any], parent_indices: Any,
              context: dict[str, Any]) -> Any:
        raise NotImplementedError


class PythonMutation(_Spec):
    def _spec(self) -> dict[str, Any]:
        return {"type": "PythonMutation", "object": self}
    def mutate(self, offspring: dict[str, Any], context: dict[str, Any]) -> Any:
        raise NotImplementedError


class PythonReferenceDirections(_Spec):
    def _spec(self) -> dict[str, Any]:
        return {"type": "PythonReferenceDirections", "object": self}
    def initialize(self, requested_count: int, objective_count: int,
                   context: dict[str, Any]) -> Any:
        raise NotImplementedError


class PythonEnvironmentSelector(_Spec):
    def _spec(self) -> dict[str, Any]:
        return {"type": "PythonEnvironmentSelector", "object": self}
    def initialize(self, info: dict[str, Any], references: Any,
                   context: dict[str, Any]) -> None:
        pass
    def prepare(self, population: dict[str, Any],
                context: dict[str, Any]) -> Any:
        return None
    def select(self, parents: dict[str, Any], offspring: dict[str, Any],
               context: dict[str, Any]) -> Any:
        raise NotImplementedError


def _expand(value: float | Sequence[float], size: int) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * size
    result = [float(v) for v in value]
    if len(result) != size:
        raise ValueError(f"expected {size} bounds, got {len(result)}")
    return result


def specification(value: Any) -> dict[str, Any]:
    if not hasattr(value, "_spec"):
        raise TypeError(f"{value!r} is not a cuda_moea strategy")
    return value._spec()
