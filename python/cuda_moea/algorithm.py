from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _C
from ._specs import (
    AdaptiveRVEADirections, CudaConfig, DasDennisDirections,
    NSGA3EnvironmentSelector, PolynomialMutation, RandomMating,
    RVEAEnvironmentSelector, SBX, TournamentMating, specification,
)
from .problem import DTLZ2


@dataclass(frozen=True)
class Population:
    variables: Any
    objectives: Any
    constraints: Any


@dataclass(frozen=True)
class Result(Population):
    auxiliary_indices: Any
    active_count: int
    total_ms: float


class Algorithm:
    def __init__(self, algorithm: str = "NSGA3", *, population_size: int = 1024,
                 max_generations: int = 100, problem: Any | None = None,
                 mating: Any | None = None, crossover: Any | None = None,
                 mutation: Any | None = None,
                 reference_directions: Any | None = None,
                 environment_selector: Any | None = None,
                 initial_population: Any | None = None,
                 cuda: CudaConfig | None = None, device: int | str | None = None,
                 seed: int | None = None, enable_warmup: bool | None = None,
                 progress_interval: int = 100, print_progress: bool = True,
                 save_data: str | Path | None = None,
                 save_interval: int = 1) -> None:
        kind = algorithm.upper().replace("-", "")
        if kind not in {"NSGA3", "RVEA"}:
            raise ValueError("algorithm must be 'NSGA3' or 'RVEA'")
        cuda = CudaConfig() if cuda is None else cuda
        if device is not None:
            cuda.device_id = _device_index(device)
        if seed is not None:
            cuda.seed = seed
        if enable_warmup is not None:
            cuda.enable_warmup = enable_warmup
        problem = DTLZ2() if problem is None else problem
        mating = (TournamentMating() if kind == "NSGA3" else RandomMating()) \
            if mating is None else mating
        crossover = SBX() if crossover is None else crossover
        mutation = PolynomialMutation() if mutation is None else mutation
        reference_directions = (
            DasDennisDirections() if kind == "NSGA3"
            else AdaptiveRVEADirections()
        ) if reference_directions is None else reference_directions
        environment_selector = (
            NSGA3EnvironmentSelector() if kind == "NSGA3"
            else RVEAEnvironmentSelector()
        ) if environment_selector is None else environment_selector
        config = {
            "population_size": population_size,
            "max_generations": max_generations,
            "progress_interval": progress_interval,
            "print_progress": print_progress,
            "cuda": vars(cuda),
            "save_enabled": save_data is not None,
            "save_directory": str(save_data or "cuda_moea_data"),
            "save_interval": save_interval,
            "initial_population": initial_population,
        }
        self._native = _C.Algorithm(
            kind, config, specification(problem), specification(mating),
            specification(crossover), specification(mutation),
            specification(reference_directions),
            specification(environment_selector),
        )

    def initialize(self) -> None:
        self._native.initialize()

    def step(self) -> None:
        self._native.step()

    def run(self, *, copy: bool = True) -> Result:
        return _result(self._native.run(copy))

    def result(self, *, copy: bool = True) -> Result:
        return _result(self._native.result(copy))

    def reset(self) -> None:
        self._native.reset()

    def synchronize(self) -> None:
        self._native.synchronize()

    @property
    def population(self) -> Population:
        value = self._native.population(False)
        return Population(value["variables"], value["objectives"],
                          value["constraints"])

    @property
    def initialized(self) -> bool:
        return self._native.initialized

    @property
    def finished(self) -> bool:
        return self._native.finished

    @property
    def generation(self) -> int:
        return self._native.generation


class NSGA3(Algorithm):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("NSGA3", **kwargs)


class RVEA(Algorithm):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("RVEA", **kwargs)


class AlgorithmBuilder:
    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm
        self.options: dict[str, Any] = {}

    def set(self, **options: Any) -> "AlgorithmBuilder":
        self.options.update(options)
        return self

    def population_size(self, value: int) -> "AlgorithmBuilder":
        return self.set(population_size=value)

    def max_generations(self, value: int) -> "AlgorithmBuilder":
        return self.set(max_generations=value)

    def problem(self, value: Any) -> "AlgorithmBuilder":
        return self.set(problem=value)

    def mating(self, value: Any) -> "AlgorithmBuilder":
        return self.set(mating=value)

    def crossover(self, value: Any) -> "AlgorithmBuilder":
        return self.set(crossover=value)

    def mutation(self, value: Any) -> "AlgorithmBuilder":
        return self.set(mutation=value)

    def reference_directions(self, value: Any) -> "AlgorithmBuilder":
        return self.set(reference_directions=value)

    def environment_selector(self, value: Any) -> "AlgorithmBuilder":
        return self.set(environment_selector=value)

    def build(self) -> Algorithm:
        return Algorithm(self.algorithm, **self.options)


def _result(value: dict[str, Any]) -> Result:
    return Result(value["variables"], value["objectives"],
                  value["constraints"], value["auxiliary_indices"],
                  value["active_count"], value["total_ms"])


def _device_index(device: int | str) -> int:
    if isinstance(device, int):
        return device
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError("CUDA-MOEA only supports CUDA devices")
