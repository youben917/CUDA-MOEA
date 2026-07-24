"""Frozen A/B/C job matrices from TEST_PLAN.md."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.benchmark.common import rotated_implementations


PROBLEMS = ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7", "ConvexDTLZ2")
STANDARD_DIMENSIONS = {"DTLZ1": 7, "DTLZ2": 12, "DTLZ3": 12, "DTLZ4": 12,
                       "DTLZ5": 12, "DTLZ6": 12, "DTLZ7": 12, "ConvexDTLZ2": 12}
POPULATIONS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
DIMENSIONS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)


@dataclass(frozen=True)
class Job:
    experiment: str
    framework: str
    algorithm: str
    problem: str
    objectives: int
    dimension: int
    nominal_population: int
    generations: int
    seed: int
    repeat_index: int
    save_objectives: bool

    @property
    def relative_directory(self) -> Path:
        return Path(self.experiment, self.problem.lower(),
                    f"m{self.objectives}_d{self.dimension}_n{self.nominal_population}",
                    f"repeat_{self.repeat_index:02d}_seed_{self.seed}",
                    f"{self.framework}_{self.algorithm}")


def _expand(experiment, configs, repeats, generations, save_objectives):
    result = []
    for config_index, (problem, dimension, population) in enumerate(configs):
        for repeat in range(repeats):
            for framework, algorithm in rotated_implementations(repeat + config_index):
                result.append(Job(experiment, framework, algorithm, problem, 3,
                                  dimension, population, generations, repeat, repeat,
                                  save_objectives))
    return result


def quality_jobs() -> list[Job]:
    return _expand("group_a", [(p, STANDARD_DIMENSIONS[p], 1024) for p in PROBLEMS],
                   30, 500, True)


def population_timing_jobs() -> list[Job]:
    return _expand("group_b", [("DTLZ1", 500, n) for n in POPULATIONS],
                   10, 100, False)


def dimension_timing_jobs() -> list[Job]:
    return _expand("group_c", [("DTLZ1", d, 1024) for d in DIMENSIONS],
                   10, 100, False)


def smoke_jobs(experiment: str = "smoke_a", *, save_objectives: bool = True) -> list[Job]:
    return _expand(experiment, [("DTLZ1", 7, 128)], 1, 5, save_objectives)
