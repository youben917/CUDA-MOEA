"""Frozen MoRobtrol D-group matrix."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from tests.benchmark.common import rotated_implementations


ENVIRONMENTS = (
    ("mo_halfcheetah", 2), ("mo_hopper_m3", 3), ("mo_humanoid", 2),
    ("mo_humanoidstandup", 2), ("mo_inverted_double_pendulum", 2),
    ("mo_pusher", 3), ("mo_reacher", 2), ("mo_swimmer", 2), ("mo_walker2d", 2),
)
QUALITY_POPULATION = 16384
CHECKPOINT_INTERVAL = 5
HIDDEN_WIDTH = 16


@dataclass(frozen=True)
class Job:
    experiment: str
    environment: str
    objectives: int
    framework: str
    algorithm: str
    nominal_population: int
    generations: int
    seed: int
    repeat_index: int
    checkpoints: bool
    hidden_width: int = HIDDEN_WIDTH
    checkpoint_interval: int = CHECKPOINT_INTERVAL

    @property
    def relative_directory(self):
        scale = Path(f"n{self.nominal_population}")
        return Path(self.experiment, self.environment, scale,
                    f"repeat_{self.repeat_index:02d}_seed_{self.seed}",
                    f"{self.framework}_{self.algorithm}")


def _jobs(experiment, environments, populations, repeats, checkpoints,
          hidden_width=HIDDEN_WIDTH, checkpoint_interval=CHECKPOINT_INTERVAL):
    result = []
    for env_index, (environment, objectives) in enumerate(environments):
        for pop_index, population in enumerate(populations):
            for repeat in range(repeats):
                order = env_index + pop_index + repeat
                for framework, algorithm in rotated_implementations(order):
                    result.append(Job(
                        experiment, environment, objectives, framework, algorithm,
                        population, 100, repeat, repeat, checkpoints,
                        hidden_width, checkpoint_interval))
    return result


def quality_jobs():
    return _jobs("group_d", ENVIRONMENTS, (QUALITY_POPULATION,), 10, True)


def smoke_jobs(experiment: str, *, checkpoints: bool):
    return [Job(experiment, ENVIRONMENTS[0][0], 2, f, a, 128, 5, 0, 0,
                checkpoints, HIDDEN_WIDTH, CHECKPOINT_INTERVAL)
            for f, a in rotated_implementations(0)]
