"""Unified MoRobtrol runners for the D-group quality experiment."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import importlib.metadata
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn

import cuda_moea as cm
from brax import envs
from evomo.problems.neuroevolution import MoRobtrol
from evox.utils import ParamsAndVector

from tests.benchmark.common import effective_population, requested_population
from tests.benchmark.adapters.shared import (
    initial_population,
    nsga3_reference_directions,
)
from tests.benchmark.MoRobtrol.plan import HIDDEN_WIDTH


_jax_clip = jnp.clip


def _compatible_clip(a, min=None, max=None, *, a_min=None, a_max=None):
    return _jax_clip(a, min=a_min if a_min is not None else min,
                     max=a_max if a_max is not None else max)


jnp.clip = _compatible_clip


class RobotPolicy(nn.Module):
    def __init__(self, observations: int, actions: int, hidden_width: int = HIDDEN_WIDTH):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observations, hidden_width), nn.Tanh(),
                                 nn.Linear(hidden_width, actions))

    def forward(self, observations):
        return torch.tanh(self.net(observations))


class CudaProblem(cm.PythonProblem):
    def __init__(self, problem, adapter, dimension, objectives, name):
        super().__init__(dimension, objectives, -5.0, 5.0, name=name)
        self.problem, self.adapter = problem, adapter

    @torch.no_grad()
    def evaluate(self, variables, context):
        return -self.problem.evaluate(self.adapter.batched_to_params(variables))


@dataclass
class RobotResult:
    total_time_ms: float | None
    requested_population: int
    actual_population: int
    active_population_count: int
    framework_version: str
    policy_parameters: int
    checkpoint_populations: dict[int, torch.Tensor] | None
    checkpoint_generations: list[int] | None
    checkpoint_interval_times_ms: list[float] | None
    checkpoint_cumulative_times_ms: list[float] | None
    checkpoint_active_counts: list[int] | None
    reference_direction_count: int | None


def _cumulative_times(interval_times: list[float]) -> list[float]:
    total = 0.0
    cumulative = [0.0]
    for value in interval_times:
        total += float(value)
        cumulative.append(total)
    return cumulative


def shared_jax_device(device: str):
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or torch_device.index is None:
        raise ValueError("device must be explicit, for example cuda:0")
    torch.cuda.set_device(torch_device)
    candidates = {d.id: d for d in jax.devices() if d.platform == "gpu"}
    if torch_device.index not in candidates:
        raise RuntimeError(f"JAX does not expose GPU {torch_device.index}; available={sorted(candidates)}")
    return candidates[torch_device.index]


def _robot_components(environment, objectives, population, seed, episodes,
                      episode_length, device, fixed_keys, hidden_width):
    env = envs.get_environment(environment)
    observations, actions = int(env.observation_size), int(env.action_size)
    model = RobotPolicy(observations, actions, hidden_width).to(device)
    adapter = ParamsAndVector(model)
    dimension = int(adapter.to_vector(dict(model.named_parameters())).numel())
    problem = MoRobtrol(
        policy=model, env_name=environment, max_episode_length=episode_length,
        num_episodes=episodes, seed=seed, pop_size=population,
        rotate_key=not fixed_keys, device=torch.device(device), num_obj=objectives,
        observation_shape=observations,
        obs_norm=torch.tensor([5.0, 1e-6, 1e6], device=device), useless=True,
    )
    return problem, adapter, dimension


def _ops():
    from evox.operators.crossover.sbx import simulated_binary
    from evox.operators.mutation.pm_mutation import polynomial_mutation

    def sbx30(x): return simulated_binary(x, pro_c=1.0, dis_c=30.0)
    def pm20(x, lb, ub): return polynomial_mutation(x, lb, ub, pro_m=1.0, dis_m=20.0)
    return sbx30, pm20


def _evox_workflow(environment, objectives, nominal, generations, algorithm,
                   seed, episodes, episode_length, device, hidden_width):
    from evox.algorithms import NSGA3, RVEA
    from evox.workflows import StdWorkflow

    actual = effective_population(algorithm, objectives, nominal)
    torch.manual_seed(seed)
    problem, adapter, dimension = _robot_components(
        environment, objectives, actual, seed, episodes, episode_length, device,
        False, hidden_width)
    lb = torch.full((dimension,), -5.0, dtype=torch.float32, device=device)
    ub = torch.full((dimension,), 5.0, dtype=torch.float32, device=device)
    sbx30, pm20 = _ops()
    common = dict(pop_size=nominal, n_objs=objectives, lb=lb, ub=ub,
                  crossover_op=sbx30, mutation_op=pm20, device=torch.device(device))
    algo = NSGA3(**common) if algorithm == "nsga3" else RVEA(
        **common, alpha=2.0, fr=0.1, max_gen=generations)
    # EvoX 1.3.0 leaves NSGA3.ref on CPU because it is not a registered
    # parameter/buffer. Move it explicitly before StdWorkflow wraps the algo.
    reference_count = None
    if algorithm == "nsga3":
        directions = nsga3_reference_directions(nominal, objectives)
        algo.ref = directions.to(device=device, dtype=torch.float32)
        reference_count = int(directions.shape[0])
    if int(algo.pop_size) != actual:
        raise RuntimeError(f"EvoX actual population {algo.pop_size} != expected {actual}")
    algo.pop.copy_(initial_population(actual, dimension, seed, -5.0, 5.0).to(device))
    workflow = StdWorkflow(algo, problem, monitor=None, opt_direction="max",
                           solution_transform=adapter, device=device)
    return workflow, dimension, reference_count


@torch.no_grad()
def _independent_evaluator(environment, objectives, population, evaluation_seed,
                           evaluation_episodes, episode_length, device, hidden_width):
    problem, adapter, dimension = _robot_components(
        environment, objectives, population, evaluation_seed, evaluation_episodes,
        episode_length, device, True, hidden_width)
    def evaluate(vectors):
        valid = torch.isfinite(vectors).all(dim=1)
        if not torch.any(valid):
            raise FloatingPointError("checkpoint contains no finite policies")
        # MoRobtrol is compiled for the full population shape. Replace RVEA's
        # empty all-NaN slots for evaluation, then discard their rewards.
        first = vectors[torch.where(valid)[0][0]]
        safe_vectors = torch.where(valid[:, None], vectors, first[None, :])
        rewards = problem.evaluate(adapter.batched_to_params(safe_vectors))
        return rewards[valid].detach().cpu()
    return evaluate, dimension


def _run_evox(environment, objectives, nominal, generations, algorithm, seed,
              episodes, episode_length, device, checkpoints, evaluation_seed,
              evaluation_episodes, hidden_width, checkpoint_interval):
    # Required by EvoX 1.3.0 NSGA3 helpers which allocate temporary tensors via
    # torch.get_default_device() during step(). The runner is a child process,
    # so this cannot leak into another framework run.
    torch.set_default_device(device)
    warmup, _, _ = _evox_workflow(environment, objectives, nominal, generations,
                                  algorithm, seed, episodes, episode_length, device,
                                  hidden_width)
    warmup.init_step(); warmup.step(); torch.cuda.synchronize()
    del warmup
    gc.collect()
    torch.cuda.empty_cache()
    workflow, dimension, reference_count = _evox_workflow(
        environment, objectives, nominal, generations, algorithm, seed, episodes,
        episode_length, device, hidden_width)
    saved = None
    if checkpoints:
        saved = {}
    workflow.init_step()
    if saved is not None:
        saved[0] = workflow.algorithm.pop.detach().cpu().clone()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    total = 0.0
    checkpoint_generations = [0] if saved is not None else None
    interval_times = [] if saved is not None else None
    active_counts = ([int(torch.isfinite(saved[0]).all(dim=1).sum())]
                     if saved is not None else None)
    start.record()
    for generation in range(1, generations + 1):
        workflow.step()
        if saved is not None and (generation % checkpoint_interval == 0
                                  or generation == generations):
            stop.record(); stop.synchronize()
            elapsed = float(start.elapsed_time(stop))
            total += elapsed
            interval_times.append(elapsed)
            checkpoint_generations.append(generation)
            print(f"Generation {generation} / {generations} completed", flush=True)
            saved[generation] = workflow.algorithm.pop.detach().cpu().clone()
            active_counts.append(int(torch.isfinite(saved[generation]).all(dim=1).sum()))
            if generation < generations:
                start.record()
    if saved is None:
        stop.record(); stop.synchronize()
        total = start.elapsed_time(stop)
    actual = int(workflow.algorithm.pop.shape[0])
    active = int(torch.isfinite(workflow.algorithm.pop).all(dim=1).sum())
    cumulative = _cumulative_times(interval_times) if interval_times is not None else None
    return RobotResult(
        total, nominal, actual, active, importlib.metadata.version("evox"), dimension,
        saved, checkpoint_generations, interval_times, cumulative, active_counts, reference_count)


def _cuda_algorithm(environment, objectives, nominal, generations, algorithm, seed,
                    episodes, episode_length, device, hidden_width, print_progress,
                    progress_interval):
    requested = requested_population("cuda_moea", algorithm, objectives, nominal)
    torch.manual_seed(seed)
    raw_problem, adapter, dimension = _robot_components(
        environment, objectives, requested, seed, episodes, episode_length, device,
        False, hidden_width)
    problem = CudaProblem(raw_problem, adapter, dimension, objectives, environment)
    options = dict(population_size=requested, max_generations=generations, problem=problem,
                   seed=seed, device=device, enable_warmup=True,
                   print_progress=print_progress, progress_interval=progress_interval,
                   save_data=None,
                   crossover=cm.SBX(eta_initial=30.0, eta_final=30.0, probability=1.0,
                                    variable_copy_probability=0.5),
                   mutation=cm.PolynomialMutation(eta_initial=20.0, eta_final=20.0,
                                                  probability=1.0))
    reference_count = None
    if algorithm == "nsga3":
        options["environment_selector"] = cm.NSGA3EnvironmentSelector(sparse_ratio=1.0)
        directions = nsga3_reference_directions(nominal, objectives)
        options["reference_directions"] = cm.UserDefinedDirections(directions)
        reference_count = int(directions.shape[0])
    else:
        options["environment_selector"] = cm.RVEAEnvironmentSelector(alpha=2.0)
        options["reference_directions"] = cm.AdaptiveRVEADirections(frequency=0.1)
    options["initial_population"] = initial_population(
        requested, dimension, seed, -5.0, 5.0)
    algo = cm.NSGA3(**options) if algorithm == "nsga3" else cm.RVEA(**options)
    return algo, requested, dimension, reference_count


def _run_cuda(environment, objectives, nominal, generations, algorithm, seed,
              episodes, episode_length, device, checkpoints, evaluation_seed,
              evaluation_episodes, hidden_width, checkpoint_interval):
    algo, requested, dimension, reference_count = _cuda_algorithm(
        environment, objectives, nominal, generations, algorithm, seed,
        episodes, episode_length, device, hidden_width, checkpoints,
        checkpoint_interval)
    if not checkpoints:
        result = algo.run()
        return RobotResult(float(result.total_ms), requested, int(result.objectives.shape[0]),
                           int(result.active_count),
                           importlib.metadata.version("cuda-moea"), dimension,
                           None, None, None, None, None, reference_count)
    saved = {}
    checkpoint_generations = [0]
    interval_times = []
    algo.initialize(); algo.synchronize()
    saved[0] = algo.population.variables.detach().cpu().clone()
    active_counts = [int(torch.isfinite(saved[0]).all(dim=1).sum())]
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    total = 0.0
    start.record()
    while not algo.finished:
        algo.step()
        if algo.generation % checkpoint_interval == 0 or algo.finished:
            algo.synchronize()
            stop.record(); stop.synchronize()
            elapsed = float(start.elapsed_time(stop))
            total += elapsed
            interval_times.append(elapsed)
            checkpoint_generations.append(int(algo.generation))
            saved[algo.generation] = algo.population.variables.detach().cpu().clone()
            active_counts.append(int(torch.isfinite(saved[algo.generation]).all(dim=1).sum()))
            if not algo.finished:
                start.record()
    result = algo.result()
    return RobotResult(total, requested, int(result.objectives.shape[0]), int(result.active_count),
                       importlib.metadata.version("cuda-moea"), dimension, saved,
                       checkpoint_generations, interval_times,
                       _cumulative_times(interval_times), active_counts, reference_count)


def run_robot(framework: str, **kwargs) -> RobotResult:
    result = (_run_cuda(**kwargs) if framework == "cuda_moea" else _run_evox(**kwargs))
    expected = effective_population(kwargs["algorithm"], kwargs["objectives"], kwargs["nominal"])
    if result.actual_population != expected:
        raise RuntimeError(f"actual population {result.actual_population} != expected {expected}")
    return result


@torch.no_grad()
def evaluate_checkpoint_populations(
    checkpoints: dict[int, torch.Tensor], *, environment: str, objectives: int,
    evaluation_seed: int, evaluation_episodes: int, episode_length: int,
    device: str, hidden_width: int,
) -> dict[int, torch.Tensor]:
    """Evaluate saved policy vectors under one fixed independent protocol."""
    if not checkpoints:
        raise ValueError("checkpoint population mapping is empty")
    generations = sorted(map(int, checkpoints))
    population = int(checkpoints[generations[0]].shape[0])
    evaluator, _ = _independent_evaluator(
        environment, objectives, population, evaluation_seed,
        evaluation_episodes, episode_length, device, hidden_width)
    rewards = {}
    for generation in generations:
        vectors = checkpoints[generation]
        if vectors.ndim != 2 or vectors.shape[0] != population:
            raise ValueError(f"invalid population shape at generation {generation}: {vectors.shape}")
        print(f"Evaluation generation {generation} started", flush=True)
        values = evaluator(vectors.to(device))
        if not torch.isfinite(values).all():
            raise FloatingPointError(
                f"independent reward matrix at generation {generation} contains non-finite values")
        rewards[generation] = values
        print(f"Evaluation generation {generation} completed", flush=True)
    return rewards
