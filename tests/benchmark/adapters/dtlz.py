"""Execute one timed DTLZ run with either framework."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import importlib.metadata
from pathlib import Path

import torch

from tests.benchmark.common import effective_population, requested_population
from tests.benchmark.adapters.shared import (
    initial_population,
    nsga3_reference_directions,
)


@dataclass
class DTLZResult:
    objectives: torch.Tensor
    total_time_ms: float
    requested_population: int
    actual_population: int
    active_population: int
    framework_version: str
    reference_direction_count: int | None


def _operators():
    from evox.operators.crossover.sbx import simulated_binary
    from evox.operators.mutation.pm_mutation import polynomial_mutation

    def sbx30(x):
        return simulated_binary(x, pro_c=1.0, dis_c=30.0)

    def pm20(x, lb, ub):
        return polynomial_mutation(x, lb, ub, pro_m=1.0, dis_m=20.0)

    return sbx30, pm20


def _evox_problem(name: str, dimension: int, objectives: int):
    from evox.core import Problem
    from evox.problems import numerical

    if name != "ConvexDTLZ2":
        return getattr(numerical, name)(d=dimension, m=objectives)

    class ConvexDTLZ2(Problem):
        def __init__(self):
            super().__init__()
            self.base = numerical.DTLZ2(d=dimension, m=objectives)

        def evaluate(self, x):
            values = self.base.evaluate(x)
            return torch.cat((values[:, :-1].pow(4), values[:, -1:].pow(2)), dim=1)

    return ConvexDTLZ2()


def _build_evox(name, algorithm, dimension, objectives, nominal, generations, seed, device):
    from evox.algorithms import NSGA3, RVEA
    from evox.workflows import StdWorkflow

    lb = torch.zeros(dimension, dtype=torch.float32, device=device)
    ub = torch.ones(dimension, dtype=torch.float32, device=device)
    sbx30, pm20 = _operators()
    common = dict(pop_size=nominal, n_objs=objectives, lb=lb, ub=ub,
                  crossover_op=sbx30, mutation_op=pm20, device=torch.device(device))
    algo = (NSGA3(**common) if algorithm == "nsga3" else
            RVEA(**common, alpha=2.0, fr=0.1, max_gen=generations))
    # EvoX 1.3.0 NSGA3 stores reference vectors as an unregistered plain
    # tensor, so Module.to(device) in StdWorkflow cannot move it.
    reference_count = None
    if algorithm == "nsga3":
        directions = nsga3_reference_directions(nominal, objectives)
        algo.ref = directions.to(device=device, dtype=torch.float32)
        reference_count = int(directions.shape[0])
    actual = int(algo.pop_size)
    algo.pop.copy_(initial_population(actual, dimension, seed, 0.0, 1.0).to(device))
    return StdWorkflow(algo, _evox_problem(name, dimension, objectives), monitor=None,
                       opt_direction="min", device=device), reference_count


def _run_evox(name, algorithm, dimension, objectives, nominal, generations, seed, device):
    # EvoX 1.3.0 NSGA3 creates temporary tensors inside step() using
    # torch.get_default_device() instead of the input tensor's device.
    # Each benchmark run is a dedicated child process, so setting this here is
    # isolated and guarantees every implicit tensor uses the requested GPU.
    torch.cuda.set_device(torch.device(device))
    torch.set_default_device(device)
    torch.manual_seed(seed)
    warmup, _ = _build_evox(name, algorithm, dimension, objectives, nominal, generations, seed, device)
    warmup.init_step()
    warmup.step()
    torch.cuda.synchronize(torch.device(device))
    del warmup
    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(seed)
    workflow, reference_count = _build_evox(
        name, algorithm, dimension, objectives, nominal, generations, seed, device)
    workflow.init_step()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(generations):
        workflow.step()
    stop.record()
    stop.synchronize()
    total_ms = start.elapsed_time(stop)
    population = workflow.algorithm.pop
    fitness = workflow.algorithm.fit
    valid_population = torch.isfinite(population).all(dim=1)
    valid_fitness = torch.isfinite(fitness).all(dim=1)
    # EvoX RVEA intentionally represents unoccupied reference-vector slots as
    # all-NaN rows. A finite solution with non-finite fitness is a real error.
    if torch.any(valid_population & ~valid_fitness):
        raise FloatingPointError("finite EvoX solution has non-finite objectives")
    valid = valid_population & valid_fitness
    if not torch.any(valid):
        raise FloatingPointError("EvoX returned no finite active solutions")
    values = fitness[valid].detach().clone()
    actual = int(workflow.algorithm.pop.shape[0])
    return DTLZResult(values, total_ms, nominal, actual, int(valid.sum().item()),
                      importlib.metadata.version("evox"), reference_count)


def _run_cuda(name, algorithm, dimension, objectives, nominal, generations, seed, device):
    import cuda_moea as cm

    requested = requested_population("cuda_moea", algorithm, objectives, nominal)
    problem = getattr(cm, name)(dimension=dimension, objectives=objectives,
                                constraint_activation_ratio=0.0)
    options = dict(
        population_size=requested, max_generations=generations, problem=problem,
        seed=seed, device=device, enable_warmup=True, print_progress=False,
        save_data=None, crossover=cm.SBX(probability=1.0, eta_initial=30.0, eta_final=30.0,
                                         variable_copy_probability=0.5),
        mutation=cm.PolynomialMutation(probability=1.0, eta_initial=20.0, eta_final=20.0),
        initial_population=initial_population(requested, dimension, seed, 0.0, 1.0),
    )
    reference_count = None
    if algorithm == "nsga3":
        options["environment_selector"] = cm.NSGA3EnvironmentSelector(sparse_ratio=1.0)
        directions = nsga3_reference_directions(nominal, objectives)
        options["reference_directions"] = cm.UserDefinedDirections(directions)
        reference_count = int(directions.shape[0])
    else:
        options["environment_selector"] = cm.RVEAEnvironmentSelector(alpha=2.0)
        options["reference_directions"] = cm.AdaptiveRVEADirections(frequency=0.1)
    result = (cm.NSGA3(**options) if algorithm == "nsga3" else cm.RVEA(**options)).run()
    values = result.objectives[:result.active_count].detach().clone()
    return DTLZResult(values, float(result.total_ms), requested,
                      int(result.objectives.shape[0]), int(result.active_count),
                      importlib.metadata.version("cuda-moea"), reference_count)


def run_dtlz(framework: str, **kwargs) -> DTLZResult:
    result = (_run_cuda(**kwargs) if framework == "cuda_moea" else _run_evox(**kwargs))
    expected = effective_population(kwargs["algorithm"], kwargs["objectives"], kwargs["nominal"])
    if result.actual_population != expected:
        raise RuntimeError(f"actual population {result.actual_population} != frozen expected {expected}")
    if not torch.isfinite(result.objectives).all():
        raise FloatingPointError("final objective matrix contains non-finite values")
    return result
