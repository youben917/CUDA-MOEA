# CUDA-MOEA Python/PyTorch API Reference

English | [简体中文](API.zh-CN.md)

This document describes the public `cuda_moea` Python package. The packaged
CUDA extension implements the algorithm lifecycle and built-in operators;
PyTorch provides tensor interchange and custom strategy callbacks.

## 1. Scope

The public API includes:

- NSGA-III and RVEA construction and lifecycle control
- 15 built-in multi-objective problems
- built-in mating, crossover, mutation, reference-direction, and environmental
  selection strategies
- PyTorch extension protocols for every strategy category
- CUDA tensor results, zero-copy population views, and optional copied results
- periodic population snapshots

The package does not expose a standalone CLI or a public native-language API.

## 2. Installation and import

Install the latest release from PyPI:

```bash
python -m pip install cuda-moea
```

A pre-built wheel is published for Linux x86_64, CPython 3.11, and PyTorch
built for CUDA 12.8, covering compute capabilities 80, 86, 89, 90, 100, and
120. On other Python versions, platforms, or PyTorch/CUDA stacks, pip builds
from the source archive instead.

A source build — from the archive or a checkout — requires Python 3.9+,
CUDA-enabled PyTorch 2.1+, CMake 3.24+, the CUDA Toolkit, OpenMP, and a
compatible host compiler. From the repository root:

```bash
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
  python -m pip install . --no-build-isolation
```

Replace both architecture values for the target GPU. Then import the package:

```python
import cuda_moea as cm

print(cm.__version__)  # 0.1.0
```

## 3. Minimal example

```python
import cuda_moea as cm

algorithm = cm.NSGA3(
    population_size=1024,
    max_generations=500,
    problem=cm.DTLZ2(dimension=12, objectives=3),
    device="cuda:0",
    seed=2887,
    print_progress=True,
)

result = algorithm.run()
print(result.variables.shape)    # torch.Size([1024, 12])
print(result.objectives.shape)   # torch.Size([1024, 3])
print(result.constraints.shape)  # torch.Size([1024])
```

By default, NSGA-III combines tournament mating, SBX, polynomial mutation,
Das–Dennis directions, and NSGA-III environmental selection. RVEA combines
random mating, SBX, polynomial mutation, adaptive RVEA directions, and RVEA
environmental selection.

## 4. Algorithm construction

### 4.1 `Algorithm`

```python
cm.Algorithm(
    algorithm="NSGA3",
    *,
    population_size=1024,
    max_generations=100,
    problem=None,
    mating=None,
    crossover=None,
    mutation=None,
    reference_directions=None,
    environment_selector=None,
    initial_population=None,
    cuda=None,
    device=None,
    seed=None,
    enable_warmup=None,
    progress_interval=100,
    print_progress=True,
    save_data=None,
    save_interval=1,
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `algorithm` | `str` | `"NSGA3"` | `"NSGA3"`, `"NSGA-3"`, or `"RVEA"` |
| `population_size` | `int` | `1024` | Requested population size |
| `max_generations` | `int` | `100` | Maximum generation count |
| `problem` | problem strategy | `DTLZ2()` | Built-in or Python problem |
| `mating` | mating strategy | algorithm default | Parent selection |
| `crossover` | crossover strategy | `SBX()` | Crossover operator |
| `mutation` | mutation strategy | `PolynomialMutation()` | Mutation operator |
| `reference_directions` | direction strategy | algorithm default | Reference provider |
| `environment_selector` | selection strategy | algorithm default | Survivor selection |
| `initial_population` | `torch.Tensor` or `None` | `None` | Optional `(N,D)` float32 initial decisions |
| `cuda` | `CudaConfig` or `None` | `CudaConfig()` | Full CUDA configuration |
| `device` | `int`, `str`, or `None` | `None` | Override `cuda.device_id` |
| `seed` | `int` or `None` | `None` | Override `cuda.seed` |
| `enable_warmup` | `bool` or `None` | `None` | Override `cuda.enable_warmup` |
| `progress_interval` | `int` | `100` | Progress-print interval |
| `print_progress` | `bool` | `True` | Enable progress output |
| `save_data` | path-like or `None` | `None` | Snapshot directory; `None` disables snapshots |
| `save_interval` | `int` | `1` | Snapshot interval in generations |

Explicit `device`, `seed`, and `enable_warmup` values take precedence over the
corresponding fields in `cuda`.

### 4.2 `NSGA3` and `RVEA`

These convenience classes accept the same keyword arguments while fixing the
algorithm name:

```python
nsga3 = cm.NSGA3(problem=cm.DTLZ1(12, 3))
rvea = cm.RVEA(problem=cm.DTLZ2(12, 3))
```

### 4.3 `AlgorithmBuilder`

```python
algorithm = (
    cm.AlgorithmBuilder("NSGA3")
    .population_size(1024)
    .max_generations(500)
    .problem(cm.DTLZ2(12, 3))
    .mating(cm.TournamentMating())
    .crossover(cm.SBX(eta_initial=30.0))
    .mutation(cm.PolynomialMutation(probability=1.0))
    .reference_directions(cm.DasDennisDirections())
    .environment_selector(cm.NSGA3EnvironmentSelector())
    .set(device="cuda:0", seed=2887, print_progress=False)
    .build()
)
```

The builder exposes `set(**options)` plus the named methods shown above.
Repeated options use the last value supplied.

## 5. Lifecycle and state

| Method | Return | Behavior |
| --- | --- | --- |
| `initialize()` | `None` | Allocate state and evaluate the initial population; repeated calls are no-ops |
| `step()` | `None` | Run one generation; initializes automatically and is a no-op after completion |
| `run(copy=True)` | `Result` | Run to `max_generations` and return the device result |
| `result(copy=True)` | `Result` | Finalize and return the current population; requires initialization |
| `reset()` | `None` | Reuse allocated resources and return to generation 0 with a new random population |
| `synchronize()` | `None` | Synchronize the internal evaluation and execution CUDA streams |

Manual generation control:

```python
algorithm.initialize()
while not algorithm.finished:
    algorithm.step()
    current = algorithm.population
final = algorithm.result()
```

Read-only properties are `initialized: bool`, `finished: bool`,
`generation: int`, and `population: Population`. Reading `population`
synchronizes internal streams; observing it every generation can reduce
throughput.

## 6. Results and tensor conventions

```python
@dataclass(frozen=True)
class Population:
    variables: torch.Tensor
    objectives: torch.Tensor
    constraints: torch.Tensor

@dataclass(frozen=True)
class Result(Population):
    auxiliary_indices: torch.Tensor
    active_count: int
    total_ms: float
```

| Field | dtype | Python shape | Meaning |
| --- | --- | --- | --- |
| `variables` | `float32` | `(N,D)` | Decision variables |
| `objectives` | `float32` | `(N,M)` | Objective values |
| `constraints` | `float32` | `(N,)` | Aggregated constraint violation; normally `<= 0` is feasible |
| `auxiliary_indices` | `int32` | algorithm-specific | NSGA-III ranks or RVEA front-zero indices |
| `active_count` | Python `int` | scalar | Active count reported by the selector |
| `total_ms` | Python `float` | scalar | CUDA timing returned by `run()` |

The native backend stores objectives in objective-major order `(M,N)`. The
Python view is `(N,M)` and may be non-contiguous; call `.contiguous()` when a
consumer requires contiguous storage.

`run(copy=True)` and `result(copy=True)` return independent tensors.
`copy=False` and `algorithm.population` return zero-copy views into native
storage. A zero-copy tensor keeps its native algorithm owner alive, but later
`step()`, `reset()`, or selection operations may overwrite its contents. Clone
data that must survive across generations.

CUDA-MOEA is an evolutionary optimizer and does not provide end-to-end
autograd. Custom model evaluation should normally use `torch.no_grad()`.

## 7. CUDA configuration

```python
cm.CudaConfig(
    device_id=0,
    evaluation_pool_ratio=0.25,
    execution_pool_ratio=0.75,
    memory_pool_policy=1,
    seed=2887,
    enable_warmup=True,
)
```

The two pool ratios must be positive. Different algorithm objects may use
different devices, but every built-in buffer and every custom callback output
for one object must remain on that object's CUDA device.

## 8. Built-in problems

Every built-in problem uses this constructor form:

```python
ProblemClass(
    dimension=12,
    objectives=3,
    constraint_activation_ratio=0.0,
)
```

Available classes are `DTLZ1` through `DTLZ7`, `ConvexDTLZ2`, `C1DTLZ1`,
`C1DTLZ3`, `C2DTLZ2`, `C2ConvexDTLZ2`, `C3DTLZ1`, `C3DTLZ4`, and `CSDP`.
`DTLZProblem` is their common specification base.

```python
problem = cm.C1DTLZ1(
    dimension=30,
    objectives=3,
    constraint_activation_ratio=0.5,
)
```

## 9. Built-in evolutionary operators

### 9.1 Mating

```python
cm.RandomMating()
cm.TournamentMating()
```

Tournament mating consumes rank/reference/score state supplied by the
environmental selector. Random mating does not depend on that state.

### 9.2 SBX crossover

```python
cm.SBX(
    eta_initial=30.0,
    eta_final=30.0,
    probability=1.0,
    variable_copy_probability=0.0,
)
```

The distribution index changes linearly from `eta_initial` to `eta_final`.
`probability` is the crossover probability. `variable_copy_probability`
controls the per-variable chance of copying a parent value directly; use `0.5`
to match the per-variable copy mask used in the repository's EvoX 1.3.0
benchmark adapter.

### 9.3 Polynomial mutation

```python
cm.PolynomialMutation(
    eta_initial=20.0,
    eta_final=20.0,
    probability=1.0,
)
```

The CUDA implementation applies a per-variable probability of
`probability / dimension`. Therefore `probability=1.0` gives the common `1/D`
mutation rate. Disable mutation with `cm.NoMutation()`.

## 10. Reference directions

```python
cm.DasDennisDirections(partitions=0)
cm.AdaptiveRVEADirections(frequency=0.1)
```

For Das–Dennis directions, `partitions=0` automatically chooses a one- or
two-layer set no larger than the requested population; a positive value sets
the partition count `H`. Adaptive RVEA's `frequency` is the adaptation interval
as a fraction of the total generation count.

User-supplied directions have shape `(K,M)`:

```python
import torch

directions = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])
provider = cm.UserDefinedDirections(directions, normalize=True)
```

Construction copies the values into core-managed device storage, so the input
may reside on the CPU.

## 11. Environmental selection

NSGA-III selection:

```python
cm.NSGA3EnvironmentSelector(
    sparse_ratio=0.5,
    cv_bins=0,
    cv_clip_upper=0.0,
    cv_log_alpha=0.0,
    feasibility_epsilon=0.0,
)
```

`sparse_ratio` controls the sparse non-dominated-sort path. The `cv_*` values
configure constraint-violation quantization, and `feasibility_epsilon` is the
feasibility tolerance.

RVEA selection:

```python
cm.RVEAEnvironmentSelector(alpha=2.0)
```

`alpha` controls how strongly angle-penalized distance increases with progress.

## 12. Custom problem protocol

```python
import torch
import cuda_moea as cm

class BiSphere(cm.PythonProblem):
    def __init__(self):
        super().__init__(
            dimension=20,
            objectives=2,
            lower_bounds=-5.0,
            upper_bounds=5.0,
            constraints=0,
            name="BiSphere",
        )

    def evaluate(self, variables, context):
        f1 = variables.square().sum(dim=1)
        f2 = (variables - 2.0).square().sum(dim=1)
        return torch.stack((f1, f2), dim=1)
```

Bounds may be scalars or sequences of length `dimension`. `evaluate()` may
return objectives, `(objectives, constraints)`, or a dictionary containing
`objectives` and `constraints`.

- `variables` is a float32 CUDA tensor with shape `(N,D)`.
- `objectives` must have shape `(N,M)`.
- `constraints`, when returned, must contain `N` values.
- Omitted constraints are cleared to zero.
- Outputs must be on the algorithm device; numeric outputs are converted to
  float32 by the binding.

The context is:

```python
{
    "generation": int,
    "max_generations": int,
    "device": int,
}
```

Initial-population evaluation uses `generation == -1`. Optional hooks are:

```python
initialize(context) -> None
prepare_parent(population, context) -> None | bool | evaluation_result
reset() -> None
```

`prepare_parent` runs before reproduction. An evaluation result replaces the
current parent objectives/constraints; `True` reports that the callback updated
the parent population in place.

## 13. Custom operator protocols

### 13.1 Mating

```python
class MyMating(cm.PythonMating):
    def initialize(self, info):
        self.population_size = info["population_size"]

    def select(self, parents, state, context):
        active = state.get("active_count", parents["variables"].shape[0])
        return torch.randint(
            0,
            active,
            (parents["variables"].shape[0],),
            device=parents["variables"].device,
            dtype=torch.int32,
        )
```

`select()` returns `N` parent indices. Selector state may include `rank`,
`reference_index`, `score`, and `active_count`; a mating implementation must not
assume every selector provides every field.

### 13.2 Crossover

```python
class CloneCrossover(cm.PythonCrossover):
    def apply(self, parents, parent_indices, context):
        return parents["variables"].index_select(0, parent_indices.long())
```

The return value must be an `(N,D)` tensor on the algorithm device.
`parent_indices` is an int32 CUDA tensor.

### 13.3 Mutation

```python
class GaussianMutation(cm.PythonMutation):
    def mutate(self, offspring, context):
        x = offspring["variables"]
        bounds = offspring["bounds"]
        mutated = x + 0.01 * torch.randn_like(x)
        return mutated.maximum(bounds[:, 0]).minimum(bounds[:, 1])
```

Mutation may return an `(N,D)` tensor or modify `offspring["variables"]` in
place and return `None`. Mating, crossover, and mutation may define an optional
`initialize(info)` method.

### 13.4 Reference directions

```python
class AxisDirections(cm.PythonReferenceDirections):
    def initialize(self, requested_count, objective_count, context):
        return torch.eye(
            objective_count,
            device=f"cuda:{context['device']}",
            dtype=torch.float32,
        )

    def update(self, population, directions, active_count, context):
        return None
```

`initialize()` returns `(K,M)`. `update()` may modify directions in place and
return `None`, or return a new tensor with the same shape. It must not change
`K` or `M`. `reset()` is optional.

## 14. Custom environmental selection

The full lifecycle is:

```python
initialize(info, references, context) -> None
prepare(population, context) -> state | None
select(parents, offspring, context) -> selection_dict
finalize(population, context) -> state | None  # optional
reset() -> None                                # optional
```

`references` is an `(K,M)` CUDA tensor, or `None` when no provider is present.

Select by indices into the concatenation `[parents; offspring]`:

```python
class FirstN(cm.PythonEnvironmentSelector):
    def select(self, parents, offspring, context):
        n = parents["variables"].shape[0]
        indices = torch.arange(n, device=parents["variables"].device)
        return {"indices": indices, "active_count": n}
```

`indices` must contain `N` values in `[0,2N)`. Alternatively, return all three
next-population tensors directly:

```python
{
    "variables": next_variables,      # (N,D)
    "objectives": next_objectives,    # (N,M)
    "constraints": next_constraints,  # (N,)
    "active_count": n,
}
```

Selector dictionaries may also contain:

| Key | Type | Purpose |
| --- | --- | --- |
| `rank` | int32 CUDA tensor | Tournament state and auxiliary result semantics |
| `reference_index` | int32 CUDA tensor | Reference association state |
| `score` | float32 CUDA tensor | Tournament comparison score |
| `active_count` | `int` | Active population count |
| `result_indices` | int32 CUDA tensor | Value exposed as `Result.auxiliary_indices` |

If custom mating depends on a state field, the selector must provide it from
`prepare()` before the first reproduction step.

## 15. Callback dictionaries

Population dictionaries contain non-owning views:

```python
{
    "variables": float32_cuda_tensor,    # (N,D)
    "objectives": float32_cuda_tensor,   # (N,M)
    "constraints": float32_cuda_tensor,  # (N,)
    "bounds": float32_cuda_tensor,       # (D,2)
}
```

Do not retain or modify these views unless the relevant protocol explicitly
permits in-place mutation.

Initialization `info` dictionaries contain:

```python
{
    "name": str,
    "population_size": int,
    "dimension": int,
    "objective_count": int,
    "max_generations": int,
    "device": int,
}
```

Generation contexts contain `generation`, `max_generations`, and `device`.

## 16. CUDA streams and callback performance

The library uses separate evaluation and execution streams. On entry to a
Python callback, the current PyTorch CUDA stream is switched to the relevant
external stream, so ordinary PyTorch operations follow algorithm dependencies.

Custom strategies should:

1. avoid `torch.cuda.synchronize()` unless host access truly requires it;
2. keep outputs on the input/algorithm device;
3. avoid CPU transfers inside callbacks;
4. reuse tensors and workspaces when possible;
5. avoid switching the default CUDA device; and
6. return without manual synchronization.

Python callbacks enter the interpreter each generation. Prefer a few batched
PyTorch operations over Python loops across individuals.

## 17. Snapshot data

```python
algorithm = cm.NSGA3(
    problem=cm.DTLZ2(12, 3),
    save_data="output/nsga3_run",
    save_interval=10,
)
```

Generation 0, interval generations, and the final generation are saved.
Decision variables use `(N,D)` row-major layout; objectives and references are
stored objective-major. Snapshotting synchronizes streams and copies data to
the host, so frequent snapshots can reduce throughput.

## 18. Troubleshooting

### Native extension import fails

Build the extension in the active Python/PyTorch environment. Check
`torch.version.cuda`, the CUDA Toolkit, and compiler ABI. Do not copy a compiled
extension from an unrelated environment.

### `must be on the algorithm CUDA device`

A callback returned a CPU tensor or a tensor on another GPU. Construct outputs
from the input tensor's `.device`.

### Shape validation fails

- problem objectives: `(N,M)`
- problem constraints: `N` values
- mating indices: `N` values
- crossover/mutation variables: `(N,D)`
- reference directions: `(K,M)`
- environmental-selection indices: `N` values

### Population reads are slow

`algorithm.population` synchronizes internal streams. Reduce per-generation
host observation or sample only selected generations.

### Results differ across versions or GPUs

Keep the seed, problem, population, CUDA architecture, and PyTorch version
fixed, then compare saved generations. Floating-point reduction order may still
produce small numeric differences.

## 19. Public symbol index

| Category | Symbols |
| --- | --- |
| Algorithms | `Algorithm`, `NSGA3`, `RVEA`, `AlgorithmBuilder` |
| Results | `Population`, `Result` |
| CUDA | `CudaConfig` |
| Problems | `PythonProblem`, `DTLZProblem`, `DTLZ1`–`DTLZ7`, `ConvexDTLZ2`, `C1DTLZ1`, `C1DTLZ3`, `C2DTLZ2`, `C2ConvexDTLZ2`, `C3DTLZ1`, `C3DTLZ4`, `CSDP` |
| Mating | `RandomMating`, `TournamentMating`, `PythonMating` |
| Crossover | `SBX`, `PythonCrossover` |
| Mutation | `PolynomialMutation`, `NoMutation`, `PythonMutation` |
| Directions | `DasDennisDirections`, `AdaptiveRVEADirections`, `UserDefinedDirections`, `PythonReferenceDirections` |
| Selection | `NSGA3EnvironmentSelector`, `RVEAEnvironmentSelector`, `PythonEnvironmentSelector` |
