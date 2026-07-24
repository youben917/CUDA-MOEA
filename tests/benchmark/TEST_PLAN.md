# CUDA-MOEA vs. EvoX Benchmark Protocol

English | [简体中文](TEST_PLAN.zh-CN.md)

This file is the authoritative parameter specification for the A–D benchmark
suite. Execution details are in [`README.md`](README.md).

## 1. Objective and comparison rules

The suite compares NSGA-III and RVEA in CUDA-MOEA and EvoX 1.3.0:

| Identifier | Framework | Algorithm |
| --- | --- | --- |
| CUDA-MOEA-NSGA-III | CUDA-MOEA | NSGA-III |
| EvoX-NSGA-III | EvoX | NSGA-III |
| CUDA-MOEA-RVEA | CUDA-MOEA | RVEA |
| EvoX-RVEA | EvoX | RVEA |

Speedup is calculated only between frameworks for the same algorithm:

$$
\text{Speedup}=\frac{T_{\text{EvoX}}}{T_{\text{CUDA-MOEA}}}.
$$

A value greater than 1 means CUDA-MOEA was faster in that recorded
configuration. NSGA-III and RVEA may be compared for solution quality, but not
as a framework speedup pair.

The suite covers eight unconstrained DTLZ problems, population and decision-
dimension scaling, and nine MoRobtrol environments with generation- and
time-indexed solution-quality analysis.

## 2. Harmonized implementations

### 2.1 Fixed settings

| Item | Setting |
| --- | --- |
| Numeric type | `float32` |
| Direction | Minimize DTLZ; minimize negative reward internally for MoRobtrol |
| SBX | probability 1.0, distribution index 30, per-variable parent-copy probability 0.5 |
| Polynomial mutation | probability parameter 1.0, distribution index 20; per-variable rate `1/D` |
| NSGA-III mating | Binary tournament |
| RVEA mating | Random |
| RVEA | `alpha=2.0`, `fr=0.1`, `max_gen` equal to the formal generation count |
| CUDA-MOEA NSGA-III | `sparse_ratio=1.0` for full non-dominated sorting |
| Execution | Same exclusive GPU; tasks run serially |

EvoX SBX must explicitly call
`simulated_binary(x, pro_c=1.0, dis_c=30.0)` because its 1.3.0 default
distribution index is 20. EvoX mutation explicitly calls
`polynomial_mutation(x, lb, ub, pro_m=1.0, dis_m=20.0)`.

Each record includes nominal, requested, and actual population. A pair with
different actual populations is excluded from speedup and paired-quality
statistics.

### 2.2 Three-objective population sizes

| Nominal population | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NSGA-III actual | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
| RVEA actual | 253 | 496 | 990 | 2016 | 4095 | 8128 | 16290 | 32640 |

EvoX RVEA changes the population to the count returned by
`uniform_sampling(nominal, 3)`. CUDA-MOEA RVEA uses that same effective size.
Plots use nominal population on the horizontal axis while raw records retain
all three population fields.

## 3. Reproducibility rules

- Freeze the GPU, CPU, driver, CUDA, Python, PyTorch, EvoX, JAX, Brax, Evomo,
  CUDA-MOEA revision, and environment manifest.
- Run every formal task serially on one device and rotate implementation order
  by repeat index.
- Use the same integer seed set for matching configurations.
- Generate shared `(N,D)` float32 initial populations with a CPU
  `torch.Generator(seed)` for Groups A–D.
- For NSGA-III, share the reference set produced by EvoX `uniform_sampling`.
- For MoRobtrol, construct matching `jax.random.PRNGKey` values, use
  `rotate_key=True` in training, and record the training-key seed.
- Run a one-seed, five-generation smoke suite before every experiment type.
- Preserve failed, timed-out, OOM, and non-finite outcomes; never silently
  reduce one implementation's configuration.
- Save raw objectives or rewards so every metric and plot can be regenerated
  after timing completes.

## 4. Timing protocol

### 4.1 CUDA-MOEA

Construct the algorithm with warm-up enabled, disable snapshots and progress
printing, call `run()`, and use `result.total_ms`. Calculate:

$$
\text{mean\_generation\_ms}=\frac{\text{total\_time\_ms}}{\text{generations}}.
$$

Do not replace `result.total_ms` with outer Python wall-clock timing.

### 4.2 EvoX

Construct the algorithm, problem, and non-monitoring `StdWorkflow`; execute
`workflow.init_step()`; warm an equivalent instance with at least one complete
`workflow.step()`; then enclose all formal steps in CUDA events. Synchronize the
ending event before reading elapsed time.

Both frameworks begin timing after initial-population evaluation. Timing
includes mating, crossover, mutation, problem evaluation, and environmental
selection, but excludes construction, first compilation, initialization,
metrics, disk I/O, and plotting.

Group D records each five-generation interval after algorithm execution and
before checkpoint I/O. Cumulative checkpoint time is the sum of intervals, with
generation 0 fixed to zero.

## 5. Experiment matrix

### 5.1 DTLZ problems

| Problem | Dimension | Characteristic |
| --- | ---: | --- |
| DTLZ1 | 7 | Linear, multimodal front |
| DTLZ2 | 12 | Spherical front |
| DTLZ3 | 12 | Strong multimodality |
| DTLZ4 | 12 | Distribution bias |
| DTLZ5 | 12 | Degenerate front |
| DTLZ6 | 12 | Degenerate front |
| DTLZ7 | 12 | Disconnected front |
| ConvexDTLZ2 | 12 | Convex transformed front |

All use three objectives, bounds `[0,1]`, and no constraints. Before formal
runs, compare CUDA-MOEA and EvoX values on a fixed random decision matrix with
maximum accepted error `1e-5`.

### 5.2 Group A: DTLZ quality and baseline timing

| Parameter | Value |
| --- | --- |
| Implementations | All four framework/algorithm combinations |
| Problems | All eight above |
| Nominal population | 1024; actual 1024 for NSGA-III and 990 for RVEA |
| Generations | 500 |
| Seeds | 0–29 |
| Saved output | Final objectives, total time, run status |

Total: `4 × 8 × 30 = 960` runs. Report IGD mean/std and median/IQR, plus
total and per-generation timing. IGD is computed offline.

### 5.3 Group B: population scaling

DTLZ1, three objectives, dimension 500, 100 generations, nominal populations
256 through 32768 from Section 2.2, and 10 timing repeats yield
`4 × 8 × 10 = 320` runs. Calculate speedup from framework medians for the same
algorithm and actual population; report a 95% bootstrap confidence interval.

### 5.4 Group C: decision-dimension scaling

DTLZ1, three objectives, nominal population 1024, 100 generations, dimensions
128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, and 131072, and 10
timing repeats yield `4 × 11 × 10 = 440` runs.

### 5.5 MoRobtrol environments

| Environment | Objectives | Policy MLP | Parameters |
| --- | ---: | --- | ---: |
| MoHalfCheetah | 2 | 17–16–6 | 390 |
| MoHopperM3 | 3 | 11–16–3 | 243 |
| MoHumanoid | 2 | 244–16–17 | 4,209 |
| MoHumanoidStandup | 2 | 244–16–17 | 4,209 |
| MoInvertedDoublePendulum | 2 | 8–16–1 | 161 |
| MoPusher | 3 | 23–16–7 | 503 |
| MoReacher | 2 | 11–16–2 | 226 |
| MoSwimmer | 2 | 8–16–2 | 178 |
| MoWalker2d | 2 | 17–16–6 | 390 |

Every policy is `Linear(observation,16) → Tanh → Linear(16,action) → Tanh`.
Both frameworks use the same evaluation implementation, `RobotPolicy`, bounds
`[-5,5]`, episode length 1000, and two training episodes. Reports convert
internally minimized negative reward back to reward maximization.

### 5.6 Group D: MoRobtrol quality

| Parameter | Value |
| --- | --- |
| Implementations | All four combinations |
| Environments | All nine above |
| Nominal population | 16384; three-objective RVEA actual 16290 |
| Generations | 100 |
| Optimization seeds | 0–9 |
| Independent evaluation | Seed 90210, 10 episodes per policy |
| Checkpoints | Generations 0, 5, 10, …, 100 |

Total: `4 × 9 × 10 = 360` runs. Preserve population and reward checkpoints,
interval and cumulative timing, and evaluation status. Compute HV, EU,
generation/time curves, quality-at-time, and time-to-target offline.

## 6. Metric definitions

### 6.1 DTLZ IGD

Remove non-finite rows, exact duplicates, and dominated points before IGD.
Share a fixed 20,000-point analytic reference front across all implementations
and seeds, using seed 2887 where sampling is required. Use the appropriate
linear, spherical, degenerate, disconnected, or convex-transformed front.
Compare rankings from 20,000 and 50,000 points; use 50,000 for any problem whose
method ordering changes.

### 6.2 MoRobtrol HV and EU

Compute exact two- and three-objective hypervolume in reward-maximization form.
For each environment, pool final rewards from the 40 formal runs to form an
empirical non-dominated front. Set the shared HV reference point 10% of the
objective range beyond the front's worst value.

Use the same empirical front to determine per-objective ideal and nadir values
for normalization. EU uses 10,000 simplex-uniform weights generated from a
fixed seed. Freeze HV and EU references in `metric_references.json` and reuse
them for every implementation, seed, and checkpoint.

### 6.3 Quality over time

Each checkpoint has generation, cumulative algorithm time, HV, and EU. For each
environment/algorithm/metric, quality-at-time budgets are the median EvoX times
at generations 25, 50, 75, and 100. For each run, use the latest checkpoint not
exceeding the budget and report missing counts.

The primary time-to-target threshold is the median final EvoX quality for the
same environment and algorithm; an optional lower target is 90% of that value.
Use the first checkpoint that reaches the target, report censored runs and reach
rate, and do not interpolate between checkpoints.

## 7. Analysis and reporting

- Select representative fronts by the run closest to median IGD (DTLZ) or
  median final HV (MoRobtrol), never the best run.
- Use common axes and views within each problem/environment.
- Label pooled MoRobtrol fronts as empirical, not theoretical.
- Use independent optimization seeds as quality samples and independent timing
  repeats as Group B/C performance samples.
- Compare frameworks within the same algorithm and problem/environment.
- Use Wilcoxon rank-sum tests unless identical random streams justify a paired
  signed-rank test. Apply Holm correction separately across the eight DTLZ
  problems and nine MoRobtrol environments at 0.05.
- Report rank-biserial correlation; a non-significant result is not evidence of
  equivalence.
- Report Group B/C medians and 95% bootstrap confidence intervals, with
  means/std only as supplementary summaries.

## 8. Required raw fields

Each record includes framework/version, algorithm, experiment, problem or
environment, objective count, dimension, nominal/requested/actual population,
generation count, seed/repeat, total and mean-generation time, run status and
error, source revision, and environment manifest. Relevant groups additionally
store final objectives, policy shape, population/reward checkpoints, interval
and cumulative checkpoint times, evaluation status, and offline metric fields.

## 9. Completion gate

Generate a final report only when all four implementations are represented,
actual populations match within every framework pair, all eight DTLZ problems
and nine MoRobtrol environments have no unexplained missing records, metrics can
be recomputed from preserved objectives/rewards, and every reported timing
excludes metric calculation, disk I/O, and plotting.
