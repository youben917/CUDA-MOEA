# A–D Benchmark Execution Guide

English | [简体中文](README.zh-CN.md)

[`TEST_PLAN.md`](TEST_PLAN.md) is the authoritative parameter specification.
Execution scripts run algorithms, preserve raw matrices, record timing, and
validate completeness. Offline analysis reads saved data to calculate IGD, HV,
EU, and statistical tables; it is excluded from algorithm timing.

All commands below run from the repository root. Formal runs require an
environment that provides CUDA-MOEA, EvoX 1.3.0, Evomo, PyTorch, JAX, and Brax.
These benchmark-only dependencies are not declared as package runtime
dependencies.

## 1. Environment and preflight

```bash
export CUDA_VISIBLE_DEVICES=0

python -m pip show cuda-moea evox evomo torch jax brax
python tests/benchmark/validate_setup.py
python -c "import torch,jax; print(torch.cuda.get_device_name(0)); print(jax.devices())"
```

Choose a unique output directory for one immutable benchmark environment:

```bash
export BENCH_ROOT="$PWD/output/benchmark_YYYYMMDD"
mkdir -p "$BENCH_ROOT"
```

The first group run freezes `$BENCH_ROOT/manifest.json`. Do not change code,
dependencies, or the GPU during a formal suite.

## 2. Five-generation smoke suite

Keep smoke output separate from formal results:

```bash
export SMOKE_ROOT="$PWD/output/benchmark_YYYYMMDD_smoke"

python tests/benchmark/DTLZ/run_population_timing.py --output "$SMOKE_ROOT" --device cuda:0 --smoke
python tests/benchmark/DTLZ/run_dimension_timing.py  --output "$SMOKE_ROOT" --device cuda:0 --smoke
python tests/benchmark/DTLZ/run_quality.py           --output "$SMOKE_ROOT" --device cuda:0 --smoke

python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$SMOKE_ROOT" --device cuda:0 --smoke
python tests/benchmark/MoRobtrol/evaluate_quality.py \
  --runs-root "$SMOKE_ROOT" --experiment smoke_d \
  --environment mo_halfcheetah --device cuda:0

python tests/benchmark/validate_smoke.py --runs-root "$SMOKE_ROOT"
```

Each command runs the four framework/algorithm combinations serially. Matching
successful results are skipped on rerun; failed, OOM, or non-finite results
retain a `run_result.json` record and are retried.

## 3. Formal execution order

### Group B: DTLZ population scaling (320 runs)

```bash
python tests/benchmark/DTLZ/run_population_timing.py --output "$BENCH_ROOT" --device cuda:0
python tests/benchmark/DTLZ/run_population_timing.py --output "$BENCH_ROOT" --device cuda:0 --validate-only
```

### Group C: DTLZ dimension scaling (440 runs)

```bash
python tests/benchmark/DTLZ/run_dimension_timing.py --output "$BENCH_ROOT" --device cuda:0
python tests/benchmark/DTLZ/run_dimension_timing.py --output "$BENCH_ROOT" --device cuda:0 --validate-only
```

### Group A: DTLZ quality (960 runs)

```bash
python tests/benchmark/DTLZ/run_quality.py --output "$BENCH_ROOT" --device cuda:0
python tests/benchmark/DTLZ/run_quality.py --output "$BENCH_ROOT" --device cuda:0 --validate-only
```

### Group D: MoRobtrol quality (360 runs)

Run optimization and independent evaluation separately for each environment:

```bash
python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer

python tests/benchmark/MoRobtrol/evaluate_quality.py \
  --runs-root "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer

python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer --validate-only
```

Optimization saves populations at generations 0, 5, 10, …, 100 and records
each five-generation interval plus cumulative algorithm time. Checkpoint I/O is
outside algorithm timing. Independent evaluation uses seed 90210 and 10
episodes per policy to generate `checkpoint_rewards.pt`; population checkpoints
remain available for reevaluation.

After every environment has been evaluated, derive MoRobtrol metrics:

```bash
python tests/benchmark/MoRobtrol/analyze_quality.py \
  --runs-root "$BENCH_ROOT" \
  --output tests/benchmark/MoRobtrol/results
```

The analysis writes frozen metric references, checkpoint metrics and summaries,
quality-at-time, and time-to-target tables under `results/data/`. Open
`tests/benchmark/MoRobtrol/plot_quality_curves.ipynb` to display the derived
plots and tables; the notebook does not write report images or PDFs.

## 4. Resume, preview, and validation

All optimization launchers support `--stop-after N`, `--dry-run`,
`--validate-only`, `--rerun-successful`, and `--fail-fast`. MoRobtrol launchers
also accept one or more `--environment` values.

- `--dry-run` prints pending commands without running optimization. It may
  create or refresh the group manifest.
- `--validate-only` checks for configuration-matching terminal
  `run_result.json` files. Group D also requires population checkpoints for a
  successful optimization.
- `--rerun-successful` deliberately overwrites the normal skip behavior; omit
  it for routine recovery.
- Do not combine `--dry-run` and `--validate-only`.
- Do not run two formal launchers concurrently; the protocol requires serial
  use of the same exclusive GPU.

Example recovery sequence:

```bash
python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer --dry-run
python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer
python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer --validate-only
python tests/benchmark/MoRobtrol/evaluate_quality.py \
  --runs-root "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer
```

`--validate-only` returns 0 when every planned optimization has a terminal
record, including recorded OOM/NaN/timeout/failure outcomes. It returns 1 for a
missing or mismatched record, or for missing Group D population checkpoints.
Independent-evaluation completeness is checked by `evaluate_quality.py` and the
final validator.

## 5. Final validation and collection

```bash
python tests/benchmark/validate_results.py --runs-root "$BENCH_ROOT"

python tests/benchmark/collect_results.py \
  --runs-root "$BENCH_ROOT" \
  --output "$BENCH_ROOT/data"
```

Use `--groups group_b group_c` for a partial-stage validation. Preserve the
entire benchmark root: later metrics, confidence intervals, tests, plots, and
reports must be derived from its raw JSON and tensor artifacts.

## 6. Reports

- [Combined report](results/REPORT.md)
- [DTLZ analysis and report](DTLZ/results/REPORT.md)
- [MoRobtrol analysis and report](MoRobtrol/results/REPORT.md)

Reports describe only the recorded hardware/software environment and experiment
matrix. They are not general performance guarantees.
