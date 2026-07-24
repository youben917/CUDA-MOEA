# DTLZ Groups A/B/C

English | [简体中文](README.zh-CN.md)

The launchers are `run_quality.py`, `run_population_timing.py`, and
`run_dimension_timing.py`. See the parent [execution guide](../README.md) for
the complete order, resume behavior, and validation commands.

Formal suite sizes are A=960, B=320, and C=440 runs. Group B uses nominal
populations 256, 512, 1024, 2048, 4096, 8192, 16384, and 32768. Group A saves
final objective matrices. Groups B/C save CUDA-event generation-loop timing,
excluding initialization, metric calculation, and disk I/O.

After completing Groups A/B/C, regenerate all derived data, statistical tests,
figures, and the optional Chinese detailed report:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cuda-moea \
python tests/benchmark/DTLZ/analyze_results.py \
  --runs-root output/benchmark_YYYYMMDD \
  --output tests/benchmark/DTLZ/results
```

Raw records remain below `output/`; committed derived tables and figures live in
`tests/benchmark/DTLZ/results/`. Update the authoritative English report from
the regenerated CSV files before release.
