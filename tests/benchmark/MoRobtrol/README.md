# MoRobtrol Group D

English | [简体中文](README.zh-CN.md)

The entry points are `run_quality.py`, `evaluate_quality.py`,
`analyze_quality.py`, and `plot_quality_curves.ipynb`. See the parent
[execution guide](../README.md#3-formal-execution-order) for exact commands.

The formal suite has 360 optimization runs with nominal population 16384 and a
fixed hidden width of 16. Policy parameters are bounded to `[-5,5]`; training
uses 2 episodes of 1000 steps. Independent evaluation uses seed 90210 and 10
episodes.

Optimization saves populations at generations 0, 5, 10, …, 100. Each
`run_result.json` records checkpoint generations, five-generation interval
times, and cumulative checkpoint times. Independent evaluation creates
`checkpoint_rewards.pt` while retaining `checkpoint_populations.pt`.

Offline analysis derives HV, EU, quality-at-time, and time-to-target tables.
The notebook displays generation/time quality curves, final-front distributions,
and timing-quality tables without modifying optimization timing.

Use `--dry-run` to preview pending optimization tasks and `--validate-only` to
check optimization completeness. A successful Group D task with missing
population checkpoints is treated as pending. See [resume and validation](../README.md#4-resume-preview-and-validation)
for exit codes and the recommended recovery sequence.
