# CUDA Result Evaluation

This directory contains two evaluation workflows:

- DTLZ: IGD and Hypervolume (HV), with the final algorithm front plotted
  against the analytic Pareto front.
- MoRobtrol: HV and Expected Utility (EU), with the selected algorithms' final
  nondominated fronts plotted together. The pooled empirical front is used
  internally for shared metric references, but is not drawn.

The implementation is split by responsibility: `snapshot_io.py` loads saved
runs, `pareto_front.py` generates analytic fronts, `visualization.py` plots
them, and `metrics/` contains one module per indicator. `evaluate_dtlz.py` and
`evaluate_morobtrol.py` are the two CLIs; matching `.ipynb` notebooks provide
small interactive tests for each workflow.

## DTLZ: IGD, HV, and analytic front

From the repository root:

```bash
python tests/evaluation/evaluate_dtlz.py \
  --run-dir output/nsga3_dtlz1 \
  --generation latest \
  --metrics igd hv \
  --pf-points 20000 \
  --plot output/nsga3_dtlz1/igd_front.png
```

`--generation latest` resolves to `metadata.json`'s `max_generations` value,
then loads the matching `generation_XXXXXX` directory.

The script expects the recorder layout documented in the project README:

- `metadata.json`
- `generation_*/snapshot.json`
- `generation_*/objectives.bin`
- optional `generation_*/constraints.bin`

`objectives.bin` is read as `float32` with objective-major layout `(M,N)` and
converted to row-major `(N,M)` for PyTorch and plotting.

For a small interactive check, open `tests/evaluation/evaluate_dtlz.ipynb` and
change only `run_dir` in its configuration cell.

HV uses the analytic front's nadir plus a 10% margin unless an explicit shared
`--reference-point` is provided. `auto` computes exact HV for one or two
objectives and reproducible scrambled Sobol Monte Carlo HV for three or more.

## MoRobtrol: HV, EU, and final fronts

Evaluate NSGA-III and RVEA outputs for one robot:

```bash
python tests/evaluation/evaluate_morobtrol.py \
  --run-root output/morobtrol/mo_swimmer \
  --generation latest \
  --hv-method auto \
  --hv-samples 16384 \
  --utility-samples 4096
```

The robot CLI restores rewards from the negative minimization objectives,
extracts each run's maximization nondominated front, and pools all selected
runs into one empirical nondominated reference front for metric normalization.
It writes `evaluation/metrics.json` and `evaluation/final_fronts.png`; the plot
contains only the algorithms' final nondominated fronts.

To pool multiple seeds or custom paths, repeat `--run LABEL=PATH`:

```bash
python tests/evaluation/evaluate_morobtrol.py \
  --run-root output/morobtrol/mo_swimmer \
  --run NSGA3-S42=output/seed42/mo_swimmer/nsga3 \
  --run RVEA-S42=output/seed42/mo_swimmer/rvea \
  --run NSGA3-S43=output/seed43/mo_swimmer/nsga3 \
  --run RVEA-S43=output/seed43/mo_swimmer/rvea
```

EU is the mean, over uniformly sampled simplex preference weights, of the best
linear utility available in the approximation set. A larger value is better.
The robot CLI normalizes all algorithms using the empirical front's shared
ideal and nadir points. HV also uses one shared reference point below the
empirical reward nadir. These shared values are saved in `metrics.json`.

For a small interactive robot comparison, open
`tests/evaluation/evaluate_morobtrol.ipynb` and change `run_root`.

The functions can also be imported directly. Both minimization and
maximization are supported:

```python
from tests.evaluation.metrics import expected_utility, hypervolume

hv = hypervolume(rewards, reference_point=[0.0, 0.0], maximize=True)
eu = expected_utility(
    rewards,
    maximize=True,
    ideal_point=[10.0, 10.0],
    nadir_point=[0.0, 0.0],
)
```

## Supported Analytic Fronts

The theoretical-front samplers cover every DTLZ problem exposed by the
CUDA-MOEA Python API:

- DTLZ1 simplex front: `DTLZ1`
- DTLZ2-style positive sphere fronts: `DTLZ2`, `DTLZ3`, `DTLZ4`
- degenerate curve fronts: `DTLZ5`, `DTLZ6`
- disconnected front: `DTLZ7`
- convex objective transform: `ConvexDTLZ2`
- constrained fronts: `C1DTLZ1`, `C1DTLZ3`, `C2DTLZ2`,
  `C2ConvexDTLZ2`, `C3DTLZ1`, `C3DTLZ4`

The C2 samplers retain only the feasible portions of the unconstrained front.
The C3 samplers construct their displaced Pareto fronts directly on the first
feasible radial constraint boundary. Constraint parameters and supported
objective counts intentionally match the CUDA kernels.
