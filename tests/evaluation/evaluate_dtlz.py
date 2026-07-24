"""Evaluate DTLZ snapshots with IGD, HV, and an analytic Pareto front."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

try:
    from .metrics import hypervolume, igd
    from .pareto_front import pareto_front_points
    from .snapshot_io import (
        default_torch_device,
        feasible_objectives,
        generation_number,
        load_metadata,
        require_torch,
        resolve_generation,
    )
    from .visualization import plot_fronts
except ImportError:  # Running as ``python tests/evaluation/evaluate_dtlz.py``.
    from metrics import hypervolume, igd
    from pareto_front import pareto_front_points
    from snapshot_io import (
        default_torch_device,
        feasible_objectives,
        generation_number,
        load_metadata,
        require_torch,
        resolve_generation,
    )
    from visualization import plot_fronts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--generation", default="latest")
    parser.add_argument(
        "--metrics", nargs="+", choices=("igd", "hv"), default=("igd", "hv")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feasibility-epsilon", type=float, default=0.0)
    parser.add_argument("--pf-points", type=int, default=20000)
    parser.add_argument("--reference-point", type=float, nargs="+")
    parser.add_argument(
        "--hv-method", choices=("auto", "exact", "monte_carlo"), default="auto"
    )
    parser.add_argument("--hv-samples", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=2887)
    parser.add_argument("--plot", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch = require_torch()
    run_dir = args.run_dir.resolve()
    metadata = load_metadata(run_dir)
    device = default_torch_device() if args.device == "auto" else args.device
    objectives = feasible_objectives(
        run_dir, args.generation, args.feasibility_epsilon
    ).to(device)
    reference = pareto_front_points(
        metadata["problem_name"],
        int(metadata["objective_count"]),
        count=args.pf_points,
        device=device,
        seed=args.seed,
    )
    ideal, nadir = reference.amin(dim=0), reference.amax(dim=0)
    span = (nadir - ideal).clamp_min(torch.finfo(reference.dtype).eps)
    hv_reference = (
        nadir + 0.1 * span
        if args.reference_point is None
        else torch.tensor(args.reference_point, dtype=objectives.dtype, device=device)
    )

    scores = {}
    if "igd" in args.metrics:
        scores["igd"] = igd(objectives, reference, device=device)
    if "hv" in args.metrics:
        scores["hypervolume"] = hypervolume(
            objectives,
            hv_reference,
            method=args.hv_method,
            samples=args.hv_samples,
            seed=args.seed,
            device=device,
        )
    generation_dir = resolve_generation(run_dir, args.generation)
    print(f"run_dir: {run_dir}")
    print(f"algorithm: {metadata.get('algorithm_name')}")
    print(f"problem: {metadata.get('problem_name')}")
    print(f"generation: {generation_number(generation_dir)}")
    print(f"feasible_points: {objectives.shape[0]}")
    for name, score in scores.items():
        print(f"{name}: {score:.8g}")
    if args.plot:
        score_text = " ".join(f"{name}={value:.4g}" for name, value in scores.items())
        plot_fronts(
            objectives,
            reference,
            args.plot.resolve(),
            title=f"{metadata.get('algorithm_name')} {score_text}",
        )
        print(f"plot: {args.plot.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
