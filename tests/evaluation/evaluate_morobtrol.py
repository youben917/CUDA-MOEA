"""Evaluate MoRobtrol runs with HV, EU, and an empirical nondominated front."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

try:
    from .metrics import expected_utility, hypervolume, nondominated_mask
    from .snapshot_io import (
        default_torch_device,
        feasible_objectives,
        load_metadata,
        require_torch,
    )
    from .visualization import plot_front_comparison
except ImportError:  # Running as ``python tests/evaluation/evaluate_morobtrol.py``.
    from metrics import expected_utility, hypervolume, nondominated_mask
    from snapshot_io import (
        default_torch_device,
        feasible_objectives,
        load_metadata,
        require_torch,
    )
    from visualization import plot_front_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Robot output directory containing nsga3/ and rvea/.",
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=("nsga3", "rvea")
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Explicit run; repeat to pool multiple algorithms or seeds.",
    )
    parser.add_argument("--generation", default="latest")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feasibility-epsilon", type=float, default=0.0)
    parser.add_argument("--reference-point", type=float, nargs="+")
    parser.add_argument("--reference-margin", type=float, default=0.1)
    parser.add_argument(
        "--hv-method", choices=("auto", "exact", "monte_carlo"), default="auto"
    )
    parser.add_argument("--hv-samples", type=int, default=16384)
    parser.add_argument("--utility-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2887)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to RUN_ROOT/evaluation.",
    )
    return parser


def _runs(args) -> list[tuple[str, Path]]:
    if args.run:
        result = []
        for value in args.run:
            if "=" not in value:
                raise ValueError(f"--run must use LABEL=PATH, got {value!r}")
            label, path = value.split("=", 1)
            if not label or not path:
                raise ValueError(f"--run must use LABEL=PATH, got {value!r}")
            result.append((label, Path(path)))
        return result
    return [(name.upper(), args.run_root / name) for name in args.algorithms]


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reference_margin < 0:
        raise ValueError("--reference-margin must be nonnegative")
    torch = require_torch()
    device = default_torch_device() if args.device == "auto" else args.device
    runs = _runs(args)
    labels = [label for label, _ in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("run labels must be unique")
    rewards = {}
    objective_count = None
    problem_name = None
    for label, path in runs:
        metadata = load_metadata(path.resolve())
        run_problem = metadata.get("problem_name")
        if run_problem:
            if problem_name is None:
                problem_name = run_problem
            elif run_problem != problem_name:
                raise ValueError("all pooled runs must use the same robot problem")
        values = -feasible_objectives(
            path.resolve(), args.generation, args.feasibility_epsilon
        ).to(device)
        if values.shape[0] == 0:
            raise ValueError(f"{path} contains no finite feasible solutions")
        if objective_count is None:
            objective_count = values.shape[1]
        elif values.shape[1] != objective_count:
            raise ValueError("all pooled runs must have the same objective count")
        rewards[label] = values
    problem_name = problem_name or args.run_root.name

    combined = torch.cat(tuple(rewards.values()), dim=0)
    empirical_front = combined[nondominated_mask(-combined)]
    fronts = {
        label: values[nondominated_mask(-values)]
        for label, values in rewards.items()
    }
    ideal = empirical_front.amax(dim=0)
    nadir = empirical_front.amin(dim=0)
    span = (ideal - nadir).clamp_min(torch.finfo(empirical_front.dtype).eps)
    utility_nadir = ideal - span
    hv_reference = (
        nadir - args.reference_margin * span
        if args.reference_point is None
        else torch.tensor(
            args.reference_point, dtype=empirical_front.dtype, device=device
        )
    )
    if hv_reference.shape != ideal.shape:
        raise ValueError("--reference-point must contain one value per objective")

    scores = {}
    for label, front in fronts.items():
        scores[label] = {
            "hypervolume": hypervolume(
                front,
                hv_reference,
                maximize=True,
                method=args.hv_method,
                samples=args.hv_samples,
                seed=args.seed,
                device=device,
            ),
            "expected_utility": expected_utility(
                front,
                maximize=True,
                ideal_point=ideal,
                nadir_point=utility_nadir,
                weight_samples=args.utility_samples,
                seed=args.seed,
                device=device,
            ),
            "final_points": int(rewards[label].shape[0]),
            "nondominated_points": int(front.shape[0]),
        }

    output = (args.output or args.run_root / "evaluation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "problem_name": problem_name,
        "generation": args.generation,
        "objective_count": objective_count,
        "empirical_front_points": int(empirical_front.shape[0]),
        "hv_reference": hv_reference.tolist(),
        "ideal_point": ideal.tolist(),
        "nadir_point": utility_nadir.tolist(),
        "hv_method": args.hv_method,
        "hv_samples": args.hv_samples,
        "utility_samples": args.utility_samples,
        "seed": args.seed,
        "scores": scores,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    plot_front_comparison(
        fronts,
        None,
        output / "final_fronts.png",
        title=f"{problem_name} — {' vs '.join(fronts)} final nondominated fronts",
    )
    print(json.dumps(report, indent=2))
    print(f"plot: {output / 'final_fronts.png'}")
    print(f"metrics: {output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
