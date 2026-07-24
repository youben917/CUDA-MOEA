"""Internal entry point for exactly one DTLZ run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.benchmark.adapters.dtlz import run_dtlz
from tests.benchmark.common import atomic_json, failure_record, requested_population


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("cuda_moea", "evox"), required=True)
    parser.add_argument("--algorithm", choices=("nsga3", "rvea"), required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--objectives", type=int, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--nominal-population", type=int, required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--save-objectives", action="store_true")
    args = parser.parse_args()
    environment = json.loads(args.environment_manifest.read_text(encoding="utf-8"))
    base = {
        "framework": args.framework, "algorithm": args.algorithm,
        "experiment": args.experiment, "problem": args.problem,
        "objectives": args.objectives, "dimension": args.dimension,
        "nominal_population": args.nominal_population,
        "generations": args.generations, "seed": args.seed,
        "repeat_index": args.repeat_index, "device": args.device,
        "initial_population_seed": args.seed,
        "initial_population_protocol": "torch_cpu_uniform_v1",
        "reference_direction_protocol": (
            "evox_uniform_sampling_v1" if args.algorithm == "nsga3" else None),
        "objective_sense": "minimize", "dtype": "float32",
        "git_commit": environment.get("git_commit"),
        "environment_manifest": str(args.environment_manifest.resolve()),
    }
    try:
        result = run_dtlz(
            args.framework, name=args.problem, algorithm=args.algorithm,
            dimension=args.dimension, objectives=args.objectives,
            nominal=args.nominal_population, generations=args.generations,
            seed=args.seed, device=args.device,
        )
        objective_path = None
        if args.save_objectives:
            objective_path = args.output / "final_objectives.pt"
            args.output.mkdir(parents=True, exist_ok=True)
            torch.save(result.objectives.cpu(), objective_path)
        record = {
            **base, "framework_version": result.framework_version,
            "requested_population": result.requested_population,
            "actual_population": result.actual_population,
            "active_population": result.active_population,
            "reference_direction_count": result.reference_direction_count,
            "total_time_ms": result.total_time_ms,
            "mean_generation_ms": result.total_time_ms / args.generations,
            "final_objectives": str(objective_path.resolve()) if objective_path else None,
            "final_igd": None, "hv": None, "eu": None,
            "time_budget_ms": None, "feasible_under_budget": None,
            "run_status": "success", "error": None,
        }
        atomic_json(args.output / "run_result.json", record)
        print("RESULT_JSON=" + json.dumps(record), flush=True)
        return 0
    except Exception as error:
        record = failure_record({
            **base, "framework_version": None,
            "requested_population": requested_population(args.framework, args.algorithm,
                                                           args.objectives, args.nominal_population),
            "actual_population": None, "total_time_ms": None,
            "active_population": None,
            "reference_direction_count": None,
            "mean_generation_ms": None, "final_objectives": None,
        }, error)
        atomic_json(args.output / "run_result.json", record)
        print("RESULT_JSON=" + json.dumps(record), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
