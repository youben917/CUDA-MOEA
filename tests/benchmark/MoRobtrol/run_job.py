"""Internal entry point for one MoRobtrol D-group run."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.benchmark.adapters.morobtrol import HIDDEN_WIDTH, run_robot, shared_jax_device
from tests.benchmark.MoRobtrol.plan import CHECKPOINT_INTERVAL
from tests.benchmark.common import atomic_json, failure_record, requested_population


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", choices=("cuda_moea", "evox"), required=True)
    p.add_argument("--algorithm", choices=("nsga3", "rvea"), required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--environment", required=True)
    p.add_argument("--objectives", type=int, required=True)
    p.add_argument("--nominal-population", type=int, required=True)
    p.add_argument("--generations", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--repeat-index", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--environment-manifest", type=Path, required=True)
    p.add_argument("--checkpoints", action="store_true")
    p.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--evaluation-seed", type=int, default=90210)
    p.add_argument("--evaluation-episodes", type=int, default=10)
    p.add_argument("--hidden-width", type=int, default=HIDDEN_WIDTH)
    args = p.parse_args()
    if args.checkpoint_interval <= 0:
        p.error("--checkpoint-interval must be positive")
    environment_manifest = json.loads(args.environment_manifest.read_text(encoding="utf-8"))
    base = {"framework": args.framework, "algorithm": args.algorithm,
            "experiment": args.experiment, "environment": args.environment,
            "objectives": args.objectives, "dimension": None,
            "nominal_population": args.nominal_population,
            "generations": args.generations, "seed": args.seed,
            "repeat_index": args.repeat_index, "episode_length": args.episode_length,
            "training_episodes": args.episodes, "hidden_width": args.hidden_width,
            "checkpoint_interval": args.checkpoint_interval,
            "initial_population_seed": args.seed,
            "initial_population_protocol": "torch_cpu_uniform_v1",
            "training_random_key_seed": args.seed,
            "training_random_key_protocol": "jax_prngkey_v1",
            "training_rotate_key": True,
            "reference_direction_protocol": (
                "evox_uniform_sampling_v1" if args.algorithm == "nsga3" else None),
            "evaluation_seed": args.evaluation_seed,
            "evaluation_episodes": args.evaluation_episodes, "device": args.device,
            "objective_sense": "reward_maximization", "dtype": "float32",
            "time_budget_ms": None}
    base.update({"git_commit": environment_manifest.get("git_commit"),
                 "environment_manifest": str(args.environment_manifest.resolve()),
                 "final_igd": None, "hv": None, "eu": None})
    print(
        f"[run start] {args.experiment} {args.framework}/{args.algorithm} "
        f"{args.environment} n={args.nominal_population} h={args.hidden_width} "
        f"seed={args.seed}",
        flush=True,
    )
    try:
        jax_device = shared_jax_device(args.device)
        with jax.default_device(jax_device):
            result = run_robot(
                args.framework, environment=args.environment, objectives=args.objectives,
                nominal=args.nominal_population, generations=args.generations,
                algorithm=args.algorithm, seed=args.seed, episodes=args.episodes,
                episode_length=args.episode_length, device=args.device,
                checkpoints=args.checkpoints, evaluation_seed=args.evaluation_seed,
                evaluation_episodes=args.evaluation_episodes,
                hidden_width=args.hidden_width,
                checkpoint_interval=args.checkpoint_interval)
        populations_path = None
        if result.checkpoint_populations is not None:
            populations_path = args.output / "checkpoint_populations.pt"
            args.output.mkdir(parents=True, exist_ok=True)
            torch.save(result.checkpoint_populations, populations_path)
        total = result.total_time_ms
        record = {**base, "dimension": result.policy_parameters,
                  "framework_version": result.framework_version,
                  "requested_population": result.requested_population,
                  "actual_population": result.actual_population,
                  "active_population_count": result.active_population_count,
                  "reference_direction_count": result.reference_direction_count,
                  "total_time_ms": total,
                  "mean_generation_ms": total / args.generations if total is not None else None,
                  "checkpoint_generations": result.checkpoint_generations,
                  "checkpoint_interval_times_ms": result.checkpoint_interval_times_ms,
                  "checkpoint_interval_generations": (
                      [[a, b] for a, b in zip(
                          result.checkpoint_generations[:-1],
                          result.checkpoint_generations[1:])]
                      if result.checkpoint_generations is not None else None),
                  "checkpoint_cumulative_times_ms": result.checkpoint_cumulative_times_ms,
                  "checkpoint_active_counts": result.checkpoint_active_counts,
                  "checkpoint_populations": str(populations_path.resolve()) if populations_path else None,
                  "checkpoint_rewards": None,
                  "evaluation_status": "pending" if populations_path else None,
                  "feasible_under_budget": None,
                  "run_status": "success", "error": None}
        atomic_json(args.output / "run_result.json", record)
        print("RESULT_JSON=" + json.dumps(record), flush=True)
        return 0
    except Exception as error:
        atomic_json(args.output / "run_result.json", failure_record({
            **base, "framework_version": None,
            "requested_population": requested_population(args.framework, args.algorithm,
                                                           args.objectives, args.nominal_population),
            "actual_population": None, "total_time_ms": None,
            "active_population_count": None,
            "mean_generation_ms": None, "checkpoint_generations": None,
            "reference_direction_count": None,
            "checkpoint_interval_times_ms": None,
            "checkpoint_interval_generations": None,
            "checkpoint_cumulative_times_ms": None,
            "checkpoint_active_counts": None,
            "checkpoint_populations": None,
            "checkpoint_rewards": None, "evaluation_status": None,
            "feasible_under_budget": False}, error))
        return 1


if __name__ == "__main__": raise SystemExit(main())
