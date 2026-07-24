"""Independently evaluate saved D-group populations while preserving them."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import traceback

import jax
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.benchmark.adapters.morobtrol import (
    evaluate_checkpoint_populations,
    shared_jax_device,
)
from tests.benchmark.common import atomic_json
from tests.benchmark.MoRobtrol.plan import ENVIRONMENTS, QUALITY_POPULATION


def _checkpoint_keys(path: Path) -> list[int]:
    values = torch.load(path, map_location="cpu", weights_only=True)
    return sorted(map(int, values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--experiment", choices=("group_d", "smoke_d"), default="group_d")
    parser.add_argument(
        "--environment", nargs="+", required=True,
        choices=tuple(environment for environment, _ in ENVIRONMENTS),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rerun-successful", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()

    paths = sorted((args.runs_root / args.experiment).glob("**/run_result.json"))
    selected = []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        current_population = (
            args.experiment != "group_d"
            or row.get("nominal_population") == QUALITY_POPULATION
        )
        if row.get("environment") in args.environment and current_population:
            selected.append((path, row))
    if not selected:
        raise FileNotFoundError(
            f"No D-group results for {args.environment} below {args.runs_root}")

    jax_device = shared_jax_device(args.device)
    failures = handled = 0
    for index, (result_path, row) in enumerate(selected, 1):
        reward_path = result_path.parent / "checkpoint_rewards.pt"
        population_value = row.get("checkpoint_populations")
        population_path = Path(population_value) if population_value else None
        if (not args.rerun_successful
                and row.get("evaluation_status") == "success"
                and reward_path.exists()):
            continue

        print(
            f"[{index}/{len(selected)}] {row.get('framework')}/{row.get('algorithm')} "
            f"{row.get('environment')} seed={row.get('seed')}",
            flush=True,
        )
        try:
            # Recover cleanly if reward saving completed before a previous interruption.
            recorded_reward = row.get("checkpoint_rewards")
            if (reward_path.exists() and recorded_reward
                    and Path(recorded_reward).resolve() == reward_path.resolve()
                    and not args.rerun_successful):
                keys = _checkpoint_keys(reward_path)
            else:
                if row.get("run_status") != "success":
                    raise RuntimeError(f"optimization status is {row.get('run_status')}")
                if population_path is None or not population_path.exists():
                    raise FileNotFoundError("checkpoint_populations.pt is missing")
                checkpoints = torch.load(
                    population_path, map_location="cpu", weights_only=True)
                with jax.default_device(jax_device):
                    rewards = evaluate_checkpoint_populations(
                        checkpoints,
                        environment=row["environment"],
                        objectives=int(row["objectives"]),
                        evaluation_seed=int(row["evaluation_seed"]),
                        evaluation_episodes=int(row["evaluation_episodes"]),
                        episode_length=int(row["episode_length"]),
                        device=args.device,
                        hidden_width=int(row["hidden_width"]),
                    )
                keys = sorted(map(int, rewards))
                temporary = reward_path.with_suffix(".pt.tmp")
                torch.save(rewards, temporary)
                temporary.replace(reward_path)
                del checkpoints, rewards

            interval = int(row.get("checkpoint_interval", 10))
            expected = list(range(0, int(row["generations"]) + 1, interval))
            if expected[-1] != int(row["generations"]):
                expected.append(int(row["generations"]))
            if keys != expected:
                raise ValueError(f"reward checkpoints {keys} != expected {expected}")
            row.update({
                "checkpoint_populations": population_value,
                "checkpoint_rewards": str(reward_path.resolve()),
                "evaluation_status": "success",
                "evaluation_error": None,
            })
            atomic_json(result_path, row)
        except Exception as error:
            failures += 1
            row.update({
                "evaluation_status": "failed",
                "evaluation_error": str(error),
                "evaluation_traceback": traceback.format_exc(),
            })
            atomic_json(result_path, row)
            print(f"ERROR {result_path}: {error}", flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        handled += 1
        if args.stop_after and handled >= args.stop_after:
            break

    print(f"Independent evaluation: selected={len(selected)} handled={handled} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
