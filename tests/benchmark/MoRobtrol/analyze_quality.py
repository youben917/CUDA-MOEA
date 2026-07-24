"""Offline HV/EU table generation for MoRobtrol group D.

The script reads optimization results plus independently evaluated checkpoint
rewards. It never launches optimization, evaluation, or plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.benchmark.MoRobtrol.plan import ENVIRONMENTS, QUALITY_POPULATION
from tests.benchmark.common import ALGORITHMS, FRAMEWORKS, atomic_json


LABELS = {
    ("cuda_moea", "nsga3"): "CUDA-MOEA–NSGA-III",
    ("evox", "nsga3"): "EvoX–NSGA-III",
    ("cuda_moea", "rvea"): "CUDA-MOEA–RVEA",
    ("evox", "rvea"): "EvoX–RVEA",
}
COLORS = {
    ("cuda_moea", "nsga3"): "#B2B1D8",
    ("evox", "nsga3"): "#EFC2B5",
    ("cuda_moea", "rvea"): "#9EB1D0",
    ("evox", "rvea"): "#DDADB9",
}


def _as_float(value):
    return "" if value is None else f"{float(value):.10g}"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_checkpoint_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row = dict(row)
            for key in ("seed", "repeat_index", "generation",
                        "reward_count", "reward_objectives"):
                row[key] = int(row[key])
            for key in ("checkpoint_cumulative_time_ms", "hv", "eu"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None,
                "q1": None, "q3": None, "iqr": None}
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def _nondominated_2d(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    best_y = -np.inf
    keep = []
    for idx in order:
        y = points[idx, 1]
        if y > best_y:
            keep.append(idx)
            best_y = y
    return points[keep]


def _nondominated_3d(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    order = np.lexsort((-points[:, 2], -points[:, 1], -points[:, 0]))
    ys: list[float] = []
    zs: list[float] = []
    keep = []
    import bisect

    for idx in order:
        y = float(points[idx, 1])
        z = float(points[idx, 2])
        pos = bisect.bisect_left(ys, y)
        if pos < len(ys) and zs[pos] >= z:
            continue
        keep.append(idx)
        left = bisect.bisect_right(ys, y)
        while left > 0 and zs[left - 1] <= z:
            left -= 1
        del ys[left:pos]
        del zs[left:pos]
        ys.insert(left, y)
        zs.insert(left, z)
    return points[keep]


def nondominated(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[1] == 2:
        return _nondominated_2d(points)
    if points.shape[1] == 3:
        return _nondominated_3d(points)
    raise ValueError(f"only 2D/3D rewards are supported, got {points.shape[1]}")


def hv_2d(points: np.ndarray, reference: np.ndarray) -> float:
    shifted = points - reference
    shifted = shifted[(shifted > 0).all(axis=1)]
    front = _nondominated_2d(shifted)
    if len(front) == 0:
        return 0.0
    front = front[np.argsort(front[:, 0])]
    area = 0.0
    previous_x = 0.0
    for x, y in front:
        area += max(0.0, x - previous_x) * max(0.0, y)
        previous_x = max(previous_x, x)
    return float(area)


def hv_3d(points: np.ndarray, reference: np.ndarray) -> float:
    shifted = points - reference
    shifted = shifted[(shifted > 0).all(axis=1)]
    front = _nondominated_3d(shifted)
    if len(front) == 0:
        return 0.0
    xs = np.unique(front[:, 0])
    xs.sort()
    volume = 0.0
    previous_x = 0.0
    for x in xs:
        if x <= previous_x:
            continue
        suffix = front[front[:, 0] >= x][:, 1:3]
        volume += (x - previous_x) * hv_2d(suffix, np.zeros(2))
        previous_x = x
    return float(volume)


def hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    return hv_2d(points, reference) if points.shape[1] == 2 else hv_3d(points, reference)


def simplex_weights(count: int, objectives: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = rng.exponential(1.0, size=(count, objectives))
    return weights / weights.sum(axis=1, keepdims=True)


def expected_utility(points: np.ndarray, ideal: np.ndarray, nadir: np.ndarray,
                     weights: np.ndarray) -> float:
    scale = np.maximum(ideal - nadir, 1e-12)
    normalized = np.clip((points - nadir) / scale, 0.0, 1.0)
    if len(normalized) == 0:
        return 0.0
    utilities = normalized @ weights.T
    return float(np.mean(np.max(utilities, axis=0)))


def load_runs(root: Path, environments: set[str], allow_incomplete: bool) -> list[dict]:
    rows = []
    for path in sorted((root / "group_d").glob("**/run_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("environment") not in environments:
            continue
        if row.get("nominal_population") != QUALITY_POPULATION:
            continue
        if row.get("run_status") != "success" or row.get("evaluation_status") != "success":
            if allow_incomplete:
                continue
            raise RuntimeError(f"incomplete D-group run: {path}")
        reward_path = Path(row["checkpoint_rewards"])
        rewards = torch.load(reward_path, map_location="cpu", weights_only=True)
        generations = list(map(int, row["checkpoint_generations"]))
        if sorted(map(int, rewards)) != generations:
            raise ValueError(f"reward/checkpoint generation mismatch: {path}")
        row["_path"] = path
        row["_rewards"] = {int(k): v.detach().cpu().numpy().astype(np.float64)
                           for k, v in rewards.items()}
        rows.append(row)
    return rows


def build_references(rows: list[dict], weight_count: int, weight_seed: int) -> dict:
    references = {}
    by_environment = defaultdict(list)
    for row in rows:
        final_generation = int(row["generations"])
        by_environment[row["environment"]].append(row["_rewards"][final_generation])
    for environment, matrices in by_environment.items():
        points = np.vstack(matrices)
        front = nondominated(points)
        worst = np.min(front, axis=0)
        best = np.max(front, axis=0)
        span = np.maximum(best - worst, 1e-12)
        reference = worst - 0.1 * span
        weights = simplex_weights(weight_count, front.shape[1], weight_seed)
        references[environment] = {
            "objectives": int(front.shape[1]),
            "hv_reference": reference.tolist(),
            "eu_ideal": best.tolist(),
            "eu_nadir": worst.tolist(),
            "eu_weight_seed": weight_seed,
            "eu_weight_count": weight_count,
            "empirical_front_points": int(front.shape[0]),
            "_weights": weights,
        }
    return references


def metric_rows(rows: list[dict], references: dict) -> list[dict]:
    result = []
    for row in rows:
        ref = references[row["environment"]]
        hv_reference = np.asarray(ref["hv_reference"], dtype=np.float64)
        ideal = np.asarray(ref["eu_ideal"], dtype=np.float64)
        nadir = np.asarray(ref["eu_nadir"], dtype=np.float64)
        weights = ref["_weights"]
        generations = list(map(int, row["checkpoint_generations"]))
        cumulative = list(map(float, row["checkpoint_cumulative_times_ms"]))
        for generation, time_ms in zip(generations, cumulative):
            points = row["_rewards"][generation]
            result.append({
                "environment": row["environment"],
                "framework": row["framework"],
                "algorithm": row["algorithm"],
                "algorithm_label": LABELS[(row["framework"], row["algorithm"])],
                "seed": int(row["seed"]),
                "repeat_index": int(row["repeat_index"]),
                "generation": int(generation),
                "checkpoint_cumulative_time_ms": float(time_ms),
                "hv": hypervolume(points, hv_reference),
                "eu": expected_utility(points, ideal, nadir, weights),
                "reward_count": int(points.shape[0]),
                "reward_objectives": int(points.shape[1]),
            })
    return result


def summarize_checkpoints(rows: list[dict]) -> list[dict]:
    groups = defaultdict(lambda: {"hv": [], "eu": [], "time": []})
    for row in rows:
        key = (row["environment"], row["framework"], row["algorithm"],
               row["algorithm_label"], row["generation"])
        groups[key]["hv"].append(row["hv"])
        groups[key]["eu"].append(row["eu"])
        groups[key]["time"].append(row["checkpoint_cumulative_time_ms"])
    summary = []
    for key, values in sorted(groups.items()):
        environment, framework, algorithm, label, generation = key
        hv = _summary(values["hv"])
        eu = _summary(values["eu"])
        tm = _summary(values["time"])
        summary.append({
            "environment": environment, "framework": framework, "algorithm": algorithm,
            "algorithm_label": label, "generation": generation,
            "hv_mean": _as_float(hv["mean"]), "hv_std": _as_float(hv["std"]),
            "hv_median": _as_float(hv["median"]), "hv_q1": _as_float(hv["q1"]),
            "hv_q3": _as_float(hv["q3"]),
            "eu_mean": _as_float(eu["mean"]), "eu_std": _as_float(eu["std"]),
            "eu_median": _as_float(eu["median"]), "eu_q1": _as_float(eu["q1"]),
            "eu_q3": _as_float(eu["q3"]),
            "time_ms_mean": _as_float(tm["mean"]),
            "time_ms_median": _as_float(tm["median"]),
            "time_ms_q1": _as_float(tm["q1"]),
            "time_ms_q3": _as_float(tm["q3"]),
        })
    return summary


def quality_at_time(rows: list[dict]) -> list[dict]:
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["environment"], row["algorithm"], row["framework"], row["seed"])].append(row)
    result = []
    for environment in sorted({r["environment"] for r in rows}):
        for algorithm in ALGORITHMS:
            evox_runs = [
                sorted(v, key=lambda r: r["generation"])
                for (env, algo, framework, _), v in by_key.items()
                if env == environment and algo == algorithm and framework == "evox"
            ]
            if not evox_runs:
                continue
            final_times = {}
            for framework in FRAMEWORKS:
                times = [
                    max(v, key=lambda r: r["generation"])["checkpoint_cumulative_time_ms"]
                    for (env, algo, fw, _), v in by_key.items()
                    if env == environment and algo == algorithm and fw == framework
                ]
                if times:
                    final_times[framework] = float(np.median(times))
            if set(final_times) != set(FRAMEWORKS):
                continue
            common_horizon = min(final_times.values())
            for budget_generation in (25, 50, 75, 100):
                times = [
                    run[[r["generation"] for r in run].index(budget_generation)]
                    ["checkpoint_cumulative_time_ms"]
                    for run in evox_runs
                    if budget_generation in {r["generation"] for r in run}
                ]
                if not times:
                    continue
                budget = float(np.median(times))
                if budget > common_horizon:
                    continue
                for framework in FRAMEWORKS:
                    for metric in ("hv", "eu"):
                        values = []
                        missing = 0
                        for (env, algo, fw, seed), run_rows in by_key.items():
                            if env != environment or algo != algorithm or fw != framework:
                                continue
                            candidates = [
                                r for r in run_rows
                                if r["checkpoint_cumulative_time_ms"] <= budget
                            ]
                            if not candidates:
                                missing += 1
                                continue
                            latest = max(candidates, key=lambda r: r["generation"])
                            values.append(float(latest[metric]))
                        stats = _summary(values)
                        result.append({
                            "environment": environment, "algorithm": algorithm,
                            "framework": framework,
                            "algorithm_label": LABELS[(framework, algorithm)],
                            "metric": metric,
                            "budget_source": f"EvoX generation {budget_generation} median",
                            "budget_time_ms": _as_float(budget),
                            "common_horizon_ms": _as_float(common_horizon),
                            "cuda_moea_final_time_median_ms": _as_float(final_times["cuda_moea"]),
                            "evox_final_time_median_ms": _as_float(final_times["evox"]),
                            "n": stats["n"], "missing": missing,
                            "mean": _as_float(stats["mean"]),
                            "std": _as_float(stats["std"]),
                            "median": _as_float(stats["median"]),
                            "q1": _as_float(stats["q1"]),
                            "q3": _as_float(stats["q3"]),
                        })
    return result


def time_to_target(rows: list[dict]) -> list[dict]:
    result = []
    by_run = defaultdict(list)
    for row in rows:
        by_run[(row["environment"], row["algorithm"], row["framework"], row["seed"])].append(row)
    for environment in sorted({r["environment"] for r in rows}):
        for algorithm in ALGORITHMS:
            for metric in ("hv", "eu"):
                evox_final = [
                    r[metric] for r in rows
                    if r["environment"] == environment and r["algorithm"] == algorithm
                    and r["framework"] == "evox" and r["generation"] == 100
                ]
                if not evox_final:
                    continue
                for target_label, target in (
                    ("EvoX final median", float(np.median(evox_final))),
                    ("90% EvoX final median", 0.9 * float(np.median(evox_final))),
                ):
                    for framework in FRAMEWORKS:
                        reached_times = []
                        censored = 0
                        for (env, algo, fw, _), run_rows in by_run.items():
                            if env != environment or algo != algorithm or fw != framework:
                                continue
                            ordered = sorted(run_rows, key=lambda r: r["generation"])
                            hit = next((r for r in ordered if r[metric] >= target), None)
                            if hit is None:
                                censored += 1
                            else:
                                reached_times.append(float(hit["checkpoint_cumulative_time_ms"]))
                        stats = _summary(reached_times)
                        total = stats["n"] + censored
                        result.append({
                            "environment": environment, "algorithm": algorithm,
                            "framework": framework,
                            "algorithm_label": LABELS[(framework, algorithm)],
                            "metric": metric,
                            "target_label": target_label,
                            "target_scope": "within observed 100-generation run",
                            "target_value": _as_float(target),
                            "reached": stats["n"],
                            "total": total,
                            "reached_rate": _as_float(stats["n"] / total if total else None),
                            "censored": censored,
                            "time_ms_median": _as_float(stats["median"]),
                            "time_ms_q1": _as_float(stats["q1"]),
                            "time_ms_q3": _as_float(stats["q3"]),
                        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/benchmark/MoRobtrol/results"))
    parser.add_argument("--environment", nargs="+",
                        choices=tuple(environment for environment, _ in ENVIRONMENTS))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--weight-count", type=int, default=10_000)
    parser.add_argument("--weight-seed", type=int, default=2887)
    parser.add_argument(
        "--reuse-checkpoint-metrics", action="store_true",
        help="reuse output/data/checkpoint_metrics.csv and only regenerate derived tables",
    )
    args = parser.parse_args()

    if args.reuse_checkpoint_metrics:
        rows = _read_checkpoint_metrics(args.output / "data/checkpoint_metrics.csv")
        _write_csv(args.output / "data/checkpoint_metric_summary.csv",
                   summarize_checkpoints(rows))
        _write_csv(args.output / "data/quality_at_time.csv", quality_at_time(rows))
        _write_csv(args.output / "data/time_to_target.csv", time_to_target(rows))
        print(f"Reused {len(rows)} checkpoint metric rows; wrote derived tables to {args.output}")
        return 0

    environments = set(args.environment or [environment for environment, _ in ENVIRONMENTS])
    runs = load_runs(args.runs_root, environments, args.allow_incomplete)
    if not runs:
        raise FileNotFoundError(f"no evaluated group_d runs found below {args.runs_root}")
    references = build_references(runs, args.weight_count, args.weight_seed)
    serializable_references = {
        env: {k: v for k, v in ref.items() if not k.startswith("_")}
        for env, ref in references.items()
    }
    atomic_json(args.output / "data/metric_references.json", serializable_references)

    rows = metric_rows(runs, references)
    _write_csv(args.output / "data/checkpoint_metrics.csv", rows)
    _write_csv(args.output / "data/checkpoint_metric_summary.csv", summarize_checkpoints(rows))
    _write_csv(args.output / "data/quality_at_time.csv", quality_at_time(rows))
    _write_csv(args.output / "data/time_to_target.csv", time_to_target(rows))
    print(f"Analyzed {len(runs)} runs; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
