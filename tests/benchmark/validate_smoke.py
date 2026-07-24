"""Validate all A--D independent five-generation smoke suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


EXPECTED = {
    "smoke_a": ("dtlz", "quality"),
    "smoke_b": ("dtlz", "timing"),
    "smoke_c": ("dtlz", "timing"),
    "smoke_d": ("robot", "quality"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    for experiment, (family, mode) in EXPECTED.items():
        manifest = args.runs_root / f"{experiment}_manifest.jsonl"
        if not manifest.exists():
            errors.append(f"缺少清单: {manifest}")
            continue
        jobs = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        if len(jobs) != 4:
            errors.append(f"{experiment}: 清单应为 4 次，实际 {len(jobs)}")
        paths = sorted((args.runs_root / experiment).glob("**/run_result.json"))
        if len(paths) != 4:
            errors.append(f"{experiment}: 结果应为 4 份，实际 {len(paths)}")
        implementations = set()
        for path in paths:
            row = json.loads(path.read_text(encoding="utf-8"))
            implementations.add((row.get("framework"), row.get("algorithm")))
            if row.get("run_status") != "success":
                errors.append(f"失败: {path} ({row.get('run_status')}: {row.get('error')})")
                continue
            if row.get("generations") != 5:
                errors.append(f"代数不是 5: {path}")
            if row.get("actual_population") is None:
                errors.append(f"缺少实际种群: {path}")
            if mode == "timing" and row.get("total_time_ms") is None:
                errors.append(f"缺少计时: {path}")
            if experiment == "smoke_a":
                value = row.get("final_objectives")
                if not value or not Path(value).exists():
                    errors.append(f"缺少最终目标矩阵: {path}")
            if experiment == "smoke_d":
                value = row.get("checkpoint_rewards")
                if not value or not Path(value).exists():
                    errors.append(f"缺少独立评估 reward: {path}")
                else:
                    checkpoints = torch.load(value, map_location="cpu", weights_only=True)
                    if sorted(map(int, checkpoints)) != [0, 5]:
                        errors.append(f"检查点应为 [0, 5]: {path}")
                if row.get("evaluation_status") != "success":
                    errors.append(f"独立评估未完成: {path}")
                if row.get("checkpoint_generations") != [0, 5]:
                    errors.append(f"检查点代数应为 [0, 5]: {path}")
                if len(row.get("checkpoint_interval_times_ms") or []) != 1:
                    errors.append(f"5 代区间耗时应为 1 个值: {path}")
                if len(row.get("checkpoint_cumulative_times_ms") or []) != 2:
                    errors.append(f"检查点累计耗时应为 2 个值: {path}")
                population = row.get("checkpoint_populations")
                if not population or not Path(population).exists():
                    errors.append(f"缺少种群检查点: {path}")
        expected_impl = {(f, a) for f in ("cuda_moea", "evox") for a in ("nsga3", "rvea")}
        if implementations != expected_impl:
            errors.append(f"{experiment}: 四种实现不完整: {sorted(implementations)}")
        print(f"{experiment}: {len(paths)}/4 result files")
    for error in errors:
        print("ERROR", error)
    print(f"Smoke validation: suites={len(EXPECTED)}, expected_runs={4 * len(EXPECTED)}, errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
