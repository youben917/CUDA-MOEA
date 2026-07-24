"""Strict structural validation of complete A--D raw benchmark data."""

from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import torch

EXPECTED = {"group_a": 960, "group_b": 320, "group_c": 440, "group_d": 360}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--groups", nargs="+", default=list(EXPECTED), choices=tuple(EXPECTED))
    args = p.parse_args()
    errors, warnings = [], []
    records = []
    for group in args.groups:
        manifest = args.runs_root / f"{group}_manifest.jsonl"
        if not manifest.exists():
            errors.append(f"missing manifest: {manifest}"); continue
        jobs = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        if len(jobs) != EXPECTED[group]: errors.append(f"{group}: manifest has {len(jobs)}, expected {EXPECTED[group]}")
        paths = sorted((args.runs_root / group).glob("**/run_result.json"))
        if group == "group_d":
            planned_populations = {job.get("nominal_population") for job in jobs}
            paths = [path for path in paths if
                     json.loads(path.read_text(encoding="utf-8")).get("nominal_population")
                     in planned_populations]
        if len(paths) != EXPECTED[group]: errors.append(f"{group}: results has {len(paths)}, expected {EXPECTED[group]}")
        for path in paths:
            row = json.loads(path.read_text(encoding="utf-8")); records.append(row)
            status = row.get("run_status")
            if status not in {"success", "oom", "nan", "timeout", "failed"}:
                errors.append(f"invalid status: {path} ({status})")
            elif status != "success":
                warnings.append(f"recorded {status}: {path}")
            if status == "success" and row.get("actual_population") is None:
                errors.append(f"missing actual population: {path}")
            if status == "success" and group in {"group_a", "group_d"}:
                key = "final_objectives" if group == "group_a" else "checkpoint_rewards"
                value = row.get(key)
                if not value or not Path(value).exists(): errors.append(f"missing {key}: {path}")
                elif group == "group_d":
                    data = torch.load(value, map_location="cpu", weights_only=True)
                    interval = int(row.get("checkpoint_interval", 5))
                    expected_checkpoints = list(range(0, int(row["generations"]) + 1, interval))
                    if expected_checkpoints[-1] != int(row["generations"]):
                        expected_checkpoints.append(int(row["generations"]))
                    if sorted(map(int, data)) != expected_checkpoints:
                        errors.append(f"bad checkpoints: {path}")
                    if row.get("checkpoint_generations") != expected_checkpoints:
                        errors.append(f"bad checkpoint generations: {path}")
                    interval_times = row.get("checkpoint_interval_times_ms")
                    cumulative_times = row.get("checkpoint_cumulative_times_ms")
                    if (not isinstance(interval_times, list)
                            or len(interval_times) != len(expected_checkpoints) - 1):
                        errors.append(f"bad interval times: {path}")
                    if (not isinstance(cumulative_times, list)
                            or len(cumulative_times) != len(expected_checkpoints)):
                        errors.append(f"bad cumulative times: {path}")
                    if row.get("evaluation_status") != "success":
                        errors.append(f"independent evaluation incomplete: {path}")
                    population_path = row.get("checkpoint_populations")
                    if not population_path or not Path(population_path).exists():
                        errors.append(f"missing population checkpoints: {path}")
    paired = defaultdict(dict)
    for row in records:
        key = (row.get("experiment"), row.get("problem", row.get("environment")),
               row.get("algorithm"), row.get("dimension"), row.get("nominal_population"),
               row.get("seed"), row.get("repeat_index"))
        if row.get("run_status") == "success":
            paired[key][row.get("framework")] = row.get("actual_population")
    for key, values in paired.items():
        if set(values) == {"cuda_moea", "evox"} and len(set(values.values())) != 1:
            errors.append(f"unmatched framework pair {key}: {values}")
        elif set(values) != {"cuda_moea", "evox"}:
            warnings.append(f"speedup/paired quality unavailable {key}: {values}")
    for message in errors[:100]: print("ERROR", message)
    if len(errors) > 100: print(f"... {len(errors)-100} additional errors")
    for message in warnings[:20]: print("WARNING", message)
    if len(warnings) > 20: print(f"... {len(warnings)-20} additional warnings")
    print(f"Validated {len(records)} records; errors={len(errors)} warnings={len(warnings)}")
    return 1 if errors else 0

if __name__ == "__main__": raise SystemExit(main())
