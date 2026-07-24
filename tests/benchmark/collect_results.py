"""Collect run_result.json files only; never calculate metrics or draw figures."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", nargs="+",
                   default=("group_a", "group_b", "group_c", "group_d"),
                   choices=("group_a", "group_b", "group_c", "group_d"))
    args = p.parse_args()
    paths = []
    planned_populations = {}
    for group in args.groups:
        manifest = args.runs_root / f"{group}_manifest.jsonl"
        if manifest.exists():
            jobs = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line
            ]
            planned_populations[group] = {job.get("nominal_population") for job in jobs}
        paths.extend(sorted((args.runs_root / group).glob("**/run_result.json")))
    rows = []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("experiment") == "group_d"
                and planned_populations.get("group_d")
                and row.get("nominal_population") not in planned_populations["group_d"]):
            continue
        row["run_directory"] = str(path.parent.resolve())
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No selected group run_result.json below {args.runs_root}")
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw_runs.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (args.output / "raw_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    counts = {}
    for row in rows: counts[row.get("run_status", "missing")] = counts.get(row.get("run_status", "missing"), 0) + 1
    print(f"Collected {len(rows)} records: {counts}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
