"""Shared serial, resumable launcher for DTLZ groups."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.benchmark.common import freeze_environment, successful, terminal_record, write_manifest


def launch(jobs, description: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--rerun-successful", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    environment_manifest = freeze_environment(args.output, args.device)
    write_manifest(args.output / f"{jobs[0].experiment}_manifest.jsonl", jobs)
    runner = Path(__file__).with_name("run_job.py")
    missing = []
    for job in jobs:
        path = args.output / job.relative_directory / "run_result.json"
        expected = {"framework": job.framework, "algorithm": job.algorithm,
                    "generations": job.generations,
                    "nominal_population": job.nominal_population}
        if not successful(path, expected):
            missing.append(job)
    print(f"Planned={len(jobs)} successful={len(jobs)-len(missing)} pending_or_failed={len(missing)}")
    if args.validate_only:
        absent = []
        for job in jobs:
            expected = {"framework": job.framework, "algorithm": job.algorithm,
                        "generations": job.generations,
                        "nominal_population": job.nominal_population}
            if not terminal_record(args.output / job.relative_directory / "run_result.json", expected):
                absent.append(job)
        for job in absent[:20]:
            print("MISSING", args.output / job.relative_directory / "run_result.json")
        return 0 if not absent else 1

    failures = handled = 0
    for index, job in enumerate(jobs, 1):
        output = args.output / job.relative_directory
        expected = {"framework": job.framework, "algorithm": job.algorithm,
                    "generations": job.generations,
                    "nominal_population": job.nominal_population}
        if not args.rerun_successful and successful(output / "run_result.json", expected):
            continue
        command = [sys.executable, str(runner), "--framework", job.framework,
                   "--algorithm", job.algorithm, "--experiment", job.experiment,
                   "--problem", job.problem, "--objectives", str(job.objectives),
                   "--dimension", str(job.dimension), "--nominal-population", str(job.nominal_population),
                   "--generations", str(job.generations), "--seed", str(job.seed),
                   "--repeat-index", str(job.repeat_index), "--device", args.device,
                   "--output", str(output), "--environment-manifest", str(environment_manifest)]
        if job.save_objectives:
            command.append("--save-objectives")
        print(f"[{index}/{len(jobs)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            code = subprocess.run(command).returncode
            failures += int(code != 0)
            if code and args.fail_fast:
                return code
        handled += 1
        if args.stop_after and handled >= args.stop_after:
            break
    return 1 if failures else 0
