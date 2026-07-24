"""Serial, resumable launcher for MoRobtrol D-group jobs."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.benchmark.common import freeze_environment, successful, terminal_record, write_manifest


def _expected(job) -> dict:
    return {
        "framework": job.framework,
        "algorithm": job.algorithm,
        "generations": job.generations,
        "nominal_population": job.nominal_population,
        "hidden_width": job.hidden_width,
        "checkpoint_interval": job.checkpoint_interval,
    }


def _optimization_successful(root: Path, job) -> bool:
    result_path = root / job.relative_directory / "run_result.json"
    if not successful(result_path, _expected(job)):
        return False
    if job.checkpoints:
        return (result_path.parent / "checkpoint_populations.pt").is_file()
    return True


def _optimization_terminal(root: Path, job) -> bool:
    result_path = root / job.relative_directory / "run_result.json"
    if not terminal_record(result_path, _expected(job)):
        return False
    try:
        status = json.loads(result_path.read_text(encoding="utf-8")).get("run_status")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return status != "success" or not job.checkpoints or (result_path.parent / "checkpoint_populations.pt").is_file()


def launch(jobs, description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--environment", nargs="+")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--rerun-successful", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--stop-after", type=int, default=0)
    args = p.parse_args()
    manifest_jobs = jobs
    if args.environment:
        jobs = [j for j in jobs if j.environment in args.environment]
    if not jobs:
        p.error("no jobs selected; check --environment")
    environment_manifest = freeze_environment(args.output, args.device)
    write_manifest(args.output / f"{jobs[0].experiment}_manifest.jsonl", manifest_jobs)
    missing = [
        job for job in jobs
        if not _optimization_successful(args.output, job)
    ]
    print(f"Planned={len(jobs)} successful={len(jobs)-len(missing)} pending_or_failed={len(missing)}")
    if args.validate_only:
        absent = []
        for job in jobs:
            if not _optimization_terminal(args.output, job):
                absent.append(job)
        return 0 if not absent else 1
    runner = Path(__file__).with_name("run_job.py")
    failures = handled = 0
    for index, job in enumerate(jobs, 1):
        output = args.output / job.relative_directory
        if not args.rerun_successful and _optimization_successful(args.output, job): continue
        command = [sys.executable, str(runner), "--framework", job.framework,
                   "--algorithm", job.algorithm, "--experiment", job.experiment,
                   "--environment", job.environment, "--objectives", str(job.objectives),
                   "--nominal-population", str(job.nominal_population),
                   "--generations", str(job.generations), "--seed", str(job.seed),
                   "--repeat-index", str(job.repeat_index), "--device", args.device,
                   "--output", str(output), "--environment-manifest", str(environment_manifest),
                   "--hidden-width", str(job.hidden_width),
                   "--checkpoint-interval", str(job.checkpoint_interval)]
        if job.checkpoints:
            command.append("--checkpoints")
        print(f"[{index}/{len(jobs)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            started = time.perf_counter()
            code = subprocess.run(command).returncode
            elapsed = time.perf_counter() - started
            print(f"[{index}/{len(jobs)}] exit={code} elapsed={elapsed:.1f}s", flush=True)
            failures += int(code != 0)
            if code and args.fail_fast: return code
        handled += 1
        if args.stop_after and handled >= args.stop_after: break
    return 1 if failures else 0
