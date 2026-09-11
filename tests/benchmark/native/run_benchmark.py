"""Paired Python/PyTorch vs native CUDA BiSphere benchmark on one visible GPU.

Run from the repository root with CUDA_VISIBLE_DEVICES selecting an idle GPU.
Raw measurements include correctness, warmup, paired samples and provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import torch
import cuda_moea as cm
from cuda_moea._native_build import build_problem, sdk_directory
from cuda_moea.native import benchmark

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples"))
from native_problem import PythonBiSphere


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def native_problem(library, dimension):
    return cm.NativeProblem(library=library, name="BiSphere", dimension=dimension,
                            objectives=2, lower_bounds=-5.0, upper_bounds=5.0,
                            parameters={"offset": 2.0, "radius": 3.0, "constrained": True})


def algorithm(kind, problem, x, generations, seed):
    selection = {"environment_selector": cm.NSGA3EnvironmentSelector(sparse_ratio=1.0)} if kind == "NSGA3" else {}
    return cm.Algorithm(kind, problem=problem, population_size=x.shape[0],
                        max_generations=generations, initial_population=x,
                        seed=seed, print_progress=False, enable_warmup=False, **selection)


def check(problem, x):
    algo = algorithm("NSGA3", problem, x, 1, 42)
    algo.initialize()
    result = algo.population
    # float64 reference distinguishes reduction-rounding error from wrong formulas.
    reference_x = x.double()
    reference_f = torch.stack((reference_x.square().sum(1),
                                (reference_x - 2.0).square().sum(1)), 1)
    reference_cv = (reference_f[:, 0] - 9.0).clamp_min(0)
    torch.testing.assert_close(result.variables, x, rtol=0, atol=0)
    torch.testing.assert_close(result.objectives.double(), reference_f, rtol=2e-5, atol=2e-4)
    torch.testing.assert_close(result.constraints.double(), reference_cv, rtol=2e-5, atol=2e-4)
    errors = {"objective_max_abs_error": (result.objectives.double() - reference_f).abs().max().item(),
              "objective_max_scaled_error": ((result.objectives.double() - reference_f).abs() /
                                               reference_f.abs().clamp_min(1)).max().item(),
              "constraint_max_abs_error": (result.constraints.double() - reference_cv).abs().max().item()}
    del result, algo
    gc.collect()
    return errors


def stats(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--populations", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--dimensions", nargs="+", type=int, default=[20, 128, 1024])
    parser.add_argument("--algorithms", nargs="+", choices=["NSGA3", "RVEA"], default=["NSGA3", "RVEA"])
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("output/native_benchmark"))
    parser.add_argument("--torch-compile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A working CUDA device is required")
    if min(args.populations + args.dimensions + [args.generations, args.samples, args.repeats, args.warmup]) <= 0:
        parser.error("All counts must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "measurements.json").exists():
        parser.error("Output already contains measurements; choose a new directory")
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": vars(args) | {"output": str(output), "nsga3_sparse_ratio": 1.0},
              "environment": {"python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "device_capability": torch.cuda.get_device_capability(), "sdk": json.loads((sdk_directory()/"build.json").read_text()),
                              "nvidia_smi_before": subprocess.check_output(["nvidia-smi"], text=True)},
              "source_sha256": {}, "cases": []}
    sources = [ROOT / "examples/native_problem.py", ROOT / "python/csrc/adapters.cpp",
               Path(__file__), *(ROOT / "examples/native/bi_sphere").glob("*")]
    for path in sources:
        if path.is_file():
            report["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = output / "measurements.json"
    save(raw, report)
    source = ROOT / "examples/native/bi_sphere"
    cache = output / "cache"
    begin = time.perf_counter()
    library = build_problem(source, cache_dir=cache, force=True)
    cold_ms = (time.perf_counter() - begin) * 1000
    hits = []
    for _ in range(7):
        begin = time.perf_counter()
        assert build_problem(source, cache_dir=cache, mode="never") == library
        hits.append((time.perf_counter() - begin) * 1000)
    report["native_build"] = {"cold_wall_ms": cold_ms, "cache_hit_wall_ms": hits,
                              "library": str(library), "architectures": json.loads((library.parent/"artifact.json").read_text())["request"]["architectures"]}
    torch.manual_seed(42)
    # Bring the device out of its idle power state; do not alter clock settings.
    warm = torch.randn((1024, 1024), device="cuda")
    for _ in range(100):
        warm = warm @ warm.T * 0.001
    torch.cuda.synchronize()
    del warm
    for n in args.populations:
        for d in args.dimensions:
            torch.manual_seed(42000 + n + d)
            x = torch.rand((n, d), device="cuda") * 10 - 5
            # Include feasible, infeasible and exactly-on-boundary points.
            x[0].zero_()
            x[1].zero_(); x[1, 0] = 3.0
            problems = {"pytorch": PythonBiSphere(d), "native": native_problem(library, d)}
            if args.torch_compile:
                p = PythonBiSphere(d)
                p.evaluate = torch.compile(p.evaluate)
                problems["torch_compile"] = p
            correctness = {name: check(p, x) for name, p in problems.items()}
            evaluations = {name: [] for name in problems}
            for sample in range(args.samples):
                order = list(problems)
                random.Random(sample + n + d).shuffle(order)
                for name in order:
                    evaluations[name].append(benchmark(problems[name], x, repeats=args.repeats, warmup=args.warmup))
            for kind in args.algorithms:
                # Warm the algorithm's kernels/workspaces before timed fresh runs.
                for p in problems.values():
                    algo = algorithm(kind, p, x, 5, 42)
                    result = algo.run(copy=False)
                    del result, algo
                samples = {name: [] for name in problems}
                for sample in range(args.samples):
                    order = list(problems)
                    random.Random(sample + n + d).shuffle(order)
                    for name in order:
                        algo = algorithm(kind, problems[name], x, args.generations, 1000 + sample)
                        algo.initialize()
                        algo.synchronize()
                        begin = time.perf_counter()
                        result = algo.run(copy=False)
                        wall_ms = (time.perf_counter() - begin) * 1000
                        samples[name].append({"wall_ms": wall_ms, "cuda_ms": result.total_ms,
                                              "active_count": result.active_count, "seed": 1000 + sample})
                        assert result.active_count > 0
                        assert torch.isfinite(result.objectives[:result.active_count]).all().item()
                        del result, algo
                evaluation_stats = {name: stats([v["wall_ms_per_eval"] for v in values]) for name, values in evaluations.items()}
                run_stats = {name: stats([v["wall_ms"] for v in values]) for name, values in samples.items()}
                row = {"population": n, "dimension": d, "algorithm": kind, "correctness": correctness,
                       "evaluation_samples": evaluations, "run_samples": samples,
                       "evaluation_wall_ms": evaluation_stats, "run_wall_ms": run_stats,
                       "evaluation_speedup": evaluation_stats["pytorch"]["median"] / evaluation_stats["native"]["median"],
                       "run_speedup": run_stats["pytorch"]["median"] / run_stats["native"]["median"]}
                report["cases"].append(row)
                save(raw, report)
                print(f"N={n} D={d} {kind}: evaluation {row['evaluation_speedup']:.2f}x; full run {row['run_speedup']:.2f}x", flush=True)
            del x, problems
            gc.collect()
    report["status"] = "complete"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["environment"]["nvidia_smi_after"] = subprocess.check_output(["nvidia-smi"], text=True)
    save(raw, report)
    print(raw, flush=True)


if __name__ == "__main__":
    main()
