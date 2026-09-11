"""Compare batch evaluation and complete runs; no fixed performance claims."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import cuda_moea as cm
from cuda_moea.native import benchmark
from native_problem import PythonBiSphere


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--torch-compile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A working CUDA device is required")
    start = time.perf_counter()
    native = cm.NativeProblem(Path(__file__).parent / "native/bi_sphere", name="BiSphere",
                              dimension=args.dimension, objectives=2,
                              lower_bounds=-5, upper_bounds=5,
                              parameters={"constrained": True})
    setup_ms = (time.perf_counter() - start) * 1000
    problems = {"pytorch": PythonBiSphere(args.dimension), "native": native}
    if args.torch_compile:
        compiled = PythonBiSphere(args.dimension)
        compiled.evaluate = torch.compile(compiled.evaluate)
        problems["torch_compile"] = compiled
    torch.manual_seed(42)
    x = torch.rand((args.population, args.dimension), device="cuda") * 10 - 5
    report = {"native_setup_ms_including_cache_or_build": setup_ms,
              "library": str(native.library), "measurements": {}}
    for name, problem in problems.items():
        evaluation = benchmark(problem, x, repeats=args.repeats, warmup=10)
        runs = []
        for _ in range(args.runs):
            algo = cm.NSGA3(problem=problem, population_size=args.population,
                           max_generations=args.generations, initial_population=x,
                           seed=42, print_progress=False, enable_warmup=False)
            algo.initialize()
            algo.synchronize()
            start = time.perf_counter()
            result = algo.run(copy=False)
            runs.append((time.perf_counter() - start) * 1000)
            del result, algo
        report["measurements"][name] = {**evaluation, "run_wall_ms_median": statistics.median(runs),
                                         "run_wall_ms_samples": runs}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
