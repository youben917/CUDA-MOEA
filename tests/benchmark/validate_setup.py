"""Fast preflight checks that do not execute optimization experiments."""

from __future__ import annotations
import importlib.metadata
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.benchmark.common import RVEA_EFFECTIVE_M3


def main():
    import cuda_moea as cm
    import evox
    from evox.algorithms import NSGA3, RVEA
    from evox.operators.sampling import uniform_sampling
    from evox.operators.crossover.sbx import simulated_binary
    from evox.operators.mutation.pm_mutation import polynomial_mutation
    if importlib.metadata.version("evox") != "1.3.0":
        raise RuntimeError("Formal benchmark requires EvoX 1.3.0")
    for nominal, expected in RVEA_EFFECTIVE_M3.items():
        actual = int(uniform_sampling(nominal, 3)[1])
        if actual != expected: raise RuntimeError(f"uniform_sampling({nominal},3)={actual}, expected {expected}")
    assert "selection_op" in inspect.signature(NSGA3).parameters
    assert "max_gen" in inspect.signature(RVEA).parameters
    assert inspect.signature(simulated_binary).parameters["dis_c"].default == 20.0
    assert inspect.signature(polynomial_mutation).parameters["dis_m"].default == 20
    for name in ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7", "ConvexDTLZ2"):
        if not hasattr(cm, name): raise RuntimeError(f"CUDA-MOEA lacks {name}")
    print("Preflight OK: EvoX 1.3.0, operators, DTLZ problems, and frozen RVEA sizes")
    return 0

if __name__ == "__main__": raise SystemExit(main())
