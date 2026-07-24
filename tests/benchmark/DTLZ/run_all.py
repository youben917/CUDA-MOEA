"""Compatibility dispatcher; prefer the three explicit group scripts."""

from __future__ import annotations
import subprocess
from pathlib import Path
import sys

if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    for name in ("run_population_timing.py", "run_dimension_timing.py", "run_quality.py"):
        code = subprocess.run([sys.executable, str(directory / name), *sys.argv[1:]]).returncode
        if code:
            raise SystemExit(code)
