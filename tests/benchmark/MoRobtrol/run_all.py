"""Compatibility dispatcher for the MoRobtrol D group."""
import subprocess, sys
from pathlib import Path
if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    for name in ("run_quality.py",):
        code = subprocess.run([sys.executable, str(directory / name), *sys.argv[1:]]).returncode
        if code: raise SystemExit(code)
