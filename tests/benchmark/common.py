"""Shared planning and result I/O for the A--D benchmark groups."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback


FRAMEWORKS = ("cuda_moea", "evox")
ALGORITHMS = ("nsga3", "rvea")
RVEA_EFFECTIVE_M3 = {
    128: 120, 256: 253, 512: 496, 1024: 990, 2048: 2016,
    4096: 4095, 8192: 8128, 16384: 16290, 32768: 32640,
}


def effective_population(algorithm: str, objectives: int, nominal: int) -> int:
    if algorithm == "rvea" and objectives == 3:
        try:
            return RVEA_EFFECTIVE_M3[nominal]
        except KeyError as exc:
            raise ValueError(f"No frozen M=3 RVEA size for nominal N={nominal}") from exc
    return nominal


def requested_population(framework: str, algorithm: str, objectives: int, nominal: int) -> int:
    """EvoX RVEA receives nominal N; CUDA-MOEA receives its effective N."""
    if framework == "cuda_moea" and algorithm == "rvea":
        return effective_population(algorithm, objectives, nominal)
    return nominal


def rotated_implementations(index: int) -> tuple[tuple[str, str], ...]:
    implementations = tuple((f, a) for a in ALGORITHMS for f in FRAMEWORKS)
    offset = index % len(implementations)
    return implementations[offset:] + implementations[:offset]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def successful(path: Path, expected: dict | None = None) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("run_status") != "success":
            return False
        return expected is None or all(record.get(key) == value for key, value in expected.items())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def terminal_record(path: Path, expected: dict | None = None) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("run_status") not in {"success", "oom", "nan", "timeout", "failed"}:
            return False
        return expected is None or all(record.get(key) == value for key, value in expected.items())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def failure_record(base: dict, error: BaseException) -> dict:
    message = str(error)
    lowered = message.lower()
    if "out of memory" in lowered or "cuda error: memory" in lowered:
        status = "oom"
    elif "nan" in lowered or "non-finite" in lowered or "nonfinite" in lowered:
        status = "nan"
    elif isinstance(error, TimeoutError):
        status = "timeout"
    else:
        status = "failed"
    return {
        **base, "run_status": status, "error_type": type(error).__name__,
        "error": message, "traceback": traceback.format_exc(),
    }


def _capture(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def freeze_environment(root: Path, device: str) -> Path:
    path = root / "manifest.json"
    if path.exists():
        return path
    packages = {}
    for name in ("cuda-moea", "evox", "evomo", "torch", "jax", "brax", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = {
        "python": sys.version, "platform": platform.platform(), "device": device,
        "packages": packages, "git_commit": _capture(["git", "rev-parse", "HEAD"]),
        "git_status": _capture(["git", "status", "--short"]),
        "nvidia_smi": _capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "nvcc": _capture(["nvcc", "--version"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    atomic_json(path, record)
    return path


def write_manifest(path: Path, jobs: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(job)) + "\n" for job in jobs), encoding="utf-8")
