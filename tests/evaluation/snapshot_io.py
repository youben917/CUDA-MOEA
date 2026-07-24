"""Load CUDA MOEA metadata and binary population snapshots."""

from __future__ import annotations

from array import array
import json
import re
from pathlib import Path
from typing import Any


GENERATION_RE = re.compile(r"generation_(\d+)$")


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for result evaluation") from error
    return torch


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(run_dir: Path) -> dict:
    return load_json(run_dir / "metadata.json")


def generation_number(path: Path) -> int:
    match = GENERATION_RE.search(path.name)
    if not match:
        raise ValueError(f"Not a generation directory: {path}")
    return int(match.group(1))


def list_generations(run_dir: Path) -> list[Path]:
    paths = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and GENERATION_RE.search(path.name)
    ]
    return sorted(paths, key=generation_number)


def resolve_generation(run_dir: Path, generation: str | int = "latest") -> Path:
    generations = list_generations(run_dir)
    if not generations:
        raise FileNotFoundError(f"No generation_* directories under {run_dir}")
    if str(generation).lower() == "latest":
        generation = int(load_metadata(run_dir)["max_generations"])
    target = f"generation_{int(generation):06d}"
    for path in generations:
        if path.name == target:
            return path
    raise FileNotFoundError(f"Could not find {target} under {run_dir}")


def read_float32(path: Path) -> array:
    values = array("f")
    if values.itemsize != 4:
        raise RuntimeError("This platform does not expose array('f') as float32")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    return values


def load_objectives(run_dir: Path, generation: str | int = "latest") -> Any:
    torch = require_torch()
    metadata = load_metadata(run_dir)
    generation_dir = resolve_generation(run_dir, generation)
    snapshot = load_json(generation_dir / "snapshot.json")
    objectives = int(snapshot.get("objective_count", metadata["objective_count"]))
    population = int(snapshot.get("population_size", metadata["population_size"]))
    values = read_float32(generation_dir / "objectives.bin")
    if len(values) != objectives * population:
        raise ValueError(
            f"objectives.bin has {len(values)} values; expected "
            f"{objectives * population}"
        )
    return torch.tensor(values, dtype=torch.float32).reshape(
        objectives, population
    ).T.contiguous()


def load_constraints(run_dir: Path, generation: str | int = "latest") -> Any | None:
    torch = require_torch()
    path = resolve_generation(run_dir, generation) / "constraints.bin"
    return torch.tensor(read_float32(path), dtype=torch.float32) if path.exists() else None


def feasible_objectives(
    run_dir: Path,
    generation: str | int = "latest",
    feasibility_epsilon: float = 0.0,
) -> Any:
    torch = require_torch()
    objectives = load_objectives(run_dir, generation)
    constraints = load_constraints(run_dir, generation)
    finite = torch.isfinite(objectives).all(dim=1)
    if constraints is None or constraints.numel() == 0:
        return objectives[finite]
    if constraints.shape[0] != objectives.shape[0]:
        raise ValueError("constraint and objective population sizes differ")
    return objectives[finite & (constraints <= feasibility_epsilon)]


def default_torch_device() -> str:
    return "cuda" if require_torch().cuda.is_available() else "cpu"
