"""Shared validation helpers for quality indicators."""

from typing import Any


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for evaluation metrics") from error
    return torch


def point_matrix(points: Any, device: str | Any | None = None) -> Any:
    torch = require_torch()
    result = torch.as_tensor(points, device=device)
    if not result.is_floating_point():
        result = result.to(torch.float64)
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError("points must have shape (point_count, objective_count)")
    return result


def objective_vector(values: Any, points: Any, name: str) -> Any:
    torch = require_torch()
    result = torch.as_tensor(values, dtype=points.dtype, device=points.device)
    if result.ndim != 1 or result.shape[0] != points.shape[1]:
        raise ValueError(f"{name} must contain one value per objective")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result
