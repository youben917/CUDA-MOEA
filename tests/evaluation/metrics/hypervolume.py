"""Exact and Sobol-estimated dominated Hypervolume (HV)."""

from typing import Any

from ._common import objective_vector, point_matrix, require_torch


def _exact(points: Any, reference: Any) -> float:
    torch = require_torch()
    dimensions = points.shape[1]
    if not points.shape[0]:
        return 0.0
    if dimensions == 1:
        return (reference[0] - points[:, 0].amin()).clamp_min(0).item()
    if dimensions == 2:
        ordered = points[torch.argsort(points[:, 0])]
        heights = reference[1] - torch.cummin(ordered[:, 1], dim=0).values
        next_x = torch.cat((ordered[1:, 0], reference[0:1]))
        return ((next_x - ordered[:, 0]).clamp_min(0) * heights.clamp_min(0)).sum().item()

    coordinates = points[:, 0].unique(sorted=True)
    volume = 0.0
    for index, coordinate in enumerate(coordinates):
        next_coordinate = (
            coordinates[index + 1]
            if index + 1 < coordinates.numel()
            else reference[0]
        )
        width = (next_coordinate - coordinate).item()
        if width > 0:
            volume += width * _exact(
                points[points[:, 0] <= coordinate, 1:], reference[1:]
            )
    return volume


def _sobol(
    points: Any,
    reference: Any,
    samples: int,
    seed: int,
    sample_chunk_size: int,
    point_chunk_size: int,
) -> float:
    torch = require_torch()
    if samples <= 0:
        raise ValueError("samples must be positive")
    lower = points.amin(dim=0)
    sides = reference - lower
    box_volume = sides.prod().item()
    if box_volume <= 0:
        return 0.0
    engine = torch.quasirandom.SobolEngine(points.shape[1], scramble=True, seed=seed)
    dominated_count = 0
    remaining = samples
    while remaining:
        count = min(sample_chunk_size, remaining)
        unit = engine.draw(count).to(device=points.device, dtype=points.dtype)
        samples_chunk = lower + unit * sides
        dominated = torch.zeros(count, dtype=torch.bool, device=points.device)
        for start in range(0, points.shape[0], point_chunk_size):
            candidates = points[start : start + point_chunk_size]
            dominated |= (
                candidates[:, None, :] <= samples_chunk[None, :, :]
            ).all(dim=2).any(dim=0)
            if dominated.all():
                break
        dominated_count += int(dominated.sum().item())
        remaining -= count
    return box_volume * dominated_count / samples


def hypervolume(
    points: Any,
    reference_point: Any,
    *,
    maximize: bool = False,
    method: str = "auto",
    samples: int = 16384,
    seed: int = 2887,
    device: str | Any | None = None,
    sample_chunk_size: int = 2048,
    point_chunk_size: int = 512,
) -> float:
    """Compute dominated HV relative to a fixed worse reference point."""

    torch = require_torch()
    values = point_matrix(points, device)
    reference = objective_vector(reference_point, values, "reference_point")
    values = values[torch.isfinite(values).all(dim=1)]
    if maximize:
        values, reference = -values, -reference
    values = values[(values <= reference).all(dim=1)]
    if not values.shape[0]:
        return 0.0
    method = method.lower().replace("-", "_")
    if method == "auto":
        method = "exact" if values.shape[1] <= 2 else "monte_carlo"
    if method == "exact":
        return _exact(values, reference)
    if method in {"monte_carlo", "sobol"}:
        return _sobol(
            values, reference, samples, seed, sample_chunk_size, point_chunk_size
        )
    raise ValueError("method must be 'auto', 'exact', or 'monte_carlo'")
