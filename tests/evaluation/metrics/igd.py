"""Inverted Generational Distance (IGD)."""

from typing import Any

from ._common import require_torch


def nondominated_mask(points: Any, chunk_size: int = 2048) -> Any:
    """Return a minimization nondominated mask."""

    torch = require_torch()
    dominated = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for start in range(0, points.shape[0], chunk_size):
        block = points[start : start + chunk_size]
        no_worse = points[:, None, :] <= block[None, :, :]
        better = points[:, None, :] < block[None, :, :]
        dominated[start : start + chunk_size] = (
            no_worse.all(dim=2) & better.any(dim=2)
        ).any(dim=0)
    return ~dominated


def igd(
    approximated_front: Any,
    reference_front: Any,
    device: str | Any = "cpu",
    chunk_size: int = 4096,
    filter_nondominated: bool = True,
) -> float:
    """Compute mean reference-to-approximation distance."""

    torch = require_torch()
    device = torch.device(device)
    approximation = torch.as_tensor(
        approximated_front, dtype=reference_front.dtype, device=device
    )
    reference = reference_front.to(device)
    approximation = approximation[torch.isfinite(approximation).all(dim=1)]
    if approximation.numel() == 0:
        raise ValueError("approximation contains no finite objective rows")
    if filter_nondominated:
        approximation = approximation[nondominated_mask(approximation)]
    distances = []
    for start in range(0, reference.shape[0], chunk_size):
        chunk = reference[start : start + chunk_size]
        distances.append(torch.cdist(chunk, approximation).amin(dim=1))
    return torch.cat(distances).mean().item()
