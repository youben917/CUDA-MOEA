"""Expected best linear utility over preference weights."""

from typing import Any

from ._common import objective_vector, point_matrix, require_torch


def preference_weights(
    count: int,
    objectives: int,
    *,
    seed: int = 2887,
    device: str | Any | None = None,
    dtype: Any | None = None,
) -> Any:
    """Sample reproducible Dirichlet(1) simplex weights."""

    torch = require_torch()
    if count <= 0 or objectives <= 0:
        raise ValueError("count and objectives must be positive")
    device = torch.device("cpu" if device is None else device)
    dtype = torch.float32 if dtype is None else dtype
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    uniform = torch.rand(
        (count, objectives), generator=generator, device=device, dtype=dtype
    )
    exponential = -torch.log(uniform.clamp_min(torch.finfo(dtype).tiny))
    return exponential / exponential.sum(dim=1, keepdim=True)


def expected_utility(
    points: Any,
    weights: Any | None = None,
    *,
    maximize: bool = False,
    ideal_point: Any | None = None,
    nadir_point: Any | None = None,
    weight_samples: int = 4096,
    seed: int = 2887,
    device: str | Any | None = None,
    chunk_size: int = 512,
) -> float:
    """Return the expected best utility supplied by an approximation set."""

    torch = require_torch()
    values = point_matrix(points, device)
    values = values[torch.isfinite(values).all(dim=1)]
    if not values.shape[0]:
        raise ValueError("points contain no finite objective rows")
    normalized = ideal_point is not None or nadir_point is not None
    if normalized and (ideal_point is None or nadir_point is None):
        raise ValueError("ideal_point and nadir_point must be supplied together")
    if normalized:
        ideal = objective_vector(ideal_point, values, "ideal_point")
        nadir = objective_vector(nadir_point, values, "nadir_point")
        scale = ideal - nadir if maximize else nadir - ideal
        if (scale <= 0).any():
            relation = "smaller" if maximize else "greater"
            raise ValueError(f"nadir_point must be {relation} than ideal_point")
        utility_values = (
            (values - nadir) / scale
            if maximize
            else 1.0 - (values - ideal) / scale
        )
    else:
        utility_values = values if maximize else -values

    if weights is None:
        preference = preference_weights(
            weight_samples,
            values.shape[1],
            seed=seed,
            device=values.device,
            dtype=values.dtype,
        )
    else:
        preference = torch.as_tensor(weights, dtype=values.dtype, device=values.device)
        if preference.ndim == 1:
            preference = preference[None, :]
        if preference.ndim != 2 or preference.shape[1] != values.shape[1]:
            raise ValueError("weights must have shape (weight_count, objectives)")
        if not torch.isfinite(preference).all() or (preference < 0).any():
            raise ValueError("weights must be finite and nonnegative")
        totals = preference.sum(dim=1, keepdim=True)
        if (totals <= 0).any():
            raise ValueError("each weight vector must have a positive sum")
        preference = preference / totals

    best = []
    for start in range(0, preference.shape[0], chunk_size):
        utilities = utility_values @ preference[start : start + chunk_size].T
        best.append(utilities.amax(dim=0))
    return torch.cat(best).mean().item()
