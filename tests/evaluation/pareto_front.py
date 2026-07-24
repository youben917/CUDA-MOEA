"""Analytic Pareto-front samplers for CUDA MOEA's DTLZ problems."""

from __future__ import annotations

from typing import Any

try:
    from .snapshot_io import require_torch
except ImportError:  # Direct script-directory import.
    from snapshot_io import require_torch


def canonical_problem_name(problem_name: str) -> str:
    return "".join(character for character in problem_name.lower() if character.isalnum())


def _simplex_points(count, dimensions, scale, device, dtype, seed):
    torch = require_torch()
    generator = torch.Generator(device=device).manual_seed(seed)
    uniform = torch.rand(
        (count, dimensions), generator=generator, device=device, dtype=dtype
    )
    exponential = -torch.log(uniform.clamp_min(torch.finfo(dtype).tiny))
    return scale * exponential / exponential.sum(dim=1, keepdim=True)


def _sphere_points(count, dimensions, device, dtype, seed):
    torch = require_torch()
    generator = torch.Generator(device=device).manual_seed(seed)
    points = torch.rand(
        (count, dimensions), generator=generator, device=device, dtype=dtype
    )
    return points / points.norm(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)


def _dtlz2_objectives_from_angles(theta):
    torch = require_torch()
    count, angle_count = theta.shape
    dimensions = angle_count + 1
    points = torch.ones((count, dimensions), device=theta.device, dtype=theta.dtype)
    for objective in range(dimensions):
        cosine_count = dimensions - 1 - objective
        if cosine_count:
            points[:, objective] *= torch.cos(theta[:, :cosine_count]).prod(dim=1)
        if objective:
            points[:, objective] *= torch.sin(theta[:, dimensions - 1 - objective])
    return points


def _dtlz5_dtlz6_points(count, dimensions, device, dtype, seed):
    torch = require_torch()
    generator = torch.Generator(device=device).manual_seed(seed)
    theta = torch.full(
        (count, dimensions - 1), torch.pi / 4.0, device=device, dtype=dtype
    )
    theta[:, 0] = torch.rand(
        count, generator=generator, device=device, dtype=dtype
    ) * (torch.pi / 2.0)
    return _dtlz2_objectives_from_angles(theta)


def _dtlz7_points(count, dimensions, device, dtype, seed):
    torch = require_torch()
    intervals = torch.tensor(
        ((0.0, 0.251411836), (0.631626531, 0.859400856)),
        device=device,
        dtype=dtype,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    regions = torch.randint(
        0, 2, (count, dimensions - 1), generator=generator, device=device
    )
    unit = torch.rand(
        (count, dimensions - 1), generator=generator, device=device, dtype=dtype
    )
    free = intervals[regions, 0] + unit * (intervals[regions, 1] - intervals[regions, 0])
    last = 2.0 * dimensions - (
        free * (1.0 + torch.sin(3.0 * torch.pi * free))
    ).sum(dim=1, keepdim=True)
    return torch.cat((free, last), dim=1)


def _convex_dtlz2_points(points):
    result = points.clone()
    result[:, :-1] = result[:, :-1].pow(4)
    result[:, -1] = result[:, -1].pow(2)
    return result


def _c2_dtlz2_radius(objective_count):
    radii = {2: 0.15, 3: 0.40, 5: 1.0, 8: 1.0, 10: 1.0, 15: 1.0}
    if objective_count not in radii:
        raise ValueError(f"C2DTLZ2 supports objective counts {sorted(radii)}")
    return radii[objective_count]


def _c2_convex_dtlz2_radius(objective_count):
    radii = {3: 0.20, 5: 0.225, 8: 0.26, 10: 0.26, 15: 0.27}
    if objective_count not in radii:
        raise ValueError(f"C2ConvexDTLZ2 supports objective counts {sorted(radii)}")
    return radii[objective_count]


def _validate_c1_dtlz3_objective_count(objective_count):
    valid = {2, 3, 5, 8, 10, 15}
    if objective_count not in valid:
        raise ValueError(f"C1DTLZ3 supports objective counts {sorted(valid)}")


def _c2_dtlz2_feasible(points, radius):
    torch = require_torch()
    dimensions = points.shape[1]
    axes = torch.eye(dimensions, device=points.device, dtype=points.dtype)
    axis_distance = ((points[:, None, :] - axes[None, :, :]) ** 2).sum(dim=2)
    center = torch.full(
        (dimensions,), 1.0 / dimensions**0.5, device=points.device, dtype=points.dtype
    )
    center_distance = ((points - center) ** 2).sum(dim=1)
    return torch.minimum(axis_distance.amin(dim=1), center_distance) <= radius**2


def _c2_convex_dtlz2_feasible(points, radius):
    centered = points - points.mean(dim=1, keepdim=True)
    return centered.square().sum(dim=1) >= radius**2


def _sample_filtered_front(sampler, predicate, count, seed, problem_name):
    torch = require_torch()
    accepted = []
    accepted_count = 0
    for attempt in range(64):
        candidates = sampler(max(1024, 2 * (count - accepted_count)), seed + attempt)
        feasible = candidates[predicate(candidates)]
        if feasible.numel():
            accepted.append(feasible)
            accepted_count += feasible.shape[0]
        if accepted_count >= count:
            return torch.cat(accepted, dim=0)[:count]
    raise RuntimeError(f"Could not sample {count} feasible points for {problem_name}")


def pareto_front_points(
    problem_name: str,
    objective_count: int,
    count: int = 20000,
    device: str | Any = "cpu",
    dtype: Any | None = None,
    seed: int = 2887,
) -> Any:
    """Generate deterministic analytic Pareto-front samples."""

    torch = require_torch()
    device = torch.device(device)
    dtype = torch.float32 if dtype is None else dtype
    if objective_count < 2:
        raise ValueError("DTLZ problems require at least two objectives")
    if count <= 0:
        raise ValueError("The number of Pareto-front points must be positive")
    problem = canonical_problem_name(problem_name)
    simplex = lambda n, s: _simplex_points(n, objective_count, s, device, dtype, seed)
    sphere = lambda n, s: _sphere_points(n, objective_count, device, dtype, s)

    if problem in {"dtlz1", "c1dtlz1"}:
        return simplex(count, 0.5)
    if problem == "c1dtlz3":
        _validate_c1_dtlz3_objective_count(objective_count)
        return sphere(count, seed)
    if problem in {"dtlz2", "dtlz3", "dtlz4"}:
        return sphere(count, seed)
    if problem in {"dtlz5", "dtlz6"}:
        return _dtlz5_dtlz6_points(count, objective_count, device, dtype, seed)
    if problem == "dtlz7":
        return _dtlz7_points(count, objective_count, device, dtype, seed)
    if problem == "convexdtlz2":
        return _convex_dtlz2_points(sphere(count, seed))
    if problem == "c2dtlz2":
        radius = _c2_dtlz2_radius(objective_count)
        return _sample_filtered_front(
            sphere, lambda points: _c2_dtlz2_feasible(points, radius), count, seed, problem_name
        )
    if problem == "c2convexdtlz2":
        radius = _c2_convex_dtlz2_radius(objective_count)
        sampler = lambda n, s: _convex_dtlz2_points(sphere(n, s))
        return _sample_filtered_front(
            sampler,
            lambda points: _c2_convex_dtlz2_feasible(points, radius),
            count,
            seed,
            problem_name,
        )
    if problem == "c3dtlz1":
        directions = simplex(count, 1.0)
        return directions * (1.0 / (2.0 - directions.amax(dim=1)))[:, None]
    if problem == "c3dtlz4":
        directions = sphere(count, seed)
        radius = (1.0 - 0.75 * directions.square().amax(dim=1)).rsqrt()
        return directions * radius[:, None]
    raise NotImplementedError(f"No analytic Pareto front for {problem_name!r}")
