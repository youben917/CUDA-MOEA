"""Deterministic inputs shared by CUDA-MOEA and EvoX benchmark runs."""

from __future__ import annotations

import torch


INITIAL_POPULATION_PROTOCOL = "torch_cpu_uniform_v1"
NSGA3_REFERENCE_PROTOCOL = "evox_uniform_sampling_v1"


def initial_population(
    population: int,
    dimension: int,
    seed: int,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Return a CPU float32 population independent of either framework RNG."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.empty(
        (population, dimension), dtype=torch.float32, device="cpu").uniform_(
        lower, upper, generator=generator)


def nsga3_reference_directions(
    population: int,
    objectives: int,
) -> torch.Tensor:
    """Return EvoX's Das--Dennis direction set in its canonical row-major layout."""
    from evox.operators.sampling import uniform_sampling

    # EvoX follows PyTorch's global default device.  Force generation on CPU
    # so the two child processes produce bit-identical direction values even
    # though the EvoX runner temporarily sets its default to CUDA.
    previous_device = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        directions, _ = uniform_sampling(population, objectives)
    finally:
        torch.set_default_device(previous_device)
    return directions.to(dtype=torch.float32, device="cpu").contiguous()
