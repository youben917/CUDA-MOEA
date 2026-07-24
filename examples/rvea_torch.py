import torch

import cuda_moea as cm


algorithm = cm.Algorithm(
    algorithm="RVEA",
    population_size=16384,
    max_generations=4001,
    problem=cm.CSDP(dimension=500, objectives=3, constraint_activation_ratio=0.5),
    seed=2887,
    device="cuda:1",
    progress_interval=1000,
    save_data="output/rvea_csdp",
    save_interval=1000
)
result = algorithm.run()
print(f"time={result.total_ms} ms")
