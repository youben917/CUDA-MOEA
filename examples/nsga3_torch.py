import torch

import cuda_moea as cm


algorithm = cm.NSGA3(
    population_size=16384,
    max_generations=4000,
    problem=cm.DTLZ1(dimension=500, objectives=3),
    environment_selector=cm.NSGA3EnvironmentSelector(sparse_ratio=1.0),
    seed=2887,
    device="cuda:1",
    progress_interval=1000,
    save_data="output/nsga3_dtlz1",
    save_interval=1000
)
result = algorithm.run()
print(f"time={result.total_ms} ms")
