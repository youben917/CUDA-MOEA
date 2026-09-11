"""Run the same constrained problem through Python or native CUDA."""
from pathlib import Path

import torch
import cuda_moea as cm


class PythonBiSphere(cm.PythonProblem):
    def __init__(self, dimension=20, offset=2.0, radius=3.0, constrained=True):
        super().__init__(dimension, 2, -5.0, 5.0, constraints=int(constrained))
        self.offset, self.radius, self.constrained = offset, radius, constrained

    def evaluate(self, x, context):
        a = x.square().sum(1)
        b = (x - self.offset).square().sum(1)
        cv = (a - self.radius**2).clamp_min(0) if self.constrained else torch.zeros_like(a)
        return torch.stack((a, b), 1), cv


if __name__ == "__main__":
    problem = cm.NativeProblem(
        source_dir=Path(__file__).parent / "native/bi_sphere",
        name="BiSphere", dimension=20, objectives=2,
        lower_bounds=-5.0, upper_bounds=5.0,
        parameters={"offset": 2.0, "radius": 3.0, "constrained": True},
    )
    result = cm.NSGA3(problem=problem, population_size=256,
                      max_generations=100).run()
    print(result.objectives)
