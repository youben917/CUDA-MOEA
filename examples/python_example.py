import torch
import cuda_moea as cm


class NeuralProblem(cm.PythonProblem):
    def __init__(self, model):
        super().__init__(32, 3, lower_bounds=-1.0, upper_bounds=1.0)
        self.model = model.eval()

    def evaluate(self, x, context):
        with torch.no_grad():
            objectives = self.model(x)
        return objectives

class MyMating(cm.PythonMating):
    def initialize(self, info):
        self.population_size = info["population_size"]

    def select(self, parents, state, context):
        return torch.randint(
            0,
            state.get("active_count", parents["variables"].shape[0]),
            (parents["variables"].shape[0],),
            device=parents["variables"].device,
            dtype=torch.int32,
        )

class CloneCrossover(cm.PythonCrossover):
    def initialize(self, info):
        pass

    def apply(self, parents, parent_indices, context):
        return parents["variables"].index_select(0, parent_indices.long())


class GaussianMutation(cm.PythonMutation):
    def mutate(self, offspring, context):
        x = offspring["variables"]
        bounds = offspring["bounds"]
        mutated = x + 0.01 * torch.randn_like(x)
        return mutated.maximum(bounds[:, 0]).minimum(bounds[:, 1])

class AxisDirections(cm.PythonReferenceDirections):
    def initialize(self, requested_count, objective_count, context):
        return torch.eye(
            objective_count,
            device=f"cuda:{context['device']}",
            dtype=torch.float32,
        )

    def update(self, population, directions, active_count, context):
        return None  # None 表示保持原方向，也可以返回相同形状的新方向

    def reset(self):
        pass


model = torch.nn.Sequential(
    torch.nn.Linear(32, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 3),
).cuda()

algorithm = cm.NSGA3(
    population_size=16384,
    max_generations=4001,
    problem=NeuralProblem(model),
    mating=MyMating(),
    crossover=CloneCrossover(),
    mutation=GaussianMutation(),
    reference_directions=AxisDirections(),
    device="cuda:0",
    seed=2887,
    progress_interval=1000
)

result = algorithm.run()
print(f"time={result.total_ms} ms")
