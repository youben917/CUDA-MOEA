import unittest

import torch

import cuda_moea as cm


class Sphere(cm.PythonProblem):
    def __init__(self):
        super().__init__(4, 2)

    def evaluate(self, variables, context):
        return torch.stack((variables.square().sum(1),
                            (variables - 1).square().sum(1)), 1)


class Mating(cm.PythonMating):
    def select(self, parents, state, context):
        return torch.arange(parents["variables"].shape[0],
                            device=parents["variables"].device,
                            dtype=torch.int32)


class Crossover(cm.PythonCrossover):
    def apply(self, parents, parent_indices, context):
        return parents["variables"][parent_indices.long()]


class Mutation(cm.PythonMutation):
    def mutate(self, offspring, context):
        return offspring["variables"]


class References(cm.PythonReferenceDirections):
    def initialize(self, requested_count, objective_count, context):
        return torch.eye(objective_count, device=f"cuda:{context['device']}")


class Selector(cm.PythonEnvironmentSelector):
    def select(self, parents, offspring, context):
        size = parents["variables"].shape[0]
        return {"indices": torch.arange(size, device=parents["variables"].device)}


class ApiTests(unittest.TestCase):
    def test_sbx_variable_copy_probability(self):
        crossover = cm.SBX(variable_copy_probability=0.5)
        self.assertEqual(crossover.variable_copy_probability, 0.5)
        self.assertEqual(crossover._spec()["variable_copy_probability"], 0.5)
        algorithm = cm.NSGA3(
            population_size=16, max_generations=1, crossover=crossover,
            enable_warmup=False,
        )
        self.assertFalse(algorithm.initialized)

    def test_initial_population_construct(self):
        initial = torch.zeros((16, 4), dtype=torch.float32)
        algorithm = cm.NSGA3(
            population_size=16, max_generations=1, problem=Sphere(),
            initial_population=initial, enable_warmup=False,
        )
        self.assertFalse(algorithm.initialized)

    def test_all_builtin_problems_construct(self):
        names = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6",
                 "DTLZ7", "ConvexDTLZ2", "C1DTLZ1", "C1DTLZ3", "C2DTLZ2",
                 "C2ConvexDTLZ2", "C3DTLZ1", "C3DTLZ4", "CSDP"]
        for name in names:
            algorithm = cm.NSGA3(population_size=16, max_generations=1,
                                 problem=getattr(cm, name)(8, 3),
                                 enable_warmup=False)
            self.assertFalse(algorithm.initialized)

    def test_all_builtin_strategies_construct(self):
        combinations = [
            dict(mating=cm.RandomMating(), crossover=cm.SBX(),
                 mutation=cm.NoMutation(),
                 reference_directions=cm.DasDennisDirections(2),
                 environment_selector=cm.NSGA3EnvironmentSelector()),
            dict(mating=cm.TournamentMating(),
                 mutation=cm.PolynomialMutation(),
                 reference_directions=cm.AdaptiveRVEADirections(),
                 environment_selector=cm.RVEAEnvironmentSelector()),
            dict(reference_directions=cm.UserDefinedDirections(torch.eye(3))),
        ]
        for options in combinations:
            self.assertFalse(cm.Algorithm("NSGA3", population_size=16,
                                         max_generations=1,
                                         enable_warmup=False,
                                         **options).initialized)

    def test_custom_strategies_construct(self):
        algorithm = cm.NSGA3(
            population_size=16, max_generations=1, problem=Sphere(),
            mating=Mating(), crossover=Crossover(), mutation=Mutation(),
            reference_directions=References(),
            environment_selector=Selector(), enable_warmup=False)
        self.assertFalse(algorithm.initialized)

    def test_builder(self):
        algorithm = (cm.AlgorithmBuilder("RVEA")
                     .population_size(16).max_generations(2)
                     .problem(cm.DTLZ2()).build())
        self.assertEqual(algorithm.generation, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_cuda_smoke(self):
        algorithm = cm.NSGA3(population_size=32, max_generations=2,
                             problem=Sphere(), enable_warmup=False,
                             print_progress=False)
        result = algorithm.run()
        self.assertEqual(tuple(result.variables.shape), (32, 4))
        self.assertEqual(tuple(result.objectives.shape), (32, 2))
        self.assertTrue(result.variables.is_cuda)


if __name__ == "__main__":
    unittest.main()
