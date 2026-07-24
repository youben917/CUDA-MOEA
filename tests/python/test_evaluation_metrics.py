import importlib.util
import importlib
from pathlib import Path
import sys
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def load_metrics_module():
    path = Path(__file__).resolve().parents[1] / "evaluation"
    sys.path.insert(0, str(path))
    return importlib.import_module("metrics")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable")
class HypervolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = load_metrics_module()
        import torch

        cls.torch = torch

    def test_single_point_exact_volume(self):
        score = self.metrics.hypervolume([[1.0, 2.0]], [3.0, 4.0])
        self.assertAlmostEqual(score, 4.0)

    def test_union_and_dominated_points(self):
        points = [[1.0, 2.0], [2.0, 1.0], [2.5, 2.5]]
        score = self.metrics.hypervolume(points, [3.0, 3.0])
        self.assertAlmostEqual(score, 3.0)

    def test_points_outside_reference_are_ignored(self):
        score = self.metrics.hypervolume(
            [[1.0, 1.0], [4.0, 0.0]], [3.0, 3.0]
        )
        self.assertAlmostEqual(score, 4.0)

    def test_maximization(self):
        score = self.metrics.hypervolume(
            [[2.0, 1.0], [1.0, 2.0]], [0.0, 0.0], maximize=True
        )
        self.assertAlmostEqual(score, 3.0)

    def test_three_dimensional_exact_volume(self):
        score = self.metrics.hypervolume(
            [[1.0, 1.0, 1.0]], [3.0, 4.0, 5.0], method="exact"
        )
        self.assertAlmostEqual(score, 24.0)

    def test_sobol_estimate_is_reproducible_and_close(self):
        arguments = dict(
            points=[[1.0, 1.0, 1.0]],
            reference_point=[3.0, 4.0, 5.0],
            method="monte_carlo",
            samples=4096,
            seed=17,
        )
        first = self.metrics.hypervolume(**arguments)
        second = self.metrics.hypervolume(**arguments)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first, 24.0, delta=0.1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable")
class ExpectedUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = load_metrics_module()
        import torch

        cls.torch = torch

    def test_raw_minimization_utility(self):
        score = self.metrics.expected_utility(
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        )
        self.assertAlmostEqual(score, -1.0 / 6.0, places=6)

    def test_normalized_minimization_utility(self):
        score = self.metrics.expected_utility(
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            ideal_point=[0.0, 0.0],
            nadir_point=[1.0, 1.0],
        )
        self.assertAlmostEqual(score, 5.0 / 6.0, places=6)

    def test_normalized_maximization_utility(self):
        score = self.metrics.expected_utility(
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            maximize=True,
            ideal_point=[1.0, 1.0],
            nadir_point=[0.0, 0.0],
        )
        self.assertAlmostEqual(score, 5.0 / 6.0, places=6)

    def test_adding_a_solution_cannot_reduce_eu(self):
        weights = self.metrics.preference_weights(128, 2, seed=11)
        small = self.metrics.expected_utility([[0.5, 0.5]], weights)
        expanded = self.metrics.expected_utility(
            [[0.5, 0.5], [0.0, 1.0]], weights
        )
        self.assertGreaterEqual(expanded, small)

    def test_sampled_weights_are_reproducible_simplex_points(self):
        first = self.metrics.preference_weights(32, 4, seed=9)
        second = self.metrics.preference_weights(32, 4, seed=9)
        self.assertTrue(self.torch.equal(first, second))
        self.assertTrue(
            self.torch.allclose(first.sum(dim=1), self.torch.ones(32))
        )

    def test_invalid_normalization_and_weights(self):
        with self.assertRaises(ValueError):
            self.metrics.expected_utility(
                [[0.0, 1.0]], ideal_point=[0.0, 0.0]
            )
        with self.assertRaises(ValueError):
            self.metrics.expected_utility([[0.0, 1.0]], [[-1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
