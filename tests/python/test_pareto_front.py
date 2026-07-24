import importlib.util
import unittest

from tests.evaluation import pareto_front
from tests.evaluation.snapshot_io import require_torch


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch unavailable")
class ParetoFrontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = pareto_front
        cls.torch = require_torch()

    def front(self, name, count=512, objectives=3):
        return self.evaluation.pareto_front_points(
            name, objectives, count=count, seed=1234
        )

    def assert_close(self, actual, expected, tolerance=2e-5):
        self.assertTrue(
            self.torch.allclose(actual, expected, atol=tolerance, rtol=tolerance),
            f"maximum error: {(actual - expected).abs().max().item()}",
        )

    def test_all_factory_dtlz_problem_names(self):
        names = [
            "DTLZ1",
            "DTLZ2",
            "DTLZ3",
            "DTLZ4",
            "DTLZ5",
            "DTLZ6",
            "DTLZ7",
            "ConvexDTLZ2",
            "C1DTLZ1",
            "C1DTLZ3",
            "C2DTLZ2",
            "C2ConvexDTLZ2",
            "C3DTLZ1",
            "C3DTLZ4",
        ]
        for name in names:
            with self.subTest(name=name):
                points = self.front(name, count=128)
                self.assertEqual(points.shape, (128, 3))
                self.assertTrue(self.torch.isfinite(points).all())
                self.assertTrue((points >= 0).all())

    def test_all_constrained_kernel_objective_counts(self):
        cases = {
            "C1DTLZ3": (2, 3, 5, 8, 10, 15),
            "C2DTLZ2": (2, 3, 5, 8, 10, 15),
            "C2ConvexDTLZ2": (3, 5, 8, 10, 15),
        }
        for name, objective_counts in cases.items():
            for objective_count in objective_counts:
                with self.subTest(name=name, objective_count=objective_count):
                    points = self.front(name, count=32, objectives=objective_count)
                    self.assertEqual(points.shape, (32, objective_count))

    def test_simplex_fronts(self):
        for name in ("DTLZ1", "C1DTLZ1"):
            with self.subTest(name=name):
                points = self.front(name)
                self.assert_close(
                    points.sum(dim=1), self.torch.full((points.shape[0],), 0.5)
                )

    def test_spherical_fronts(self):
        for name in ("DTLZ2", "DTLZ3", "DTLZ4", "C1DTLZ3"):
            with self.subTest(name=name):
                points = self.front(name)
                self.assert_close(
                    points.norm(dim=1), self.torch.ones(points.shape[0])
                )

    def test_dtlz5_and_dtlz6_degenerate_curve(self):
        for name in ("DTLZ5", "DTLZ6"):
            with self.subTest(name=name):
                points = self.front(name, objectives=4)
                self.assert_close(points.norm(dim=1), self.torch.ones(points.shape[0]))
                self.assert_close(points[:, 0], points[:, 1])
                self.assert_close(points[:, 2], points[:, 0] * 2.0**0.5)

    def test_dtlz7_disconnected_front(self):
        points = self.front("DTLZ7")
        free = points[:, :-1]
        in_first = (free >= 0.0) & (free <= 0.251411836)
        in_second = (free >= 0.631626531) & (free <= 0.859400856)
        self.assertTrue((in_first | in_second).all())
        expected_last = 6.0 - (
            free * (1.0 + self.torch.sin(3.0 * self.torch.pi * free))
        ).sum(dim=1)
        self.assert_close(points[:, -1], expected_last)

    def test_convex_dtlz2_surface(self):
        points = self.front("ConvexDTLZ2")
        surface = points[:, :-1].sqrt().sum(dim=1) + points[:, -1]
        self.assert_close(surface, self.torch.ones(points.shape[0]))

    def test_c2_fronts_are_feasible(self):
        points = self.front("C2DTLZ2")
        self.assertTrue(
            self.evaluation._c2_dtlz2_feasible(points, 0.4).all()
        )

        points = self.front("C2ConvexDTLZ2")
        self.assertTrue(
            self.evaluation._c2_convex_dtlz2_feasible(points, 0.2).all()
        )

    def test_c3_dtlz1_lies_on_first_feasible_boundary(self):
        points = self.front("C3DTLZ1")
        constraints = points + 1.0 - 2.0 * points.sum(dim=1, keepdim=True)
        self.assertTrue((constraints <= 2e-5).all())
        self.assert_close(constraints.max(dim=1).values, self.torch.zeros(points.shape[0]))

    def test_c3_dtlz4_lies_on_first_feasible_boundary(self):
        points = self.front("C3DTLZ4")
        sum_squares = points.square().sum(dim=1, keepdim=True)
        constraints = 0.75 * points.square() - sum_squares + 1.0
        self.assertTrue((constraints <= 2e-5).all())
        self.assert_close(constraints.max(dim=1).values, self.torch.zeros(points.shape[0]))

    def test_sampling_is_reproducible(self):
        for name in ("DTLZ7", "C2DTLZ2", "C2ConvexDTLZ2", "C3DTLZ4"):
            with self.subTest(name=name):
                self.assertTrue(self.torch.equal(self.front(name), self.front(name)))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.evaluation.pareto_front_points("DTLZ2", 1)
        with self.assertRaises(ValueError):
            self.evaluation.pareto_front_points("DTLZ2", 3, count=0)
        with self.assertRaises(ValueError):
            self.evaluation.pareto_front_points("C2DTLZ2", 4)
        with self.assertRaises(ValueError):
            self.evaluation.pareto_front_points("C2ConvexDTLZ2", 2)


if __name__ == "__main__":
    unittest.main()
