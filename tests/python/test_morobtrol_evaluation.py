from array import array
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.evaluation import evaluate_morobtrol


class MoRobtrolEvaluationTests(unittest.TestCase):
    def write_run(self, directory: Path, rewards: list[list[float]]) -> None:
        directory.mkdir(parents=True)
        objectives = len(rewards[0])
        population = len(rewards)
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "max_generations": 1,
                    "problem_name": "MoSwimmer",
                    "objective_count": objectives,
                    "population_size": population,
                }
            ),
            encoding="utf-8",
        )
        generation = directory / "generation_000001"
        generation.mkdir()
        (generation / "snapshot.json").write_text(
            json.dumps(
                {"objective_count": objectives, "population_size": population}
            ),
            encoding="utf-8",
        )
        # Recorder layout is objective-major and MoRobtrol stores -reward.
        values = [
            -rewards[row][column]
            for column in range(objectives)
            for row in range(population)
        ]
        with (generation / "objectives.bin").open("wb") as handle:
            array("f", values).tofile(handle)

    def test_two_runs_share_empirical_front_and_metric_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_run(root / "nsga3", [[1.0, 0.0], [0.5, 0.5]])
            self.write_run(root / "rvea", [[0.0, 1.0], [0.6, 0.4]])
            output = root / "evaluation"
            with mock.patch.object(evaluate_morobtrol, "plot_front_comparison"):
                status = evaluate_morobtrol.main(
                    [
                        "--run-root",
                        str(root),
                        "--device",
                        "cpu",
                        "--hv-method",
                        "exact",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads((output / "metrics.json").read_text())
            self.assertEqual(report["objective_count"], 2)
            self.assertEqual(report["problem_name"], "MoSwimmer")
            self.assertEqual(report["empirical_front_points"], 4)
            self.assertEqual(set(report["scores"]), {"NSGA3", "RVEA"})
            for value in report["hv_reference"]:
                self.assertAlmostEqual(value, -0.1, places=6)
            for metrics in report["scores"].values():
                self.assertGreater(metrics["hypervolume"], 0)
                self.assertGreaterEqual(metrics["expected_utility"], 0)


if __name__ == "__main__":
    unittest.main()
