"""Static checks for TEST_PLAN.md execution matrices."""

from collections import Counter
import unittest

from tests.benchmark.common import effective_population, requested_population
from tests.benchmark.DTLZ.plan import (DIMENSIONS, POPULATIONS, quality_jobs,
    population_timing_jobs, dimension_timing_jobs)
from tests.benchmark.MoRobtrol.plan import (
    CHECKPOINT_INTERVAL,
    QUALITY_POPULATION,
    quality_jobs as robot_quality_jobs,
)


class BenchmarkPlanTests(unittest.TestCase):
    def test_dtlz_group_a(self):
        jobs = quality_jobs()
        self.assertEqual(len(jobs), 960)
        self.assertEqual({j.nominal_population for j in jobs}, {1024})
        self.assertEqual({j.generations for j in jobs}, {500})
        self.assertEqual({j.seed for j in jobs}, set(range(30)))
        self.assertTrue(all(j.save_objectives for j in jobs))
        self.assertEqual(set(Counter((j.problem, j.framework, j.algorithm) for j in jobs).values()), {30})

    def test_dtlz_timing_groups(self):
        b, c = population_timing_jobs(), dimension_timing_jobs()
        self.assertEqual((len(b), len(c)), (320, 440))
        self.assertEqual({j.nominal_population for j in b}, set(POPULATIONS))
        self.assertNotIn(128, POPULATIONS)
        self.assertIn(32768, POPULATIONS)
        self.assertEqual({j.dimension for j in b}, {500})
        self.assertEqual({j.dimension for j in c}, set(DIMENSIONS))
        self.assertEqual({j.nominal_population for j in c}, {1024})
        self.assertTrue(all(j.generations == 100 and not j.save_objectives for j in b + c))

    def test_population_contract(self):
        self.assertEqual(effective_population("rvea", 3, 1024), 990)
        self.assertEqual(requested_population("cuda_moea", "rvea", 3, 1024), 990)
        self.assertEqual(requested_population("evox", "rvea", 3, 1024), 1024)
        self.assertEqual(effective_population("nsga3", 3, 1024), 1024)

    def test_morobtrol_groups(self):
        d = robot_quality_jobs()
        self.assertEqual(len(d), 360)
        self.assertEqual({j.seed for j in d}, set(range(10)))
        self.assertTrue(all(j.checkpoints for j in d))
        self.assertEqual({j.nominal_population for j in d}, {QUALITY_POPULATION})
        self.assertEqual({j.generations for j in d}, {100})
        self.assertEqual({j.checkpoint_interval for j in d}, {CHECKPOINT_INTERVAL})
        self.assertEqual(CHECKPOINT_INTERVAL, 5)
        self.assertEqual(
            set(Counter((j.environment, j.framework, j.algorithm) for j in d).values()),
            {10},
        )

    def test_implementation_order_rotates(self):
        jobs = population_timing_jobs()[:8]
        self.assertNotEqual([(j.framework, j.algorithm) for j in jobs[:4]],
                            [(j.framework, j.algorithm) for j in jobs[4:8]])


if __name__ == "__main__": unittest.main()
