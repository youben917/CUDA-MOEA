"""Cache/loader tests run without a GPU; numerical tests require CUDA."""
import concurrent.futures
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import torch
import cuda_moea as cm
from cuda_moea import _C
from cuda_moea import _native_build as builder
from cuda_moea.native import NativeProblem, benchmark


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/native/bi_sphere"


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source"
        self.source.mkdir()
        (self.source / "CMakeLists.txt").write_text("# mocked build\n")
        (self.source / "formula.h").write_text("version1")
        self.sdk = root / "sdk"
        self.sdk.mkdir()
        (self.sdk / "build.json").write_text(json.dumps({"build_id": "sdk1", "architectures": "89", "abi": 1}))
        self.cache = root / "cache"
        self.tools = {"cmake": ["cmake", "1"], "cxx": ["c++", "1"], "nvcc": ["nvcc", "1"]}
        self.commands = []
        self.output = None
        self.external = root / "external.h"
        self.external.write_text("external1")
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(builder, "_tools", return_value=self.tools).start()
        mock.patch.object(builder, "_run", side_effect=self.run_build).start()
        mock.patch.object(builder, "_dependencies", side_effect=lambda _: {
            str(self.external): builder._hash(self.external)}).start()

    def run_build(self, command, log, verbose):
        self.commands.append(command)
        if "-S" in command:
            self.output = Path(next(x.split("=", 1)[1] for x in command if x.startswith("-DCUDA_MOEA_PLUGIN_OUTPUT_DIR=")))
        else:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / "problem.so").write_bytes(b"test artifact")

    def build(self, **kwargs):
        return builder.build_problem(self.source, sdk_dir=self.sdk, cache_dir=self.cache, **kwargs)

    def test_cache_hit_never_and_missing_tools(self):
        first = self.build()
        self.assertEqual(first, self.build(mode="never"))
        with mock.patch.object(builder, "_tools", return_value={}):
            self.assertEqual(first, self.build())
        self.assertEqual(len(self.commands), 2)

    def test_never_miss_does_not_build(self):
        with self.assertRaises(FileNotFoundError):
            self.build(mode="never")
        self.assertFalse(self.commands)

    def test_source_header_and_options_invalidate(self):
        first = self.build()
        header = self.source / "formula.h"
        timestamp = header.stat().st_mtime_ns
        header.write_text("version2")
        os.utime(header, ns=(timestamp, timestamp))
        second = self.build()
        self.assertNotEqual(first, second)
        self.assertIn("--clean-first", self.commands[-1])
        self.assertNotEqual(second, self.build(debug=True))
        self.assertNotEqual(second, self.build(architectures="90"))
        self.assertNotEqual(second, self.build(cmake_options={"MY_FEATURE": "ON"}))

    def test_external_header_and_toolchain_invalidate(self):
        first = self.build()
        self.external.write_text("external2")
        second = self.build()
        self.assertNotEqual(first, second)
        self.tools["nvcc"] = ["nvcc", "2"]
        self.assertNotEqual(second, self.build())

    def test_sdk_invalidation_and_corruption(self):
        first = self.build()
        first.write_bytes(b"truncated")
        self.assertNotEqual(first, self.build())
        (self.sdk / "build.json").write_text(json.dumps({"build_id": "sdk2", "architectures": "89", "abi": 1}))
        with self.assertRaises(FileNotFoundError):
            self.build(mode="never")

    def test_failed_force_preserves_cached_artifact(self):
        first = self.build()
        with mock.patch.object(builder, "_run", side_effect=RuntimeError("compile error")):
            with self.assertRaisesRegex(RuntimeError, "compile error"):
                self.build(force=True)
        self.assertEqual(first, self.build(mode="never"))

    def test_concurrent_build_only_compiles_once(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            paths = list(pool.map(lambda _: self.build(), range(2)))
        self.assertEqual(paths[0], paths[1])
        self.assertEqual(len(self.commands), 2)

    def test_export_bundled_lookup_and_clean(self):
        first = self.build()
        bundled = builder.export_problem(first, self.sdk.parent / "native_problems" / self.source.name)
        with self.assertRaises(FileExistsError):
            builder.export_problem(first, bundled.parent)
        builder.clear_cache(self.cache)
        self.assertFalse(first.exists())
        self.assertEqual(bundled, self.build(mode="never"))
        (self.source / "formula.h").write_text("modified")
        with self.assertRaises(FileNotFoundError):
            self.build(mode="never")

    def test_no_source_changes_during_compilation(self):
        def mutate(*args):
            self.run_build(*args)
            (self.source / "formula.h").write_text("modified during compilation")
        with mock.patch.object(builder, "_run", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "changed during compilation"):
                self.build()
        self.assertFalse(list(self.cache.glob("artifacts/**/artifact.json")))


@unittest.skipUnless(shutil.which("nvcc") and shutil.which("cmake") and
                     (builder.sdk_directory() / "build.json").exists(), "CUDA toolchain/SDK unavailable")
class NativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "bi_sphere"
        shutil.copytree(EXAMPLE, cls.source)
        cls.external = cls.root / "external_config.h"
        cls.external.write_text("constexpr double default_radius = 3.0;\n")
        registration = cls.source / "registration.cpp"
        registration.write_text(f'#include "{cls.external}"\n' + registration.read_text().replace(
            '{"radius", 3.0,', '{"radius", default_radius,'))
        cls.cache = cls.root / "cache"
        cls.library = builder.build_problem(cls.source, cache_dir=cls.cache)

    def problem(self, **kwargs):
        options = dict(library=self.library, name="BiSphere", dimension=7,
                       objectives=2, lower_bounds=-5.0, upper_bounds=5.0)
        options.update(kwargs)
        return NativeProblem(**options)

    def test_registration_and_native_algorithm_construction(self):
        p = self.problem(parameters={"offset": 4.0, "constrained": True})
        self.assertEqual(p.schema["parameters"]["radius"]["default"], 3.0)
        self.assertEqual(p._spec()["type"], "NativeProblem")
        for algorithm in (cm.NSGA3, cm.RVEA):
            instance = algorithm(problem=p, population_size=32, max_generations=2, enable_warmup=False)
            self.assertFalse(instance.initialized)
            del instance
        gc.collect()

    def test_parameters_do_not_compile_or_alias(self):
        with mock.patch.object(builder, "_run", side_effect=AssertionError("unexpected compilation")):
            first = NativeProblem(self.source, name="BiSphere", dimension=7, objectives=2,
                                  parameters={"offset": 1.0}, cache_dir=self.cache)
            second = NativeProblem(self.source, name="BiSphere", dimension=129, objectives=2,
                                   parameters={"offset": 7.0}, cache_dir=self.cache, build="never")
        self.assertEqual(first.library, second.library)
        spec = first._spec()
        spec["parameters"]["offset"] = 99
        self.assertEqual(first._spec()["parameters"]["offset"], 1.0)

    def test_invalid_parameters_names_and_dimensions(self):
        for params in ({"offest": 2}, {"offset": True}, {"offset": float("nan")}, {"constrained": 1}):
            with self.assertRaises(ValueError):
                self.problem(parameters=params)
        with self.assertRaises(ValueError):
            self.problem(name="unknown")
        with self.assertRaises(ValueError):
            self.problem(lower_bounds=5.0)
        with self.assertRaisesRegex(ValueError, "objectives=2"):
            cm.NSGA3(problem=self.problem(objectives=3))

    def test_built_header_edit_and_failed_build(self):
        header = self.source / "problem.cuh"
        original = header.read_text()
        header.write_text(original + "\n// tracked header edit\n")
        rebuilt = builder.build_problem(self.source, cache_dir=self.cache)
        self.assertNotEqual(self.library, rebuilt)
        self.assertEqual(_C.native_problem_schema(str(rebuilt))["name"], "BiSphere")
        header.write_text(original + "\nthis is not valid C++;\n")
        with self.assertRaisesRegex(RuntimeError, "build failed"):
            builder.build_problem(self.source, cache_dir=self.cache)
        header.write_text(original)
        self.assertEqual(self.library, builder.build_problem(self.source, cache_dir=self.cache))

    def test_mismatched_sdk_rejected_before_factory(self):
        source = self.root / "mismatch.cpp"
        source.write_text('extern "C" { struct Plugin { unsigned abi; const char* id; }; '
                          'const Plugin* cuda_moea_problem_v1() { static Plugin p{1,"wrong"}; return &p; } }')
        library = self.root / "mismatch.so"
        subprocess.run([os.environ.get("CXX", "c++"), "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
        with self.assertRaisesRegex(ValueError, "SDK mismatch"):
            _C.native_problem_schema(str(library))

    def test_external_depfile_invalidates_real_compilation(self):
        manifest = json.loads((self.library.parent / "artifact.json").read_text())
        self.assertIn(str(self.external), manifest["dependencies"])
        previous = self.external.stat().st_mtime_ns
        try:
            self.external.write_text("constexpr double default_radius = 4.0;\n")
            os.utime(self.external, ns=(previous, previous))
            rebuilt = builder.build_problem(self.source, cache_dir=self.cache)
            self.assertNotEqual(rebuilt, self.library)
            schema = _C.native_problem_schema(str(rebuilt))
            self.assertEqual(schema["parameters"]["radius"]["default"], 4.0)
        finally:
            self.external.write_text("constexpr double default_radius = 3.0;\n")

    def test_bridge_revalidates_parameters(self):
        p = self.problem()
        p._config["parameters"] = {"offset": True}
        with self.assertRaisesRegex(ValueError, "Invalid type/value"):
            cm.NSGA3(problem=p)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_gpu_unconstrained_and_benchmark(self):
        p = self.problem(parameters={"offset": 2.0})
        algo = cm.NSGA3(problem=p, population_size=32, max_generations=1,
                        enable_warmup=False, print_progress=False)
        result = algo.run()
        torch.testing.assert_close(result.constraints, torch.zeros_like(result.constraints))
        timing = benchmark(p, result.variables, repeats=3, warmup=1)
        self.assertGreater(timing["wall_ms_per_eval"], 0)
        self.assertGreater(timing["cuda_ms_per_eval"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_gpu_objectives_constraints_streams_reset_and_lifetime(self):
        for algorithm in (cm.NSGA3, cm.RVEA):
            for dimension in (7, 129):
                p = self.problem(dimension=dimension, parameters={"offset": 1.5, "radius": 2.0, "constrained": True})
                with torch.cuda.stream(torch.cuda.Stream()):
                    algo = algorithm(problem=p, population_size=32, max_generations=2,
                                     enable_warmup=False, print_progress=False)
                    algo.initialize()
                    for action in (lambda: None, algo.step, algo.reset):
                        action()
                        population = algo.population
                        # RVEA marks inactive tail individuals with NaN variables
                        # and sentinel objectives; only live rows represent points.
                        valid = torch.isfinite(population.variables).all(1)
                        self.assertTrue(valid.any())
                        x = population.variables[valid]
                        expected = torch.stack((x.square().sum(1), (x - 1.5).square().sum(1)), 1)
                        torch.testing.assert_close(population.objectives[valid], expected, rtol=2e-5, atol=2e-4)
                        torch.testing.assert_close(population.constraints[valid], (expected[:, 0] - 4).clamp_min(0), rtol=2e-5, atol=2e-4)
                    del population, x, expected, valid
                    result = algo.run(copy=False)
                    del algo, p
                    gc.collect()
                    self.assertGreater(result.active_count, 0)
                    self.assertTrue(torch.isfinite(result.objectives[:result.active_count]).all())


if __name__ == "__main__":
    unittest.main()
