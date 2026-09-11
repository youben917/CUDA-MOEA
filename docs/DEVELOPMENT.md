# CUDA-MOEA Development Guide

English | [简体中文](DEVELOPMENT.zh-CN.md)

This guide covers local development, validation, packaging, and benchmark
report maintenance. Run all commands from the repository root unless noted.

## Source layout

- `python/cuda_moea/` contains the public Python API.
- `python/csrc/` contains Python bindings and the package-private CUDA backend.
- `examples/` contains directly runnable Python examples.
- `tests/python/` contains API, metric, Pareto-front, and benchmark-plan tests.
- `tests/evaluation/` contains offline metrics and visualization utilities.
- `tests/benchmark/` contains the DTLZ and MoRobtrol benchmark harnesses,
  derived data, figures, and reports.
- `docs/API.md` is the authoritative public API reference.

The native backend builds the shared core and `cuda_moea._C`. The packaged SDK
supports `IProblemEvaluator` extensions; its exact build identifier rejects
incompatible binaries. See [native problem development](NATIVE_PROBLEMS.md)
for independent builds, caching, debugging, and optional wheel bundling.
Validate backend changes through the Python API and `tests/python/`.

## Toolchain

The build metadata requires Python 3.9+, setuptools 68+, CMake 3.24+, and
PyTorch 2.1+. Native compilation also requires the CUDA Toolkit, OpenMP, and a
host compiler compatible with the selected PyTorch/CUDA combination. C++17 and
CUDA C++17 are enabled by the root CMake project.

No lockfile or tested operating-system/compiler matrix is present. Record the
exact Python, PyTorch, CUDA, compiler, GPU, and CMake versions used for a release
candidate.

## Editable installation

Activate a Python environment that already contains CUDA-enabled PyTorch, then
run:

```bash
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
  python -m pip install -e . --no-build-isolation
```

Set both architecture values for the target GPU. `--no-build-isolation` is
required by the documented path so the build uses PyTorch from the active
environment.

## Normal validation loop

Compile Python sources without importing the native extension:

```bash
python -m compileall -q python examples tests
```

Run all Python unit tests after installing the extension:

```bash
python -m unittest discover -s tests/python -v
```

For a narrower check, pass a module such as
`tests.python.test_evaluation_metrics`. CUDA-specific tests in
`tests/python/test_api.py` skip when CUDA is unavailable.

The repository does not define formatter, linter, type-checker, or CI commands.
Do not present ad hoc local tooling as a required project check without first
establishing it in project configuration.

## Build a wheel

```bash
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
  python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
```

The wheel contains the Python package, compiled extension, shared core, SDK,
and any explicitly bundled native problems. The source
manifest includes the native extension sources plus `docs/`, `examples/`, and
`tests/`. Build outputs, caches, and raw benchmark `output/` directories should
not be treated as source artifacts.

Before publishing, inspect both archives rather than assuming the manifest is
correct:

```bash
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
python -m zipfile -l dist/cuda_moea-0.2.0-*.whl
```

Build the source distribution the same way the release workflow
(`.github/workflows/publish.yml`) does:

```bash
python -m build --sdist
```

Registry upload happens only through that tag-triggered workflow; do not
upload local builds by hand.

## Benchmark and report maintenance

Read the [benchmark protocol](../tests/benchmark/TEST_PLAN.md) before updating
results. Keep the GPU, software versions, seeds, initial-population protocol,
and experiment matrix fixed. The execution guide and validation commands are
in [`tests/benchmark/README.md`](../tests/benchmark/README.md).

The DTLZ and MoRobtrol analysis entry points are:

```bash
python tests/benchmark/DTLZ/analyze_results.py --help
python tests/benchmark/MoRobtrol/analyze_quality.py --help
```

Regenerate derived values with these scripts; do not hand-edit numerical
results. Raw runs belong under the ignored `output/` tree, while committed
reports, figures, and derived tables live below the relevant `results/`
directory.

## Release preparation

Use the [pre-release checklist](RELEASE_CHECKLIST.md). In particular, public
release is blocked until a license and a publishing procedure are explicitly
chosen. A security policy, contribution policy, compatibility promise, and
support commitment should only be added when maintainers establish them.
