# CUDA-MOEA

English | [简体中文](README.zh-CN.md)

CUDA-MOEA is a Python/PyTorch package for running multi-objective evolutionary algorithms on NVIDIA GPUs.

It features a fully CUDA-native backend with each kernels carefully optimized for maximum computational efficiency and GPU performance.

Currently, it renders **NSGA-III** and **RVEA** with DTLZ benchmark problems.

The supported public interface is Python. The native CUDA backend is packaged
as the private `cuda_moea._C` extension; it is not a standalone CLI or a public
C++ API.

## Features

- NSGA-III and RVEA algorithm co entry points
- Tournament mating, simulated binary crossover (SBX), and polynomial mutation
- DTLZ1–7, ConvexDTLZ2, C1/C2/C3-DTLZ, and CSDP problems
- Custom problems and operators written with PyTorch

## Requirements

- Python 3.9 or later
- A CUDA-enabled build of PyTorch 2.1 or later
- CUDA Toolkit (12.8 or later is recommended)
- CMake 3.24 or later
- A host compiler compatible with the selected PyTorch/CUDA toolchain
- OpenMP
- An NVIDIA GPU supported by that toolchain

PyTorch, the CUDA Toolkit, and the host compiler must be ABI-compatible. The
repository does not currently define a tested operating-system matrix.

## Install from source

From the repository root, in an environment that already contains a
CUDA-enabled PyTorch installation, run:

```bash
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
  python -m pip install . --no-build-isolation
```

Replace `89` and `8.9` with the compute capability of the target GPU. For an
editable development installation, replace `install .` with `install -e .`.

## Quick start

```python
import cuda_moea as cm

algorithm = cm.NSGA3(
    population_size=1024,
    max_generations=500,
    problem=cm.DTLZ2(dimension=12, objectives=3),
    device="cuda:0",
    seed=2887,
)

result = algorithm.run(copy=True)
print(result.variables.shape)    # torch.Size([1024, 12])
print(result.objectives.shape)   # torch.Size([1024, 3])
print(result.constraints.shape)  # torch.Size([1024])
```

To control the lifecycle one generation at a time:

```python
algorithm.initialize()
while not algorithm.finished:
    algorithm.step()
result = algorithm.result()
```

See [`examples/`](examples/) for additional runnable programs.

## Documentation

- [Python/PyTorch API reference](docs/API.md)
- [Development and packaging guide](docs/DEVELOPMENT.md)
- [Pre-release checklist](docs/RELEASE_CHECKLIST.md)
- [Evaluation guide](tests/evaluation/README.md)
- [Benchmark execution guide](tests/benchmark/README.md)
- [Benchmark protocol](tests/benchmark/TEST_PLAN.md)
- [Combined benchmark report](tests/benchmark/results/REPORT.md)
- [DTLZ report](tests/benchmark/DTLZ/results/REPORT.md)
- [MoRobtrol report](tests/benchmark/MoRobtrol/results/REPORT.md)

## Test

After installing the package, run the Python test suite from the repository
root:

```bash
python -m unittest discover -s tests/python -v
```

CUDA-dependent API tests are skipped when CUDA is unavailable. Benchmark runs
have additional dependencies and substantially higher hardware and time costs;
follow the dedicated [benchmark guide](tests/benchmark/README.md).

## Repository layout

```text
CUDA-MOEA/
├── docs/                 # API, development, and release documentation
├── examples/             # Python examples
├── python/
│   ├── cuda_moea/        # Public Python package
│   └── csrc/              # Private native CUDA extension
├── tests/                 # Unit, evaluation, and benchmark suites
├── CMakeLists.txt         # Native build configuration
├── pyproject.toml         # Python package metadata
└── setup.py               # CMake-backed extension build
```

## Release status

The package metadata identifies the current version as `0.1.0`. This checkout
does not yet contain an automated publishing workflow or a stated
support/security channel.

## License

CUDA-MOEA is licensed under the [MIT License](LICENSE).
