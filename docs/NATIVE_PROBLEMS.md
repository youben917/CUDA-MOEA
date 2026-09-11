# Native C++/CUDA problems

English | [简体中文](NATIVE_PROBLEMS.zh-CN.md)

`cm.NativeProblem` lets a Python algorithm use an independently compiled
`IProblemEvaluator`. Ordinary `cm.PythonProblem` subclasses remain supported:
their PyTorch computations already run on the GPU. Native implementations can
remove evaluation callbacks, fuse kernels, reuse workspaces, and write directly
to population buffers; speedups depend on the problem and kernel implementation.

## Requirements and quick start

This extension mechanism currently supports Linux. Install CUDA-MOEA first.
Compilation requires CMake 3.24+, Make, and the same host compiler identity/version
and CUDA compiler version used to build its SDK. The SDK applies the core's
PyTorch C++ ABI flags. Set `CXX` and `CUDACXX` to compiler executables when needed.
Source compilation needs no GPU. Running an optimization needs a compatible
NVIDIA GPU and driver. Loading a compatible prebuilt library does not invoke a
compiler.

From a source checkout, after installing the package:

```bash
python -m cuda_moea.native build examples/native/bi_sphere --verbose
python examples/native_problem.py
```

The build command prints the cached `problem.so` path. Subsequent uses with the
same inputs reuse that immutable file. To build on first use instead:

```python
import cuda_moea as cm

problem = cm.NativeProblem(
    source_dir="examples/native/bi_sphere",
    name="BiSphere",
    dimension=20,
    objectives=2,
    lower_bounds=-5.0,
    upper_bounds=5.0,
    parameters={"offset": 2.0, "radius": 3.0, "constrained": True},
    build="auto",
)
algorithm = cm.NSGA3(problem=problem, population_size=256, max_generations=100)
result = algorithm.run()
```

The existing `AlgorithmBuilder.problem(problem)` path also accepts this object.
No `problem.json` is required. Registration provides parameter types, defaults,
and descriptions; `problem.schema` exposes that information.

## Writing a problem

Copy [the BiSphere directory](../examples/native/bi_sphere/). Its files separate
the class declaration (`problem.cuh`), implementation/kernel (`problem.cu`), and
registration (`registration.cpp`). Small problems may combine translation units.

Implement the existing interface:

```cpp
#include "cuda_moea/native/plugin.cuh"

class MyProblem final : public cuda_moea::IProblemEvaluator {
public:
    explicit MyProblem(const cuda_moea::native::ProblemConfig& config);
    const cuda_moea::ProblemInfo& info() const noexcept override;
    void initialize(cuda_moea::CudaContext&) override;
    void evaluate(cuda_moea::PopulationView,
                  const cuda_moea::EvaluationContext&, cudaStream_t) override;
    // Optional: prepare_parent(...) and reset().
};
```

The constructor receives dimensions, expanded float32 bounds, and validated
parameters. `ProblemInfo` must report the requested dimensions and bounds; reject
unsupported configurations explicitly. For example, BiSphere requires two
objectives. Constructors must not allocate GPU resources: the algorithm selects
the CUDA device before `initialize()`.

Register exactly one problem per extension:

```cpp
const std::vector<cuda_moea::native::ParameterSpec> parameters{
    {"offset", 2.0, "Center of the second objective"},
    {"constrained", false, "Enable constraints"},
};
CUDA_MOEA_REGISTER_PROBLEM(MyProblem, "MyProblem", parameters)
```

`ParameterSpec` infers types from the default variant: `std::int64_t`, `double`,
`bool`, `std::string`, or `std::vector<double>`. Retrieve with
`config.get<double>("offset")`, for example. Python passes int/float, bool, str,
or a list/tuple of numbers as appropriate. Unknown keys, incorrect types, and
nonfinite numeric inputs are rejected. The class owns range/semantic validation
and any conversion/upload of vector parameters into reusable device buffers.

`evaluate()` is a host C++ method that launches CUDA kernels or calls GPU
libraries. Ordinary CPU loops are not automatically converted to GPU code.
Follow the original data and stream contract:

- Variables: float32 `(N,D)`, row-major; objectives: float32 `(M,N)`, objective-major.
- Constraints: `(N,)`, aggregate nonnegative violation; zero means feasible.
  Write this buffer on every evaluation, including unconstrained problems.
- Use the supplied stream and honor the actual population size. Views are
  non-owning; do not retain them across evaluations.
- Allocate/reuse owned workspaces in `initialize()` or on size changes. Enqueue
  frees on the appropriate stream. Destructors must not throw.
- Initial population evaluation uses `generation == -1`. `prepare_parent()` may
  refresh dynamic parent objectives and return true; `reset()` retains resources
  required by the next evaluation.
- Extra streams are the plugin's responsibility: join them to the supplied
  stream before returning, including before destruction.

The loader keeps the module alive through destruction of the problem, and waits
for the algorithm streams before/after destroying initialized native objects.
The optimization loop invokes the native object directly, without a Python
evaluation trampoline.

The problem CMake file is small:

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_problem LANGUAGES CXX CUDA)
find_package(CudaMoeaNative CONFIG REQUIRED)
cuda_moea_add_problem(my_problem problem.cu registration.cpp)
```

Add external libraries with `target_link_libraries(my_problem PRIVATE ...)`.
Build outputs must stay outside the problem directory. The SDK includes headers,
CMake helpers and an exact-build identifier; both `_C` and plugins link the
packaged `libcuda_moea_native.so`. This does not restore a standalone C++
application distribution or promise cross-version binary compatibility.

## Build and cache controls

`NativeProblem` requires exactly one of `source_dir` or `library`, plus `name`,
`dimension`, and `objectives`. Bounds default to `[0,1]`; scalars expand to the
dimension. Configuration is copied at construction. Create a new problem and
algorithm to change parameters; compatible machine code is reused. In-place
updates during optimization are not supported.

With `source_dir`, build options are:

| Option | Meaning |
| --- | --- |
| `build="auto"` | Reuse a matching bundled binary/cache; otherwise compile |
| `build="never"` | Reuse only; fail on a miss without compiling |
| `cache_dir=None` | Override `CUDA_MOEA_NATIVE_CACHE` or XDG cache default |
| `architectures=None` | Explicit CMake architectures, e.g. `"89;90"`; otherwise environment `CMAKE_CUDA_ARCHITECTURES`, then SDK default |
| `debug=False` | Debug builds include device debug information (`-G`, `-lineinfo`) |
| `force=False` | Force a clean rebuild into a new immutable artifact |
| `verbose=False` | Stream build output; logs are always retained |
| `cmake_options=None` | Dictionary of additional CMake `-D` settings; SDK/output/toolchain controls are reserved |
| `dependencies=()` | External library, data, or build-file paths/directories to fingerprint |
| `jobs=None` | Positive build parallelism; default is up to 8 workers |

Cache inputs include the full problem directory, explicit dependencies, SDK,
architectures, build options, relevant environment, and available tool identities.
Compiler depfiles track included headers outside the problem directory. Declare
external linked libraries, CMake modules and other build inputs in `dependencies`;
they are not automatically discovered from arbitrary custom CMake commands.
Declared dependency directories track additions/removals as well as content.
Runtime parameter values are excluded from the compilation key.

Unchanged source with a compatible cache needs no compiler; available changed
tool versions invalidate source-build caches. Portable prebuilt libraries are
loaded explicitly by path and checked against the SDK, without tool probing.
No GPU-architecture detection occurs at import: choose architectures that cover
the deployment GPU.

CMake workspaces are reused for incremental compilation. Artifact directories
are immutable, builds/cache cleanup are locked, and failed builds leave previous
artifacts intact. Load changed code through a new problem object; restarting the
Python process is recommended while debugging. Stop processes using the cache
before cleanup, since a stored problem configuration may instantiate later.

```bash
python -m cuda_moea.native build examples/native/bi_sphere --debug --force
python -m cuda_moea.native inspect /path/to/problem.so
python -m cuda_moea.native sdk
python -m cuda_moea.native clean
```

## Precompilation and wheels

Export an independently compiled problem to a new directory:

```bash
python -m cuda_moea.native build examples/native/bi_sphere --output dist/bi_sphere
```

Then use `cm.NativeProblem(library="dist/bi_sphere/problem.so", name="BiSphere",
dimension=20, objectives=2, ...)`. The loader needs the same SDK build, compatible
GPU architecture and runtime libraries. A mismatch reports that the extension
must be rebuilt. Load only trusted native code.

To include problems while building CUDA-MOEA's wheel, set a Linux colon-separated
list of source directories with unique basenames:

```bash
CUDA_MOEA_NATIVE_PROBLEMS=examples/native/bi_sphere \
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
```

Bundled binaries reside in `cuda_moea/native_problems/<directory-name>/problem.so`:

```python
from pathlib import Path
library = Path(cm.__file__).parent / "native_problems/bi_sphere/problem.so"
problem = cm.NativeProblem(library=library, name="BiSphere", dimension=20, objectives=2)
```

When a matching source directory is supplied, automatic mode also checks the
bundled binary's source fingerprint. A source archive includes repository example
sources; external directories named by the environment variable are not added to
the source archive automatically. Explicit library loading needs no source tree.

## Validation and performance

```bash
python -m unittest discover -s tests/python -p test_native_problem.py -v
python examples/benchmark_native_problem.py --population 1024 --dimension 128
python examples/benchmark_native_problem.py --torch-compile
```

Cache tests run without CUDA. Build/loader tests require a toolkit and SDK; GPU
tests skip when a device is unavailable. GPU checks cover objectives, constraints,
nondefault streams, reset and result ownership. The benchmark reports setup time
(cache lookup or compilation), steady batch evaluation time and complete
optimization runs separately. `cuda_moea.native.benchmark(problem, variables,
repeats=100, warmup=10)` accepts either problem implementation and a float32 CUDA
tensor. CUDA-event intervals include GPU idle gaps caused by host dispatch; wall
time also includes the final wait. Neither is a sum of kernel execution durations.
Compare numerical results before interpreting timing differences.

Build errors include a log path and compiler output. Missing kernels usually
mean the chosen architectures do not cover the GPU. Undefined symbols indicate
missing link dependencies. An SDK mismatch requires rebuilding against the
currently installed package rather than copying an old binary over the new one.
