# 原生 C++/CUDA 自定义问题

[English](NATIVE_PROBLEMS.md) | 简体中文

`cm.NativeProblem` 让 Python 算法直接使用独立编译的 `IProblemEvaluator`。
原有 `cm.PythonProblem` 保持可用，其 PyTorch 计算本身已在 GPU 上执行。
原生实现可以省去评估回调、融合内核、复用工作区并直接写入种群显存；实际加速幅度取决于问题和实现。

## 环境与快速使用

当前扩展机制支持 Linux。先安装 CUDA-MOEA。编译需要 CMake 3.24+、Make，
以及与 SDK 构建时相同的宿主编译器类型/版本和 CUDA 编译器版本。
SDK 自动沿用核心的 PyTorch C++ ABI 编译选项；必要时通过 `CXX`、`CUDACXX`
指定编译器可执行文件。编译不需要 GPU，运行优化需要兼容的 NVIDIA GPU 和驱动。
加载兼容的预编译动态库不会调用编译器。

在源码仓库根目录、安装包之后执行：

```bash
python -m cuda_moea.native build examples/native/bi_sphere --verbose
python examples/native_problem.py
```

第一条命令输出缓存中的 `problem.so` 路径。也可以首次使用时自动编译：

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

现有 `AlgorithmBuilder.problem(problem)` 同样适用。不需要 `problem.json`；
参数类型、默认值和描述由 C++ 注册信息定义，可通过 `problem.schema` 查看。

## 编写问题

复制 [BiSphere 示例目录](../examples/native/bi_sphere/)。`problem.cuh` 声明类，
`problem.cu` 实现类和 CUDA 内核，`registration.cpp` 注册问题；简单问题可合并文件。

沿用原项目接口：

```cpp
#include "cuda_moea/native/plugin.cuh"

class MyProblem final : public cuda_moea::IProblemEvaluator {
public:
    explicit MyProblem(const cuda_moea::native::ProblemConfig& config);
    const cuda_moea::ProblemInfo& info() const noexcept override;
    void initialize(cuda_moea::CudaContext&) override;
    void evaluate(cuda_moea::PopulationView,
                  const cuda_moea::EvaluationContext&, cudaStream_t) override;
    // 可选实现 prepare_parent(...) 和 reset()。
};
```

构造函数接收维度、目标数量、展开后的 float32 边界和经过类型校验的参数。
`ProblemInfo` 必须与请求的维度、目标数量和边界一致；不支持的配置应明确报错。
例如 BiSphere 要求两个目标。构造函数不要分配 GPU 资源，算法会在 `initialize()`
之前选择正确的 CUDA device。

每个扩展注册一个问题：

```cpp
const std::vector<cuda_moea::native::ParameterSpec> parameters{
    {"offset", 2.0, "Center of the second objective"},
    {"constrained", false, "Enable constraints"},
};
CUDA_MOEA_REGISTER_PROBLEM(MyProblem, "MyProblem", parameters)
```

参数类型由默认值确定，支持 `std::int64_t`、`double`、`bool`、`std::string`、
`std::vector<double>`，通过 `config.get<double>("offset")` 等方式读取。
Python 对应传入整数/浮点数、布尔值、字符串、数字 list/tuple。
未知参数、类型错误和非有限数值会被拒绝；取值范围与业务约束由问题类校验。
数组参数先作为宿主配置传入，按需在初始化时上传到自有 GPU 缓冲区。

`evaluate()` 本身是 CPU 上的 C++ 调度函数，内部启动 CUDA kernel 或调用 GPU 库。
普通 CPU 循环不会自动转换成 GPU 代码。遵守原接口约定：

- 变量是 float32 `(N,D)` 行优先布局；目标是 float32 `(M,N)` 目标优先布局。
- 约束是 `(N,)` 汇总非负违反量，零表示可行；无约束问题也必须每次写零。
- 使用传入的 stream 和实际种群大小；view 不拥有内存，不跨评估保存其指针。
- 初始化或规模变化时分配工作区并复用，在适当的 stream 上释放；析构函数不得抛异常。
- 初始化评估的 `generation == -1`；动态问题可在 `prepare_parent()` 更新父代并返回 true。
  `reset()` 后仍需保留下一次评估所需资源。
- 自建 CUDA stream 的同步由问题负责，返回前应将工作汇合到接口 stream，析构前亦然。

加载器会让动态库存活到问题析构完成，并在已初始化原生问题析构前后等待算法 streams。
每代由 C++ 算法直接调用原生对象，不经过 Python 评估回调。

构建文件：

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_problem LANGUAGES CXX CUDA)
find_package(CudaMoeaNative CONFIG REQUIRED)
cuda_moea_add_problem(my_problem problem.cu registration.cpp)
```

需要外部库时对目标使用 `target_link_libraries(my_problem PRIVATE ...)`。
构建输出应放在源码目录外。随包 SDK 包含头文件、CMake 辅助配置和精确构建标识；
Python 扩展与问题扩展共同链接包中的 `libcuda_moea_native.so`。
这提供的是问题扩展接口，不承诺不同版本之间的二进制兼容，也不是独立 C++ 应用发行包。

## 编译与缓存

`NativeProblem` 的 `source_dir` 和 `library` 必须且只能提供一个；`name`、
`dimension`、`objectives` 必填。边界默认 `[0,1]`，标量自动展开。
构造时复制配置；修改参数应创建新的问题和算法，已有兼容内核可以复用。
第一版不支持优化过程中的原地参数修改。

使用 `source_dir` 时可传入：

| 参数 | 行为 |
| --- | --- |
| `build="auto"` | 优先匹配随包产物/缓存，缺失时编译 |
| `build="never"` | 只复用已有产物，缺失时报错，不编译 |
| `cache_dir=None` | 覆盖 `CUDA_MOEA_NATIVE_CACHE` 或 XDG 默认缓存目录 |
| `architectures=None` | CMake 架构，如 `"89;90"`；未指定时依次使用环境变量 `CMAKE_CUDA_ARCHITECTURES`、SDK 默认值 |
| `debug=False` | Debug 构建加入设备调试信息 `-G`、`-lineinfo` |
| `force=False` | 强制清理重建，生成新的不可变产物 |
| `verbose=False` | 实时输出编译信息；无论是否启用都会保留日志 |
| `cmake_options=None` | 额外 CMake `-D` 设置字典；SDK、输出路径、工具链控制项保留 |
| `dependencies=()` | 显式纳入指纹的外部库、数据、构建文件或目录 |
| `jobs=None` | 正整数并行数，默认最多 8 个构建任务 |

缓存记录整个问题目录、显式依赖、SDK、架构、编译配置、相关环境变量以及可用工具的身份。
编译器依赖文件还会记录源码目录外的被包含头文件。外部链接库、CMake 模块以及自定义
构建命令读取的其他文件，需要通过 `dependencies` 显式声明；不会从任意 CMake
命令自动推导全部输入。目录依赖也检测新增和删除文件。运行时参数不进入编译缓存键。

缓存兼容时，没有编译器也可使用；若本机存在的工具版本发生变化，则源码构建缓存失效。
显式通过 `library` 加载预编译库只做 SDK 兼容校验，不探测编译器。
导入时不自动探测 GPU 架构，构建时应明确覆盖部署设备。

CMake 工作目录支持增量构建，最终产物目录不可变；构建和清理使用锁，失败不会覆盖
已有产物。修改源码后创建新问题对象加载新版本；调试时建议重启 Python 进程。
清理缓存前停止使用它的进程，因为已创建的问题配置仍可能在稍后实例化算法。

```bash
python -m cuda_moea.native build examples/native/bi_sphere --debug --force
python -m cuda_moea.native inspect /path/to/problem.so
python -m cuda_moea.native sdk
python -m cuda_moea.native clean
```

## 预编译与随包发布

导出到新目录：

```bash
python -m cuda_moea.native build examples/native/bi_sphere --output dist/bi_sphere
```

之后通过 `cm.NativeProblem(library="dist/bi_sphere/problem.so", name="BiSphere",
dimension=20, objectives=2, ...)` 使用。需要相同 SDK 构建、兼容 GPU 架构和运行库。
SDK 不匹配时明确要求重新编译；只加载可信原生代码。

构建 Python wheel 时，通过 Linux 冒号分隔的目录列表指定一起编译的问题，目录名不能重复：

```bash
CUDA_MOEA_NATIVE_PROBLEMS=examples/native/bi_sphere \
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
```

产物位于包内 `native_problems/<目录名>/problem.so`：

```python
from pathlib import Path
library = Path(cm.__file__).parent / "native_problems/bi_sphere/problem.so"
problem = cm.NativeProblem(library=library, name="BiSphere", dimension=20, objectives=2)
```

提供源码目录时，自动模式也会根据源码指纹匹配随包产物。源码发行包包含仓库示例源码；
环境变量指定的外部目录不会被自动纳入源码发行包。显式加载动态库不需要源码目录。

## 验证与性能

```bash
python -m unittest discover -s tests/python -p test_native_problem.py -v
python examples/benchmark_native_problem.py --population 1024 --dimension 128
python examples/benchmark_native_problem.py --torch-compile
```

缓存测试不需要 CUDA；构建/加载测试需要工具链和 SDK；没有 GPU 时跳过数值测试。
GPU 测试覆盖目标、约束、非默认 stream、reset 和结果所有权。
基准分别报告初始化耗时（含编译或缓存查询）、稳定批量评估耗时与完整优化耗时。
`cuda_moea.native.benchmark(problem, variables, repeats=100, warmup=10)`
接受两种问题实现和 float32 CUDA Tensor。
CUDA event 时间包含 CPU 发射不及时导致的 GPU 空闲间隔；墙钟时间还包含最后等待，
两者都不是单纯的 kernel 时长之和。解释性能之前应先确认数值一致性。

编译错误会给出日志路径和编译器输出。找不到适用 kernel 通常表示架构不覆盖 GPU；
未定义符号通常表示缺少链接依赖；SDK 不匹配时应针对当前包重新编译，不要覆盖旧库冒充兼容版本。
