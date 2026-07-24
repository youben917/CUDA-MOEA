# CUDA-MOEA

[English](README.md) | 中文

CUDA-MOEA 是面向 NVIDIA GPU 的多目标进化算法 Python/PyTorch 包，提供 NSGA-III、
RVEA、常用 DTLZ 测试问题以及可组合的交配、交叉、变异、参考方向和环境选择策略。

本仓库只提供 Python 公共 API。构建时使用的原生 CUDA 后端位于包内部，不提供独立的
命令行工具或原生语言 API。

## 功能

- NSGA-III 与 RVEA；
- DTLZ1–7、ConvexDTLZ2、C1/C2/C3-DTLZ 与 CSDP；
- Random/Tournament mating、SBX crossover、Polynomial/None mutation；
- Das-Dennis、Adaptive RVEA 和用户自定义参考方向；
- 支持使用 PyTorch 编写自定义 problem 和算子；
- CUDA Tensor 输入输出、零拷贝种群视图和运行快照。

## 环境要求

- Python 3.9 或更高版本；
- 支持 CUDA 的 PyTorch 2.1 或更高版本；
- CUDA Toolkit；
- CMake 3.24 或更高版本；
- 与当前 PyTorch/CUDA ABI 兼容的宿主编译器；
- OpenMP。

## 安装

在已经安装 CUDA 版 PyTorch 的环境中执行：

```bash
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
  pip install . --no-build-isolation
```

请将 `89` 和 `8.9` 改为目标 GPU 的计算能力。

开发模式：

```bash
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
  pip install -e . --no-build-isolation
```

## 快速开始

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
print(result.variables.shape)
print(result.objectives.shape)
print(result.constraints.shape)
```

逐代执行：

```python
algorithm.initialize()
while not algorithm.finished:
    algorithm.step()
result = algorithm.result()
```

更多用法见 [`examples/`](examples/) 和 [Python API 参考](docs/API.zh-CN.md)。

## 文档与报告

- [Python API 参考](docs/API.zh-CN.md)：类型、参数、自定义策略、生命周期和快照格式；
- [评估指南](tests/evaluation/README.md)：IGD、HV、EU、Pareto front 与可视化；
- [Benchmark 指南](tests/benchmark/README.zh-CN.md)：测试环境、运行入口和结果校验；
- [Benchmark 测试方案](tests/benchmark/TEST_PLAN.zh-CN.md)：公平性约束、测试矩阵与统计方法；
- [综合测试报告](tests/benchmark/results/REPORT.zh-CN.md)：DTLZ 与 MoRobtrol 汇总结论；
- [DTLZ 报告](tests/benchmark/DTLZ/results/REPORT.zh-CN.md)；
- [MoRobtrol 报告](tests/benchmark/MoRobtrol/results/REPORT.zh-CN.md)。
- [开发指南](docs/DEVELOPMENT.zh-CN.md)：源码结构、构建、测试与报告维护。

## 测试

安装项目后执行：

```bash
python -m unittest discover -s tests/python -v
```

## 项目结构

```text
CUDA-MOEA/
├── examples/             # Python 示例
├── python/
│   ├── cuda_moea/       # Python 公共 API
│   └── csrc/             # 包私有的原生 CUDA 扩展
├── docs/                 # Python API 文档
├── tests/                # API、评估与 benchmark 测试及报告
├── CMakeLists.txt        # 扩展构建配置
├── pyproject.toml
└── setup.py
```

## 许可证

CUDA-MOEA 基于 [MIT License](LICENSE) 开源。
