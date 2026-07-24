<p align="center">
  <img src="assets/logo.svg" alt="CUDA-MOEA" width="640">
</p>

# CUDA-MOEA

[English](README.md) | 简体中文

面向 PyTorch 的 GPU 加速多目标进化算法库。

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)
![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.1-orange)
![License](https://img.shields.io/badge/license-MIT-green)

CUDA-MOEA 是运行在 NVIDIA GPU 上的多目标进化算法（MOEA）Python/PyTorch 包。
整个进化循环——问题评估、交配选择、交叉、变异、参考方向维护和环境选择——
全部以 CUDA 原生内核执行，种群数据在世代之间始终驻留 GPU，无需回传 host。

本仓库只提供 Python 公共 API。构建时使用的原生 CUDA 后端以私有的
`cuda_moea._C` 扩展形式打包，不提供独立的命令行工具或公共 C++ API。

对应的研究论文将于 2026 年 8 月发布。

## 功能

- **算法**：NSGA-III 与 RVEA，可通过 `NSGA3` / `RVEA` 入口或通用的
  `Algorithm` / `AlgorithmBuilder` 接口使用。
- **CUDA 原生流水线**：世代循环的每个阶段都是精心优化的 CUDA 内核；输入与
  结果均为 `torch.Tensor`，提供零拷贝种群视图。
- **内置问题**：DTLZ1–7、ConvexDTLZ2、C1/C2/C3-DTLZ 约束变体和 CSDP。
- **可组合算子**：锦标赛/随机交配、模拟二进制交叉（SBX）、多项式/无变异、
  Das-Dennis、自适应 RVEA 及用户自定义参考方向。
- **PyTorch 可扩展性**：用纯 PyTorch 编写自定义问题和算子——包括神经网络
  目标——直接接入 GPU 进化循环。
- **完整生命周期控制**：可一次运行到底，也可逐代步进、查看实时种群，并保存
  周期性运行快照。

## 性能

CUDA-MOEA 与 [EvoX](https://github.com/EMI-Group/evox) 1.3.0 的对比测试
（CUDA-MOEA 0.1.0、PyTorch 2.12.0、NVIDIA RTX PRO 6000 Blackwell）。
DTLZ 测试集上的世代耗时的中位加速比：

| 扩展性研究 | NSGA-III | RVEA |
| --- | ---: | ---: |
| 基准（Group A，8 个问题） | 5.79–12.42× | 12.08–12.73× |
| 种群规模扩展（Group B，最大可比点） | 271.92×（`N=32768`） | 247.26×（`N=16384`） |
| 维度扩展（Group C，`D=131072`） | 10.12× | 10.27× |

在标称 `N=32768` 时，EvoX RVEA 10 次重复全部显存溢出（OOM），而 CUDA-MOEA
10 次全部完成。

左列为每代运行耗时，右列为相对 EvoX 的加速比；上下分别为种群规模扩展和
维度扩展：

<p><img src="tests/benchmark/DTLZ/results/images/population_generation_time.png" alt="每代运行耗时随种群规模变化" width="49%"> <img src="tests/benchmark/DTLZ/results/images/population_speedup.png" alt="种群规模扩展加速比" width="49%"></p>
<p><img src="tests/benchmark/DTLZ/results/images/dimension_generation_time.png" alt="每代运行耗时随决策维度变化" width="49%"> <img src="tests/benchmark/DTLZ/results/images/dimension_speedup.png" alt="维度扩展加速比" width="49%"></p>

解质量（IGD）取决于具体问题和算法：两个框架各有胜负，上述结果不可外推至
未测试的 GPU、软件栈或问题。计时范围为初始化之后的世代循环。完整的测试
方案、统计分析和 MoRobtrol 控制套件结果见
[综合测试报告](tests/benchmark/results/REPORT.zh-CN.md)。

## 内置组件

| 类别 | 组件 |
| --- | --- |
| 算法 | `NSGA3`、`RVEA`（另有通用的 `Algorithm`、`AlgorithmBuilder`） |
| 问题 | `DTLZ1`–`DTLZ7`、`ConvexDTLZ2`、`C1DTLZ1`、`C1DTLZ3`、`C2DTLZ2`、`C2ConvexDTLZ2`、`C3DTLZ1`、`C3DTLZ4`、`CSDP` |
| 交配选择 | `TournamentMating`、`RandomMating` |
| 交叉 | `SBX` |
| 变异 | `PolynomialMutation`、`NoMutation` |
| 参考方向 | `DasDennisDirections`、`AdaptiveRVEADirections`、`UserDefinedDirections` |
| 环境选择 | `NSGA3EnvironmentSelector`、`RVEAEnvironmentSelector` |
| 自定义策略 | `PythonProblem`、`PythonMating`、`PythonCrossover`、`PythonMutation`、`PythonReferenceDirections`、`PythonEnvironmentSelector` |

## 环境要求

- Python 3.9 或更高版本
- 支持 CUDA 的 PyTorch 2.1 或更高版本
- CUDA Toolkit（推荐 12.8 或更高版本）
- CMake 3.24 或更高版本
- 与所选 PyTorch/CUDA 工具链兼容的宿主编译器
- OpenMP
- 该工具链支持的 NVIDIA GPU

PyTorch、CUDA Toolkit 和宿主编译器必须 ABI 兼容。本仓库目前未定义经过
测试的操作系统矩阵。

## 从源码安装

在仓库根目录、已安装 CUDA 版 PyTorch 的环境中执行：

```bash
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
  python -m pip install . --no-build-isolation
```

请将 `89` 和 `8.9` 替换为目标 GPU 的计算能力。常见取值：

| `CMAKE_CUDA_ARCHITECTURES` | 代表性 GPU |
| ---: | --- |
| 80 | A100 |
| 86 | RTX 30 系 |
| 89 | RTX 40 系 |
| 90 | H100 / H200 |
| 100 | B100 / B200 |
| 120 | RTX 50 系、RTX PRO 6000 Blackwell |

开发模式安装请将 `install .` 替换为 `install -e .`。

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
print(result.variables.shape)    # torch.Size([1024, 12])
print(result.objectives.shape)   # torch.Size([1024, 3])
print(result.constraints.shape)  # torch.Size([1024])
```

逐代控制生命周期：

```python
algorithm.initialize()
while not algorithm.finished:
    algorithm.step()
result = algorithm.result()
```

自定义问题就是纯 PyTorch。`evaluate` 内可以使用任意 `torch` 计算，
包括神经网络：

```python
import torch
import cuda_moea as cm

class MyProblem(cm.PythonProblem):
    def __init__(self):
        super().__init__(dimension=32, objectives=3,
                         lower_bounds=-1.0, upper_bounds=1.0)

    def evaluate(self, x, context):
        # x: (N, 32) CUDA 张量 -> 目标: (N, 3) CUDA 张量
        return torch.stack([(x ** 2).sum(dim=1),
                            ((x - 0.5) ** 2).sum(dim=1),
                            ((x + 0.5) ** 2).sum(dim=1)], dim=1)

algorithm = cm.NSGA3(
    population_size=4096,
    max_generations=500,
    problem=MyProblem(),
    device="cuda:0",
)
result = algorithm.run()
```

## 示例

[`examples/`](examples/) 目录中的可运行程序：

- [`nsga3_torch.py`](examples/nsga3_torch.py)——NSGA-III 求解 500 维
  DTLZ1，种群规模 16384，周期性保存快照。
- [`rvea_torch.py`](examples/rvea_torch.py)——通过通用 `Algorithm` 入口
  运行 RVEA 求解带约束的 CSDP 问题。
- [`python_example.py`](examples/python_example.py)——完全自定义配置：
  神经网络目标，外加自定义交配、交叉、变异和参考方向。

## 文档

- [Python/PyTorch API 参考](docs/API.zh-CN.md)
- [开发与打包指南](docs/DEVELOPMENT.zh-CN.md)
- [发布前检查清单](docs/RELEASE_CHECKLIST.md)
- [评估指南](tests/evaluation/README.md)
- [Benchmark 执行指南](tests/benchmark/README.zh-CN.md)
- [Benchmark 测试方案](tests/benchmark/TEST_PLAN.zh-CN.md)
- [综合测试报告](tests/benchmark/results/REPORT.zh-CN.md)
- [DTLZ 报告](tests/benchmark/DTLZ/results/REPORT.zh-CN.md)
- [MoRobtrol 报告](tests/benchmark/MoRobtrol/results/REPORT.zh-CN.md)

## 测试

安装项目后，在仓库根目录运行 Python 测试套件：

```bash
python -m unittest discover -s tests/python -v
```

CUDA 不可用时，依赖 CUDA 的 API 测试会被跳过。Benchmark 运行有额外的
依赖以及更高的硬件和时间成本，请参阅专门的
[benchmark 指南](tests/benchmark/README.zh-CN.md)。

## 项目结构

```text
CUDA-MOEA/
├── docs/                 # API、开发与发布文档
├── examples/             # Python 示例
├── python/
│   ├── cuda_moea/        # Python 公共包
│   └── csrc/             # 包私有的原生 CUDA 扩展
├── tests/                # 单元、评估与 benchmark 测试
├── CMakeLists.txt        # 原生构建配置
├── pyproject.toml        # Python 包元数据
└── setup.py              # 基于 CMake 的扩展构建
```

## 发布状态

包元数据标识的当前版本为 `0.1.0`。当前检出尚未包含自动化发布流程，
也未声明支持/安全渠道。

## 许可证

CUDA-MOEA 基于 [MIT License](LICENSE) 开源。
