# CUDA-MOEA 开发指南

[English](DEVELOPMENT.md) | 中文

本文说明 Python 发布仓库的本地构建、测试和报告维护流程。所有命令均从项目根目录执行。

## 源码结构

- `python/cuda_moea/`：Python 公共 API；
- `python/csrc/`：Python 扩展绑定和包私有的 CUDA 后端；
- `examples/`：可直接运行的 Python 示例；
- `tests/python/`：API、指标、Pareto front 与 benchmark 计划单元测试；
- `tests/evaluation/`：离线指标和可视化工具；
- `tests/benchmark/`：DTLZ、MoRobtrol benchmark、派生数据、图表和报告；
- `docs/API.md`：公开 API 参考。

原生后端仅用于构建 `cuda_moea._C`，不是独立的公共接口。修改后端时应通过 Python API
和 `tests/python/` 验证行为。

## 开发安装

先激活已经安装 CUDA 版 PyTorch 的 Python 环境，再执行：

```bash
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
  python -m pip install -e . --no-build-isolation
```

请按目标 GPU 修改 `89` 和 `8.9`。PyTorch、CUDA Toolkit 与宿主编译器必须 ABI 兼容。

## 检查与测试

语法检查：

```bash
python -m compileall -q python examples tests
```

完整 Python 单元测试：

```bash
python -m unittest discover -s tests/python -v
```

`tests/python/test_api.py` 中需要 GPU 的用例会在 CUDA 不可用时跳过。benchmark 依赖和
正式执行流程见 [`tests/benchmark/README.zh-CN.md`](../tests/benchmark/README.zh-CN.md)，其运行时间和
硬件需求远高于单元测试，不属于常规快速检查。

## 构建发布包

构建 wheel：

```bash
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
  python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
```

wheel 包含 Python API 和编译后的扩展。源码发布包通过 `MANIFEST.in` 收录扩展源码、文档、
示例、测试、派生数据、图表和报告；构建产物、缓存与 `output/` 原始运行目录不会提交。

## 报告维护

仓库保留用于复核报告的 CSV、图表和生成脚本。更新报告前应先阅读
[`tests/benchmark/TEST_PLAN.zh-CN.md`](../tests/benchmark/TEST_PLAN.zh-CN.md)，保持 GPU、软件版本、
seed 和测试矩阵一致。DTLZ 与 MoRobtrol 的分析入口分别是：

```bash
python tests/benchmark/DTLZ/analyze_results.py --help
python tests/benchmark/MoRobtrol/analyze_quality.py --help
```

不要手工修改派生数值来替代脚本重算。原始运行数据写入 `output/`，该目录由 Git 忽略。
