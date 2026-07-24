# CUDA-MOEA Python/PyTorch API 参考

[English](API.md) | 中文

本文是 `cuda_moea` Python 包的完整用户与扩展 API 手册。包内置 CUDA 原生后端，
公开接口均通过 Python/PyTorch 提供。

## 1. 功能范围

Python API 提供：

- `NSGA3` 与 `RVEA` 两种算法入口；
- 初始化、单步、完整运行、结果读取、同步和重置生命周期；
- 全部 15 种内置 problem；
- Random/Tournament 交配、SBX 交叉、Polynomial/None 变异；
- Das-Dennis、Adaptive RVEA 和用户给定参考方向；
- NSGA-III、RVEA 环境选择；
- 使用 PyTorch 编写 problem、交配、交叉、变异、参考方向和环境选择；
- CUDA Tensor 结果、零拷贝种群视图和可选独立结果副本；
- 数据快照功能。

算法生命周期和内置算子由包内的 CUDA 后端执行，PyTorch 用于张量交互和执行用户定义
的策略回调。

## 2. 安装与导入

### 2.1 环境要求

- Python 3.9 或更高版本；
- CUDA 版 PyTorch 2.1 或更高版本；
- CMake 3.24 或更高版本；
- CUDA Toolkit；
- CUDA Toolkit 支持的宿主编译器；
- OpenMP。

PyTorch、CUDA Toolkit 和宿主编译器必须 ABI 兼容。建议在目标 PyTorch 环境中构建，
不要把一个环境生成的 `_C` 扩展复制到另一个 PyTorch 环境。

### 2.2 pip 安装

从 PyPI 安装最新发布版本：

```bash
python -m pip install cuda-moea
```

预编译 wheel 面向 Linux x86_64、CPython 3.11 和 CUDA 12.8 版 PyTorch，覆盖
计算能力 80、86、89、90、100、120；其他环境会改用源码包本地编译。

从源码检出安装则在项目根目录执行：

```bash
cd CUDA-MOEA
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
  pip install . --no-build-isolation
```

`89`/`8.9` 应替换为目标 GPU 的计算能力。`--no-build-isolation` 会直接使用当前环境中
已安装的 PyTorch，通常更适合 CUDA 扩展。

开发模式安装：

```bash
CMAKE_CUDA_ARCHITECTURES=89 TORCH_CUDA_ARCH_LIST=8.9 \
  pip install -e . --no-build-isolation
```

### 2.3 导入与版本

```python
import cuda_moea as cm

print(cm.__version__)  # 0.1.0
```

## 3. 最小示例

```python
import cuda_moea as cm

algorithm = cm.NSGA3(
    population_size=1024,
    max_generations=500,
    problem=cm.DTLZ2(dimension=12, objectives=3),
    device="cuda:0",
    seed=2887,
    print_progress=True,
)

result = algorithm.run()
print(result.variables.shape)     # torch.Size([1024, 12])
print(result.objectives.shape)    # torch.Size([1024, 3])
print(result.constraints.shape)   # torch.Size([1024])
```

`NSGA3` 使用 Tournament、SBX、Polynomial、Das-Dennis 和 NSGA-III 环境选择作为默认
组合；`RVEA` 使用 Random、SBX、Polynomial、Adaptive RVEA 和 RVEA 环境选择。

## 4. 算法构造 API

### 4.1 `Algorithm`

通用构造函数：

```python
cm.Algorithm(
    algorithm="NSGA3",
    *,
    population_size=1024,
    max_generations=100,
    problem=None,
    mating=None,
    crossover=None,
    mutation=None,
    reference_directions=None,
    environment_selector=None,
    initial_population=None,
    cuda=None,
    device=None,
    seed=None,
    enable_warmup=None,
    progress_interval=100,
    print_progress=True,
    save_data=None,
    save_interval=1,
)
```

参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `algorithm` | `str` | `"NSGA3"` | `"NSGA3"`、`"NSGA-3"` 或 `"RVEA"` |
| `population_size` | `int` | `1024` | 种群大小 `N` |
| `max_generations` | `int` | `100` | 最大代数 |
| `problem` | problem 对象 | `DTLZ2()` | 内置或 Python problem |
| `mating` | mating 对象 | 算法默认 | 交配选择 |
| `crossover` | crossover 对象 | `SBX()` | 交叉算子 |
| `mutation` | mutation 对象 | `PolynomialMutation()` | 变异算子 |
| `reference_directions` | 参考方向对象 | 算法默认 | 参考方向提供器 |
| `environment_selector` | 环境选择对象 | 算法默认 | 环境选择器 |
| `initial_population` | `torch.Tensor` / `None` | `None` | 可选的 `(N,D)` float32 初始决策矩阵；设置后跳过 CUDA 随机初始化 |
| `cuda` | `CudaConfig` | `CudaConfig()` | 完整 CUDA 配置 |
| `device` | `int`/`str` | `None` | 覆盖 `cuda.device_id`，支持 `0`、`"cuda"`、`"cuda:0"` |
| `seed` | `int` | `None` | 覆盖 `cuda.seed` |
| `enable_warmup` | `bool` | `None` | 覆盖 `cuda.enable_warmup` |
| `progress_interval` | `int` | `100` | 进度输出间隔 |
| `print_progress` | `bool` | `True` | 是否打印进度 |
| `save_data` | path-like | `None` | 非 `None` 时启用快照保存 |
| `save_interval` | `int` | `1` | 快照代间隔 |

传入 `cuda` 后再传 `device`、`seed` 或 `enable_warmup` 时，后三者优先。

### 4.2 `NSGA3` 与 `RVEA`

两者接受与 `Algorithm` 相同的关键字参数，但算法名称固定：

```python
nsga3 = cm.NSGA3(problem=cm.DTLZ1(12, 3))
rvea = cm.RVEA(problem=cm.DTLZ2(12, 3))
```

RVEA 当前为 Beta 路径。新问题、约束配置或大规模参数应保存快照并进行质量回归。

### 4.3 `AlgorithmBuilder`

Builder 是可选的 Python 链式接口：

```python
algorithm = (
    cm.AlgorithmBuilder("NSGA3")
    .population_size(1024)
    .max_generations(500)
    .problem(cm.DTLZ2(12, 3))
    .mating(cm.TournamentMating())
    .crossover(cm.SBX(eta_initial=30.0))
    .mutation(cm.PolynomialMutation(probability=1 / 12))
    .reference_directions(cm.DasDennisDirections())
    .environment_selector(cm.NSGA3EnvironmentSelector())
    .set(device="cuda:0", seed=2887, print_progress=False)
    .build()
)
```

公开方法包括 `set(**options)`、`population_size()`、`max_generations()`、`problem()`、
`mating()`、`crossover()`、`mutation()`、`reference_directions()`、
`environment_selector()` 和 `build()`。重复设置时最后一次设置生效。

## 5. 生命周期与状态

### 5.1 方法

| 方法 | 返回值 | 行为 |
|---|---|---|
| `initialize()` | `None` | 创建 CUDA 上下文、种群、策略工作区并评估初始种群；重复调用无操作 |
| `step()` | `None` | 执行一代；未初始化时自动初始化；完成后无操作 |
| `run(copy=True)` | `Result` | 运行到最大代数并返回设备结果 |
| `result(copy=True)` | `Result` | finalize 当前种群并返回结果；未初始化时抛出异常 |
| `reset()` | `None` | 复用已分配资源，重新随机初始化并回到 generation 0 |
| `synchronize()` | `None` | 同步算法内部 evaluation/execution CUDA streams |

手动逐代运行：

```python
algorithm.initialize()
while not algorithm.finished:
    algorithm.step()
    current = algorithm.population
final = algorithm.result()
```

### 5.2 属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `initialized` | `bool` | 是否已完成初始化 |
| `finished` | `bool` | generation 是否达到最大代数 |
| `generation` | `int` | 当前已完成的代数 |
| `population` | `Population` | 当前父代的零拷贝 CUDA Tensor 视图 |

读取 `population` 前必须初始化。该属性会先同步内部 streams，适合观测和调试；在每一代
都读取会引入同步开销。

## 6. `Population`、`Result` 与张量约定

### 6.1 `Population`

```python
@dataclass(frozen=True)
class Population:
    variables: torch.Tensor
    objectives: torch.Tensor
    constraints: torch.Tensor
```

### 6.2 `Result`

```python
@dataclass(frozen=True)
class Result(Population):
    auxiliary_indices: torch.Tensor
    active_count: int
    total_ms: float
```

字段约定：

| 字段 | dtype | Python 形状 | 含义 |
|---|---|---|---|
| `variables` | `float32` | `(N,D)` | 决策变量 |
| `objectives` | `float32` | `(N,M)` | 每个个体的目标值 |
| `constraints` | `float32` | `(N,)` | 聚合约束违反值，通常 `<= 0` 视为可行 |
| `auxiliary_indices` | `int32` | 算法相关 | NSGA-III rank 或 RVEA front-zero 索引 |
| `active_count` | Python `int` | 标量 | 选择器报告的有效个体数 |
| `total_ms` | Python `float` | 标量 | `run()` 的 CUDA 计时结果 |

原生后端内部 objectives 是 objective-major `(M,N)`；Python API 将其暴露为常用的
`(N,M)`。零拷贝 objectives 可能不是 contiguous，需要连续存储时调用
`tensor.contiguous()`。

### 6.3 `copy` 与生命周期

- `run(copy=True)`/`result(copy=True)`：返回独立 Tensor，可以安全保留或修改；
- `run(copy=False)`/`result(copy=False)`：返回底层种群的零拷贝视图；
- `algorithm.population`：始终返回零拷贝视图。

零拷贝 Tensor 会保持底层 native Algorithm 存活，但其内容可能在后续 `step()`、
`reset()` 或选择过程中被覆盖。需要跨代保存时应使用 `clone()` 或 `copy=True`。

算法本身是进化优化器，不提供端到端 autograd。自定义回调中如使用神经网络推理，建议
显式使用 `torch.no_grad()`，并不要让返回 Tensor 携带不必要的计算图。

## 7. CUDA 配置

```python
cm.CudaConfig(
    device_id=0,
    evaluation_pool_ratio=0.25,
    execution_pool_ratio=0.75,
    memory_pool_policy=1,
    seed=2887,
    enable_warmup=True,
)
```

| 字段 | 说明 |
|---|---|
| `device_id` | CUDA device 编号 |
| `evaluation_pool_ratio` | problem evaluation memory pool 比例 |
| `execution_pool_ratio` | 繁殖和选择 memory pool 比例 |
| `memory_pool_policy` | CUDA memory pool 策略编号 |
| `seed` | 初始种群及全部内置随机算子的种子 |
| `enable_warmup` | 初始化时是否 warm up cuBLAS/cuSOLVER |

两个 pool ratio 必须为正值。多个算法对象可以选择不同 device，但每个对象的所有内置
数据和 Python 回调输出必须位于该对象对应的 CUDA device。

## 8. 内置 problem API

所有内置 problem 使用统一构造形式：

```python
ProblemClass(
    dimension=12,
    objectives=3,
    constraint_activation_ratio=0.0,
)
```

可用类：

| 类 | 说明 |
|---|---|
| `DTLZ1`–`DTLZ7` | 标准 DTLZ 问题 |
| `ConvexDTLZ2` | 凸变换 DTLZ2 |
| `C1DTLZ1`, `C1DTLZ3` | C1 约束问题 |
| `C2DTLZ2`, `C2ConvexDTLZ2` | C2 约束问题 |
| `C3DTLZ1`, `C3DTLZ4` | C3 约束问题 |
| `CSDP` | 稀疏/约束 problem 路径 |

示例：

```python
problem = cm.C1DTLZ1(
    dimension=30,
    objectives=3,
    constraint_activation_ratio=0.5,
)
```

`DTLZProblem` 是这些类共享的基础规格类，通常直接实例化具体问题类即可。

## 9. 内置交配、交叉与变异

### 9.1 交配

```python
cm.RandomMating()
cm.TournamentMating()
```

Tournament 会使用环境选择器提供的 rank/reference/score 状态；Random 不依赖这些状态。

### 9.2 SBX

```python
cm.SBX(
    eta_initial=30.0,
    eta_final=30.0,
    probability=1.0,
    variable_copy_probability=0.0,
)
```

`eta` 可在运行期间从 initial 线性变化到 final。`probability` 是交叉概率。

`variable_copy_probability` 控制已配对个体中每个变量直接保留父代值的概率，默认 `0.0`。
将其设为 `0.5` 可匹配 EvoX 1.3.0 SBX 的逐变量复制掩码：

```python
cm.SBX(variable_copy_probability=0.5)
```

### 9.3 Polynomial mutation

```python
cm.PolynomialMutation(
    eta_initial=20.0,
    eta_final=20.0,
    probability=1.0,
)
```

当前 CUDA 实现将 `probability` 作为每个个体的期望变异强度，并对每个变量使用
`probability / dimension`。如果希望典型的逐变量概率 `1/D`，可保持 `probability=1.0`。

关闭变异：

```python
cm.NoMutation()
```

## 10. 内置参考方向

### 10.1 Das-Dennis

```python
cm.DasDennisDirections(partitions=0)
```

`partitions=0` 自动选择不超过种群规模的最大一层/两层参考点集；正数表示使用精确的
Das-Dennis 分区数 `H`。

### 10.2 Adaptive RVEA

```python
cm.AdaptiveRVEADirections(frequency=0.1)
```

`frequency` 表示参考向量自适应触发间隔相对于总迭代代数的比例。

### 10.3 用户给定方向

```python
directions = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])
provider = cm.UserDefinedDirections(directions, normalize=True)
```

输入形状为 `(K,M)`。构建算法时方向会复制到核心管理的设备缓冲，因此输入 Tensor 可以
位于 CPU。`normalize=True` 会规范化参考方向。

## 11. 内置环境选择

### 11.1 NSGA-III

```python
cm.NSGA3EnvironmentSelector(
    sparse_ratio=0.5,
    cv_bins=0,
    cv_clip_upper=0.0,
    cv_log_alpha=0.0,
    feasibility_epsilon=0.0,
)
```

| 参数 | 含义 |
|---|---|
| `sparse_ratio` | 非支配排序稀疏路径阈值/比例 |
| `cv_bins` | 约束违反量量化 bins，`0` 使用默认/关闭显式量化配置 |
| `cv_clip_upper` | CV 量化上界 |
| `cv_log_alpha` | CV 对数量化系数 |
| `feasibility_epsilon` | 可行性容差 |

### 11.2 RVEA

```python
cm.RVEAEnvironmentSelector(alpha=2.0)
```

`alpha` 控制 angle-penalized distance 随迭代进度增加的惩罚强度。

## 12. 自定义 problem

### 12.1 构造

```python
class MyProblem(cm.PythonProblem):
    def __init__(self):
        super().__init__(
            dimension=20,
            objectives=2,
            lower_bounds=-5.0,
            upper_bounds=5.0,
            constraints=0,
            name="BiSphere",
        )

    def evaluate(self, variables, context):
        f1 = variables.square().sum(dim=1)
        f2 = (variables - 2.0).square().sum(dim=1)
        return torch.stack((f1, f2), dim=1)
```

`lower_bounds`/`upper_bounds` 可以是标量，也可以是长度为 `dimension` 的序列。

### 12.2 `evaluate`

```python
evaluate(variables, context) -> objectives
evaluate(variables, context) -> (objectives, constraints)
evaluate(variables, context) -> {
    "objectives": objectives,
    "constraints": constraints,
}
```

- `variables`：`float32` CUDA Tensor，形状 `(N,D)`；
- `objectives`：CUDA Tensor，形状必须为 `(N,M)`；
- `constraints`：CUDA Tensor，必须包含 `N` 个值；
- 没有返回 constraints 时核心将其清零；
- 返回 Tensor 必须位于算法 CUDA device，dtype 会转换为 `float32`。

`context` 字段：

```python
{
    "generation": int,
    "max_generations": int,
    "device": int,
}
```

初始化种群的第一次评估使用 `generation == -1`。

### 12.3 可选生命周期 hook

自定义 problem 可以额外实现：

```python
initialize(context) -> None
prepare_parent(population, context) -> None | bool | evaluation_result
reset() -> None
```

`prepare_parent` 在每代繁殖前调用。返回 evaluation result 时会覆盖当前父代的 objectives
和 constraints，并通知环境选择器重新 prepare；返回 `True` 表示调用方已就地更新父代。

## 13. 自定义交配

```python
class MyMating(cm.PythonMating):
    def initialize(self, info):
        self.population_size = info["population_size"]

    def select(self, parents, state, context):
        return torch.randint(
            0,
            state.get("active_count", parents["variables"].shape[0]),
            (parents["variables"].shape[0],),
            device=parents["variables"].device,
            dtype=torch.int32,
        )
```

`select()` 必须返回 `N` 个父代索引。`state` 可能包含：

- `rank: int32 Tensor`；
- `reference_index: int32 Tensor`；
- `score: float32 Tensor`；
- `active_count: int`。

自定义环境选择器可以产生这些字段。交配实现不应假定每种选择器都会提供全部字段。

## 14. 自定义交叉

```python
class CloneCrossover(cm.PythonCrossover):
    def initialize(self, info):
        pass

    def apply(self, parents, parent_indices, context):
        return parents["variables"].index_select(0, parent_indices.long())
```

返回 Tensor 必须位于算法 device，形状为 `(N,D)`。`parents` 是 population 字典，
`parent_indices` 是 `int32` CUDA Tensor。

## 15. 自定义变异

返回新 Tensor：

```python
class GaussianMutation(cm.PythonMutation):
    def mutate(self, offspring, context):
        x = offspring["variables"]
        bounds = offspring["bounds"]
        mutated = x + 0.01 * torch.randn_like(x)
        return mutated.maximum(bounds[:, 0]).minimum(bounds[:, 1])
```

也可以原地修改 `offspring["variables"]` 并返回 `None`。返回 Tensor 时形状必须为
`(N,D)`。可选 `initialize(info)` 与交叉、交配相同。

## 16. 自定义参考方向

```python
class AxisDirections(cm.PythonReferenceDirections):
    def initialize(self, requested_count, objective_count, context):
        return torch.eye(
            objective_count,
            device=f"cuda:{context['device']}",
            dtype=torch.float32,
        )

    def update(self, population, directions, active_count, context):
        return None  # None 表示保持原方向，也可以返回相同形状的新方向

    def reset(self):
        pass
```

`initialize()` 返回 `(K,M)` CUDA Tensor。`update()` 不能改变 `K` 或 `M`；它可以原地
修改 directions 并返回 `None`，也可以返回相同形状的新 CUDA Tensor。`reset()` 可选。

## 17. 自定义环境选择

环境选择是自定义协议中能力最完整、责任也最大的一层。

### 17.1 生命周期

```python
initialize(info, references, context) -> None
prepare(population, context) -> state | None
select(parents, offspring, context) -> selection_dict
finalize(population, context) -> state | None       # 可选
reset() -> None                                     # 可选
```

`references` 是 `(K,M)` CUDA Tensor；算法没有参考方向提供器时为 `None`。

### 17.2 通过合并种群索引选择

```python
class FirstN(cm.PythonEnvironmentSelector):
    def select(self, parents, offspring, context):
        n = parents["variables"].shape[0]
        indices = torch.arange(n, device=parents["variables"].device)
        return {
            "indices": indices,
            "active_count": n,
        }
```

`indices` 必须包含 `N` 个索引，索引对象是 `[parents; offspring]`，范围为 `[0,2N)`。
核心会用同一组索引收集 variables、objectives 和 constraints。

### 17.3 直接返回下一代

```python
return {
    "variables": next_variables,       # (N,D)
    "objectives": next_objectives,     # (N,M)
    "constraints": next_constraints,   # (N,)
    "active_count": n,
}
```

必须同时提供三个种群字段。所有 Tensor 必须在算法 device 上。

### 17.4 mating/result 状态

`prepare()`、`select()` 或 `finalize()` 返回的字典还可以包含：

| 键 | 类型 | 用途 |
|---|---|---|
| `rank` | `int32` CUDA Tensor | Tournament mating 与最终辅助索引语义 |
| `reference_index` | `int32` CUDA Tensor | 参考方向关联状态 |
| `score` | `float32` CUDA Tensor | Tournament 比较分数 |
| `active_count` | `int` | 有效种群数量 |
| `result_indices` | `int32` CUDA Tensor | 写入 `Result.auxiliary_indices` |

状态由选择器对象持有到下一次更新。若自定义 mating 依赖某字段，选择器必须在首次繁殖前
通过 `prepare()` 提供它。

## 18. 回调中的通用字典

### 18.1 population 字典

```python
{
    "variables": float32 CUDA Tensor[N,D],
    "objectives": float32 CUDA Tensor[N,M],
    "constraints": float32 CUDA Tensor[N],
    "bounds": float32 CUDA Tensor[D,2],
}
```

这些都是核心缓冲区的非拥有视图。除明确允许就地修改的接口外，不要长期保存或修改。

### 18.2 info 字典

交配、交叉、变异的 `initialize(info)` 以及环境选择器的 initialize 会收到：

```python
{
    "name": str,
    "population_size": int,
    "dimension": int,
    "objective_count": int,
    "max_generations": int,
    "device": int,
}
```

### 18.3 generation context

```python
{
    "generation": int,
    "max_generations": int,
    "device": int,
}
```

## 19. CUDA stream、设备与性能约定

库内部使用 evaluation stream 和 execution stream。Python 回调进入时，PyTorch 当前
CUDA stream 已切换为对应的外部 stream，因此回调中的普通 PyTorch 操作会按算法依赖
顺序排入正确 stream。

自定义策略应遵守：

1. 不调用 `torch.cuda.synchronize()`，除非确实需要 host 读取；
2. 不把输出放到另一张 GPU；
3. 不把输出移到 CPU；
4. 尽量复用 Tensor/workspace，避免每代分配大型临时量；
5. 不在回调中切换默认 CUDA device；
6. 返回前无需手动同步，stream 顺序会保证后续核心操作可见结果。

Python 回调每代会进入解释器，性能低于纯 CUDA 内置策略。通常将复杂计算写成少量批量
PyTorch 操作，而不是 Python 个体循环。

## 20. 数据保存

```python
algorithm = cm.NSGA3(
    problem=cm.DTLZ2(12, 3),
    save_data="output/nsga3_run",
    save_interval=10,
)
```

generation 0、最终代和间隔代会保存。快照包含 variables
为 `(N,D)` row-major，objectives 和 references 在文件中为 objective-major。

保存会同步 CUDA streams 并复制到 host，高频保存可能明显降低吞吐。

## 21. 完整组合示例

```python
import torch
import cuda_moea as cm


class NeuralProblem(cm.PythonProblem):
    def __init__(self, model):
        super().__init__(32, 3, lower_bounds=-1.0, upper_bounds=1.0)
        self.model = model.eval()

    def evaluate(self, x, context):
        with torch.no_grad():
            objectives = self.model(x)
        return objectives


model = torch.nn.Sequential(
    torch.nn.Linear(32, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 3),
).cuda()

algorithm = cm.NSGA3(
    population_size=2048,
    max_generations=300,
    problem=NeuralProblem(model),
    mating=cm.TournamentMating(),
    crossover=cm.SBX(eta_initial=30.0, eta_final=20.0),
    mutation=cm.PolynomialMutation(
        eta_initial=20.0,
        eta_final=30.0,
        probability=1.0,
    ),
    reference_directions=cm.DasDennisDirections(),
    environment_selector=cm.NSGA3EnvironmentSelector(
        sparse_ratio=0.5,
        feasibility_epsilon=1e-6,
    ),
    device="cuda:0",
    seed=2887,
    print_progress=False,
)

result = algorithm.run(copy=True)
torch.save({
    "variables": result.variables.cpu(),
    "objectives": result.objectives.cpu(),
    "constraints": result.constraints.cpu(),
}, "result.pt")
```

## 22. 异常与排查

### 扩展导入失败

确认扩展是在当前 Python/PyTorch 环境中构建，并检查 `torch.version.cuda`、CUDA Toolkit、
编译器 ABI。不要直接复制其他环境生成的 `.so`。

### `must be on the algorithm CUDA device`

自定义回调返回了 CPU Tensor 或另一张 GPU 上的 Tensor。使用输入 Tensor 的 `.device`
创建输出最稳妥。

### shape 错误

- problem objectives：`(N,M)`；
- problem constraints：`N` 个值；
- mating indices：`N` 个值；
- crossover/mutation variables：`(N,D)`；
- reference directions：`(K,M)`；
- environment selected indices：`N` 个值。

### 读取 population 很慢

`algorithm.population` 会同步内部 streams。减少逐代 host 观测，或只在需要的代读取。

### 结果随版本或 GPU 有差异

固定 `seed`，保持相同 problem、种群、CUDA 架构和 PyTorch 版本，并使用保存快照进行逐代
比较。浮点归约顺序可能造成微小数值差异。

## 23. 公共符号速查

| 分类 | Python 符号 |
|---|---|
| 算法 | `Algorithm`, `NSGA3`, `RVEA`, `AlgorithmBuilder` |
| 结果 | `Population`, `Result` |
| CUDA | `CudaConfig` |
| problem | `PythonProblem`, `DTLZProblem`, `DTLZ1`–`DTLZ7`, `ConvexDTLZ2`, `C1DTLZ1`, `C1DTLZ3`, `C2DTLZ2`, `C2ConvexDTLZ2`, `C3DTLZ1`, `C3DTLZ4`, `CSDP` |
| 交配 | `RandomMating`, `TournamentMating`, `PythonMating` |
| 交叉 | `SBX`, `PythonCrossover` |
| 变异 | `PolynomialMutation`, `NoMutation`, `PythonMutation` |
| 参考方向 | `DasDennisDirections`, `AdaptiveRVEADirections`, `UserDefinedDirections`, `PythonReferenceDirections` |
| 环境选择 | `NSGA3EnvironmentSelector`, `RVEAEnvironmentSelector`, `PythonEnvironmentSelector` |
