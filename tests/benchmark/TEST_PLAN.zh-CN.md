# CUDA-MOEA 与 EvoX 最终测试方案

[English](TEST_PLAN.md) | 中文

## 1. 测试目的

比较 CUDA-MOEA 与 EvoX 1.3.0 中 NSGA-III、RVEA 的运行速度和解质量。正式比较对象为：

| 标识 | 框架 | 算法 |
|---|---|---|
| CUDA-MOEA-NSGA-III | CUDA-MOEA | NSGA-III |
| EvoX-NSGA-III | EvoX | NSGA-III |
| CUDA-MOEA-RVEA | CUDA-MOEA | RVEA |
| EvoX-RVEA | EvoX | RVEA |

加速比只在相同算法之间计算：

$$
Speedup=\frac{T_{EvoX}}{T_{CUDA\_MOEA}}。
$$

`Speedup > 1` 表示 CUDA-MOEA 更快。NSGA-III 与 RVEA 之间只比较解质量，不计算框架加速比。

测试包括：

1. 8 个无约束 DTLZ 问题：IGD、最终前沿、平均每代时间；
2. DTLZ 随种群规模变化的加速性能；
3. DTLZ 随决策维数变化的加速性能；
4. 9 个 MoRobtrol 环境：HV、EU、最终前沿、Generation–HV/EU、Time–HV/EU、Quality-at-time 和 Time-to-target。


## 2. 两端实现与统一参数

### 2.1 已确认的实现差异

- CUDA-MOEA 的 `run()` 在初始化完成后，用 CUDA event 计量完整代循环，并通过 `result.total_ms` 返回总运行时间；
- EvoX 使用 `StdWorkflow` 驱动算法。EvoX 没有与 `result.total_ms` 对应的字段，因此测试脚本应在 `workflow.init_step()` 完成后，用 `torch.cuda.Event` 包住后续全部 `workflow.step()`；
- EvoX NSGA-III 和 RVEA 使用 `uniform_sampling(pop_size, n_objs)` 生成参考向量；EvoX RVEA 会将实际种群规模改成生成的参考向量数量；
- CUDA-MOEA 默认 SBX 分布指数为 30，而 EvoX 1.3.0 默认值为 20；
- 两端多项式变异的分布指数均为 20，`probability/pro_m=1` 均表示每个决策变量以 `1/D` 的概率变异；
- 两端 NSGA-III 均使用二元锦标赛，RVEA 均使用随机交配；
- 两端 RVEA 的 `alpha=2.0`、参考向量更新频率 `fr=0.1`。

### 2.2 正式统一设置

| 项目 | 设置 |
|---|---|
| 数值类型 | `float32` |
| 优化方向 | DTLZ 最小化；MoRobtrol 内部最小化负 reward |
| SBX | `probability/pro_c=1.0`，分布指数 30；逐变量 0.5 概率直接复制亲本值 |
| 多项式变异 | `probability/pro_m=1.0`，分布指数 20 |
| NSGA-III 交配 | 二元锦标赛 |
| RVEA 交配 | 随机交配 |
| RVEA | `alpha=2.0`，`fr=0.1`，`max_gen=正式代数` |
| CUDA-MOEA NSGA-III | `sparse_ratio=1.0`，使用完整非支配排序 |
| 设备 | 同一块独占 GPU，所有任务串行执行 |

EvoX 的 SBX 必须通过包装函数显式调用 `simulated_binary(x, pro_c=1.0, dis_c=30.0)`，不能使用其默认 `dis_c=20`。EvoX 多项式变异显式调用 `polynomial_mutation(x, lb, ub, pro_m=1.0, dis_m=20.0)`。

每个配置都必须记录请求种群规模和实际种群规模。若两端实际规模不一致，该配置不得计算加速比或配对解质量统计。

### 2.3 三目标种群规模

三目标实验使用以下标准 2 次幂作为名义种群规模：

`P_nominal ∈ {256, 512, 1024, 2048, 4096, 8192, 16384, 32768}`。

EvoX NSGA-III 保持请求种群规模不变，因此两端 NSGA-III 均直接使用 `P_nominal`。EvoX RVEA 会把种群规模改成 `uniform_sampling(P_nominal, 3)` 生成的参考向量数；为保证 RVEA 跨框架公平，CUDA-MOEA-RVEA 使用相同的有效种群规模 `P_effective`。

| 名义规模 `P_nominal` | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NSGA-III 实际规模 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
| RVEA 实际规模 `P_effective` | 253 | 496 | 990 | 2016 | 4095 | 8128 | 16290 | 32640 |

图表横轴使用名义规模；原始数据同时保存名义、请求和实际规模。加速比只比较同一算法在相同实际规模下的两种框架，不直接比较 NSGA-III 与 RVEA 的时间。

## 3. 通用执行规范

- 固定 GPU、CPU、驱动、CUDA、Python、PyTorch、EvoX、JAX、Brax、Evomo 和 CUDA-MOEA commit；
- 所有正式任务在同一环境与同一设备上串行运行；
- 四种实现的运行顺序按重复编号轮换，避免固定先后顺序；
- 相同配置使用相同整数种子集合；交叉、变异、平局处理等随机流仍由各框架独立生成，但初始种群按下述协议完全相同；
- A--D 组从同一 CPU `torch.Generator(seed)` 生成 `(N,D)` float32 初始种群并交给两端；NSGA-III 还共享 EvoX `uniform_sampling` 生成的参考方向集。EvoX 对方向顺序的代内重排仅用于平局处理；
- MoRobtrol 两端以相同 `seed` 构造 `jax.random.PRNGKey`，训练阶段 `rotate_key=True`，并在原始结果中记录训练 key 种子；
- 正式运行前，每个实验类型执行 1 个种子、5 代的 smoke test；
- 保存所有失败、超时、OOM 和非有限值记录，不静默缩小单个实现的配置；
- 原始目标值或 reward 必须保存，使 IGD、HV、EU 和图片可在算法结束后独立生成；
- 指标计算和绘图脚本与算法运行脚本分开。

## 4. 统一计时方法

### 4.1 CUDA-MOEA

1. 构造算法对象，启用已有 warm-up；
2. 关闭 `save_data` 和逐代输出；
3. 调用 `algorithm.run()`；
4. 读取 `result.total_ms`；
5. 计算：

$$
mean\_generation\_ms=\frac{total\_time\_ms}{generations}。
$$

不得在 Python 外层重新计时替代 `result.total_ms`。

### 4.2 EvoX

1. 构造算法、问题和不保存历史的 `StdWorkflow`；
2. 调用一次 `workflow.init_step()`，完成初始种群评价；
3. 正式计时前，用同一配置另建实例完成至少一次完整 `workflow.step()`，触发惰性编译和 warm-up；
4. 在正式实例的代循环前后记录 `torch.cuda.Event`；
5. 执行与 CUDA-MOEA 相同数量的 `workflow.step()`；
6. 终点 event 后调用 `torch.cuda.synchronize()`，读取 `total_time_ms`；
7. 用相同公式计算 `mean_generation_ms`。

两端计时都从初始种群已经评价完成后开始，包含每一代的交配、交叉、变异、问题评价和环境选择。两端都不包含对象构造、首次编译、初始种群评价、IGD/HV/EU、写盘和绘图。

质量实验可以记录同一次算法运行的 `total_time_ms`。D 组优化阶段每 5 代保存一次种群检查点，并记录每个 5 代区间的算法耗时；检查点写盘必须发生在该区间计时结束之后，不计入算法时间。算法完成后记录 `total_time_ms` 和各检查点累计时间，再由独立脚本读取检查点计算 reward。

## 5. 实验分组

全部正式运行划分为 A–D 四组。实验组只负责执行算法并保存原始结果；IGD、HV、EU、统计分析和绘图均在实验结束后离线完成。

### 5.1 DTLZ 测试问题

| 问题 | 决策维数 | 特性 |
|---|---:|---|
| DTLZ1 | 7 | 线性、多峰前沿 |
| DTLZ2 | 12 | 球面前沿 |
| DTLZ3 | 12 | 强多峰 |
| DTLZ4 | 12 | 分布偏置 |
| DTLZ5 | 12 | 退化前沿 |
| DTLZ6 | 12 | 退化前沿 |
| DTLZ7 | 12 | 不连续前沿 |
| ConvexDTLZ2 | 12 | 凸前沿 |

全部为无约束、3 目标、变量范围 `[0,1]`。正式测试前，在固定随机决策矩阵上验证 CUDA-MOEA 与 EvoX 问题函数输出误差不超过 `1e-5`。

### 5.2 A 组：DTLZ 解质量与平均时间

| 参数 | 取值 |
|---|---|
| 实现 | 四个框架–算法组合 |
| 问题 | 全部 8 个问题 |
| 名义种群规模 | 1024；NSGA-III 实际为 1024，RVEA 实际为 990 |
| 代数 | 500 |
| 独立种子 | 30 个：`0–29` |
| 保存内容 | 最终目标矩阵、总运行时间、运行状态 |

总运行数：`4 × 8 × 30 = 960`。

本组只运行算法并保存原始结果。每个“问题–实现”后续汇总：

- IGD：`mean ± std`、`median [IQR]`；
- `total_time_ms`：`mean ± std`；
- `mean_generation_ms`：先对每次运行计算，再报告 `mean ± std`。

时间统计不包括 IGD、写盘和绘图，因为这些操作均在算法计时结束后执行。IGD 按第 6.1 节计算。

### 5.3 B 组：DTLZ 随种群规模变化的加速性能

| 参数 | 取值 |
|---|---|
| 问题 | DTLZ1，无约束，3 目标 |
| 决策维数 | 500 |
| 名义种群规模 | 256、512、1024、2048、4096、8192、16384、32768 |
| 代数 | 100 |
| 独立计时重复 | 10 |
| 指标 | `total_time_ms`、`mean_generation_ms`、Speedup |

总运行数：`4 × 8 × 10 = 320`。对每个种群规模和算法，用两框架 10 次时间的中位数计算主加速比，同时报告加速比的 95% bootstrap 置信区间。

本组保存每次计时记录和 OOM/失败状态；曲线按第 7.1 节生成。

### 5.4 C 组：DTLZ 随决策维数变化的加速性能

| 参数 | 取值 |
|---|---|
| 问题 | DTLZ1，无约束，3 目标 |
| 名义种群规模 | 1024；NSGA-III 实际为 1024，RVEA 实际为 990 |
| 决策维数 | 128、256、512、1024、2048、4096、8192、16384、32768、65536、131072 |
| 代数 | 100 |
| 独立计时重复 | 10 |
| 指标 | `total_time_ms`、`mean_generation_ms`、Speedup |

总运行数：`4 × 11 × 10 = 440`。本组保存每次计时记录和 OOM/失败状态；曲线按第 7.1 节生成。

### 5.5 MoRobtrol 测试环境

| 环境 | 目标数 | MLP 结构 | 参数量 |
|---|---:|---|---:|
| MoHalfCheetah | 2 | 17–16–6 | 390 |
| MoHopperM3 | 3 | 11–16–3 | 243 |
| MoHumanoid | 2 | 244–16–17 | 4,209 |
| MoHumanoidStandup | 2 | 244–16–17 | 4,209 |
| MoInvertedDoublePendulum | 2 | 8–16–1 | 161 |
| MoPusher | 3 | 23–16–7 | 503 |
| MoReacher | 2 | 11–16–2 | 226 |
| MoSwimmer | 2 | 8–16–2 | 178 |
| MoWalker2d | 2 | 17–16–6 | 390 |

策略网络统一为 `Linear(observation, 16) → Tanh → Linear(16, action) → Tanh`。参数量包含两层权重和偏置：

$$
L=(observation\times16+16)+(16\times action+action)。
$$

两端必须使用仓库现有的同一 MoRobtrol/EVOMO 评价代码、相同 `RobotPolicy`、参数范围 `[-5,5]`、episode 长度 1000、训练 episode 数 2。优化时对 reward 取负，指标和图中恢复为 reward maximization。

### 5.6 D 组：MoRobtrol 解质量

| 参数 | 取值 |
|---|---|
| 实现 | 四个框架–算法组合 |
| 环境 | 全部 9 个环境 |
| 名义种群规模 | 16384；2 目标环境实际为 16384，3 目标环境中 RVEA 实际为 16290 |
| 代数 | 100 |
| 优化种子 | `0–9` |
| 独立评估 | 固定种子 90210，每个策略 10 个 episode |
| 保存检查点 | 第 0、5、10、……、100 代 |
| 区间计时 | 记录 `(0,5]`、`(5,10]`、……、`(95,100]` 每个 5 代区间耗时 |

总运行数：`4 × 9 × 10 = 360`。

本组用于离线计算 HV、EU、收敛曲线和时间质量表，不单独设置 MoRobtrol 速度实验。优化阶段每 5 代输出一次进度，并保存第 0、5、10、……、100 代的种群决策变量；100 代完成后记录算法 `total_time_ms`、`mean_generation_ms`、20 个 5 代区间耗时和 21 个检查点累计时间。第 0 代累计时间记为 0，第 `g` 代累计时间为到该检查点为止的区间耗时之和。随后按环境运行独立评估脚本，每个检查点使用相同的独立评估随机键集合生成 reward 矩阵。训练随机键按代更新，独立评估随机键固定且不参与优化。评估成功后仍保留种群检查点，以便调整种子或 episode 数后重新评估。指标按第 6.2、6.3、6.4 节计算。

### 5.7 实验组汇总

| 组别 | 数据集 | 目的 | 核心输出 |
|---|---|---|---|
| A | 8 个 DTLZ | 解质量与平均时间 | 最终目标矩阵、IGD、运行时间 |
| B | DTLZ1 | 种群规模扩展 | 不同种群规模的时间与加速比 |
| C | DTLZ1 | 决策维数扩展 | 不同维数的时间与加速比 |
| D | 9 个 MoRobtrol | 解质量与时间收敛 | checkpoint reward、checkpoint time、HV、EU、Quality-at-time、Time-to-target |

## 6. 离线指标定义

指标脚本只读取实验组保存的目标值或 reward，不启动优化算法，也不参与算法计时。

### 6.1 DTLZ：IGD

对 A 组最终目标矩阵先移除非有限行、精确去重并保留非支配解，再计算 IGD（越小越好）。

- 每个问题固定一份 20,000 点理论 Pareto 前沿，四种实现和全部种子共享；
- DTLZ1 使用理论线性前沿；DTLZ2–4 使用理论球面前沿；
- DTLZ5、DTLZ6 使用退化理论前沿；DTLZ7 使用不连续理论前沿；
- ConvexDTLZ2 使用目标变换后的理论前沿；
- 参考前沿生成种子固定为 `2887`；
- 正式运行前比较 20,000 点与 50,000 点所得 IGD。若任一问题的方法排序发生变化，该问题正式使用 50,000 点。

### 6.2 MoRobtrol：HV

- 使用 D 组保存的 reward，以 maximization 形式计算，越大越好；
- 2、3 目标均使用精确 HV；
- 每个环境使用一个固定参考点，四种实现、全部种子和检查点共享；
- 每个环境合并该环境 40 次正式运行（4 种实现 × 10 个种子）的最终 reward，构造经验非支配前沿；
- 参考点为经验前沿逐目标最差值向劣方向扩展该目标范围的 10%。

### 6.3 MoRobtrol：EU

- 使用与 HV 相同的经验前沿确定逐目标 ideal/nadir，并对 reward 归一化；
- 使用固定种子生成 10,000 个均匀单纯形权重；
- 四种实现、全部种子和检查点共享 ideal、nadir 和权重集合；
- EU 越大越好。

HV 参考点、EU ideal/nadir 和权重集合统一保存到 `metric_references.json`，生成后不得变化。每个“环境–实现”报告最终 HV、EU 的 `mean ± std` 和 `median [IQR]`。

### 6.4 MoRobtrol：从 Generation 转换到 Time

D 组每个检查点都同时拥有：

- `checkpoint_generation`：`0、5、10、……、100`；
- `checkpoint_cumulative_time_ms`：从第 1 代到该检查点的累计算法时间；
- `checkpoint_hv` 和 `checkpoint_eu`：离线独立评估后计算的指标。

因此同一份 D 组数据可同时生成 Generation–HV/EU 和 Time–HV/EU 曲线。时间曲线使用 `checkpoint_cumulative_time_ms` 作为横轴，第 0 代时间为 0。

Quality-at-time 表按“环境–算法–指标”分别生成。推荐时间预算使用 EvoX 同一算法在第 25、50、75、100 代的中位累计时间：

$$
\tau \in \{median(T^{EvoX}_{25}), median(T^{EvoX}_{50}), median(T^{EvoX}_{75}), median(T^{EvoX}_{100})\}。
$$

对每次运行，在不超过预算 `τ` 的检查点中选择最新一个检查点，报告该检查点 HV/EU；若某次运行在 `τ` 前没有任何有效检查点，则记为缺失并报告缺失数量。Quality-at-time 表按相同时间预算比较 CUDA-MOEA 与 EvoX 在同一算法上的质量。

Time-to-target 表按“环境–算法–指标”分别生成。主目标值定义为 EvoX 同一算法第 100 代最终指标的中位数；可附加报告该最终中位数的 90% 作为较低目标。对每次运行，查找首次达到或超过目标值的检查点累计时间；未达到则记为 censored，并同时报告达到率。Time-to-target 不用插值，避免把两个 checkpoint 之间的未知质量变化假设为线性。

## 7. 离线可视化

绘图脚本只读取原始数据和离线指标结果，不参与算法计时。

### 7.1 DTLZ 图表

- A 组：全部 8 个问题绘制理论前沿和四种实现最终非支配前沿；
- 代表运行选择 IGD 最接近该实现 30 次中位数的运行，不选择最好运行；
- 同一问题使用相同坐标范围、视角、点大小和理论前沿；
- 绘制最终 IGD 分布图和各问题平均每代时间图；
- B 组绘制 `Population–mean_generation_ms` 和 `Population–Speedup`；
- C 组绘制 `Dimension–mean_generation_ms` 和 `Dimension–Speedup`。

### 7.2 MoRobtrol 图表

- D 组绘制 `Generation–HV`、`Generation–EU`、`Time–HV`、`Time–EU` 和最终 HV/EU 分布图；
- 2 目标环境绘制二维最终非支配前沿，3 目标环境绘制三维前沿；
- 代表运行选择最终 HV 最接近该实现 10 次中位数的运行；
- 同一环境使用相同坐标范围、视角和点大小；坐标使用真实 reward 名称并注明越大越好；
- 经验合并前沿只能标记为经验参考前沿，不得称为理论前沿；
- Time–HV/EU 图使用 D 组记录的检查点累计算法时间，不包含独立评估、指标计算、写盘和绘图时间。

## 8. 统计分析

- 解质量以独立优化种子为样本；B、C 组专用速度实验以独立计时重复为样本；D 组 Time–HV/EU、Quality-at-time 和 Time-to-target 仍以独立优化种子为样本；
- 每个问题/环境内，CUDA-MOEA 与 EvoX 按相同算法比较；
- IGD、HV、EU 使用 Wilcoxon rank-sum 检验。只有在脚本成功注入完全相同初始种群和随机序列时才使用 signed-rank 配对检验；
- DTLZ 8 个问题和 MoRobtrol 9 个环境分别使用 Holm 校正，显著性水平 0.05；
- 同时报告 rank-biserial correlation，不以 `p≥0.05` 表示两者等价；
- B、C 组时间和加速比主要报告中位数及 95% bootstrap 置信区间，平均值和标准差作为补充。

## 9. 原始数据字段

每个运行至少保存：

| 字段 | 说明 |
|---|---|
| framework / framework_version | 框架及版本 |
| algorithm | NSGA-III 或 RVEA |
| experiment | 实验类型 |
| problem / environment | 问题或环境 |
| objectives / dimension | 目标数与决策维数 |
| nominal_population / requested_population / actual_population | 名义、传入实现的请求值与最终实际种群规模 |
| hidden_width / policy_parameters | 策略隐藏层宽度与参数量；仅 MoRobtrol 使用 |
| generations | 实际完成代数 |
| seed / repeat_index | 优化种子或计时重复编号 |
| total_time_ms | 算法返回或 CUDA event 测得的总代循环时间 |
| mean_generation_ms | `total_time_ms / generations` |
| checkpoint_generations | D 组检查点代数：`0、5、10、……、100` |
| checkpoint_interval_times_ms | D 组每 5 代区间耗时；第 0 代无区间耗时 |
| checkpoint_cumulative_times_ms | D 组检查点累计算法时间，第 0 代为 0 |
| checkpoint_populations / checkpoint_rewards | D 组种群检查点与独立评估 reward 文件；两者均保留 |
| evaluation_status / evaluation_error | D 组独立评估状态与错误信息 |
| final_igd / hv / eu | 对应实验的离线指标 |
| quality_at_time / time_to_target | D 组离线派生表字段 |
| run_status / error | 成功、超时、OOM、NaN 或其他错误 |
| git_commit / environment_manifest | 代码和环境标识 |

## 10. 脚本与执行顺序

建议脚本按以下职责拆分：

```text
tests/benchmark/
├── adapters/                 # CUDA-MOEA 与 EvoX 统一运行接口
├── DTLZ/
│   ├── run_quality.py
│   ├── run_population_timing.py
│   └── run_dimension_timing.py
├── MoRobtrol/
│   ├── run_quality.py
│   └── evaluate_quality.py
├── metrics/                  # IGD、HV、EU，仅读取运行结果
├── plots/                    # 前沿和统计图，仅读取派生数据
└── validate_results.py
```

执行顺序：

1. 冻结环境和 `manifest.json`；
2. 验证问题输出、算子参数、参考向量数量和实际种群规模；
3. 执行 smoke test；
4. 执行 B、C 组 DTLZ 专用速度实验；
5. 执行 A 组 DTLZ 解质量实验；
6. 离线生成 DTLZ 理论前沿并计算 IGD；
7. 按环境执行 D 组 MoRobtrol 优化，保存种群检查点和每 5 代区间耗时；
8. 按环境执行 D 组独立评估，同时保留 reward 和种群检查点；
9. 离线冻结 HV/EU 参考数据并计算 HV、EU；
10. 由 D 组检查点累计时间生成 Quality-at-time 和 Time-to-target 表；
11. 校验运行数量和失败记录；
12. 最后执行统计分析和绘图。

只有满足以下条件才生成最终报告：四种实现均纳入；两端实际种群规模一致；DTLZ 8 个问题和 MoRobtrol 9 个环境无未解释缺失；所有指标能从保存的目标值或 reward 复算；所有时间均不包含指标计算、写盘和绘图。
