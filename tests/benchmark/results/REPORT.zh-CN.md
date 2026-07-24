# CUDA-MOEA 与 EvoX 综合测试报告

[English](REPORT.md) | 中文

生成日期：2026-07-15  
覆盖范围：DTLZ A/B/C 组与 MoRobtrol D 组  
详细子报告：[DTLZ](../DTLZ/results/REPORT.zh-CN.md) · [MoRobtrol](../MoRobtrol/results/REPORT.zh-CN.md)

## 1. 执行摘要

- **计算性能：CUDA-MOEA 在 DTLZ 的全部可比较配置上更快。** A 组逐问题中位加速比为 NSGA-III 5.79–12.42×、RVEA 12.08–12.73×；扩大种群后最高分别达到 271.92× 和 247.26×。EvoX–RVEA 在最大种群 `N=32768` 的 10 次运行全部 OOM，而 CUDA-MOEA 10/10 成功。
- **DTLZ 解质量：不存在单一框架全面占优。** CUDA-MOEA 的中位 IGD 在 NSGA-III 的 5/8 个问题、RVEA 的 4/8 个问题上更低；其中 CUDA-MOEA 的优势经 Holm 校正后分别有 2 个和 4 个问题显著。其余问题需结合绝对差、IQR、显著性和代表前沿判断。
- **MoRobtrol / NSGA-III：CUDA-MOEA 的时间效率总体更高。** Hopper 中 CUDA-MOEA 的最终质量更高且达到 EvoX 最终中位目标更快；Swimmer 最终质量近似持平，但在共同时间预算下 CUDA-MOEA 的 HV/EU 更高。
- **MoRobtrol / RVEA：结论依赖环境。** Hopper 后程 EvoX 的 HV/EU 更高；Swimmer 则由 CUDA-MOEA 在最终质量和等时质量上显著领先，说明不能把单一环境的结果外推到全部控制任务。
- **MoRobtrol 的速度优势受仿真成本稀释。** 策略在仿真环境中的评估占每代耗时较大，且是两框架共同承担的成本；只有种群规模增大、进化算子的并行计算占比上升后，CUDA-MOEA 的算法速度优势才更明显。环境仿真越耗时，共同评估成本占比越高，整体速度优势通常越小。
- **工程判断：** 若重点是大种群、高吞吐或受显存约束的优化，结果强烈支持 CUDA-MOEA；若重点是最终解质量，应按“问题 × 算法”选择实现，并保留多 seed 统计与等时质量比较。

## 2. 测试范围与统计口径

| 测试组 | 目的 | 规模 | 主要指标 | 汇总方式 |
|---|---|---:|---|---|
| DTLZ A | 8 个解析测试问题的质量与时间 | 960 次，全部成功 | 最终 IGD、每代时间 | 每实现每问题 30 个 seed；mean/std 与 median/IQR |
| DTLZ B | 种群规模扩展 | 320 次，310 成功、10 OOM | 每代时间、加速比 | 每配置 10 次；中位数与 10,000 次 bootstrap 95% CI |
| DTLZ C | 决策维数扩展 | 440 次，全部成功 | 每代时间、加速比 | 每配置 10 次；中位数与 10,000 次 bootstrap 95% CI |
| MoRobtrol D | 控制任务的收敛质量与时间效率 | 2 环境 × 2 算法 × 2 框架 × 10 seed | HV、EU、Quality-at-time、Time-to-target | 曲线为中位数，阴影/表格为 IQR |

DTLZ 的 IGD 越小越好；MoRobtrol 的 HV 与 EU 越大越好。DTLZ 时间为初始化完成后的代循环 CUDA-MOEA event 时间；MoRobtrol 时间为检查点累计算法时间。两者都不应解释为包含数据准备、独立评估、指标计算、写盘和绘图的端到端墙钟时间。

MoRobtrol 的“算法时间”包含优化期间与仿真环境交互得到策略回报的成本，但不包含优化完成后的独立评估。可将每代耗时概念性地写为 `T_total = T_simulation + T_evolution`：CUDA-MOEA 主要降低后者。当 `T_simulation` 占主导时，即使进化算子本身显著加速，总耗时加速比也会接近 1；扩大种群会提高 `T_evolution` 的占比，更有利于显现 CUDA-MOEA 的并行优势。更复杂、更耗时的仿真环境则会提高 `T_simulation` 占比，从而缩小可观察到的整体速度优势。

### 2.1 统一图表配色

DTLZ 与 MoRobtrol 现在使用同一套实现配色：

| 实现 | 颜色 | 图形编码 |
|---|---|---|
| CUDA-MOEA–NSGA-III | <span style="color:#8CAFCF">■ `#8CAFCF`</span> | 圆点、实线 |
| EvoX–NSGA-III | <span style="color:#EA8675">■ `#EA8675`</span> | 圆点、实线 |
| CUDA-MOEA–RVEA | <span style="color:#BDB8D9">■ `#BDB8D9`</span> | 方点、虚线 |
| EvoX–RVEA | <span style="color:#F5C184">■ `#F5C184`</span> | 方点、虚线 |

理论 Pareto 前沿使用中性灰，避免与任一实现混淆。

## 3. DTLZ A 组：质量与基础性能

### 3.1 解质量

| 算法 | CUDA-MOEA 中位 IGD 更低 | EvoX 中位 IGD 更低 | CUDA-MOEA 显著更低 | EvoX 显著更低 | 不显著 |
|---|---:|---:|---:|---:|---:|
| NSGA-III | 5/8 | 3/8 | 2/8 | 3/8 | 3/8 |
| RVEA | 4/8 | 4/8 | 4/8 | 3/8 | 1/8 |

NSGA-III 中，CUDA-MOEA 在 DTLZ1、DTLZ4 上的优势经 Holm 校正后显著；EvoX 在 DTLZ5、DTLZ6、ConvexDTLZ2 上显著更低。DTLZ2、DTLZ3、DTLZ7 的差异不显著。RVEA 中，CUDA-MOEA 在 DTLZ1、DTLZ3、DTLZ5、DTLZ7 显著更低；EvoX 在 DTLZ2、DTLZ4、DTLZ6 显著更低；ConvexDTLZ2 不显著。

![DTLZ IGD 分布](../DTLZ/results/images/igd_distributions.png)

箱线图使用完整 30 次运行计算并采用对数纵轴。统计显著不等于工程差异一定很大：例如部分问题的两套中位 IGD 十分接近，应同时查看绝对差和代表前沿；DTLZ1、DTLZ3 的 RVEA 差异则更明显。

### 3.2 每代时间

| 算法 | 8 个问题加速比范围 | 结论 |
|---|---:|---|
| NSGA-III | 5.79–12.42× | CUDA-MOEA 在 8/8 个问题更快 |
| RVEA | 12.08–12.73× | CUDA-MOEA 在 8/8 个问题更快 |

![DTLZ A 组每代时间](../DTLZ/results/images/quality_mean_generation_time.png)

加速比定义为 `median(EvoX) / median(CUDA-MOEA)`，仅在相同算法内比较；NSGA-III 与 RVEA 的柱高不能用于推导跨算法加速比。

### 3.3 代表前沿

代表运行按各“问题–算法–框架”的 IGD 最接近 30-seed 中位数选择，而非挑选最好一次。同一问题共享坐标范围、理论前沿和观察视角。

<p><img src="../DTLZ/results/images/fronts/dtlz1_fronts.png" alt="DTLZ1 fronts" width="49%"> <img src="../DTLZ/results/images/fronts/dtlz3_fronts.png" alt="DTLZ3 fronts" width="49%"></p>

其余 6 个问题的前沿图及逐问题完整统计见 [DTLZ 子报告](../DTLZ/results/REPORT.zh-CN.md)。

## 4. DTLZ B/C 组：扩展性

### 4.1 种群规模扩展

| 算法 | 最小规模 `N=256` | 最大可比规模 | 最大规模加速比 | 最大规模状态 |
|---|---:|---:|---:|---|
| NSGA-III | 5.23× | `N=32768` | 271.92× | CUDA-MOEA/EvoX 均 10/10 成功 |
| RVEA | 12.49× | `N=16384` | 247.26× | `N=32768` 时 CUDA-MOEA 10/10 成功、EvoX 0/10（OOM） |

<p><img src="../DTLZ/results/images/population_generation_time.png" alt="Population generation time" width="49%"> <img src="../DTLZ/results/images/population_speedup.png" alt="Population speedup" width="49%"></p>

小规模时框架固定开销占比较高；超过 `N=2048` 后，EvoX 每代时间增长明显快于 CUDA-MOEA。RVEA 的末端缺点表示 EvoX 无成功样本，不是零时间。

### 4.2 决策维数扩展

| 算法 | 加速比最低点 | `D=131072` 加速比 | 全范围结论 |
|---|---:|---:|---|
| NSGA-III | 4.73×（`D=8192`） | 10.12× | 全部维数 CUDA-MOEA 更快 |
| RVEA | 7.81×（`D=8192`） | 10.27× | 全部维数 CUDA-MOEA 更快 |

<p><img src="../DTLZ/results/images/dimension_generation_time.png" alt="Dimension generation time" width="49%"> <img src="../DTLZ/results/images/dimension_speedup.png" alt="Dimension speedup" width="49%"></p>

两种算法都呈现加速比先下降、后回升的趋势：中等维数时固定开销被摊薄，而高维计算量主导后 CUDA-MOEA 的并行扩展优势再次扩大。

## 5. MoRobtrol D 组：控制任务质量与时间

### 5.1 第 100 代最终质量与时间

| 环境 / 算法 | CUDA-MOEA HV / EU | EvoX HV / EU | 100 代中位时间（CUDA-MOEA / EvoX） | 判断 |
|---|---:|---:|---:|---|
| Hopper / NSGA-III | 4.846e9 / 0.4906 | 4.538e9 / 0.4856 | 9.99 / 11.65 min | CUDA-MOEA 质量更高且更快 |
| Hopper / RVEA | 1.229e10 / 0.6995 | 1.318e10 / 0.7051 | 11.17 / 11.46 min | EvoX 后程质量更高，时间接近 |
| Swimmer / NSGA-III | 33.87 / 0.74095 | 34.06 / 0.74092 | 11.59 / 19.19 min | 最终质量近似持平，CUDA-MOEA 更快 |
| Swimmer / RVEA | 11.43 / 0.5163 | 0.7765 / 0.05566 | 11.57 / 11.83 min | CUDA-MOEA 质量显著更高，时间接近 |

### 5.2 相同时间预算下的质量

| 环境 / 算法 | 共同预算 | 等时质量结论 |
|---|---|---|
| Hopper / NSGA-III | 2.93、5.85、8.75 min | CUDA-MOEA HV 高 9.5%、7.4%、13.3%；EU 高 3.4%、2.0%、3.8% |
| Hopper / RVEA | 2.87、5.73、8.59 min | 2.87 min 时 CUDA-MOEA 领先；后两个预算 EvoX HV 高 13.5%、14.7%，EU 高 7.0%、5.4% |
| Swimmer / NSGA-III | 4.84、9.64 min | CUDA-MOEA HV 高 10.1%、7.7%；EU 高 4.3%、1.7% |
| Swimmer / RVEA | 2.96、5.92、8.88 min | CUDA-MOEA HV 为 26.6×、17.2×、15.1×；EU 为 9.4×、8.3×、9.1× |

等时比较比单纯比较第 100 代更接近固定计算预算下的工程选择。Hopper/RVEA 还展示了结论随预算变化：早期 CUDA-MOEA 略优，后程由 EvoX 反超。

### 5.3 收敛与时间质量曲线

#### Hopper：按代数的收敛质量

![Hopper Generation–HV/EU](../MoRobtrol/results/images/mo_hopper_m3_generation.png)

#### Hopper：按算法时间的收敛质量

![Hopper Time–HV/EU](../MoRobtrol/results/images/mo_hopper_m3_time.png)

#### Swimmer：按代数的收敛质量

![Swimmer Generation–HV/EU](../MoRobtrol/results/images/mo_swimmer_generation.png)

#### Swimmer：按算法时间的收敛质量

![Swimmer Time–HV/EU](../MoRobtrol/results/images/mo_swimmer_time.png)

四张曲线图均独占一行。按代数曲线用于比较相同迭代预算下的收敛过程；按时间曲线用于比较相同算法时间下的质量。

### 5.4 最终非支配前沿

<p><img src="../MoRobtrol/results/images/mo_hopper_m3_final_front.png" alt="Hopper final nondominated front" width="49%"> <img src="../MoRobtrol/results/images/mo_swimmer_final_front.png" alt="Swimmer final nondominated front" width="49%"></p>

前沿图使用第 100 代 HV 最接近各实现 10-seed 中位数的代表运行，用于展示典型覆盖形态，不替代完整 seed 统计。

### 5.5 达标时间与解释限制

- Hopper/NSGA-III 达到 EvoX 最终中位 HV/EU 时，CUDA-MOEA 的达到率为 90%/80%，EvoX 为 70%/60%；仅在实际达到目标的 seed 中，CUDA-MOEA 时间中位数加速为 1.59×/1.52×。
- Hopper/RVEA 的 CUDA-MOEA 没有 seed 达到 EvoX 最终中位 HV，因此不能用运行时间略短推导同质量加速。
- Swimmer/RVEA 以 EvoX 最终中位数设定的目标过低，部分目标在初始检查点即可达到；0 min 只说明目标区分度不足，不表示无限加速。

完整收敛曲线、最终前沿以及 Quality-at-time/Time-to-target 表见 [MoRobtrol 子报告](../MoRobtrol/results/REPORT.zh-CN.md)。

## 6. 跨测试综合判断

### 6.1 可支持的结论

1. CUDA-MOEA 的核心优势是计算吞吐与扩展性：在 DTLZ 全部可比较时间配置中更快，且优势随大种群显著扩大。
2. 速度优势没有以系统性牺牲 DTLZ 解质量为代价，但也不能据此声称 CUDA-MOEA 在所有问题上质量更高；质量优胜随问题和算法变化。
3. 在已完成的 MoRobtrol 环境中，CUDA-MOEA–NSGA-III 的等时质量一致占优；RVEA 则对环境更敏感。
4. MoRobtrol 的整体速度是仿真评估与进化计算的共同结果；当前结果不能单独视为进化算子的纯性能对比。大种群更能显现 CUDA-MOEA 优势，而更耗时的环境会稀释该优势。
5. 固定代数、固定时间和固定质量目标回答的是不同问题，报告决策时应同时保留三种口径。

### 6.2 不能外推的结论

- MoRobtrol 当前只覆盖 Hopper 与 Swimmer，不能代表计划中的全部控制环境。
- DTLZ 与 MoRobtrol 的计时边界不同，不能直接把 DTLZ 的百倍级加速套用到强化学习控制任务。
- MoRobtrol 不同环境的仿真成本不同，因此某一环境测得的总耗时加速比不能直接外推到更复杂或更耗时的环境。
- 代表前沿只用于展示典型形态，不能替代全部 seed 的统计。
- DTLZ 的显著性检验是非配对 rank-sum；显著性受样本量影响，仍需结合效应方向和绝对差。

## 7. 建议

- 大种群或显存敏感任务优先采用 CUDA-MOEA，并在目标 GPU 上复核实际吞吐与峰值显存。
- 最终质量关键时按具体问题选择实现：先比较等时质量，再检查最终质量、IQR、达到率和代表前沿。
- 扩充 MoRobtrol 至计划中的其余环境后再给出跨环境总体排名；同时保留当前共同时间范围约束。
- 对 Swimmer/RVEA 重新设置更有区分度的 Time-to-target 目标，避免初始检查点即达标。

## 8. 数据、复现与文件索引

| 内容 | 路径 |
|---|---|
| DTLZ 完整报告 | [`DTLZ/results/REPORT.zh-CN.md`](../DTLZ/results/REPORT.zh-CN.md) |
| DTLZ 派生数据 | [`DTLZ/results/data/`](../DTLZ/results/data/) |
| DTLZ 分析与报告脚本 | [`DTLZ/analyze_results.py`](../DTLZ/analyze_results.py) |
| MoRobtrol 完整报告 | [`MoRobtrol/results/REPORT.zh-CN.md`](../MoRobtrol/results/REPORT.zh-CN.md) |
| MoRobtrol 派生数据 | [`MoRobtrol/results/data/`](../MoRobtrol/results/data/) |
| MoRobtrol 报告生成脚本 | [`MoRobtrol/generate_quality_report.py`](../MoRobtrol/generate_quality_report.py) |

DTLZ 本轮基线提交为 `2dc367fda14d5b2860e57ad5d0a6a6e2e7d32088`，设备为 `cuda:1`，GPU 为 NVIDIA RTX PRO 6000 Blackwell Workstation Edition；软件版本为 CUDA-MOEA 0.1.0、EvoX 1.3.0、PyTorch 2.12.0。完整数值应以各子报告的 CSV 为准，图片用于辅助理解，不应从像素反推数据。
