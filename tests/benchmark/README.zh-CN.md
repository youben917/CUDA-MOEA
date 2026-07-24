# A–D 基准测试执行手册

[English](README.md) | 中文

`TEST_PLAN.zh-CN.md` 是中文参数规范。运行入口负责执行算法、保存原始矩阵、记录时间与校验完整性；离线分析脚本只读取已保存数据，用于计算 IGD/HV/EU 和统计派生表，不参与算法计时。绘图由 notebook 读取派生表完成。

## 1. 环境与预检

所有命令从项目根目录执行：

```bash
# 先激活已安装 CUDA-MOEA 与 benchmark 依赖的 Python 环境
export CUDA_VISIBLE_DEVICES=0

python -m pip show cuda-moea evox evomo torch jax brax
python tests/benchmark/validate_setup.py
python -c "import torch,jax; print(torch.cuda.get_device_name(0)); print(jax.devices())"
```

建议为本轮数据建立唯一目录：

```bash
export BENCH_ROOT="$PWD/output/benchmark_20260707"
mkdir -p "$BENCH_ROOT"
```

第一次执行任一组时会冻结 `$BENCH_ROOT/manifest.json`。不要在正式测试中途更换代码、环境或 GPU。

## 2. 5 代 smoke test

各类入口都先做 smoke；输出与正式数据分离：

```bash
export SMOKE_ROOT="$PWD/output/benchmark_20260707_smoke"

python tests/benchmark/DTLZ/run_population_timing.py --output "$SMOKE_ROOT" --device cuda:0 --smoke
python tests/benchmark/DTLZ/run_dimension_timing.py  --output "$SMOKE_ROOT" --device cuda:0 --smoke
python tests/benchmark/DTLZ/run_quality.py           --output "$SMOKE_ROOT" --device cuda:0 --smoke

python tests/benchmark/MoRobtrol/run_quality.py             --output "$SMOKE_ROOT" --device cuda:0 --smoke
python tests/benchmark/MoRobtrol/evaluate_quality.py \
  --runs-root "$SMOKE_ROOT" --experiment smoke_d --environment mo_halfcheetah --device cuda:0

python tests/benchmark/validate_smoke.py --runs-root "$SMOKE_ROOT"
```

smoke 分别写入对应临时实验目录。每个命令串行运行四种实现。成功结果可重复执行，脚本会自动跳过；失败/OOM/NaN 会保留 `run_result.json`，下次执行会重试。

## 3. 正式执行顺序

### B 组：DTLZ 种群规模测速（320 次）

```bash
python tests/benchmark/DTLZ/run_population_timing.py --output "$BENCH_ROOT" --device cuda:0
python tests/benchmark/DTLZ/run_population_timing.py --output "$BENCH_ROOT" --device cuda:0 --validate-only
```

### C 组：DTLZ 决策维数测速（440 次）

```bash
python tests/benchmark/DTLZ/run_dimension_timing.py --output "$BENCH_ROOT" --device cuda:0
python tests/benchmark/DTLZ/run_dimension_timing.py --output "$BENCH_ROOT" --device cuda:0 --validate-only
```

### A 组：DTLZ 解质量（960 次）

```bash
python tests/benchmark/DTLZ/run_quality.py --output "$BENCH_ROOT" --device cuda:0
python tests/benchmark/DTLZ/run_quality.py --output "$BENCH_ROOT" --device cuda:0 --validate-only
```

### D 组：MoRobtrol 解质量（360 次）

```bash
python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer

python tests/benchmark/MoRobtrol/evaluate_quality.py \
  --runs-root "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer

python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer --validate-only
```

D 组按环境分两阶段运行。优化脚本每 5 代输出进度并保存第 0、5、10、…、100 代种群到 `checkpoint_populations.pt`，同时记录每个 5 代区间耗时和检查点累计时间；检查点写盘不计入算法时间。独立评估脚本使用固定种子 90210、每个策略 10 episodes 生成 `checkpoint_rewards.pt`；评估后保留种群文件，方便调整评估配置后重新评估。后续可由这些检查点生成 Generation–HV/EU、Time–HV/EU、Quality-at-time 和 Time-to-target。

D 组全部环境独立评估完成后，先生成 MoRobtrol 离线指标和派生表：

```bash
python tests/benchmark/MoRobtrol/analyze_quality.py \
  --runs-root "$BENCH_ROOT" \
  --output tests/benchmark/MoRobtrol/results
```

主要输出包括：

- `data/metric_references.json`：每个环境冻结的 HV reference、EU ideal/nadir 和权重配置；
- `data/checkpoint_metrics.csv`：每个“环境–实现–seed–检查点”的 HV、EU 和累计时间；
- `data/checkpoint_metric_summary.csv`：Generation–HV/EU 和 Time–HV/EU 作图用的中位数、IQR 与时间汇总；
- `data/quality_at_time.csv`：按 EvoX 第 25/50/75/100 代中位累计时间预算生成的 Quality-at-time 表；
- `data/time_to_target.csv`：以 EvoX 第 100 代最终中位质量及其 90% 为目标的 Time-to-target 表。

图表和表格展示使用 notebook：

```bash
jupyter notebook tests/benchmark/MoRobtrol/plot_quality_curves.ipynb
```

该 notebook 读取上述 CSV 和 `checkpoint_rewards.pt`，显示 Generation–HV/EU 图、Time–HV/EU 图、最终前沿分布图、Quality-at-time 表和 Time-to-target 表，不输出图片文件或 PDF。

## 4. 中断、分环境与重跑

- 所有入口都支持 `--stop-after N`，适合先执行少量任务核查输出。
- MoRobtrol 入口支持 `--environment mo_swimmer` 或多个环境名。
- 默认跳过配置匹配的成功结果；D 组还要求 `checkpoint_populations.pt` 存在，否则自动补跑。失败记录会自动重试。
- 如确需覆盖成功任务，增加 `--rerun-successful`。
- 如希望首次错误立即停止，增加 `--fail-fast`。
- 正式任务必须串行，不要同时启动两个入口。

### 4.1 `--dry-run`：预览待执行任务

`--dry-run` 只生成并打印执行计划，不启动优化。适合在正式运行或补跑前确认待执行的 seed、框架、算法、设备和输出路径：

```bash
python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$BENCH_ROOT" \
  --device cuda:0 \
  --environment mo_swimmer \
  --dry-run
```

输出首先给出计划统计，例如：

```text
Planned=40 successful=27 pending_or_failed=13
```

随后只打印这 13 个待执行任务的完整命令。`--dry-run` 不会修改已有运行结果，但会创建或更新本组任务清单。

### 4.2 `--validate-only`：校验任务完整性

`--validate-only` 不启动优化，只检查所选范围内每个计划任务是否已有配置匹配的终态 `run_result.json`。对于成功的 D 组任务，还会检查 `checkpoint_populations.pt` 是否存在：

```bash
python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$BENCH_ROOT" \
  --device cuda:0 \
  --environment mo_swimmer \
  --validate-only
```

退出码含义：

- `0`：所有计划任务都有终态记录；终态可以是成功，也可以是已记录的 OOM、NaN、超时或失败；
- `1`：存在缺失结果、配置不匹配，或成功的 D 组任务缺少种群检查点。

`--validate-only` 只校验优化阶段；D 组独立评估是否完成由 `evaluate_quality.py` 的退出码以及最终的 `validate_results.py` 检查。

推荐补跑顺序：

```bash
# 1. 预览缺失任务
python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer --dry-run

# 2. 只执行缺失或失败任务
python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer

# 3. 校验优化阶段
python tests/benchmark/MoRobtrol/run_quality.py \
  --output "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer --validate-only

# 4. 独立评估；已成功且 reward 文件存在的任务会跳过
python tests/benchmark/MoRobtrol/evaluate_quality.py \
  --runs-root "$BENCH_ROOT" --device cuda:0 --environment mo_swimmer
```

不要同时使用 `--dry-run` 和 `--validate-only`。常规补跑也不要添加 `--rerun-successful`，否则所选范围内的成功任务会全部重跑。

## 5. 完整性校验与原始表汇总

全部四组结束后执行：

```bash
python tests/benchmark/validate_results.py --runs-root "$BENCH_ROOT"

python tests/benchmark/collect_results.py \
  --runs-root "$BENCH_ROOT" \
  --output "$BENCH_ROOT/data"
```

输出为：

- `data/raw_runs.json`：完整逐运行记录；
- `data/raw_runs.csv`：便于后续统计整理的平面表；
- A 组各运行的 `final_objectives.pt`；
- D 组的 `checkpoint_populations.pt` 与独立评估生成的 `checkpoint_rewards.pt`；
- D 组每次运行 JSON 中的 `checkpoint_generations`、`checkpoint_interval_times_ms` 和 `checkpoint_cumulative_times_ms`；
- 每组的 `group_?_manifest.jsonl` 与全局环境 `manifest.json`。

若只完成部分组，可用例如 `--groups group_b group_c` 做阶段校验。请保留整个 `$BENCH_ROOT`；后续 IGD/HV/EU、置信区间、显著性检验、图表和报告都应从这些原始文件离线生成。
