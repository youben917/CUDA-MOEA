# MoRobtrol D 组

[English](README.md) | 中文

入口为 `run_quality.py`、`evaluate_quality.py`、`analyze_quality.py` 和 `plot_quality_curves.ipynb`。完整执行顺序和命令见上级 `README.zh-CN.md`。

D 组优化完成后，按环境运行 `evaluate_quality.py`。该脚本生成 `checkpoint_rewards.pt`，同时保留 `checkpoint_populations.pt`，方便后续重新评估。

正式规模：D=360。D 组名义种群规模为 16384，隐藏层固定 16。参数范围 `[-5,5]`，训练为 2×1000 steps；独立评估固定种子 90210、10 episodes。

优化阶段保存第 0、5、10、…、100 代 `checkpoint_populations.pt`，并在 `run_result.json` 中记录 `checkpoint_generations`、每 5 代区间耗时 `checkpoint_interval_times_ms` 和检查点累计时间 `checkpoint_cumulative_times_ms`。离线 `analyze_quality.py` 会读取这些字段和独立评估 reward，只生成 HV/EU、Quality-at-time 和 Time-to-target 数据表；`plot_quality_curves.ipynb` 负责显示 Generation–HV/EU 图、Time–HV/EU 图、最终前沿分布图和两张时间质量表。

运行前可给任一优化入口增加 `--dry-run` 预览待执行任务，运行后增加 `--validate-only` 校验优化结果。D 组补跑时还会将缺少 `checkpoint_populations.pt` 的成功记录视为待执行。详细命令、退出码和推荐顺序见上级 [`README.zh-CN.md`](../README.zh-CN.md#4-中断分环境与重跑)。
