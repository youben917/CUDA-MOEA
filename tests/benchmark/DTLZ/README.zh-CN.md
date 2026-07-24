# DTLZ A/B/C

[English](README.md) | 中文

入口分别为 `run_quality.py`、`run_population_timing.py`、`run_dimension_timing.py`。完整执行顺序、断点续跑和校验命令见上级 `README.zh-CN.md`。

正式规模：A=960、B=320、C=440。B 的名义种群规模为 256、512、1024、2048、4096、8192、16384、32768；A 保存最终目标矩阵；B/C 只保存不含初始化、指标和写盘的代循环 CUDA event 时间。

完成 A/B/C 后生成全部派生数据、统计检验、图和报告：

```bash
MPLCONFIGDIR=/tmp/matplotlib-cuda-moea \
python tests/benchmark/DTLZ/analyze_results.py \
  --runs-root output/paper_benchmark_20260707 \
  --output tests/benchmark/DTLZ/results
```

原始运行记录继续保存在 `output/`；最终报告、派生表和图表统一保存在
`tests/benchmark/DTLZ/results/`。
