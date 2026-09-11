"""Generate CSV summaries and a Chinese report from a completed paired run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/benchmark/native/results"))
    parser.add_argument("--visualize", action="store_true", help="Add charts and export a Chinese PDF (requires matplotlib)")
    args = parser.parse_args()
    data = json.loads(args.measurements.read_text())
    if data["status"] != "complete":
        parser.error("Only completed runs can be summarized")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = data["cases"]
    evaluations = {}
    for case in cases:
        evaluations.setdefault((case["population"], case["dimension"]), case)
    with (args.output / "evaluation.csv").open("w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["population", "dimension", "pytorch_us", "native_us", "speedup", "pytorch_min_us", "pytorch_max_us", "native_min_us", "native_max_us"])
        for (n, d), c in evaluations.items():
            p, v = c["evaluation_wall_ms"]["pytorch"], c["evaluation_wall_ms"]["native"]
            writer.writerow([n, d, p["median"]*1000, v["median"]*1000, c["evaluation_speedup"], p["min"]*1000, p["max"]*1000, v["min"]*1000, v["max"]*1000])
    with (args.output / "optimization.csv").open("w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["algorithm", "population", "dimension", "pytorch_ms", "native_ms", "speedup", "pytorch_min_ms", "pytorch_max_ms", "native_min_ms", "native_max_ms"])
        for c in cases:
            p, v = c["run_wall_ms"]["pytorch"], c["run_wall_ms"]["native"]
            writer.writerow([c["algorithm"], c["population"], c["dimension"], p["median"], v["median"], c["run_speedup"], p["min"], p["max"], v["min"], v["max"]])
    protocol, env = data["protocol"], data["environment"]
    root = Path(__file__).resolve().parents[3]
    run_output = Path(protocol["output"])
    if run_output.is_relative_to(root):
        run_output = run_output.relative_to(root)
    reproduce = ["python", "tests/benchmark/native/run_benchmark.py",
                 "--populations", *map(str, protocol["populations"]),
                 "--dimensions", *map(str, protocol["dimensions"]),
                 "--algorithms", *protocol["algorithms"],
                 "--generations", str(protocol["generations"]),
                 "--samples", str(protocol["samples"]),
                 "--repeats", str(protocol["repeats"]),
                 "--warmup", str(protocol["warmup"]), "--output", str(run_output)]
    if protocol["torch_compile"]:
        reproduce.append("--torch-compile")
    eval_ratios = [c["evaluation_speedup"] for c in evaluations.values()]
    max_error = max(item["objective_max_scaled_error"] for c in cases for item in c["correctness"].values())
    lines = ["# Python/PyTorch 与原生 C++/CUDA 自定义问题性能实测", "",
             f"测试开始：`{data['started_utc']}`；完成：`{data['finished_utc']}`。", "",
             f"本次测试使用同一约束 BiSphere 问题，批量评估加速比为 **{min(eval_ratios):.2f}–{max(eval_ratios):.2f}×**。"]
    for kind in protocol["algorithms"]:
        ratios = [c["run_speedup"] for c in cases if c["algorithm"] == kind]
        lines.append(f"{kind} 完整代循环加速比为 **{min(ratios):.2f}–{max(ratios):.2f}×**。")
    lines += ["", "加速比统一为 Python 耗时 / 原生耗时；大于 1 表示原生实现更快，小于 1 表示本次测量原生更慢。", "",
              "## 环境与口径", "",
              f"- GPU：{env['gpu']}，物理 GPU `{env['visible_devices']}`，计算能力 `{env['device_capability']}`。",
              f"- PyTorch：`{env['torch']}`；CUDA：`{env['torch_cuda']}`；编译目标：`{data['native_build']['architectures']}`。",
              f"- 核心 SDK 构建标识：`{env['sdk']['build_id']}`。",
              f"- 种群：`{protocol['populations']}`；维度：`{protocol['dimensions']}`；两个目标。",
              "- 目标：`f1=sum(x²)`、`f2=sum((x-2)²)`；约束违反量：`max(f1-9, 0)`；边界 `[-5,5]`。",
              "- PyTorch 基线使用批量 eager 算子；原生版本使用单个融合 CUDA kernel，每个个体由一个 128 线程 block 处理。",
              f"- 评估：每个样本预热 {protocol['warmup']} 次，然后连续评估 {protocol['repeats']} 次，取平均单次墙钟时间；共 {protocol['samples']} 个样本，表中列中位数。",
              f"- 完整优化：每个算法先预热，然后用相同初始种群和配对随机种子运行 {protocol['generations']} 代，共 {protocol['samples']} 次，表中列中位数。",
              "- 两种实现交替随机测试顺序；完整优化计时排除编译、构造、初始化和数据写盘，包含代循环、最终同步和零拷贝结果包装。",
              "- 评估计时包含 Python 回调、Tensor 适配、GPU 算子及结果复制。它衡量完整评估路径，不单独估计 Python 回调本身。",
              f"- NSGA-III 设置 `sparse_ratio={protocol['nsga3_sparse_ratio']}`，为支配关系分配完整容量，避免默认稀疏容量溢出使结果近似。",
              "- GPU 时钟未锁定；CSV 保留最小/最大值，微小差异不应直接视为稳定优势。", "",
              "## 数值验证", "",
              "每个规模的两种实现均先对相同输入进行评估，与 float64 公式参考比较；输入包含可行点、边界点和不可行点。",
              "容差为 `rtol=2e-5, atol=2e-4`。所有已记录配置通过验证。",
              f"目标值最大缩放误差 `abs(error)/max(abs(reference),1)` 为 `{max_error:.3e}`。",
              "GPU 单元测试另覆盖非默认 stream、reset 和结果生命周期。完整进化轨迹未要求逐位一致：浮点归约顺序差异可能影响选择与最终种群。", "",
              "## 批量评估", "", "单位：μs / 次；每次评估整个种群。", "",
              "| N | D | Python/PyTorch | C++/CUDA | 加速比 |", "|---:|---:|---:|---:|---:|"]
    for (n, d), c in evaluations.items():
        lines.append(f"| {n} | {d} | {c['evaluation_wall_ms']['pytorch']['median']*1000:.2f} | {c['evaluation_wall_ms']['native']['median']*1000:.2f} | {c['evaluation_speedup']:.2f}× |")
    lines += ["", "## 完整优化", "", f"单位：ms / {protocol['generations']} 代。", "",
              "| 算法 | N | D | Python/PyTorch | C++/CUDA | 加速比 |", "|---|---:|---:|---:|---:|---:|"]
    for c in cases:
        lines.append(f"| {c['algorithm']} | {c['population']} | {c['dimension']} | {c['run_wall_ms']['pytorch']['median']:.2f} | {c['run_wall_ms']['native']['median']:.2f} | {c['run_speedup']:.2f}× |")
    hits = data["native_build"]["cache_hit_wall_ms"]
    lines += ["", "## 编译与缓存", "",
              f"- 空缓存首次编译墙钟时间：{data['native_build']['cold_wall_ms']/1000:.3f} s。",
              f"- 同源码缓存命中查询中位时间：{statistics.median(hits):.2f} ms（范围 {min(hits):.2f}–{max(hits):.2f} ms，共 {len(hits)} 次）。",
              "- 缓存查询包含源码/头文件哈希和工具身份检查，不是 GPU 评估时间；这些成本不会在每代重复。",
              "- 表中完整优化时间不包含上述启动成本。对短任务，缓存检查或首次编译可能超过代循环省下的时间；应复用已创建的问题对象，或进行足够长/足够多次的优化以摊薄成本。", "",
              "## 解释范围", "",
              "原生评估的收益同时来自融合 kernel、减少临时 Tensor、减少输出复制以及绕过 Python 回调，不能全部归因于 C++ 语言本身。",
              "完整优化还包含繁殖、排序、参考方向和环境选择，评估部分的加速不能直接作为整个优化过程的加速。",
              "本问题的约束会形成较多支配层，尤其大种群 NSGA-III 的排序成本较高；这些数据不代表所有目标/约束结构。",
              "完整代循环也可能出现原生版本更慢的配置，表中按实测保留。相同初始种群和随机种子不保证浮点实现的进化轨迹逐位一致；未进行逐代剖析，不能将反向差异直接归因于某一种开销。",
              "本结果仅代表当前双球解析问题、所测 GPU 和配置，不外推为神经网络、复杂仿真或已融合 PyTorch 实现的通用加速比。", "",
              "## 复现与数据", "", "从仓库根目录，在 CUDA-MOEA 已按目标 GPU 编译安装的环境运行：", "", "```bash",
              f"CUDA_VISIBLE_DEVICES={shlex.quote(env['visible_devices'] or '0')} PYTHONPATH=python " + shlex.join(reproduce),
              shlex.join(["python", "tests/benchmark/native/summarize.py", str(run_output / "measurements.json"), "--output", str(args.output), *(["--visualize"] if args.visualize else [])]), "```", "",
              "原始文件保存在忽略提交的 `output/` 下。已存在结果时运行器会拒绝覆盖；重复实验请使用新的输出目录。",
              f"原始文件 SHA256：`{hashlib.sha256(args.measurements.read_bytes()).hexdigest()}`。", "",
              "- [批量评估汇总](evaluation.csv)", "- [完整优化汇总](optimization.csv)",
              "- [源码校验和](source_sha256.json)", ""]
    if args.visualize:
        from visualize_report import generate_charts, export_pdf

        figures = generate_charts(data, args.output)
        insertions = {
            "## 环境与口径": [
                "![评估与完整优化加速比](figures/speedup.png)", "",
                "图 1：每格为 Python 与原生耗时中位数之比；N 为种群大小，D 为变量维度。评估与完整优化使用不同色标，两个优化算法共用色标。小于 1 的格子表示原生更慢。", "",
            ],
            "## 批量评估": [
                "![批量评估耗时](figures/evaluation.png)", "",
                "图 2：柱高为 7 个样本的中位数，误差线为实测最小值至最大值，不是置信区间。每个样本是连续 500 次评估的平均单次耗时；三个面板共用纵轴。", "",
            ],
            "## 完整优化": [
                "![完整优化耗时](figures/optimization.png)", "",
                "图 3：100 代墙钟时间；柱高为 7 次运行的中位数，误差线为实测最小值至最大值，不是置信区间。各面板纵轴独立，均从零开始；请按刻度比较绝对耗时。编译、构造与初始化不计入。", "",
            ],
        }
        expanded = []
        for line in lines:
            if line == "## 环境与口径":
                expanded.extend(insertions[line])
            expanded.append(line)
            if line in ("## 批量评估", "## 完整优化"):
                expanded.extend(["", *insertions[line]])
        lines = expanded
        lines += ["", "图表与 PDF 导出依赖 Matplotlib（本次使用 3.11.0）及本机中文字体；不需要重新运行 GPU 测试。",
                  "图表同时提供 PNG 与矢量 PDF，位于 `figures/` 目录。", ""]
    report = args.output / "REPORT.zh-CN.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    if args.visualize:
        export_pdf(report, figures)
    (args.output / "source_sha256.json").write_text(json.dumps(data["source_sha256"], indent=2) + "\n")
    print(args.output / "REPORT.zh-CN.md")


if __name__ == "__main__":
    main()
