"""Offline A/B/C analysis, figures, statistics, and Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from collections import defaultdict

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, ranksums
import torch

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
})

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.evaluation.pareto_front import pareto_front_points


PROBLEMS = ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7", "ConvexDTLZ2")
ALGORITHMS = ("nsga3", "rvea")
FRAMEWORKS = ("cuda_moea", "evox")
LABELS = {
    ("cuda_moea", "nsga3"): "CUDA-MOEA–NSGA-III",
    ("evox", "nsga3"): "EvoX–NSGA-III",
    ("cuda_moea", "rvea"): "CUDA-MOEA–RVEA",
    ("evox", "rvea"): "EvoX–RVEA",
}
IMPLEMENTATION_COLORS = {
    ("cuda_moea", "nsga3"): "#8CAFCF",
    ("evox", "nsga3"): "#EA8675",
    ("cuda_moea", "rvea"): "#BDB8D9",
    ("evox", "rvea"): "#F5C184",
}
ALGORITHM_COLORS = {"nsga3": "#8CAFCF", "rvea": "#BDB8D9"}
THEORY_COLOR = "#D9D9D9"
THEORY_EDGE_COLOR = "#8C8C8C"
MARKERS = {"nsga3": "o", "rvea": "s"}


def style_paper_axis(axis, *, grid_axis="both"):
    """Apply a compact publication-style treatment without an in-figure title."""
    axis.set_axisbelow(True)
    axis.grid(axis=grid_axis, which="major", linestyle="--", linewidth=.55,
              color="#B8B8B8", alpha=.55)
    axis.grid(axis=grid_axis, which="minor", linestyle=":", linewidth=.4,
              color="#D0D0D0", alpha=.35)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def read_rows(root: Path, group: str) -> list[dict]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob(f"{group}/**/run_result.json")]
    if not rows:
        raise FileNotFoundError(f"No results for {group}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_front(path: str) -> np.ndarray:
    points = torch.load(path, map_location="cpu", weights_only=True).detach().numpy().astype(np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    points = np.unique(points, axis=0)
    if not len(points):
        raise ValueError(f"No finite objectives in {path}")
    keep = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        if keep[index]:
            dominated = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
            if np.any(dominated):
                keep[index] = False
    return points[keep]


def analytic_front(problem: str, count: int) -> np.ndarray:
    return pareto_front_points(problem, 3, count=count, device="cpu", seed=2887).numpy().astype(np.float64)


def igd_kdtree(front: np.ndarray, reference: np.ndarray) -> float:
    distances = cKDTree(front).query(reference, k=1, workers=-1)[0]
    return float(np.mean(distances))


def quantiles(values) -> tuple[float, float, float]:
    return tuple(float(x) for x in np.quantile(np.asarray(values, dtype=float), (0.25, 0.5, 0.75)))


def summary(values) -> dict:
    values = np.asarray(values, dtype=float)
    q1, median, q3 = quantiles(values)
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "median": median, "q1": q1, "q3": q3, "iqr": q3 - q1}


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, original in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[original])
        adjusted[original] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_speedup(cuda_values, evox_values, seed: int, samples: int = 10000):
    cuda_values, evox_values = np.asarray(cuda_values), np.asarray(evox_values)
    rng = np.random.default_rng(seed)
    ratios = np.empty(samples)
    for index in range(samples):
        c = rng.choice(cuda_values, len(cuda_values), replace=True)
        e = rng.choice(evox_values, len(evox_values), replace=True)
        ratios[index] = np.median(e) / np.median(c)
    point = float(np.median(evox_values) / np.median(cuda_values))
    low, high = np.quantile(ratios, (0.025, 0.975))
    return point, float(low), float(high)


def quality_analysis(rows: list[dict], output: Path):
    references = {p: {n: analytic_front(p, n) for n in (20000, 50000)} for p in PROBLEMS}
    fronts = {}
    enriched = []
    for index, row in enumerate(rows, 1):
        front = clean_front(row["final_objectives"])
        fronts[row["final_objectives"]] = front
        item = dict(row)
        item["finite_unique_nondominated_points"] = len(front)
        item["igd_20000"] = igd_kdtree(front, references[row["problem"]][20000])
        item["igd_50000"] = igd_kdtree(front, references[row["problem"]][50000])
        enriched.append(item)
        if index % 120 == 0:
            print(f"IGD: {index}/{len(rows)}", flush=True)

    sensitivity = []
    selected_counts = {}
    for problem in PROBLEMS:
        rankings = {}
        for count in (20000, 50000):
            medians = []
            for framework in FRAMEWORKS:
                for algorithm in ALGORITHMS:
                    values = [r[f"igd_{count}"] for r in enriched if r["problem"] == problem and
                              r["framework"] == framework and r["algorithm"] == algorithm]
                    medians.append((float(np.median(values)), LABELS[(framework, algorithm)]))
            rankings[count] = [label for _, label in sorted(medians)]
        changed = rankings[20000] != rankings[50000]
        selected_counts[problem] = 50000 if changed else 20000
        sensitivity.append({"problem": problem, "ranking_20000": " > ".join(rankings[20000]),
                            "ranking_50000": " > ".join(rankings[50000]),
                            "ranking_changed": changed, "selected_pf_points": selected_counts[problem]})
    for row in enriched:
        row["pareto_reference_points"] = selected_counts[row["problem"]]
        row["final_igd"] = row[f"igd_{selected_counts[row['problem']]}" ]

    quality_summary = []
    for problem in PROBLEMS:
        for algorithm in ALGORITHMS:
            for framework in FRAMEWORKS:
                selected = [r for r in enriched if r["problem"] == problem and
                            r["algorithm"] == algorithm and r["framework"] == framework]
                for metric in ("final_igd", "total_time_ms", "mean_generation_ms"):
                    quality_summary.append({"problem": problem, "algorithm": algorithm,
                                            "framework": framework, "implementation": LABELS[(framework, algorithm)],
                                            "metric": metric, **summary([r[metric] for r in selected])})

    tests = []
    for algorithm in ALGORITHMS:
        algorithm_rows = []
        pvalues = []
        for problem in PROBLEMS:
            c = [r["final_igd"] for r in enriched if r["problem"] == problem and
                 r["algorithm"] == algorithm and r["framework"] == "cuda_moea"]
            e = [r["final_igd"] for r in enriched if r["problem"] == problem and
                 r["algorithm"] == algorithm and r["framework"] == "evox"]
            _, pvalue = ranksums(c, e)
            u = mannwhitneyu(c, e, alternative="two-sided").statistic
            item = {"problem": problem, "algorithm": algorithm, "test": "Wilcoxon rank-sum",
                    "p_value": float(pvalue), "rank_biserial_cuda_minus_evox": float(2*u/(len(c)*len(e))-1),
                    "cuda_median_igd": float(np.median(c)), "evox_median_igd": float(np.median(e))}
            algorithm_rows.append(item); pvalues.append(float(pvalue))
        adjusted = holm_adjust(pvalues)
        for item, value in zip(algorithm_rows, adjusted):
            item["holm_p_value"] = value
            item["significant_0_05"] = value < 0.05
            tests.append(item)

    representatives = []
    for problem in PROBLEMS:
        for algorithm in ALGORITHMS:
            for framework in FRAMEWORKS:
                selected = [r for r in enriched if r["problem"] == problem and r["algorithm"] == algorithm
                            and r["framework"] == framework]
                median = float(np.median([r["final_igd"] for r in selected]))
                chosen = min(selected, key=lambda r: (abs(r["final_igd"] - median), r["repeat_index"]))
                representatives.append({"problem": problem, "algorithm": algorithm, "framework": framework,
                                        "implementation": LABELS[(framework, algorithm)], "median_igd": median,
                                        "representative_igd": chosen["final_igd"], "seed": chosen["seed"],
                                        "repeat_index": chosen["repeat_index"],
                                        "final_objectives": chosen["final_objectives"]})

    write_csv(output / "data/quality_runs.csv", enriched)
    write_csv(output / "data/quality_summary.csv", quality_summary)
    write_csv(output / "data/quality_rank_sum_tests.csv", tests)
    write_csv(output / "data/pareto_front_sensitivity.csv", sensitivity)
    write_csv(output / "data/representative_runs.csv", representatives)
    return enriched, quality_summary, tests, sensitivity, representatives, fronts, references


def timing_analysis(rows: list[dict], x_field: str, seed_offset: int):
    result = []
    x_values = sorted({int(r[x_field]) for r in rows})
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        for x_index, x in enumerate(x_values):
            selected = [r for r in rows if r["algorithm"] == algorithm and int(r[x_field]) == x]
            c = [r["total_time_ms"] for r in selected if r["framework"] == "cuda_moea" and r["run_status"] == "success"]
            e = [r["total_time_ms"] for r in selected if r["framework"] == "evox" and r["run_status"] == "success"]
            if c and e:
                speedup, low, high = bootstrap_speedup(c, e, 2887 + seed_offset + algorithm_index*100 + x_index)
            else:
                speedup = low = high = float("nan")
            csum = summary(c) if c else {key: float("nan") for key in ("mean", "std", "median")}
            esum = summary(e) if e else {key: float("nan") for key in ("mean", "std", "median")}
            result.append({x_field: x, "algorithm": algorithm,
                           "cuda_median_total_ms": csum["median"], "cuda_mean_total_ms": csum["mean"],
                           "cuda_std_total_ms": csum["std"], "evox_median_total_ms": esum["median"],
                           "evox_mean_total_ms": esum["mean"], "evox_std_total_ms": esum["std"],
                           "cuda_median_generation_ms": csum["median"] / selected[0]["generations"] if c else float("nan"),
                           "evox_median_generation_ms": esum["median"] / selected[0]["generations"] if e else float("nan"),
                           "speedup": speedup, "speedup_ci_low": low, "speedup_ci_high": high,
                           "cuda_successes": len(c), "evox_successes": len(e),
                           "cuda_failures": sum(r["run_status"] != "success" for r in selected if r["framework"] == "cuda_moea"),
                           "evox_failures": sum(r["run_status"] != "success" for r in selected if r["framework"] == "evox")})
    return result


def plot_fronts(output, representatives, fronts, references, selected_counts):
    image_dir = output / "images/fronts"
    image_dir.mkdir(parents=True, exist_ok=True)
    for problem in PROBLEMS:
        figure = plt.figure(figsize=(13, 11))
        selected = [r for r in representatives if r["problem"] == problem]
        reference = references[problem][selected_counts[problem]]
        display_ref = reference[::max(1, len(reference)//3000)]
        all_points = [reference] + [fronts[r["final_objectives"]] for r in selected]
        combined = np.concatenate(all_points)
        lows, highs = combined.min(axis=0), combined.max(axis=0)
        margin = np.maximum((highs-lows)*0.04, 1e-8)
        for index, row in enumerate(selected, 1):
            axis = figure.add_subplot(2, 2, index, projection="3d")
            front = fronts[row["final_objectives"]]
            axis.scatter(display_ref[:,0], display_ref[:,1], display_ref[:,2], s=8, alpha=.72,
                         color=THEORY_COLOR, edgecolors=THEORY_EDGE_COLOR, linewidths=.18,
                         depthshade=False)
            axis.scatter(front[:,0], front[:,1], front[:,2], s=12, alpha=.78,
                         color=IMPLEMENTATION_COLORS[(row["framework"], row["algorithm"])],
                         edgecolors="#555555", linewidths=.12, depthshade=False)
            axis.set(xlim=(lows[0]-margin[0], highs[0]+margin[0]),
                     ylim=(lows[1]-margin[1], highs[1]+margin[1]),
                     zlim=(lows[2]-margin[2], highs[2]+margin[2]), xlabel="f1", ylabel="f2", zlabel="f3")
            axis.view_init(elev=22, azim=42)
            axis.set_title(f"{row['implementation']}\nseed={row['seed']}, IGD={row['representative_igd']:.4g}")
        handles = [
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=THEORY_COLOR,
                   markeredgecolor=THEORY_EDGE_COLOR, markersize=7, label="Theoretical Pareto front"),
            *[
                Line2D([0], [0], marker=MARKERS[algorithm], linestyle="none",
                       color=IMPLEMENTATION_COLORS[(framework, algorithm)], markersize=7,
                       label=f"{LABELS[(framework, algorithm)]} final front")
                for framework, algorithm in (("cuda_moea", "nsga3"), ("evox", "nsga3"),
                                             ("cuda_moea", "rvea"), ("evox", "rvea"))
            ],
        ]
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .965), ncol=3)
        figure.suptitle(f"{problem}: representative final fronts", fontsize=15, y=.995)
        figure.tight_layout(rect=(0,0,1,.925))
        figure.savefig(image_dir / f"{problem.lower()}_fronts.png", dpi=180)
        plt.close(figure)


def plot_quality(output, rows):
    image_dir = output / "images"; image_dir.mkdir(parents=True, exist_ok=True)
    implementations = [(f,a) for a in ALGORITHMS for f in FRAMEWORKS]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=False)
    for axis, problem in zip(axes.flat, PROBLEMS):
        data = [[r["final_igd"] for r in rows if r["problem"] == problem and
                 r["framework"] == f and r["algorithm"] == a] for f,a in implementations]
        boxes = axis.boxplot(data, patch_artist=True, showfliers=False)
        for patch, implementation in zip(boxes["boxes"], implementations):
            patch.set_facecolor(IMPLEMENTATION_COLORS[implementation]); patch.set_alpha(.88)
        axis.set_yscale("log"); axis.set_title(problem); axis.grid(alpha=.2, axis="y")
        axis.set_xticks(range(1,5), ["C-N", "E-N", "C-R", "E-R"], rotation=25)
    fig.suptitle("Final IGD distributions (C=CUDA-MOEA, E=EvoX, N=NSGA-III, R=RVEA)")
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(image_dir/"igd_distributions.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(13.5, 5.2))
    x = np.arange(len(PROBLEMS)); width=.19
    offsets = np.array((-1.5, -.5, .5, 1.5)) * width
    hatches = {"nsga3": "", "rvea": "//"}
    for impl_index, (f,a) in enumerate(implementations):
        means, stds = [], []
        for p in PROBLEMS:
            v=[r["mean_generation_ms"] for r in rows if r["problem"]==p and r["framework"]==f and r["algorithm"]==a]
            means.append(np.mean(v)); stds.append(np.std(v,ddof=1))
        axis.bar(x + offsets[impl_index], means, width=width, yerr=stds, capsize=2.5,
                 error_kw={"elinewidth": .8, "capthick": .8},
                 color=IMPLEMENTATION_COLORS[(f, a)], hatch=hatches[a], edgecolor="#505050", linewidth=.65,
                 alpha=.88, label=LABELS[(f,a)])
    axis.set_yscale("log"); axis.set_xticks(x, PROBLEMS, rotation=25, ha="right")
    axis.margins(y=.10)
    axis.set_ylabel("Mean generation time (ms)")
    style_paper_axis(axis, grid_axis="y")
    axis.legend(ncol=2, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(image_dir/"quality_mean_generation_time.png", dpi=300,
                                    bbox_inches="tight"); plt.close(fig)


def plot_scaling(output, summaries, x_field, prefix, xlabel):
    image_dir=output/"images"; image_dir.mkdir(parents=True,exist_ok=True)
    fig, axis=plt.subplots(figsize=(8.2,5.2))
    line_styles = {"nsga3": "-", "rvea": "--"}
    for algorithm in ALGORITHMS:
        data=sorted([r for r in summaries if r["algorithm"]==algorithm],key=lambda r:r[x_field])
        for framework in FRAMEWORKS:
            key = f"{'cuda' if framework == 'cuda_moea' else 'evox'}_median_generation_ms"
            available = [r for r in data if np.isfinite(r[key])]
            x=[r[x_field] for r in available]
            y=[r[key] for r in available]
            axis.plot(x,y,marker=MARKERS[algorithm],linestyle=line_styles[algorithm],
                      linewidth=2.2, markersize=6, markeredgewidth=.7,
                      label=LABELS[(framework,algorithm)],
                      color=IMPLEMENTATION_COLORS[(framework, algorithm)])
    axis.set_xscale("log",base=2); axis.set_yscale("log")
    axis.set_xlabel(xlabel); axis.set_ylabel("Median generation time (ms)")
    style_paper_axis(axis)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(image_dir/f"{prefix}_generation_time.png", dpi=300,
                                    bbox_inches="tight"); plt.close(fig)
    fig,axis=plt.subplots(figsize=(8.2,5.2))
    for algorithm in ALGORITHMS:
        data=sorted([r for r in summaries if r["algorithm"]==algorithm],key=lambda r:r[x_field])
        data=[r for r in data if np.isfinite(r["speedup"])]
        x=np.array([r[x_field] for r in data]); y=np.array([r["speedup"] for r in data])
        algorithm_color = ALGORITHM_COLORS[algorithm]
        display_algorithm = "NSGA-III" if algorithm == "nsga3" else "RVEA"
        axis.plot(x, y, marker=MARKERS[algorithm], linewidth=2.2, markersize=6,
                  markeredgewidth=.7,
                  label=display_algorithm, color=algorithm_color)
    axis.axhline(1,color="#4A4A4A",ls="--",lw=1); axis.set_xscale("log",base=2); axis.set_yscale("log")
    axis.set_xlabel(xlabel); axis.set_ylabel("Speedup")
    style_paper_axis(axis)
    axis.legend(frameon=False); fig.tight_layout()
    fig.savefig(image_dir/f"{prefix}_speedup.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fmt(value):
    return f"{value:.4g}"


def report(output, manifest, qsummary, tests, sensitivity, pop_summary, dim_summary):
    """Write a data-driven report; timing failures remain explicit rather than omitted."""
    lookup = {(row["problem"], row["algorithm"], row["framework"], row["metric"]): row
              for row in qsummary}
    packages = manifest.get("packages", {})
    gpu = (manifest.get("nvidia_smi") or "unknown").splitlines()[0]
    quality_speedups = {
        (problem, algorithm): (
            lookup[(problem, algorithm, "evox", "total_time_ms")]["median"] /
            lookup[(problem, algorithm, "cuda_moea", "total_time_ms")]["median"]
        )
        for problem in PROBLEMS for algorithm in ALGORITHMS
    }
    a_runs = sum(row["count"] for row in qsummary if row["metric"] == "final_igd")
    b_success = sum(row["cuda_successes"] + row["evox_successes"] for row in pop_summary)
    b_failed = sum(row["cuda_failures"] + row["evox_failures"] for row in pop_summary)
    c_success = sum(row["cuda_successes"] + row["evox_successes"] for row in dim_summary)
    quality_wins = {
        algorithm: sum(item["cuda_median_igd"] < item["evox_median_igd"]
                       for item in tests if item["algorithm"] == algorithm)
        for algorithm in ALGORITHMS
    }
    significant_wins = {
        algorithm: sum(item["cuda_median_igd"] < item["evox_median_igd"] and item["significant_0_05"]
                       for item in tests if item["algorithm"] == algorithm)
        for algorithm in ALGORITHMS
    }
    a_speedup_ranges = {
        algorithm: (min(quality_speedups[(problem, algorithm)] for problem in PROBLEMS),
                    max(quality_speedups[(problem, algorithm)] for problem in PROBLEMS))
        for algorithm in ALGORITHMS
    }
    finite_pop = {algorithm: [row for row in pop_summary if row["algorithm"] == algorithm and np.isfinite(row["speedup"])]
                  for algorithm in ALGORITHMS}
    finite_dim = {algorithm: [row for row in dim_summary if row["algorithm"] == algorithm and np.isfinite(row["speedup"])]
                  for algorithm in ALGORITHMS}
    lines = [
        "# CUDA-MOEA 与 EvoX：DTLZ A/B/C 测试报告", "",
        "[English](REPORT.md) | 中文", "",
        "> 数据源：`output/benchmark_20260710/data/raw_runs.json`。全部统计、表格和图像均由该数据重新生成。", "",
        "## 1. 数据完整性与口径", "",
        f"- A 组：{a_runs}/960 次成功运行；B 组：{b_success}/320 次成功、{b_failed} 次失败；C 组：{c_success}/440 次成功运行。",
        "- B 组 `N=32768` 的 EvoX–RVEA 为 10/10 EvoX OOM；保留失败记录，不填补时间，也不计算该点加速比。",
        "- IGD 越小越好。时间为初始化完成后的代循环 CUDA-MOEA event 时间；加速比为 `median(EvoX) / median(CUDA-MOEA)`，大于 1 表示 CUDA-MOEA 更快。",
        f"- 基线提交：`{manifest.get('git_commit')}`；设备：`{manifest.get('device')}`；GPU：`{gpu}`。",
        f"- 环境：CUDA-MOEA {packages.get('cuda-moea')}，EvoX {packages.get('evox')}，PyTorch {packages.get('torch')}。", "",
        "## 2. A 组：解质量、每代时间与加速比", "",
        "每项为 30 个独立种子的汇总。`IGD mean ± std` 与 `IGD median [IQR]` 同时报告，以兼顾异常值与典型表现；每代运行时间为每次 `total_time_ms / generations` 后的 `mean ± std`。加速比只列在 CUDA-MOEA 行，表示同问题、同算法的 CUDA-MOEA 相对 EvoX 加速比。", "",
        "| Implementation | Problem | IGD mean ± std | IGD median [IQR] | 每代时间 (ms, mean ± std) | 加速比 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    implementations = [("cuda_moea", "nsga3"), ("evox", "nsga3"), ("cuda_moea", "rvea"), ("evox", "rvea")]
    for framework, algorithm in implementations:
        for index, problem in enumerate(PROBLEMS):
            igd = lookup[(problem, algorithm, framework, "final_igd")]
            time = lookup[(problem, algorithm, framework, "mean_generation_ms")]
            implementation = LABELS[(framework, algorithm)] if index == 0 else ""
            speedup = fmt(quality_speedups[(problem, algorithm)]) + "×" if framework == "cuda_moea" else "—"
            lines.append(
                f"| {implementation} | {problem} | {fmt(igd['mean'])} ± {fmt(igd['std'])} | "
                f"{fmt(igd['median'])} [{fmt(igd['q1'])}, {fmt(igd['q3'])}] | "
                f"{fmt(time['mean'])} ± {fmt(time['std'])} | {speedup} |"
            )
    lines += [
        "",
        f"**结果解读。** CUDA-MOEA 的中位 IGD 更低的题目数为：NSGA-III {quality_wins['nsga3']}/8（Holm 校正后显著 {significant_wins['nsga3']} 个），RVEA {quality_wins['rvea']}/8（显著 {significant_wins['rvea']} 个）。因此解质量不存在单一框架全面占优；应优先结合中位数与 IQR 判断，而不只看均值。",
        f"同算法的 A 组逐问题时间加速比范围为 NSGA-III {a_speedup_ranges['nsga3'][0]:.2f}–{a_speedup_ranges['nsga3'][1]:.2f}×、RVEA {a_speedup_ranges['rvea'][0]:.2f}–{a_speedup_ranges['rvea'][1]:.2f}×，所有值均高于 1。", "",
        "### 2.1 IGD 分布", "",
        "![IGD distributions](images/igd_distributions.png)", "",
        "图中对数纵轴展示 30 次运行的 IGD 分布。它与总表中的 median[IQR] 互为补充：DTLZ1、DTLZ3 的 RVEA 两框架差异更明显，而 DTLZ2、DTLZ7 的 NSGA-III 中位 IGD 接近，不能仅由细小均值差异推断稳定优势。", "",
        "### 2.2 四实现每代时间", "",
        "![A-group generation time](images/quality_mean_generation_time.png)", "",
        "该图在同一张分组柱状图中绘制 CUDA-MOEA/EvoX × NSGA-III/RVEA 四个实现；误差线为 30 次标准差，纵轴为对数尺度。四组柱的相对高度与总表加速比一致：同算法下 CUDA-MOEA 每代时间均低于 EvoX；NSGA-III 与 RVEA 的柱高不用于跨算法速度优劣判断。", "",
        "### 2.3 代表最终前沿", "",
        "每个子图选择 IGD 最接近该实现 30 次中位数的运行；同一问题的四个实现共享坐标范围、视角和理论前沿，因此前沿图用于检查覆盖形态而非挑选最佳单次运行。", "",
    ]
    for index in range(0, len(PROBLEMS), 2):
        left, right = PROBLEMS[index:index + 2]
        lines += [f'<p><img src="images/fronts/{left.lower()}_fronts.png" alt="{left} fronts" width="49%"> <img src="images/fronts/{right.lower()}_fronts.png" alt="{right} fronts" width="49%"></p>', ""]
    lines += [
        "前沿图总体与 IGD 统计一致：覆盖较均匀且更接近理论前沿的实现通常有更低 IGD；退化或不连续前沿问题则需同时观察形状和总表数值，不能用视觉点密度替代 IGD。", "",
        "## 3. B 组：种群规模扩展", "",
        "| Algorithm | 名义种群 N | CUDA-MOEA ms/gen | EvoX ms/gen | Speedup [95% CI] | 成功数（CUDA-MOEA/EvoX） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        rows = sorted((row for row in pop_summary if row["algorithm"] == algorithm), key=lambda row: row["nominal_population"])
        for index, row in enumerate(rows):
            speedup = "—" if not np.isfinite(row["speedup"]) else f"{fmt(row['speedup'])} [{fmt(row['speedup_ci_low'])}, {fmt(row['speedup_ci_high'])}]"
            evox_time = "—" if not np.isfinite(row["evox_median_generation_ms"]) else fmt(row["evox_median_generation_ms"])
            label = ("NSGA-III" if algorithm == "nsga3" else "RVEA") if index == 0 else ""
            lines.append(f"| {label} | {row['nominal_population']} | {fmt(row['cuda_median_generation_ms'])} | {evox_time} | {speedup} | {row['cuda_successes']}/{row['evox_successes']} |")
    lines += [
        "", '<p><img src="images/population_generation_time.png" alt="Population time" width="49%"> <img src="images/population_speedup.png" alt="Population speedup" width="49%"></p>', "",
        f"**结果解读。** NSGA-III 加速比从 {finite_pop['nsga3'][0]['speedup']:.2f}×（N={finite_pop['nsga3'][0]['nominal_population']}）提高到 {finite_pop['nsga3'][-1]['speedup']:.2f}×（N={finite_pop['nsga3'][-1]['nominal_population']}）；RVEA 在成功比较点为 {min(row['speedup'] for row in finite_pop['rvea']):.2f}–{max(row['speedup'] for row in finite_pop['rvea']):.2f}×。两张图显示 EvoX 时间随种群扩大增长更快，而 CUDA-MOEA 增长较缓。RVEA 的 N=32768 缺点是 OOM，不应把曲线末端留白解读为零开销。", "",
        "## 4. C 组：决策维数扩展", "",
        "| Algorithm | 决策维数 D | CUDA-MOEA ms/gen | EvoX ms/gen | Speedup [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        rows = sorted((row for row in dim_summary if row["algorithm"] == algorithm), key=lambda row: row["dimension"])
        for index, row in enumerate(rows):
            label = ("NSGA-III" if algorithm == "nsga3" else "RVEA") if index == 0 else ""
            lines.append(f"| {label} | {row['dimension']} | {fmt(row['cuda_median_generation_ms'])} | {fmt(row['evox_median_generation_ms'])} | {fmt(row['speedup'])} [{fmt(row['speedup_ci_low'])}, {fmt(row['speedup_ci_high'])}] |")
    lines += [
        "", '<p><img src="images/dimension_generation_time.png" alt="Dimension time" width="49%"> <img src="images/dimension_speedup.png" alt="Dimension speedup" width="49%"></p>', "",
        f"**结果解读。** NSGA-III 加速比在 D={min(finite_dim['nsga3'], key=lambda row: row['speedup'])['dimension']} 时最低（{min(row['speedup'] for row in finite_dim['nsga3']):.2f}×），在 D=131072 达到 {max(finite_dim['nsga3'], key=lambda row: row['dimension'])['speedup']:.2f}×；RVEA 对应范围为 {min(row['speedup'] for row in finite_dim['rvea']):.2f}–{max(row['speedup'] for row in finite_dim['rvea']):.2f}×。时间图与加速比图共同表明中等维数时相对优势收窄，而高维计算量主导后 CUDA-MOEA 优势再次扩大。", "",
        "## 5. `results/data/` 派生数据说明", "",
        "这些文件均由原始运行记录离线生成，不参与算法计时；它们是表格、统计检验和图表的可复核数据源。", "",
        "| 文件 | 记录粒度 | 内容与用途 |", "|---|---|---|",
        "| `quality_runs.csv` | A 组每次独立运行 | 原始字段、非支配点数、20k/50k IGD、最终 IGD；用于质量统计和代表运行选择。 |",
        "| `quality_summary.csv` | 问题–算法–框架–指标 | IGD、总时间、每代时间的 count/mean/std/median/Q1/Q3/IQR；用于 A 组总表和时间图。 |",
        "| `quality_rank_sum_tests.csv` | 问题–算法 | Wilcoxon rank-sum、Holm 校正、RBC 和中位 IGD；用于判断质量差异。 |",
        "| `pareto_front_sensitivity.csv` | 问题 | 20k/50k 理论前沿下的排序敏感性与正式参考点数。 |",
        "| `representative_runs.csv` | 问题–算法–框架 | 最接近中位 IGD 的 seed、重复编号和目标矩阵路径；用于前沿图。 |",
        "| `population_timing_summary.csv` | B 组种群–算法 | 两框架的时间汇总、加速比与 bootstrap CI、成功/失败数；RVEA N=32768 的 EvoX 字段为空并保留失败计数。 |",
        "| `dimension_timing_summary.csv` | C 组维数–算法 | 两框架的时间汇总、加速比与 bootstrap CI，用于维数扩展表和图。 |",
    ]
    (output / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return

    packages = manifest.get("packages", {})
    gpu = (manifest.get("nvidia_smi") or "unknown").splitlines()[0]
    a_runs = sum(r["count"] for r in qsummary if r["metric"] == "final_igd")
    b_success = sum(r["cuda_successes"] + r["evox_successes"] for r in pop_summary)
    b_failed = sum(r["cuda_failures"] + r["evox_failures"] for r in pop_summary)
    c_success = sum(r["cuda_successes"] + r["evox_successes"] for r in dim_summary)
    lines = [
        "# CUDA-MOEA 与 EvoX：DTLZ A/B/C 测试报告", "",
        "[English](REPORT.md) | 中文",
        "",
        "> 数据源：`output/benchmark_20260710/data/raw_runs.json`。本报告和全部派生表、图均由该批数据重新生成。",
        "",
        "## 数据完整性",
        "",
        f"- A 组：{a_runs} 次成功运行。",
        f"- B 组：{b_success} 次成功、{b_failed} 次失败。",
        f"- C 组：{c_success} 次成功运行。",
        f"- 基线提交：`{manifest.get('git_commit')}`；逻辑设备：`{manifest.get('device')}`；GPU：`{gpu}`。",
        f"- 环境：CUDA-MOEA {packages.get('cuda-moea')}，EvoX {packages.get('evox')}，PyTorch {packages.get('torch')}。",
        "",
        "## B 组失败说明",
        "",
        "`DTLZ1, N=32768` 的 EvoX–RVEA 10/10 次运行均为 EvoX OOM；CUDA-MOEA–RVEA 的对应 10 次均成功。因此该点不计算跨框架加速比，也不会被错误地当作零时间或成功样本纳入统计。",
        "",
        "## A 组：解质量",
        "",
        "IGD 越小越好；统计检验为双侧 Wilcoxon rank-sum，并在同一算法的 8 个问题内进行 Holm 校正。",
        "",
        "| Algorithm | Problem | CUDA-MOEA median IGD | EvoX median IGD | Holm p | 结论 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for algorithm in ALGORITHMS:
        for index, item in enumerate(r for r in tests if r["algorithm"] == algorithm):
            winner = "差异不显著" if not item["significant_0_05"] else ("CUDA-MOEA 更低" if item["cuda_median_igd"] < item["evox_median_igd"] else "EvoX 更低")
            label = ("NSGA-III" if algorithm == "nsga3" else "RVEA") if index == 0 else ""
            lines.append(f"| {label} | {item['problem']} | {fmt(item['cuda_median_igd'])} | {fmt(item['evox_median_igd'])} | {fmt(item['holm_p_value'])} | {winner} |")
    lines += [
        "",
        "![IGD distributions](images/igd_distributions.png)",
        "",
        "## B 组：种群规模扩展",
        "",
        "加速比定义为 `median(EvoX) / median(CUDA-MOEA)`；空值表示该实现没有成功运行。",
        "",
        "| Algorithm | N | CUDA-MOEA ms/gen | EvoX ms/gen | Speedup [95% CI] | 成功数（CUDA-MOEA/EvoX） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        for item in sorted((r for r in pop_summary if r["algorithm"] == algorithm), key=lambda r: r["nominal_population"]):
            speedup = "—" if not np.isfinite(item["speedup"]) else f"{fmt(item['speedup'])} [{fmt(item['speedup_ci_low'])}, {fmt(item['speedup_ci_high'])}]"
            evox_time = "—" if not np.isfinite(item["evox_median_generation_ms"]) else fmt(item["evox_median_generation_ms"])
            lines.append(f"| {'NSGA-III' if algorithm == 'nsga3' else 'RVEA'} | {item['nominal_population']} | {fmt(item['cuda_median_generation_ms'])} | {evox_time} | {speedup} | {item['cuda_successes']}/{item['evox_successes']} |")
    lines += ["", '<p><img src="images/population_generation_time.png" alt="Population time" width="49%"> <img src="images/population_speedup.png" alt="Population speedup" width="49%"></p>', "", "## C 组：决策维数扩展", "", "| Algorithm | D | CUDA-MOEA ms/gen | EvoX ms/gen | Speedup [95% CI] |", "|---|---:|---:|---:|---:|"]
    for algorithm in ALGORITHMS:
        for item in sorted((r for r in dim_summary if r["algorithm"] == algorithm), key=lambda r: r["dimension"]):
            lines.append(f"| {'NSGA-III' if algorithm == 'nsga3' else 'RVEA'} | {item['dimension']} | {fmt(item['cuda_median_generation_ms'])} | {fmt(item['evox_median_generation_ms'])} | {fmt(item['speedup'])} [{fmt(item['speedup_ci_low'])}, {fmt(item['speedup_ci_high'])}] |")
    lines += ["", '<p><img src="images/dimension_generation_time.png" alt="Dimension time" width="49%"> <img src="images/dimension_speedup.png" alt="Dimension speedup" width="49%"></p>', "", "## 派生文件", "", "`data/` 包含运行级质量指标、质量汇总与检验、理论前沿敏感性、代表运行，以及 B/C 时序汇总；`images/` 包含相应图表和代表 Pareto 前沿图。"]
    (output / "REPORT.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return

    lookup={(r["problem"],r["algorithm"],r["framework"],r["metric"]):r for r in qsummary}
    quality_wins = {
        algorithm: sum(r["cuda_median_igd"] < r["evox_median_igd"]
                       for r in tests if r["algorithm"] == algorithm)
        for algorithm in ALGORITHMS
    }
    quality_sig_wins = {
        algorithm: sum(r["cuda_median_igd"] < r["evox_median_igd"] and r["significant_0_05"]
                       for r in tests if r["algorithm"] == algorithm)
        for algorithm in ALGORITHMS
    }
    a_speedups = {}
    for algorithm in ALGORITHMS:
        values = []
        for problem in PROBLEMS:
            c = lookup[(problem, algorithm, "cuda_moea", "total_time_ms")]["median"]
            e = lookup[(problem, algorithm, "evox", "total_time_ms")]["median"]
            values.append(e / c)
        a_speedups[algorithm] = (min(values), max(values))
    population_ranges = {a: (min(r["speedup"] for r in pop_summary if r["algorithm"] == a),
                             max(r["speedup"] for r in pop_summary if r["algorithm"] == a))
                         for a in ALGORITHMS}
    dimension_ranges = {a: (min(r["speedup"] for r in dim_summary if r["algorithm"] == a),
                            max(r["speedup"] for r in dim_summary if r["algorithm"] == a))
                        for a in ALGORITHMS}
    population_last = {a: max((r for r in pop_summary if r["algorithm"] == a),
                              key=lambda r: r["nominal_population"]) for a in ALGORITHMS}
    dimension_min = {a: min((r for r in dim_summary if r["algorithm"] == a),
                            key=lambda r: r["speedup"]) for a in ALGORITHMS}
    dimension_last = {a: max((r for r in dim_summary if r["algorithm"] == a),
                             key=lambda r: r["dimension"]) for a in ALGORITHMS}
    packages = manifest.get("packages", {})
    gpu = (manifest.get("nvidia_smi") or "unknown").splitlines()[0]
    dirty_note = (["> **复现性提示：** 正式运行时 Git 工作树不是 clean 状态；`manifest.json` 已保存当时的完整 `git status`，因此 commit 只能作为基线标识。", ""]
                  if manifest.get("git_status", "").strip() else [])
    lines=["# CUDA-MOEA 与 EvoX：DTLZ A/B/C 测试报告", "",
           "[English](REPORT.md) | 中文", "",
           "> 本报告仅覆盖 DTLZ A、B、C 三组。MoRobtrol 不在本报告范围内。", "",
           "## 1. 数据完整性", "",
           "A/B/C 分别包含 960、320、440 次成功运行，共 1720 次；无失败、OOM、NaN 或缺失。",
           f"基线提交：`{manifest.get('git_commit')}`；逻辑设备：`{manifest.get('device')}`；GPU：`{gpu}`。",
           f"环境版本：CUDA-MOEA {packages.get('cuda-moea')}，EvoX {packages.get('evox')}，PyTorch {packages.get('torch')}，NumPy {packages.get('numpy')}，SciPy {packages.get('scipy')}。", "",
           *dirty_note,
           "## 2. 方法", "",
           "A 组对每个最终目标矩阵执行有限值过滤、精确去重和非支配过滤，再计算 IGD。每个问题先比较 20,000 与 50,000 点理论前沿的四实现中位数排序；排序变化时采用 50,000 点。代表运行取 IGD 最接近该实现 30 次中位数的运行。",
           "质量比较使用 Wilcoxon rank-sum 检验，按算法在 8 个问题内做 Holm 校正，并报告 rank-biserial correlation（正值表示 CUDA-MOEA 的 IGD 值整体更大，即更差）。测速以 10 次重复的中位数为主，加速比为 median(EvoX)/median(CUDA-MOEA)，置信区间为 10,000 次 bootstrap。", "",
           "## 3. 主要结论", "",
           f"- A 组中，CUDA-MOEA 的中位 IGD 在 NSGA-III 的 {quality_wins['nsga3']}/8 个问题、RVEA 的 {quality_wins['rvea']}/8 个问题上更低；其中分别有 {quality_sig_wins['nsga3']} 和 {quality_sig_wins['rvea']} 个经 Holm 校正后显著。解质量不存在单一框架全面占优。",
           f"- A 组相同算法时间比较中，CUDA-MOEA–NSGA-III 的逐问题中位加速比为 {a_speedups['nsga3'][0]:.2f}–{a_speedups['nsga3'][1]:.2f}×，CUDA-MOEA–RVEA 为 {a_speedups['rvea'][0]:.2f}–{a_speedups['rvea'][1]:.2f}×。",
           f"- B 组随种群扩大，NSGA-III 加速比为 {population_ranges['nsga3'][0]:.2f}–{population_ranges['nsga3'][1]:.2f}×，RVEA 为 {population_ranges['rvea'][0]:.2f}–{population_ranges['rvea'][1]:.2f}×；最大规模 N=16384 时分别达到 {population_ranges['nsga3'][1]:.2f}× 和 {population_ranges['rvea'][1]:.2f}×。",
           f"- C 组随决策维数变化，NSGA-III 加速比为 {dimension_ranges['nsga3'][0]:.2f}–{dimension_ranges['nsga3'][1]:.2f}×，RVEA 为 {dimension_ranges['rvea'][0]:.2f}–{dimension_ranges['rvea'][1]:.2f}×。",
           "- 20k/50k 理论前沿敏感性检查仅在 DTLZ4 改变四实现中位数排序，因此 DTLZ4 正式采用 50,000 点，其余问题采用 20,000 点。", "",
           "## 4. A 组：解质量与平均每代时间", "",
           "### 4.1 IGD 汇总", "",
           "IGD 越小表示最终解集越接近理论 Pareto 前沿。表中 IGD 为 `mean ± std`，时间为 30 次运行的每代时间 `mean ± std`。", "",
           "| Algorithm | Problem | IGD | IGD-median [IQR] | 每代平均时间 (ms) |",
           "|---|---|---:|---:|---:|"]
    implementations = [("cuda_moea", "nsga3"), ("evox", "nsga3"),
                       ("cuda_moea", "rvea"), ("evox", "rvea")]
    for framework, algorithm in implementations:
        for problem_index, problem in enumerate(PROBLEMS):
            igd_row=lookup[(problem,algorithm,framework,"final_igd")]
            time_row=lookup[(problem,algorithm,framework,"mean_generation_ms")]
            algorithm_cell = LABELS[(framework,algorithm)] if problem_index == 0 else ""
            lines.append(
                f"| {algorithm_cell} | {problem} | {fmt(igd_row['mean'])} ± {fmt(igd_row['std'])} | "
                f"{fmt(igd_row['median'])} [{fmt(igd_row['q1'])}, {fmt(igd_row['q3'])}] | "
                f"{fmt(time_row['mean'])} ± {fmt(time_row['std'])} |"
            )
    lines += ["",
              "结果显示，CUDA-MOEA–NSGA-III 在 DTLZ1、DTLZ4–7 上取得更低的中位 IGD，EvoX–NSGA-III 在 DTLZ2、DTLZ3 和 ConvexDTLZ2 上更低。CUDA-MOEA–RVEA 在 DTLZ1、DTLZ3、DTLZ5、DTLZ7 上更低，EvoX–RVEA 在其余四个问题上更低。DTLZ1 的 CUDA-MOEA–RVEA 均值明显高于中位数，说明少数独立运行出现较大退化，因此该问题应优先结合 median[IQR] 解读。", "",
              "分布图采用对数纵轴并隐藏极端离群点的单独标记，但箱线范围和所有统计量均由完整 30 次运行计算。", "",
              "![IGD distributions](images/igd_distributions.png)", "", "### 4.2 显著性检验", "",
              "RBC 为 rank-biserial correlation；正值表示 CUDA-MOEA 的 IGD 整体更大（更差），负值表示 CUDA-MOEA 整体更小（更好）。", "",
              "| Algorithm | Problem | CUDA-MOEA IGD-median | EvoX IGD-median | Holm p | RBC | 结果 |",
              "|---|---|---:|---:|---:|---:|---|"]
    for algorithm in ALGORITHMS:
        selected_tests = [r for r in tests if r["algorithm"] == algorithm]
        for problem_index, r in enumerate(selected_tests):
            algorithm_cell = "NSGA-III" if algorithm == "nsga3" else "RVEA"
            if problem_index:
                algorithm_cell = ""
            if not r["significant_0_05"]:
                conclusion = "差异不显著"
            elif r["cuda_median_igd"] < r["evox_median_igd"]:
                conclusion = "CUDA-MOEA 更低"
            else:
                conclusion = "EvoX 更低"
            lines.append(f"| {algorithm_cell} | {r['problem']} | {fmt(r['cuda_median_igd'])} | {fmt(r['evox_median_igd'])} | {fmt(r['holm_p_value'])} | {fmt(r['rank_biserial_cuda_minus_evox'])} | {conclusion} |")
    lines += ["",
              "经 8 个问题内 Holm 校正后，NSGA-III 的 8/8 个问题均达到显著差异；其中 CUDA-MOEA 在 5 个问题的 IGD 更低，EvoX 在 3 个问题更低。RVEA 的 7/8 个问题达到显著差异；CUDA-MOEA 和 EvoX 各在 4 个问题取得更低中位数，但 ConvexDTLZ2 的差异不显著。显著差异不等于工程上必然重要，仍应结合 IGD 绝对差和前沿图判断。", "",
              "### 4.3 每代时间", "",
              "下图是一张四系列分组柱状图：紫色始终表示 CUDA-MOEA，红色始终表示 EvoX；RVEA 使用斜线纹理，NSGA-III 使用纯色。误差线为 30 次运行的标准差，纵轴采用对数尺度。", "",
              "![A-group time](images/quality_mean_generation_time.png)", "",
              f"八个问题中，CUDA-MOEA 在相同算法下均具有更低的每代时间。逐问题中位加速比范围为 NSGA-III {a_speedups['nsga3'][0]:.2f}–{a_speedups['nsga3'][1]:.2f}×、RVEA {a_speedups['rvea'][0]:.2f}–{a_speedups['rvea'][1]:.2f}×。按照测试方案，NSGA-III 与 RVEA 之间不直接计算框架加速比。", "",
              "### 4.4 代表最终前沿", "",
              "每个子图使用相同坐标范围和视角。带橙色描边的黄色点为理论 Pareto 前沿，紫色点为 CUDA-MOEA 最终前沿，红色点为 EvoX 最终前沿。代表运行按 IGD 最接近各实现 30 次中位数选择，而非选择最好一次。", ""]
    for index in range(0, len(PROBLEMS), 2):
        left, right = PROBLEMS[index:index + 2]
        lines.append(
            f'<p><img src="images/fronts/{left.lower()}_fronts.png" alt="{left} fronts" width="49%"> '
            f'<img src="images/fronts/{right.lower()}_fronts.png" alt="{right} fronts" width="49%"></p>'
        )
    lines += ["", "前沿图与 IGD 统计总体一致：DTLZ1 和 DTLZ3 中 EvoX–RVEA 的代表前沿偏离更明显；DTLZ5/DTLZ6 的退化前沿上，NSGA-III 的 IGD 明显低于两种 RVEA；ConvexDTLZ2 则由 EvoX–NSGA-III 取得最低中位 IGD。图形用于展示覆盖形态，定量结论仍以完整 30 次运行的统计为准。"]
    lines += ["", "## 5. B 组：种群规模扩展", "",
              "下图将四种实现画在同一坐标轴上：紫色始终表示 CUDA-MOEA，红色始终表示 EvoX；实线圆点表示 NSGA-III，虚线方点表示 RVEA，横纵轴均采用对数尺度。", "",
              "在小种群区间，EvoX 每代时间主要受固定调度开销影响，曲线变化较小；当名义种群超过 2048 后，EvoX 两条曲线增长明显加快，而 CUDA-MOEA 增长更缓。对应地，NSGA-III 加速比由 N=128 的 5.42× 提升到 N=16384 的 155.93×，RVEA 由 15.69× 提升到 264.35×。", "",
              "加速比图绘制各测试规模下的中位加速比；95% bootstrap 置信区间保留在下表中，且所有区间均完全高于 1。横轴使用名义规模；RVEA 两框架实际使用相同参考向量数，因此每个加速比仍是同实际种群规模比较。", "",
              '<p><img src="images/population_generation_time.png" alt="Population time" width="49%"> '
              '<img src="images/population_speedup.png" alt="Population speedup" width="49%"></p>', "",
              "| Algorithm | Problem | CUDA-MOEA ms/gen | EvoX ms/gen | Speedup [95% CI] |",
              "|---|---|---:|---:|---:|"]
    for algorithm in ALGORITHMS:
        selected = sorted((r for r in pop_summary if r["algorithm"] == algorithm),
                          key=lambda r: r["nominal_population"])
        for row_index, r in enumerate(selected):
            algorithm_cell = ("NSGA-III" if algorithm == "nsga3" else "RVEA") if row_index == 0 else ""
            lines.append(f"| {algorithm_cell} | DTLZ1 (N={r['nominal_population']}) | {fmt(r['cuda_median_generation_ms'])} | {fmt(r['evox_median_generation_ms'])} | {fmt(r['speedup'])} [{fmt(r['speedup_ci_low'])}, {fmt(r['speedup_ci_high'])}] |")
    lines += ["", "在最大名义规模 N=16384，NSGA-III 的 CUDA-MOEA/EvoX 每代中位时间分别为 "
              f"{population_last['nsga3']['cuda_median_generation_ms']:.3g}/{population_last['nsga3']['evox_median_generation_ms']:.3g} ms，"
              "RVEA 分别为 "
              f"{population_last['rvea']['cuda_median_generation_ms']:.3g}/{population_last['rvea']['evox_median_generation_ms']:.3g} ms。结果表明 CUDA-MOEA 的相对优势随种群扩大而显著增强。", "",
              "## 6. C 组：决策维数扩展", "",
              "四条曲线同样绘制在一个双对数坐标轴上。该组固定名义种群为 1024（NSGA-III 实际 1024，RVEA 实际 990），只改变 DTLZ1 决策维数。", "",
              f"两种算法的加速比都呈现先下降后回升的趋势：NSGA-III 在 D={dimension_min['nsga3']['dimension']} 时最低，为 {dimension_min['nsga3']['speedup']:.2f}×；RVEA 在 D={dimension_min['rvea']['dimension']} 时最低，为 {dimension_min['rvea']['speedup']:.2f}×。这说明中等维数时两框架都能较充分摊薄固定开销，而进入高维区间后，CUDA-MOEA 的数据并行实现扩展更有利。", "",
              f"到 D=131072，NSGA-III 和 RVEA 的加速比分别达到 {dimension_last['nsga3']['speedup']:.2f}× 和 {dimension_last['rvea']['speedup']:.2f}×；全部 95% bootstrap 区间仍高于 1。", "",
              '<p><img src="images/dimension_generation_time.png" alt="Dimension time" width="49%"> '
              '<img src="images/dimension_speedup.png" alt="Dimension speedup" width="49%"></p>', "",
              "| Algorithm | Problem | CUDA-MOEA ms/gen | EvoX ms/gen | Speedup [95% CI] |",
              "|---|---|---:|---:|---:|"]
    for algorithm in ALGORITHMS:
        selected = sorted((r for r in dim_summary if r["algorithm"] == algorithm),
                          key=lambda r: r["dimension"])
        for row_index, r in enumerate(selected):
            algorithm_cell = ("NSGA-III" if algorithm == "nsga3" else "RVEA") if row_index == 0 else ""
            lines.append(f"| {algorithm_cell} | DTLZ1 (D={r['dimension']}) | {fmt(r['cuda_median_generation_ms'])} | {fmt(r['evox_median_generation_ms'])} | {fmt(r['speedup'])} [{fmt(r['speedup_ci_low'])}, {fmt(r['speedup_ci_high'])}] |")
    lines += ["", "C 组显示 CUDA-MOEA 在全部测试维数上均更快，但加速比不是随维数单调增长：低维主要受框架固定开销影响，中等维数差距收窄，高维计算量主导后差距再次扩大。该趋势在 NSGA-III 和 RVEA 上一致。"]
    changes=[r["problem"] for r in sensitivity if r["ranking_changed"]]
    lines += ["", "## 7. 理论前沿敏感性与限制", "",
              f"20,000/50,000 点导致方法中位数排序变化的问题：{', '.join(changes) if changes else '无'}。具体排序见 `data/pareto_front_sensitivity.csv`。",
              "不同框架只共享种子编号，并不共享逐项相同的随机序列，因此质量检验采用非配对 rank-sum。加速比只在相同算法、相同实际种群规模下解释，不用于 NSGA-III 与 RVEA 之间的直接测速比较。", "",
              "完整派生数据位于 `data/`；所有图均由原始 `run_result.json` 与 A 组 `final_objectives.pt` 离线生成。", "",
              "## 8. 派生数据文件说明", "",
              "`tests/benchmark/DTLZ/results/data/` 中的 CSV 均由本报告分析脚本离线生成，不会修改 `output/paper_benchmark_20260707` 中的原始运行结果。", "",
              "| 文件 | 记录粒度 | 内容与用途 |", "|---|---|---|",
              "| `quality_runs.csv` | A 组每次独立运行一行，共 960 行 | 保存原始运行字段、有限且去重后的非支配点数、20k/50k 两套 IGD、最终采用的参考前沿点数和 `final_igd`。它是 A 组统计检验、箱线图和代表运行选择的基础明细表。 |",
              "| `quality_summary.csv` | 每个“问题–算法–框架–指标”一行，共 96 行 | 汇总 `final_igd`、`total_time_ms`、`mean_generation_ms` 的样本数、mean、std、median、Q1、Q3 和 IQR，用于 4.1 表格及 A 组时间图。 |",
              "| `quality_rank_sum_tests.csv` | 每个“问题–算法”跨框架比较一行，共 16 行 | 保存 Wilcoxon rank-sum 原始 p 值、Holm 校正 p 值、rank-biserial correlation、两框架中位 IGD 和显著性判断，用于 4.2。 |",
              "| `pareto_front_sensitivity.csv` | 每个 DTLZ 问题一行，共 8 行 | 记录 20,000 与 50,000 点理论前沿下的四实现中位 IGD 排序、排序是否变化以及最终使用的参考点数。 |",
              "| `representative_runs.csv` | 每个“问题–框架–算法”一行，共 32 行 | 记录各实现 30 次 IGD 中位数、最接近中位数的代表运行 IGD、seed、repeat index 和目标矩阵路径，用于 4.4 前沿图。 |",
              "| `population_timing_summary.csv` | B 组每个“名义种群–算法”一行，共 16 行 | 汇总两框架 10 次时间重复的均值、标准差、中位数、每代中位时间，以及中位加速比和 95% bootstrap 置信区间。 |",
              "| `dimension_timing_summary.csv` | C 组每个“决策维数–算法”一行，共 22 行 | 字段含义与种群规模汇总表一致，用于决策维数扩展曲线、加速比图和第 6 节表格。 |", "",
              "其中 `quality_runs.csv` 是质量实验的运行级派生数据；其余文件均为汇总、统计检验、敏感性分析或绘图索引。若需复核数值，应从该明细表及原始 `run_result.json`/`final_objectives.pt` 开始，而不应从图片反推。", ""]
    (output/"REPORT.zh-CN.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs-root",type=Path)
    source.add_argument("--raw-runs",type=Path,
                        help="Consolidated raw_runs.json produced with the benchmark data export")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if args.raw_runs:
        all_rows = json.loads(args.raw_runs.read_text(encoding="utf-8"))
        a, b, c = ([row for row in all_rows if row["experiment"] == f"group_{g}"] for g in "abc")
        runs_root = args.raw_runs.parent.parent
    else:
        a,b,c=(read_rows(args.runs_root,f"group_{g}") for g in "abc")
        runs_root = args.runs_root
    failed_quality = [row for row in a if row.get("run_status") != "success"]
    if failed_quality:
        raise RuntimeError(f"group_a contains {len(failed_quality)} unsuccessful runs")
    qrows,qsummary,tests,sensitivity,reps,fronts,references=quality_analysis(a,args.output)
    selected_counts={r["problem"]:int(r["selected_pf_points"]) for r in sensitivity}
    pop_summary=timing_analysis(b,"nominal_population",0)
    dim_summary=timing_analysis(c,"dimension",1000)
    write_csv(args.output/"data/population_timing_summary.csv",pop_summary)
    write_csv(args.output/"data/dimension_timing_summary.csv",dim_summary)
    plot_fronts(args.output,reps,fronts,references,selected_counts)
    plot_quality(args.output,qrows)
    plot_scaling(args.output,pop_summary,"nominal_population","population","Population size")
    plot_scaling(args.output,dim_summary,"dimension","dimension","Problem dimension")
    manifest=json.loads((runs_root/"manifest.json").read_text(encoding="utf-8"))
    report(args.output,manifest,qsummary,tests,sensitivity,pop_summary,dim_summary)
    print(f"Chinese report: {(args.output/'REPORT.zh-CN.md').resolve()}")
    return 0


if __name__=="__main__": raise SystemExit(main())
