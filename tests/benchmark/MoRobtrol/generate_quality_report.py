#!/usr/bin/env python3
"""Build a Markdown report from the saved outputs of plot_quality_curves.ipynb."""

from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "plot_quality_curves.ipynb"
RESULTS = ROOT / "results"
IMAGES = RESULTS / "images"

ENVIRONMENTS = ["mo_hopper_m3", "mo_swimmer"]
OUTPUT_CELLS = {
    5: "generation",
    7: "time",
    14: "final_front",
}


def image_outputs(cell: dict) -> list[str]:
    images = []
    for output in cell.get("outputs", []):
        encoded = output.get("data", {}).get("image/png")
        if encoded:
            images.append("".join(encoded) if isinstance(encoded, list) else encoded)
    return images


def html_outputs(cell: dict) -> list[str]:
    tables = []
    for output in cell.get("outputs", []):
        html = output.get("data", {}).get("text/html")
        if html:
            tables.append("".join(html) if isinstance(html, list) else html)
    return tables


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    IMAGES.mkdir(parents=True, exist_ok=True)

    paths: dict[tuple[str, str], str] = {}
    for cell_index, chart_kind in OUTPUT_CELLS.items():
        images = image_outputs(cells[cell_index])
        if len(images) != len(ENVIRONMENTS):
            raise RuntimeError(
                f"cell {cell_index} expected {len(ENVIRONMENTS)} charts, found {len(images)}"
            )
        for environment, encoded in zip(ENVIRONMENTS, images):
            filename = f"{environment}_{chart_kind}.png"
            (IMAGES / filename).write_bytes(base64.b64decode(encoded))
            paths[(environment, chart_kind)] = f"images/{filename}"

    quality_tables = html_outputs(cells[9])
    target_tables = html_outputs(cells[11])
    if len(quality_tables) != 2 or len(target_tables) != 2:
        raise RuntimeError("notebook must contain both HV and EU tables")
    sections = []
    for environment in ENVIRONMENTS:
        sections.append(
            f"""## {environment}

### 按代数的收敛质量

![{environment} Generation–HV/EU]({paths[(environment, 'generation')]})

### 按算法时间的收敛质量

![{environment} Time–HV/EU]({paths[(environment, 'time')]})

### 最终非支配前沿

![{environment} 最终非支配前沿]({paths[(environment, 'final_front')]})
"""
        )

    report = f"""# MoRobtrol D 组质量测试报告

[English](REPORT.md) | 中文

生成日期：2026-07-15  
数据范围：`mo_hopper_m3`、`mo_swimmer`；CUDA-MOEA 与 EvoX；NSGA-III 与 RVEA；每组 10 个 seed；名义种群 16384；第 0–100 代（每 5 代检查点）。

## 结论摘要

- **Hopper / NSGA-III：CUDA-MOEA 综合占优。** 第 100 代中位 HV 为 4.846e9（EvoX 4.538e9），EU 为 0.4906（EvoX 0.4856），100 代中位算法时间为 9.99 min（EvoX 11.65 min）。在相同的 2.93、5.85、8.75 min 预算下，CUDA-MOEA 的中位 HV 分别高 9.5%、7.4%、13.3%，中位 EU 分别高 3.4%、2.0%、3.8%。达到 EvoX 最终中位目标时，CUDA-MOEA 的 HV/EU 达到率分别为 90%/80%，EvoX 为 70%/60%；已达目标 seed 的时间中位数加速分别为 1.59×/1.52×。
- **Hopper / RVEA：后程 EvoX 质量更高。** EvoX 第 100 代中位 HV/EU 为 1.318e10/0.7051，CUDA-MOEA 为 1.229e10/0.6995。相同时间下 CUDA-MOEA 仅在 2.87 min 的早期预算领先（HV +5.7%、EU +2.0%）；到 5.73 和 8.59 min，EvoX 的 HV 分别高 13.5%、14.7%，EU 分别高 7.0%、5.4%。CUDA-MOEA 100 代时间略短（11.17 min 对 11.46 min），但未达到 EvoX 最终中位 HV，不能将小幅时间优势解释为同质量加速。
- **Swimmer / NSGA-III：最终质量基本持平，CUDA-MOEA 时间效率更高。** 第 100 代中位 HV 为 33.87 对 34.06，EU 为 0.74095 对 0.74092；CUDA-MOEA 100 代中位时间 11.59 min，EvoX 为 19.19 min（约 1.66× 总时间差）。在相同的 4.84 和 9.64 min 预算下，CUDA-MOEA 的中位 HV 分别高 10.1%、7.7%，中位 EU 分别高 4.3%、1.7%。
- **Swimmer / RVEA：CUDA-MOEA 在最终质量和等时质量上均显著占优。** 第 100 代中位 HV 为 11.43（EvoX 0.7765），EU 为 0.5163（EvoX 0.05566）；两者总时间接近。在相同的 2.96、5.92、8.88 min 预算下，CUDA-MOEA 的中位 HV 是 EvoX 的 26.6×、17.2×、15.1×，中位 EU 是 9.4×、8.3×、9.1×。由于以 EvoX 最终值定义的目标在初始检查点即可达到，相关 Time-to-target 的 0 min 结果只表示目标区分度不足，不代表无限加速。

## 判读口径

- 曲线实线/虚线为 10 个 seed 的中位数，阴影为四分位距（IQR）；HV 与 EU 均为越大越好。
- 时间仅统计优化算法的检查点累计时间，不含独立评估、指标计算、写盘和绘图。
- MoRobtrol 优化阶段包含仿真环境中的策略评估。仿真评估在每代耗时中占比较大且两框架都必须承担，因此会稀释 CUDA-MOEA 在进化算子上的速度优势；种群越大，选择、排序和种群操作的并行计算占比越高，速度优势才越容易体现。对于单步仿真更耗时的环境，共同的评估成本占比更高，观察到的整体速度优势通常会进一步减小。
- Quality-at-time 只使用两套实现共同可观测时间范围内的预算；`missing` 表示该预算下无可用检查点的 seed 数。
- Time-to-target 的目标取 EvoX 最终中位数及其 90%，仅在已观测的 100 代内判断；时间统计只覆盖实际达到目标的 seed。
- 最终前沿使用各“环境–实现–算法”中第 100 代 HV 最接近 10-seed 中位数的代表运行。

{''.join(sections)}

## Quality-at-time

### HV

{quality_tables[0]}

### EU

{quality_tables[1]}

## Time-to-target

### HV

{target_tables[0]}

### EU

{target_tables[1]}

## 数据与复现

- 原始派生表位于 [`data/`](data/)；完整数值应以 CSV 为准。
- 图表和表格由 [`plot_quality_curves.ipynb`](../plot_quality_curves.ipynb) 的已保存输出提取，本报告生成器只负责整理，不重新优化或独立评估。
- 重新执行 notebook 后，运行 `python tests/benchmark/MoRobtrol/generate_quality_report.py` 可刷新本报告。
"""
    (RESULTS / "REPORT.zh-CN.md").write_text(report, encoding="utf-8")
    print(f"Wrote {RESULTS / 'REPORT.zh-CN.md'} and {len(paths)} images")


if __name__ == "__main__":
    main()
