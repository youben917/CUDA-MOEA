"""Charts and paginated PDF for summarize.py --visualize; no GPU needed.

Requires matplotlib and an installed CJK font. Set CUDA_MOEA_REPORT_FONT to
override the font path. The PDF renderer supports this report's Markdown subset.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cuda-moea-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np


BLUE, TEAL = "#3768ad", "#008578"
INK, MUTED = "#1c3047", "#53657a"


def configure():
    override = os.environ.get("CUDA_MOEA_REPORT_FONT")
    candidates = [Path(override)] if override else [
        Path("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_path = next((p for p in candidates if p.is_file()), None)
    if font_path is None:
        raise RuntimeError("Install a CJK font or set CUDA_MOEA_REPORT_FONT to its path")
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", family], "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "text.color": INK, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c6d0dc", "xtick.color": MUTED, "ytick.color": MUTED,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def generate_charts(data, output):
    configure()
    target = output / "figures"
    target.mkdir(exist_ok=True)
    cases = data["cases"]
    populations, dimensions = data["protocol"]["populations"], data["protocol"]["dimensions"]
    evaluations = {}
    runs = {}
    for case in cases:
        evaluations.setdefault((case["population"], case["dimension"]), case)
        runs[case["algorithm"], case["population"], case["dimension"]] = case
    figures = {}

    def save(fig, name):
        fig.savefig(target / f"{name}.png", dpi=220)
        fig.savefig(target / f"{name}.pdf", metadata={"Title": f"CUDA-MOEA {name}"})
        figures[name] = fig

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    all_ratios = [c["run_speedup"] for c in cases]
    run_norm = colors.TwoSlopeNorm(vmin=min(0.95, min(all_ratios)), vcenter=1,
                                  vmax=max(1.26, max(all_ratios)))
    for ax, kind in zip(axes, ("批量评估", "NSGA3", "RVEA")):
        values = np.array([[evaluations[n, d]["evaluation_speedup"] if kind == "批量评估"
                            else runs[kind, n, d]["run_speedup"] for d in dimensions] for n in populations])
        if kind == "批量评估":
            im = ax.imshow(values, cmap="Blues", vmin=0, vmax=max(16, values.max()))
        else:
            im = ax.imshow(values, cmap="RdBu", norm=run_norm)
        ax.set(xticks=range(len(dimensions)), xticklabels=dimensions,
               yticks=range(len(populations)), yticklabels=populations,
               xlabel="变量维度 D", title=kind)
        ax.set_ylabel("种群大小 N")
        for (i, j), value in np.ndenumerate(values):
            rgba = im.cmap(im.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, f"{value:.2f}×" if kind == "批量评估" else f"{value:.3f}×",
                    ha="center", va="center", color="white" if luminance < .5 else INK, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=.75, pad=.035).set_label("Python / 原生")
    save(fig, "speedup")

    def bars(ax, selected, key, scale=1):
        x = np.arange(len(dimensions))
        for offset, impl, label, color in [(-.19, "pytorch", "Python / PyTorch", BLUE),
                                           (.19, "native", "C++ / CUDA", TEAL)]:
            stats = [c[key][impl] for c in selected]
            med = np.array([s["median"] * scale for s in stats])
            err = [[(s["median"] - s["min"]) * scale for s in stats],
                   [(s["max"] - s["median"]) * scale for s in stats]]
            ax.bar(x + offset, med, width=.35, color=color, label=label,
                   yerr=err, capsize=3, error_kw={"elinewidth": .8, "ecolor": INK})
        ax.set(xticks=x, xticklabels=dimensions, xlabel="变量维度 D", ylim=(0, None))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, len(populations), figsize=(10.5, 3.7), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), populations):
        bars(ax, [evaluations[n, d] for d in dimensions], "evaluation_wall_ms", 1000)
        ax.set_title(f"N = {n}")
    np.atleast_1d(axes)[0].set_ylabel("批量评估墙钟时间（μs / 次）")
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=.08, right=.98, bottom=.18, top=.8, wspace=.12)
    save(fig, "evaluation")

    fig, axes = plt.subplots(2, len(populations), figsize=(10.5, 5.3), squeeze=False)
    for row, algorithm in enumerate(("NSGA3", "RVEA")):
        for ax, n in zip(axes[row], populations):
            bars(ax, [runs[algorithm, n, d] for d in dimensions], "run_wall_ms")
            ax.set_title(f"{algorithm} · N = {n}")
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[row, 0].set_ylabel(f"{data['protocol']['generations']} 代墙钟时间（ms）")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=.08, right=.98, bottom=.1, top=.87, hspace=.65, wspace=.32)
    save(fig, "optimization")
    return figures


def plain(text):
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)
    return text.replace("**", "").replace("`", "")


class ReportPDF:
    """A4 layout in typographic points, with measured CJK wrapping."""

    width, height = 595.28, 841.89
    left, right, top, bottom = 43, 43, 58, 48

    def __init__(self):
        self.pages = []
        self.new_page()

    def new_page(self):
        self.fig = Figure(figsize=(self.width / 72, self.height / 72), dpi=100)
        FigureCanvasAgg(self.fig)
        self.pages.append(self.fig)
        self.y = self.height - self.top
        self.text_at("CUDA-MOEA  /  自定义问题性能实测", self.left, self.height - 27, 8, MUTED)
        self.fig.add_artist(plt.Line2D([self.left / self.width, 1-self.right / self.width],
                                      [1-38/self.height]*2, transform=self.fig.transFigure,
                                      color="#d8e2ed", linewidth=.7))

    def text_at(self, text, x, y, size=10, color=INK):
        return self.fig.text(x / self.width, y / self.height, text, fontsize=size,
                             va="top", color=color, parse_math=False)

    def ensure(self, height):
        if self.y - height < self.bottom:
            self.new_page()

    def wrap(self, text, size, width):
        renderer = self.fig.canvas.get_renderer()
        font = font_manager.FontProperties(family=plt.rcParams["font.family"], size=size)
        # Preserve words and paths where possible, while allowing long hashes and
        # command arguments to wrap without running beyond the page margin.
        tokens = re.findall(r"[\x21-\x7e]+|[^\x21-\x7e]", text)
        lines, current = [], ""

        def fits(value):
            return renderer.get_text_width_height_descent(value, font, False)[0] * 72 / self.fig.dpi <= width

        for token in tokens:
            if fits(current + token):
                current += token
                continue
            if current.strip():
                lines.append(current.rstrip())
                current = ""
            for char in token.lstrip():
                if not fits(current + char):
                    lines.append(current)
                    current = ""
                current += char
        if current.strip():
            lines.append(current.rstrip())
        for i in range(1, len(lines)):
            if lines[i][0] in "，。；：！？、）】" and len(lines[i-1]) > 1:
                lines[i] = lines[i-1][-1] + lines[i]
                lines[i-1] = lines[i-1][:-1]
        return lines

    def paragraph(self, text, size=10, gap=5, color=INK):
        rows = self.wrap(text, size, self.width-self.left-self.right)
        for row in rows:
            self.ensure(size * 1.6)
            self.text_at(row, self.left, self.y, size, color)
            self.y -= size * 1.6
        self.y -= gap

    def heading(self, text, level):
        self.ensure(65)
        self.y -= 4
        self.paragraph(text, size=19 if level == 1 else 14, gap=9)

    def table(self, rows):
        ncols = len(rows[0])
        widths = [1, 1, 2, 2, 1.3] if ncols == 5 else [1.2, 1, 1, 2, 2, 1.3]
        widths = np.array(widths) / sum(widths) * (self.width-self.left-self.right)
        row_height = 19
        self.ensure(len(rows)*row_height+12)
        for index, cells in enumerate(rows):
            color = "#e7eef6" if index == 0 else ("#f4f7fa" if index % 2 else "white")
            self.fig.add_artist(plt.Rectangle((self.left/self.width, (self.y-row_height)/self.height),
                                             sum(widths)/self.width, row_height/self.height,
                                             transform=self.fig.transFigure, color=color, linewidth=0))
            x = self.left
            for cell, width in zip(cells, widths):
                self.text_at(cell, x+5, self.y-3, 8.5)
                x += width
            self.y -= row_height
        self.y -= 10

    def chart(self, figure):
        figure.set_dpi(220)
        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba()).copy()
        width = self.width-self.left-self.right
        height = width * pixels.shape[0] / pixels.shape[1]
        self.ensure(height+12)
        ax = self.fig.add_axes([self.left/self.width, (self.y-height)/self.height,
                                width/self.width, height/self.height])
        ax.imshow(pixels)
        ax.axis("off")
        self.y -= height+10

    def save(self, path):
        with PdfPages(path, metadata={"Title": "Python/PyTorch 与原生 C++/CUDA 自定义问题性能实测",
                                      "Author": "CUDA-MOEA", "Subject": "BiSphere GPU benchmark"}) as pdf:
            for i, fig in enumerate(self.pages, 1):
                fig.text(.5, 25/self.height, f"{i} / {len(self.pages)}", ha="center", fontsize=8, color=MUTED)
                pdf.savefig(fig)


def export_pdf(report, figures):
    document = ReportPDF()
    lines = report.read_text(encoding="utf-8").splitlines()
    page_sections = {"环境与口径", "批量评估", "完整优化", "编译与缓存", "复现与数据"}
    index, in_code = 0, False
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line:
            continue
        if line.startswith("```"):
            in_code = not in_code
        elif in_code:
            document.paragraph(line, size=8, color=MUTED, gap=10)
        elif line.startswith("#"):
            level = len(line)-len(line.lstrip("#"))
            title = line.lstrip("# ")
            if title in page_sections:
                document.new_page()
            document.heading(title, level)
        elif line.startswith("!["):
            name = Path(re.search(r"\(([^)]+)\)", line).group(1)).stem
            document.chart(figures[name])
        elif line.startswith("|"):
            rows = [[cell.strip() for cell in line.strip("|").split("|")]]
            index += 1  # Markdown alignment row.
            while index < len(lines) and lines[index].startswith("|"):
                rows.append([cell.strip() for cell in lines[index].strip("|").split("|")])
                index += 1
            document.table(rows)
        else:
            document.paragraph(plain(line), size=9 if line.startswith("图 ") else 10,
                               color=MUTED if line.startswith("图 ") else INK)
    output = report.with_suffix(".pdf")
    document.save(output)
    for fig in figures.values():
        plt.close(fig)
    print(output)
