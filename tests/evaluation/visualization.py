"""Pareto-front visualization helpers."""

from itertools import cycle
from pathlib import Path
from typing import Any


def plot_fronts(
    objectives: Any,
    reference_front: Any,
    output_path: Path | None = None,
    title: str | None = None,
    reference_alpha: float = 0.30,
    result_size: float = 9.0,
    result_alpha: float = 0.65,
    result_linewidth: float = 0.35,
    reference_label: str = "Theory PF",
    result_label: str = "CUDA-MOEA PF",
) -> None:
    """Plot a result front against its theoretical reference front."""

    import matplotlib.pyplot as plt

    reference = reference_front.detach().cpu()
    objectives = objectives.detach().cpu()
    dimensions = objectives.shape[1]
    if dimensions == 3:
        figure = plt.figure(figsize=(7, 6))
        axis = figure.add_subplot(111, projection="3d")
        axis.scatter(
            *[reference[:, index].tolist() for index in range(3)],
            s=3,
            alpha=reference_alpha,
            color="#4C78A8",
            label=reference_label,
            depthshade=False,
        )
        axis.scatter(
            *[objectives[:, index].tolist() for index in range(3)],
            s=result_size,
            alpha=result_alpha,
            facecolors="none",
            edgecolors="#F28E2B",
            linewidths=result_linewidth,
            label=result_label,
            depthshade=False,
        )
        axis.set_zlabel("f3")
    else:
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(
            reference[:, 0].tolist(),
            reference[:, 1].tolist(),
            s=3,
            alpha=reference_alpha,
            color="#4C78A8",
            label=reference_label,
        )
        axis.scatter(
            objectives[:, 0].tolist(),
            objectives[:, 1].tolist(),
            s=result_size,
            alpha=result_alpha,
            facecolors="none",
            edgecolors="#F28E2B",
            linewidths=result_linewidth,
            label=result_label,
        )
        if dimensions > 3 and title is None:
            axis.set_title("First two objectives")
    axis.set_xlabel("f1")
    axis.set_ylabel("f2")
    if title:
        axis.set_title(title)
    axis.legend(loc="best")
    figure.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
    plt.show()
    plt.close(figure)


def plot_front_comparison(
    fronts: dict[str, Any],
    reference_front: Any | None = None,
    output_path: Path | None = None,
    title: str | None = None,
    reference_label: str = "Empirical nondominated PF",
) -> None:
    """Plot multiple approximation fronts against a shared reference front."""

    import matplotlib.pyplot as plt

    reference = (
        None if reference_front is None else reference_front.detach().cpu()
    )
    values = {name: front.detach().cpu() for name, front in fronts.items()}
    if not values:
        raise ValueError("fronts must contain at least one approximation set")
    dimensions = next(iter(values.values())).shape[1]
    colors = ("#F28E2B", "#59A14F", "#E15759", "#B07AA1")
    markers = ("s", "o", "D", "P")
    if dimensions == 3:
        figure = plt.figure(figsize=(9, 7.5))
        axis = figure.add_subplot(111, projection="3d")
        if reference is not None:
            axis.scatter(
                *[reference[:, index].tolist() for index in range(3)],
                s=58,
                color="#4C78A8",
                alpha=0.88,
                marker="^",
                edgecolors="#244A73",
                linewidths=0.8,
                label=reference_label,
                depthshade=False,
            )
        for color, marker, (name, front) in zip(
            cycle(colors), cycle(markers), values.items()
        ):
            axis.scatter(
                *[front[:, index].tolist() for index in range(3)],
                s=46,
                color=color,
                edgecolors="#2B2B2B",
                linewidths=0.75,
                marker=marker,
                alpha=0.88,
                label=name,
                depthshade=False,
            )
        axis.set_zlabel("reward 3")
    else:
        figure, axis = plt.subplots(figsize=(9, 6.5))
        if reference is not None:
            axis.scatter(
                reference[:, 0].tolist(),
                reference[:, 1].tolist(),
                s=58,
                color="#4C78A8",
                alpha=0.88,
                marker="^",
                edgecolors="#244A73",
                linewidths=0.8,
                label=reference_label,
            )
        for color, marker, (name, front) in zip(
            cycle(colors), cycle(markers), values.items()
        ):
            axis.scatter(
                front[:, 0].tolist(),
                front[:, 1].tolist(),
                s=46,
                color=color,
                edgecolors="#2B2B2B",
                linewidths=0.75,
                marker=marker,
                alpha=0.88,
                label=name,
            )
        if dimensions > 3 and title is None:
            axis.set_title("First two rewards")
    axis.set_xlabel("Reward objective 1")
    axis.set_ylabel("Reward objective 2")
    if title:
        axis.set_title(title)
    axis.grid(True, alpha=0.22)
    axis.legend(loc="best", framealpha=0.95, markerscale=1.15)
    figure.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
    plt.show()
    plt.close(figure)
