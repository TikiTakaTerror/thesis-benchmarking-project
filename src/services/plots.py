"""Plot-generation helpers for comparison views, benchmark summaries, and sweeps."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .reporting import METRIC_LABELS
from .run_manager import RunRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_ROOT = PROJECT_ROOT / "results" / "plots"


def generate_comparison_plots(
    records: Sequence[RunRecord],
    *,
    metric_names: Sequence[str],
    output_basename: str,
    plots_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Generate comparison plots for a fixed set of run records."""

    resolved_plots_root = _resolve_plots_root(plots_root)
    resolved_plots_root.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, str]] = []

    score_metrics = [
        metric_name
        for metric_name in metric_names
        if metric_name in {
            "benchmark_primary_score",
            "test_accuracy",
            "id_accuracy",
            "ood_accuracy",
            "test_concept_accuracy",
            "id_concept_accuracy",
            "rsbench_shortcut_gap",
            "rsbench_concept_gap",
        }
    ]
    if score_metrics:
        score_path = resolved_plots_root / f"{output_basename}__comparison_scores.png"
        _plot_metric_grid(
            records,
            metrics=score_metrics[:6],
            title="Run Comparison: Score Metrics",
            output_path=score_path,
        )
        assets.append(_build_plot_asset("Comparison Scores", score_path, resolved_plots_root))

    has_robustness = any(
        "id_accuracy" in record.metrics and "ood_accuracy" in record.metrics
        for record in records
    )
    if has_robustness:
        robustness_path = resolved_plots_root / f"{output_basename}__comparison_robustness.png"
        _plot_robustness_bars(
            records,
            title="Run Comparison: ID vs OOD Robustness",
            output_path=robustness_path,
        )
        assets.append(_build_plot_asset("Comparison Robustness", robustness_path, resolved_plots_root))

    return assets


def generate_benchmark_summary_plots(
    summary_rows: Sequence[dict[str, Any]],
    *,
    output_basename: str,
    plots_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Generate grouped benchmark-summary plots from row dictionaries."""

    resolved_plots_root = _resolve_plots_root(plots_root)
    resolved_plots_root.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        return []

    overview_path = resolved_plots_root / f"{output_basename}__benchmark_overview.png"
    _plot_benchmark_overview(summary_rows, output_path=overview_path)
    assets = [_build_plot_asset("Benchmark Overview", overview_path, resolved_plots_root)]

    has_shortcut_gap = any(
        row.get("mean_shortcut_gap_value") is not None for row in summary_rows
    )
    if has_shortcut_gap:
        shortcut_path = resolved_plots_root / f"{output_basename}__shortcut_gap.png"
        _plot_benchmark_shortcut_gap(summary_rows, output_path=shortcut_path)
        assets.append(_build_plot_asset("Benchmark Shortcut Gap", shortcut_path, resolved_plots_root))

    return assets


def generate_seed_sweep_plots(
    aggregate_rows: Sequence[dict[str, Any]],
    *,
    output_basename: str,
    plots_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Generate a mean/std plot from seed-sweep aggregate rows."""

    resolved_plots_root = _resolve_plots_root(plots_root)
    resolved_plots_root.mkdir(parents=True, exist_ok=True)
    if not aggregate_rows:
        return []

    path = resolved_plots_root / f"{output_basename}__seed_sweep.png"
    _plot_seed_sweep_summary(aggregate_rows, output_path=path)
    return [_build_plot_asset("Seed Sweep Summary", path, resolved_plots_root)]


def _plot_metric_grid(
    records: Sequence[RunRecord],
    *,
    metrics: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    cols = 2 if len(metrics) > 1 else 1
    rows = math.ceil(len(metrics) / cols)
    figure, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.6 * rows))
    axes_list = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]
    run_labels = [_short_run_label(record) for record in records]

    for axis, metric_name in zip(axes_list, metrics):
        values = [
            float(record.metrics.get(metric_name, float("nan")))
            if metric_name in record.metrics
            else float("nan")
            for record in records
        ]
        cleaned_values = [0.0 if math.isnan(value) else value for value in values]
        axis.bar(run_labels, cleaned_values, color="#1f6a5a")
        axis.set_title(METRIC_LABELS.get(metric_name, metric_name))
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        if "accuracy" in metric_name or "score" in metric_name or "drop" in metric_name or "gap" in metric_name:
            axis.set_ylim(bottom=min(-0.05, min(cleaned_values) - 0.05), top=max(1.05, max(cleaned_values) + 0.05))

    for axis in axes_list[len(metrics):]:
        axis.axis("off")

    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_robustness_bars(
    records: Sequence[RunRecord],
    *,
    title: str,
    output_path: Path,
) -> None:
    labels = [_short_run_label(record) for record in records]
    id_values = [float(record.metrics.get("id_accuracy", 0.0)) for record in records]
    ood_values = [float(record.metrics.get("ood_accuracy", 0.0)) for record in records]
    gap_values = [float(record.metrics.get("rsbench_shortcut_gap", 0.0)) for record in records]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.36
    indices = list(range(len(records)))

    axes[0].bar([index - width / 2 for index in indices], id_values, width=width, label="ID", color="#1f6a5a")
    axes[0].bar([index + width / 2 for index in indices], ood_values, width=width, label="OOD", color="#8fb7ab")
    axes[0].set_xticks(indices, labels, rotation=20)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("ID vs OOD Accuracy")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend()

    axes[1].bar(labels, gap_values, color="#7f2e1d")
    axes[1].axhline(0.0, color="#6f6659", linewidth=1)
    axes[1].set_title("Shortcut Gap (ID - OOD)")
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_benchmark_overview(
    summary_rows: Sequence[dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    labels = [_summary_label(row) for row in summary_rows]
    primary_scores = [
        float(row["mean_primary_score_value"])
        for row in summary_rows
        if row.get("mean_primary_score_value") is not None
    ]
    usable_rows = [
        row for row in summary_rows if row.get("mean_primary_score_value") is not None
    ]
    labels = [_summary_label(row) for row in usable_rows]
    runtimes = [
        float(row["mean_runtime_seconds_value"])
        if row.get("mean_runtime_seconds_value") is not None
        else 0.0
        for row in usable_rows
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(labels, primary_scores, color="#1f6a5a")
    axes[0].set_ylim(0.0, max(1.05, max(primary_scores, default=1.0) + 0.05))
    axes[0].set_title("Mean Primary Score by Benchmark Group")
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)

    axes[1].bar(labels, runtimes, color="#c5b89a")
    axes[1].set_title("Mean Run Time (s) by Benchmark Group")
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    figure.suptitle("Benchmark Summary Overview", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_benchmark_shortcut_gap(
    summary_rows: Sequence[dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    usable_rows = [
        row for row in summary_rows if row.get("mean_shortcut_gap_value") is not None
    ]
    labels = [_summary_label(row) for row in usable_rows]
    gaps = [float(row["mean_shortcut_gap_value"]) for row in usable_rows]

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(labels, gaps, color="#7f2e1d")
    axis.axhline(0.0, color="#6f6659", linewidth=1)
    axis.set_title("Mean Shortcut Gap by Benchmark Group")
    axis.tick_params(axis="x", labelrotation=30)
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_seed_sweep_summary(
    aggregate_rows: Sequence[dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    labels = [str(row["label"]) for row in aggregate_rows]
    means = [float(row["mean"]) for row in aggregate_rows]
    stds = [float(row["std"]) for row in aggregate_rows]

    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    axis.bar(labels, means, yerr=stds, capsize=4, color="#1f6a5a")
    axis.set_title("Seed Sweep Mean +/- Std")
    axis.tick_params(axis="x", labelrotation=30)
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _build_plot_asset(label: str, path: Path, plots_root: Path) -> dict[str, str]:
    return {
        "label": label,
        "path": str(path.resolve()),
        "url": _plot_url(path, plots_root),
    }


def _plot_url(path: Path, plots_root: Path) -> str:
    relative_path = path.resolve().relative_to(plots_root.resolve())
    return "/plots/" + relative_path.as_posix()


def _resolve_plots_root(plots_root: str | Path | None) -> Path:
    if plots_root is None:
        return DEFAULT_PLOTS_ROOT.resolve()
    path = Path(plots_root).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _short_run_label(record: RunRecord) -> str:
    return f"{record.selection.model_family}-{record.selection.seed}"


def _summary_label(row: dict[str, Any]) -> str:
    return (
        f"{row['benchmark_suite']}\n"
        f"{row['model_family']}\n"
        f"{row['supervision']}"
    )
