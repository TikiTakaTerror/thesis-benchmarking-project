"""Helpers for UI-friendly run comparison and benchmark summary views."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha1
from statistics import mean
from typing import Any, Sequence

from .run_manager import RunRecord


DEFAULT_COMPARISON_METRICS = [
    "benchmark_primary_score",
    "test_accuracy",
    "id_accuracy",
    "ood_accuracy",
    "test_concept_accuracy",
    "id_concept_accuracy",
    "run_runtime_seconds",
]

METRIC_LABELS = {
    "benchmark_primary_score": "Primary Score",
    "test_accuracy": "Test Accuracy",
    "id_accuracy": "ID Accuracy",
    "ood_accuracy": "OOD Accuracy",
    "test_concept_accuracy": "Test Concept Accuracy",
    "id_concept_accuracy": "ID Concept Accuracy",
    "run_runtime_seconds": "Run Time (s)",
}


def build_comparison_table(
    records: Sequence[RunRecord],
    *,
    metric_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return comparison cards, columns, and rows for a fixed set of runs."""

    resolved_metric_names = list(metric_names or DEFAULT_COMPARISON_METRICS)
    rows: list[dict[str, Any]] = []
    for record in records:
        metric_values = {
            metric_name: record.metrics.get(metric_name) for metric_name in resolved_metric_names
        }
        rows.append(
            {
                "run_id": record.run_id,
                "run_name": record.run_name,
                "status": record.status,
                "model_family": record.selection.model_family,
                "benchmark_suite": record.selection.benchmark_suite,
                "supervision": record.selection.supervision,
                "seed": record.selection.seed,
                "metrics": metric_values,
            }
        )

    return {
        "cards": _build_comparison_cards(records),
        "metric_columns": [
            {
                "name": metric_name,
                "label": METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title()),
            }
            for metric_name in resolved_metric_names
        ],
        "rows": rows,
    }


def build_benchmark_summary(records: Sequence[RunRecord]) -> dict[str, Any]:
    """Aggregate runs into benchmark-facing summary cards and table rows."""

    total_runs = len(records)
    completed_runs = sum(1 for record in records if record.status == "completed")
    benchmark_groups: dict[tuple[str, str, str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        key = (
            record.selection.benchmark_suite,
            record.selection.dataset,
            record.selection.model_family,
            record.selection.supervision,
        )
        benchmark_groups[key].append(record)

    best_primary_score = _best_metric_any(records, ["benchmark_primary_score", "test_accuracy", "id_accuracy"])

    cards = [
        {"label": "Stored Runs", "value": str(total_runs)},
        {"label": "Completed", "value": str(completed_runs)},
        {"label": "Benchmark Groups", "value": str(len(benchmark_groups))},
        {"label": "Best Primary Score", "value": best_primary_score},
    ]

    rows: list[dict[str, Any]] = []
    for key in sorted(benchmark_groups.keys()):
        suite, dataset, model_family, supervision = key
        grouped_records = sorted(
            benchmark_groups[key],
            key=lambda record: record.created_at,
            reverse=True,
        )

        rows.append(
            {
                "benchmark_suite": suite,
                "dataset": dataset,
                "model_family": model_family,
                "supervision": supervision,
                "run_count": len(grouped_records),
                "completed_runs": sum(
                    1 for record in grouped_records if record.status == "completed"
                ),
                "failed_runs": sum(1 for record in grouped_records if record.status == "failed"),
                "best_primary_score": _best_metric_any(
                    grouped_records,
                    ["benchmark_primary_score", "test_accuracy", "id_accuracy"],
                ),
                "mean_primary_score": _mean_metric_any(
                    grouped_records,
                    ["benchmark_primary_score", "test_accuracy", "id_accuracy"],
                ),
                "mean_concept_accuracy": _mean_metric_any(
                    grouped_records,
                    ["test_concept_accuracy", "id_concept_accuracy"],
                ),
                "mean_runtime_seconds": _mean_metric(
                    grouped_records, "run_runtime_seconds"
                ),
                "latest_run_id": grouped_records[0].run_id,
                "latest_run_name": grouped_records[0].run_name,
                "latest_created_at": grouped_records[0].created_at,
                "compare_run_ids": [
                    record.run_id
                    for record in grouped_records
                    if record.status == "completed"
                ][:3],
            }
        )

    return {"cards": cards, "rows": rows}


def build_comparison_export_basename(run_ids: Sequence[str]) -> str:
    """Return a stable basename so repeated UI comparisons reuse the same export paths."""

    digest = sha1("|".join(run_ids).encode("utf-8")).hexdigest()[:10]
    return f"ui_compare__{digest}"


def _build_comparison_cards(records: Sequence[RunRecord]) -> list[dict[str, str]]:
    runtime_values = [
        record.metrics["run_runtime_seconds"]
        for record in records
        if "run_runtime_seconds" in record.metrics
    ]
    return [
        {"label": "Selected Runs", "value": str(len(records))},
        {
            "label": "Best Primary Score",
            "value": _best_metric_any(records, ["benchmark_primary_score", "test_accuracy", "id_accuracy"]),
        },
        {
            "label": "Best Concept Acc.",
            "value": _best_metric_any(records, ["test_concept_accuracy", "id_concept_accuracy"]),
        },
        {
            "label": "Fastest Run (s)",
            "value": "n/a"
            if not runtime_values
            else f"{min(runtime_values):.4f}",
        },
    ]


def _best_metric(records: Sequence[RunRecord], metric_name: str) -> str:
    values = [
        float(record.metrics[metric_name])
        for record in records
        if metric_name in record.metrics
    ]
    if not values:
        return "n/a"
    return f"{max(values):.4f}"


def _best_metric_any(records: Sequence[RunRecord], metric_names: Sequence[str]) -> str:
    values = []
    for record in records:
        metric_value = _first_metric_value(record, metric_names)
        if metric_value is not None:
            values.append(metric_value)
    if not values:
        return "n/a"
    return f"{max(values):.4f}"


def _mean_metric(records: Sequence[RunRecord], metric_name: str) -> str:
    values = [
        float(record.metrics[metric_name])
        for record in records
        if metric_name in record.metrics
    ]
    if not values:
        return "n/a"
    return f"{mean(values):.4f}"


def _mean_metric_any(records: Sequence[RunRecord], metric_names: Sequence[str]) -> str:
    values = []
    for record in records:
        metric_value = _first_metric_value(record, metric_names)
        if metric_value is not None:
            values.append(metric_value)
    if not values:
        return "n/a"
    return f"{mean(values):.4f}"


def _first_metric_value(record: RunRecord, metric_names: Sequence[str]) -> float | None:
    for metric_name in metric_names:
        if metric_name in record.metrics:
            return float(record.metrics[metric_name])
    return None
