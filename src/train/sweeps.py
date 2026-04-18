"""Multi-seed sweep orchestration for managed runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..services import (
    ProjectConfig,
    RunManager,
    RunRecord,
    build_seed_sweep_summary,
    generate_seed_sweep_plots,
)
from .real_data import REAL_MNLOGIC_DATASET_NAME, execute_real_mnlogic_managed_run
from .runner import RunExecutionResult
from .synthetic import SYNTHETIC_DATASET_NAME, execute_synthetic_managed_run


@dataclass(frozen=True)
class SeedSweepResult:
    """Outcome of a multi-seed managed-run sweep."""

    sweep_name: str
    records: list[RunRecord]
    csv_path: str
    json_path: str
    plot_paths: list[str]
    summary: dict[str, Any]


def execute_seed_sweep(
    run_manager: RunManager,
    *,
    project_config: ProjectConfig,
    dataset: str,
    model_family: str,
    benchmark_suite: str,
    supervision: str,
    seeds: Sequence[int],
    sweep_name: str,
    training_overrides: Mapping[str, Any] | None = None,
    limit_per_split: int | None = None,
    metric_names: Sequence[str] | None = None,
) -> SeedSweepResult:
    """Execute one managed run per seed and write aggregated summary exports."""

    resolved_seeds = _normalize_seeds(seeds)
    run_results: list[RunExecutionResult] = []

    for seed in resolved_seeds:
        run_name = f"{sweep_name}_seed_{seed}"
        if dataset == REAL_MNLOGIC_DATASET_NAME:
            result = execute_real_mnlogic_managed_run(
                run_manager,
                project_config=project_config,
                model_family=model_family,
                seed=seed,
                benchmark_suite=benchmark_suite,
                supervision=supervision,
                run_name=run_name,
                training_overrides=training_overrides,
                limit_per_split=limit_per_split,
            )
        elif dataset == SYNTHETIC_DATASET_NAME:
            result = execute_synthetic_managed_run(
                run_manager,
                project_config=project_config,
                model_family=model_family,
                seed=seed,
                benchmark_suite=benchmark_suite,
                supervision=supervision,
                run_name=run_name,
                training_overrides=training_overrides,
            )
        else:
            raise ValueError(f"Unsupported dataset for seed sweep: {dataset}")

        run_results.append(result)

    records = [result.record for result in run_results]
    summary_payload = build_seed_sweep_summary(records, metric_names=metric_names)
    export_basename = _build_seed_sweep_export_basename(
        sweep_name=sweep_name,
        dataset=dataset,
        model_family=model_family,
        benchmark_suite=benchmark_suite,
        supervision=supervision,
    )
    csv_path, json_path = _write_seed_sweep_exports(
        run_manager.paths.summaries_root,
        export_basename=export_basename,
        dataset=dataset,
        model_family=model_family,
        benchmark_suite=benchmark_suite,
        supervision=supervision,
        seeds=resolved_seeds,
        records=records,
        summary_payload=summary_payload,
    )
    plot_assets = generate_seed_sweep_plots(
        summary_payload.get("aggregate_rows", []),
        output_basename=export_basename,
        plots_root=project_config.paths.plots_root,
    )

    return SeedSweepResult(
        sweep_name=sweep_name,
        records=records,
        csv_path=str(csv_path),
        json_path=str(json_path),
        plot_paths=[asset["path"] for asset in plot_assets],
        summary=summary_payload,
    )


def _normalize_seeds(seeds: Sequence[int]) -> list[int]:
    if not seeds:
        raise ValueError("Seed sweep requires at least one seed.")

    resolved = [int(seed) for seed in seeds]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Seed sweep received duplicate seeds.")
    return resolved


def _build_seed_sweep_export_basename(
    *,
    sweep_name: str,
    dataset: str,
    model_family: str,
    benchmark_suite: str,
    supervision: str,
) -> str:
    safe_name = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in sweep_name.strip().lower()
    ).strip("_")
    safe_name = safe_name or "seed_sweep"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"seed_sweep__{safe_name}__{dataset}__{model_family}__"
        f"{benchmark_suite}__{supervision}__{timestamp}"
    )


def _write_seed_sweep_exports(
    summaries_root: Path,
    *,
    export_basename: str,
    dataset: str,
    model_family: str,
    benchmark_suite: str,
    supervision: str,
    seeds: Sequence[int],
    records: Sequence[RunRecord],
    summary_payload: Mapping[str, Any],
) -> tuple[Path, Path]:
    summaries_root.mkdir(parents=True, exist_ok=True)
    csv_path = summaries_root / f"{export_basename}.csv"
    json_path = summaries_root / f"{export_basename}.json"

    aggregate_rows = list(summary_payload.get("aggregate_rows", []))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric_name",
                "label",
                "count",
                "mean",
                "std",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    payload = {
        "sweep_name": export_basename,
        "selection": {
            "dataset": dataset,
            "model_family": model_family,
            "benchmark_suite": benchmark_suite,
            "supervision": supervision,
            "seeds": list(seeds),
        },
        "run_ids": [record.run_id for record in records],
        "metric_names": list(summary_payload.get("metric_names", [])),
        "aggregate_rows": aggregate_rows,
        "run_rows": list(summary_payload.get("run_rows", [])),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    return csv_path, json_path
