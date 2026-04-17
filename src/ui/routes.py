"""Server-rendered UI routes for the minimal frontend and comparison views."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..services.runtime import get_project_config, get_run_manager
from ..services.catalog import list_available_options
from ..services.reporting import (
    build_benchmark_summary,
    build_comparison_export_basename,
    build_comparison_table,
)
from ..train.synthetic import SYNTHETIC_DATASET_NAME, execute_synthetic_managed_run


UI_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(UI_ROOT / "templates"))


def create_ui_router() -> APIRouter:
    """Create the Phase 11 UI router."""

    router = APIRouter(include_in_schema=False)

    @router.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        project_config = get_project_config()
        run_manager = get_run_manager()
        runs = run_manager.list_runs()[:12]
        options = list_available_options()
        context = {
            "request": request,
            "page_title": "Experiment Control",
            "project_name": project_config.name,
            "phase": project_config.phase,
            "launch_dataset": SYNTHETIC_DATASET_NAME,
            "configured_datasets": options["datasets"],
            "model_families": options["model_families"],
            "benchmark_suites": options["benchmark_suites"],
            "supervision_settings": options["supervision_settings"],
            "defaults": {
                "model_family": project_config.defaults.model_family,
                "benchmark_suite": project_config.defaults.benchmark_suite,
                "supervision": project_config.defaults.supervision,
                "seed": project_config.defaults.seed,
            },
            "summary": _build_dashboard_summary(runs),
            "runs": [_record_to_ui_row(record) for record in runs],
            "error_message": None,
        }
        return TEMPLATES.TemplateResponse(request, "dashboard.html", context)

    @router.get("/compare", response_class=HTMLResponse)
    def compare_runs(
        request: Request,
        run_id: list[str] | None = Query(default=None),
    ) -> HTMLResponse:
        project_config = get_project_config()
        run_manager = get_run_manager()
        available_runs = run_manager.list_runs()[:20]
        selected_run_ids = _deduplicate_preserve_order(run_id or [])

        comparison = None
        export_paths = None
        error_message = None
        if selected_run_ids:
            if len(selected_run_ids) < 2:
                error_message = "Select at least two runs to open the comparison view."
            else:
                try:
                    records = [
                        run_manager.get_run(selected_id) for selected_id in selected_run_ids
                    ]
                except ValueError as exc:
                    error_message = str(exc)
                else:
                    comparison = build_comparison_table(records)
                    export_paths = run_manager.compare_runs(
                        selected_run_ids,
                        metric_names=[
                            column["name"] for column in comparison["metric_columns"]
                        ],
                        output_basename=build_comparison_export_basename(selected_run_ids),
                    )

        context = {
            "request": request,
            "page_title": "Run Comparison",
            "project_name": project_config.name,
            "phase": project_config.phase,
            "available_runs": [_record_to_ui_row(record) for record in available_runs],
            "selected_run_ids": selected_run_ids,
            "comparison": _format_comparison_payload(comparison),
            "export_paths": export_paths,
            "error_message": error_message,
        }
        return TEMPLATES.TemplateResponse(request, "compare.html", context)

    @router.get("/benchmarks", response_class=HTMLResponse)
    def benchmark_summary(request: Request) -> HTMLResponse:
        project_config = get_project_config()
        records = get_run_manager().list_runs()
        summary = build_benchmark_summary(records)
        rows = []
        for row in summary["rows"]:
            compare_ids = row["compare_run_ids"]
            compare_href = None
            if len(compare_ids) >= 2:
                compare_href = "/compare?" + urlencode(
                    [("run_id", run_id) for run_id in compare_ids]
                )
            rows.append(
                {
                    **row,
                    "compare_href": compare_href,
                    "best_test_accuracy": row["best_test_accuracy"],
                    "mean_test_accuracy": row["mean_test_accuracy"],
                    "mean_concept_accuracy": row["mean_concept_accuracy"],
                    "mean_runtime_seconds": row["mean_runtime_seconds"],
                }
            )

        context = {
            "request": request,
            "page_title": "Benchmark Summary",
            "project_name": project_config.name,
            "phase": project_config.phase,
            "summary_cards": summary["cards"],
            "summary_rows": rows,
        }
        return TEMPLATES.TemplateResponse(request, "benchmark_summary.html", context)

    @router.post("/ui/launch", response_model=None)
    def launch_from_ui(
        request: Request,
        dataset: str = Form(...),
        model_family: str = Form(...),
        benchmark_suite: str = Form(...),
        supervision: str = Form(...),
        seed: int = Form(...),
        run_name: str = Form(""),
    ) -> RedirectResponse | HTMLResponse:
        project_config = get_project_config()
        run_manager = get_run_manager()
        options = list_available_options()

        if dataset != SYNTHETIC_DATASET_NAME:
            return _render_dashboard_with_error(
                request,
                error_message=(
                    "Phase 10 can only launch the synthetic backend dataset. "
                    "Real dataset-backed launches remain pending."
                ),
            )
        if model_family not in options["model_families"]:
            return _render_dashboard_with_error(
                request,
                error_message=f"Unsupported model family: {model_family}",
            )

        result = execute_synthetic_managed_run(
            run_manager,
            project_config=project_config,
            model_family=model_family,
            seed=int(seed),
            benchmark_suite=benchmark_suite,
            supervision=supervision,
            run_name=run_name.strip() or f"ui_{model_family}_seed_{seed}",
        )
        return RedirectResponse(
            url=f"/runs/{result.record.run_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @router.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: str) -> HTMLResponse:
        record = get_run_manager().get_run(run_id)
        primary_metrics = _build_primary_metrics(record.metrics)
        metric_rows = [
            {"name": metric_name, "value": _format_metric_value(metric_value)}
            for metric_name, metric_value in sorted(record.metrics.items())
        ]
        context = {
            "request": request,
            "page_title": f"Run {record.run_name}",
            "run": _record_to_ui_row(record),
            "primary_metrics": primary_metrics,
            "metric_rows": metric_rows,
            "artifacts": sorted(record.artifacts.items()),
        }
        return TEMPLATES.TemplateResponse(request, "run_detail.html", context)

    return router


def _render_dashboard_with_error(request: Request, *, error_message: str) -> HTMLResponse:
    project_config = get_project_config()
    run_manager = get_run_manager()
    runs = run_manager.list_runs()[:12]
    options = list_available_options()
    context = {
        "request": request,
        "page_title": "Experiment Control",
        "project_name": project_config.name,
        "phase": project_config.phase,
        "launch_dataset": SYNTHETIC_DATASET_NAME,
        "configured_datasets": options["datasets"],
        "model_families": options["model_families"],
        "benchmark_suites": options["benchmark_suites"],
        "supervision_settings": options["supervision_settings"],
        "defaults": {
            "model_family": project_config.defaults.model_family,
            "benchmark_suite": project_config.defaults.benchmark_suite,
            "supervision": project_config.defaults.supervision,
            "seed": project_config.defaults.seed,
        },
        "summary": _build_dashboard_summary(runs),
        "runs": [_record_to_ui_row(record) for record in runs],
        "error_message": error_message,
    }
    return TEMPLATES.TemplateResponse(request, "dashboard.html", context, status_code=400)


def _format_comparison_payload(comparison: dict | None) -> dict | None:
    if comparison is None:
        return None

    rows = []
    for row in comparison["rows"]:
        rows.append(
            {
                **row,
                "metrics": {
                    metric_name: _format_metric_value(metric_value)
                    for metric_name, metric_value in row["metrics"].items()
                },
            }
        )

    return {
        "cards": comparison["cards"],
        "metric_columns": comparison["metric_columns"],
        "rows": rows,
    }


def _build_dashboard_summary(records) -> list[dict[str, str]]:
    total_runs = len(records)
    completed_runs = sum(1 for record in records if record.status == "completed")
    failed_runs = sum(1 for record in records if record.status == "failed")

    latest_test_accuracy = "n/a"
    for record in records:
        if "test_accuracy" in record.metrics:
            latest_test_accuracy = _format_metric_value(record.metrics["test_accuracy"])
            break

    return [
        {"label": "Visible Runs", "value": str(total_runs)},
        {"label": "Completed", "value": str(completed_runs)},
        {"label": "Failed", "value": str(failed_runs)},
        {"label": "Latest Test Acc.", "value": latest_test_accuracy},
    ]


def _build_primary_metrics(metrics: dict[str, float]) -> list[dict[str, str]]:
    metric_order = [
        ("test_accuracy", "Test Accuracy"),
        ("test_concept_accuracy", "Test Concept Accuracy"),
        ("train_label_accuracy", "Train Label Accuracy"),
        ("run_runtime_seconds", "Run Time (s)"),
    ]
    cards: list[dict[str, str]] = []
    for metric_name, label in metric_order:
        if metric_name in metrics:
            cards.append({"label": label, "value": _format_metric_value(metrics[metric_name])})
    return cards


def _record_to_ui_row(record) -> dict[str, str | dict[str, float]]:
    return {
        "run_id": record.run_id,
        "run_name": record.run_name,
        "status": record.status,
        "dataset": record.selection.dataset,
        "model_family": record.selection.model_family,
        "benchmark_suite": record.selection.benchmark_suite,
        "supervision": record.selection.supervision,
        "seed": record.selection.seed,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "run_dir": record.run_dir,
        "checkpoint_path": record.checkpoint_path,
        "error_message": record.error_message,
        "metrics": record.metrics,
        "artifacts": record.artifacts,
        "test_accuracy": _format_metric_value(record.metrics.get("test_accuracy")),
        "test_concept_accuracy": _format_metric_value(
            record.metrics.get("test_concept_accuracy")
        ),
        "runtime_seconds": _format_metric_value(record.metrics.get("run_runtime_seconds")),
    }


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _format_metric_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.4f}"
    return str(value)
