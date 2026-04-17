"""Minimal server-rendered UI routes for Phase 10."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..services.runtime import get_project_config, get_run_manager
from ..services.catalog import list_available_options
from ..train.synthetic import SYNTHETIC_DATASET_NAME, execute_synthetic_managed_run


UI_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(UI_ROOT / "templates"))


def create_ui_router() -> APIRouter:
    """Create the minimal Phase 10 UI router."""

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


def _format_metric_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.4f}"
    return str(value)
