"""Minimal FastAPI backend for run control, stored-result inspection, and UI serving."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from ..services import RunRecord, get_project_config, get_run_manager
from ..services.catalog import list_available_options
from ..train.real_data import execute_real_mnlogic_managed_run
from ..train.synthetic import execute_synthetic_managed_run
from ..ui import create_ui_router
from .schemas import (
    AvailableOptionsResponse,
    HealthResponse,
    RealMNLogicRunLaunchRequest,
    RealMNLogicRunLaunchResponse,
    RunCompareRequest,
    RunCompareResponse,
    RunListResponse,
    RunSelectionResponse,
    RunSnapshotResponse,
    RunSummaryResponse,
    SyntheticRunLaunchRequest,
    SyntheticRunLaunchResponse,
)


def create_app() -> FastAPI:
    """Create the Phase 11 backend and UI application."""

    project_config = get_project_config()
    app = FastAPI(
        title="Thesis Benchmarking Backend API",
        version="0.12.0",
        description=(
            "Minimal API for listing stored runs, inspecting run artifacts, "
            "comparing runs, launching synthetic or real-MNLogic managed runs, "
            "and serving the minimal UI."
        ),
    )
    static_dir = Path(__file__).resolve().parents[1] / "ui" / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    app.mount(
        "/plots",
        StaticFiles(directory=str(project_config.paths.plots_root)),
        name="plots",
    )
    app.include_router(create_ui_router())

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            project_name=project_config.name,
            phase=project_config.phase,
            timestamp=_utc_now(),
        )

    @app.get("/api/v1/options", response_model=AvailableOptionsResponse)
    def available_options() -> AvailableOptionsResponse:
        project_config = get_project_config()
        options = list_available_options()
        return AvailableOptionsResponse(
            **options,
            defaults=RunSelectionResponse(
                dataset=project_config.defaults.dataset,
                model_family=project_config.defaults.model_family,
                benchmark_suite=project_config.defaults.benchmark_suite,
                supervision=project_config.defaults.supervision,
                seed=project_config.defaults.seed,
            ),
        )

    @app.get("/api/v1/runs", response_model=RunListResponse)
    def list_runs(
        status: str | None = None,
        dataset: str | None = None,
        model_family: str | None = None,
        benchmark_suite: str | None = None,
        supervision: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
    ) -> RunListResponse:
        records = get_run_manager().list_runs(
            status=status,
            dataset=dataset,
            model_family=model_family,
            benchmark_suite=benchmark_suite,
            supervision=supervision,
        )
        records = records[:limit]
        return RunListResponse(
            total_runs=len(records),
            runs=[_record_to_summary(record) for record in records],
        )

    @app.get("/api/v1/runs/{run_id}", response_model=RunSummaryResponse)
    def get_run(run_id: str) -> RunSummaryResponse:
        return _record_to_summary(_require_run(run_id))

    @app.get(
        "/api/v1/runs/{run_id}/snapshot/{snapshot_type}",
        response_model=RunSnapshotResponse,
    )
    def get_run_snapshot(
        run_id: str,
        snapshot_type: Literal["metadata", "metrics", "artifacts", "config"],
    ) -> RunSnapshotResponse:
        record = _require_run(run_id)
        payload = _load_snapshot_payload(record, snapshot_type)
        return RunSnapshotResponse(
            run_id=run_id,
            snapshot_type=snapshot_type,
            payload=payload,
        )

    @app.post("/api/v1/runs/compare", response_model=RunCompareResponse)
    def compare_runs(request: RunCompareRequest) -> RunCompareResponse:
        manager = get_run_manager()
        try:
            paths = manager.compare_runs(
                request.run_ids,
                metric_names=request.metric_names,
                output_basename=request.output_basename,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        comparison_payload = _load_json(Path(paths["json_path"]))
        return RunCompareResponse(
            csv_path=paths["csv_path"],
            json_path=paths["json_path"],
            metric_names=[
                str(metric_name)
                for metric_name in comparison_payload.get("metric_names", [])
            ],
            rows=[
                row
                for row in comparison_payload.get("rows", [])
                if isinstance(row, dict)
            ],
        )

    @app.post(
        "/api/v1/runs/launch/synthetic",
        response_model=SyntheticRunLaunchResponse,
    )
    def launch_synthetic_run(
        request: SyntheticRunLaunchRequest,
    ) -> SyntheticRunLaunchResponse:
        manager = get_run_manager()
        project_config = get_project_config()
        options = list_available_options()
        if request.model_family not in options["model_families"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model_family: {request.model_family}",
            )

        try:
            result = execute_synthetic_managed_run(
                manager,
                project_config=project_config,
                model_family=request.model_family,
                seed=request.seed,
                benchmark_suite=request.benchmark_suite,
                supervision=request.supervision,
                run_name=request.run_name,
                training_overrides=request.training_overrides,
                total_samples=request.total_samples,
                train_size=request.train_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SyntheticRunLaunchResponse(
            run=_record_to_summary(result.record),
            checkpoint_path=result.checkpoint_path,
            training_metrics=result.training_metrics,
            evaluation_metrics=result.evaluation_metrics,
        )

    @app.post(
        "/api/v1/runs/launch/mnlogic",
        response_model=RealMNLogicRunLaunchResponse,
    )
    def launch_real_mnlogic_run(
        request: RealMNLogicRunLaunchRequest,
    ) -> RealMNLogicRunLaunchResponse:
        manager = get_run_manager()
        project_config = get_project_config()
        options = list_available_options()
        if request.model_family not in options["model_families"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model_family: {request.model_family}",
            )

        try:
            result = execute_real_mnlogic_managed_run(
                manager,
                project_config=project_config,
                model_family=request.model_family,
                seed=request.seed,
                benchmark_suite=request.benchmark_suite,
                supervision=request.supervision,
                run_name=request.run_name,
                training_overrides=request.training_overrides,
                limit_per_split=request.limit_per_split,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        config_payload = _load_snapshot_payload(result.record, "config")
        dataset_warnings = []
        if isinstance(config_payload.get("dataset"), dict):
            dataset_warnings = [
                str(item)
                for item in config_payload["dataset"].get("warnings", [])
            ]

        return RealMNLogicRunLaunchResponse(
            run=_record_to_summary(result.record),
            checkpoint_path=result.checkpoint_path,
            training_metrics=result.training_metrics,
            evaluation_metrics=result.evaluation_metrics,
            dataset_warnings=dataset_warnings,
        )

    return app

def _require_run(run_id: str) -> RunRecord:
    try:
        return get_run_manager().get_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _record_to_summary(record: RunRecord) -> RunSummaryResponse:
    return RunSummaryResponse(
        run_id=record.run_id,
        run_name=record.run_name,
        selection=RunSelectionResponse(
            dataset=record.selection.dataset,
            model_family=record.selection.model_family,
            benchmark_suite=record.selection.benchmark_suite,
            supervision=record.selection.supervision,
            seed=record.selection.seed,
        ),
        status=record.status,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        run_dir=record.run_dir,
        config_path=record.config_path,
        metadata_path=record.metadata_path,
        metrics_path=record.metrics_path,
        artifacts_path=record.artifacts_path,
        checkpoint_path=record.checkpoint_path,
        error_message=record.error_message,
        metrics=record.metrics,
        artifacts=record.artifacts,
    )


def _load_snapshot_payload(record: RunRecord, snapshot_type: str) -> dict:
    path_map = {
        "metadata": Path(record.metadata_path),
        "metrics": Path(record.metrics_path),
        "artifacts": Path(record.artifacts_path),
        "config": Path(record.config_path),
    }
    path = path_map[snapshot_type]
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Snapshot file not found for {snapshot_type}: {path}",
        )

    if snapshot_type == "config":
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    else:
        payload = _load_json(path)

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Invalid snapshot payload for {snapshot_type}: {path}",
        )
    return payload


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


app = create_app()
