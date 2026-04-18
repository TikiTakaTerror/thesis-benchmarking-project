"""Pydantic schemas for the backend API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    project_name: str
    phase: int
    timestamp: str


class RunSelectionResponse(BaseModel):
    dataset: str
    model_family: str
    benchmark_suite: str
    supervision: str
    seed: int


class AvailableOptionsResponse(BaseModel):
    datasets: list[str]
    model_families: list[str]
    benchmark_suites: list[str]
    supervision_settings: list[str]
    run_presets: list[str]
    defaults: RunSelectionResponse


class RunSummaryResponse(BaseModel):
    run_id: str
    run_name: str
    selection: RunSelectionResponse
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    run_dir: str
    config_path: str
    metadata_path: str
    metrics_path: str
    artifacts_path: str
    checkpoint_path: str | None
    error_message: str | None
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)


class RunListResponse(BaseModel):
    total_runs: int
    runs: list[RunSummaryResponse]


class RunSnapshotResponse(BaseModel):
    run_id: str
    snapshot_type: str
    payload: dict[str, Any]


class RunCompareRequest(BaseModel):
    run_ids: list[str] = Field(min_length=1)
    metric_names: list[str] | None = None
    output_basename: str | None = None


class RunCompareResponse(BaseModel):
    csv_path: str
    json_path: str
    metric_names: list[str]
    rows: list[dict[str, Any]]


class SyntheticRunLaunchRequest(BaseModel):
    model_family: str = "pipeline"
    seed: int = 42
    benchmark_suite: str = "rsbench"
    supervision: str = "full"
    run_name: str | None = None
    training_overrides: dict[str, float] = Field(default_factory=dict)
    total_samples: int = 64
    train_size: int = 48


class SyntheticRunLaunchResponse(BaseModel):
    run: RunSummaryResponse
    checkpoint_path: str
    training_metrics: dict[str, float]
    evaluation_metrics: dict[str, float]


class RealMNLogicRunLaunchRequest(BaseModel):
    model_family: str = "pipeline"
    seed: int = 42
    benchmark_suite: str = "rsbench"
    supervision: str = "full"
    run_name: str | None = None
    training_overrides: dict[str, float] = Field(default_factory=dict)
    limit_per_split: int | None = Field(default=None, ge=1)


class RealMNLogicRunLaunchResponse(BaseModel):
    run: RunSummaryResponse
    checkpoint_path: str
    training_metrics: dict[str, float]
    evaluation_metrics: dict[str, float]
    dataset_warnings: list[str] = Field(default_factory=list)
