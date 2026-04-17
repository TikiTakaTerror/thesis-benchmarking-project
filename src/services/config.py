"""Project-level config loading for storage and run services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "base.yaml"


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project paths used by run storage and related services."""

    project_root: Path
    data_root: Path
    raw_data_root: Path
    processed_data_root: Path
    external_root: Path
    results_root: Path
    runs_root: Path
    summaries_root: Path
    plots_root: Path

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        project_root: Path,
    ) -> "ProjectPaths":
        payload = dict(payload or {})

        def resolve(relative_path: str, default: str) -> Path:
            return (project_root / str(payload.get(relative_path, default))).resolve()

        return cls(
            project_root=project_root.resolve(),
            data_root=resolve("data_root", "data"),
            raw_data_root=resolve("raw_data_root", "data/raw"),
            processed_data_root=resolve("processed_data_root", "data/processed"),
            external_root=resolve("external_root", "external"),
            results_root=resolve("results_root", "results"),
            runs_root=resolve("runs_root", "results/runs"),
            summaries_root=resolve("summaries_root", "results/summaries"),
            plots_root=resolve("plots_root", "results/plots"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "data_root": str(self.data_root),
            "raw_data_root": str(self.raw_data_root),
            "processed_data_root": str(self.processed_data_root),
            "external_root": str(self.external_root),
            "results_root": str(self.results_root),
            "runs_root": str(self.runs_root),
            "summaries_root": str(self.summaries_root),
            "plots_root": str(self.plots_root),
        }


@dataclass(frozen=True)
class ProjectDefaults:
    """Default run selection fields defined in the base config."""

    benchmark_suite: str
    dataset: str
    model_family: str
    supervision: str
    seed: int
    device: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ProjectDefaults":
        payload = dict(payload or {})
        return cls(
            benchmark_suite=str(payload.get("benchmark_suite", "rsbench")),
            dataset=str(payload.get("dataset", "mnlogic")),
            model_family=str(payload.get("model_family", "pipeline")),
            supervision=str(payload.get("supervision", "full")),
            seed=int(payload.get("seed", 42)),
            device=str(payload.get("device", "cpu")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_suite": self.benchmark_suite,
            "dataset": self.dataset,
            "model_family": self.model_family,
            "supervision": self.supervision,
            "seed": self.seed,
            "device": self.device,
        }


@dataclass(frozen=True)
class ProjectStorage:
    """Storage backend settings for run metadata and summaries."""

    run_registry_backend: str
    sqlite_path: Path

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        project_root: Path,
    ) -> "ProjectStorage":
        payload = dict(payload or {})
        sqlite_path = (project_root / str(payload.get("sqlite_path", "results/experiment_registry.sqlite3"))).resolve()
        return cls(
            run_registry_backend=str(payload.get("run_registry_backend", "sqlite")),
            sqlite_path=sqlite_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_registry_backend": self.run_registry_backend,
            "sqlite_path": str(self.sqlite_path),
        }


@dataclass(frozen=True)
class ProjectConfig:
    """Typed base project config used by the run-management layer."""

    name: str
    phase: int
    description: str
    paths: ProjectPaths
    defaults: ProjectDefaults
    storage: ProjectStorage

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        project_root: Path,
    ) -> "ProjectConfig":
        project_payload = dict(payload.get("project", {}))
        return cls(
            name=str(project_payload.get("name", "thesis-benchmarking-project")),
            phase=int(project_payload.get("phase", 0)),
            description=str(project_payload.get("description", "")),
            paths=ProjectPaths.from_dict(payload.get("paths"), project_root=project_root),
            defaults=ProjectDefaults.from_dict(payload.get("defaults")),
            storage=ProjectStorage.from_dict(
                payload.get("storage"),
                project_root=project_root,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": {
                "name": self.name,
                "phase": self.phase,
                "description": self.description,
            },
            "paths": self.paths.to_dict(),
            "defaults": self.defaults.to_dict(),
            "storage": self.storage.to_dict(),
        }


def load_project_config(path: str | Path | None = None) -> ProjectConfig:
    """Load the typed project config from the base YAML file."""

    config_path = Path(path).expanduser().resolve() if path is not None else BASE_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid project config file: {config_path}")

    return ProjectConfig.from_dict(payload, project_root=PROJECT_ROOT)
