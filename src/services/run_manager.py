"""Run registry and result-storage services."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import yaml

from .config import ProjectConfig


RUN_STATUS_CREATED = "created"
RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"


@dataclass(frozen=True)
class RunSelection:
    """Stable selection fields that define one experiment run."""

    dataset: str
    model_family: str
    benchmark_suite: str
    supervision: str
    seed: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunSelection":
        return cls(
            dataset=str(payload["dataset"]),
            model_family=str(payload["model_family"]),
            benchmark_suite=str(payload["benchmark_suite"]),
            supervision=str(payload["supervision"]),
            seed=int(payload["seed"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model_family": self.model_family,
            "benchmark_suite": self.benchmark_suite,
            "supervision": self.supervision,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RunRecord:
    """Serializable snapshot of one stored experiment run."""

    run_id: str
    run_name: str
    selection: RunSelection
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
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "selection": self.selection.to_dict(),
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "run_dir": self.run_dir,
            "config_path": self.config_path,
            "metadata_path": self.metadata_path,
            "metrics_path": self.metrics_path,
            "artifacts_path": self.artifacts_path,
            "checkpoint_path": self.checkpoint_path,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }


class RunManager:
    """Persist run metadata, metrics, and comparison exports."""

    def __init__(self, project_config: ProjectConfig) -> None:
        self.project_config = project_config
        self.paths = project_config.paths
        self.sqlite_path = project_config.storage.sqlite_path

        self.paths.runs_root.mkdir(parents=True, exist_ok=True)
        self.paths.summaries_root.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def create_run(
        self,
        *,
        run_name: str,
        selection: RunSelection,
        config_snapshot: Mapping[str, Any],
    ) -> RunRecord:
        """Create a new run folder and registry entry."""

        created_at = _utc_now()
        run_id = _build_run_id(run_name)
        run_dir = self.paths.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_dir / "config_snapshot.yaml"
        metadata_path = run_dir / "metadata.json"
        metrics_path = run_dir / "metrics.json"
        artifacts_path = run_dir / "artifacts.json"

        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(config_snapshot), handle, sort_keys=False)

        self._write_json(metadata_path, {})
        self._write_json(metrics_path, {})
        self._write_json(artifacts_path, {})

        record = RunRecord(
            run_id=run_id,
            run_name=run_name,
            selection=selection,
            status=RUN_STATUS_CREATED,
            created_at=created_at,
            started_at=None,
            finished_at=None,
            run_dir=str(run_dir),
            config_path=str(config_path),
            metadata_path=str(metadata_path),
            metrics_path=str(metrics_path),
            artifacts_path=str(artifacts_path),
            checkpoint_path=None,
            error_message=None,
        )

        self._write_metadata(record)
        self._append_event(run_dir / "events.jsonl", "created", {"run_id": run_id})

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, run_name, status, dataset, model_family, benchmark_suite,
                    supervision, seed, created_at, started_at, finished_at, run_dir,
                    config_path, metadata_path, metrics_path, artifacts_path,
                    checkpoint_path, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.run_name,
                    record.status,
                    record.selection.dataset,
                    record.selection.model_family,
                    record.selection.benchmark_suite,
                    record.selection.supervision,
                    record.selection.seed,
                    record.created_at,
                    record.started_at,
                    record.finished_at,
                    record.run_dir,
                    record.config_path,
                    record.metadata_path,
                    record.metrics_path,
                    record.artifacts_path,
                    record.checkpoint_path,
                    record.error_message,
                ),
            )

        self._write_registry_exports()
        return record

    def mark_run_started(self, run_id: str) -> RunRecord:
        """Mark a created run as running."""

        record = self.get_run(run_id)
        started_at = _utc_now()
        updated = RunRecord(
            run_id=record.run_id,
            run_name=record.run_name,
            selection=record.selection,
            status=RUN_STATUS_RUNNING,
            created_at=record.created_at,
            started_at=started_at,
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
        self._update_run_row(updated)
        self._write_metadata(updated)
        self._append_event(Path(updated.run_dir) / "events.jsonl", "started", {})
        self._write_registry_exports()
        return updated

    def complete_run(
        self,
        run_id: str,
        *,
        metrics: Mapping[str, float],
        artifacts: Mapping[str, str | Path] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> RunRecord:
        """Store final metrics and mark the run as completed."""

        record = self.get_run(run_id)
        finished_at = _utc_now()
        artifact_payload = {
            artifact_name: str(Path(artifact_path).expanduser().resolve())
            for artifact_name, artifact_path in dict(artifacts or {}).items()
        }
        checkpoint_value = (
            str(Path(checkpoint_path).expanduser().resolve())
            if checkpoint_path is not None
            else record.checkpoint_path
        )
        numeric_metrics = {
            str(metric_name): float(metric_value)
            for metric_name, metric_value in metrics.items()
        }

        updated = RunRecord(
            run_id=record.run_id,
            run_name=record.run_name,
            selection=record.selection,
            status=RUN_STATUS_COMPLETED,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=finished_at,
            run_dir=record.run_dir,
            config_path=record.config_path,
            metadata_path=record.metadata_path,
            metrics_path=record.metrics_path,
            artifacts_path=record.artifacts_path,
            checkpoint_path=checkpoint_value,
            error_message=None,
            metrics=numeric_metrics,
            artifacts=artifact_payload,
        )

        self._write_json(Path(updated.metrics_path), numeric_metrics)
        self._write_json(Path(updated.artifacts_path), artifact_payload)
        self._write_metadata(updated)
        self._append_event(
            Path(updated.run_dir) / "events.jsonl",
            "completed",
            {
                "metric_count": len(numeric_metrics),
                "artifact_count": len(artifact_payload),
            },
        )

        with self._connect() as connection:
            connection.execute(
                "DELETE FROM metrics WHERE run_id = ?",
                (updated.run_id,),
            )
            connection.executemany(
                """
                INSERT INTO metrics (run_id, metric_name, metric_value)
                VALUES (?, ?, ?)
                """,
                [
                    (updated.run_id, metric_name, metric_value)
                    for metric_name, metric_value in numeric_metrics.items()
                ],
            )

        self._update_run_row(updated)
        self._write_registry_exports()
        return updated

    def fail_run(self, run_id: str, error_message: str) -> RunRecord:
        """Mark a run as failed and store the error message."""

        record = self.get_run(run_id)
        finished_at = _utc_now()
        updated = RunRecord(
            run_id=record.run_id,
            run_name=record.run_name,
            selection=record.selection,
            status=RUN_STATUS_FAILED,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=finished_at,
            run_dir=record.run_dir,
            config_path=record.config_path,
            metadata_path=record.metadata_path,
            metrics_path=record.metrics_path,
            artifacts_path=record.artifacts_path,
            checkpoint_path=record.checkpoint_path,
            error_message=str(error_message),
            metrics=record.metrics,
            artifacts=record.artifacts,
        )

        self._update_run_row(updated)
        self._write_metadata(updated)
        self._append_event(
            Path(updated.run_dir) / "events.jsonl",
            "failed",
            {"error_message": updated.error_message},
        )
        self._write_registry_exports()
        return updated

    def get_run(self, run_id: str) -> RunRecord:
        """Load one run record and its metric map from storage."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown run_id: {run_id}")

            metric_rows = connection.execute(
                "SELECT metric_name, metric_value FROM metrics WHERE run_id = ? ORDER BY metric_name",
                (run_id,),
            ).fetchall()

        artifacts = self._read_json(Path(row["artifacts_path"]))
        metrics = {entry["metric_name"]: float(entry["metric_value"]) for entry in metric_rows}
        return self._row_to_record(row, metrics=metrics, artifacts=artifacts)

    def list_runs(
        self,
        *,
        status: str | None = None,
        dataset: str | None = None,
        model_family: str | None = None,
        benchmark_suite: str | None = None,
        supervision: str | None = None,
    ) -> list[RunRecord]:
        """List runs with optional selection/status filters."""

        query = "SELECT * FROM runs WHERE 1=1"
        parameters: list[Any] = []

        for field_name, field_value in (
            ("status", status),
            ("dataset", dataset),
            ("model_family", model_family),
            ("benchmark_suite", benchmark_suite),
            ("supervision", supervision),
        ):
            if field_value is not None:
                query += f" AND {field_name} = ?"
                parameters.append(field_value)

        query += " ORDER BY created_at DESC"

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
            metric_rows = connection.execute(
                "SELECT run_id, metric_name, metric_value FROM metrics ORDER BY run_id, metric_name"
            ).fetchall()

        metrics_by_run: dict[str, dict[str, float]] = {}
        for metric_row in metric_rows:
            metrics_by_run.setdefault(metric_row["run_id"], {})[metric_row["metric_name"]] = float(
                metric_row["metric_value"]
            )

        records: list[RunRecord] = []
        for row in rows:
            artifacts = self._read_json(Path(row["artifacts_path"]))
            records.append(
                self._row_to_record(
                    row,
                    metrics=metrics_by_run.get(row["run_id"], {}),
                    artifacts=artifacts,
                )
            )
        return records

    def compare_runs(
        self,
        run_ids: Sequence[str],
        *,
        metric_names: Sequence[str] | None = None,
        output_basename: str | None = None,
    ) -> dict[str, str]:
        """Write CSV and JSON comparison exports for a fixed set of runs."""

        records = [self.get_run(run_id) for run_id in run_ids]
        if not records:
            raise ValueError("compare_runs requires at least one run_id")

        if metric_names is None:
            metric_union = set()
            for record in records:
                metric_union.update(record.metrics.keys())
            metric_names = sorted(metric_union)
        else:
            metric_names = [str(metric_name) for metric_name in metric_names]

        output_stem = output_basename or f"comparison__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        csv_path = self.paths.summaries_root / f"{output_stem}.csv"
        json_path = self.paths.summaries_root / f"{output_stem}.json"

        rows: list[dict[str, Any]] = []
        for record in records:
            row = {
                "run_id": record.run_id,
                "run_name": record.run_name,
                "status": record.status,
                "dataset": record.selection.dataset,
                "model_family": record.selection.model_family,
                "benchmark_suite": record.selection.benchmark_suite,
                "supervision": record.selection.supervision,
                "seed": record.selection.seed,
            }
            for metric_name in metric_names:
                row[metric_name] = record.metrics.get(metric_name)
            rows.append(row)

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        payload = {
            "run_ids": list(run_ids),
            "metric_names": list(metric_names),
            "rows": rows,
        }
        self._write_json(json_path, payload)

        return {"csv_path": str(csv_path), "json_path": str(json_path)}

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    model_family TEXT NOT NULL,
                    benchmark_suite TEXT NOT NULL,
                    supervision TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    run_dir TEXT NOT NULL,
                    config_path TEXT NOT NULL,
                    metadata_path TEXT NOT NULL,
                    metrics_path TEXT NOT NULL,
                    artifacts_path TEXT NOT NULL,
                    checkpoint_path TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    PRIMARY KEY (run_id, metric_name),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
                CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset);
                CREATE INDEX IF NOT EXISTS idx_runs_model_family ON runs(model_family);
                CREATE INDEX IF NOT EXISTS idx_metrics_metric_name ON metrics(metric_name);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _update_run_row(self, record: RunRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, started_at = ?, finished_at = ?, checkpoint_path = ?, error_message = ?
                WHERE run_id = ?
                """,
                (
                    record.status,
                    record.started_at,
                    record.finished_at,
                    record.checkpoint_path,
                    record.error_message,
                    record.run_id,
                ),
            )

    def _write_metadata(self, record: RunRecord) -> None:
        metadata = {
            "run_id": record.run_id,
            "run_name": record.run_name,
            "selection": record.selection.to_dict(),
            "status": record.status,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "run_dir": record.run_dir,
            "config_path": record.config_path,
            "metrics_path": record.metrics_path,
            "artifacts_path": record.artifacts_path,
            "checkpoint_path": record.checkpoint_path,
            "error_message": record.error_message,
        }
        self._write_json(Path(record.metadata_path), metadata)

    def _write_registry_exports(self) -> None:
        records = self.list_runs()
        json_path = self.paths.summaries_root / "run_registry.json"
        csv_path = self.paths.summaries_root / "run_registry.csv"

        payload = [record.to_dict() for record in records]
        self._write_json(json_path, payload)

        fieldnames = [
            "run_id",
            "run_name",
            "status",
            "dataset",
            "model_family",
            "benchmark_suite",
            "supervision",
            "seed",
            "created_at",
            "started_at",
            "finished_at",
            "checkpoint_path",
            "error_message",
        ]
        metric_names = sorted(
            {
                metric_name
                for record in records
                for metric_name in record.metrics.keys()
            }
        )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames + metric_names)
            writer.writeheader()
            for record in records:
                row = {
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
                    "checkpoint_path": record.checkpoint_path,
                    "error_message": record.error_message,
                }
                row.update(record.metrics)
                writer.writerow(row)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
        *,
        metrics: Mapping[str, float],
        artifacts: Mapping[str, str],
    ) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            run_name=row["run_name"],
            selection=RunSelection(
                dataset=row["dataset"],
                model_family=row["model_family"],
                benchmark_suite=row["benchmark_suite"],
                supervision=row["supervision"],
                seed=int(row["seed"]),
            ),
            status=row["status"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            run_dir=row["run_dir"],
            config_path=row["config_path"],
            metadata_path=row["metadata_path"],
            metrics_path=row["metrics_path"],
            artifacts_path=row["artifacts_path"],
            checkpoint_path=row["checkpoint_path"],
            error_message=row["error_message"],
            metrics={str(key): float(value) for key, value in metrics.items()},
            artifacts={str(key): str(value) for key, value in artifacts.items()},
        )

    @staticmethod
    def _append_event(path: Path, event_type: str, payload: Mapping[str, Any]) -> None:
        event_payload = {
            "timestamp": _utc_now(),
            "event": event_type,
            **dict(payload),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload, sort_keys=True) + "\n")


def _build_run_id(run_name: str) -> str:
    safe_name = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in run_name.strip().lower())
    safe_name = safe_name.strip("_") or "run"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}__{safe_name}__{uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
