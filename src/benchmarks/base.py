"""Benchmark suite adapter contracts and config parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..eval import evaluate_named_splits


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    """Typed benchmark-suite configuration loaded from YAML."""

    name: str
    root_dir: str | None
    supported_datasets: tuple[str, ...]
    suite_metrics: tuple[str, ...]
    notes: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkSuiteConfig":
        dataset_support = payload.get("dataset_support", {})
        if not isinstance(dataset_support, Mapping):
            raise ValueError("benchmark config dataset_support must be a mapping")

        supported_datasets = tuple(
            sorted(str(dataset_name) for dataset_name in dataset_support.keys())
        )
        suite_metrics = payload.get("suite_metrics", [])
        if not isinstance(suite_metrics, list):
            raise ValueError("benchmark config suite_metrics must be a list")

        paths = payload.get("paths", {})
        root_dir = None
        if isinstance(paths, Mapping) and "root_dir" in paths:
            raw_root_dir = paths["root_dir"]
            if raw_root_dir not in {None, ""}:
                root_dir = str(raw_root_dir)

        return cls(
            name=str(payload.get("name", "")),
            root_dir=root_dir,
            supported_datasets=supported_datasets,
            suite_metrics=tuple(str(metric_name) for metric_name in suite_metrics),
            notes=dict(payload.get("notes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root_dir": self.root_dir,
            "supported_datasets": list(self.supported_datasets),
            "suite_metrics": list(self.suite_metrics),
            "notes": dict(self.notes),
        }


class BenchmarkSuiteAdapter(ABC):
    """Shared interface for benchmark-suite adapters."""

    suite_name: str

    def __init__(self, config: BenchmarkSuiteConfig) -> None:
        self.config = config

    def list_datasets(self) -> list[str]:
        """List datasets supported by the suite config."""

        return list(self.config.supported_datasets)

    def prepare_dataset(
        self,
        dataset_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Validate dataset support and return benchmark-specific split plans."""

        normalized_dataset_name = self._normalize_dataset_name(dataset_name)
        if normalized_dataset_name not in self.config.supported_datasets:
            raise ValueError(
                f"Benchmark suite '{self.suite_name}' does not support dataset '{dataset_name}'."
            )
        return self._prepare_dataset(normalized_dataset_name, **kwargs)

    def run_evaluation(
        self,
        model: Any,
        split_batches: Mapping[str, list[dict[str, Any]]],
        *,
        seed: int | None = None,
        label_loss_weight: float = 1.0,
        concept_loss_weight: float = 1.0,
        external_environment: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        """Run shared evaluation and attach suite-specific metrics."""

        metrics = evaluate_named_splits(
            model,
            split_batches,
            seed=seed,
            label_loss_weight=label_loss_weight,
            concept_loss_weight=concept_loss_weight,
        )
        metrics.update(self.compute_suite_specific_metrics(metrics))
        if external_environment:
            metrics.update(self.compute_external_environment_metrics(external_environment))
        return metrics

    def build_external_environment(
        self,
        *,
        dataset_name: str,
        model_family: str,
    ) -> dict[str, Any]:
        """Return optional benchmark-environment metadata for one run."""

        return {}

    def compute_external_environment_metrics(
        self,
        environment: Mapping[str, Any],
    ) -> dict[str, float]:
        """Convert benchmark-environment metadata into numeric stored metrics."""

        return {}

    @abstractmethod
    def _prepare_dataset(
        self,
        dataset_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return benchmark-specific train/eval split structure."""

    @abstractmethod
    def compute_suite_specific_metrics(
        self,
        metrics: Mapping[str, float],
    ) -> dict[str, float]:
        """Compute suite-level metrics from shared evaluation outputs."""

    @staticmethod
    def _normalize_dataset_name(dataset_name: str) -> str:
        dataset_name = str(dataset_name).strip().lower()
        if dataset_name.startswith("synthetic_"):
            return dataset_name.removeprefix("synthetic_")
        return dataset_name

    @property
    def root_dir(self) -> Path | None:
        """Return the configured external root if one exists."""

        if not self.config.root_dir:
            return None
        path = Path(self.config.root_dir).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (Path(__file__).resolve().parents[2] / path).resolve()
