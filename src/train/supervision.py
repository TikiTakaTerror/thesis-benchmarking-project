"""Config-driven supervision policies for managed runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml

from ..services.config import PROJECT_ROOT


SUPERVISION_CONFIG_DIR = PROJECT_ROOT / "src" / "configs" / "supervision"


@dataclass(frozen=True)
class SupervisionConfig:
    """Typed supervision policy loaded from YAML."""

    name: str
    labels: bool
    concepts: bool
    logic_constraints: bool
    concept_supervision_fraction: float
    notes: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisionConfig":
        signals = payload.get("signals", {})
        if not isinstance(signals, Mapping):
            raise ValueError("supervision config signals must be a mapping")

        concept_supervision = payload.get("concept_supervision", {})
        if not isinstance(concept_supervision, Mapping):
            raise ValueError("supervision config concept_supervision must be a mapping")

        fraction = float(concept_supervision.get("fraction", 0.0))
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("concept supervision fraction must be between 0.0 and 1.0")

        return cls(
            name=str(payload.get("name", "")),
            labels=bool(signals.get("labels", True)),
            concepts=bool(signals.get("concepts", False)),
            logic_constraints=bool(signals.get("logic_constraints", False)),
            concept_supervision_fraction=fraction,
            notes=dict(payload.get("notes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signals": {
                "labels": self.labels,
                "concepts": self.concepts,
                "logic_constraints": self.logic_constraints,
            },
            "concept_supervision": {
                "fraction": self.concept_supervision_fraction,
            },
            "notes": dict(self.notes),
        }


@dataclass(frozen=True)
class AppliedSupervision:
    """Adjusted batches and effective training settings after applying a supervision policy."""

    train_batches: list[dict[str, Any]]
    train_kwargs: dict[str, Any]
    summary: dict[str, Any]


def load_supervision_config(supervision_name: str) -> SupervisionConfig:
    """Load one supervision config from the repo config directory."""

    config_path = SUPERVISION_CONFIG_DIR / f"{supervision_name.strip().lower()}.yaml"
    if not config_path.exists():
        raise ValueError(f"Unknown supervision setting: {supervision_name}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid supervision config file: {config_path}")
    return SupervisionConfig.from_dict(payload)


def apply_supervision(
    *,
    model_family: str,
    supervision_name: str,
    seed: int,
    train_batches: Iterable[dict[str, Any]],
    train_kwargs: Mapping[str, Any] | None = None,
) -> AppliedSupervision:
    """Apply a config-driven supervision policy to the training batches and kwargs."""

    supervision = load_supervision_config(supervision_name)
    materialized_batches = _materialize_batches(train_batches)
    if not supervision.labels:
        raise ValueError(
            f"Supervision setting '{supervision_name}' disables task labels, "
            "but the current model families require label supervision."
        )

    adjusted_batches, batch_summary = _apply_concept_supervision_to_batches(
        materialized_batches,
        supervision=supervision,
        seed=seed,
    )
    adjusted_train_kwargs, kwargs_summary = _apply_logic_supervision_to_kwargs(
        dict(train_kwargs or {}),
        supervision=supervision,
        model_family=model_family,
    )
    if batch_summary.get("effective_concept_supervision_fraction", 0.0) <= 0.0:
        adjusted_train_kwargs["concept_loss_weight"] = 0.0
    kwargs_summary["effective_concept_loss_weight"] = float(
        adjusted_train_kwargs.get("concept_loss_weight", 0.0)
    )

    summary = {
        "supervision_name": supervision.name,
        "labels_enabled": supervision.labels,
        "concepts_enabled": supervision.concepts,
        "logic_constraints_enabled": supervision.logic_constraints,
        "requested_concept_supervision_fraction": supervision.concept_supervision_fraction,
        **batch_summary,
        **kwargs_summary,
    }

    return AppliedSupervision(
        train_batches=adjusted_batches,
        train_kwargs=adjusted_train_kwargs,
        summary=summary,
    )


def _materialize_batches(
    train_batches: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(train_batches, list):
        return [dict(batch) for batch in train_batches]
    return [dict(batch) for batch in train_batches]


def _apply_concept_supervision_to_batches(
    train_batches: list[dict[str, Any]],
    *,
    supervision: SupervisionConfig,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total_examples = 0
    total_concept_examples = 0
    for batch in train_batches:
        label_ids = batch.get("label_ids")
        if isinstance(label_ids, torch.Tensor):
            total_examples += int(label_ids.shape[0])
        concept_targets = batch.get("concept_targets")
        if isinstance(concept_targets, torch.Tensor):
            total_concept_examples += int(concept_targets.shape[0])

    if total_examples == 0:
        return train_batches, {
            "concept_supervised_examples": 0,
            "concept_unsupervised_examples": 0,
            "effective_concept_supervision_fraction": 0.0,
        }

    if not supervision.concepts or supervision.concept_supervision_fraction <= 0.0:
        for batch in train_batches:
            batch.pop("concept_targets", None)
            batch.pop("concept_supervision_mask", None)
        return train_batches, {
            "concept_supervised_examples": 0,
            "concept_unsupervised_examples": total_examples,
            "effective_concept_supervision_fraction": 0.0,
        }

    for batch in train_batches:
        if "concept_targets" not in batch:
            raise ValueError(
                f"Supervision setting '{supervision.name}' requires concept targets, "
                "but the current train batches do not provide them."
            )

    requested_fraction = supervision.concept_supervision_fraction
    num_supervised_examples = int(round(total_examples * requested_fraction))
    num_supervised_examples = max(0, min(total_examples, num_supervised_examples))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 5000)
    shuffled_indices = torch.randperm(total_examples, generator=generator).tolist()
    supervised_indices = set(shuffled_indices[:num_supervised_examples])

    running_offset = 0
    for batch in train_batches:
        label_ids = batch["label_ids"]
        if not isinstance(label_ids, torch.Tensor):
            raise ValueError("Each training batch must contain tensor label_ids")
        batch_size = int(label_ids.shape[0])
        local_mask = torch.tensor(
            [
                (running_offset + local_index) in supervised_indices
                for local_index in range(batch_size)
            ],
            dtype=torch.bool,
        )
        running_offset += batch_size
        batch["concept_supervision_mask"] = local_mask

    effective_fraction = num_supervised_examples / total_examples
    return train_batches, {
        "concept_supervised_examples": num_supervised_examples,
        "concept_unsupervised_examples": total_examples - num_supervised_examples,
        "effective_concept_supervision_fraction": effective_fraction,
        "dataset_concept_examples_available": total_concept_examples,
    }


def _apply_logic_supervision_to_kwargs(
    train_kwargs: dict[str, Any],
    *,
    supervision: SupervisionConfig,
    model_family: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = {
        "logic_constraint_loss_supported": False,
        "effective_satisfaction_weight": None,
    }

    if model_family == "ltn":
        summary["logic_constraint_loss_supported"] = True
        if supervision.logic_constraints:
            effective_weight = float(train_kwargs.get("satisfaction_weight", 1.0))
        else:
            effective_weight = 0.0
            train_kwargs["satisfaction_weight"] = 0.0
        summary["effective_satisfaction_weight"] = effective_weight

    return train_kwargs, summary
