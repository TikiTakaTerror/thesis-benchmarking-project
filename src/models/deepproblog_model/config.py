"""Configuration objects for the DeepProbLog model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..heads import PredictionHeadConfig
from ..shared_encoder import SharedEncoderConfig


@dataclass(frozen=True)
class DeepProbLogLabelConfig:
    """One label definition used by the DeepProbLog family."""

    id: int
    name: str
    description: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeepProbLogLabelConfig":
        return cls(
            id=int(payload["id"]),
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
        }


@dataclass(frozen=True)
class DeepProbLogTrainingConfig:
    """Training defaults for the DeepProbLog adapter."""

    batch_size: int = 16
    learning_rate: float = 1e-3
    max_epochs: int = 12
    label_loss_weight: float = 1.0
    concept_loss_weight: float = 1.0

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "DeepProbLogTrainingConfig":
        payload = dict(payload or {})
        return cls(
            batch_size=int(payload.get("batch_size", 16)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            max_epochs=int(payload.get("max_epochs", 12)),
            label_loss_weight=float(payload.get("label_loss_weight", 1.0)),
            concept_loss_weight=float(payload.get("concept_loss_weight", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "label_loss_weight": self.label_loss_weight,
            "concept_loss_weight": self.concept_loss_weight,
        }


@dataclass(frozen=True)
class DeepProbLogProgramConfig:
    """ProbLog program settings used by the DeepProbLog adapter."""

    tensor_source_name: str
    positive_label: str
    positive_rule: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "DeepProbLogProgramConfig":
        payload = dict(payload or {})
        positive_rule = payload.get("positive_rule")
        if not isinstance(positive_rule, Mapping) or not positive_rule:
            raise ValueError(
                "deepproblog logic_program.positive_rule must be a non-empty mapping"
            )

        return cls(
            tensor_source_name=str(payload.get("tensor_source_name", "runtime")),
            positive_label=str(payload.get("positive_label", "positive")),
            positive_rule=dict(positive_rule),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_source_name": self.tensor_source_name,
            "positive_label": self.positive_label,
            "positive_rule": self.positive_rule,
        }


@dataclass(frozen=True)
class DeepProbLogConfig:
    """Typed config for the DeepProbLog model family."""

    family: str
    shared_encoder: SharedEncoderConfig
    concept_head: PredictionHeadConfig
    concepts: tuple[str, ...]
    labels: tuple[DeepProbLogLabelConfig, ...]
    logic_program: DeepProbLogProgramConfig
    training_defaults: DeepProbLogTrainingConfig

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeepProbLogConfig":
        concepts = payload.get("concepts", [])
        labels = payload.get("labels", [])
        if not isinstance(concepts, list) or not concepts:
            raise ValueError("deepproblog config must define a non-empty 'concepts' list")
        if not isinstance(labels, list) or not labels:
            raise ValueError("deepproblog config must define a non-empty 'labels' list")

        shared_encoder = SharedEncoderConfig.from_dict(payload.get("shared_encoder"))
        concept_head = PredictionHeadConfig.from_dict(
            payload.get("heads", {}).get("concept_head"),
            input_dim=shared_encoder.feature_dim,
            output_dim=2,
        )
        label_configs = tuple(
            sorted(
                (DeepProbLogLabelConfig.from_dict(item) for item in labels),
                key=lambda item: item.id,
            )
        )
        if len(label_configs) != 2:
            raise ValueError(
                "deepproblog currently supports exactly two labels in Phase 7"
            )

        logic_program = DeepProbLogProgramConfig.from_dict(payload.get("logic_program"))
        label_names = {label.name for label in label_configs}
        if logic_program.positive_label not in label_names:
            raise ValueError(
                "deepproblog logic_program.positive_label must match one label name"
            )

        return cls(
            family=str(payload.get("family", "deepproblog")),
            shared_encoder=shared_encoder,
            concept_head=concept_head,
            concepts=tuple(str(name) for name in concepts),
            labels=label_configs,
            logic_program=logic_program,
            training_defaults=DeepProbLogTrainingConfig.from_dict(
                payload.get("training_defaults")
            ),
        )

    @classmethod
    def default(
        cls,
        shared_encoder: SharedEncoderConfig | None = None,
    ) -> "DeepProbLogConfig":
        """Fallback config used when only shared encoder settings are available."""

        shared_encoder = shared_encoder or SharedEncoderConfig()
        payload = {
            "family": "deepproblog",
            "shared_encoder": shared_encoder.to_dict(),
            "heads": {"concept_head": {"hidden_dim": 64, "dropout": 0.1}},
            "concepts": ["has_triangle", "is_red", "count_is_two"],
            "labels": [
                {"id": 0, "name": "negative"},
                {"id": 1, "name": "positive"},
            ],
            "logic_program": {
                "tensor_source_name": "runtime",
                "positive_label": "positive",
                "positive_rule": {
                    "op": "and",
                    "args": [
                        {"concept": "has_triangle"},
                        {"concept": "is_red"},
                        {"concept": "count_is_two"},
                    ],
                },
            },
            "training_defaults": {
                "batch_size": 16,
                "learning_rate": 1e-2,
                "max_epochs": 8,
                "label_loss_weight": 1.0,
                "concept_loss_weight": 4.0,
            },
        }
        return cls.from_dict(payload)

    @property
    def concept_names(self) -> tuple[str, ...]:
        return self.concepts

    @property
    def label_names(self) -> tuple[str, ...]:
        return tuple(label.name for label in self.labels)

    @property
    def positive_label_name(self) -> str:
        return self.logic_program.positive_label

    @property
    def negative_label_name(self) -> str:
        for label in self.labels:
            if label.name != self.logic_program.positive_label:
                return label.name
        raise ValueError("deepproblog config must define one negative label")

    @property
    def num_concepts(self) -> int:
        return len(self.concepts)

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "shared_encoder": self.shared_encoder.to_dict(),
            "heads": {
                "concept_head": {
                    "hidden_dim": self.concept_head.hidden_dim,
                    "dropout": self.concept_head.dropout,
                }
            },
            "concepts": list(self.concepts),
            "labels": [label.to_dict() for label in self.labels],
            "logic_program": self.logic_program.to_dict(),
            "training_defaults": self.training_defaults.to_dict(),
        }
