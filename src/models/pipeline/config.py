"""Configuration objects for the custom concept-first symbolic pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..heads import PredictionHeadConfig
from ..shared_encoder import SharedEncoderConfig


@dataclass(frozen=True)
class PipelineLabelConfig:
    """One label definition used by the symbolic pipeline."""

    id: int
    name: str
    description: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineLabelConfig":
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
class PipelineTrainingConfig:
    """Training defaults for the symbolic pipeline adapter."""

    batch_size: int = 16
    learning_rate: float = 1e-3
    max_epochs: int = 12
    label_loss_weight: float = 1.0
    concept_loss_weight: float = 1.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "PipelineTrainingConfig":
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
class PipelineSymbolicLayerConfig:
    """Symbolic layer settings and label rule definitions."""

    executor_type: str
    threshold: float
    epsilon: float
    rules: dict[str, dict[str, Any]]

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "PipelineSymbolicLayerConfig":
        payload = dict(payload or {})
        rules = payload.get("rules", {})
        if not isinstance(rules, Mapping) or not rules:
            raise ValueError("pipeline symbolic_layer.rules must be a non-empty mapping")

        return cls(
            executor_type=str(payload.get("executor_type", "soft_logic")),
            threshold=float(payload.get("threshold", 0.5)),
            epsilon=float(payload.get("epsilon", 1e-6)),
            rules={str(name): dict(rule) for name, rule in rules.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "executor_type": self.executor_type,
            "threshold": self.threshold,
            "epsilon": self.epsilon,
            "rules": self.rules,
        }


@dataclass(frozen=True)
class PipelineConfig:
    """Typed config for the custom concept-first symbolic pipeline."""

    family: str
    shared_encoder: SharedEncoderConfig
    concept_head: PredictionHeadConfig
    concepts: tuple[str, ...]
    labels: tuple[PipelineLabelConfig, ...]
    symbolic_layer: PipelineSymbolicLayerConfig
    training_defaults: PipelineTrainingConfig

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineConfig":
        concepts = payload.get("concepts", [])
        labels = payload.get("labels", [])
        if not isinstance(concepts, list) or not concepts:
            raise ValueError("pipeline config must define a non-empty 'concepts' list")
        if not isinstance(labels, list) or not labels:
            raise ValueError("pipeline config must define a non-empty 'labels' list")

        shared_encoder = SharedEncoderConfig.from_dict(payload.get("shared_encoder"))
        concept_head = PredictionHeadConfig.from_dict(
            payload.get("heads", {}).get("concept_head"),
            input_dim=shared_encoder.feature_dim,
            output_dim=len(concepts),
        )
        label_configs = tuple(
            sorted(
                (PipelineLabelConfig.from_dict(item) for item in labels),
                key=lambda item: item.id,
            )
        )

        return cls(
            family=str(payload.get("family", "pipeline")),
            shared_encoder=shared_encoder,
            concept_head=concept_head,
            concepts=tuple(str(name) for name in concepts),
            labels=label_configs,
            symbolic_layer=PipelineSymbolicLayerConfig.from_dict(
                payload.get("symbolic_layer")
            ),
            training_defaults=PipelineTrainingConfig.from_dict(
                payload.get("training_defaults")
            ),
        )

    @classmethod
    def default(
        cls,
        shared_encoder: SharedEncoderConfig | None = None,
    ) -> "PipelineConfig":
        """Fallback config used when only a shared encoder config is available."""

        shared_encoder = shared_encoder or SharedEncoderConfig()
        payload = {
            "family": "pipeline",
            "shared_encoder": shared_encoder.to_dict(),
            "heads": {"concept_head": {"hidden_dim": 64, "dropout": 0.1}},
            "concepts": ["has_triangle", "is_red", "count_is_two"],
            "labels": [
                {"id": 0, "name": "negative"},
                {"id": 1, "name": "positive"},
            ],
            "symbolic_layer": {
                "executor_type": "soft_logic",
                "threshold": 0.5,
                "epsilon": 1e-6,
                "rules": {
                    "negative": {
                        "op": "or",
                        "args": [
                            {"op": "not", "args": [{"concept": "has_triangle"}]},
                            {"op": "not", "args": [{"concept": "is_red"}]},
                        ],
                    },
                    "positive": {
                        "op": "and",
                        "args": [
                            {"concept": "has_triangle"},
                            {"concept": "is_red"},
                        ],
                    },
                },
            },
            "training_defaults": {
                "batch_size": 16,
                "learning_rate": 3e-3,
                "max_epochs": 20,
                "label_loss_weight": 1.0,
                "concept_loss_weight": 2.0,
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
            "symbolic_layer": self.symbolic_layer.to_dict(),
            "training_defaults": self.training_defaults.to_dict(),
        }
