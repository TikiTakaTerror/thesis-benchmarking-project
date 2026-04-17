"""Configuration objects for the LTNtorch model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..heads import PredictionHeadConfig
from ..shared_encoder import SharedEncoderConfig


@dataclass(frozen=True)
class LTNLabelConfig:
    """One label definition used by the LTN model family."""

    id: int
    name: str
    description: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LTNLabelConfig":
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
class LTNTrainingConfig:
    """Training defaults for the LTN model adapter."""

    batch_size: int = 16
    learning_rate: float = 1e-3
    max_epochs: int = 12
    label_loss_weight: float = 1.0
    concept_loss_weight: float = 1.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "LTNTrainingConfig":
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
class LTNLogicConfig:
    """Logical constraints and inference settings used by the LTN model."""

    satisfaction_weight: float
    final_prediction_logic_blend: float
    aggregator_p: int
    label_logic_rules: dict[str, dict[str, Any]]
    formulas: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "LTNLogicConfig":
        payload = dict(payload or {})
        label_logic_rules = payload.get("label_logic_rules", {})
        formulas = payload.get("formulas", [])
        if not isinstance(label_logic_rules, Mapping) or not label_logic_rules:
            raise ValueError("ltn logic_constraints.label_logic_rules must be a non-empty mapping")
        if not isinstance(formulas, list) or not formulas:
            raise ValueError("ltn logic_constraints.formulas must be a non-empty list")

        return cls(
            satisfaction_weight=float(payload.get("satisfaction_weight", 1.0)),
            final_prediction_logic_blend=float(
                payload.get("final_prediction_logic_blend", 0.5)
            ),
            aggregator_p=int(payload.get("aggregator_p", 2)),
            label_logic_rules={
                str(name): dict(rule) for name, rule in label_logic_rules.items()
            },
            formulas=tuple(dict(item) for item in formulas),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "satisfaction_weight": self.satisfaction_weight,
            "final_prediction_logic_blend": self.final_prediction_logic_blend,
            "aggregator_p": self.aggregator_p,
            "label_logic_rules": self.label_logic_rules,
            "formulas": list(self.formulas),
        }


@dataclass(frozen=True)
class LTNConfig:
    """Typed config for the LTNtorch model family."""

    family: str
    shared_encoder: SharedEncoderConfig
    concept_head: PredictionHeadConfig
    label_head: PredictionHeadConfig
    concepts: tuple[str, ...]
    labels: tuple[LTNLabelConfig, ...]
    logic_constraints: LTNLogicConfig
    training_defaults: LTNTrainingConfig

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LTNConfig":
        concepts = payload.get("concepts", [])
        labels = payload.get("labels", [])
        if not isinstance(concepts, list) or not concepts:
            raise ValueError("ltn config must define a non-empty 'concepts' list")
        if not isinstance(labels, list) or not labels:
            raise ValueError("ltn config must define a non-empty 'labels' list")

        shared_encoder = SharedEncoderConfig.from_dict(payload.get("shared_encoder"))
        head_payload = payload.get("heads", {})
        concept_head = PredictionHeadConfig.from_dict(
            head_payload.get("concept_head"),
            input_dim=shared_encoder.feature_dim,
            output_dim=len(concepts),
        )
        label_configs = tuple(
            sorted(
                (LTNLabelConfig.from_dict(item) for item in labels),
                key=lambda item: item.id,
            )
        )
        label_head = PredictionHeadConfig.from_dict(
            head_payload.get("label_head"),
            input_dim=shared_encoder.feature_dim,
            output_dim=len(label_configs),
        )

        return cls(
            family=str(payload.get("family", "ltn")),
            shared_encoder=shared_encoder,
            concept_head=concept_head,
            label_head=label_head,
            concepts=tuple(str(name) for name in concepts),
            labels=label_configs,
            logic_constraints=LTNLogicConfig.from_dict(payload.get("logic_constraints")),
            training_defaults=LTNTrainingConfig.from_dict(payload.get("training_defaults")),
        )

    @classmethod
    def default(
        cls,
        shared_encoder: SharedEncoderConfig | None = None,
    ) -> "LTNConfig":
        """Fallback config used when only shared encoder settings are available."""

        shared_encoder = shared_encoder or SharedEncoderConfig()
        payload = {
            "family": "ltn",
            "shared_encoder": shared_encoder.to_dict(),
            "heads": {
                "concept_head": {"hidden_dim": 64, "dropout": 0.1},
                "label_head": {"hidden_dim": 64, "dropout": 0.1},
            },
            "concepts": ["has_triangle", "is_red", "count_is_two"],
            "labels": [
                {"id": 0, "name": "negative"},
                {"id": 1, "name": "positive"},
            ],
            "logic_constraints": {
                "satisfaction_weight": 1.0,
                "final_prediction_logic_blend": 0.5,
                "aggregator_p": 2,
                "label_logic_rules": {
                    "negative": {
                        "op": "not",
                        "args": [
                            {
                                "op": "and",
                                "args": [
                                    {"concept": "has_triangle"},
                                    {"concept": "is_red"},
                                ],
                            }
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
                "formulas": [
                    {
                        "name": "positive_matches_triangle_and_red",
                        "expression": {
                            "op": "equiv",
                            "args": [
                                {"label": "positive"},
                                {
                                    "op": "and",
                                    "args": [
                                        {"concept": "has_triangle"},
                                        {"concept": "is_red"},
                                    ],
                                },
                            ],
                        },
                    },
                    {
                        "name": "negative_matches_not_positive",
                        "expression": {
                            "op": "equiv",
                            "args": [
                                {"label": "negative"},
                                {"op": "not", "args": [{"label": "positive"}]},
                            ],
                        },
                    },
                    {
                        "name": "positive_implies_count_is_two",
                        "expression": {
                            "op": "implies",
                            "args": [
                                {"label": "positive"},
                                {"concept": "count_is_two"},
                            ],
                        },
                    },
                ],
            },
            "training_defaults": {
                "batch_size": 16,
                "learning_rate": 3e-3,
                "max_epochs": 24,
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
                },
                "label_head": {
                    "hidden_dim": self.label_head.hidden_dim,
                    "dropout": self.label_head.dropout,
                },
            },
            "concepts": list(self.concepts),
            "labels": [label.to_dict() for label in self.labels],
            "logic_constraints": self.logic_constraints.to_dict(),
            "training_defaults": self.training_defaults.to_dict(),
        }
