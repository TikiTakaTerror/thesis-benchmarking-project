"""Reusable prediction heads for concept and label outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from torch import nn


@dataclass(frozen=True)
class PredictionHeadConfig:
    """Configuration for a simple MLP prediction head."""

    input_dim: int
    output_dim: int
    hidden_dim: int | None = None
    dropout: float = 0.0

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        input_dim: int,
        output_dim: int,
    ) -> "PredictionHeadConfig":
        payload = dict(payload or {})
        hidden_dim = payload.get("hidden_dim")
        if hidden_dim is not None:
            hidden_dim = int(hidden_dim)

        return cls(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            hidden_dim=hidden_dim,
            dropout=float(payload.get("dropout", 0.0)),
        )


class PredictionHead(nn.Module):
    """Small head used for concept logits or label logits."""

    def __init__(self, config: PredictionHeadConfig) -> None:
        super().__init__()
        self.config = config

        if config.hidden_dim is None:
            self.network = nn.Linear(config.input_dim, config.output_dim)
        else:
            self.network = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.output_dim),
            )

    def forward(self, features):
        return self.network(features)


def build_prediction_head(config: PredictionHeadConfig) -> PredictionHead:
    """Factory for a reusable prediction head."""

    return PredictionHead(config)

