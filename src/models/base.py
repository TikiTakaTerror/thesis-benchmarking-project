"""Common model adapter contracts shared by all families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class ModelOutputs:
    """Common container for neural outputs returned by a model family."""

    encoder_features: torch.Tensor
    concept_logits: torch.Tensor | None = None
    label_logits: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Experiment-facing interface implemented by every model family adapter."""

    family_name: str

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Run family-specific training and return training summary metrics."""

    @abstractmethod
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Predict task labels or label logits from an input image batch."""

    @abstractmethod
    def predict_concepts(self, images: torch.Tensor) -> torch.Tensor:
        """Predict concept logits or probabilities from an input image batch."""

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Run evaluation and return shared metric outputs."""

    @abstractmethod
    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Save a model-family checkpoint."""

    @classmethod
    @abstractmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> "ModelAdapter":
        """Load a model-family checkpoint."""


class PhaseStubModelAdapter(ModelAdapter):
    """Stub adapter used until a family reaches its implementation phase."""

    family_name = "stub"
    implementation_phase = 0

    def __init__(self, shared_encoder_config: Any | None = None) -> None:
        self.shared_encoder_config = shared_encoder_config

    def train(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        raise self._not_ready("train")

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        raise self._not_ready("predict")

    def predict_concepts(self, images: torch.Tensor) -> torch.Tensor:
        raise self._not_ready("predict_concepts")

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        raise self._not_ready("evaluate")

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        raise self._not_ready("save_checkpoint")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> "PhaseStubModelAdapter":
        raise NotImplementedError(
            f"{cls.__name__}.load_checkpoint() is planned for Phase "
            f"{cls.implementation_phase}."
        )

    @classmethod
    def status_message(cls) -> str:
        """Summarize the implementation status for this family."""

        return (
            f"{cls.__name__} is a Phase {cls.implementation_phase} stub. "
            "The shared model contract exists, but family-specific logic is not "
            "implemented yet."
        )

    def _not_ready(self, method_name: str) -> NotImplementedError:
        return NotImplementedError(
            f"{self.__class__.__name__}.{method_name}() is planned for Phase "
            f"{self.implementation_phase}."
        )

