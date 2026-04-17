"""Typed containers used by the evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EvaluationTensors:
    """Aggregated tensors collected across evaluation batches."""

    label_targets: torch.Tensor | None = None
    label_predictions: torch.Tensor | None = None
    label_logits: torch.Tensor | None = None
    concept_targets: torch.Tensor | None = None
    concept_predictions: torch.Tensor | None = None
    concept_probabilities: torch.Tensor | None = None
    concept_logits: torch.Tensor | None = None
    symbolic_label_predictions: torch.Tensor | None = None
    hard_rule_scores: torch.Tensor | None = None
    soft_rule_scores: torch.Tensor | None = None

