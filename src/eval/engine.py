"""Shared evaluation engine for model families."""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from .metrics import (
    compute_classification_metrics,
    compute_concept_metrics,
    compute_control_metrics,
    compute_semantic_metrics,
)
from .types import EvaluationTensors


def evaluate_model(
    model: Any,
    eval_batches: Iterable[dict[str, torch.Tensor]],
    *,
    seed: int | None = None,
    label_loss_weight: float = 1.0,
    concept_loss_weight: float = 1.0,
) -> dict[str, float]:
    """Run the common evaluator on a sequence of tensor batches."""

    batches = list(eval_batches)
    if not batches:
        raise ValueError("eval_batches must contain at least one batch")

    start_time = time.perf_counter()
    collected = _collect_evaluation_tensors(model, batches)
    elapsed_seconds = time.perf_counter() - start_time

    metrics: dict[str, float] = {}
    metrics.update(
        compute_classification_metrics(
            collected.label_targets,
            collected.label_predictions,
        )
    )
    metrics.update(
        compute_concept_metrics(
            collected.concept_targets,
            collected.concept_predictions,
        )
    )
    metrics.update(
        compute_semantic_metrics(
            label_targets=collected.label_targets,
            label_predictions=collected.label_predictions,
            symbolic_label_predictions=collected.symbolic_label_predictions,
            hard_rule_scores=collected.hard_rule_scores,
        )
    )
    metrics.update(
        _compute_loss_metrics(
            collected,
            label_loss_weight=label_loss_weight,
            concept_loss_weight=concept_loss_weight,
        )
    )
    metrics.update(
        compute_control_metrics(
            model=model,
            num_examples=int(collected.label_predictions.shape[0]),
            num_batches=len(batches),
            evaluation_time_seconds=elapsed_seconds,
            seed=seed,
        )
    )
    return metrics


def evaluate_named_splits(
    model: Any,
    split_batches: Mapping[str, Iterable[dict[str, torch.Tensor]]],
    *,
    seed: int | None = None,
    label_loss_weight: float = 1.0,
    concept_loss_weight: float = 1.0,
) -> dict[str, float]:
    """Evaluate multiple named splits and prefix the metric keys per split."""

    metrics: dict[str, float] = {}
    for split_name, batches in split_batches.items():
        split_metrics = evaluate_model(
            model,
            batches,
            seed=seed,
            label_loss_weight=label_loss_weight,
            concept_loss_weight=concept_loss_weight,
        )
        metrics.update(
            {
                f"{split_name}_{metric_name}": metric_value
                for metric_name, metric_value in split_metrics.items()
            }
        )

    return metrics


def _collect_evaluation_tensors(
    model: Any,
    batches: list[dict[str, torch.Tensor]],
) -> EvaluationTensors:
    if hasattr(model, "_set_training_mode"):
        model._set_training_mode(False)

    label_targets: list[torch.Tensor] = []
    label_predictions: list[torch.Tensor] = []
    label_logits: list[torch.Tensor] = []
    concept_targets: list[torch.Tensor] = []
    concept_predictions: list[torch.Tensor] = []
    concept_probabilities: list[torch.Tensor] = []
    concept_logits: list[torch.Tensor] = []
    symbolic_label_predictions: list[torch.Tensor] = []
    hard_rule_scores: list[torch.Tensor] = []
    soft_rule_scores: list[torch.Tensor] = []

    device = _resolve_model_device(model)

    with torch.no_grad():
        for batch in batches:
            images = _require_batch_field(batch, "images").to(device).float()
            batch_label_targets = batch.get("label_ids")
            batch_concept_targets = batch.get("concept_targets")

            outputs = model.forward(images) if hasattr(model, "forward") else None
            if outputs is None:
                raise ValueError(
                    "The current evaluation engine expects model.forward(...) to be available."
                )

            batch_label_predictions = outputs.extras.get("hard_label_predictions")
            if batch_label_predictions is None:
                if outputs.label_logits is None:
                    raise ValueError("Model outputs do not provide label predictions.")
                batch_label_predictions = outputs.label_logits.argmax(dim=-1)

            batch_concept_probabilities = outputs.extras.get("concept_probs")
            if batch_concept_probabilities is None and outputs.concept_logits is not None:
                batch_concept_probabilities = torch.sigmoid(outputs.concept_logits)

            batch_concept_predictions = outputs.extras.get("hard_concepts")
            if batch_concept_predictions is None and batch_concept_probabilities is not None:
                batch_concept_predictions = (batch_concept_probabilities >= 0.5).float()

            batch_symbolic_label_predictions = outputs.extras.get(
                "symbolic_label_predictions",
                outputs.extras.get("hard_label_predictions"),
            )
            batch_hard_rule_scores = outputs.extras.get("hard_rule_scores")
            batch_soft_rule_scores = outputs.extras.get("soft_rule_scores")

            if (
                hasattr(model, "symbolic_executor")
                and batch_concept_probabilities is not None
                and batch_hard_rule_scores is None
            ):
                batch_hard_rule_scores = model.symbolic_executor.evaluate_hard(
                    batch_concept_probabilities
                )
                batch_soft_rule_scores = model.symbolic_executor.evaluate_soft(
                    batch_concept_probabilities
                )
                batch_symbolic_label_predictions = model.symbolic_executor.predict_label_ids(
                    batch_concept_probabilities
                )

            if batch_label_targets is not None:
                label_targets.append(batch_label_targets.detach().cpu().long())
            label_predictions.append(batch_label_predictions.detach().cpu().long())

            if outputs.label_logits is not None:
                label_logits.append(outputs.label_logits.detach().cpu().float())

            if batch_concept_targets is not None:
                concept_targets.append(batch_concept_targets.detach().cpu().float())

            if batch_concept_predictions is not None:
                concept_predictions.append(batch_concept_predictions.detach().cpu().float())

            if batch_concept_probabilities is not None:
                concept_probabilities.append(
                    batch_concept_probabilities.detach().cpu().float()
                )

            if outputs.concept_logits is not None:
                concept_logits.append(outputs.concept_logits.detach().cpu().float())

            if batch_symbolic_label_predictions is not None:
                symbolic_label_predictions.append(
                    batch_symbolic_label_predictions.detach().cpu().long()
                )

            if batch_hard_rule_scores is not None:
                hard_rule_scores.append(batch_hard_rule_scores.detach().cpu().float())

            if batch_soft_rule_scores is not None:
                soft_rule_scores.append(batch_soft_rule_scores.detach().cpu().float())

    return EvaluationTensors(
        label_targets=_concat_or_none(label_targets),
        label_predictions=_concat_or_none(label_predictions),
        label_logits=_concat_or_none(label_logits),
        concept_targets=_concat_or_none(concept_targets),
        concept_predictions=_concat_or_none(concept_predictions),
        concept_probabilities=_concat_or_none(concept_probabilities),
        concept_logits=_concat_or_none(concept_logits),
        symbolic_label_predictions=_concat_or_none(symbolic_label_predictions),
        hard_rule_scores=_concat_or_none(hard_rule_scores),
        soft_rule_scores=_concat_or_none(soft_rule_scores),
    )


def _compute_loss_metrics(
    tensors: EvaluationTensors,
    *,
    label_loss_weight: float,
    concept_loss_weight: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total_loss = 0.0

    if tensors.label_targets is not None and tensors.label_logits is not None:
        label_loss = nn.CrossEntropyLoss()(
            tensors.label_logits,
            tensors.label_targets.long(),
        )
        metrics["label_loss"] = float(label_loss.item())
        total_loss += label_loss_weight * float(label_loss.item())

    if tensors.concept_targets is not None and tensors.concept_logits is not None:
        concept_loss = nn.BCEWithLogitsLoss()(
            tensors.concept_logits,
            tensors.concept_targets.float(),
        )
        metrics["concept_loss"] = float(concept_loss.item())
        total_loss += concept_loss_weight * float(concept_loss.item())

    if metrics:
        metrics["loss"] = total_loss

    return metrics


def _concat_or_none(chunks: list[torch.Tensor]) -> torch.Tensor | None:
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def _resolve_model_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)

    if hasattr(model, "parameters"):
        try:
            return next(iter(model.parameters())).device
        except StopIteration:
            return torch.device("cpu")

    return torch.device("cpu")


def _require_batch_field(
    batch: dict[str, torch.Tensor],
    field_name: str,
) -> torch.Tensor:
    if field_name not in batch:
        raise ValueError(f"Each evaluation batch must contain '{field_name}'")
    return batch[field_name]
