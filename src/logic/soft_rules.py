"""Differentiable soft-logic rule execution for concept-first models."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


class SoftLogicRuleExecutor:
    """Evaluate symbolic label rules from concept probabilities or hard concepts."""

    def __init__(
        self,
        *,
        concept_names: Sequence[str],
        label_names: Sequence[str],
        rules: Mapping[str, Mapping[str, Any]],
        threshold: float = 0.5,
        epsilon: float = 1e-6,
    ) -> None:
        self.concept_names = tuple(concept_names)
        self.label_names = tuple(label_names)
        self.rules = {str(name): dict(rule) for name, rule in rules.items()}
        self.threshold = float(threshold)
        self.epsilon = float(epsilon)

        missing_rules = [
            label_name for label_name in self.label_names if label_name not in self.rules
        ]
        if missing_rules:
            raise ValueError(f"Missing rules for labels: {missing_rules}")

        self._validate_rule_references()

    def evaluate_soft(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """Evaluate soft rule scores from concept probabilities in [0, 1]."""

        concept_map = self._concept_map(concept_probs)
        scores = [
            self._evaluate_expression(self.rules[label_name], concept_map)
            for label_name in self.label_names
        ]
        return torch.stack(scores, dim=-1)

    def binarize_concepts(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """Convert concept probabilities to hard binary predictions."""

        return (concept_probs >= self.threshold).float()

    def evaluate_hard(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """Evaluate hard rule scores after thresholding concepts."""

        hard_concepts = self.binarize_concepts(concept_probs)
        return self.evaluate_soft(hard_concepts)

    def rule_scores_to_logits(self, rule_scores: torch.Tensor) -> torch.Tensor:
        """Convert rule scores in [0, 1] to logits for optimization."""

        clipped_scores = rule_scores.clamp(min=self.epsilon, max=1.0 - self.epsilon)
        return torch.logit(clipped_scores)

    def predict_label_ids(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """Predict label ids using hard thresholded concepts and symbolic rules."""

        hard_rule_scores = self.evaluate_hard(concept_probs)
        return hard_rule_scores.argmax(dim=-1)

    def _concept_map(self, concept_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if concept_values.ndim != 2:
            raise ValueError(
                "Concept values must be a 2D tensor with shape [batch_size, num_concepts]"
            )

        if concept_values.shape[1] != len(self.concept_names):
            raise ValueError(
                f"Expected {len(self.concept_names)} concepts, got {concept_values.shape[1]}"
            )

        return {
            concept_name: concept_values[:, index]
            for index, concept_name in enumerate(self.concept_names)
        }

    def _evaluate_expression(
        self,
        expression: Mapping[str, Any],
        concept_map: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if "concept" in expression:
            concept_name = str(expression["concept"])
            if concept_name not in concept_map:
                raise ValueError(f"Unknown concept referenced in rule: {concept_name}")
            return concept_map[concept_name]

        if "value" in expression:
            reference_tensor = next(iter(concept_map.values()))
            return torch.full_like(reference_tensor, float(expression["value"]))

        op = str(expression.get("op", "")).lower()
        args = expression.get("args", [])
        if not isinstance(args, list) or not args:
            raise ValueError(f"Rule operator '{op}' requires a non-empty args list.")

        evaluated_args = [self._evaluate_expression(arg, concept_map) for arg in args]
        if op == "not":
            if len(evaluated_args) != 1:
                raise ValueError("The 'not' operator requires exactly one argument.")
            return 1.0 - evaluated_args[0]

        if op == "and":
            result = torch.ones_like(evaluated_args[0])
            for value in evaluated_args:
                result = result * value
            return result

        if op == "or":
            inverse_result = torch.ones_like(evaluated_args[0])
            for value in evaluated_args:
                inverse_result = inverse_result * (1.0 - value)
            return 1.0 - inverse_result

        raise ValueError(f"Unsupported symbolic operator: {op}")

    def _validate_rule_references(self) -> None:
        for label_name in self.label_names:
            self._validate_expression(self.rules[label_name])

    def _validate_expression(self, expression: Mapping[str, Any]) -> None:
        if "concept" in expression:
            concept_name = str(expression["concept"])
            if concept_name not in self.concept_names:
                raise ValueError(f"Unknown concept in symbolic rule: {concept_name}")
            return

        if "value" in expression:
            return

        op = str(expression.get("op", "")).lower()
        args = expression.get("args", [])
        if op not in {"and", "or", "not"}:
            raise ValueError(f"Unsupported symbolic operator: {op}")
        if not isinstance(args, list) or not args:
            raise ValueError(f"Operator '{op}' requires a non-empty args list.")
        if op == "not" and len(args) != 1:
            raise ValueError("Operator 'not' requires exactly one argument.")

        for argument in args:
            if not isinstance(argument, Mapping):
                raise ValueError(f"Invalid rule expression node: {argument}")
            self._validate_expression(argument)

