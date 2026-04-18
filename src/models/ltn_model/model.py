"""LTNtorch model-family implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import ltn
import torch
from torch import nn

from ...eval import evaluate_model
from ..base import ModelAdapter, ModelOutputs
from ..checkpoints import load_module_bundle, save_module_bundle
from ..heads import build_prediction_head
from ..shared_encoder import SharedEncoderConfig, build_shared_encoder
from .config import LTNConfig


class LTNModelAdapter(ModelAdapter):
    """LTN family using shared neural heads plus differentiable logical satisfaction."""

    family_name = "ltn"

    def __init__(
        self,
        config: LTNConfig | None = None,
        *,
        shared_encoder_config: SharedEncoderConfig | None = None,
    ) -> None:
        self.config = config or LTNConfig.default(shared_encoder=shared_encoder_config)
        self.encoder = build_shared_encoder(self.config.shared_encoder)
        self.concept_head = build_prediction_head(self.config.concept_head)
        self.label_head = build_prediction_head(self.config.label_head)

        self._concept_index = {
            concept_name: index for index, concept_name in enumerate(self.config.concept_names)
        }
        self._label_index = {
            label_name: index for index, label_name in enumerate(self.config.label_names)
        }
        self._sample_size = self.config.num_concepts + self.config.num_labels

        self._not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self._and = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self._or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        self._implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self._equiv = ltn.Connective(
            ltn.fuzzy_ops.Equiv(
                and_op=ltn.fuzzy_ops.AndProd(),
                implies_op=ltn.fuzzy_ops.ImpliesReichenbach(),
            )
        )
        self._forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(
                p=self.config.logic_constraints.aggregator_p
            ),
            quantifier="f",
        )
        self._sat_agg = ltn.fuzzy_ops.SatAgg()

        self._concept_predicates = {
            concept_name: ltn.Predicate(
                func=lambda sample_tensor, idx=index: sample_tensor[:, idx]
            )
            for concept_name, index in self._concept_index.items()
        }
        self._label_predicates = {
            label_name: ltn.Predicate(
                func=lambda sample_tensor, idx=self.config.num_concepts + index: sample_tensor[:, idx]
            )
            for label_name, index in self._label_index.items()
        }

    @classmethod
    def from_config_dict(cls, payload: dict[str, Any]) -> "LTNModelAdapter":
        """Construct the LTN adapter from a parsed config dictionary."""

        return cls(config=LTNConfig.from_dict(payload))

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def to(self, device: str | torch.device) -> "LTNModelAdapter":
        """Move the internal neural modules to a device."""

        torch_device = torch.device(device)
        for module in self._modules().values():
            module.to(torch_device)
        return self

    def forward(self, images: torch.Tensor) -> ModelOutputs:
        """Run the encoder, neural heads, and logic-guided inference."""

        encoder_features = self.encoder(images)
        concept_logits = self.concept_head(encoder_features)
        label_logits = self.label_head(encoder_features)

        concept_probs = torch.sigmoid(concept_logits)
        label_probs = torch.softmax(label_logits, dim=-1)
        soft_logic_scores = self._compute_logic_label_scores(concept_probs, label_probs)
        hard_concepts = (concept_probs >= 0.5).float()
        hard_logic_scores = self._compute_logic_label_scores(hard_concepts, label_probs)

        blend = self.config.logic_constraints.final_prediction_logic_blend
        combined_label_scores = (1.0 - blend) * label_probs + blend * soft_logic_scores
        hard_label_predictions = combined_label_scores.argmax(dim=-1)
        symbolic_label_predictions = hard_logic_scores.argmax(dim=-1)
        satisfaction = self._compute_satisfaction(concept_probs, label_probs)

        return ModelOutputs(
            encoder_features=encoder_features,
            concept_logits=concept_logits,
            label_logits=label_logits,
            extras={
                "concept_probs": concept_probs,
                "label_probs": label_probs,
                "hard_concepts": hard_concepts,
                "soft_rule_scores": soft_logic_scores,
                "hard_rule_scores": hard_logic_scores,
                "combined_label_scores": combined_label_scores,
                "hard_label_predictions": hard_label_predictions,
                "symbolic_label_predictions": symbolic_label_predictions,
                "logic_satisfaction": satisfaction,
            },
        )

    def train(
        self,
        train_batches: Iterable[dict[str, torch.Tensor]],
        *,
        val_batches: Iterable[dict[str, torch.Tensor]] | None = None,
        epochs: int | None = None,
        learning_rate: float | None = None,
        label_loss_weight: float | None = None,
        concept_loss_weight: float | None = None,
        satisfaction_weight: float | None = None,
        shuffle: bool = True,
    ) -> dict[str, float]:
        """Train the LTN model on batches of tensors."""

        materialized_train_batches = self._materialize_batches(train_batches)
        materialized_val_batches = (
            self._materialize_batches(val_batches) if val_batches is not None else None
        )
        if not materialized_train_batches:
            raise ValueError("train_batches must contain at least one batch")

        epochs = epochs or self.config.training_defaults.max_epochs
        learning_rate = learning_rate or self.config.training_defaults.learning_rate
        label_loss_weight = (
            self.config.training_defaults.label_loss_weight
            if label_loss_weight is None
            else float(label_loss_weight)
        )
        concept_loss_weight = (
            self.config.training_defaults.concept_loss_weight
            if concept_loss_weight is None
            else float(concept_loss_weight)
        )
        satisfaction_weight = (
            self.config.logic_constraints.satisfaction_weight
            if satisfaction_weight is None
            else float(satisfaction_weight)
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        final_train_metrics: dict[str, float] = {}
        for _ in range(int(epochs)):
            final_train_metrics = self._run_epoch(
                materialized_train_batches,
                optimizer=optimizer,
                label_loss_weight=label_loss_weight,
                concept_loss_weight=concept_loss_weight,
                satisfaction_weight=satisfaction_weight,
                shuffle=shuffle,
            )

        result = {
            f"train_{metric_name}": metric_value
            for metric_name, metric_value in final_train_metrics.items()
        }
        result["epochs"] = float(epochs)
        result["learning_rate"] = float(learning_rate)

        if materialized_val_batches is not None:
            val_metrics = self.evaluate(
                materialized_val_batches,
                label_loss_weight=label_loss_weight,
                concept_loss_weight=concept_loss_weight,
            )
            result.update(
                {
                    f"val_{metric_name}": metric_value
                    for metric_name, metric_value in val_metrics.items()
                }
            )

        return result

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Predict final task labels using neural logits blended with logic scores."""

        self._set_training_mode(False)
        images = images.to(self.device)
        with torch.no_grad():
            outputs = self.forward(images)
        return outputs.extras["hard_label_predictions"]

    def predict_concepts(self, images: torch.Tensor) -> torch.Tensor:
        """Predict concept probabilities from images."""

        self._set_training_mode(False)
        images = images.to(self.device)
        with torch.no_grad():
            outputs = self.forward(images)
        return outputs.extras["concept_probs"]

    def supports_symbolic_ablation(self) -> bool:
        """The LTN family exposes a direct neural label head."""

        return True

    def predict_without_symbolic_layer(
        self,
        images: torch.Tensor,
        *,
        reference_outputs: ModelOutputs | None = None,
    ) -> torch.Tensor:
        """Predict labels from the raw neural label head only."""

        self._set_training_mode(False)
        if reference_outputs is None:
            images = images.to(self.device)
            with torch.no_grad():
                reference_outputs = self.forward(images)

        if reference_outputs.label_logits is None:
            raise ValueError("LTN ablation requires label_logits to be available.")
        return reference_outputs.label_logits.argmax(dim=-1)

    def supports_concept_intervention(self) -> bool:
        """The LTN family can recompute logic-guided predictions from intervened concepts."""

        return True

    def predict_from_concepts(
        self,
        concept_values: torch.Tensor,
        *,
        reference_outputs: ModelOutputs | None = None,
    ) -> torch.Tensor:
        """Predict labels after replacing concept values and keeping the neural label path fixed."""

        concept_values = concept_values.to(self.device).float()
        if reference_outputs is not None:
            label_probs = reference_outputs.extras.get("label_probs")
        else:
            label_probs = None

        if label_probs is None:
            label_probs = torch.full(
                (concept_values.shape[0], self.config.num_labels),
                fill_value=1.0 / self.config.num_labels,
                device=self.device,
                dtype=concept_values.dtype,
            )
        else:
            label_probs = label_probs.to(self.device).float()

        with torch.no_grad():
            logic_scores = self._compute_logic_label_scores(concept_values, label_probs)
            blend = self.config.logic_constraints.final_prediction_logic_blend
            combined_scores = (1.0 - blend) * label_probs + blend * logic_scores
        return combined_scores.argmax(dim=-1)

    def evaluate(
        self,
        eval_batches: Iterable[dict[str, torch.Tensor]],
        *,
        label_loss_weight: float | None = None,
        concept_loss_weight: float | None = None,
    ) -> dict[str, float]:
        """Evaluate the LTN model with the common evaluation engine."""

        materialized_eval_batches = self._materialize_batches(eval_batches)
        if not materialized_eval_batches:
            raise ValueError("eval_batches must contain at least one batch")

        label_loss_weight = (
            self.config.training_defaults.label_loss_weight
            if label_loss_weight is None
            else float(label_loss_weight)
        )
        concept_loss_weight = (
            self.config.training_defaults.concept_loss_weight
            if concept_loss_weight is None
            else float(concept_loss_weight)
        )

        return evaluate_model(
            self,
            materialized_eval_batches,
            label_loss_weight=label_loss_weight,
            concept_loss_weight=concept_loss_weight,
        )

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Save the LTN encoder and neural heads."""

        save_module_bundle(
            checkpoint_path,
            modules=self._modules(),
            config=self.config.to_dict(),
            metadata={"family_name": self.family_name},
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> "LTNModelAdapter":
        """Load an LTN checkpoint and rebuild the logical components from config."""

        path = Path(checkpoint_path).expanduser().resolve()
        payload = torch.load(path, map_location=kwargs.get("map_location", "cpu"))
        config_payload = payload.get("config")
        if not isinstance(config_payload, dict):
            raise ValueError(f"LTN checkpoint is missing config data: {path}")

        model = cls.from_config_dict(config_payload)
        load_module_bundle(path, modules=model._modules(), map_location="cpu")
        return model

    def parameters(self) -> Sequence[nn.Parameter]:
        """Expose trainable parameters for optimizers and utilities."""

        return [
            parameter
            for module in self._modules().values()
            for parameter in module.parameters()
        ]

    def _modules(self) -> dict[str, nn.Module]:
        return {
            "encoder": self.encoder,
            "concept_head": self.concept_head,
            "label_head": self.label_head,
        }

    def _set_training_mode(self, is_training: bool) -> None:
        for module in self._modules().values():
            module.train(mode=is_training)

    def _run_epoch(
        self,
        batches: list[dict[str, torch.Tensor]],
        *,
        optimizer: torch.optim.Optimizer | None,
        label_loss_weight: float,
        concept_loss_weight: float,
        satisfaction_weight: float,
        shuffle: bool,
    ) -> dict[str, float]:
        is_training = optimizer is not None
        self._set_training_mode(is_training)

        ordered_batches = list(batches)
        if shuffle and is_training and len(ordered_batches) > 1:
            permutation = torch.randperm(len(ordered_batches)).tolist()
            ordered_batches = [ordered_batches[index] for index in permutation]

        label_loss_fn = nn.CrossEntropyLoss()
        concept_loss_fn = nn.BCEWithLogitsLoss()

        total_examples = 0
        total_loss_sum = 0.0
        label_loss_sum = 0.0
        concept_loss_sum = 0.0
        logic_loss_sum = 0.0
        satisfaction_sum = 0.0
        label_correct = 0.0
        concept_correct_sum = 0.0
        concept_batches = 0

        for batch in ordered_batches:
            prepared_batch = self._prepare_batch(batch)
            images = prepared_batch["images"]
            label_ids = prepared_batch["label_ids"]
            concept_targets = prepared_batch.get("concept_targets")
            concept_supervision_mask = prepared_batch.get("concept_supervision_mask")
            batch_size = int(images.shape[0])

            if is_training:
                optimizer.zero_grad()

            outputs = self.forward(images)
            label_loss = label_loss_fn(outputs.label_logits, label_ids)
            total_loss = label_loss_weight * label_loss

            if concept_targets is not None:
                if concept_supervision_mask is not None:
                    supervised_mask = concept_supervision_mask.bool()
                else:
                    supervised_mask = torch.ones(
                        concept_targets.shape[0],
                        dtype=torch.bool,
                        device=concept_targets.device,
                    )

                if bool(supervised_mask.any().item()):
                    supervised_count = int(supervised_mask.sum().item())
                    concept_loss = concept_loss_fn(
                        outputs.concept_logits[supervised_mask],
                        concept_targets[supervised_mask],
                    )
                    total_loss = total_loss + concept_loss_weight * concept_loss
                    concept_loss_sum += float(concept_loss.item()) * supervised_count
                    concept_accuracy = (
                        outputs.extras["hard_concepts"][supervised_mask]
                        == concept_targets[supervised_mask]
                    ).float().mean()
                    concept_correct_sum += float(concept_accuracy.item()) * supervised_count
                    concept_batches += supervised_count
                else:
                    concept_loss = None
            else:
                concept_loss = None

            logic_satisfaction = self._compute_satisfaction(
                outputs.extras["concept_probs"],
                outputs.extras["label_probs"],
            )
            logic_loss = 1.0 - logic_satisfaction
            total_loss = total_loss + satisfaction_weight * logic_loss

            if is_training:
                total_loss.backward()
                optimizer.step()

            total_examples += batch_size
            total_loss_sum += float(total_loss.item()) * batch_size
            label_loss_sum += float(label_loss.item()) * batch_size
            logic_loss_sum += float(logic_loss.item()) * batch_size
            satisfaction_sum += float(logic_satisfaction.item()) * batch_size
            label_predictions = outputs.extras["hard_label_predictions"]
            label_correct += float((label_predictions == label_ids).float().sum().item())

        metrics = {
            "loss": total_loss_sum / total_examples,
            "label_loss": label_loss_sum / total_examples,
            "label_accuracy": label_correct / total_examples,
            "logic_loss": logic_loss_sum / total_examples,
            "logic_satisfaction": satisfaction_sum / total_examples,
        }

        if concept_batches > 0:
            metrics["concept_loss"] = concept_loss_sum / concept_batches
            metrics["concept_accuracy"] = concept_correct_sum / concept_batches
        metrics["concept_supervised_examples"] = float(concept_batches)
        metrics["concept_supervision_fraction"] = concept_batches / total_examples

        return metrics

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "images" not in batch or "label_ids" not in batch:
            raise ValueError("Each batch must contain 'images' and 'label_ids'")

        images = batch["images"].to(self.device).float()
        label_ids = batch["label_ids"].to(self.device).long()
        if images.ndim != 4:
            raise ValueError(
                "Batch field 'images' must have shape [batch, channels, height, width]"
            )
        if label_ids.ndim != 1:
            raise ValueError("Batch field 'label_ids' must have shape [batch]")
        if images.shape[0] != label_ids.shape[0]:
            raise ValueError(
                "Batch fields 'images' and 'label_ids' must have matching batch sizes"
            )

        prepared_batch: dict[str, torch.Tensor] = {
            "images": images,
            "label_ids": label_ids,
        }

        concept_targets = batch.get("concept_targets")
        if concept_targets is not None:
            concept_targets = concept_targets.to(self.device).float()
            if concept_targets.ndim != 2:
                raise ValueError(
                    "Batch field 'concept_targets' must have shape [batch, num_concepts]"
                )
            if concept_targets.shape != (images.shape[0], self.config.num_concepts):
                raise ValueError(
                    f"Expected concept_targets with shape {(images.shape[0], self.config.num_concepts)}, "
                    f"got {tuple(concept_targets.shape)}"
                )
            prepared_batch["concept_targets"] = concept_targets

        concept_supervision_mask = batch.get("concept_supervision_mask")
        if concept_supervision_mask is not None:
            concept_supervision_mask = concept_supervision_mask.to(self.device).bool()
            if concept_supervision_mask.ndim != 1:
                raise ValueError(
                    "Batch field 'concept_supervision_mask' must have shape [batch]"
                )
            if concept_supervision_mask.shape[0] != images.shape[0]:
                raise ValueError(
                    f"Expected concept_supervision_mask with length {images.shape[0]}, "
                    f"got {tuple(concept_supervision_mask.shape)}"
                )
            prepared_batch["concept_supervision_mask"] = concept_supervision_mask

        return prepared_batch

    def _compute_satisfaction(
        self,
        concept_probs: torch.Tensor,
        label_probs: torch.Tensor,
    ) -> torch.Tensor:
        sample_variable = self._build_sample_variable(concept_probs, label_probs)
        formulas = []
        for formula_cfg in self.config.logic_constraints.formulas:
            expression = formula_cfg.get("expression")
            if not isinstance(expression, Mapping):
                raise ValueError("Each LTN formula must define an 'expression' mapping")
            formulas.append(
                self._forall(
                    sample_variable,
                    self._evaluate_formula_expression(expression, sample_variable),
                )
            )

        return self._sat_agg(*formulas)

    def _compute_logic_label_scores(
        self,
        concept_values: torch.Tensor,
        label_probs: torch.Tensor,
    ) -> torch.Tensor:
        sample_variable = self._build_sample_variable(concept_values, label_probs)
        scores = []
        for label_name in self.config.label_names:
            expression = self.config.logic_constraints.label_logic_rules[label_name]
            scores.append(
                self._evaluate_formula_expression(expression, sample_variable).value
            )
        return torch.stack(scores, dim=-1)

    def _build_sample_variable(
        self,
        concept_values: torch.Tensor,
        label_probs: torch.Tensor,
    ) -> ltn.Variable:
        sample_tensor = torch.cat([concept_values, label_probs], dim=-1)
        if sample_tensor.shape[1] != self._sample_size:
            raise ValueError(
                f"Expected sample tensor width {self._sample_size}, got {sample_tensor.shape[1]}"
            )
        return ltn.Variable("sample", sample_tensor)

    def _evaluate_formula_expression(
        self,
        expression: Mapping[str, Any],
        sample_variable: ltn.Variable,
    ):
        if "concept" in expression:
            concept_name = str(expression["concept"])
            if concept_name not in self._concept_predicates:
                raise ValueError(f"Unknown concept in LTN expression: {concept_name}")
            return self._concept_predicates[concept_name](sample_variable)

        if "label" in expression:
            label_name = str(expression["label"])
            if label_name not in self._label_predicates:
                raise ValueError(f"Unknown label in LTN expression: {label_name}")
            return self._label_predicates[label_name](sample_variable)

        op = str(expression.get("op", "")).lower()
        args = expression.get("args", [])
        if not isinstance(args, list) or not args:
            raise ValueError(f"LTN operator '{op}' requires a non-empty args list")

        evaluated_args = [
            self._evaluate_formula_expression(argument, sample_variable)
            for argument in args
        ]

        if op == "not":
            if len(evaluated_args) != 1:
                raise ValueError("LTN operator 'not' requires exactly one argument")
            return self._not(evaluated_args[0])
        if op == "and":
            result = evaluated_args[0]
            for value in evaluated_args[1:]:
                result = self._and(result, value)
            return result
        if op == "or":
            result = evaluated_args[0]
            for value in evaluated_args[1:]:
                result = self._or(result, value)
            return result
        if op == "implies":
            if len(evaluated_args) != 2:
                raise ValueError("LTN operator 'implies' requires exactly two arguments")
            return self._implies(evaluated_args[0], evaluated_args[1])
        if op == "equiv":
            if len(evaluated_args) != 2:
                raise ValueError("LTN operator 'equiv' requires exactly two arguments")
            return self._equiv(evaluated_args[0], evaluated_args[1])

        raise ValueError(f"Unsupported LTN operator: {op}")

    @staticmethod
    def _materialize_batches(
        batches: Iterable[dict[str, torch.Tensor]] | None,
    ) -> list[dict[str, torch.Tensor]]:
        if batches is None:
            return []
        if isinstance(batches, list):
            return batches
        return list(batches)
