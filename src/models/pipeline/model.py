"""Custom concept-first symbolic pipeline implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from ...logic import SoftLogicRuleExecutor
from ..base import ModelAdapter, ModelOutputs
from ..checkpoints import load_module_bundle, save_module_bundle
from ..heads import build_prediction_head
from ..shared_encoder import SharedEncoderConfig, build_shared_encoder
from .config import PipelineConfig


class PipelineModelAdapter(ModelAdapter):
    """Concept-first pipeline with a shared encoder, concept head, and rule executor."""

    family_name = "pipeline"

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        shared_encoder_config: SharedEncoderConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig.default(shared_encoder=shared_encoder_config)
        self.encoder = build_shared_encoder(self.config.shared_encoder)
        self.concept_head = build_prediction_head(self.config.concept_head)
        self.symbolic_executor = SoftLogicRuleExecutor(
            concept_names=self.config.concept_names,
            label_names=self.config.label_names,
            rules=self.config.symbolic_layer.rules,
            threshold=self.config.symbolic_layer.threshold,
            epsilon=self.config.symbolic_layer.epsilon,
        )

    @classmethod
    def from_config_dict(cls, payload: dict[str, Any]) -> "PipelineModelAdapter":
        """Construct the pipeline adapter from a parsed config dictionary."""

        return cls(config=PipelineConfig.from_dict(payload))

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def to(self, device: str | torch.device) -> "PipelineModelAdapter":
        """Move the internal neural modules to a device."""

        torch_device = torch.device(device)
        for module in self._modules().values():
            module.to(torch_device)
        return self

    def forward(self, images: torch.Tensor) -> ModelOutputs:
        """Run the neural encoder, concept head, and symbolic executor."""

        encoder_features = self.encoder(images)
        concept_logits = self.concept_head(encoder_features)
        concept_probs = torch.sigmoid(concept_logits)
        soft_rule_scores = self.symbolic_executor.evaluate_soft(concept_probs)
        hard_concepts = self.symbolic_executor.binarize_concepts(concept_probs)
        hard_rule_scores = self.symbolic_executor.evaluate_soft(hard_concepts)
        label_logits = self.symbolic_executor.rule_scores_to_logits(soft_rule_scores)
        hard_label_predictions = hard_rule_scores.argmax(dim=-1)

        return ModelOutputs(
            encoder_features=encoder_features,
            concept_logits=concept_logits,
            label_logits=label_logits,
            extras={
                "concept_probs": concept_probs,
                "hard_concepts": hard_concepts,
                "soft_rule_scores": soft_rule_scores,
                "hard_rule_scores": hard_rule_scores,
                "hard_label_predictions": hard_label_predictions,
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
        shuffle: bool = True,
    ) -> dict[str, float]:
        """Train the symbolic pipeline on batches of tensors."""

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

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        final_train_metrics: dict[str, float] = {}
        for _ in range(int(epochs)):
            final_train_metrics = self._run_epoch(
                materialized_train_batches,
                optimizer=optimizer,
                label_loss_weight=label_loss_weight,
                concept_loss_weight=concept_loss_weight,
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
        """Predict final task labels using the symbolic layer."""

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

    def evaluate(
        self,
        eval_batches: Iterable[dict[str, torch.Tensor]],
        *,
        label_loss_weight: float | None = None,
        concept_loss_weight: float | None = None,
    ) -> dict[str, float]:
        """Evaluate the symbolic pipeline on batches of tensors."""

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

        return self._run_epoch(
            materialized_eval_batches,
            optimizer=None,
            label_loss_weight=label_loss_weight,
            concept_loss_weight=concept_loss_weight,
            shuffle=False,
        )

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Save the pipeline encoder and concept head."""

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
    ) -> "PipelineModelAdapter":
        """Load a pipeline checkpoint and rebuild the symbolic executor from config."""

        path = Path(checkpoint_path).expanduser().resolve()
        payload = torch.load(path, map_location=kwargs.get("map_location", "cpu"))
        config_payload = payload.get("config")
        if not isinstance(config_payload, dict):
            raise ValueError(f"Pipeline checkpoint is missing config data: {path}")

        model = cls.from_config_dict(config_payload)
        load_module_bundle(path, modules=model._modules(), map_location="cpu")
        return model

    def parameters(self) -> Sequence[nn.Parameter]:
        """Expose trainable parameters for optimizers and utilities."""

        return [parameter for module in self._modules().values() for parameter in module.parameters()]

    def _modules(self) -> dict[str, nn.Module]:
        return {
            "encoder": self.encoder,
            "concept_head": self.concept_head,
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
        label_correct = 0.0
        concept_correct_sum = 0.0
        concept_batches = 0

        for batch in ordered_batches:
            prepared_batch = self._prepare_batch(batch)
            images = prepared_batch["images"]
            label_ids = prepared_batch["label_ids"]
            concept_targets = prepared_batch.get("concept_targets")
            batch_size = int(images.shape[0])

            if is_training:
                optimizer.zero_grad()

            outputs = self.forward(images)
            label_loss = label_loss_fn(outputs.label_logits, label_ids)
            total_loss = label_loss_weight * label_loss

            if concept_targets is not None:
                concept_loss = concept_loss_fn(outputs.concept_logits, concept_targets)
                total_loss = total_loss + concept_loss_weight * concept_loss
                concept_loss_sum += float(concept_loss.item()) * batch_size
                concept_accuracy = (
                    outputs.extras["hard_concepts"] == concept_targets
                ).float().mean()
                concept_correct_sum += float(concept_accuracy.item()) * batch_size
                concept_batches += batch_size
            else:
                concept_loss = None

            if is_training:
                total_loss.backward()
                optimizer.step()

            total_examples += batch_size
            total_loss_sum += float(total_loss.item()) * batch_size
            label_loss_sum += float(label_loss.item()) * batch_size
            label_predictions = outputs.extras["hard_label_predictions"]
            label_correct += float((label_predictions == label_ids).float().sum().item())

        metrics = {
            "loss": total_loss_sum / total_examples,
            "label_loss": label_loss_sum / total_examples,
            "label_accuracy": label_correct / total_examples,
        }

        if concept_batches > 0:
            metrics["concept_loss"] = concept_loss_sum / concept_batches
            metrics["concept_accuracy"] = concept_correct_sum / concept_batches

        return metrics

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "images" not in batch or "label_ids" not in batch:
            raise ValueError("Each batch must contain 'images' and 'label_ids'")

        images = batch["images"].to(self.device).float()
        label_ids = batch["label_ids"].to(self.device).long()
        if images.ndim != 4:
            raise ValueError("Batch field 'images' must have shape [batch, channels, height, width]")
        if label_ids.ndim != 1:
            raise ValueError("Batch field 'label_ids' must have shape [batch]")
        if images.shape[0] != label_ids.shape[0]:
            raise ValueError("Batch fields 'images' and 'label_ids' must have matching batch sizes")

        prepared_batch: dict[str, torch.Tensor] = {
            "images": images,
            "label_ids": label_ids,
        }

        concept_targets = batch.get("concept_targets")
        if concept_targets is not None:
            concept_targets = concept_targets.to(self.device).float()
            if concept_targets.ndim != 2:
                raise ValueError("Batch field 'concept_targets' must have shape [batch, num_concepts]")
            if concept_targets.shape != (images.shape[0], self.config.num_concepts):
                raise ValueError(
                    f"Expected concept_targets with shape {(images.shape[0], self.config.num_concepts)}, "
                    f"got {tuple(concept_targets.shape)}"
                )
            prepared_batch["concept_targets"] = concept_targets

        return prepared_batch

    @staticmethod
    def _materialize_batches(
        batches: Iterable[dict[str, torch.Tensor]] | None,
    ) -> list[dict[str, torch.Tensor]]:
        if batches is None:
            return []
        if isinstance(batches, list):
            return batches
        return list(batches)

