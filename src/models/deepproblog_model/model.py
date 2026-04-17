"""DeepProbLog model-family implementation."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

warnings.filterwarnings(
    "ignore",
    message="ApproximateEngine is not available as PySwip could not be found",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.*",
    category=UserWarning,
)

import torch
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.query import Query
from problog.logic import Constant, Term, Var
from torch import nn

from ...eval import evaluate_model
from ..base import ModelAdapter, ModelOutputs
from ..checkpoints import load_module_bundle, save_module_bundle
from ..heads import build_prediction_head
from ..shared_encoder import SharedEncoderConfig, build_shared_encoder
from .config import DeepProbLogConfig


class _SharedConceptBackbone(nn.Module):
    """Shared encoder plus one binary head per concept predicate."""

    def __init__(self, config: DeepProbLogConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = build_shared_encoder(config.shared_encoder)
        self.concept_heads = nn.ModuleDict(
            {
                concept_name: build_prediction_head(config.concept_head)
                for concept_name in config.concept_names
            }
        )

    def forward_all(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        encoder_features = self.encoder(images)
        raw_concept_logits = {
            concept_name: head(encoder_features)
            for concept_name, head in self.concept_heads.items()
        }
        concept_logits = torch.stack(
            [
                raw_concept_logits[concept_name][:, 1]
                - raw_concept_logits[concept_name][:, 0]
                for concept_name in self.config.concept_names
            ],
            dim=1,
        )
        concept_probs = torch.stack(
            [
                torch.softmax(raw_concept_logits[concept_name], dim=-1)[:, 1]
                for concept_name in self.config.concept_names
            ],
            dim=1,
        )
        return encoder_features, raw_concept_logits, concept_logits, concept_probs

    def forward_concept_distribution(
        self,
        concept_name: str,
        images: torch.Tensor,
    ) -> torch.Tensor:
        encoder_features = self.encoder(images)
        raw_logits = self.concept_heads[concept_name](encoder_features)
        return torch.softmax(raw_logits, dim=-1)


class _ConceptPredicateView(nn.Module):
    """Per-predicate view that DeepProbLog calls as a standalone network."""

    def __init__(self, backbone: _SharedConceptBackbone, concept_name: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.concept_name = concept_name

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_concept_distribution(self.concept_name, images)


class DeepProbLogModelAdapter(ModelAdapter):
    """DeepProbLog family using neural predicates and exact probabilistic logic."""

    family_name = "deepproblog"

    def __init__(
        self,
        config: DeepProbLogConfig | None = None,
        *,
        shared_encoder_config: SharedEncoderConfig | None = None,
    ) -> None:
        self.config = config or DeepProbLogConfig.default(
            shared_encoder=shared_encoder_config
        )
        self._epsilon = 1e-8
        self._label_index = {
            label_name: index for index, label_name in enumerate(self.config.label_names)
        }
        self._network_names = {
            concept_name: f"concept_pred__{concept_name}"
            for concept_name in self.config.concept_names
        }

        self.backbone = _SharedConceptBackbone(self.config)
        self._predicate_modules = nn.ModuleDict(
            {
                concept_name: _ConceptPredicateView(self.backbone, concept_name)
                for concept_name in self.config.concept_names
            }
        )
        self.program_text = self._build_program_string()
        self.logic_model = self._build_logic_model()

    @classmethod
    def from_config_dict(cls, payload: dict[str, Any]) -> "DeepProbLogModelAdapter":
        """Construct the DeepProbLog adapter from a parsed config dictionary."""

        return cls(config=DeepProbLogConfig.from_dict(payload))

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def to(self, device: str | torch.device) -> "DeepProbLogModelAdapter":
        """Move the shared neural modules to a device."""

        self.backbone.to(torch.device(device))
        return self

    def forward(self, images: torch.Tensor) -> ModelOutputs:
        """Run the shared encoder and exact DeepProbLog label inference."""

        images = images.to(self.device).float()
        encoder_features, _, concept_logits, concept_probs = self.backbone.forward_all(images)
        label_probs = self._solve_label_distribution(images)
        label_logits = label_probs.clamp_min(self._epsilon).log()
        hard_concepts = (concept_probs >= 0.5).float()
        hard_label_predictions = label_probs.argmax(dim=-1)
        hard_rule_scores = torch.zeros_like(label_probs)
        hard_rule_scores.scatter_(1, hard_label_predictions.unsqueeze(1), 1.0)

        return ModelOutputs(
            encoder_features=encoder_features,
            concept_logits=concept_logits,
            label_logits=label_logits,
            extras={
                "concept_probs": concept_probs,
                "hard_concepts": hard_concepts,
                "label_probs": label_probs,
                "soft_rule_scores": label_probs,
                "hard_rule_scores": hard_rule_scores,
                "combined_label_scores": label_probs,
                "hard_label_predictions": hard_label_predictions,
                "symbolic_label_predictions": hard_label_predictions,
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
        """Train the DeepProbLog model on batches of tensors."""

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
        """Predict task labels using DeepProbLog label probabilities."""

        self._set_training_mode(False)
        with torch.no_grad():
            outputs = self.forward(images)
        return outputs.extras["hard_label_predictions"]

    def predict_concepts(self, images: torch.Tensor) -> torch.Tensor:
        """Predict concept probabilities from images."""

        self._set_training_mode(False)
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
        """Evaluate the DeepProbLog model with the common evaluation engine."""

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
        """Save the shared neural backbone and the parsed config."""

        save_module_bundle(
            checkpoint_path,
            modules=self._modules(),
            config=self.config.to_dict(),
            metadata={
                "family_name": self.family_name,
                "program_text": self.program_text,
            },
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> "DeepProbLogModelAdapter":
        """Load a DeepProbLog checkpoint and rebuild the logic program from config."""

        path = Path(checkpoint_path).expanduser().resolve()
        payload = torch.load(path, map_location=kwargs.get("map_location", "cpu"))
        config_payload = payload.get("config")
        if not isinstance(config_payload, dict):
            raise ValueError(f"DeepProbLog checkpoint is missing config data: {path}")

        model = cls.from_config_dict(config_payload)
        load_module_bundle(path, modules=model._modules(), map_location="cpu")
        return model

    def parameters(self) -> Sequence[nn.Parameter]:
        """Expose trainable parameters for optimizers and utilities."""

        return list(self.backbone.parameters())

    def _modules(self) -> dict[str, nn.Module]:
        return {"backbone": self.backbone}

    def _set_training_mode(self, is_training: bool) -> None:
        self.backbone.train(mode=is_training)

    def _build_logic_model(self) -> Model:
        networks = [
            Network(
                self._predicate_modules[concept_name],
                self._network_names[concept_name],
                batching=True,
            )
            for concept_name in self.config.concept_names
        ]
        model = Model(self.program_text, networks, load=False)
        model.set_engine(ExactEngine(model), cache=False)
        return model

    def _build_program_string(self) -> str:
        lines = []
        for concept_name in self.config.concept_names:
            lines.append(
                "nn({network},[X],Y,[false,true]) :: {predicate}(X,Y).".format(
                    network=self._atom(self._network_names[concept_name]),
                    predicate=self._atom(concept_name),
                )
            )

        positive_rule_name = "positive_label_logic"
        positive_rule_body = self._compile_rule_body(
            self.config.logic_program.positive_rule,
            variable_name="X",
        )
        lines.append(f"{positive_rule_name}(X) :- {positive_rule_body}.")
        lines.append(
            "label(X,{positive}) :- {helper}(X).".format(
                positive=self._atom(self.config.positive_label_name),
                helper=positive_rule_name,
            )
        )
        lines.append(
            "label(X,{negative}) :- \\+ {helper}(X).".format(
                negative=self._atom(self.config.negative_label_name),
                helper=positive_rule_name,
            )
        )
        return "\n".join(lines)

    def _compile_rule_body(self, expression: Mapping[str, Any], *, variable_name: str) -> str:
        if "concept" in expression:
            concept_name = str(expression["concept"])
            if concept_name not in self.config.concept_names:
                raise ValueError(f"Unknown concept in DeepProbLog rule: {concept_name}")
            return f"{self._atom(concept_name)}({variable_name},true)"

        operator = str(expression.get("op", "")).lower()
        args = expression.get("args", [])
        if operator not in {"and", "or", "not"}:
            raise ValueError(
                "DeepProbLog positive_rule only supports 'and', 'or', and 'not'"
            )
        if not isinstance(args, list) or not args:
            raise ValueError(
                f"DeepProbLog rule operator '{operator}' must define a non-empty args list"
            )

        if operator == "not":
            return f"\\+ ({self._compile_rule_body(args[0], variable_name=variable_name)})"

        rendered_args = [
            self._compile_rule_body(arg, variable_name=variable_name) for arg in args
        ]
        joiner = ", " if operator == "and" else " ; "
        if len(rendered_args) == 1:
            return rendered_args[0]
        return "(" + joiner.join(rendered_args) + ")"

    def _solve_label_distribution(self, images: torch.Tensor) -> torch.Tensor:
        tensor_mapping, queries = self._build_runtime_batch(images)
        self.logic_model.tensor_sources[self.config.logic_program.tensor_source_name] = (
            tensor_mapping
        )
        results = self.logic_model.solve(queries)

        probabilities: list[torch.Tensor] = []
        for result in results:
            label_row = torch.zeros(
                self.config.num_labels,
                device=images.device,
                dtype=torch.float32,
            )
            for term, value in result.result.items():
                label_term = term.args[1]
                label_name = getattr(label_term, "functor", str(label_term))
                label_index = self._label_index[label_name]
                if isinstance(value, torch.Tensor):
                    label_row[label_index] = value.to(images.device).float()
                else:
                    label_row[label_index] = float(value)

            row_sum = label_row.sum().clamp_min(self._epsilon)
            probabilities.append(label_row / row_sum)

        return torch.stack(probabilities, dim=0)

    def _build_runtime_batch(
        self,
        images: torch.Tensor,
    ) -> tuple[dict[tuple[Constant], torch.Tensor], list[Query]]:
        tensor_source_name = self.config.logic_program.tensor_source_name
        tensor_mapping: dict[tuple[Constant], torch.Tensor] = {}
        queries: list[Query] = []

        for sample_index in range(images.shape[0]):
            sample_constant = Constant(int(sample_index))
            tensor_mapping[(sample_constant,)] = images[sample_index]
            image_term = Term("tensor", Term(tensor_source_name, sample_constant))
            queries.append(
                Query(
                    Term("label", image_term, Var("Y")),
                    output_ind=(1,),
                )
            )

        return tensor_mapping, queries

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
            label_probs = outputs.extras["label_probs"].clamp_min(self._epsilon)
            selected_probs = label_probs.gather(1, label_ids.unsqueeze(1)).squeeze(1)
            label_loss = -torch.log(selected_probs).mean()
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

    @staticmethod
    def _atom(name: str) -> str:
        if re.fullmatch(r"[a-z][a-zA-Z0-9_]*", name):
            return name
        escaped = name.replace("'", "\\'")
        return f"'{escaped}'"
