"""Real-data managed-run helpers for prepared datasets such as MNLogic and Kand-Logic."""

from __future__ import annotations

import ast
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch

from ..benchmarks import create_benchmark_adapter
from ..data import build_prepared_dataloaders, create_dataset_adapter
from ..models import load_model_config
from ..services import ProjectConfig, RunManager, RunSelection
from .runner import RunExecutionResult, execute_training_run
from .supervision import apply_supervision


REAL_MNLOGIC_DATASET_NAME = "mnlogic"
REAL_KAND_LOGIC_DATASET_NAME = "kand_logic"
REAL_PREPARED_DATASET_NAMES = (
    REAL_MNLOGIC_DATASET_NAME,
    REAL_KAND_LOGIC_DATASET_NAME,
)


def execute_real_prepared_managed_run(
    run_manager: RunManager,
    *,
    dataset_name: str,
    project_config: ProjectConfig,
    model_family: str,
    seed: int,
    benchmark_suite: str = "rsbench",
    supervision: str = "full",
    run_name: str | None = None,
    training_overrides: Mapping[str, Any] | None = None,
    limit_per_split: int | None = None,
    dataset_root: str | Path | None = None,
    num_workers: int = 0,
) -> RunExecutionResult:
    """Execute one managed run on a prepared real dataset."""

    canonical_dataset_name = normalize_real_dataset_name(dataset_name)
    if canonical_dataset_name not in REAL_PREPARED_DATASET_NAMES:
        raise ValueError(
            f"Unsupported real prepared dataset '{dataset_name}'. Supported values are "
            f"{list(REAL_PREPARED_DATASET_NAMES)}."
        )

    random.seed(seed)
    torch.manual_seed(seed)

    benchmark_adapter = create_benchmark_adapter(benchmark_suite)
    if canonical_dataset_name not in benchmark_adapter.list_datasets():
        raise ValueError(
            f"Benchmark suite '{benchmark_suite}' does not support dataset "
            f"'{canonical_dataset_name}'."
        )

    runtime_context = build_prepared_runtime_context(
        dataset_name=canonical_dataset_name,
        model_family=model_family,
        dataset_root=dataset_root,
    )
    runtime_model_config = runtime_context["model_config"]
    source_info = runtime_context["source_info"]
    external_environment = benchmark_adapter.build_external_environment(
        dataset_name=canonical_dataset_name,
        model_family=model_family,
    )

    selection = RunSelection(
        dataset=canonical_dataset_name,
        model_family=model_family,
        benchmark_suite=benchmark_suite,
        supervision=supervision,
        seed=seed,
    )

    training_payload = {
        **default_real_training_kwargs(runtime_model_config),
        **dict(training_overrides or {}),
    }
    batch_size = int(
        training_payload.pop(
            "batch_size",
            runtime_model_config.get("training_defaults", {}).get("batch_size", 16),
        )
    )

    dataloaders = build_prepared_dataloaders(
        dataset_name=canonical_dataset_name,
        model_family=model_family,
        model_config=runtime_model_config,
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=True,
        limit_per_split=limit_per_split,
    )
    supervision_result = apply_supervision(
        model_family=model_family,
        supervision_name=supervision,
        seed=seed,
        train_batches=list(dataloaders.train_loader),
        train_kwargs=training_payload,
    )
    train_batches = supervision_result.train_batches
    effective_training_payload = supervision_result.train_kwargs
    val_batches = list(dataloaders.val_loader)
    evaluation_splits = build_real_evaluation_splits(
        benchmark_suite=benchmark_suite,
        test_batches=list(dataloaders.test_loader),
        ood_batches=(
            list(dataloaders.ood_loader)
            if dataloaders.ood_loader is not None
            else None
        ),
    )

    resolved_run_name = run_name or f"real_{canonical_dataset_name}_{model_family}_seed_{seed}"
    dataset_snapshot = {
        "name": canonical_dataset_name,
        "root": str(runtime_context["dataset_root"]),
        "concept_names": list(runtime_context["concept_names"]),
        "label_names": list(runtime_context["label_names"]),
        "logic_expression": source_info.get("logic_expression"),
        "aggregator_logic_expression": source_info.get("aggregator_logic_expression"),
        "warnings": list(source_info.get("warnings", [])),
        "tensor_config": {
            "input_channels": dataloaders.tensor_config.input_channels,
            "input_size": list(dataloaders.tensor_config.input_size),
            "batch_size": dataloaders.tensor_config.batch_size,
            "num_workers": dataloaders.tensor_config.num_workers,
        },
        "split_sizes": {
            "train": len(dataloaders.train_loader.dataset),
            "val": len(dataloaders.val_loader.dataset),
            "test": len(dataloaders.test_loader.dataset),
            "ood": (
                len(dataloaders.ood_loader.dataset)
                if dataloaders.ood_loader is not None
                else 0
            ),
        },
    }
    for optional_key in (
        "concept_representation",
        "n_figures",
        "n_objects_per_figure",
        "colors",
        "shapes",
        "compiled_positive_rule_stats",
    ):
        if optional_key in source_info:
            dataset_snapshot[optional_key] = deepcopy(source_info[optional_key])

    config_snapshot = {
        "project": project_config.to_dict(),
        "run": {
            "name": resolved_run_name,
            "phase": 14,
            "seed": seed,
            "source": f"r9_real_{canonical_dataset_name}_managed_run",
        },
        "selection": selection.to_dict(),
        "model": runtime_model_config,
        "dataset": dataset_snapshot,
        "benchmark": {
            "suite": benchmark_suite,
            "config": benchmark_adapter.config.to_dict(),
            "external_environment": external_environment,
            "evaluation_splits": list(evaluation_splits.keys()),
        },
        "supervision_policy": supervision_result.summary,
        "training": {
            **effective_training_payload,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "limit_per_split": limit_per_split,
        },
    }

    return execute_training_run(
        run_manager,
        run_name=resolved_run_name,
        selection=selection,
        config_snapshot=config_snapshot,
        model_config=runtime_model_config,
        train_batches=train_batches,
        evaluation_splits=evaluation_splits,
        evaluation_callback=lambda model, split_batches: benchmark_adapter.run_evaluation(
            model,
            split_batches,
            seed=seed,
            label_loss_weight=float(
                effective_training_payload.get("label_loss_weight", 1.0)
            ),
            concept_loss_weight=float(
                effective_training_payload.get("concept_loss_weight", 1.0)
            ),
            external_environment=external_environment,
        ),
        train_kwargs={
            **effective_training_payload,
            "val_batches": val_batches,
        },
    )


def execute_real_mnlogic_managed_run(
    run_manager: RunManager,
    *,
    project_config: ProjectConfig,
    model_family: str,
    seed: int,
    benchmark_suite: str = "rsbench",
    supervision: str = "full",
    run_name: str | None = None,
    training_overrides: Mapping[str, Any] | None = None,
    limit_per_split: int | None = None,
    dataset_root: str | Path | None = None,
    num_workers: int = 0,
) -> RunExecutionResult:
    """Backward-compatible wrapper for real MNLogic runs."""

    return execute_real_prepared_managed_run(
        run_manager,
        dataset_name=REAL_MNLOGIC_DATASET_NAME,
        project_config=project_config,
        model_family=model_family,
        seed=seed,
        benchmark_suite=benchmark_suite,
        supervision=supervision,
        run_name=run_name,
        training_overrides=training_overrides,
        limit_per_split=limit_per_split,
        dataset_root=dataset_root,
        num_workers=num_workers,
    )


def execute_real_kand_logic_managed_run(
    run_manager: RunManager,
    *,
    project_config: ProjectConfig,
    model_family: str,
    seed: int,
    benchmark_suite: str = "rsbench",
    supervision: str = "full",
    run_name: str | None = None,
    training_overrides: Mapping[str, Any] | None = None,
    limit_per_split: int | None = None,
    dataset_root: str | Path | None = None,
    num_workers: int = 0,
) -> RunExecutionResult:
    """Managed real-data wrapper for prepared Kand-Logic runs."""

    return execute_real_prepared_managed_run(
        run_manager,
        dataset_name=REAL_KAND_LOGIC_DATASET_NAME,
        project_config=project_config,
        model_family=model_family,
        seed=seed,
        benchmark_suite=benchmark_suite,
        supervision=supervision,
        run_name=run_name,
        training_overrides=training_overrides,
        limit_per_split=limit_per_split,
        dataset_root=dataset_root,
        num_workers=num_workers,
    )


def build_prepared_runtime_context(
    *,
    dataset_name: str,
    model_family: str,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a dataset-aware runtime config for a prepared real dataset."""

    canonical_dataset_name = normalize_real_dataset_name(dataset_name)
    adapter = create_dataset_adapter(canonical_dataset_name, dataset_root=dataset_root)
    adapter.validate_layout()
    dataset_root_path = adapter.dataset_root
    concept_names = tuple(concept.name for concept in adapter.get_concept_schema())
    label_definitions = [
        {
            "id": label.id,
            "name": label.name,
            "description": label.description,
        }
        for label in adapter.get_label_schema()
    ]
    label_names = tuple(label["name"] for label in label_definitions)

    source_info_path = dataset_root_path / "metadata" / "source_info.json"
    source_info = _load_json(source_info_path)
    positive_rule = _resolve_positive_rule(
        source_info=source_info,
        concept_names=set(concept_names),
    )

    positive_label_name, negative_label_name = _resolve_binary_label_names(label_definitions)
    negative_rule = {"op": "not", "args": [positive_rule]}

    model_config = deepcopy(load_model_config(model_family))
    model_config["concepts"] = list(concept_names)
    model_config["labels"] = label_definitions

    if model_family == "pipeline":
        model_config.setdefault("symbolic_layer", {})
        model_config["symbolic_layer"]["rules"] = {
            negative_label_name: negative_rule,
            positive_label_name: positive_rule,
        }
    elif model_family == "ltn":
        model_config.setdefault("logic_constraints", {})
        model_config["logic_constraints"]["label_logic_rules"] = {
            negative_label_name: negative_rule,
            positive_label_name: positive_rule,
        }
        model_config["logic_constraints"]["formulas"] = [
            {
                "name": "positive_matches_dataset_logic",
                "expression": {
                    "op": "equiv",
                    "args": [
                        {"label": positive_label_name},
                        positive_rule,
                    ],
                },
            },
            {
                "name": "negative_matches_not_positive",
                "expression": {
                    "op": "equiv",
                    "args": [
                        {"label": negative_label_name},
                        {"op": "not", "args": [{"label": positive_label_name}]},
                    ],
                },
            },
        ]
    elif model_family == "deepproblog":
        model_config.setdefault("logic_program", {})
        model_config["logic_program"]["positive_label"] = positive_label_name
        model_config["logic_program"]["positive_rule"] = positive_rule
    else:
        raise ValueError(
            f"Unsupported model family for runtime config: {model_family}"
        )

    notes = dict(model_config.get("notes", {}))
    notes.update(
        {
            "runtime_dataset": canonical_dataset_name,
            "runtime_logic_expression": source_info.get("logic_expression", ""),
            "runtime_aggregator_logic_expression": source_info.get(
                "aggregator_logic_expression", ""
            ),
            "runtime_generated": True,
        }
    )
    model_config["notes"] = notes

    return {
        "dataset_root": dataset_root_path,
        "concept_names": concept_names,
        "label_names": label_names,
        "positive_label_name": positive_label_name,
        "negative_label_name": negative_label_name,
        "positive_rule": positive_rule,
        "source_info": source_info,
        "model_config": model_config,
    }


def build_mnlogic_runtime_context(
    *,
    model_family: str,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for the MNLogic runtime builder."""

    return build_prepared_runtime_context(
        dataset_name=REAL_MNLOGIC_DATASET_NAME,
        model_family=model_family,
        dataset_root=dataset_root,
    )


def build_kand_logic_runtime_context(
    *,
    model_family: str,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Runtime builder for prepared Kand-Logic runs."""

    return build_prepared_runtime_context(
        dataset_name=REAL_KAND_LOGIC_DATASET_NAME,
        model_family=model_family,
        dataset_root=dataset_root,
    )


def default_real_training_kwargs(model_config: Mapping[str, Any]) -> dict[str, float]:
    """Return stable training defaults for a real prepared-dataset run."""

    training_defaults = dict(model_config.get("training_defaults", {}))
    payload = {
        "epochs": float(training_defaults.get("max_epochs", 12)),
        "learning_rate": float(training_defaults.get("learning_rate", 1e-3)),
        "batch_size": float(training_defaults.get("batch_size", 16)),
    }
    if "label_loss_weight" in training_defaults:
        payload["label_loss_weight"] = float(training_defaults["label_loss_weight"])
    if "concept_loss_weight" in training_defaults:
        payload["concept_loss_weight"] = float(training_defaults["concept_loss_weight"])
    return payload


def build_real_evaluation_splits(
    *,
    benchmark_suite: str,
    test_batches: list[dict[str, Any]],
    ood_batches: list[dict[str, Any]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Build evaluation splits for a real prepared dataset run."""

    if benchmark_suite == "core_eval":
        return {"test": test_batches}
    if benchmark_suite == "rsbench":
        evaluation_splits = {"id": test_batches}
        if ood_batches:
            evaluation_splits["ood"] = ood_batches
        return evaluation_splits
    raise ValueError(f"Unsupported real-data benchmark suite: {benchmark_suite}")


def normalize_real_dataset_name(dataset_name: str) -> str:
    return str(dataset_name).strip().lower().replace("-", "_")


def _resolve_positive_rule(
    *,
    source_info: Mapping[str, Any],
    concept_names: set[str],
) -> dict[str, Any]:
    compiled_rule = source_info.get("compiled_positive_rule")
    if isinstance(compiled_rule, Mapping) and compiled_rule:
        return deepcopy(dict(compiled_rule))

    logic_expression = str(source_info.get("logic_expression", "")).strip()
    if not logic_expression:
        raise ValueError(
            "Prepared real dataset runtime config requires either "
            "'compiled_positive_rule' or 'logic_expression' in source_info.json."
        )

    return _parse_logic_expression(logic_expression, concept_names=concept_names)


def _resolve_binary_label_names(
    label_definitions: list[dict[str, Any]],
) -> tuple[str, str]:
    if len(label_definitions) != 2:
        raise ValueError("Real prepared runtime config currently expects exactly two labels.")

    by_id = {int(label["id"]): str(label["name"]) for label in label_definitions}
    positive_label_name = by_id.get(1)
    negative_label_name = by_id.get(0)
    if positive_label_name is None or negative_label_name is None:
        raise ValueError(
            "Real prepared runtime config expects label IDs 0 and 1 to be present."
        )
    return positive_label_name, negative_label_name


def _parse_logic_expression(
    expression: str,
    *,
    concept_names: set[str],
) -> dict[str, Any]:
    parsed = ast.parse(expression, mode="eval")
    return _convert_logic_node(parsed.body, concept_names=concept_names)


def _convert_logic_node(
    node: ast.AST,
    *,
    concept_names: set[str],
) -> dict[str, Any]:
    if isinstance(node, ast.Name):
        if node.id not in concept_names:
            raise ValueError(f"Unknown concept in logic expression: {node.id}")
        return {"concept": node.id}

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.BitAnd):
            return _compose_logic_expression(
                "and",
                [
                    _convert_logic_node(node.left, concept_names=concept_names),
                    _convert_logic_node(node.right, concept_names=concept_names),
                ],
            )
        if isinstance(node.op, ast.BitOr):
            return _compose_logic_expression(
                "or",
                [
                    _convert_logic_node(node.left, concept_names=concept_names),
                    _convert_logic_node(node.right, concept_names=concept_names),
                ],
            )
        raise ValueError(f"Unsupported runtime logic binary operator: {ast.dump(node.op)}")

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        operator_name = node.func.id.strip().lower()
        if operator_name not in {"and", "or", "not"}:
            raise ValueError(f"Unsupported logic operator in runtime expression: {node.func.id}")
        rendered_args = [
            _convert_logic_node(argument, concept_names=concept_names)
            for argument in node.args
        ]
        return _compose_logic_expression(operator_name, rendered_args)

    raise ValueError(f"Unsupported logic-expression node: {ast.dump(node)}")


def _compose_logic_expression(
    operator_name: str,
    args: list[dict[str, Any]],
) -> dict[str, Any]:
    if operator_name == "not":
        if len(args) != 1:
            raise ValueError("Runtime logic operator 'not' requires exactly one argument.")
        return {"op": "not", "args": args}

    flattened_args: list[dict[str, Any]] = []
    for argument in args:
        if argument.get("op") == operator_name:
            flattened_args.extend(argument.get("args", []))
        else:
            flattened_args.append(argument)

    if not flattened_args:
        raise ValueError(f"Runtime logic operator '{operator_name}' requires arguments.")
    if len(flattened_args) == 1:
        return flattened_args[0]
    return {"op": operator_name, "args": flattened_args}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload
