#!/usr/bin/env python3
"""Execute managed synthetic runs and verify result storage."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.registry import load_model_config
from src.services import RunManager, RunSelection, load_project_config
from src.train import execute_training_run


RUN_CONFIG_PATH = (
    PROJECT_ROOT / "src" / "configs" / "runs" / "phase8_pipeline_storage_smoke.yaml"
)


def main() -> int:
    project_config = load_project_config()
    run_config = load_yaml(RUN_CONFIG_PATH)
    run_manager = RunManager(project_config)

    print(f"[OK] Project config loaded: {project_config.name}")
    print(f"[OK] SQLite registry path: {project_config.storage.sqlite_path}")

    created_run_ids: list[str] = []
    for seed in (11, 17):
        selection_payload = dict(run_config["selection"])
        selection_payload["seed"] = seed
        selection = RunSelection.from_dict(selection_payload)

        images, label_ids, concept_targets = build_synthetic_dataset(total_samples=64, seed=seed)
        train_batches, test_batches = split_into_batches(
            images,
            label_ids,
            concept_targets,
            batch_size=16,
            train_size=48,
        )

        config_snapshot = {
            "project": project_config.to_dict(),
            "run": {
                "name": run_config["run"]["name"],
                "phase": run_config["run"]["phase"],
                "seed": seed,
            },
            "selection": selection.to_dict(),
            "model": load_model_config(selection.model_family),
            "training": run_config["training"],
        }

        result = execute_training_run(
            run_manager,
            run_name=f"{run_config['run']['name']}_seed_{seed}",
            selection=selection,
            config_snapshot=config_snapshot,
            train_batches=train_batches,
            evaluation_splits={"test": test_batches},
            train_kwargs=dict(run_config["training"]),
        )
        created_run_ids.append(result.record.run_id)

        test_accuracy = result.evaluation_metrics.get("test_accuracy", 0.0)
        test_concept_accuracy = result.evaluation_metrics.get("test_concept_accuracy", 0.0)
        if test_accuracy < 0.95:
            print("[ERROR] Stored test accuracy is too low.", file=sys.stderr)
            return 1
        if test_concept_accuracy < 0.90:
            print("[ERROR] Stored test concept accuracy is too low.", file=sys.stderr)
            return 1

        checkpoint_path = Path(result.checkpoint_path)
        if not checkpoint_path.exists():
            print("[ERROR] Stored checkpoint path is missing.", file=sys.stderr)
            return 1

        print(
            f"[OK] Completed managed run: {result.record.run_id} "
            f"(test_accuracy={test_accuracy:.4f})"
        )

    completed_runs = {
        record.run_id: record
        for record in run_manager.list_runs(status="completed", model_family="pipeline")
        if record.run_id in created_run_ids
    }
    if len(completed_runs) != 2:
        print("[ERROR] Failed to list the newly completed runs.", file=sys.stderr)
        return 1

    comparison_paths = run_manager.compare_runs(
        created_run_ids,
        metric_names=[
            "train_label_accuracy",
            "test_accuracy",
            "test_concept_accuracy",
            "run_runtime_seconds",
        ],
        output_basename=str(run_config["storage"]["comparison_name"]),
    )

    registry_csv = project_config.paths.summaries_root / "run_registry.csv"
    registry_json = project_config.paths.summaries_root / "run_registry.json"
    if not registry_csv.exists() or not registry_json.exists():
        print("[ERROR] Registry summary exports are missing.", file=sys.stderr)
        return 1

    if not Path(comparison_paths["csv_path"]).exists() or not Path(
        comparison_paths["json_path"]
    ).exists():
        print("[ERROR] Comparison exports are missing.", file=sys.stderr)
        return 1

    print(f"[OK] Listed completed runs: {len(completed_runs)}")
    print(f"[OK] Registry CSV updated: {registry_csv}")
    print(f"[OK] Comparison CSV written: {comparison_paths['csv_path']}")
    print("[OK] Run management smoke check passed.")
    return 0


def build_synthetic_dataset(
    *,
    total_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)

    base_patterns = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=torch.float32,
    )
    repeats = (total_samples + len(base_patterns) - 1) // len(base_patterns)
    concept_targets = base_patterns.repeat(repeats, 1)[:total_samples]
    concept_targets = concept_targets[torch.randperm(total_samples)]

    positive_mask = (concept_targets[:, 0] == 1) & (concept_targets[:, 1] == 1)
    label_ids = positive_mask.long()

    images = torch.zeros(total_samples, 3, 64, 64)
    for sample_index, concepts in enumerate(concept_targets):
        if concepts[0] > 0.5:
            images[sample_index, 0, 4:28, 4:28] = 1.0
        if concepts[1] > 0.5:
            images[sample_index, 1, 4:28, 36:60] = 1.0
        if concepts[2] > 0.5:
            images[sample_index, 2, 36:60, 20:44] = 1.0

    images = (images + 0.01 * torch.randn_like(images)).clamp(0.0, 1.0)
    return images, label_ids, concept_targets


def split_into_batches(
    images: torch.Tensor,
    label_ids: torch.Tensor,
    concept_targets: torch.Tensor,
    *,
    batch_size: int,
    train_size: int,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    train_batches = make_batches(
        images[:train_size],
        label_ids[:train_size],
        concept_targets[:train_size],
        batch_size=batch_size,
    )
    test_batches = make_batches(
        images[train_size:],
        label_ids[train_size:],
        concept_targets[train_size:],
        batch_size=batch_size,
    )
    return train_batches, test_batches


def make_batches(
    images: torch.Tensor,
    label_ids: torch.Tensor,
    concept_targets: torch.Tensor,
    *,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for start_index in range(0, images.shape[0], batch_size):
        end_index = start_index + batch_size
        batches.append(
            {
                "images": images[start_index:end_index],
                "label_ids": label_ids[start_index:end_index],
                "concept_targets": concept_targets[start_index:end_index],
            }
        )
    return batches


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML file: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
