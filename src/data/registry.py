"""Dataset registry and config helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import DatasetAdapter
from .mnlogic import MNLogicDatasetAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG_DIR = PROJECT_ROOT / "src" / "configs" / "datasets"


def get_dataset_config(dataset_name: str) -> dict:
    """Load the YAML config for a dataset."""

    config_path = DATASET_CONFIG_DIR / f"{dataset_name.replace('-', '_')}.yaml"
    if not config_path.exists():
        raise ValueError(f"Unknown dataset config: {dataset_name}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid dataset config file: {config_path}")

    return payload


def create_dataset_adapter(
    dataset_name: str,
    dataset_root: str | Path | None = None,
) -> DatasetAdapter:
    """Instantiate a dataset adapter, using config defaults when needed."""

    canonical_name = dataset_name.strip().lower()
    if canonical_name != "mnlogic":
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Phase 2 only implements MNLogic."
        )

    if dataset_root is None:
        config = get_dataset_config(canonical_name)
        dataset_root = PROJECT_ROOT / config["paths"]["prepared_root"]

    return MNLogicDatasetAdapter(dataset_root=dataset_root)

