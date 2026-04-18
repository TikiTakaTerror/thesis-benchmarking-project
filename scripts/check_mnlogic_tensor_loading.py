#!/usr/bin/env python3
"""Verify real prepared MNLogic image loading and batching."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_prepared_dataloaders


def main() -> int:
    torch.manual_seed(0)

    loaders = build_prepared_dataloaders(
        dataset_name="mnlogic",
        model_family="pipeline",
        limit_per_split=8,
        shuffle_train=False,
    )

    print("[OK] Built real MNLogic dataloaders for model family: pipeline")
    print(
        "[OK] Tensor config: channels={channels}, input_size={size}, batch_size={batch}".format(
            channels=loaders.tensor_config.input_channels,
            size=loaders.tensor_config.input_size,
            batch=loaders.tensor_config.batch_size,
        )
    )
    print(f"[OK] Dataset concept names: {list(loaders.dataset_concept_names)}")
    print(f"[OK] Model concept names: {list(loaders.model_concept_names)}")
    if loaders.concept_names_match:
        print("[OK] Dataset and model concept names currently align.")
    else:
        print(
            "[WARN] Dataset and model concept names do not currently align. "
            "This is expected after R2 and will be fixed during the real-run wiring phase."
        )

    train_batch = next(iter(loaders.train_loader))
    test_batch = next(iter(loaders.test_loader))
    print(f"[OK] Train batch image shape: {tuple(train_batch['images'].shape)}")
    print(f"[OK] Train batch label shape: {tuple(train_batch['label_ids'].shape)}")
    print(
        "[OK] Train batch concept shape: {shape}".format(
            shape=tuple(train_batch["concept_targets"].shape)
        )
    )
    print(
        "[OK] Train batch pixel range: min={min_value:.4f}, max={max_value:.4f}".format(
            min_value=float(train_batch["images"].min().item()),
            max_value=float(train_batch["images"].max().item()),
        )
    )
    print(
        "[OK] Example train sample ids: {sample_ids}".format(
            sample_ids=list(train_batch["sample_id"][:3])
        )
    )
    print(f"[OK] Test batch image shape: {tuple(test_batch['images'].shape)}")

    if loaders.ood_loader is not None:
        ood_batch = next(iter(loaders.ood_loader))
        print(f"[OK] OOD batch image shape: {tuple(ood_batch['images'].shape)}")
    else:
        print("[OK] OOD loader: not present")

    print("[OK] MNLogic tensor loading check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
