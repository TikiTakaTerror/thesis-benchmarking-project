#!/usr/bin/env python3
"""Validate a prepared Kand-Logic dataset and print a short summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DatasetValidationError, create_dataset_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Prepared dataset root. If omitted, use the default Kand-Logic root "
            "from src/configs/datasets/kand_logic.yaml."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_root = None
    if args.dataset_dir:
        dataset_root = Path(args.dataset_dir)
        if not dataset_root.is_absolute():
            dataset_root = PROJECT_ROOT / dataset_root

    try:
        adapter = create_dataset_adapter("kand_logic", dataset_root=dataset_root)
        adapter.validate_layout()
    except DatasetValidationError as exc:
        print(f"[ERROR] Dataset validation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Unexpected error: {exc}", file=sys.stderr)
        return 1

    concept_schema = adapter.get_concept_schema()
    label_schema = adapter.get_label_schema()
    summary = adapter.summarize()

    print(f"[OK] Dataset root: {adapter.dataset_root}")
    print(f"[OK] Concepts loaded: {len(concept_schema)}")
    print(f"[OK] Labels loaded: {len(label_schema)}")
    print(f"[OK] train split: {summary['train']} samples")
    print(f"[OK] val split: {summary['val']} samples")
    print(f"[OK] test split: {summary['test']} samples")
    if "ood" in summary:
        print(f"[OK] ood split: {summary['ood']} samples")
    else:
        print("[OK] ood split: not present")
    print("[OK] Dataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
