#!/usr/bin/env python3
"""Create a tiny prepared MNLogic-style dataset for Phase 2 verification."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8GfkcAAAAASUVORK5CYII="
)

CONCEPT_SCHEMA = {
    "dataset_name": "mnlogic_demo",
    "concepts": [
        {
            "name": "has_triangle",
            "index": 0,
            "type": "binary",
            "description": "Whether a triangle is present.",
        },
        {
            "name": "is_red",
            "index": 1,
            "type": "binary",
            "description": "Whether the main object is red.",
        },
        {
            "name": "count_is_two",
            "index": 2,
            "type": "binary",
            "description": "Whether the object count equals two.",
        },
    ],
}

LABEL_SCHEMA = {
    "dataset_name": "mnlogic_demo",
    "labels": [
        {"id": 0, "name": "negative", "description": "Rule outcome is false."},
        {"id": 1, "name": "positive", "description": "Rule outcome is true."},
    ],
}

SPLIT_ROWS = {
    "train": [
        ("train_000", 1, 1, 1, 0, "id"),
        ("train_001", 0, 1, 0, 0, "id"),
        ("train_002", 1, 1, 1, 1, "id"),
        ("train_003", 0, 0, 1, 1, "id"),
    ],
    "val": [
        ("val_000", 1, 1, 1, 1, "id"),
        ("val_001", 0, 0, 1, 0, "id"),
    ],
    "test": [
        ("test_000", 1, 1, 1, 1, "id"),
        ("test_001", 0, 1, 0, 1, "id"),
    ],
    "ood": [
        ("ood_000", 0, 0, 0, 1, "ood"),
        ("ood_001", 1, 1, 0, 0, "ood"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/processed/mnlogic_demo",
        help="Dataset directory to create relative to the project root unless absolute.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    splits_dir = output_dir / "splits"

    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    write_json(metadata_dir / "concept_schema.json", CONCEPT_SCHEMA)
    write_json(metadata_dir / "label_schema.json", LABEL_SCHEMA)

    total_images = 0
    for split_name, rows in SPLIT_ROWS.items():
        split_path = splits_dir / f"{split_name}.csv"
        with split_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "sample_id",
                    "image_path",
                    "label_id",
                    "concept__has_triangle",
                    "concept__is_red",
                    "concept__count_is_two",
                    "group",
                ]
            )

            for sample_id, label_id, has_triangle, is_red, count_is_two, group in rows:
                relative_image_path = f"images/{sample_id}.png"
                image_path = output_dir / relative_image_path
                image_path.write_bytes(PNG_BYTES)
                total_images += 1

                writer.writerow(
                    [
                        sample_id,
                        relative_image_path,
                        label_id,
                        has_triangle,
                        is_red,
                        count_is_two,
                        group,
                    ]
                )

    print(f"[OK] Demo dataset created at: {output_dir}")
    print(f"[OK] Metadata files written: 2")
    print(f"[OK] Split files written: {len(SPLIT_ROWS)}")
    print(f"[OK] Image files written: {total_images}")
    return 0


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())

