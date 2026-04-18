#!/usr/bin/env python3
"""Convert raw rsbench MNLogic output into the local prepared-manifest format."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DatasetValidationError, create_dataset_adapter


DEFAULT_RAW_DIR = "data/raw/mnlogic/rsbench_generator_output"
DEFAULT_OUTPUT_DIR = "data/processed/mnlogic"
DEFAULT_CONFIG_PATH = "data/raw/mnlogic/xor_r1.yml"
SPLIT_NAMES = ("train", "val", "test", "ood")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help="Raw rsbench MNLogic output directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Prepared MNLogic output directory.",
    )
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Generator config used to create the raw dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the prepared output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = resolve_path(args.raw_dir)
    output_dir = resolve_path(args.output_dir)
    config_path = resolve_path(args.config_path)

    if not raw_dir.exists():
        print(f"[ERROR] Raw directory not found: {raw_dir}", file=sys.stderr)
        return 1
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return 1

    if output_dir.exists():
        if not args.overwrite:
            print(
                f"[ERROR] Output directory already exists: {output_dir}\n"
                "Use --overwrite to replace it.",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(output_dir)

    generator_config = load_yaml(config_path)
    concept_names = load_concept_names(generator_config, raw_dir)
    logic_expression = generator_config.get("logic")

    images_root = output_dir / "images"
    metadata_root = output_dir / "metadata"
    splits_root = output_dir / "splits"
    images_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)

    write_json(
        metadata_root / "concept_schema.json",
        build_concept_schema(concept_names),
    )
    write_json(
        metadata_root / "label_schema.json",
        build_label_schema(),
    )

    split_summaries: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    total_samples = 0

    for split_name in SPLIT_NAMES:
        split_summary = convert_split(
            raw_dir=raw_dir,
            output_dir=output_dir,
            split_name=split_name,
            concept_names=concept_names,
        )
        split_summaries[split_name] = split_summary
        total_samples += int(split_summary["num_samples"])

        label_distribution = split_summary["label_distribution"]
        if len(label_distribution) <= 1:
            warnings.append(
                "Raw split '{split}' contains a single label only: {labels}. "
                "This comes from the upstream rsbench xor generator output and is "
                "not a prepared-dataset conversion bug.".format(
                    split=split_name,
                    labels=sorted(label_distribution.keys()),
                )
            )

    write_json(
        metadata_root / "source_info.json",
        {
            "dataset_name": "mnlogic",
            "source": "rsbench-code/rssgen xor generator",
            "raw_root": str(raw_dir),
            "generator_config_path": str(config_path),
            "logic_expression": str(logic_expression),
            "concept_names": concept_names,
            "split_summaries": split_summaries,
            "warnings": warnings,
        },
    )

    try:
        adapter = create_dataset_adapter("mnlogic", dataset_root=output_dir)
        adapter.validate_layout()
    except DatasetValidationError as exc:
        print(f"[ERROR] Prepared dataset validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Raw MNLogic root: {raw_dir}")
    print(f"[OK] Prepared MNLogic root: {output_dir}")
    print(f"[OK] Generator config used: {config_path}")
    print(f"[OK] Concepts written: {len(concept_names)}")
    print(f"[OK] Total samples converted: {total_samples}")
    for split_name in SPLIT_NAMES:
        split_summary = split_summaries[split_name]
        print(
            "[OK] {split} split: {count} samples, labels={labels}".format(
                split=split_name,
                count=split_summary["num_samples"],
                labels=split_summary["label_distribution"],
            )
        )
    for warning in warnings:
        print(f"[WARN] {warning}")
    print("[OK] Prepared MNLogic dataset validation passed.")
    return 0


def convert_split(
    *,
    raw_dir: Path,
    output_dir: Path,
    split_name: str,
    concept_names: list[str],
) -> dict[str, Any]:
    raw_split_dir = raw_dir / split_name
    if not raw_split_dir.exists():
        raise FileNotFoundError(f"Raw split directory not found: {raw_split_dir}")

    split_files = collect_split_files(raw_split_dir)
    image_split_dir = output_dir / "images" / split_name
    image_split_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "splits" / f"{split_name}.csv"

    fieldnames = [
        "sample_id",
        "image_path",
        "label_id",
        *[f"concept__{concept_name}" for concept_name in concept_names],
        "raw_split",
        "raw_index",
    ]

    label_distribution: Counter[int] = Counter()
    concept_distribution: Counter[tuple[int, ...]] = Counter()

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for raw_index, image_path, metadata_path in split_files:
            metadata = joblib.load(metadata_path)
            concept_values = normalize_concepts(metadata["meta"]["concepts"])
            label_id = 1 if bool(metadata["label"]) else 0
            sample_id = f"{split_name}_{raw_index:06d}"
            prepared_image_path = image_split_dir / f"{sample_id}.png"
            shutil.copy2(image_path, prepared_image_path)

            row = {
                "sample_id": sample_id,
                "image_path": str(prepared_image_path.relative_to(output_dir)),
                "label_id": label_id,
                "raw_split": split_name,
                "raw_index": raw_index,
            }
            for concept_name, concept_value in zip(concept_names, concept_values):
                row[f"concept__{concept_name}"] = concept_value

            writer.writerow(row)
            label_distribution[label_id] += 1
            concept_distribution[tuple(concept_values)] += 1

    return {
        "num_samples": len(split_files),
        "label_distribution": {
            str(label_id): count for label_id, count in sorted(label_distribution.items())
        },
        "num_unique_concept_vectors": len(concept_distribution),
    }


def collect_split_files(split_dir: Path) -> list[tuple[int, Path, Path]]:
    png_by_index = {int(path.stem): path for path in split_dir.glob("*.png")}
    joblib_by_index = {int(path.stem): path for path in split_dir.glob("*.joblib")}
    all_indices = sorted(set(png_by_index) | set(joblib_by_index))
    if not all_indices:
        raise ValueError(f"Split contains no files: {split_dir}")

    pairs: list[tuple[int, Path, Path]] = []
    for raw_index in all_indices:
        image_path = png_by_index.get(raw_index)
        metadata_path = joblib_by_index.get(raw_index)
        if image_path is None or metadata_path is None:
            raise ValueError(
                f"Missing paired image/joblib for index {raw_index} in {split_dir}"
            )
        pairs.append((raw_index, image_path, metadata_path))
    return pairs


def normalize_concepts(values: Any) -> list[int]:
    if not isinstance(values, list):
        raise ValueError(f"Expected concept list in raw metadata, got: {values!r}")
    return [1 if bool(value) else 0 for value in values]


def build_concept_schema(concept_names: list[str]) -> dict[str, Any]:
    return {
        "dataset_name": "mnlogic",
        "concepts": [
            {
                "name": concept_name,
                "index": index,
                "type": "binary",
                "description": (
                    "Boolean concept derived from the rsbench MNLogic symbol "
                    f"'{concept_name}'."
                ),
            }
            for index, concept_name in enumerate(concept_names)
        ],
    }


def build_label_schema() -> dict[str, Any]:
    return {
        "dataset_name": "mnlogic",
        "labels": [
            {
                "id": 0,
                "name": "negative",
                "description": "The symbolic MNLogic formula evaluates to false.",
            },
            {
                "id": 1,
                "name": "positive",
                "description": "The symbolic MNLogic formula evaluates to true.",
            },
        ],
    }


def load_concept_names(generator_config: dict[str, Any], raw_dir: Path) -> list[str]:
    raw_symbols = generator_config.get("symbols")
    if isinstance(raw_symbols, list) and raw_symbols:
        return [str(symbol) for symbol in raw_symbols]

    first_joblib = next((raw_dir / "train").glob("*.joblib"), None)
    if first_joblib is None:
        raise ValueError("Could not infer concept names because the train split is empty.")
    metadata = joblib.load(first_joblib)
    concepts = metadata["meta"]["concepts"]
    return [f"concept_{index}" for index in range(len(concepts))]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
