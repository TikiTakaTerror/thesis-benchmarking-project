#!/usr/bin/env python3
"""Convert raw rsbench Kand-Logic output into the local prepared-manifest format."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import joblib
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DatasetValidationError, create_dataset_adapter


DEFAULT_RAW_DIR = "data/raw/kand_logic/rsbench_generator_output"
DEFAULT_OUTPUT_DIR = "data/processed/kand_logic"
DEFAULT_CONFIG_PATH = "external/rsbench-code/rssgen/examples_config/kandinksy.yml"
SPLIT_NAMES = ("train", "val", "test", "ood")
COLOR_ID_TO_NAME = {1: "red", 2: "yellow", 3: "blue"}
SHAPE_ID_TO_NAME = {4: "square", 5: "circle", 6: "triangle"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_DIR,
        help="Raw rsbench Kand-Logic output directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Prepared Kand-Logic output directory.",
    )
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Generator config used to create the raw Kand-Logic dataset.",
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
        return fail(f"Raw directory not found: {raw_dir}")
    if not config_path.exists():
        return fail(f"Config file not found: {config_path}")

    if output_dir.exists():
        if not args.overwrite:
            return fail(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    generator_config = load_yaml(config_path)
    colors = [str(color) for color in generator_config["colors"]]
    shapes = [str(shape) for shape in generator_config["shapes"]]
    n_figures = int(generator_config["n_figures"])
    n_objects = int(generator_config["n_shapes"])
    concept_names = build_concept_names(
        colors=colors,
        shapes=shapes,
        n_figures=n_figures,
        n_objects=n_objects,
    )
    compiled_positive_rule = compile_positive_rule(
        logic_expression=str(generator_config["logic"]),
        aggregator_expression=str(generator_config["aggregator_logic"]),
        colors=colors,
        shapes=shapes,
        n_figures=n_figures,
    )

    images_root = output_dir / "images"
    metadata_root = output_dir / "metadata"
    splits_root = output_dir / "splits"
    images_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)

    write_json(
        metadata_root / "concept_schema.json",
        build_concept_schema(
            concept_names=concept_names,
            colors=colors,
            shapes=shapes,
            n_figures=n_figures,
            n_objects=n_objects,
        ),
    )
    write_json(metadata_root / "label_schema.json", build_label_schema())

    split_summaries: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    total_samples = 0

    for split_name in SPLIT_NAMES:
        split_summary = convert_split(
            raw_dir=raw_dir,
            output_dir=output_dir,
            split_name=split_name,
            concept_names=concept_names,
            colors=colors,
            shapes=shapes,
            n_figures=n_figures,
            n_objects=n_objects,
        )
        split_summaries[split_name] = split_summary
        total_samples += int(split_summary["num_samples"])
        if len(split_summary["label_distribution"]) <= 1:
            warnings.append(
                "Raw split '{split}' contains a single label only: {labels}.".format(
                    split=split_name,
                    labels=sorted(split_summary["label_distribution"].keys()),
                )
            )

    write_json(
        metadata_root / "source_info.json",
        {
            "dataset_name": "kand_logic",
            "source": "rsbench-code/rssgen kandinsky generator",
            "raw_root": str(raw_dir),
            "generator_config_path": str(config_path),
            "logic_expression": str(generator_config["logic"]),
            "aggregator_logic_expression": str(generator_config["aggregator_logic"]),
            "concept_representation": "one_hot_binary_shape_and_color_slots",
            "colors": colors,
            "shapes": shapes,
            "n_figures": n_figures,
            "n_objects_per_figure": n_objects,
            "concept_names": concept_names,
            "compiled_positive_rule": compiled_positive_rule,
            "compiled_positive_rule_stats": summarize_expression(compiled_positive_rule),
            "split_summaries": split_summaries,
            "warnings": warnings,
        },
    )

    try:
        adapter = create_dataset_adapter("kand_logic", dataset_root=output_dir)
        adapter.validate_layout()
    except DatasetValidationError as exc:
        return fail(f"Prepared Kand-Logic dataset validation failed: {exc}")

    print(f"[OK] Raw Kand-Logic root: {raw_dir}")
    print(f"[OK] Prepared Kand-Logic root: {output_dir}")
    print(f"[OK] Generator config used: {config_path}")
    print(f"[OK] Concepts written: {len(concept_names)}")
    print(f"[OK] Total samples converted: {total_samples}")
    print(f"[OK] Figures per image: {n_figures}")
    print(f"[OK] Objects per figure: {n_objects}")
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
    print("[OK] Prepared Kand-Logic dataset validation passed.")
    return 0


def convert_split(
    *,
    raw_dir: Path,
    output_dir: Path,
    split_name: str,
    concept_names: list[str],
    colors: list[str],
    shapes: list[str],
    n_figures: int,
    n_objects: int,
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
    active_concept_distribution: Counter[int] = Counter()

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for raw_index, image_path, metadata_path in split_files:
            metadata = joblib.load(metadata_path)
            concept_vector = expand_raw_concepts(
                metadata["meta"]["concepts"],
                colors=colors,
                shapes=shapes,
                n_figures=n_figures,
                n_objects=n_objects,
                concept_names=concept_names,
            )
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
            for concept_name, concept_value in zip(concept_names, concept_vector):
                row[f"concept__{concept_name}"] = concept_value

            writer.writerow(row)
            label_distribution[label_id] += 1
            active_concept_distribution[int(sum(concept_vector))] += 1

    return {
        "num_samples": len(split_files),
        "label_distribution": {
            str(label_id): count for label_id, count in sorted(label_distribution.items())
        },
        "active_concept_counts": {
            str(active_count): count
            for active_count, count in sorted(active_concept_distribution.items())
        },
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


def expand_raw_concepts(
    raw_concepts: Any,
    *,
    colors: list[str],
    shapes: list[str],
    n_figures: int,
    n_objects: int,
    concept_names: list[str],
) -> list[int]:
    if not isinstance(raw_concepts, list) or len(raw_concepts) != n_figures:
        raise ValueError(
            "Expected raw Kand-Logic concepts to contain one list per figure, "
            f"got {raw_concepts!r}"
        )

    encoded: dict[str, int] = {concept_name: 0 for concept_name in concept_names}
    for figure_index, figure_values in enumerate(raw_concepts, start=1):
        if not isinstance(figure_values, list) or len(figure_values) != n_objects * 2:
            raise ValueError(
                f"Expected figure {figure_index} to expose {n_objects * 2} flattened "
                f"shape/color values, got {figure_values!r}"
            )

        for object_index in range(1, n_objects + 1):
            shape_id = int(figure_values[(object_index - 1) * 2])
            color_id = int(figure_values[(object_index - 1) * 2 + 1])
            shape_name = SHAPE_ID_TO_NAME.get(shape_id)
            color_name = COLOR_ID_TO_NAME.get(color_id)
            if shape_name is None or color_name is None:
                raise ValueError(
                    f"Unexpected raw Kand-Logic concept IDs: shape={shape_id}, color={color_id}"
                )

            for shape in shapes:
                encoded[
                    concept_name_for(
                        figure_index=figure_index,
                        object_index=object_index,
                        attribute_kind="shape",
                        attribute_value=shape,
                    )
                ] = 1 if shape_name == shape else 0
            for color in colors:
                encoded[
                    concept_name_for(
                        figure_index=figure_index,
                        object_index=object_index,
                        attribute_kind="color",
                        attribute_value=color,
                    )
                ] = 1 if color_name == color else 0

    return [encoded[concept_name] for concept_name in concept_names]


def build_concept_names(
    *,
    colors: list[str],
    shapes: list[str],
    n_figures: int,
    n_objects: int,
) -> list[str]:
    concept_names: list[str] = []
    for figure_index in range(1, n_figures + 1):
        for object_index in range(1, n_objects + 1):
            for shape in shapes:
                concept_names.append(
                    concept_name_for(
                        figure_index=figure_index,
                        object_index=object_index,
                        attribute_kind="shape",
                        attribute_value=shape,
                    )
                )
            for color in colors:
                concept_names.append(
                    concept_name_for(
                        figure_index=figure_index,
                        object_index=object_index,
                        attribute_kind="color",
                        attribute_value=color,
                    )
                )
    return concept_names


def build_concept_schema(
    *,
    concept_names: list[str],
    colors: list[str],
    shapes: list[str],
    n_figures: int,
    n_objects: int,
) -> dict[str, Any]:
    return {
        "dataset_name": "kand_logic",
        "concept_representation": "one_hot_binary_shape_and_color_slots",
        "n_figures": n_figures,
        "n_objects_per_figure": n_objects,
        "colors": colors,
        "shapes": shapes,
        "concepts": [
            {
                "name": concept_name,
                "index": index,
                "type": "binary",
                "description": describe_concept(concept_name),
            }
            for index, concept_name in enumerate(concept_names)
        ],
    }


def build_label_schema() -> dict[str, Any]:
    return {
        "dataset_name": "kand_logic",
        "labels": [
            {
                "id": 0,
                "name": "negative",
                "description": "Image does not satisfy the Kand-Logic pattern across all figures.",
            },
            {
                "id": 1,
                "name": "positive",
                "description": "Image satisfies the Kand-Logic pattern across all figures.",
            },
        ],
    }


def describe_concept(concept_name: str) -> str:
    figure_token, object_token, attribute_kind, attribute_value = concept_name.split("_", 3)
    figure_index = figure_token.replace("fig", "")
    object_index = object_token.replace("obj", "")
    return (
        f"One-hot indicator that figure {figure_index}, object {object_index} has "
        f"{attribute_kind} '{attribute_value}'."
    )


def concept_name_for(
    *,
    figure_index: int,
    object_index: int,
    attribute_kind: str,
    attribute_value: str,
) -> str:
    return f"fig{figure_index}_obj{object_index}_{attribute_kind}_{attribute_value}"


def compile_positive_rule(
    *,
    logic_expression: str,
    aggregator_expression: str,
    colors: list[str],
    shapes: list[str],
    n_figures: int,
) -> dict[str, Any]:
    per_figure_rule = ast.parse(logic_expression, mode="eval").body
    aggregator_rule = ast.parse(aggregator_expression, mode="eval").body

    figure_pattern_rules = {
        f"pattern_{figure_index}": _compile_kand_node(
            per_figure_rule,
            figure_index=figure_index,
            colors=colors,
            shapes=shapes,
            pattern_map={},
        )
        for figure_index in range(1, n_figures + 1)
    }
    return _compile_kand_node(
        aggregator_rule,
        figure_index=None,
        colors=colors,
        shapes=shapes,
        pattern_map=figure_pattern_rules,
    )


def _compile_kand_node(
    node: ast.AST,
    *,
    figure_index: int | None,
    colors: list[str],
    shapes: list[str],
    pattern_map: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(node, ast.Name):
        symbol_name = node.id
        if symbol_name in pattern_map:
            return pattern_map[symbol_name]
        raise ValueError(f"Unknown Kand logic symbol: {symbol_name}")

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.BitAnd):
            return compose_expression(
                "and",
                [
                    _compile_kand_node(
                        node.left,
                        figure_index=figure_index,
                        colors=colors,
                        shapes=shapes,
                        pattern_map=pattern_map,
                    ),
                    _compile_kand_node(
                        node.right,
                        figure_index=figure_index,
                        colors=colors,
                        shapes=shapes,
                        pattern_map=pattern_map,
                    ),
                ],
            )
        if isinstance(node.op, ast.BitOr):
            return compose_expression(
                "or",
                [
                    _compile_kand_node(
                        node.left,
                        figure_index=figure_index,
                        colors=colors,
                        shapes=shapes,
                        pattern_map=pattern_map,
                    ),
                    _compile_kand_node(
                        node.right,
                        figure_index=figure_index,
                        colors=colors,
                        shapes=shapes,
                        pattern_map=pattern_map,
                    ),
                ],
            )
        raise ValueError(f"Unsupported Kand logic binary operator: {ast.dump(node.op)}")

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        operator_name = node.func.id.strip().lower()
        if operator_name in {"and", "or"}:
            return compose_expression(
                operator_name,
                [
                    _compile_kand_node(
                        argument,
                        figure_index=figure_index,
                        colors=colors,
                        shapes=shapes,
                        pattern_map=pattern_map,
                    )
                    for argument in node.args
                ],
            )
        if operator_name == "not":
            return compose_expression(
                "not",
                [
                    _compile_kand_node(
                        node.args[0],
                        figure_index=figure_index,
                        colors=colors,
                        shapes=shapes,
                        pattern_map=pattern_map,
                    )
                ],
            )
        if operator_name in {"eq", "ne"}:
            if figure_index is None:
                raise ValueError(
                    f"Kand logic operator '{node.func.id}' requires a concrete figure index."
                )
            if len(node.args) != 2:
                raise ValueError(
                    f"Kand logic operator '{node.func.id}' requires exactly two arguments."
                )
            left_attribute, left_slot = parse_attribute_symbol(node.args[0])
            right_attribute, right_slot = parse_attribute_symbol(node.args[1])
            if left_attribute != right_attribute:
                raise ValueError(
                    f"Kand logic equality requires matching attributes, got "
                    f"{left_attribute} and {right_attribute}."
                )
            values = shapes if left_attribute == "shape" else colors
            equality_expression = compose_expression(
                "or",
                [
                    compose_expression(
                        "and",
                        [
                            {
                                "concept": concept_name_for(
                                    figure_index=figure_index,
                                    object_index=left_slot,
                                    attribute_kind=left_attribute,
                                    attribute_value=value,
                                )
                            },
                            {
                                "concept": concept_name_for(
                                    figure_index=figure_index,
                                    object_index=right_slot,
                                    attribute_kind=left_attribute,
                                    attribute_value=value,
                                )
                            },
                        ],
                    )
                    for value in values
                ],
            )
            if operator_name == "eq":
                return equality_expression
            return {"op": "not", "args": [equality_expression]}

        raise ValueError(f"Unsupported Kand logic operator: {node.func.id}")

    raise ValueError(f"Unsupported Kand logic AST node: {ast.dump(node)}")


def parse_attribute_symbol(node: ast.AST) -> tuple[str, int]:
    if not isinstance(node, ast.Name):
        raise ValueError(f"Kand attribute symbol must be a bare name, got {ast.dump(node)}")
    symbol_name = node.id
    parts = symbol_name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid Kand attribute symbol: {symbol_name}")
    attribute_kind = parts[0].strip().lower()
    slot_index = int(parts[1])
    if attribute_kind not in {"shape", "color"}:
        raise ValueError(f"Unsupported Kand attribute kind: {attribute_kind}")
    return attribute_kind, slot_index


def compose_expression(operator_name: str, args: list[dict[str, Any]]) -> dict[str, Any]:
    if operator_name == "not":
        if len(args) != 1:
            raise ValueError("compose_expression('not') requires exactly one argument")
        return {"op": "not", "args": args}

    flattened: list[dict[str, Any]] = []
    for argument in args:
        if argument.get("op") == operator_name:
            flattened.extend(argument.get("args", []))
        else:
            flattened.append(argument)

    if not flattened:
        raise ValueError(f"compose_expression('{operator_name}') requires arguments")
    if len(flattened) == 1:
        return flattened[0]
    return {"op": operator_name, "args": flattened}


def summarize_expression(expression: Mapping[str, Any]) -> dict[str, int]:
    concept_count = 0
    node_count = 1
    operator_count = 0

    if "concept" in expression:
        concept_count = 1
    else:
        operator_count = 1
        for child in expression.get("args", []):
            child_summary = summarize_expression(child)
            concept_count += child_summary["concept_count"]
            node_count += child_summary["node_count"]
            operator_count += child_summary["operator_count"]

    return {
        "concept_count": concept_count,
        "node_count": node_count,
        "operator_count": operator_count,
    }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def fail(message: str) -> int:
    print(f"[ERROR] {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
