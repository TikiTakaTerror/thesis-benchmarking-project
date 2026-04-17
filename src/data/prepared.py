"""Reusable adapter for prepared datasets backed by schema JSON and split CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .base import DatasetAdapter
from .exceptions import DatasetValidationError
from .types import ConceptDefinition, DatasetRecord, DatasetSplit, LabelDefinition


class PreparedManifestDatasetAdapter(DatasetAdapter):
    """Loads datasets from a simple prepared layout that later families can share."""

    required_split_names = ("train", "val", "test")
    optional_split_names = ("ood",)
    required_columns = ("sample_id", "image_path", "label_id")
    concept_prefix = "concept__"

    def __init__(self, dataset_root: str | Path) -> None:
        super().__init__(dataset_root)
        self._concept_schema: list[ConceptDefinition] | None = None
        self._label_schema: list[LabelDefinition] | None = None

    @property
    def images_dir(self) -> Path:
        return self.dataset_root / "images"

    @property
    def metadata_dir(self) -> Path:
        return self.dataset_root / "metadata"

    @property
    def splits_dir(self) -> Path:
        return self.dataset_root / "splits"

    @property
    def concept_schema_path(self) -> Path:
        return self.metadata_dir / "concept_schema.json"

    @property
    def label_schema_path(self) -> Path:
        return self.metadata_dir / "label_schema.json"

    def load_train_split(self, limit: int | None = None) -> DatasetSplit:
        return self._load_split("train", limit=limit)

    def load_val_split(self, limit: int | None = None) -> DatasetSplit:
        return self._load_split("val", limit=limit)

    def load_test_split(self, limit: int | None = None) -> DatasetSplit:
        return self._load_split("test", limit=limit)

    def load_ood_split(self, limit: int | None = None) -> DatasetSplit | None:
        split_path = self._split_path("ood")
        if not split_path.exists():
            return None

        return self._load_split("ood", limit=limit)

    def get_concept_schema(self) -> list[ConceptDefinition]:
        if self._concept_schema is None:
            self._concept_schema = self._load_concept_schema()

        return self._concept_schema

    def get_label_schema(self) -> list[LabelDefinition]:
        if self._label_schema is None:
            self._label_schema = self._load_label_schema()

        return self._label_schema

    def validate_layout(self) -> None:
        self._validate_required_paths()

        label_ids = {label.id for label in self.get_label_schema()}
        split_names = list(self.required_split_names)

        for optional_split in self.optional_split_names:
            if self._split_path(optional_split).exists():
                split_names.append(optional_split)

        for split_name in split_names:
            split = self._load_split(split_name)
            if len(split) == 0:
                raise DatasetValidationError(
                    f"Split '{split_name}' contains zero rows in {self.dataset_root}"
                )

            seen_sample_ids: set[str] = set()
            for record in split.records:
                if record.sample_id in seen_sample_ids:
                    raise DatasetValidationError(
                        f"Duplicate sample_id '{record.sample_id}' in split '{split_name}'"
                    )

                seen_sample_ids.add(record.sample_id)

                if record.label_id not in label_ids:
                    raise DatasetValidationError(
                        f"Unknown label_id {record.label_id} in split '{split_name}'"
                    )

                if not record.image_path.exists():
                    raise DatasetValidationError(
                        f"Missing image file for sample '{record.sample_id}': "
                        f"{record.image_path}"
                    )

    def summarize(self) -> dict[str, int]:
        """Return split sizes for all available splits."""

        summary = {
            "train": len(self.load_train_split()),
            "val": len(self.load_val_split()),
            "test": len(self.load_test_split()),
        }

        ood_split = self.load_ood_split()
        if ood_split is not None:
            summary["ood"] = len(ood_split)

        return summary

    def _validate_required_paths(self) -> None:
        required_paths = [
            self.dataset_root,
            self.images_dir,
            self.metadata_dir,
            self.splits_dir,
            self.concept_schema_path,
            self.label_schema_path,
        ]

        required_paths.extend(
            self._split_path(split_name) for split_name in self.required_split_names
        )

        for path in required_paths:
            if not path.exists():
                raise DatasetValidationError(f"Required path missing: {path}")

    def _load_concept_schema(self) -> list[ConceptDefinition]:
        payload = self._load_json(self.concept_schema_path)
        raw_concepts = payload.get("concepts")
        if not isinstance(raw_concepts, list) or not raw_concepts:
            raise DatasetValidationError(
                f"Invalid concept schema file: {self.concept_schema_path}"
            )

        concepts: list[ConceptDefinition] = []
        for index, item in enumerate(raw_concepts):
            if not isinstance(item, dict) or "name" not in item:
                raise DatasetValidationError(
                    f"Invalid concept schema entry at index {index}"
                )

            concepts.append(
                ConceptDefinition(
                    name=str(item["name"]),
                    index=int(item.get("index", index)),
                    value_type=str(item.get("type", "binary")),
                    description=str(item.get("description", "")),
                )
            )

        concepts.sort(key=lambda concept: concept.index)
        return concepts

    def _load_label_schema(self) -> list[LabelDefinition]:
        payload = self._load_json(self.label_schema_path)
        raw_labels = payload.get("labels")
        if not isinstance(raw_labels, list) or not raw_labels:
            raise DatasetValidationError(
                f"Invalid label schema file: {self.label_schema_path}"
            )

        labels: list[LabelDefinition] = []
        for index, item in enumerate(raw_labels):
            if not isinstance(item, dict) or "id" not in item or "name" not in item:
                raise DatasetValidationError(f"Invalid label schema entry at index {index}")

            labels.append(
                LabelDefinition(
                    id=int(item["id"]),
                    name=str(item["name"]),
                    description=str(item.get("description", "")),
                )
            )

        labels.sort(key=lambda label: label.id)
        return labels

    def _load_split(self, split_name: str, limit: int | None = None) -> DatasetSplit:
        split_path = self._split_path(split_name)
        if not split_path.exists():
            raise DatasetValidationError(f"Split file not found: {split_path}")

        concept_definitions = self.get_concept_schema()
        label_definitions = self.get_label_schema()
        concept_names = [concept.name for concept in concept_definitions]
        concept_columns = [f"{self.concept_prefix}{name}" for name in concept_names]

        with split_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DatasetValidationError(f"Split file has no header: {split_path}")

            fieldnames = set(reader.fieldnames)
            missing_columns = set(self.required_columns).difference(fieldnames)
            if missing_columns:
                raise DatasetValidationError(
                    f"Split file {split_path} is missing required columns: "
                    f"{sorted(missing_columns)}"
                )

            missing_concept_columns = [
                column for column in concept_columns if column not in fieldnames
            ]
            if missing_concept_columns:
                raise DatasetValidationError(
                    f"Split file {split_path} is missing concept columns: "
                    f"{missing_concept_columns}"
                )

            records: list[DatasetRecord] = []
            for row_index, row in enumerate(reader, start=2):
                if limit is not None and len(records) >= limit:
                    break

                sample_id = self._require_value(
                    row=row,
                    field_name="sample_id",
                    split_name=split_name,
                    row_index=row_index,
                )
                image_path = self._resolve_image_path(
                    self._require_value(
                        row=row,
                        field_name="image_path",
                        split_name=split_name,
                        row_index=row_index,
                    )
                )

                label_value = self._require_value(
                    row=row,
                    field_name="label_id",
                    split_name=split_name,
                    row_index=row_index,
                )

                try:
                    label_id = int(label_value)
                except ValueError as exc:
                    raise DatasetValidationError(
                        f"Invalid label_id '{label_value}' in split '{split_name}' "
                        f"at CSV row {row_index}"
                    ) from exc

                concepts = {
                    concept_name: self._parse_numeric_value(
                        self._require_value(
                            row=row,
                            field_name=f"{self.concept_prefix}{concept_name}",
                            split_name=split_name,
                            row_index=row_index,
                        )
                    )
                    for concept_name in concept_names
                }

                metadata = {
                    key: value
                    for key, value in row.items()
                    if key
                    and key not in self.required_columns
                    and key not in concept_columns
                    and value not in (None, "")
                }

                records.append(
                    DatasetRecord(
                        sample_id=sample_id,
                        image_path=image_path,
                        label_id=label_id,
                        concepts=concepts,
                        split_name=split_name,
                        metadata=metadata,
                    )
                )

        return DatasetSplit(
            name=split_name,
            records=records,
            concept_definitions=concept_definitions,
            label_definitions=label_definitions,
        )

    def _split_path(self, split_name: str) -> Path:
        return self.splits_dir / f"{split_name}.csv"

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path

        return (self.dataset_root / path).resolve()

    @staticmethod
    def _load_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise DatasetValidationError(f"Expected a JSON object in {path}")

        return payload

    @staticmethod
    def _require_value(
        row: dict[str, str | None],
        field_name: str,
        split_name: str,
        row_index: int,
    ) -> str:
        value = row.get(field_name)
        if value is None or value == "":
            raise DatasetValidationError(
                f"Missing value for '{field_name}' in split '{split_name}' "
                f"at CSV row {row_index}"
            )

        return value

    @staticmethod
    def _parse_numeric_value(raw_value: str) -> int | float:
        if any(character in raw_value for character in (".", "e", "E")):
            return float(raw_value)

        return int(raw_value)

