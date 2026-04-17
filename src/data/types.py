"""Typed records used by dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ConceptDefinition:
    """Describes one concept exposed by a dataset."""

    name: str
    index: int
    value_type: str = "binary"
    description: str = ""


@dataclass(frozen=True)
class LabelDefinition:
    """Describes one dataset label."""

    id: int
    name: str
    description: str = ""


@dataclass(frozen=True)
class DatasetRecord:
    """One dataset sample with path, task label, and concept targets."""

    sample_id: str
    image_path: Path
    label_id: int
    concepts: dict[str, int | float]
    split_name: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetSplit:
    """Container for records loaded from one named split."""

    name: str
    records: list[DatasetRecord]
    concept_definitions: list[ConceptDefinition]
    label_definitions: list[LabelDefinition]

    def __len__(self) -> int:
        return len(self.records)

