"""Common dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .types import ConceptDefinition, DatasetSplit, LabelDefinition


class DatasetAdapter(ABC):
    """Base interface implemented by all dataset adapters."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical dataset name."""

    @abstractmethod
    def load_train_split(self, limit: int | None = None) -> DatasetSplit:
        """Load the training split."""

    @abstractmethod
    def load_val_split(self, limit: int | None = None) -> DatasetSplit:
        """Load the validation split."""

    @abstractmethod
    def load_test_split(self, limit: int | None = None) -> DatasetSplit:
        """Load the in-distribution test split."""

    @abstractmethod
    def load_ood_split(self, limit: int | None = None) -> DatasetSplit | None:
        """Load the OOD split when available."""

    @abstractmethod
    def get_concept_schema(self) -> list[ConceptDefinition]:
        """Return the concept schema in a stable order."""

    @abstractmethod
    def get_label_schema(self) -> list[LabelDefinition]:
        """Return the label schema."""

    @abstractmethod
    def validate_layout(self) -> None:
        """Raise if the dataset layout does not satisfy the adapter contract."""

