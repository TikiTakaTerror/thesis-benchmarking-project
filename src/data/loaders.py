"""PyTorch dataset and dataloader helpers for prepared datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .base import DatasetAdapter
from .registry import create_dataset_adapter
from .types import DatasetRecord, DatasetSplit
from ..models.registry import load_model_config
from ..models.shared_encoder import SharedEncoderConfig


@dataclass(frozen=True)
class ImageTensorConfig:
    """Resolved image-to-tensor settings for one model family."""

    input_channels: int
    input_size: tuple[int, int]
    batch_size: int
    num_workers: int = 0

    @classmethod
    def from_model_family(
        cls,
        model_family: str,
        *,
        model_config: Mapping[str, object] | None = None,
        batch_size: int | None = None,
        num_workers: int = 0,
    ) -> "ImageTensorConfig":
        resolved_model_config = dict(model_config or load_model_config(model_family))
        shared_encoder = SharedEncoderConfig.from_dict(
            resolved_model_config.get("shared_encoder")
        )
        training_defaults = dict(resolved_model_config.get("training_defaults", {}))
        resolved_batch_size = int(
            batch_size if batch_size is not None else training_defaults.get("batch_size", 16)
        )
        return cls(
            input_channels=shared_encoder.input_channels,
            input_size=shared_encoder.input_size,
            batch_size=resolved_batch_size,
            num_workers=int(num_workers),
        )


@dataclass(frozen=True)
class PreparedDataLoaders:
    """Grouped train/eval dataloaders plus resolved dataset/model metadata."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    ood_loader: DataLoader | None
    tensor_config: ImageTensorConfig
    dataset_concept_names: tuple[str, ...]
    dataset_label_names: tuple[str, ...]
    model_concept_names: tuple[str, ...]
    concept_names_match: bool


class PreparedImageTensorDataset(Dataset):
    """Wrap a prepared dataset split as a PyTorch dataset."""

    def __init__(
        self,
        split: DatasetSplit,
        *,
        tensor_config: ImageTensorConfig,
    ) -> None:
        self.split = split
        self.tensor_config = tensor_config
        self.records = list(split.records)
        self.concept_names = tuple(
            concept.name for concept in split.concept_definitions
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image_tensor = self._load_image(record.image_path)
        concept_targets = torch.tensor(
            [float(record.concepts[concept_name]) for concept_name in self.concept_names],
            dtype=torch.float32,
        )
        return {
            "images": image_tensor,
            "label_ids": torch.tensor(record.label_id, dtype=torch.long),
            "concept_targets": concept_targets,
            "sample_id": record.sample_id,
            "image_path": str(record.image_path),
            "split_name": record.split_name,
        }

    def _load_image(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            if self.tensor_config.input_channels == 1:
                image = image.convert("L")
            elif self.tensor_config.input_channels == 3:
                image = image.convert("RGB")
            else:
                raise ValueError(
                    "PreparedImageTensorDataset only supports 1 or 3 input channels, "
                    f"got {self.tensor_config.input_channels}."
                )

            image = TF.resize(
                image,
                size=list(self.tensor_config.input_size),
                interpolation=InterpolationMode.BILINEAR,
            )
            tensor = TF.pil_to_tensor(image).float() / 255.0
        return tensor


def build_split_dataloader(
    split: DatasetSplit,
    *,
    tensor_config: ImageTensorConfig,
    shuffle: bool,
) -> DataLoader:
    """Build a PyTorch DataLoader for one prepared dataset split."""

    dataset = PreparedImageTensorDataset(
        split,
        tensor_config=tensor_config,
    )
    return DataLoader(
        dataset,
        batch_size=tensor_config.batch_size,
        shuffle=shuffle,
        num_workers=tensor_config.num_workers,
    )


def build_prepared_dataloaders(
    *,
    dataset_name: str,
    model_family: str,
    model_config: Mapping[str, object] | None = None,
    dataset_root: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    shuffle_train: bool = True,
    limit_per_split: int | None = None,
) -> PreparedDataLoaders:
    """Build model-compatible PyTorch dataloaders from a prepared dataset."""

    adapter = create_dataset_adapter(dataset_name, dataset_root=dataset_root)
    adapter.validate_layout()
    tensor_config = ImageTensorConfig.from_model_family(
        model_family,
        model_config=model_config,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    train_split = adapter.load_train_split(limit=limit_per_split)
    val_split = adapter.load_val_split(limit=limit_per_split)
    test_split = adapter.load_test_split(limit=limit_per_split)
    ood_split = adapter.load_ood_split(limit=limit_per_split)

    dataset_concept_names = tuple(
        concept.name for concept in adapter.get_concept_schema()
    )
    dataset_label_names = tuple(
        label.name for label in adapter.get_label_schema()
    )
    resolved_model_config = dict(model_config or load_model_config(model_family))
    model_concept_names = tuple(
        str(name) for name in resolved_model_config.get("concepts", [])
    )

    if len(dataset_concept_names) != len(model_concept_names):
        raise ValueError(
            f"Dataset '{dataset_name}' exposes {len(dataset_concept_names)} concepts, "
            f"but model family '{model_family}' expects {len(model_concept_names)}."
        )

    return PreparedDataLoaders(
        train_loader=build_split_dataloader(
            train_split,
            tensor_config=tensor_config,
            shuffle=shuffle_train,
        ),
        val_loader=build_split_dataloader(
            val_split,
            tensor_config=tensor_config,
            shuffle=False,
        ),
        test_loader=build_split_dataloader(
            test_split,
            tensor_config=tensor_config,
            shuffle=False,
        ),
        ood_loader=(
            build_split_dataloader(
                ood_split,
                tensor_config=tensor_config,
                shuffle=False,
            )
            if ood_split is not None
            else None
        ),
        tensor_config=tensor_config,
        dataset_concept_names=dataset_concept_names,
        dataset_label_names=dataset_label_names,
        model_concept_names=model_concept_names,
        concept_names_match=(dataset_concept_names == model_concept_names),
    )
