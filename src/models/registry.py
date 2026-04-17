"""Model config helpers and adapter-class registry."""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import ModelAdapter
from .deepproblog_model import DeepProbLogModelAdapter
from .ltn_model import LTNModelAdapter
from .pipeline import PipelineModelAdapter
from .shared_encoder import SharedEncoderConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_DIR = PROJECT_ROOT / "src" / "configs" / "models"

MODEL_ADAPTERS: dict[str, type[ModelAdapter]] = {
    "pipeline": PipelineModelAdapter,
    "ltn": LTNModelAdapter,
    "deepproblog": DeepProbLogModelAdapter,
}


def load_model_config(family_name: str) -> dict:
    """Load the YAML config for a model family."""

    canonical_name = family_name.strip().lower()
    config_path = MODEL_CONFIG_DIR / f"{canonical_name}.yaml"
    if not config_path.exists():
        raise ValueError(f"Unknown model config: {family_name}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid model config file: {config_path}")

    return payload


def load_shared_encoder_config(family_name: str) -> SharedEncoderConfig:
    """Load the shared encoder config from one family config file."""

    model_config = load_model_config(family_name)
    return SharedEncoderConfig.from_dict(model_config.get("shared_encoder"))


def get_model_adapter_class(family_name: str) -> type[ModelAdapter]:
    """Resolve the adapter class for a model family."""

    canonical_name = family_name.strip().lower()
    if canonical_name not in MODEL_ADAPTERS:
        raise ValueError(f"Unsupported model family: {family_name}")

    return MODEL_ADAPTERS[canonical_name]


def create_model_adapter(family_name: str) -> ModelAdapter:
    """Instantiate the best currently available adapter for a model family."""

    canonical_name = family_name.strip().lower()
    adapter_cls = get_model_adapter_class(canonical_name)
    if canonical_name in {"pipeline", "ltn", "deepproblog"}:
        model_config = load_model_config(canonical_name)
        return adapter_cls.from_config_dict(model_config)

    shared_encoder_config = load_shared_encoder_config(canonical_name)
    return adapter_cls(shared_encoder_config=shared_encoder_config)


def create_model_adapter_stub(family_name: str) -> ModelAdapter:
    """Backward-compatible alias used by earlier smoke-check scripts."""

    return create_model_adapter(family_name)
