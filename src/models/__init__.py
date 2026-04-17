"""Model family packages and shared model abstractions."""

from .base import ModelAdapter, ModelOutputs, PhaseStubModelAdapter
from .checkpoints import load_module_bundle, save_module_bundle
from .heads import PredictionHead, PredictionHeadConfig, build_prediction_head
from .registry import (
    create_model_adapter_stub,
    get_model_adapter_class,
    load_model_config,
    load_shared_encoder_config,
)
from .shared_encoder import (
    SharedEncoderConfig,
    SmallCNNEncoder,
    build_shared_encoder,
    count_trainable_parameters,
)

__all__ = [
    "ModelAdapter",
    "ModelOutputs",
    "PhaseStubModelAdapter",
    "PredictionHead",
    "PredictionHeadConfig",
    "SharedEncoderConfig",
    "SmallCNNEncoder",
    "build_prediction_head",
    "build_shared_encoder",
    "count_trainable_parameters",
    "create_model_adapter_stub",
    "get_model_adapter_class",
    "load_model_config",
    "load_module_bundle",
    "load_shared_encoder_config",
    "save_module_bundle",
]
