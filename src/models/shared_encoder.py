"""Reusable shared encoder implementation and config parsing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from torch import nn


@dataclass(frozen=True)
class SharedEncoderConfig:
    """Configuration for the common image encoder used across families."""

    name: str = "small_cnn"
    pretrained: bool = False
    input_channels: int = 3
    input_size: tuple[int, int] = (64, 64)
    conv_channels: tuple[int, ...] = (32, 64)
    feature_dim: int = 128
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SharedEncoderConfig":
        payload = dict(payload or {})
        raw_input_size = payload.get("input_size", (64, 64))
        if not isinstance(raw_input_size, (list, tuple)) or len(raw_input_size) != 2:
            raise ValueError("shared_encoder.input_size must contain exactly two items")

        raw_conv_channels = payload.get("conv_channels", (32, 64))
        if not isinstance(raw_conv_channels, (list, tuple)) or not raw_conv_channels:
            raise ValueError("shared_encoder.conv_channels must be a non-empty sequence")

        return cls(
            name=str(payload.get("name", "small_cnn")),
            pretrained=bool(payload.get("pretrained", False)),
            input_channels=int(payload.get("input_channels", 3)),
            input_size=(int(raw_input_size[0]), int(raw_input_size[1])),
            conv_channels=tuple(int(channel) for channel in raw_conv_channels),
            feature_dim=int(payload.get("feature_dim", 128)),
            dropout=float(payload.get("dropout", 0.1)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to a JSON/YAML-friendly dictionary."""

        payload = asdict(self)
        payload["input_size"] = list(self.input_size)
        payload["conv_channels"] = list(self.conv_channels)
        return payload


class SmallCNNEncoder(nn.Module):
    """Compact CNN encoder used as the shared baseline across families."""

    def __init__(self, config: SharedEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.output_dim = config.feature_dim

        layers: list[nn.Module] = []
        in_channels = config.input_channels
        for out_channels in config.conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, config.feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, images):
        features = self.feature_extractor(images)
        return self.projector(features)


def build_shared_encoder(config: SharedEncoderConfig) -> nn.Module:
    """Factory for the shared encoder."""

    if config.name != "small_cnn":
        raise ValueError(
            f"Unsupported shared encoder '{config.name}'. "
            "Phase 3 only provides 'small_cnn'."
        )

    return SmallCNNEncoder(config)


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in a module."""

    return sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )

