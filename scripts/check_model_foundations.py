#!/usr/bin/env python3
"""Smoke-check the Phase 3 model foundations."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    PredictionHeadConfig,
    build_prediction_head,
    build_shared_encoder,
    count_trainable_parameters,
    create_model_adapter_stub,
    load_model_config,
    load_module_bundle,
    load_shared_encoder_config,
    save_module_bundle,
)


def main() -> int:
    torch.manual_seed(7)

    model_config = load_model_config("pipeline")
    print(f"[OK] Loaded model config: {model_config['family']}")

    encoder_config = load_shared_encoder_config("pipeline")
    encoder = build_shared_encoder(encoder_config)
    encoder.eval()
    print(f"[OK] Shared encoder built: {encoder_config.name}")
    print(f"[OK] Encoder trainable parameters: {count_trainable_parameters(encoder)}")

    head_settings = model_config.get("heads", {})
    concept_head_config = PredictionHeadConfig.from_dict(
        head_settings.get("concept_head"),
        input_dim=encoder_config.feature_dim,
        output_dim=3,
    )
    label_head_config = PredictionHeadConfig.from_dict(
        head_settings.get("label_head"),
        input_dim=encoder_config.feature_dim,
        output_dim=2,
    )

    concept_head = build_prediction_head(concept_head_config)
    label_head = build_prediction_head(label_head_config)
    concept_head.eval()
    label_head.eval()

    dummy_images = torch.randn(
        2,
        encoder_config.input_channels,
        encoder_config.input_size[0],
        encoder_config.input_size[1],
    )

    with torch.no_grad():
        encoder_features = encoder(dummy_images)
        concept_logits = concept_head(encoder_features)
        label_logits = label_head(encoder_features)

    print(f"[OK] Encoder output shape: {tuple(encoder_features.shape)}")
    print(f"[OK] Concept logits shape: {tuple(concept_logits.shape)}")
    print(f"[OK] Label logits shape: {tuple(label_logits.shape)}")

    checkpoint_path = (
        PROJECT_ROOT
        / "results"
        / "runs"
        / "phase3_shared_encoder_smoke"
        / "shared_components.pt"
    )
    save_module_bundle(
        checkpoint_path,
        modules={
            "encoder": encoder,
            "concept_head": concept_head,
            "label_head": label_head,
        },
        config={
            "shared_encoder": encoder_config.to_dict(),
            "concept_head": concept_head_config.__dict__,
            "label_head": label_head_config.__dict__,
        },
        metadata={"phase": 3, "purpose": "shared_encoder_smoke_check"},
    )

    reloaded_encoder = build_shared_encoder(encoder_config)
    reloaded_concept_head = build_prediction_head(concept_head_config)
    reloaded_label_head = build_prediction_head(label_head_config)
    reloaded_encoder.eval()
    reloaded_concept_head.eval()
    reloaded_label_head.eval()

    load_module_bundle(
        checkpoint_path,
        modules={
            "encoder": reloaded_encoder,
            "concept_head": reloaded_concept_head,
            "label_head": reloaded_label_head,
        },
    )

    with torch.no_grad():
        reloaded_features = reloaded_encoder(dummy_images)
        reloaded_concept_logits = reloaded_concept_head(reloaded_features)
        reloaded_label_logits = reloaded_label_head(reloaded_features)

    if not torch.allclose(encoder_features, reloaded_features):
        print("[ERROR] Encoder checkpoint roundtrip mismatch.", file=sys.stderr)
        return 1

    if not torch.allclose(concept_logits, reloaded_concept_logits):
        print("[ERROR] Concept head checkpoint roundtrip mismatch.", file=sys.stderr)
        return 1

    if not torch.allclose(label_logits, reloaded_label_logits):
        print("[ERROR] Label head checkpoint roundtrip mismatch.", file=sys.stderr)
        return 1

    print("[OK] Checkpoint roundtrip matched.")

    for family_name in ("pipeline", "ltn", "deepproblog"):
        adapter = create_model_adapter_stub(family_name)
        print(
            f"[OK] Registered adapter: {family_name} -> "
            f"{adapter.__class__.__name__}"
        )

    print("[OK] Model foundations check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
