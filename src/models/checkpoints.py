"""Generic checkpoint helpers for shared model components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def save_module_bundle(
    checkpoint_path: str | Path,
    *,
    modules: Mapping[str, nn.Module],
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a named set of modules to one checkpoint file."""

    path = Path(checkpoint_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": config,
        "metadata": metadata or {},
        "modules": {
            module_name: module.state_dict() for module_name, module in modules.items()
        },
    }
    torch.save(payload, path)
    return path


def load_module_bundle(
    checkpoint_path: str | Path,
    *,
    modules: Mapping[str, nn.Module],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a named set of modules from a checkpoint file."""

    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location=map_location)
    state_dicts = payload.get("modules")
    if not isinstance(state_dicts, dict):
        raise ValueError(f"Checkpoint does not contain a valid module bundle: {path}")

    missing_modules = [name for name in modules if name not in state_dicts]
    if missing_modules:
        raise ValueError(
            f"Checkpoint {path} is missing module state for: {missing_modules}"
        )

    for module_name, module in modules.items():
        module.load_state_dict(state_dicts[module_name])

    return payload

