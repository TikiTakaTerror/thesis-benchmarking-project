"""Helpers for exposing available config-backed options."""

from __future__ import annotations

from pathlib import Path

from .config import PROJECT_ROOT


CONFIG_ROOT = PROJECT_ROOT / "src" / "configs"


def list_available_options() -> dict[str, list[str]]:
    """List dataset/model/benchmark/supervision/run preset names from config files."""

    return {
        "datasets": _list_stems(CONFIG_ROOT / "datasets"),
        "model_families": _list_stems(CONFIG_ROOT / "models"),
        "benchmark_suites": _list_stems(CONFIG_ROOT / "benchmarks"),
        "supervision_settings": _list_stems(CONFIG_ROOT / "supervision"),
        "run_presets": _list_stems(CONFIG_ROOT / "runs"),
    }


def _list_stems(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))
