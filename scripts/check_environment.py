#!/usr/bin/env python3
"""Smoke-check the Phase 1 Python environment and repository structure."""

from __future__ import annotations

import importlib
import pathlib
import sys


REQUIRED_IMPORTS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("deepproblog", "deepproblog"),
    ("problog", "problog"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("yaml", "PyYAML"),
    ("pydantic", "pydantic"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("multipart", "python-multipart"),
    ("jinja2", "Jinja2"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
]

REQUIRED_PATHS = [
    "external/rsbench-code",
    "external/deepproblog",
    "external/LTNtorch",
    "src/configs",
    "results/runs",
]


def main() -> int:
    if sys.version_info[:2] != (3, 10):
        print(
            f"[ERROR] Expected Python 3.10.x, got {sys.version.split()[0]}",
            file=sys.stderr,
        )
        return 1

    print(f"[OK] Python version: {sys.version.split()[0]}")

    for module_name, display_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            print(f"[ERROR] Failed to import {display_name}: {exc}", file=sys.stderr)
            return 1

        version = getattr(module, "__version__", "unknown")
        print(f"[OK] Imported {display_name} ({version})")

    project_root = pathlib.Path(__file__).resolve().parent.parent

    for relative_path in REQUIRED_PATHS:
        target_path = project_root / relative_path
        if not target_path.exists():
            print(
                f"[ERROR] Required path missing: {relative_path}",
                file=sys.stderr,
            )
            return 1

        print(f"[OK] Required path exists: {relative_path}")

    print("[OK] Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
