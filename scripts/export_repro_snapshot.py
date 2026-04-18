#!/usr/bin/env python3
"""Export a reproducibility snapshot for the current project state."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.catalog import list_available_options
from src.services.config import load_project_config


PACKAGE_NAMES = [
    "torch",
    "torchvision",
    "LTNtorch",
    "deepproblog",
    "problog",
    "numpy",
    "pandas",
    "scikit-learn",
    "PyYAML",
    "pydantic",
    "fastapi",
    "uvicorn",
    "python-multipart",
    "Jinja2",
    "matplotlib",
    "tqdm",
    "pytest",
    "httpx",
    "ruff",
]

VERIFICATION_COMMANDS = [
    "python scripts/check_project_ready.py",
    "python scripts/check_environment.py",
    "python scripts/check_benchmark_adapters.py",
]


def main() -> int:
    project_config = load_project_config()
    output_path = _build_output_path(project_config.paths.summaries_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": project_config.to_dict(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {package_name: _safe_version(package_name) for package_name in PACKAGE_NAMES},
        "git": _collect_git_info(),
        "available_options": list_available_options(),
        "verification_commands": VERIFICATION_COMMANDS,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(f"[OK] Reproducibility snapshot written: {output_path}")
    git_commit = snapshot["git"].get("commit")
    if git_commit:
        print(f"[OK] Git commit: {git_commit}")
    print(f"[OK] Project phase captured: {project_config.phase}")
    return 0


def _build_output_path(summaries_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return summaries_root / f"repro_snapshot__{timestamp}.json"


def _safe_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _collect_git_info() -> dict[str, object]:
    commit = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    status_output = _git(["status", "--short"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status_output),
        "status_short": status_output.splitlines() if status_output else [],
    }


def _git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
