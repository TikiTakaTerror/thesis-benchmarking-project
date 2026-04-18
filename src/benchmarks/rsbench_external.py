"""Helpers for inspecting the local official rsbench-code repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_XOR_REFERENCE_MODELS: dict[str, str | None] = {
    "pipeline": "xorcbm",
    "deepproblog": "xordpl",
    "ltn": None,
}
XOR_COMPATIBLE_DATASETS = {"mnlogic", "synthetic_mnlogic"}


def inspect_rsbench_external_environment(
    *,
    benchmark_root: str | Path | None,
    dataset_name: str,
    model_family: str,
) -> dict[str, Any]:
    """Inspect the official rsbench-code checkout used by local runs."""

    resolved_root = _resolve_benchmark_root(benchmark_root)
    repo_present = resolved_root.exists()
    components = {
        "rssgen": (resolved_root / "rssgen").is_dir(),
        "rsseval": (resolved_root / "rsseval").is_dir(),
        "rsscount": (resolved_root / "rsscount").is_dir(),
    }
    git = _inspect_git_repository(resolved_root) if repo_present else _missing_git_payload()
    available_xor_models = _discover_xor_reference_models(resolved_root)

    normalized_dataset = str(dataset_name).strip().lower()
    normalized_model_family = str(model_family).strip().lower()
    mapped_reference_model = (
        OFFICIAL_XOR_REFERENCE_MODELS.get(normalized_model_family)
        if normalized_dataset in XOR_COMPATIBLE_DATASETS
        else None
    )
    mapped_reference_model_available = bool(
        mapped_reference_model and mapped_reference_model in available_xor_models
    )
    rsscount = _inspect_rsscount_availability(resolved_root)

    warnings: list[str] = []
    if not repo_present:
        warnings.append(
            f"Configured rsbench root does not exist: {resolved_root}"
        )
    if repo_present and not git.get("is_git_repository", False):
        warnings.append(
            f"Configured rsbench root is not a git repository: {resolved_root}"
        )
    if git.get("is_dirty", False):
        warnings.append("Official rsbench checkout is dirty; benchmark metadata may be non-reproducible.")
    if not mapped_reference_model_available and mapped_reference_model is not None:
        warnings.append(
            f"Mapped official XOR reference model is missing: {mapped_reference_model}"
        )
    if not rsscount.get("exact_available", False):
        unavailable_reason = str(rsscount.get("unavailable_reason", "")).strip()
        if unavailable_reason:
            warnings.append(f"Official rsscount exact counter unavailable: {unavailable_reason}")

    return {
        "suite_name": "rsbench",
        "root_dir": str(resolved_root),
        "repo_present": repo_present,
        "dataset_name": normalized_dataset,
        "model_family": normalized_model_family,
        "components": components,
        "git": git,
        "official_reference_models": {
            "available_xor_models": available_xor_models,
            "mapped_reference_model": mapped_reference_model,
            "mapped_reference_model_available": mapped_reference_model_available,
        },
        "rsscount": rsscount,
        "warnings": warnings,
    }


def extract_rsbench_external_metrics(environment: dict[str, Any]) -> dict[str, float]:
    """Flatten inspected rsbench environment metadata into numeric metrics."""

    components = dict(environment.get("components", {}))
    git = dict(environment.get("git", {}))
    reference_models = dict(environment.get("official_reference_models", {}))
    rsscount = dict(environment.get("rsscount", {}))

    return {
        "rsbench_external_repo_present": float(bool(environment.get("repo_present", False))),
        "rsbench_external_repo_dirty": float(bool(git.get("is_dirty", False))),
        "rsbench_external_rssgen_present": float(bool(components.get("rssgen", False))),
        "rsbench_external_rsseval_present": float(bool(components.get("rsseval", False))),
        "rsbench_external_rsscount_present": float(bool(components.get("rsscount", False))),
        "rsbench_official_xor_model_count": float(
            len(list(reference_models.get("available_xor_models", [])))
        ),
        "rsbench_reference_model_available": float(
            bool(reference_models.get("mapped_reference_model_available", False))
        ),
        "rsbench_rsscount_exact_available": float(
            bool(rsscount.get("exact_available", False))
        ),
    }


def _resolve_benchmark_root(benchmark_root: str | Path | None) -> Path:
    if benchmark_root is None:
        return (PROJECT_ROOT / "external" / "rsbench-code").resolve()
    path = Path(benchmark_root).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _inspect_git_repository(repo_root: Path) -> dict[str, Any]:
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return _missing_git_payload()

    return {
        "is_git_repository": True,
        "commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "branch": _git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "remote_url": _git_output(repo_root, "config", "--get", "remote.origin.url"),
        "is_dirty": bool(_git_output(repo_root, "status", "--porcelain")),
    }


def _missing_git_payload() -> dict[str, Any]:
    return {
        "is_git_repository": False,
        "commit": None,
        "branch": None,
        "remote_url": None,
        "is_dirty": False,
    }


def _discover_xor_reference_models(repo_root: Path) -> list[str]:
    models_root = repo_root / "rsseval" / "rss" / "models"
    if not models_root.is_dir():
        return []
    return sorted(
        path.stem
        for path in models_root.glob("xor*.py")
        if path.is_file() and path.stem != "__init__"
    )


def _inspect_rsscount_availability(repo_root: Path) -> dict[str, Any]:
    script_path = repo_root / "rsscount" / "gen-rss-count.py"
    if not script_path.exists():
        return {
            "script_path": str(script_path),
            "exact_available": False,
            "checked_with_python": sys.executable,
            "import_probe": {"sklearn": False, "pyeda.inter": False},
            "unavailable_reason": "rsscount/gen-rss-count.py is missing",
        }

    import_probe = {
        "sklearn": _import_available("sklearn"),
        "pyeda.inter": _import_available("pyeda.inter"),
    }
    if not import_probe["pyeda.inter"]:
        return {
            "script_path": str(script_path),
            "exact_available": False,
            "checked_with_python": sys.executable,
            "import_probe": import_probe,
            "unavailable_reason": "pyeda.inter is not importable in the current environment",
        }

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime probe
        return {
            "script_path": str(script_path),
            "exact_available": False,
            "checked_with_python": sys.executable,
            "import_probe": import_probe,
            "unavailable_reason": f"rsscount help probe failed: {exc}",
        }

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        return {
            "script_path": str(script_path),
            "exact_available": False,
            "checked_with_python": sys.executable,
            "import_probe": import_probe,
            "unavailable_reason": stderr or "rsscount help probe returned a non-zero exit code",
        }

    return {
        "script_path": str(script_path),
        "exact_available": True,
        "checked_with_python": sys.executable,
        "import_probe": import_probe,
        "unavailable_reason": None,
    }


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:  # pragma: no cover - defensive runtime probe
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _import_available(module_name: str) -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return probe.returncode == 0
