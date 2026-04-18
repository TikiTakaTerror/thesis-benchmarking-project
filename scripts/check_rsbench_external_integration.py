#!/usr/bin/env python3
"""Verify that rsbench runs capture the real external benchmark environment."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import create_app
from src.benchmarks import create_benchmark_adapter
from src.services.runtime import get_project_config, get_run_manager
from src.train import execute_real_mnlogic_managed_run


def main() -> int:
    adapter = create_benchmark_adapter("rsbench")
    environment = adapter.build_external_environment(
        dataset_name="mnlogic",
        model_family="pipeline",
    )

    if not environment.get("repo_present", False):
        _fail("rsbench external environment did not detect the official repo")
    components = dict(environment.get("components", {}))
    if not components.get("rssgen", False):
        _fail("rsbench external environment is missing rssgen")
    if not components.get("rsseval", False):
        _fail("rsbench external environment is missing rsseval")
    if not components.get("rsscount", False):
        _fail("rsbench external environment is missing rsscount")
    print(
        "[OK] rsbench external repo detected with rssgen/rsseval/rsscount components"
    )

    git_payload = dict(environment.get("git", {}))
    remote_url = str(git_payload.get("remote_url", "") or "")
    if "unitn-sml/rsbench-code" not in remote_url:
        _fail(f"Unexpected rsbench remote URL: {remote_url}")
    print(
        "[OK] rsbench git metadata captured: "
        f"commit={git_payload.get('commit')}, dirty={git_payload.get('is_dirty')}"
    )

    reference_models = dict(environment.get("official_reference_models", {}))
    xor_models = list(reference_models.get("available_xor_models", []))
    if len(xor_models) < 3:
        _fail(f"Expected at least 3 official xor models, got {xor_models}")
    if reference_models.get("mapped_reference_model") != "xorcbm":
        _fail("Pipeline should map to the official xorcbm reference model")
    if not reference_models.get("mapped_reference_model_available", False):
        _fail("Mapped official xorcbm reference model is not available")
    print(
        "[OK] Official XOR reference models discovered: "
        f"{xor_models}"
    )

    rsscount_payload = dict(environment.get("rsscount", {}))
    if "exact_available" not in rsscount_payload:
        _fail("rsscount availability probe did not return exact_available")
    print(
        "[OK] rsscount exact availability captured: "
        f"{rsscount_payload.get('exact_available')} "
        f"({rsscount_payload.get('unavailable_reason')})"
    )

    manager = get_run_manager()
    project_config = get_project_config()
    result = execute_real_mnlogic_managed_run(
        manager,
        project_config=project_config,
        model_family="pipeline",
        seed=77,
        benchmark_suite="rsbench",
        supervision="full",
        run_name="r7_rsbench_external_real_mnlogic",
        training_overrides={
            "epochs": 1,
            "learning_rate": 0.003,
            "concept_loss_weight": 2.0,
        },
        limit_per_split=16,
    )
    config_payload = yaml.safe_load(Path(result.record.config_path).read_text())
    external_environment = (
        config_payload.get("benchmark", {}).get("external_environment", {})
    )
    if not external_environment.get("repo_present", False):
        _fail("Real MNLogic rsbench run did not store external benchmark metadata")
    required_metrics = {
        "rsbench_external_repo_present",
        "rsbench_official_xor_model_count",
        "rsbench_reference_model_available",
        "rsbench_rsscount_exact_available",
    }
    missing_metrics = sorted(required_metrics.difference(result.record.metrics.keys()))
    if missing_metrics:
        _fail(f"Real MNLogic rsbench run is missing external benchmark metrics: {missing_metrics}")
    print(
        "[OK] Real MNLogic rsbench run stored external benchmark metadata and metrics"
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/runs/launch/mnlogic",
        json={
            "model_family": "pipeline",
            "seed": 78,
            "benchmark_suite": "rsbench",
            "supervision": "full",
            "run_name": "r7_api_rsbench_external_real_mnlogic",
            "training_overrides": {
                "epochs": 1,
                "learning_rate": 0.003,
                "concept_loss_weight": 2.0,
            },
            "limit_per_split": 16,
        },
    )
    if response.status_code != 200:
        _fail(
            f"Backend real-MNLogic rsbench launch failed with {response.status_code}: "
            f"{response.text}"
        )
    payload = response.json()
    evaluation_metrics = dict(payload.get("evaluation_metrics", {}))
    if "rsbench_external_repo_present" not in evaluation_metrics:
        _fail("Backend rsbench launch response is missing external benchmark metrics")

    run_id = str(payload["run"]["run_id"])
    config_response = client.get(f"/api/v1/runs/{run_id}/snapshot/config")
    if config_response.status_code != 200:
        _fail(
            f"Backend config snapshot fetch failed with {config_response.status_code}: "
            f"{config_response.text}"
        )
    config_snapshot = config_response.json().get("payload", {})
    api_environment = config_snapshot.get("benchmark", {}).get("external_environment", {})
    if not api_environment.get("official_reference_models", {}).get(
        "mapped_reference_model_available", False
    ):
        _fail("Backend API run snapshot is missing mapped official reference model metadata")
    print("[OK] Backend API stored external rsbench metadata for real MNLogic launch")

    print("[OK] rsbench external integration check passed.")
    return 0


def _fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
