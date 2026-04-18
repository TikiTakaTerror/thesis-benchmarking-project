#!/usr/bin/env python3
"""Verify the real-MNLogic managed run path introduced in R4."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api import create_app
from src.train import build_mnlogic_runtime_context, execute_real_mnlogic_managed_run
from src.services.runtime import get_project_config, get_run_manager


def main() -> int:
    for family in ("pipeline", "ltn", "deepproblog"):
        context = build_mnlogic_runtime_context(model_family=family)
        if context["concept_names"] != ("a", "b", "c"):
            _fail(f"Runtime concept names for {family} do not match MNLogic")
        print(
            f"[OK] Runtime config built for {family}: "
            f"concepts={list(context['concept_names'])}, "
            f"logic={context['source_info'].get('logic_expression')}"
        )

    manager = get_run_manager()
    project_config = get_project_config()
    for family in ("pipeline", "ltn", "deepproblog"):
        overrides = {"epochs": 1, "learning_rate": 0.003}
        if family == "pipeline":
            overrides["concept_loss_weight"] = 2.0

        result = execute_real_mnlogic_managed_run(
            manager,
            project_config=project_config,
            model_family=family,
            seed=41,
            benchmark_suite="core_eval",
            supervision="full",
            run_name=f"r4_real_mnlogic_smoke_{family}",
            training_overrides=overrides,
            limit_per_split=16,
        )
        if result.record.selection.dataset != "mnlogic":
            _fail(f"Managed run for {family} did not store dataset=mnlogic")
        if not Path(result.checkpoint_path).exists():
            _fail(f"Managed run for {family} did not write a checkpoint")
        if "test_accuracy" not in result.evaluation_metrics:
            _fail(f"Managed run for {family} is missing test_accuracy")
        print(
            f"[OK] Real MNLogic managed run completed for {family}: "
            f"{result.record.run_id} "
            f"(test_accuracy={result.evaluation_metrics['test_accuracy']:.4f})"
        )

    client = TestClient(create_app())

    api_response = client.post(
        "/api/v1/runs/launch/mnlogic",
        json={
            "model_family": "pipeline",
            "seed": 99,
            "benchmark_suite": "rsbench",
            "supervision": "full",
            "run_name": "r4_api_real_mnlogic",
            "training_overrides": {
                "epochs": 1,
                "learning_rate": 0.003,
                "concept_loss_weight": 2.0,
            },
            "limit_per_split": 16,
        },
    )
    _require_ok(api_response, "real MNLogic API launch")
    api_payload = api_response.json()
    if api_payload["run"]["selection"]["dataset"] != "mnlogic":
        _fail("Real MNLogic API launch did not store dataset=mnlogic")
    if not api_payload["dataset_warnings"]:
        _fail("Real MNLogic API launch did not expose dataset warnings")
    print(
        "[OK] Backend API launched a real MNLogic run and returned dataset warnings"
    )

    ui_response = client.post(
        "/ui/launch",
        data={
            "dataset": "mnlogic",
            "model_family": "pipeline",
            "benchmark_suite": "rsbench",
            "supervision": "full",
            "seed": 100,
            "run_name": "r4_ui_real_mnlogic",
            "limit_per_split": 16,
        },
        follow_redirects=False,
    )
    if ui_response.status_code != 303:
        _fail(
            f"Real MNLogic UI launch expected 303 redirect, got "
            f"{ui_response.status_code}: {ui_response.text}"
        )
    redirect_target = ui_response.headers.get("location", "")
    if not redirect_target.startswith("/runs/"):
        _fail(f"Real MNLogic UI launch did not redirect to a run page: {redirect_target}")
    print(f"[OK] Minimal frontend launched a real MNLogic run: {redirect_target}")

    print("[OK] Real MNLogic managed-run check passed.")
    return 0


def _require_ok(response, name: str) -> None:
    if response.status_code != 200:
        _fail(f"{name} failed with status {response.status_code}: {response.text}")


def _fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
