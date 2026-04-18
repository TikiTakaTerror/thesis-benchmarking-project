#!/usr/bin/env python3
"""Verify the real Kand-Logic managed run path introduced in R9."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api import create_app
from src.train import build_kand_logic_runtime_context, execute_real_kand_logic_managed_run
from src.services.runtime import get_project_config, get_run_manager


EXPECTED_CONCEPT_COUNT = 54


def main() -> int:
    for family in ("pipeline", "ltn", "deepproblog"):
        context = build_kand_logic_runtime_context(model_family=family)
        if len(context["concept_names"]) != EXPECTED_CONCEPT_COUNT:
            _fail(
                f"Runtime concept names for {family} do not match Kand-Logic "
                f"({len(context['concept_names'])} != {EXPECTED_CONCEPT_COUNT})"
            )
        print(
            f"[OK] Runtime config built for {family}: "
            f"concepts={len(context['concept_names'])}, "
            f"logic={context['source_info'].get('logic_expression')}, "
            f"aggregator={context['source_info'].get('aggregator_logic_expression')}"
        )

    manager = get_run_manager()
    project_config = get_project_config()
    family_overrides = {
        "pipeline": {"epochs": 1, "learning_rate": 0.002, "concept_loss_weight": 2.0},
        "ltn": {"epochs": 1, "learning_rate": 0.002, "concept_loss_weight": 1.0},
        "deepproblog": {"epochs": 1, "learning_rate": 0.001, "concept_loss_weight": 0.0},
    }
    family_limits = {"pipeline": 4, "ltn": 4, "deepproblog": 1}
    family_supervision = {"pipeline": "full", "ltn": "full", "deepproblog": "label_only"}
    for family in ("pipeline", "ltn", "deepproblog"):
        result = execute_real_kand_logic_managed_run(
            manager,
            project_config=project_config,
            model_family=family,
            seed=43,
            benchmark_suite="core_eval",
            supervision=family_supervision[family],
            run_name=f"r9_real_kand_logic_smoke_{family}",
            training_overrides=family_overrides[family],
            limit_per_split=family_limits[family],
        )
        if result.record.selection.dataset != "kand_logic":
            _fail(f"Managed run for {family} did not store dataset=kand_logic")
        if not Path(result.checkpoint_path).exists():
            _fail(f"Managed run for {family} did not write a checkpoint")
        if "test_accuracy" not in result.evaluation_metrics:
            _fail(f"Managed run for {family} is missing test_accuracy")
        print(
            f"[OK] Real Kand-Logic managed run completed for {family}: "
            f"{result.record.run_id} "
            f"(test_accuracy={result.evaluation_metrics['test_accuracy']:.4f})"
        )

    client = TestClient(create_app())

    api_response = client.post(
        "/api/v1/runs/launch/kand_logic",
        json={
            "model_family": "pipeline",
            "seed": 98,
            "benchmark_suite": "rsbench",
            "supervision": "full",
            "run_name": "r9_api_real_kand_logic",
            "training_overrides": {
                "epochs": 1,
                "learning_rate": 0.002,
                "concept_loss_weight": 2.0,
            },
            "limit_per_split": 4,
        },
    )
    _require_ok(api_response, "real Kand-Logic API launch")
    api_payload = api_response.json()
    if api_payload["run"]["selection"]["dataset"] != "kand_logic":
        _fail("Real Kand-Logic API launch did not store dataset=kand_logic")
    print("[OK] Backend API launched a real Kand-Logic run")

    ui_response = client.post(
        "/ui/launch",
        data={
            "dataset": "kand_logic",
            "model_family": "pipeline",
            "benchmark_suite": "rsbench",
            "supervision": "full",
            "seed": 101,
            "run_name": "r9_ui_real_kand_logic",
            "limit_per_split": 4,
        },
        follow_redirects=False,
    )
    if ui_response.status_code != 303:
        _fail(
            f"Real Kand-Logic UI launch expected 303 redirect, got "
            f"{ui_response.status_code}: {ui_response.text}"
        )
    redirect_target = ui_response.headers.get("location", "")
    if not redirect_target.startswith("/runs/"):
        _fail(
            f"Real Kand-Logic UI launch did not redirect to a run page: {redirect_target}"
        )
    print(f"[OK] Minimal frontend launched a real Kand-Logic run: {redirect_target}")

    print("[OK] Real Kand-Logic managed-run check passed.")
    return 0


def _require_ok(response, name: str) -> None:
    if response.status_code != 200:
        _fail(f"{name} failed with status {response.status_code}: {response.text}")


def _fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
