#!/usr/bin/env python3
"""Verify the Phase 9 backend API with FastAPI TestClient."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api import create_app


def main() -> int:
    client = TestClient(create_app())

    health_response = client.get("/api/v1/health")
    _require_ok(health_response, "health")
    health_payload = health_response.json()
    if health_payload["status"] != "ok":
        _fail("Health endpoint did not return status=ok")
    print(
        f"[OK] Health endpoint responded for phase {health_payload['phase']}"
    )

    options_response = client.get("/api/v1/options")
    _require_ok(options_response, "options")
    options_payload = options_response.json()
    required_model_families = {"pipeline", "ltn", "deepproblog"}
    if not required_model_families.issubset(set(options_payload["model_families"])):
        _fail("Options endpoint is missing one or more model families")
    if "kand_logic" not in options_payload["datasets"]:
        _fail("Options endpoint is missing kand_logic in the dataset list")
    print(
        f"[OK] Options endpoint returned {len(options_payload['model_families'])} model families"
    )

    launched_run_ids: list[str] = []
    for seed in (11, 17):
        response = client.post(
            "/api/v1/runs/launch/synthetic",
            json={
                "model_family": "pipeline",
                "seed": seed,
                "benchmark_suite": "core_eval",
                "run_name": f"phase9_api_smoke_seed_{seed}",
                "training_overrides": {
                    "epochs": 12,
                    "learning_rate": 0.003,
                    "concept_loss_weight": 6.0,
                },
            },
        )
        _require_ok(response, f"launch synthetic seed {seed}")
        payload = response.json()
        run_id = payload["run"]["run_id"]
        launched_run_ids.append(run_id)

        test_accuracy = payload["evaluation_metrics"].get("test_accuracy", 0.0)
        test_concept_accuracy = payload["evaluation_metrics"].get(
            "test_concept_accuracy", 0.0
        )
        if test_accuracy < 0.95:
            _fail("API launch returned test accuracy below threshold")
        if test_concept_accuracy < 0.90:
            _fail("API launch returned test concept accuracy below threshold")

        print(
            f"[OK] Synthetic launch completed: {run_id} "
            f"(test_accuracy={test_accuracy:.4f})"
        )

    list_response = client.get(
        "/api/v1/runs",
        params={"model_family": "pipeline", "status": "completed", "limit": 20},
    )
    _require_ok(list_response, "list runs")
    listed_run_ids = {item["run_id"] for item in list_response.json()["runs"]}
    if not set(launched_run_ids).issubset(listed_run_ids):
        _fail("List runs endpoint did not return the launched runs")
    print("[OK] Run listing endpoint returned launched runs")

    detail_response = client.get(f"/api/v1/runs/{launched_run_ids[0]}")
    _require_ok(detail_response, "run detail")
    detail_payload = detail_response.json()
    if detail_payload["status"] != "completed":
        _fail("Run detail endpoint did not return completed status")
    print(f"[OK] Run detail endpoint returned {launched_run_ids[0]}")

    snapshot_response = client.get(
        f"/api/v1/runs/{launched_run_ids[0]}/snapshot/metrics"
    )
    _require_ok(snapshot_response, "run metrics snapshot")
    snapshot_payload = snapshot_response.json()
    if "test_accuracy" not in snapshot_payload["payload"]:
        _fail("Run metrics snapshot is missing test_accuracy")
    print("[OK] Metrics snapshot endpoint returned stored metrics")

    compare_response = client.post(
        "/api/v1/runs/compare",
        json={
            "run_ids": launched_run_ids,
            "metric_names": [
                "test_accuracy",
                "test_concept_accuracy",
                "run_runtime_seconds",
            ],
            "output_basename": "phase9_api_smoke_comparison",
        },
    )
    _require_ok(compare_response, "compare runs")
    compare_payload = compare_response.json()
    if len(compare_payload["rows"]) != 2:
        _fail("Compare runs endpoint did not return two rows")
    if not Path(compare_payload["csv_path"]).exists():
        _fail("Compare runs endpoint returned a missing CSV path")
    print(f"[OK] Compare endpoint wrote: {compare_payload['csv_path']}")

    print("[OK] Backend API smoke check passed.")
    return 0


def _require_ok(response, name: str) -> None:
    if response.status_code != 200:
        _fail(f"{name} endpoint failed with status {response.status_code}: {response.text}")


def _fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
