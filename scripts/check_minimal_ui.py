#!/usr/bin/env python3
"""Verify the Phase 10 minimal frontend with FastAPI TestClient."""

from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.api import create_app


def main() -> int:
    client = TestClient(create_app())

    dashboard_response = client.get("/")
    _require_ok(dashboard_response, "dashboard")
    dashboard_html = dashboard_response.text
    if "Experiment Control" not in dashboard_html:
        _fail("Dashboard page is missing the main heading")
    if "Launch Train + Evaluate" not in dashboard_html:
        _fail("Dashboard page is missing the launch button")
    if "kand_logic" not in dashboard_html:
        _fail("Dashboard page did not expose kand_logic as a launchable dataset")
    print("[OK] Dashboard page rendered")

    styles_response = client.get("/static/styles.css")
    _require_ok(styles_response, "static stylesheet")
    if "--accent" not in styles_response.text:
        _fail("Stylesheet content looks incomplete")
    print("[OK] Static stylesheet served")

    launch_response = client.post(
        "/ui/launch",
        data={
            "dataset": "synthetic_mnlogic",
            "model_family": "pipeline",
            "benchmark_suite": "core_eval",
            "supervision": "full",
            "seed": 23,
            "run_name": "phase10_ui_smoke_seed_23",
        },
        follow_redirects=False,
    )
    if launch_response.status_code != 303:
        _fail(
            f"UI launch expected 303 redirect, got {launch_response.status_code}: {launch_response.text}"
        )
    redirect_target = launch_response.headers.get("location", "")
    if not redirect_target.startswith("/runs/"):
        _fail(f"UI launch did not redirect to a run detail page: {redirect_target}")
    print(f"[OK] Launch form redirected to {redirect_target}")

    detail_response = client.get(redirect_target)
    _require_ok(detail_response, "run detail page")
    detail_html = detail_response.text
    if "Stored Run" not in detail_html:
        _fail("Run detail page is missing the stored-run heading")
    if "All Metrics" not in detail_html:
        _fail("Run detail page is missing the metrics section")
    if "test_accuracy" not in detail_html:
        _fail("Run detail page did not render stored test metrics")

    match = re.search(r"/runs/([A-Za-z0-9T_\-]+)", redirect_target)
    if not match:
        _fail("Failed to parse run_id from redirect target")
    run_id = match.group(1)
    print(f"[OK] Run detail page rendered for {run_id}")

    refreshed_dashboard = client.get("/")
    _require_ok(refreshed_dashboard, "refreshed dashboard")
    if "phase10_ui_smoke_seed_23" not in refreshed_dashboard.text:
        _fail("Dashboard did not list the newly launched UI run")
    print("[OK] Dashboard listed the launched run")

    print("[OK] Minimal frontend smoke check passed.")
    return 0


def _require_ok(response, name: str) -> None:
    if response.status_code != 200:
        _fail(f"{name} failed with status {response.status_code}: {response.text}")


def _fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
