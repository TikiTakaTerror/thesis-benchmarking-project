#!/usr/bin/env python3
"""Run the full project smoke-check suite for the final handoff."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    python_executable = sys.executable

    print("[OK] Starting full project verification")
    print(f"[OK] Python executable: {python_executable}")

    with tempfile.TemporaryDirectory(
        prefix="phase14_mnlogic_demo_",
        dir=PROJECT_ROOT / "results",
    ) as dataset_dir:
        checks = [
            (
                "Environment",
                [python_executable, "scripts/check_environment.py"],
            ),
            (
                "MNLogic Demo Dataset Creation",
                [
                    python_executable,
                    "scripts/create_mnlogic_demo_dataset.py",
                    "--output-dir",
                    dataset_dir,
                ],
            ),
            (
                "MNLogic Dataset Validation",
                [
                    python_executable,
                    "scripts/check_mnlogic_dataset.py",
                    "--dataset-dir",
                    dataset_dir,
                ],
            ),
            (
                "Model Foundations",
                [python_executable, "scripts/check_model_foundations.py"],
            ),
            (
                "Pipeline Family",
                [python_executable, "scripts/check_pipeline_model.py"],
            ),
            (
                "Evaluation Engine",
                [python_executable, "scripts/check_evaluation_engine.py"],
            ),
            (
                "LTN Family",
                [python_executable, "scripts/check_ltn_model.py"],
            ),
            (
                "DeepProbLog Family",
                [python_executable, "scripts/check_deepproblog_model.py"],
            ),
            (
                "Run Management",
                [python_executable, "scripts/check_run_management.py"],
            ),
            (
                "Backend API",
                [python_executable, "scripts/check_backend_api.py"],
            ),
            (
                "Minimal Frontend",
                [python_executable, "scripts/check_minimal_ui.py"],
            ),
            (
                "Comparison Views",
                [python_executable, "scripts/check_benchmark_views.py"],
            ),
            (
                "Ablation Tooling",
                [python_executable, "scripts/check_ablation_tooling.py"],
            ),
            (
                "Benchmark Adapters",
                [python_executable, "scripts/check_benchmark_adapters.py"],
            ),
            (
                "rsbench External Integration",
                [python_executable, "scripts/check_rsbench_external_integration.py"],
            ),
        ]

        for label, command in checks:
            print(f"\n[RUN] {label}")
            print("[CMD] " + " ".join(command))
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    print("\n[OK] Full project verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
