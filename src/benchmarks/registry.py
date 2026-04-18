"""Benchmark suite config helpers and adapter registry."""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import BenchmarkSuiteAdapter, BenchmarkSuiteConfig
from .core_eval import CoreEvalBenchmarkAdapter
from .rsbench import RSBenchBenchmarkAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_DIR = PROJECT_ROOT / "src" / "configs" / "benchmarks"

BENCHMARK_ADAPTERS: dict[str, type[BenchmarkSuiteAdapter]] = {
    "core_eval": CoreEvalBenchmarkAdapter,
    "rsbench": RSBenchBenchmarkAdapter,
}


def load_benchmark_config(suite_name: str) -> dict:
    """Load the YAML config for a benchmark suite."""

    canonical_name = suite_name.strip().lower()
    config_path = BENCHMARK_CONFIG_DIR / f"{canonical_name}.yaml"
    if not config_path.exists():
        raise ValueError(f"Unknown benchmark suite config: {suite_name}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid benchmark config file: {config_path}")

    return payload


def get_benchmark_adapter_class(suite_name: str) -> type[BenchmarkSuiteAdapter]:
    """Resolve the adapter class for a benchmark suite."""

    canonical_name = suite_name.strip().lower()
    if canonical_name not in BENCHMARK_ADAPTERS:
        raise ValueError(f"Unsupported benchmark suite: {suite_name}")
    return BENCHMARK_ADAPTERS[canonical_name]


def create_benchmark_adapter(suite_name: str) -> BenchmarkSuiteAdapter:
    """Instantiate the benchmark suite adapter from its config."""

    adapter_cls = get_benchmark_adapter_class(suite_name)
    config = BenchmarkSuiteConfig.from_dict(load_benchmark_config(suite_name))
    return adapter_cls(config)
