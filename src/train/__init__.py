"""Training orchestration and run execution helpers."""
"""Training orchestration helpers."""

from .runner import RunExecutionResult, execute_training_run
from .synthetic import (
    SYNTHETIC_DATASET_NAME,
    build_synthetic_dataset,
    default_synthetic_training_kwargs,
    execute_synthetic_managed_run,
    make_batches,
    split_tensor_batches,
)

__all__ = [
    "RunExecutionResult",
    "SYNTHETIC_DATASET_NAME",
    "build_synthetic_dataset",
    "default_synthetic_training_kwargs",
    "execute_synthetic_managed_run",
    "execute_training_run",
    "make_batches",
    "split_tensor_batches",
]
