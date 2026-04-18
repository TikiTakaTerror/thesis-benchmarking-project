"""Training orchestration and run execution helpers."""
"""Training orchestration helpers."""

from .runner import RunExecutionResult, execute_training_run
from .real_data import (
    REAL_MNLOGIC_DATASET_NAME,
    build_mnlogic_runtime_context,
    build_real_evaluation_splits,
    default_real_training_kwargs,
    execute_real_mnlogic_managed_run,
)
from .supervision import (
    AppliedSupervision,
    SupervisionConfig,
    apply_supervision,
    load_supervision_config,
)
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
    "AppliedSupervision",
    "REAL_MNLOGIC_DATASET_NAME",
    "SYNTHETIC_DATASET_NAME",
    "SupervisionConfig",
    "apply_supervision",
    "build_synthetic_dataset",
    "build_mnlogic_runtime_context",
    "build_real_evaluation_splits",
    "default_real_training_kwargs",
    "default_synthetic_training_kwargs",
    "execute_real_mnlogic_managed_run",
    "execute_synthetic_managed_run",
    "execute_training_run",
    "load_supervision_config",
    "make_batches",
    "split_tensor_batches",
]
