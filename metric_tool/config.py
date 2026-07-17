from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from postprocess.paths import DEFAULT_TABDIFF_DATASET_NAME


@dataclass
class MetricConfig:
    dataset_name: str = DEFAULT_TABDIFF_DATASET_NAME
    exp_name: str = "adult_tgm_w1"
    artifact_dir: Path | None = None
    shared_artifact_dir: Path | None = None
    seed: int = 20260420
    holdout_fraction: float = 0.1
    eval_device: str = "auto"
    privacy_version: str = "v2"
    density_reference_size: int = 5000
    nn_device: str = "auto"
    nn_query_batch_size: int = 2048
    nn_reference_chunk_size: int = 8192
    utility_exact_evaluator: str = "tabdiff_mle"
    utility_exact_torch_epochs: int = 6
    utility_exact_torch_importance_sample_size: int = 2000
