from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CoreSelectionConfig


@dataclass
class ArtifactPaths:
    artifact_dir: Path
    input_dir: Path
    cards_dir: Path
    validation_dir: Path
    selection_dir: Path
    versions_dir: Path
    report_dir: Path


@dataclass
class CoreSelectionOutputs:
    preselected_valid_df: pd.DataFrame
    fidelity_ceiling_df: pd.DataFrame
    random_full_df: pd.DataFrame
    scalar_df: pd.DataFrame
    pareto_df: pd.DataFrame
    preselected_valid_records: list[dict[str, Any]] = field(default_factory=list)
    fidelity_ceiling_records: list[dict[str, Any]] = field(default_factory=list)
    random_full_records: list[dict[str, Any]] = field(default_factory=list)
    scalar_records: list[dict[str, Any]] = field(default_factory=list)
    pareto_records: list[dict[str, Any]] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionState:
    config: CoreSelectionConfig
    paths: ArtifactPaths
    dataset_ctx: Any
    synthetic_csv: Path

    train_df: pd.DataFrame
    holdout_df: pd.DataFrame
    test_df: pd.DataFrame
    synthetic_df: pd.DataFrame
    candidate_source_by_id: dict[int, str] = field(default_factory=dict)
    theta_s_pool_report: dict[str, Any] = field(default_factory=dict)

    cards: Any | None = None
    selector: Any | None = None
    evaluator: Any | None = None

    valid_df: pd.DataFrame | None = None
    valid_records: list[dict[str, Any]] = field(default_factory=list)
    rejected_records: list[dict[str, Any]] = field(default_factory=list)
    validation_report: dict[str, Any] = field(default_factory=dict)

    pool_df: pd.DataFrame | None = None
    pool_records: list[dict[str, Any]] = field(default_factory=list)
    d_cur_df: pd.DataFrame | None = None

    desired_keep_k: int = 0
    requested_preselect_target: int = 0
    effective_keep_k: int = 0
    effective_preselect_target: int = 0

    surrogate_records_all: list[dict[str, Any]] = field(default_factory=list)
    preselected_valid: list[dict[str, Any]] = field(default_factory=list)
    preselected_surrogates: list[dict[str, Any]] = field(default_factory=list)
    preselect_gate: dict[str, Any] = field(default_factory=dict)
    preselect_status: dict[str, Any] = field(default_factory=dict)

    global_exact_records: list[dict[str, Any]] = field(default_factory=list)
    global_baselines: dict[str, Any] = field(default_factory=dict)
    timing_report: dict[str, Any] = field(default_factory=dict)

    utility_proxy_bundle: dict[str, Any] = field(default_factory=dict)
    utility_proxy_merge_report: dict[str, Any] = field(default_factory=dict)
    fidelity_ceiling_df: pd.DataFrame | None = None
    fidelity_ceiling_records: list[dict[str, Any]] = field(default_factory=list)
    fidelity_ceiling_report: dict[str, Any] = field(default_factory=dict)
    initial_fidelity_ceiling_df: pd.DataFrame | None = None
    initial_fidelity_ceiling_records: list[dict[str, Any]] = field(default_factory=list)
    initial_fidelity_ceiling_report: dict[str, Any] = field(default_factory=dict)
    floor_reference: dict[str, Any] = field(default_factory=dict)
