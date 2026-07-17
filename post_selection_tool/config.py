from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TABDIFF_DATASET_NAME = "adult_tgm_w1"


@dataclass
class CoreSelectionConfig:
    synthetic_csv: Path | None = None
    theta_s_pool_manifest: Path | None = None
    theta_s_pool_sample_root: Path = Path("third_party/sample")
    theta_s_pool_sample_id: int = 0
    dataset_name: str = DEFAULT_TABDIFF_DATASET_NAME
    exp_name: str = "adult_tgm_w1"
    artifact_dir: Path | None = None
    shared_artifact_dir: Path | None = None
    seed: int = 20260420
    source: str = "tabdiff"

    keep_k: int = 50
    preselect_target: int = 50
    d_cur_size: int = 200
    d_cur_source: str = "synthetic"
    holdout_fraction: float = 0.1

    scalar_fidelity_weight: float = 0.5
    scalar_privacy_weight: float = 0.3
    scalar_utility_weight: float = 0.2

    lambda_penalty: float = 1.0
    gamma: float = 0.5
    privacy_version: str = "v2"
    nn_device: str = "auto"
    nn_query_batch_size: int = 2048
    nn_reference_chunk_size: int = 8192
    density_reference_size: int = 5000

    final_fidelity_floor_eps: float = 0.01
    final_trend_floor_eps: float = 0.01
    fidelity_ceiling_utility_weight: float = 0.04
    fidelity_ceiling_refine_utility_weight: float = 0.15
    fidelity_ceiling_second_pass_utility_weight: float = 0.08
    fidelity_ceiling_second_pass_refine_utility_weight: float = 0.20

    pareto_floor_mode: str = "soft"
    pareto_soft_fidelity_floor_eps: float = 0.02
    pareto_soft_trend_floor_eps: float = 0.02
    pareto_soft_privacy_floor_eps: float = 0.005
    pareto_soft_utility_floor_eps: float = 0.005
    pareto_soft_min_score_delta: float = 0.0

    reward_candidate_v2_enabled: bool = False
    reward_candidate_v2_pre_repair_enabled: bool = True
    reward_candidate_v2_max_swap_fraction: float = 0.16
    reward_candidate_v2_max_candidate_sizes: int = 10
    reward_candidate_v2_min_proxy_delta: float = 0.0
    reward_candidate_v2_fidelity_floor_eps: float = 0.015
    reward_candidate_v2_utility_floor_eps: float = 0.02

    direct_dcr_repair_v19_enabled: bool = True
    direct_dcr_repair_v19_target_margin: float = 0.03
    direct_dcr_repair_v19_max_swap_fraction: float = 0.30
    direct_dcr_repair_v19_candidate_neighbors: int = 64
    direct_dcr_repair_v19_margin_weight: float = 0.10
    direct_dcr_repair_v19_utility_weight: float = 0.65
    direct_dcr_repair_v19_cat_weight: float = 1.0
    direct_dcr_repair_v19_large_keep_k_threshold: int = 50_000
    direct_dcr_repair_v19_large_pool_rows_threshold: int = 180_000
    direct_dcr_repair_v19_large_candidate_rows: int = 72_000
    direct_dcr_repair_v19_large_reference_rows: int = 0
    direct_dcr_repair_v19_large_max_swaps: int = 20_000
    direct_dcr_repair_v19_large_candidate_neighbors: int = 28
    direct_dcr_repair_v19_min_pair_utility_gain: float = -0.08
    direct_dcr_repair_v19_fallback_min_pair_utility_gain: float = -0.18
    direct_dcr_repair_v19_signal_query_batch_size: int = 0
    direct_dcr_repair_v19_signal_reference_chunk_size: int = 65536
    dcr_signal_full_reference: bool = False
    direct_dcr_repair_v19_report_id_limit: int = 64
    direct_dcr_repair_v19_target_bins: int = 12
    direct_dcr_repair_v19_quality_weight: float = 0.20
    direct_dcr_repair_v19_target_mismatch_penalty: float = 4.0
    direct_dcr_repair_v19_generic_remove_budget: int = 20_000
    preselect_gate_fidelity_max_drop: float = 0.01
    preselect_gate_trend_max_drop: float = 0.01
    preselect_gate_dcr_min_gain: float = 0.02
    preselect_gate_candidate_vs_baseline_max_drop: float = 0.001
    preselect_gate_candidate_vs_baseline_min_dcr_gain: float = 0.002
    preselect_dcr_balance_enabled: bool = False
    preselect_dcr_balance_target_fraction: float = 0.50
    preselect_dcr_balance_max_exchange_fraction: float = 0.30

    fidelity_1d_columns: list[str] | None = None
    fidelity_2d_anchor_columns: list[str] | None = None
    privacy_columns: list[str] | None = None
    utility_balance_column: str | None = None
    utility_source_prior: str | None = None
    utility_source_prior_default_weight: float = 1.0
    allow_target_in_fidelity_columns: bool = False
    allow_target_in_privacy_columns: bool = False
    privacy_encoding_column_mode: str = "privacy_columns"
    max_theta_pairs: int | None = None
    theta_col_ps_all_columns: bool = False
    theta_default_fidelity_columns: bool = False
    theta_default_utility_balance: bool = False
    theta_guidance_report: dict[str, Any] | None = None

    high_cardinality_enabled: bool | None = None
    high_cardinality_threshold: int = 256
    high_cardinality_top_k: int = 64
    high_cardinality_tail_clusters: int = 16

    eval_device: str = "auto"
    quiet: bool = False
    log_file: Path | None = None
    disable_progress: bool = False
    save_validation_records: bool = True


def progress_enabled(config: CoreSelectionConfig) -> bool:
    return not bool(config.disable_progress or config.quiet)
