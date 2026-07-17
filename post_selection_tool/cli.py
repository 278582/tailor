from __future__ import annotations

import argparse
from pathlib import Path

from .config import CoreSelectionConfig
from .logging_utils import get_logger
from .pipeline import run_core_selection
from .theta_guidance import resolve_theta_guidance, resolve_theta_synthetic_pool


def _parse_column_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    columns = [item.strip() for item in value.split(",") if item.strip()]
    return columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate five core post-selection tabular selections.")
    parser.add_argument("--synthetic-csv", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, default=CoreSelectionConfig.dataset_name)
    parser.add_argument("--exp-name", type=str, default=CoreSelectionConfig.exp_name)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=CoreSelectionConfig.seed)
    parser.add_argument("--source", type=str, default=CoreSelectionConfig.source)
    parser.add_argument("--keep-k", type=int, default=CoreSelectionConfig.keep_k)
    parser.add_argument("--preselect-target", type=int, default=CoreSelectionConfig.preselect_target)
    parser.add_argument("--d-cur-size", type=int, default=CoreSelectionConfig.d_cur_size)
    parser.add_argument("--d-cur-source", choices=["synthetic", "train"], default=CoreSelectionConfig.d_cur_source)
    parser.add_argument("--holdout-fraction", type=float, default=CoreSelectionConfig.holdout_fraction)
    parser.add_argument("--scalar-fidelity-weight", type=float, default=CoreSelectionConfig.scalar_fidelity_weight)
    parser.add_argument("--scalar-privacy-weight", type=float, default=CoreSelectionConfig.scalar_privacy_weight)
    parser.add_argument("--scalar-utility-weight", type=float, default=CoreSelectionConfig.scalar_utility_weight)
    parser.add_argument("--lambda-penalty", type=float, default=CoreSelectionConfig.lambda_penalty)
    parser.add_argument("--gamma", type=float, default=CoreSelectionConfig.gamma)
    parser.add_argument("--privacy-version", choices=["v1", "v2", "v3"], default=CoreSelectionConfig.privacy_version)
    parser.add_argument("--nn-device", type=str, default=CoreSelectionConfig.nn_device)
    parser.add_argument("--nn-query-batch-size", type=int, default=CoreSelectionConfig.nn_query_batch_size)
    parser.add_argument("--nn-reference-chunk-size", type=int, default=CoreSelectionConfig.nn_reference_chunk_size)
    parser.add_argument("--density-reference-size", type=int, default=CoreSelectionConfig.density_reference_size)
    parser.add_argument("--final-fidelity-floor-eps", type=float, default=CoreSelectionConfig.final_fidelity_floor_eps)
    parser.add_argument("--final-trend-floor-eps", type=float, default=CoreSelectionConfig.final_trend_floor_eps)
    parser.add_argument("--fidelity-ceiling-utility-weight", type=float, default=CoreSelectionConfig.fidelity_ceiling_utility_weight)
    parser.add_argument("--fidelity-ceiling-refine-utility-weight", type=float, default=CoreSelectionConfig.fidelity_ceiling_refine_utility_weight)
    parser.add_argument(
        "--fidelity-ceiling-second-pass-utility-weight",
        type=float,
        default=CoreSelectionConfig.fidelity_ceiling_second_pass_utility_weight,
    )
    parser.add_argument(
        "--fidelity-ceiling-second-pass-refine-utility-weight",
        type=float,
        default=CoreSelectionConfig.fidelity_ceiling_second_pass_refine_utility_weight,
    )
    parser.add_argument("--pareto-floor-mode", choices=["hard", "soft"], default=CoreSelectionConfig.pareto_floor_mode)
    parser.add_argument("--pareto-soft-fidelity-floor-eps", type=float, default=CoreSelectionConfig.pareto_soft_fidelity_floor_eps)
    parser.add_argument("--pareto-soft-trend-floor-eps", type=float, default=CoreSelectionConfig.pareto_soft_trend_floor_eps)
    parser.add_argument("--pareto-soft-privacy-floor-eps", type=float, default=CoreSelectionConfig.pareto_soft_privacy_floor_eps)
    parser.add_argument("--pareto-soft-utility-floor-eps", type=float, default=CoreSelectionConfig.pareto_soft_utility_floor_eps)
    parser.add_argument("--pareto-soft-min-score-delta", type=float, default=CoreSelectionConfig.pareto_soft_min_score_delta)
    parser.add_argument("--enable-reward-candidate-v2", action="store_true")
    parser.add_argument("--disable-reward-candidate-v2", action="store_true")
    parser.add_argument("--disable-reward-candidate-v2-pre-repair", action="store_true")
    parser.add_argument(
        "--reward-candidate-v2-max-swap-fraction",
        type=float,
        default=CoreSelectionConfig.reward_candidate_v2_max_swap_fraction,
    )
    parser.add_argument(
        "--reward-candidate-v2-max-candidate-sizes",
        type=int,
        default=CoreSelectionConfig.reward_candidate_v2_max_candidate_sizes,
    )
    parser.add_argument(
        "--reward-candidate-v2-min-proxy-delta",
        type=float,
        default=CoreSelectionConfig.reward_candidate_v2_min_proxy_delta,
    )
    parser.add_argument(
        "--reward-candidate-v2-fidelity-floor-eps",
        type=float,
        default=CoreSelectionConfig.reward_candidate_v2_fidelity_floor_eps,
    )
    parser.add_argument(
        "--reward-candidate-v2-utility-floor-eps",
        type=float,
        default=CoreSelectionConfig.reward_candidate_v2_utility_floor_eps,
    )
    parser.add_argument("--disable-direct-dcr-repair-v19", action="store_true")
    parser.add_argument(
        "--direct-dcr-repair-v19-target-margin",
        type=float,
        default=CoreSelectionConfig.direct_dcr_repair_v19_target_margin,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-max-swap-fraction",
        type=float,
        default=CoreSelectionConfig.direct_dcr_repair_v19_max_swap_fraction,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-candidate-neighbors",
        type=int,
        default=CoreSelectionConfig.direct_dcr_repair_v19_candidate_neighbors,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-margin-weight",
        type=float,
        default=CoreSelectionConfig.direct_dcr_repair_v19_margin_weight,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-utility-weight",
        type=float,
        default=CoreSelectionConfig.direct_dcr_repair_v19_utility_weight,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-cat-weight",
        type=float,
        default=CoreSelectionConfig.direct_dcr_repair_v19_cat_weight,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-large-keep-k-threshold",
        type=int,
        default=CoreSelectionConfig.direct_dcr_repair_v19_large_keep_k_threshold,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-large-pool-rows-threshold",
        type=int,
        default=CoreSelectionConfig.direct_dcr_repair_v19_large_pool_rows_threshold,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-large-candidate-rows",
        type=int,
        default=CoreSelectionConfig.direct_dcr_repair_v19_large_candidate_rows,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-large-reference-rows",
        type=int,
        default=CoreSelectionConfig.direct_dcr_repair_v19_large_reference_rows,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-large-max-swaps",
        type=int,
        default=CoreSelectionConfig.direct_dcr_repair_v19_large_max_swaps,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-large-candidate-neighbors",
        type=int,
        default=CoreSelectionConfig.direct_dcr_repair_v19_large_candidate_neighbors,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-min-pair-utility-gain",
        type=float,
        default=CoreSelectionConfig.direct_dcr_repair_v19_min_pair_utility_gain,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-fallback-min-pair-utility-gain",
        type=float,
        default=CoreSelectionConfig.direct_dcr_repair_v19_fallback_min_pair_utility_gain,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-signal-query-batch-size",
        type=int,
        default=CoreSelectionConfig.direct_dcr_repair_v19_signal_query_batch_size,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-signal-reference-chunk-size",
        type=int,
        default=CoreSelectionConfig.direct_dcr_repair_v19_signal_reference_chunk_size,
    )
    parser.add_argument(
        "--dcr-signal-full-reference",
        action="store_true",
        help="Use the full schema column order for DCR repair signal even when theta guidance narrows privacy columns.",
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-report-id-limit",
        type=int,
        default=CoreSelectionConfig.direct_dcr_repair_v19_report_id_limit,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-target-bins",
        type=int,
        default=CoreSelectionConfig.direct_dcr_repair_v19_target_bins,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-quality-weight",
        type=float,
        default=CoreSelectionConfig.direct_dcr_repair_v19_quality_weight,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-target-mismatch-penalty",
        type=float,
        default=CoreSelectionConfig.direct_dcr_repair_v19_target_mismatch_penalty,
    )
    parser.add_argument(
        "--direct-dcr-repair-v19-generic-remove-budget",
        type=int,
        default=CoreSelectionConfig.direct_dcr_repair_v19_generic_remove_budget,
    )
    parser.add_argument(
        "--preselect-gate-fidelity-max-drop",
        type=float,
        default=CoreSelectionConfig.preselect_gate_fidelity_max_drop,
    )
    parser.add_argument(
        "--preselect-gate-trend-max-drop",
        type=float,
        default=CoreSelectionConfig.preselect_gate_trend_max_drop,
    )
    parser.add_argument(
        "--preselect-gate-dcr-min-gain",
        type=float,
        default=CoreSelectionConfig.preselect_gate_dcr_min_gain,
    )
    parser.add_argument(
        "--preselect-gate-candidate-vs-baseline-max-drop",
        type=float,
        default=CoreSelectionConfig.preselect_gate_candidate_vs_baseline_max_drop,
    )
    parser.add_argument(
        "--preselect-gate-candidate-vs-baseline-min-dcr-gain",
        type=float,
        default=CoreSelectionConfig.preselect_gate_candidate_vs_baseline_min_dcr_gain,
    )
    parser.add_argument("--enable-preselect-dcr-balance", action="store_true")
    parser.add_argument("--disable-preselect-dcr-balance", action="store_true")
    parser.add_argument(
        "--preselect-dcr-balance-target-fraction",
        type=float,
        default=CoreSelectionConfig.preselect_dcr_balance_target_fraction,
    )
    parser.add_argument(
        "--preselect-dcr-balance-max-exchange-fraction",
        type=float,
        default=CoreSelectionConfig.preselect_dcr_balance_max_exchange_fraction,
    )
    parser.add_argument(
        "--theta-json",
        type=Path,
        default=None,
        help="Path to theta_star.json or rollout theta.json. Enables theta-guided Pareto selection.",
    )
    parser.add_argument(
        "--theta-source",
        choices=["none", "final", "best-rollout", "best-artifact", "auto"],
        default="none",
        help="Theta source under --theta-mcts-dir, or under --theta-artifact-root/--dataset-name/--theta-run-name/mcts_v2 or mcts.",
    )
    parser.add_argument("--theta-mcts-dir", type=Path, default=None)
    parser.add_argument("--theta-artifact-root", type=Path, default=Path("artifacts/llm_mcts_llm"))
    parser.add_argument("--theta-run-name", type=str, default="auto")
    parser.add_argument(
        "--theta-s-pool-manifest",
        type=Path,
        default=None,
        help="Optional LLM-MCTS s_nodes/s_*/synthetic_pool_manifest.json used to rebuild a mixed source pool.",
    )
    parser.add_argument(
        "--theta-s-pool-sample-root",
        type=Path,
        default=CoreSelectionConfig.theta_s_pool_sample_root,
        help="Root containing source/dataset/sample_{sample_id}.csv files for theta-guided mixed pool rebuilding.",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=CoreSelectionConfig.theta_s_pool_sample_id,
        help="Sample id used with --theta-s-pool-sample-root for mixed theta pools, e.g. 1 -> sample_1.csv.",
    )
    parser.add_argument("--fidelity-1d-columns", type=str, default=None)
    parser.add_argument("--fidelity-2d-anchor-columns", type=str, default=None)
    parser.add_argument("--privacy-columns", type=str, default=None)
    parser.add_argument("--utility-balance-column", type=str, default=None)
    parser.add_argument(
        "--utility-source-prior",
        type=str,
        default=CoreSelectionConfig.utility_source_prior,
        help="Optional comma-separated source utility multipliers, for example tabdiff:1.0,tabsyn:0.7,smote:0.5,great:0.5.",
    )
    parser.add_argument(
        "--utility-source-prior-default-weight",
        type=float,
        default=CoreSelectionConfig.utility_source_prior_default_weight,
    )
    parser.add_argument("--max-theta-pairs", type=int, default=None)
    parser.add_argument(
        "--theta-col-ps-all-columns",
        action="store_true",
        help="Replace theta col_ps with all non-target feature columns before theta-guided Pareto selection.",
    )
    parser.add_argument(
        "--theta-default-fidelity-columns",
        action="store_true",
        help="Keep best-theta col_u but use selector default fidelity columns instead of theta col_1ds/col_2ds.",
    )
    parser.add_argument(
        "--theta-default-utility-balance",
        action="store_true",
        help="Keep theta guidance but use the selector default utility balance instead of theta col_u.",
    )
    parser.add_argument("--disable-high-cardinality-compression", action="store_true")
    parser.add_argument("--high-cardinality-threshold", type=int, default=CoreSelectionConfig.high_cardinality_threshold)
    parser.add_argument("--high-cardinality-top-k", type=int, default=CoreSelectionConfig.high_cardinality_top_k)
    parser.add_argument("--high-cardinality-tail-clusters", type=int, default=CoreSelectionConfig.high_cardinality_tail_clusters)
    parser.add_argument("--eval-device", type=str, default=CoreSelectionConfig.eval_device)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--skip-validation-records", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> CoreSelectionConfig:
    theta_source = args.theta_source
    if theta_source == "none" and args.theta_mcts_dir is not None:
        theta_source = "auto"
    theta_guidance = resolve_theta_guidance(
        theta_json=args.theta_json,
        theta_source=theta_source,
        dataset_name=args.dataset_name,
        theta_artifact_root=args.theta_artifact_root,
        theta_run_name=args.theta_run_name,
        theta_mcts_dir=args.theta_mcts_dir,
    )

    fidelity_1d_columns = _parse_column_list(args.fidelity_1d_columns)
    fidelity_2d_anchor_columns = _parse_column_list(args.fidelity_2d_anchor_columns)
    privacy_columns = _parse_column_list(args.privacy_columns)
    explicit_fidelity_1d_columns = fidelity_1d_columns is not None
    explicit_fidelity_2d_anchor_columns = fidelity_2d_anchor_columns is not None
    utility_balance_column = (
        None if args.utility_balance_column is None or not args.utility_balance_column.strip() else args.utility_balance_column.strip()
    )
    theta_guidance_report = {"enabled": False}
    synthetic_csv = args.synthetic_csv
    theta_s_pool_manifest = args.theta_s_pool_manifest
    if theta_guidance is not None:
        theta = theta_guidance.theta
        fidelity_1d_columns = list(theta["col_1ds"]) if fidelity_1d_columns is None else fidelity_1d_columns
        fidelity_2d_anchor_columns = (
            list(theta["col_2ds"]) if fidelity_2d_anchor_columns is None else fidelity_2d_anchor_columns
        )
        privacy_columns = list(theta["col_ps"]) if privacy_columns is None else privacy_columns
        utility_balance_column = theta["col_u"] if utility_balance_column is None else utility_balance_column
        theta_guidance_report = theta_guidance.to_manifest()
        theta_synthetic_csv, theta_synthetic_manifest = resolve_theta_synthetic_pool(theta_guidance)
        if theta_s_pool_manifest is None and theta_synthetic_manifest is not None:
            theta_s_pool_manifest = theta_synthetic_manifest
            theta_guidance_report["theta_s_pool_manifest_auto_applied"] = True
        else:
            theta_guidance_report["theta_s_pool_manifest_auto_applied"] = False
        theta_guidance_report["synthetic_csv_direct_reuse"] = False
        if theta_synthetic_csv is not None:
            theta_guidance_report["llm_mcts_s_pool_csv"] = str(theta_synthetic_csv)
        if theta_synthetic_manifest is not None:
            theta_guidance_report["synthetic_pool_manifest"] = str(theta_synthetic_manifest)
    if args.theta_default_fidelity_columns:
        if not explicit_fidelity_1d_columns:
            fidelity_1d_columns = None
        if not explicit_fidelity_2d_anchor_columns:
            fidelity_2d_anchor_columns = None
    if args.theta_default_utility_balance:
        theta_guidance_report["utility_balance_override"] = {
            "enabled": True,
            "applied": theta_guidance is not None,
            "mode": "selector_default_gate_stratum",
            "original_column": utility_balance_column,
            "replacement_column": None,
        }
        utility_balance_column = None

    if bool(args.disable_reward_candidate_v2):
        reward_candidate_v2_enabled = False
    elif bool(args.enable_reward_candidate_v2):
        reward_candidate_v2_enabled = True
    elif theta_guidance is not None:
        reward_candidate_v2_enabled = True
        theta_guidance_report["reward_candidate_v2_auto_enabled"] = True
    else:
        reward_candidate_v2_enabled = CoreSelectionConfig.reward_candidate_v2_enabled
    if theta_guidance is not None:
        theta_guidance_report["reward_candidate_v2_enabled"] = bool(reward_candidate_v2_enabled)

    return CoreSelectionConfig(
        synthetic_csv=synthetic_csv,
        theta_s_pool_manifest=theta_s_pool_manifest,
        theta_s_pool_sample_root=args.theta_s_pool_sample_root,
        theta_s_pool_sample_id=args.sample_id,
        dataset_name=args.dataset_name,
        exp_name=args.exp_name,
        artifact_dir=args.artifact_dir,
        seed=args.seed,
        source=args.source,
        keep_k=args.keep_k,
        preselect_target=args.preselect_target,
        d_cur_size=args.d_cur_size,
        d_cur_source=args.d_cur_source,
        holdout_fraction=args.holdout_fraction,
        scalar_fidelity_weight=args.scalar_fidelity_weight,
        scalar_privacy_weight=args.scalar_privacy_weight,
        scalar_utility_weight=args.scalar_utility_weight,
        lambda_penalty=args.lambda_penalty,
        gamma=args.gamma,
        privacy_version=args.privacy_version,
        nn_device=args.nn_device,
        nn_query_batch_size=args.nn_query_batch_size,
        nn_reference_chunk_size=args.nn_reference_chunk_size,
        density_reference_size=args.density_reference_size,
        final_fidelity_floor_eps=args.final_fidelity_floor_eps,
        final_trend_floor_eps=args.final_trend_floor_eps,
        fidelity_ceiling_utility_weight=args.fidelity_ceiling_utility_weight,
        fidelity_ceiling_refine_utility_weight=args.fidelity_ceiling_refine_utility_weight,
        fidelity_ceiling_second_pass_utility_weight=args.fidelity_ceiling_second_pass_utility_weight,
        fidelity_ceiling_second_pass_refine_utility_weight=args.fidelity_ceiling_second_pass_refine_utility_weight,
        pareto_floor_mode=args.pareto_floor_mode,
        pareto_soft_fidelity_floor_eps=args.pareto_soft_fidelity_floor_eps,
        pareto_soft_trend_floor_eps=args.pareto_soft_trend_floor_eps,
        pareto_soft_privacy_floor_eps=args.pareto_soft_privacy_floor_eps,
        pareto_soft_utility_floor_eps=args.pareto_soft_utility_floor_eps,
        pareto_soft_min_score_delta=args.pareto_soft_min_score_delta,
        reward_candidate_v2_enabled=reward_candidate_v2_enabled,
        reward_candidate_v2_pre_repair_enabled=not bool(args.disable_reward_candidate_v2_pre_repair),
        reward_candidate_v2_max_swap_fraction=args.reward_candidate_v2_max_swap_fraction,
        reward_candidate_v2_max_candidate_sizes=args.reward_candidate_v2_max_candidate_sizes,
        reward_candidate_v2_min_proxy_delta=args.reward_candidate_v2_min_proxy_delta,
        reward_candidate_v2_fidelity_floor_eps=args.reward_candidate_v2_fidelity_floor_eps,
        reward_candidate_v2_utility_floor_eps=args.reward_candidate_v2_utility_floor_eps,
        direct_dcr_repair_v19_enabled=not bool(args.disable_direct_dcr_repair_v19),
        direct_dcr_repair_v19_target_margin=args.direct_dcr_repair_v19_target_margin,
        direct_dcr_repair_v19_max_swap_fraction=args.direct_dcr_repair_v19_max_swap_fraction,
        direct_dcr_repair_v19_candidate_neighbors=args.direct_dcr_repair_v19_candidate_neighbors,
        direct_dcr_repair_v19_margin_weight=args.direct_dcr_repair_v19_margin_weight,
        direct_dcr_repair_v19_utility_weight=args.direct_dcr_repair_v19_utility_weight,
        direct_dcr_repair_v19_cat_weight=args.direct_dcr_repair_v19_cat_weight,
        direct_dcr_repair_v19_large_keep_k_threshold=args.direct_dcr_repair_v19_large_keep_k_threshold,
        direct_dcr_repair_v19_large_pool_rows_threshold=args.direct_dcr_repair_v19_large_pool_rows_threshold,
        direct_dcr_repair_v19_large_candidate_rows=args.direct_dcr_repair_v19_large_candidate_rows,
        direct_dcr_repair_v19_large_reference_rows=args.direct_dcr_repair_v19_large_reference_rows,
        direct_dcr_repair_v19_large_max_swaps=args.direct_dcr_repair_v19_large_max_swaps,
        direct_dcr_repair_v19_large_candidate_neighbors=args.direct_dcr_repair_v19_large_candidate_neighbors,
        direct_dcr_repair_v19_min_pair_utility_gain=args.direct_dcr_repair_v19_min_pair_utility_gain,
        direct_dcr_repair_v19_fallback_min_pair_utility_gain=args.direct_dcr_repair_v19_fallback_min_pair_utility_gain,
        direct_dcr_repair_v19_signal_query_batch_size=args.direct_dcr_repair_v19_signal_query_batch_size,
        direct_dcr_repair_v19_signal_reference_chunk_size=args.direct_dcr_repair_v19_signal_reference_chunk_size,
        dcr_signal_full_reference=bool(args.dcr_signal_full_reference),
        direct_dcr_repair_v19_report_id_limit=args.direct_dcr_repair_v19_report_id_limit,
        direct_dcr_repair_v19_target_bins=args.direct_dcr_repair_v19_target_bins,
        direct_dcr_repair_v19_quality_weight=args.direct_dcr_repair_v19_quality_weight,
        direct_dcr_repair_v19_target_mismatch_penalty=args.direct_dcr_repair_v19_target_mismatch_penalty,
        direct_dcr_repair_v19_generic_remove_budget=args.direct_dcr_repair_v19_generic_remove_budget,
        preselect_gate_fidelity_max_drop=args.preselect_gate_fidelity_max_drop,
        preselect_gate_trend_max_drop=args.preselect_gate_trend_max_drop,
        preselect_gate_dcr_min_gain=args.preselect_gate_dcr_min_gain,
        preselect_gate_candidate_vs_baseline_max_drop=args.preselect_gate_candidate_vs_baseline_max_drop,
        preselect_gate_candidate_vs_baseline_min_dcr_gain=args.preselect_gate_candidate_vs_baseline_min_dcr_gain,
        preselect_dcr_balance_enabled=(
            bool(args.enable_preselect_dcr_balance)
            if bool(args.enable_preselect_dcr_balance) or bool(args.disable_preselect_dcr_balance)
            else CoreSelectionConfig.preselect_dcr_balance_enabled
        ),
        preselect_dcr_balance_target_fraction=args.preselect_dcr_balance_target_fraction,
        preselect_dcr_balance_max_exchange_fraction=args.preselect_dcr_balance_max_exchange_fraction,
        fidelity_1d_columns=fidelity_1d_columns,
        fidelity_2d_anchor_columns=fidelity_2d_anchor_columns,
        privacy_columns=privacy_columns,
        utility_balance_column=utility_balance_column,
        utility_source_prior=args.utility_source_prior,
        utility_source_prior_default_weight=args.utility_source_prior_default_weight,
        max_theta_pairs=args.max_theta_pairs,
        theta_col_ps_all_columns=bool(args.theta_col_ps_all_columns),
        theta_default_fidelity_columns=bool(args.theta_default_fidelity_columns),
        theta_default_utility_balance=bool(args.theta_default_utility_balance),
        theta_guidance_report=theta_guidance_report,
        allow_target_in_fidelity_columns=CoreSelectionConfig.allow_target_in_fidelity_columns,
        allow_target_in_privacy_columns=CoreSelectionConfig.allow_target_in_privacy_columns,
        high_cardinality_enabled=False if args.disable_high_cardinality_compression else None,
        high_cardinality_threshold=args.high_cardinality_threshold,
        high_cardinality_top_k=args.high_cardinality_top_k,
        high_cardinality_tail_clusters=args.high_cardinality_tail_clusters,
        eval_device=args.eval_device,
        quiet=bool(args.quiet),
        log_file=args.log_file,
        disable_progress=args.disable_progress,
        save_validation_records=not bool(args.skip_validation_records),
    )


def main() -> None:
    config = config_from_args(parse_args())
    state, _ = run_core_selection(config)
    logger = get_logger()
    if config.theta_guidance_report and config.theta_guidance_report.get("enabled"):
        source = config.theta_guidance_report.get("source_kind")
        path = config.theta_guidance_report.get("source_path")
        reward = config.theta_guidance_report.get("reward")
        logger.info("Theta guidance: source=%s path=%s reward=%s", source, path, reward)
    logger.info("Core selections saved to %s", state.paths.versions_dir)


if __name__ == "__main__":
    main()
