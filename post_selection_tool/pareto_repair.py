from __future__ import annotations

import time
from typing import Any

import pandas as pd

from .direct_dcr_repair_v19 import apply_direct_dcr_repair_v19
from .logging_utils import get_logger
from .timing import record_pipeline_stage


def dcr_signal_schema_card(config: Any, selector: Any) -> dict[str, Any]:
    dcr_schema_card = dict(selector.schema_card)
    theta_enabled = bool((getattr(config, "theta_guidance_report", None) or {}).get("enabled", False))
    theta_all_columns = bool(getattr(config, "theta_col_ps_all_columns", False))
    full_reference_requested = bool(getattr(config, "dcr_signal_full_reference", False))
    dcr_schema_card["theta_guidance_enabled"] = theta_enabled
    dcr_schema_card["dcr_repeats"] = int(getattr(config, "dcr_repeats", 10))
    dcr_schema_card["dcr_cap"] = int(getattr(config, "dcr_cap", 0) or 0)
    dcr_schema_card["dcr_seed"] = int(getattr(config, "dcr_seed", getattr(config, "seed", 20260420)))
    dcr_schema_card["dcr_align_mode"] = "paired_nn_coreset" if int(dcr_schema_card["dcr_cap"]) <= 0 else "independent_thinning"
    if full_reference_requested:
        dcr_schema_card["dcr_signal_column_source"] = "full_reference_override"
        return dcr_schema_card
    signal_columns = list(getattr(selector, "privacy_columns", None) or getattr(selector, "feature_columns", []))
    if signal_columns:
        dcr_schema_card["dcr_signal_column_order"] = signal_columns
        dcr_schema_card["target_in_signal"] = bool(getattr(selector, "target_column", None) in set(signal_columns))
        if theta_all_columns and bool(getattr(config, "theta_default_fidelity_columns", False)):
            source = "theta_default_fidelity_col_ps_all_columns"
        elif theta_all_columns:
            source = "theta_col_ps_all_columns"
        elif theta_enabled:
            source = "theta_col_ps"
        else:
            source = "selector_privacy_columns"
        dcr_schema_card["dcr_signal_column_source"] = source
    return dcr_schema_card


def apply_pareto_post_selection_repairs(
    state: Any,
    pareto_df: pd.DataFrame,
    pareto_records: list[dict[str, Any]],
    pareto_report: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    if state.selector is None:
        raise RuntimeError("selector is required before Pareto post-selection repairs")

    config = state.config
    selector = state.selector
    started = time.perf_counter()
    direct_dcr_v19_report: dict[str, Any] = {
        "enabled": bool(getattr(config, "direct_dcr_repair_v19_enabled", False)),
        "version": "direct_dcr_repair_v19",
        "applied": False,
    }
    if getattr(config, "direct_dcr_repair_v19_enabled", False):
        pareto_df, pareto_records, direct_dcr_v19_report = apply_direct_dcr_repair_v19(
            pool_records=state.pool_records,
            selected_records=pareto_records,
            exact_records=state.global_exact_records,
            surrogate_records=state.surrogate_records_all,
            train_df=state.train_df,
            test_df=state.test_df,
            schema_card=dcr_signal_schema_card(config, selector),
            column_order=selector.column_order,
            target_margin=config.direct_dcr_repair_v19_target_margin,
            max_swap_fraction=config.direct_dcr_repair_v19_max_swap_fraction,
            candidate_neighbors=config.direct_dcr_repair_v19_candidate_neighbors,
            margin_weight=config.direct_dcr_repair_v19_margin_weight,
            utility_weight=config.direct_dcr_repair_v19_utility_weight,
            cat_weight=config.direct_dcr_repair_v19_cat_weight,
            large_keep_k_threshold=config.direct_dcr_repair_v19_large_keep_k_threshold,
            large_pool_rows_threshold=config.direct_dcr_repair_v19_large_pool_rows_threshold,
            large_candidate_rows=config.direct_dcr_repair_v19_large_candidate_rows,
            large_reference_rows=config.direct_dcr_repair_v19_large_reference_rows,
            large_max_swaps=config.direct_dcr_repair_v19_large_max_swaps,
            large_candidate_neighbors=config.direct_dcr_repair_v19_large_candidate_neighbors,
            min_pair_utility_gain=config.direct_dcr_repair_v19_min_pair_utility_gain,
            fallback_min_pair_utility_gain=config.direct_dcr_repair_v19_fallback_min_pair_utility_gain,
            signal_query_batch_size=config.direct_dcr_repair_v19_signal_query_batch_size,
            signal_reference_chunk_size=config.direct_dcr_repair_v19_signal_reference_chunk_size,
            signal_device=config.nn_device,
            report_id_limit=config.direct_dcr_repair_v19_report_id_limit,
            target_bins=config.direct_dcr_repair_v19_target_bins,
            quality_weight=config.direct_dcr_repair_v19_quality_weight,
            target_mismatch_penalty=config.direct_dcr_repair_v19_target_mismatch_penalty,
            generic_remove_budget=config.direct_dcr_repair_v19_generic_remove_budget,
        )
    elapsed = float(time.perf_counter() - started)
    record_pipeline_stage(
        state.timing_report,
        "dcr_repair",
        elapsed,
        dcr_repair_enabled=bool(direct_dcr_v19_report.get("enabled", False)),
        dcr_repair_applied=bool(direct_dcr_v19_report.get("applied", False)),
        dcr_repair_internal_elapsed_seconds=direct_dcr_v19_report.get("elapsed_seconds"),
    )
    get_logger().info(
        "[pipeline_stages] dcr_repair done applied=%s elapsed=%.2fs",
        bool(direct_dcr_v19_report.get("applied", False)),
        elapsed,
    )
    return (
        pareto_df,
        pareto_records,
        {
            **pareto_report,
            "direct_dcr_repair_v19": direct_dcr_v19_report,
        },
    )
