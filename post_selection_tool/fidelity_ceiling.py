from __future__ import annotations

import time

from .config import progress_enabled
from .io import records_to_df
from .logging_utils import get_logger
from .state import SelectionState
from .utility_proxy import (
    apply_utility_source_prior_to_proxy_scores,
    attach_utility_proxy_fields,
    build_static_balanced_utility_scores,
)


def build_fidelity_ceiling(state: SelectionState) -> SelectionState:
    if state.selector is None:
        raise RuntimeError("selector is required before build_fidelity_ceiling")
    if not state.preselected_valid or not state.global_exact_records:
        raise RuntimeError("preselected_valid and global_exact_records are required before build_fidelity_ceiling")

    config = state.config
    selector = state.selector
    selection_records = state.preselected_valid
    progress = progress_enabled(config)
    logger = get_logger()
    logger.info(
        "[objective_score] utility start rows=%d objective=utility_proxy",
        len(selection_records),
    )
    utility_started = time.perf_counter()
    pre_ceiling_static_utility_bundle = build_static_balanced_utility_scores(
        selector=selector,
        preselected_records=selection_records,
        random_state=config.seed,
        show_progress=progress,
    )
    utility_elapsed = float(time.perf_counter() - utility_started)
    objective_timing = state.timing_report.setdefault("objective_scoring", {})
    objective_timing.update(
        {
            "schema_version": 1,
            "candidate_rows": int(len(selection_records)),
            "objectives": [
                "pareto_fid_1d_obj",
                "pareto_fid_2d_obj",
                "pareto_priv_obj",
                "pareto_util_proxy_obj",
            ],
            "utility_proxy_seconds": utility_elapsed,
            "utility_proxy_recorded": True,
            "timing_basis": "wall_clock_perf_counter",
            "shared_by": ["scalar", "pareto"],
        }
    )
    logger.info(
        "[objective_score] utility done rows=%d elapsed=%.2fs objective=utility_proxy",
        len(pre_ceiling_static_utility_bundle.get("static_scores", [])),
        utility_elapsed,
    )
    (
        state.initial_fidelity_ceiling_df,
        state.initial_fidelity_ceiling_records,
        state.initial_fidelity_ceiling_report,
    ) = selector.construct_fidelity_ceiling_subset(
        preselected_records=selection_records,
        exact_records=state.global_exact_records,
        keep_k=state.effective_keep_k,
        utility_scores_by_id=pre_ceiling_static_utility_bundle["score_by_id"],
        utility_weight=config.fidelity_ceiling_utility_weight,
        refine_utility_weight=config.fidelity_ceiling_refine_utility_weight,
        utility_score_name="utility_static_balanced",
        show_progress=progress,
        progress_desc="initial fidelity ceiling",
    )
    utility_postprocess_started = time.perf_counter()
    static_scores = pre_ceiling_static_utility_bundle["static_scores"]
    anchor_ids = {int(anchor.get("candidate_id", -1)) for anchor in state.initial_fidelity_ceiling_records}
    proxy_scores = [
        {
            "candidate_id": int(record["candidate_id"]),
            "u_static": float(record.get("u_static", 0.0)),
            "u_static_raw": float(record.get("u_static_raw", record.get("u_static", 0.0))),
            "target_label": record.get("target_label"),
            "gate_stratum": int(record.get("gate_stratum", -1)),
            "balance_bucket": record.get("balance_bucket"),
            "u_static_group_rank": float(record.get("u_static_group_rank", 0.0)),
            "density_weight": float(record.get("density_weight", 0.0)),
            "coverage_gain": float(record.get("coverage_gain", 0.0)),
            "u_static_balanced": float(record.get("u_static_balanced", 0.0)),
            "u_static_norm": float(record.get("u_static_balanced", 0.0)),
            "u_proxy": float(record.get("u_static_balanced", 0.0)),
            "is_anchor_member": int(record["candidate_id"]) in anchor_ids,
            "task_type": record.get("task_type"),
        }
        for record in static_scores
    ]
    proxy_scores, source_prior_report = apply_utility_source_prior_to_proxy_scores(
        proxy_scores,
        source_by_id=state.candidate_source_by_id,
        prior=config.utility_source_prior,
        default_weight=config.utility_source_prior_default_weight,
    )
    state.utility_proxy_bundle = {
        "static_scores": static_scores,
        "proxy_scores": proxy_scores,
        "manifest": {
            **pre_ceiling_static_utility_bundle.get("manifest", {}),
            "mode": "static",
            "proxy_formula": "u_static_balanced",
            "source_prior": source_prior_report,
            "theta_s_pool": state.theta_s_pool_report,
            "dynamic_utility": {"available": False, "reason": "removed_from_main_chain"},
            "final_test_used": False,
        },
        "teacher_manifest": pre_ceiling_static_utility_bundle.get("teacher_manifest", {}),
    }
    state.global_exact_records, state.utility_proxy_merge_report = attach_utility_proxy_fields(
        state.global_exact_records,
        state.utility_proxy_bundle["proxy_scores"],
    )
    utility_postprocess_elapsed = float(time.perf_counter() - utility_postprocess_started)
    exact_elapsed = objective_timing.get("exact_fidelity_privacy_seconds")
    exact_recorded = bool(objective_timing.get("exact_fidelity_privacy_recorded", False))
    four_objective_total = float(exact_elapsed or 0.0) + utility_elapsed + utility_postprocess_elapsed
    objective_timing.update(
        {
            "utility_postprocess_seconds": utility_postprocess_elapsed,
            "utility_postprocess_recorded": True,
            "four_objective_total_seconds": four_objective_total,
            "complete": exact_recorded,
            "excludes": [
                "preselection",
                "fidelity_ceiling_subset_construction",
                "random_selection",
                "scalar_selection",
                "pareto_selection",
                "pareto_post_repair",
            ],
        }
    )
    logger.info(
        "[objective_score] utility postprocess done rows=%d elapsed=%.2fs",
        len(state.global_exact_records),
        utility_postprocess_elapsed,
    )
    logger.info(
        "[objective_score] four-objective summary rows=%d exact=%.2fs utility=%.2fs "
        "postprocess=%.2fs total=%.2fs complete=%s",
        len(state.global_exact_records),
        float(exact_elapsed or 0.0),
        utility_elapsed,
        utility_postprocess_elapsed,
        four_objective_total,
        exact_recorded,
    )
    state.fidelity_ceiling_df = state.initial_fidelity_ceiling_df
    state.fidelity_ceiling_records = state.initial_fidelity_ceiling_records
    state.fidelity_ceiling_report = state.initial_fidelity_ceiling_report
    state.fidelity_ceiling_report["second_pass"] = {
        "applied": False,
        "reason": "dynamic_utility_removed_from_main_chain",
        "utility_source": "static_balanced_only",
    }

    if state.fidelity_ceiling_df is None:
        state.fidelity_ceiling_df = records_to_df(state.fidelity_ceiling_records, selector.column_order)
    state.floor_reference = state.fidelity_ceiling_report.get(
        "reference",
        {
            "name": "preselected_fidelity_ceiling_keep_k",
            "rows": len(state.fidelity_ceiling_records),
            "fidelity_1d": selector.compute_dataset_fidelity(state.fidelity_ceiling_df),
            "fidelity_2d": selector.compute_dataset_pair_fidelity(state.fidelity_ceiling_df),
            "privacy_mean": selector.compute_dataset_privacy(state.fidelity_ceiling_df),
        },
    )
    state.utility_proxy_bundle["pre_ceiling_static"] = pre_ceiling_static_utility_bundle
    return state
