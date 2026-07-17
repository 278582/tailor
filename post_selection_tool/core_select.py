from __future__ import annotations

import time
from typing import Any

from .io import records_to_df
from .logging_utils import get_logger
from .pareto_repair import apply_pareto_post_selection_repairs
from .reward_candidate_v2 import refine_selection_for_reward_v2
from .state import CoreSelectionOutputs, SelectionState


def _log(message: str) -> None:
    get_logger().info("[core_select] %s", message)


def _candidate_id(record: dict[str, Any], fallback: int) -> int:
    try:
        return int(record.get("candidate_id", fallback))
    except (TypeError, ValueError):
        return int(fallback)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number != number or number in {float("inf"), float("-inf")}:
        return float(default)
    return float(number)


def _records_by_candidate_id(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        _candidate_id(record, idx): record
        for idx, record in enumerate(records)
    }


def _reward_proxy_record(
    *,
    candidate_id: int,
    candidate_index: int,
    exact_by_id: dict[int, dict[str, Any]],
    surrogate_by_id: dict[int, dict[str, Any]],
    utility_by_id: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    exact_record = exact_by_id.get(candidate_id)
    if exact_record is not None:
        proxy_record = dict(exact_record)
        proxy_record["candidate_id"] = int(candidate_index)
        proxy_record["candidate_index"] = int(candidate_index)
        return proxy_record, "exact"

    surrogate = surrogate_by_id.get(candidate_id, {})
    utility = utility_by_id.get(candidate_id, {})
    return {
        "candidate_id": int(candidate_index),
        "candidate_index": int(candidate_index),
        "pareto_fid_1d_obj": _finite_float(
            surrogate.get("s_pareto_fid_1d_sur", surrogate.get("s_fid_sur_1d_rank")),
            0.5,
        ),
        "pareto_fid_2d_obj": _finite_float(
            surrogate.get("s_pareto_fid_2d_sur", surrogate.get("s_fid_sur_2d_rank")),
            0.5,
        ),
        "pareto_priv_obj": _finite_float(
            surrogate.get(
                "s_pareto_priv_sur",
                surrogate.get("s_priv_sur_selected", surrogate.get("density_normalized_nn_distance")),
            ),
            0.5,
        ),
        "pareto_util_proxy_obj": _finite_float(
            utility.get("u_static_balanced", utility.get("u_proxy", utility.get("u_static_norm"))),
            0.5,
        ),
        "holdout_gap": _finite_float(surrogate.get("holdout_gap"), 0.0),
    }, "surrogate"


def _apply_post_dcr_signal_override(
    state: SelectionState,
    candidate_records: list[dict[str, Any]],
    proxy_records: list[dict[str, Any]],
    selected_rows: int,
) -> dict[str, Any]:
    if state.selector is None:
        return {"applied": False, "reason": "missing_selector"}
    if not candidate_records or not proxy_records:
        return {"applied": False, "reason": "empty_records"}
    try:
        from .direct_dcr_repair_v10 import _row_dcr_signal_batched_l1
        from .pareto_repair import dcr_signal_schema_card

        config = state.config
        selector = state.selector
        column_order = selector.column_order
        candidate_df = records_to_df(candidate_records, column_order)
        signal = _row_dcr_signal_batched_l1(
            pool_df=candidate_df[column_order],
            train_df=state.train_df[column_order],
            test_df=state.test_df[column_order],
            schema_card=dcr_signal_schema_card(config, selector),
            column_order=column_order,
            cat_weight=config.direct_dcr_repair_v19_cat_weight,
            nn_device=config.nn_device,
            query_batch_size=config.direct_dcr_repair_v19_signal_query_batch_size,
            reference_chunk_size=config.direct_dcr_repair_v19_signal_reference_chunk_size,
        )
    except Exception as exc:
        return {"applied": False, "reason": "signal_override_failed", "error": str(exc)}

    margins = [float(value) for value in signal["margin"]]
    real_flags = [bool(value) for value in signal["is_real_closer"]]
    for proxy_record, margin, is_real_closer in zip(proxy_records, margins, real_flags):
        proxy_record["holdout_gap"] = float(margin)
        proxy_record["direct_dcr_real_closer"] = bool(is_real_closer)

    selected_flags = real_flags[: max(0, min(int(selected_rows), len(real_flags)))]
    selected_dcr = float(sum(1 for flag in selected_flags if flag) / len(selected_flags)) if selected_flags else 0.5
    return {
        "applied": True,
        "source": "direct_dcr_repair_signal",
        "selected_dcr_proxy": selected_dcr,
        "candidate_rows": int(len(candidate_records)),
        "selected_rows": int(len(selected_flags)),
        "signal_backend": signal.get("signal_backend"),
        "signal_column_source": signal.get("signal_column_source"),
        "signal_column_count": int(signal.get("signal_column_count") or 0),
    }


def _post_dcr_reward_refine_inputs(
    state: SelectionState,
    selected_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    exact_by_id = _records_by_candidate_id(state.global_exact_records)
    surrogate_by_id = _records_by_candidate_id(state.surrogate_records_all)
    utility_by_id = _records_by_candidate_id(state.utility_proxy_bundle.get("proxy_scores", []))
    candidate_records: list[dict[str, Any]] = []
    proxy_records: list[dict[str, Any]] = []
    selected_refine_records: list[dict[str, Any]] = []
    selected_source_ids: set[int] = set()
    source_counts = {"exact": 0, "surrogate": 0}

    def append_record(record: dict[str, Any], source_candidate_id: int) -> dict[str, Any]:
        refine_id = len(candidate_records)
        refine_record = dict(record)
        refine_record["candidate_id"] = int(refine_id)
        refine_record["_reward_refine_source_candidate_id"] = int(source_candidate_id)
        proxy_record, proxy_source = _reward_proxy_record(
            candidate_id=int(source_candidate_id),
            candidate_index=int(refine_id),
            exact_by_id=exact_by_id,
            surrogate_by_id=surrogate_by_id,
            utility_by_id=utility_by_id,
        )
        candidate_records.append(refine_record)
        proxy_records.append(proxy_record)
        source_counts[proxy_source] += 1
        return refine_record

    for idx, record in enumerate(selected_records):
        source_candidate_id = _candidate_id(record, idx)
        selected_source_ids.add(source_candidate_id)
        selected_refine_records.append(append_record(record, source_candidate_id))

    for idx, record in enumerate(state.pool_records):
        source_candidate_id = _candidate_id(record, idx)
        if source_candidate_id in selected_source_ids:
            continue
        append_record(record, source_candidate_id)

    signal_override_report = _apply_post_dcr_signal_override(
        state,
        candidate_records,
        proxy_records,
        len(selected_refine_records),
    )
    return candidate_records, proxy_records, selected_refine_records, {
        "proxy_scope": "post_dcr_expanded_pool",
        "candidate_rows": int(len(candidate_records)),
        "selected_rows": int(len(selected_refine_records)),
        "pool_rows": int(len(state.pool_records)),
        "selected_unique_source_ids": int(len(selected_source_ids)),
        "selected_duplicate_source_rows": int(len(selected_records) - len(selected_source_ids)),
        "exact_proxy_rows": int(source_counts["exact"]),
        "surrogate_proxy_rows": int(source_counts["surrogate"]),
        "utility_proxy_rows": int(len(utility_by_id)),
        "post_dcr_signal_override": signal_override_report,
    }


def build_core_selections(state: SelectionState) -> CoreSelectionOutputs:
    if state.selector is None:
        raise RuntimeError("selector is required before build_core_selections")
    if not state.pool_records or not state.preselected_valid or not state.global_exact_records:
        raise RuntimeError("pool_records, preselected_valid, and global_exact_records are required")
    if state.fidelity_ceiling_df is None:
        raise RuntimeError("build_fidelity_ceiling must run before build_core_selections")

    config = state.config
    selector = state.selector
    preselected_valid_df = records_to_df(state.preselected_valid, selector.column_order)

    t0 = time.perf_counter()
    _log(f"random_full start rows={len(state.pool_records)} keep_k={state.effective_keep_k}")
    random_full_df, random_full_records, random_full_report = selector.select_keep_random(
        candidate_records=state.pool_records,
        keep_k=state.effective_keep_k,
        rng_seed=config.seed,
    )
    _log(f"random_full done rows={len(random_full_df)} elapsed={time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    _log(f"scalar start rows={len(state.preselected_valid)} keep_k={state.effective_keep_k}")
    scalar_df, scalar_records, scalar_report = selector.select_keep_scalarization(
        preselected_records=state.preselected_valid,
        exact_records=state.global_exact_records,
        keep_k=state.effective_keep_k,
        fidelity_1d_weight=0.5 * config.scalar_fidelity_weight,
        fidelity_2d_weight=0.5 * config.scalar_fidelity_weight,
        privacy_weight=config.scalar_privacy_weight,
        utility_weight=config.scalar_utility_weight,
        mode="matched",
        floor_reference=state.floor_reference,
    )
    _log(f"scalar done rows={len(scalar_df)} elapsed={time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    _log(f"pareto start rows={len(state.preselected_valid)} keep_k={state.effective_keep_k}")
    pareto_df, pareto_records, pareto_report = selector.select_keep(
        preselected_records=state.preselected_valid,
        surrogate_records=[],
        exact_records=state.global_exact_records,
        keep_k=state.effective_keep_k,
        floor_reference=state.floor_reference,
        constraint_reference_records=state.fidelity_ceiling_records,
        floor_mode=config.pareto_floor_mode,
        soft_fidelity_floor_eps=config.pareto_soft_fidelity_floor_eps,
        soft_trend_floor_eps=config.pareto_soft_trend_floor_eps,
        soft_privacy_floor_eps=config.pareto_soft_privacy_floor_eps,
        soft_utility_floor_eps=config.pareto_soft_utility_floor_eps,
        soft_min_score_delta=config.pareto_soft_min_score_delta,
        allow_reference_anchor=True,
    )
    if config.reward_candidate_v2_enabled and config.reward_candidate_v2_pre_repair_enabled:
        pareto_df, pareto_records, reward_v2_pre_report = refine_selection_for_reward_v2(
            preselected_records=state.preselected_valid,
            exact_records=state.global_exact_records,
            selected_records=pareto_records,
            keep_k=state.effective_keep_k,
            column_order=selector.column_order,
            max_swap_fraction=config.reward_candidate_v2_max_swap_fraction,
            max_candidate_sizes=config.reward_candidate_v2_max_candidate_sizes,
            min_proxy_delta=config.reward_candidate_v2_min_proxy_delta,
            fidelity_floor_eps=config.reward_candidate_v2_fidelity_floor_eps,
            utility_floor_eps=config.reward_candidate_v2_utility_floor_eps,
        )
        reward_v2_pre_report = {
            **reward_v2_pre_report,
            "stage": "pre_direct_dcr_repair",
        }
    else:
        reward_v2_pre_report = {
            "enabled": bool(config.reward_candidate_v2_enabled),
            "version": "reward_candidate_v2",
            "applied": False,
            "reason": (
                "pre_repair_disabled"
                if config.reward_candidate_v2_enabled
                else "disabled"
            ),
            "stage": "pre_direct_dcr_repair",
        }
    pareto_df, pareto_records, pareto_report = apply_pareto_post_selection_repairs(
        state=state,
        pareto_df=pareto_df,
        pareto_records=pareto_records,
        pareto_report=pareto_report,
    )
    if config.reward_candidate_v2_enabled:
        post_candidate_records, post_proxy_records, post_selected_records, post_proxy_report = (
            _post_dcr_reward_refine_inputs(state, pareto_records)
        )
        pareto_df, pareto_records, reward_v2_post_report = refine_selection_for_reward_v2(
            preselected_records=post_candidate_records,
            exact_records=post_proxy_records,
            selected_records=post_selected_records,
            keep_k=state.effective_keep_k,
            column_order=selector.column_order,
            max_swap_fraction=config.reward_candidate_v2_max_swap_fraction,
            max_candidate_sizes=config.reward_candidate_v2_max_candidate_sizes,
            min_proxy_delta=config.reward_candidate_v2_min_proxy_delta,
            fidelity_floor_eps=config.reward_candidate_v2_fidelity_floor_eps,
            utility_floor_eps=config.reward_candidate_v2_utility_floor_eps,
            dcr_proxy_max_balance_error=config.direct_dcr_repair_v19_target_margin,
            allow_duplicate_adds=True,
        )
        reward_v2_post_report = {
            **reward_v2_post_report,
            "stage": "post_direct_dcr_repair",
            "proxy_records": post_proxy_report,
        }
    else:
        reward_v2_post_report = {
            "enabled": False,
            "version": "reward_candidate_v2",
            "applied": False,
            "reason": "disabled",
            "stage": "post_direct_dcr_repair",
        }
    pareto_report = {
        **pareto_report,
        "reward_candidate_v2": reward_v2_post_report,
        "reward_candidate_v2_pre_direct_dcr_repair": reward_v2_pre_report,
        "reward_candidate_v2_post_direct_dcr_repair": reward_v2_post_report,
    }
    _log(f"pareto done rows={len(pareto_df)} elapsed={time.perf_counter() - t0:.2f}s")

    return CoreSelectionOutputs(
        preselected_valid_df=preselected_valid_df,
        fidelity_ceiling_df=state.fidelity_ceiling_df,
        random_full_df=random_full_df,
        scalar_df=scalar_df,
        pareto_df=pareto_df,
        preselected_valid_records=state.preselected_valid,
        fidelity_ceiling_records=state.fidelity_ceiling_records,
        random_full_records=random_full_records,
        scalar_records=scalar_records,
        pareto_records=pareto_records,
        reports={
            "random_full": random_full_report,
            "scalar": scalar_report,
            "pareto": pareto_report,
        },
    )
