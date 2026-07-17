from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .direct_dcr_repair_v4 import (
    _build_pairs,
    _candidate_id,
    _finite_float,
    _row_dcr_signal,
    _utility_scores,
)
from .direct_dcr_repair_v6 import apply_direct_dcr_repair_v6
from .io import records_to_df
from .logging_utils import get_logger


def _log(message: str) -> None:
    get_logger().info("[direct_dcr_repair_v9] %s", message)


def _reference_frame(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    max_rows = int(max_rows)
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    positions = np.linspace(0, len(df) - 1, num=max_rows, dtype=np.int64)
    return df.iloc[positions].reset_index(drop=True)


def _truncate_ids(values: list[int], limit: int) -> list[int]:
    limit = max(0, int(limit))
    if limit == 0:
        return []
    return values[:limit]


def _surrogate_holdout_gap(record: dict[str, Any]) -> float:
    if record.get("holdout_gap") is not None:
        return _finite_float(record, "holdout_gap", np.nan)
    return _finite_float(record, "nn_distance_holdout", np.nan) - _finite_float(record, "nn_distance_train", np.nan)


def _surrogate_quality(record: dict[str, Any]) -> float:
    stage = _finite_float(record, "s_preselect_stage_b", _finite_float(record, "s_preselect_band", 0.5))
    fid1 = _finite_float(record, "s_pareto_fid_1d_sur", _finite_float(record, "s_fid_sur_1d_rank", 0.5))
    fid2 = _finite_float(record, "s_pareto_fid_2d_sur", _finite_float(record, "s_fid_sur_2d_rank", 0.5))
    support = _finite_float(record, "s_preselect_support_tiebreak", 0.5)
    privacy = _finite_float(record, "s_preselect_priv_tiebreak", 0.5)
    return float(np.clip(0.30 * stage + 0.25 * fid1 + 0.25 * fid2 + 0.12 * support + 0.08 * privacy, 0.0, 1.0))


def _signal_prior_arrays(
    *,
    pool_records: list[dict[str, Any]],
    exact_records: list[dict[str, Any]],
    surrogate_records: list[dict[str, Any]] | None,
    pool_id_to_index: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exact_quality = np.full(len(pool_records), np.nan, dtype=float)
    holdout_gap = np.full(len(pool_records), np.nan, dtype=float)
    surrogate_quality = np.full(len(pool_records), 0.5, dtype=float)

    for record in exact_records:
        pool_idx = pool_id_to_index.get(_candidate_id(record, -1))
        if pool_idx is None:
            continue
        utility = _finite_float(record, "pareto_util_proxy_obj", 0.5)
        privacy = _finite_float(record, "pareto_priv_obj", 0.5)
        fid = _finite_float(
            record,
            "pareto_fid_obj",
            0.5 * (
                _finite_float(record, "pareto_fid_1d_obj", 0.5)
                + _finite_float(record, "pareto_fid_2d_obj", 0.5)
            ),
        )
        exact_quality[pool_idx] = float(np.clip(0.55 * utility + 0.25 * fid + 0.20 * privacy, 0.0, 1.0))
        if record.get("holdout_gap") is not None:
            holdout_gap[pool_idx] = _surrogate_holdout_gap(record)

    if surrogate_records is not None:
        for record in surrogate_records:
            pool_idx = pool_id_to_index.get(_candidate_id(record, -1))
            if pool_idx is None:
                continue
            if not np.isfinite(holdout_gap[pool_idx]):
                holdout_gap[pool_idx] = _surrogate_holdout_gap(record)
            surrogate_quality[pool_idx] = _surrogate_quality(record)

    return exact_quality, holdout_gap, surrogate_quality


def _take_ranked(indices: np.ndarray, scores: np.ndarray, limit: int) -> list[int]:
    limit = max(0, int(limit))
    if limit <= 0 or indices.size == 0:
        return []
    local_scores = np.nan_to_num(scores[indices], nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    order = np.lexsort((indices, -local_scores))
    return [int(idx) for idx in indices[order[:limit]].tolist()]


def _add_unique(target: list[int], seen: set[int], values: list[int], limit: int) -> None:
    if len(target) >= limit:
        return
    for value in values:
        idx = int(value)
        if idx in seen:
            continue
        seen.add(idx)
        target.append(idx)
        if len(target) >= limit:
            break


def _candidate_pool_indices_v9(
    *,
    selected_mask: np.ndarray,
    selected_pool_indices: np.ndarray,
    utility: np.ndarray,
    exact_quality: np.ndarray,
    holdout_gap: np.ndarray,
    surrogate_quality: np.ndarray,
    max_rows: int,
) -> tuple[list[int], dict[str, Any]]:
    max_rows = max(0, int(max_rows))
    unselected = np.flatnonzero(~selected_mask)
    if max_rows <= 0 or unselected.size == 0:
        return [], {"reason": "empty_unselected_or_zero_budget"}
    if unselected.size <= max_rows:
        return [int(idx) for idx in unselected.tolist()], {
            "reason": "unselected_within_budget",
            "unselected_rows": int(unselected.size),
        }

    quality = np.where(np.isfinite(exact_quality), exact_quality, surrogate_quality)
    abs_gap = np.nan_to_num(np.abs(holdout_gap), nan=0.0, posinf=0.0, neginf=0.0)
    finite_gap = abs_gap[unselected][np.isfinite(abs_gap[unselected])]
    gap_scale = float(np.percentile(finite_gap, 95)) if finite_gap.size else 1.0
    if gap_scale <= 1e-12:
        gap_scale = 1.0
    scaled_abs_gap = np.clip(abs_gap / gap_scale, 0.0, 1.0)
    rank_score = (
        0.55 * np.nan_to_num(quality, nan=0.5)
        + 0.25 * np.nan_to_num(utility, nan=0.5)
        + 0.20 * scaled_abs_gap
    )

    selected_gap = holdout_gap[selected_pool_indices]
    finite_selected_gap = selected_gap[np.isfinite(selected_gap)]
    surrogate_base_real = None
    if finite_selected_gap.size:
        surrogate_base_real = float(np.mean(finite_selected_gap >= 0.0))

    exact_budget = int(round(max_rows * 0.25))
    primary_budget = int(round(max_rows * 0.45))
    secondary_budget = int(round(max_rows * 0.15))
    fill_budget = max_rows

    selected: list[int] = []
    seen: set[int] = set()
    exact_candidates = unselected[np.isfinite(exact_quality[unselected])]
    _add_unique(selected, seen, _take_ranked(exact_candidates, rank_score, exact_budget), max_rows)

    finite_gap_unselected = unselected[np.isfinite(holdout_gap[unselected])]
    if surrogate_base_real is None:
        primary_mask = holdout_gap[finite_gap_unselected] < 0.0
        secondary_mask = ~primary_mask
        primary_budget = int(round(max_rows * 0.30))
        secondary_budget = int(round(max_rows * 0.30))
    elif surrogate_base_real >= 0.5:
        primary_mask = holdout_gap[finite_gap_unselected] < 0.0
        secondary_mask = holdout_gap[finite_gap_unselected] >= 0.0
    else:
        primary_mask = holdout_gap[finite_gap_unselected] >= 0.0
        secondary_mask = holdout_gap[finite_gap_unselected] < 0.0

    primary = finite_gap_unselected[primary_mask]
    secondary = finite_gap_unselected[secondary_mask]
    _add_unique(selected, seen, _take_ranked(primary, rank_score, primary_budget), max_rows)
    _add_unique(selected, seen, _take_ranked(secondary, rank_score, secondary_budget), max_rows)
    _add_unique(selected, seen, _take_ranked(unselected, rank_score, fill_budget), max_rows)

    selected_arr = np.asarray(selected, dtype=np.int64)
    selected_gap_values = holdout_gap[selected_arr] if selected_arr.size else np.asarray([], dtype=float)
    finite_selected = np.isfinite(selected_gap_values)
    return selected, {
        "reason": "ranked_surrogate_direction_budget",
        "unselected_rows": int(unselected.size),
        "candidate_rows": int(len(selected)),
        "exact_candidate_rows": int(exact_candidates.size),
        "surrogate_base_real_fraction": surrogate_base_real,
        "candidate_gap_finite_rows": int(np.count_nonzero(finite_selected)),
        "candidate_gap_real_rows": int(np.count_nonzero(selected_gap_values[finite_selected] >= 0.0)) if np.any(finite_selected) else 0,
        "candidate_gap_holdout_rows": int(np.count_nonzero(selected_gap_values[finite_selected] < 0.0)) if np.any(finite_selected) else 0,
    }


def _progressive_pair_filter(
    pairs: list[tuple[int, int, float, float, float]],
    *,
    desired_swaps: int,
    max_swaps: int,
    min_pair_utility_gain: float,
    fallback_min_pair_utility_gain: float,
) -> tuple[list[tuple[int, int, float, float, float]], dict[str, Any]]:
    target = max(0, min(int(desired_swaps), int(max_swaps)))
    if target <= 0 or not pairs:
        return [], {"pair_filter_mode": "empty_or_zero_target", "strict_pair_count": 0, "fallback_pair_count": 0}

    strict = [pair for pair in pairs if float(pair[4]) >= float(min_pair_utility_gain)]
    if len(strict) >= target:
        return strict, {
            "pair_filter_mode": "strict",
            "active_min_pair_utility_gain": float(min_pair_utility_gain),
            "strict_pair_count": int(len(strict)),
            "fallback_pair_count": int(len(strict)),
        }

    fallback_floor = min(float(min_pair_utility_gain), float(fallback_min_pair_utility_gain))
    fallback = [pair for pair in pairs if float(pair[4]) >= fallback_floor]
    if fallback:
        return fallback, {
            "pair_filter_mode": "fallback",
            "active_min_pair_utility_gain": float(fallback_floor),
            "strict_pair_count": int(len(strict)),
            "fallback_pair_count": int(len(fallback)),
        }

    return pairs, {
        "pair_filter_mode": "unfiltered_last_resort",
        "active_min_pair_utility_gain": None,
        "strict_pair_count": int(len(strict)),
        "fallback_pair_count": 0,
    }


def apply_direct_dcr_repair_v9(
    *,
    pool_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    exact_records: list[dict[str, Any]],
    surrogate_records: list[dict[str, Any]] | None,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_card: dict[str, Any],
    column_order: list[str],
    target_margin: float = 0.05,
    max_swap_fraction: float = 0.26,
    candidate_neighbors: int = 64,
    margin_weight: float = 0.08,
    utility_weight: float = 0.50,
    cat_weight: float = 1.0,
    large_keep_k_threshold: int = 50_000,
    large_pool_rows_threshold: int = 180_000,
    large_candidate_rows: int = 72_000,
    large_reference_rows: int = 12_000,
    large_max_swaps: int = 18_000,
    large_candidate_neighbors: int = 20,
    min_pair_utility_gain: float = -0.08,
    fallback_min_pair_utility_gain: float = -0.20,
    report_id_limit: int = 64,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    report_base: dict[str, Any] = {
        "enabled": True,
        "version": "direct_dcr_repair_v9",
        "candidate_full_eval_used": False,
        "intermediate_candidate_count": 0,
        "selection_signal": "full_selected_surrogate_biased_candidates_budgeted_train_vs_test_dcr_direction",
    }
    if not pool_records or not selected_records:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {**report_base, "applied": False, "reason": "empty_inputs"},
        )

    use_bounded = (
        len(selected_records) > int(large_keep_k_threshold)
        or len(pool_records) > int(large_pool_rows_threshold)
    )
    if not use_bounded:
        final_df, final_records, v6_report = apply_direct_dcr_repair_v6(
            pool_records=pool_records,
            selected_records=selected_records,
            exact_records=exact_records,
            surrogate_records=surrogate_records,
            train_df=train_df,
            test_df=test_df,
            schema_card=schema_card,
            column_order=column_order,
            target_margin=target_margin,
            max_swap_fraction=max_swap_fraction,
            candidate_neighbors=candidate_neighbors,
            margin_weight=margin_weight,
            utility_weight=utility_weight,
            cat_weight=cat_weight,
        )
        return final_df, final_records, {
            **v6_report,
            "version": "direct_dcr_repair_v9",
            "base_strategy": "exact_v6_small",
            "bounded_mode": False,
        }

    t_all = time.perf_counter()
    _log(
        f"bounded start pool_rows={len(pool_records)} selected_rows={len(selected_records)} "
        f"candidate_budget={int(large_candidate_rows)} reference_budget={int(large_reference_rows)}"
    )
    pool_id_to_index = {
        _candidate_id(record, idx): idx
        for idx, record in enumerate(pool_records)
    }
    selected_pool_indices_list: list[int] = []
    for pos, record in enumerate(selected_records):
        idx = pool_id_to_index.get(_candidate_id(record, pos))
        if idx is None:
            return (
                records_to_df(selected_records, column_order),
                selected_records,
                {**report_base, "applied": False, "reason": "selected_record_not_in_pool"},
            )
        selected_pool_indices_list.append(int(idx))

    selected_pool_indices = np.asarray(selected_pool_indices_list, dtype=np.int64)
    selected_mask = np.zeros(len(pool_records), dtype=bool)
    selected_mask[selected_pool_indices] = True
    if int(selected_mask.sum()) != len(selected_records):
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {**report_base, "applied": False, "reason": "duplicate_selected_pool_indices"},
        )

    utility, utility_report = _utility_scores(
        pool_records=pool_records,
        exact_records=exact_records,
        surrogate_records=surrogate_records,
    )
    exact_quality, holdout_gap, surrogate_quality = _signal_prior_arrays(
        pool_records=pool_records,
        exact_records=exact_records,
        surrogate_records=surrogate_records,
        pool_id_to_index=pool_id_to_index,
    )
    add_pool_indices, candidate_report = _candidate_pool_indices_v9(
        selected_mask=selected_mask,
        selected_pool_indices=selected_pool_indices,
        utility=utility,
        exact_quality=exact_quality,
        holdout_gap=holdout_gap,
        surrogate_quality=surrogate_quality,
        max_rows=int(large_candidate_rows),
    )
    if not add_pool_indices:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "empty_bounded_candidate_pool",
                "bounded_mode": True,
                "pool_rows": int(len(pool_records)),
                "keep_k": int(len(selected_records)),
                "candidate_pool": candidate_report,
            },
        )
    _log(f"candidate pool ready rows={len(add_pool_indices)} report={candidate_report}")

    local_pool_indices = np.concatenate(
        [
            selected_pool_indices,
            np.asarray(add_pool_indices, dtype=np.int64),
        ]
    )
    local_selected_count = int(selected_pool_indices.size)
    local_records = [pool_records[int(idx)] for idx in local_pool_indices.tolist()]
    local_df = records_to_df(local_records, column_order)
    train_ref = _reference_frame(train_df[column_order], int(large_reference_rows))
    test_ref = _reference_frame(test_df[column_order], int(large_reference_rows))

    t_signal = time.perf_counter()
    _log(f"signal start local_rows={len(local_records)} train_ref={len(train_ref)} test_ref={len(test_ref)}")
    signal = _row_dcr_signal(
        pool_df=local_df[column_order],
        train_df=train_ref[column_order],
        test_df=test_ref[column_order],
        schema_card=schema_card,
        column_order=column_order,
        cat_weight=cat_weight,
    )
    signal_elapsed = time.perf_counter() - t_signal
    _log(f"signal done elapsed={signal_elapsed:.2f}s")

    is_real_closer = np.asarray(signal["is_real_closer"], dtype=bool)
    base_dcr = float(np.mean(is_real_closer[:local_selected_count]))
    target_margin = abs(float(target_margin))
    lower_target = 0.5 - target_margin
    upper_target = 0.5 + target_margin
    if base_dcr > upper_target:
        reduce_dcr = True
        target_dcr = upper_target
        desired_swaps = int(round((base_dcr - upper_target) * float(len(selected_records))))
    elif base_dcr < lower_target:
        reduce_dcr = False
        target_dcr = lower_target
        desired_swaps = int(round((lower_target - base_dcr) * float(len(selected_records))))
    else:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "base_dcr_within_target_band",
                "bounded_mode": True,
                "sampled_estimate": True,
                "base_dcr_estimate": base_dcr,
                "target_band": [float(lower_target), float(upper_target)],
                "base_dcr_privacy_estimate": float(1.0 - abs(base_dcr - 0.5)),
                "pool_rows": int(len(pool_records)),
                "keep_k": int(len(selected_records)),
                "local_selected_rows": int(local_selected_count),
                "candidate_rows": int(len(add_pool_indices)),
                "reference_rows": [int(len(train_ref)), int(len(test_ref))],
                "candidate_pool": candidate_report,
                "signal_elapsed_seconds": float(signal_elapsed),
                "elapsed_seconds": float(time.perf_counter() - t_all),
            },
        )

    max_swaps = max(0, int(round(float(len(selected_records)) * max(0.0, float(max_swap_fraction)))))
    max_swaps = min(max_swaps, max(0, int(large_max_swaps)))
    if desired_swaps <= 0 or max_swaps <= 0:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "no_required_bounded_swaps",
                "bounded_mode": True,
                "sampled_estimate": True,
                "base_dcr_estimate": base_dcr,
                "target_dcr": float(target_dcr),
                "target_band": [float(lower_target), float(upper_target)],
                "desired_swaps": int(desired_swaps),
                "max_swaps": int(max_swaps),
            },
        )

    local_selected_rows = np.arange(local_selected_count, dtype=np.int64)
    local_selected_mask = np.zeros(len(local_pool_indices), dtype=bool)
    local_selected_mask[:local_selected_count] = True
    local_utility = utility[local_pool_indices]
    t_pairs = time.perf_counter()
    _log(f"pairs start desired_swaps={desired_swaps} max_swaps={max_swaps} reduce_dcr={reduce_dcr}")
    raw_pairs = _build_pairs(
        selected_pool_indices=local_selected_rows,
        selected_mask=local_selected_mask,
        is_real_closer=is_real_closer,
        margin=np.asarray(signal["margin"], dtype=float),
        features=np.asarray(signal["features"], dtype=np.float32),
        utility_scores=local_utility,
        reduce_dcr=reduce_dcr,
        candidate_neighbors=min(int(candidate_neighbors), int(large_candidate_neighbors)),
        margin_weight=margin_weight,
        utility_weight=utility_weight,
    )
    pairs, pair_filter_report = _progressive_pair_filter(
        raw_pairs,
        desired_swaps=desired_swaps,
        max_swaps=max_swaps,
        min_pair_utility_gain=min_pair_utility_gain,
        fallback_min_pair_utility_gain=fallback_min_pair_utility_gain,
    )
    pair_elapsed = time.perf_counter() - t_pairs
    selected_swaps = min(int(desired_swaps), int(max_swaps), int(len(pairs)))
    _log(
        f"pairs done raw={len(raw_pairs)} filtered={len(pairs)} selected_swaps={selected_swaps} "
        f"elapsed={pair_elapsed:.2f}s filter={pair_filter_report}"
    )
    if selected_swaps <= 0:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "no_feasible_bounded_pairs",
                "bounded_mode": True,
                "sampled_estimate": True,
                "base_dcr_estimate": base_dcr,
                "target_dcr": float(target_dcr),
                "target_band": [float(lower_target), float(upper_target)],
                "desired_swaps": int(desired_swaps),
                "max_swaps": int(max_swaps),
                "pair_count": int(len(pairs)),
                "raw_pair_count": int(len(raw_pairs)),
                **pair_filter_report,
            },
        )

    final_records = [dict(record) for record in selected_records]
    local_selected_flags = is_real_closer[:local_selected_count].copy()
    removed_ids: list[int] = []
    added_ids: list[int] = []
    prefix = pairs[:selected_swaps]
    for remove_local_pos, add_local_idx, _, _, _ in prefix:
        actual_selected_pos = int(remove_local_pos)
        actual_add_pool_idx = int(local_pool_indices[int(add_local_idx)])
        final_records[actual_selected_pos] = dict(pool_records[actual_add_pool_idx])
        local_selected_flags[int(remove_local_pos)] = bool(is_real_closer[int(add_local_idx)])
        removed_ids.append(_candidate_id(selected_records[actual_selected_pos], actual_selected_pos))
        added_ids.append(_candidate_id(pool_records[actual_add_pool_idx], actual_add_pool_idx))

    final_dcr = float(np.mean(local_selected_flags))
    final_df = records_to_df(final_records, column_order)
    return (
        final_df,
        final_records,
        {
            **report_base,
            "applied": True,
            "bounded_mode": True,
            "sampled_estimate": True,
            "base_strategy": "full_selected_surrogate_biased_candidates_v9",
            "base_dcr_estimate": base_dcr,
            "target_dcr": float(target_dcr),
            "target_band": [float(lower_target), float(upper_target)],
            "final_dcr_estimate": final_dcr,
            "base_dcr_privacy_estimate": float(1.0 - abs(base_dcr - 0.5)),
            "final_dcr_privacy_estimate": float(1.0 - abs(final_dcr - 0.5)),
            "desired_swaps": int(desired_swaps),
            "max_swaps": int(max_swaps),
            "selected_swaps": int(selected_swaps),
            "pair_count": int(len(pairs)),
            "raw_pair_count": int(len(raw_pairs)),
            "reduce_dcr": bool(reduce_dcr),
            "candidate_neighbors": int(min(int(candidate_neighbors), int(large_candidate_neighbors))),
            "margin_weight": float(margin_weight),
            "utility_weight": float(utility_weight),
            "min_pair_utility_gain": float(min_pair_utility_gain),
            "fallback_min_pair_utility_gain": float(fallback_min_pair_utility_gain),
            **pair_filter_report,
            "cat_weight": float(cat_weight),
            "mean_pair_distance": float(np.mean([item[2] for item in prefix])),
            "mean_pair_utility_gain": float(np.mean([item[4] for item in prefix])),
            "sum_pair_utility_gain": float(np.sum([item[4] for item in prefix])),
            "utility_scores": utility_report,
            "pool_rows": int(len(pool_records)),
            "keep_k": int(len(selected_records)),
            "local_selected_rows": int(local_selected_count),
            "candidate_rows": int(len(add_pool_indices)),
            "reference_rows": [int(len(train_ref)), int(len(test_ref))],
            "candidate_pool": candidate_report,
            "signal_elapsed_seconds": float(signal_elapsed),
            "pair_elapsed_seconds": float(pair_elapsed),
            "elapsed_seconds": float(time.perf_counter() - t_all),
            "report_id_limit": int(report_id_limit),
            "removed_candidate_ids": _truncate_ids(removed_ids, report_id_limit),
            "added_candidate_ids": _truncate_ids(added_ids, report_id_limit),
            "removed_candidate_id_count": int(len(removed_ids)),
            "added_candidate_id_count": int(len(added_ids)),
        },
    )
