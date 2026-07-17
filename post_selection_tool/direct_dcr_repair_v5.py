from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .direct_dcr_repair_v4 import (
    _build_pairs,
    _candidate_id,
    _row_dcr_signal,
    _utility_scores,
)
from .io import records_to_df


def apply_direct_dcr_repair_v5(
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
    max_swap_fraction: float = 0.22,
    candidate_neighbors: int = 64,
    margin_weight: float = 0.05,
    utility_weight: float = 0.35,
    cat_weight: float = 1.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    report_base: dict[str, Any] = {
        "enabled": True,
        "version": "direct_dcr_repair_v5",
        "candidate_full_eval_used": False,
        "intermediate_candidate_count": 0,
        "selection_signal": "row_level_train_vs_test_dcr_direction",
    }
    if not pool_records or not selected_records:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {**report_base, "applied": False, "reason": "empty_inputs"},
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

    pool_df = records_to_df(pool_records, column_order)
    signal = _row_dcr_signal(
        pool_df=pool_df,
        train_df=train_df[column_order],
        test_df=test_df[column_order],
        schema_card=schema_card,
        column_order=column_order,
        cat_weight=cat_weight,
    )
    is_real_closer = np.asarray(signal["is_real_closer"], dtype=bool)
    base_dcr = float(np.mean(is_real_closer[selected_pool_indices]))
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
                "base_dcr_estimate": base_dcr,
                "target_band": [float(lower_target), float(upper_target)],
                "base_dcr_privacy_estimate": float(1.0 - abs(base_dcr - 0.5)),
            },
        )

    keep_k = len(selected_records)
    max_swaps = max(0, int(round(float(keep_k) * max(0.0, float(max_swap_fraction)))))
    if desired_swaps <= 0 or max_swaps <= 0:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "no_required_swaps",
                "base_dcr_estimate": base_dcr,
                "target_dcr": target_dcr,
                "target_band": [float(lower_target), float(upper_target)],
            },
        )

    utility, utility_report = _utility_scores(
        pool_records=pool_records,
        exact_records=exact_records,
        surrogate_records=surrogate_records,
    )
    pairs = _build_pairs(
        selected_pool_indices=selected_pool_indices,
        selected_mask=selected_mask,
        is_real_closer=is_real_closer,
        margin=np.asarray(signal["margin"], dtype=float),
        features=np.asarray(signal["features"], dtype=np.float32),
        utility_scores=utility,
        reduce_dcr=reduce_dcr,
        candidate_neighbors=candidate_neighbors,
        margin_weight=margin_weight,
        utility_weight=utility_weight,
    )
    selected_swaps = min(int(desired_swaps), int(max_swaps), int(len(pairs)))
    if selected_swaps <= 0:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "no_feasible_pairs",
                "base_dcr_estimate": base_dcr,
                "target_dcr": target_dcr,
                "target_band": [float(lower_target), float(upper_target)],
                "desired_swaps": int(desired_swaps),
                "pair_count": int(len(pairs)),
            },
        )

    final_records = [dict(record) for record in selected_records]
    final_pool_indices = selected_pool_indices.copy()
    for remove_base_pos, add_pool_idx, _, _, _ in pairs[:selected_swaps]:
        final_records[int(remove_base_pos)] = dict(pool_records[int(add_pool_idx)])
        final_pool_indices[int(remove_base_pos)] = int(add_pool_idx)

    final_dcr = float(np.mean(is_real_closer[final_pool_indices]))
    final_df = records_to_df(final_records, column_order)
    prefix = pairs[:selected_swaps]
    return (
        final_df,
        final_records,
        {
            **report_base,
            "applied": True,
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
            "reduce_dcr": bool(reduce_dcr),
            "candidate_neighbors": int(candidate_neighbors),
            "margin_weight": float(margin_weight),
            "utility_weight": float(utility_weight),
            "cat_weight": float(cat_weight),
            "mean_pair_distance": float(np.mean([item[2] for item in prefix])),
            "mean_pair_utility_gain": float(np.mean([item[4] for item in prefix])),
            "sum_pair_utility_gain": float(np.sum([item[4] for item in prefix])),
            "utility_scores": utility_report,
            "removed_candidate_ids": [
                _candidate_id(selected_records[int(item[0])], int(item[0]))
                for item in prefix
            ],
            "added_candidate_ids": [
                _candidate_id(pool_records[int(item[1])], int(item[1]))
                for item in prefix
            ],
        },
    )
