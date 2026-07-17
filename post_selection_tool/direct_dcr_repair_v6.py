from __future__ import annotations

from typing import Any

import pandas as pd

from .direct_dcr_repair_v4 import apply_direct_dcr_repair_v4
from .direct_dcr_repair_v5 import apply_direct_dcr_repair_v5
from .io import records_to_df


def apply_direct_dcr_repair_v6(
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
    v5_df, v5_records, v5_report = apply_direct_dcr_repair_v5(
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
    if bool(v5_report.get("applied", False)) or v5_report.get("reason") != "base_dcr_within_target_band":
        return v5_df, v5_records, {**v5_report, "version": "direct_dcr_repair_v6", "base_strategy": "v5_out_of_band"}

    v4_df, v4_records, v4_report = apply_direct_dcr_repair_v4(
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
    lower, upper = v5_report.get("target_band", [0.5 - abs(float(target_margin)), 0.5 + abs(float(target_margin))])
    final_dcr = v4_report.get("final_dcr_estimate")
    mean_utility_gain = float(v4_report.get("mean_pair_utility_gain", 0.0))
    accepted = bool(
        v4_report.get("applied", False)
        and final_dcr is not None
        and float(lower) <= float(final_dcr) <= float(upper)
        and mean_utility_gain > 0.0
    )
    if accepted:
        return (
            v4_df,
            v4_records,
            {
                **v4_report,
                "version": "direct_dcr_repair_v6",
                "base_strategy": "in_band_utility_positive_v4",
                "target_band": [float(lower), float(upper)],
                "in_band_candidate_accepted": True,
                "in_band_v5_report": v5_report,
            },
        )
    return (
        records_to_df(selected_records, column_order),
        selected_records,
        {
            **v5_report,
            "version": "direct_dcr_repair_v6",
            "applied": False,
            "reason": "base_dcr_within_target_band_no_positive_safe_utility_swap",
            "base_strategy": "in_band_guarded_skip",
            "in_band_candidate_accepted": False,
            "in_band_v4_report": v4_report,
        },
    )
