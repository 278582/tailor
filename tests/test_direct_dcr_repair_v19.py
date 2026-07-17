from __future__ import annotations

import pandas as pd

from post_selection_tool.direct_dcr_repair_v19 import apply_direct_dcr_repair_v19


def _schema_card() -> dict:
    return {
        "dataset": "toy",
        "column_order": ["x", "target"],
        "target_column": "target",
        "columns": {
            "x": {"type": "numerical", "is_target": False},
            "target": {"type": "categorical", "is_target": True, "legal_values": ["0"]},
        },
    }


def _record(idx: int, x: float) -> dict:
    return {"candidate_id": idx, "row": {"x": x, "target": "0"}}


def test_direct_dcr_repair_v19_uses_center_target_duplicate_fill() -> None:
    train_df = pd.DataFrame({"x": [0.0, 10.0], "target": ["0", "0"]})
    test_df = pd.DataFrame({"x": [100.0, 110.0], "target": ["0", "0"]})
    selected_real = [_record(idx, 0.1 + idx * 0.01) for idx in range(18)]
    selected_test = [_record(18 + idx, 100.1 + idx * 0.01) for idx in range(2)]
    extra_test = [_record(20 + idx, 100.3 + idx * 0.01) for idx in range(3)]
    extra_real = [_record(23 + idx, 0.3 + idx * 0.01) for idx in range(8)]
    pool_records = selected_real + selected_test + extra_test + extra_real
    selected_records = selected_real + selected_test

    exact_records = [
        {"candidate_id": idx, "pareto_util_proxy_obj": 0.8, "pareto_priv_obj": 0.5, "pareto_fid_obj": 0.5}
        for idx in range(20)
    ]
    surrogate_records = []
    for idx in range(len(pool_records)):
        is_test_like = 18 <= idx < 23
        surrogate_records.append(
            {
                "candidate_id": idx,
                "holdout_gap": -2.0 if is_test_like else 2.0,
                "s_preselect_stage_b": 0.9,
                "s_pareto_fid_1d_sur": 0.9,
                "s_pareto_fid_2d_sur": 0.9,
                "s_preselect_support_tiebreak": 0.9,
                "s_preselect_priv_tiebreak": 0.9,
            }
        )

    _, final_records, report = apply_direct_dcr_repair_v19(
        pool_records=pool_records,
        selected_records=selected_records,
        exact_records=exact_records,
        surrogate_records=surrogate_records,
        train_df=train_df,
        test_df=test_df,
        schema_card=_schema_card(),
        column_order=["x", "target"],
        target_margin=0.03,
        max_swap_fraction=1.0,
        candidate_neighbors=8,
        large_keep_k_threshold=50_000,
        large_pool_rows_threshold=180_000,
        large_candidate_rows=4,
        large_reference_rows=0,
        large_max_swaps=8,
        large_candidate_neighbors=4,
        min_pair_utility_gain=0.0,
        fallback_min_pair_utility_gain=-1.0,
        signal_query_batch_size=4,
        signal_reference_chunk_size=4,
        signal_device="cpu",
        report_id_limit=3,
        generic_remove_budget=8,
    )

    final_ids = [record["candidate_id"] for record in final_records]
    assert report["applied"] is True
    assert report["version"] == "direct_dcr_repair_v19"
    assert report["v19_source_strategy"] == "direct_dcr_repair_v15"
    assert report["candidate_full_eval_used"] is False
    assert report["intermediate_candidate_count"] == 0
    assert report["v19_center_target_margin"] == 0.03
    assert report["duplicate_fill_pair_count"] > 0
    assert report["selected_swaps"] == report["desired_swaps"]
    assert len(set(final_ids)) < len(final_ids)
