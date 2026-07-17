from __future__ import annotations

import pandas as pd

from post_selection_tool.direct_dcr_repair_v11 import apply_direct_dcr_repair_v11


def _schema_card() -> dict:
    return {
        "dataset": "toy",
        "column_order": ["x", "target"],
        "target_column": "target",
        "columns": {
            "x": {"type": "numerical", "is_target": False},
            "target": {"type": "categorical", "is_target": True, "legal_values": ["0", "1"]},
        },
    }


def _record(idx: int, x: float, target: str = "0") -> dict:
    return {"candidate_id": idx, "row": {"x": x, "target": target}}


def test_direct_dcr_repair_v11_prefers_same_target_pairs() -> None:
    train_df = pd.DataFrame({"x": [0.0, 10.0], "target": ["0", "1"]})
    test_df = pd.DataFrame({"x": [100.0, 110.0], "target": ["0", "1"]})
    selected_real = [_record(idx, 0.1 + idx * 0.01, "0") for idx in range(18)]
    selected_test = [_record(18 + idx, 100.1 + idx * 0.01, "0") for idx in range(2)]
    extra_test_same_target = [_record(20 + idx, 100.3 + idx * 0.01, "0") for idx in range(16)]
    extra_test_cross_target = [_record(36 + idx, 100.3 + idx * 0.01, "1") for idx in range(16)]
    extra_real = [_record(52 + idx, 0.3 + idx * 0.01, "0") for idx in range(8)]
    pool_records = selected_real + selected_test + extra_test_same_target + extra_test_cross_target + extra_real
    selected_records = selected_real + selected_test

    exact_records = [
        {"candidate_id": idx, "pareto_util_proxy_obj": 0.8, "pareto_priv_obj": 0.5, "pareto_fid_obj": 0.5}
        for idx in range(20)
    ]
    surrogate_records = []
    for idx in range(len(pool_records)):
        is_test_like = 18 <= idx < 52
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

    _, final_records, report = apply_direct_dcr_repair_v11(
        pool_records=pool_records,
        selected_records=selected_records,
        exact_records=exact_records,
        surrogate_records=surrogate_records,
        train_df=train_df,
        test_df=test_df,
        schema_card=_schema_card(),
        column_order=["x", "target"],
        target_margin=0.05,
        max_swap_fraction=1.0,
        candidate_neighbors=6,
        large_keep_k_threshold=1,
        large_pool_rows_threshold=1,
        large_candidate_rows=24,
        large_reference_rows=0,
        large_max_swaps=8,
        large_candidate_neighbors=4,
        min_pair_utility_gain=0.0,
        fallback_min_pair_utility_gain=-1.0,
        signal_query_batch_size=4,
        signal_reference_chunk_size=4,
        signal_device="cpu",
        report_id_limit=3,
        allow_cross_target_fallback=False,
    )

    final_ids = {record["candidate_id"] for record in final_records}
    assert report["applied"] is True
    assert report["version"] == "direct_dcr_repair_v11"
    assert report["candidate_full_eval_used"] is False
    assert report["intermediate_candidate_count"] == 0
    assert report["pair_builder_mode"] == "target_aware_grouped"
    assert report["target_group_mode"] == "exact_target_labels"
    assert report["same_target_candidate_edges"] > 0
    assert report["selected_target_match_fraction"] == 1.0
    assert report["selected_swaps"] > 0
    assert report["final_dcr_estimate"] < report["base_dcr_estimate"]
    assert final_ids & set(range(20, 36))
