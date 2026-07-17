from __future__ import annotations

import pandas as pd

from post_selection_tool.direct_dcr_repair_v8 import apply_direct_dcr_repair_v8


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


def test_direct_dcr_repair_v8_uses_full_selected_signal_in_large_mode() -> None:
    train_df = pd.DataFrame({"x": [0.0, 10.0], "target": ["0", "0"]})
    test_df = pd.DataFrame({"x": [100.0, 110.0], "target": ["0", "0"]})
    selected_real = [_record(idx, 0.1 + idx * 0.01) for idx in range(18)]
    selected_test = [_record(18 + idx, 100.1 + idx * 0.01) for idx in range(2)]
    extra_test = [_record(20 + idx, 100.3 + idx * 0.01) for idx in range(12)]
    pool_records = selected_real + selected_test + extra_test
    selected_records = selected_real + selected_test
    exact_records = [
        {"candidate_id": idx, "pareto_util_proxy_obj": 0.1, "pareto_priv_obj": 0.5, "pareto_fid_obj": 0.5}
        for idx in range(20)
    ] + [
        {"candidate_id": 20 + idx, "pareto_util_proxy_obj": 0.9, "pareto_priv_obj": 0.9, "pareto_fid_obj": 0.9}
        for idx in range(12)
    ]

    _, final_records, report = apply_direct_dcr_repair_v8(
        pool_records=pool_records,
        selected_records=selected_records,
        exact_records=exact_records,
        surrogate_records=None,
        train_df=train_df,
        test_df=test_df,
        schema_card=_schema_card(),
        column_order=["x", "target"],
        target_margin=0.05,
        max_swap_fraction=1.0,
        candidate_neighbors=4,
        large_keep_k_threshold=1,
        large_pool_rows_threshold=1,
        large_candidate_rows=8,
        large_reference_rows=0,
        large_max_swaps=5,
        large_candidate_neighbors=3,
        min_pair_utility_gain=0.0,
        report_id_limit=2,
    )

    final_ids = {record["candidate_id"] for record in final_records}
    assert report["applied"] is True
    assert report["version"] == "direct_dcr_repair_v8"
    assert report["bounded_mode"] is True
    assert report["sampled_estimate"] is False
    assert report["intermediate_candidate_count"] == 0
    assert report["candidate_full_eval_used"] is False
    assert report["selected_swaps"] <= 5
    assert report["final_dcr_estimate"] < report["base_dcr_estimate"]
    assert len(report["removed_candidate_ids"]) <= 2
    assert len(report["added_candidate_ids"]) <= 2
    assert report["removed_candidate_id_count"] == report["selected_swaps"]
    assert report["added_candidate_id_count"] == report["selected_swaps"]
    assert final_ids & {20, 21, 22, 23, 24, 25, 26, 27}
