from __future__ import annotations

import pandas as pd

from post_selection_tool.direct_dcr_repair_v6 import apply_direct_dcr_repair_v6


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


def test_direct_dcr_repair_v6_accepts_in_band_positive_utility_swap() -> None:
    train_df = pd.DataFrame({"x": [0.0, 10.0], "target": ["0", "0"]})
    test_df = pd.DataFrame({"x": [100.0, 110.0], "target": ["0", "0"]})
    selected_real = [_record(idx, 0.1 + idx * 0.01) for idx in range(10)]
    selected_test = [_record(10 + idx, 100.1 + idx * 0.01) for idx in range(10)]
    extra_test = [_record(20 + idx, 100.3 + idx * 0.01) for idx in range(10)]
    pool_records = selected_real + selected_test + extra_test
    selected_records = selected_real + selected_test
    exact_records = [
        {"candidate_id": idx, "pareto_util_proxy_obj": 0.1}
        for idx in range(20)
    ]
    surrogate_records = [
        {"candidate_id": idx, "s_preselect_stage_b": 0.9}
        for idx in range(30)
    ]

    _, final_records, report = apply_direct_dcr_repair_v6(
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
        candidate_neighbors=3,
    )

    final_ids = {record["candidate_id"] for record in final_records}
    assert report["applied"] is True
    assert report["version"] == "direct_dcr_repair_v6"
    assert report["base_strategy"] == "in_band_utility_positive_v4"
    assert report["intermediate_candidate_count"] == 0
    assert report["candidate_full_eval_used"] is False
    assert report["target_band"] == [0.45, 0.55]
    assert 0.45 <= report["final_dcr_estimate"] <= 0.55
    assert final_ids & {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}
