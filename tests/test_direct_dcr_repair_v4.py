from __future__ import annotations

import pandas as pd

from post_selection_tool.direct_dcr_repair_v4 import apply_direct_dcr_repair_v4


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


def test_direct_dcr_repair_v4_swaps_once_to_target_direction() -> None:
    train_df = pd.DataFrame({"x": [0.0, 10.0], "target": ["0", "0"]})
    test_df = pd.DataFrame({"x": [100.0, 110.0], "target": ["0", "0"]})
    pool_records = [
        _record(0, 0.1),
        _record(1, 0.2),
        _record(2, 0.3),
        _record(3, 0.4),
        _record(4, 100.1),
        _record(5, 100.2),
        _record(6, 100.3),
        _record(7, 100.4),
    ]
    selected_records = pool_records[:4]
    exact_records = [
        {"candidate_id": idx, "pareto_util_proxy_obj": 0.8}
        for idx in range(4)
    ]
    surrogate_records = [
        {"candidate_id": idx, "s_preselect_stage_b": 0.9}
        for idx in range(8)
    ]

    final_df, final_records, report = apply_direct_dcr_repair_v4(
        pool_records=pool_records,
        selected_records=selected_records,
        exact_records=exact_records,
        surrogate_records=surrogate_records,
        train_df=train_df,
        test_df=test_df,
        schema_card=_schema_card(),
        column_order=["x", "target"],
        target_margin=0.0,
        max_swap_fraction=1.0,
        candidate_neighbors=2,
    )

    final_ids = {record["candidate_id"] for record in final_records}
    assert len(final_df) == 4
    assert len(final_records) == 4
    assert report["applied"] is True
    assert report["intermediate_candidate_count"] == 0
    assert report["candidate_full_eval_used"] is False
    assert report["selected_swaps"] == 2
    assert report["base_dcr_estimate"] == 1.0
    assert report["final_dcr_estimate"] == 0.5
    assert final_ids != {0, 1, 2, 3}
    assert final_ids & {4, 5, 6, 7}
