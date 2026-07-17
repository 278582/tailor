from __future__ import annotations

import pandas as pd

from post_selection_tool.direct_dcr_repair_v5 import apply_direct_dcr_repair_v5


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


def test_direct_dcr_repair_v5_skips_when_base_dcr_is_inside_target_band() -> None:
    train_df = pd.DataFrame({"x": [0.0, 10.0], "target": ["0", "0"]})
    test_df = pd.DataFrame({"x": [100.0, 110.0], "target": ["0", "0"]})
    pool_records = [
        _record(0, 0.1),
        _record(1, 0.2),
        _record(2, 100.1),
        _record(3, 100.2),
    ]
    selected_records = list(pool_records)

    final_df, final_records, report = apply_direct_dcr_repair_v5(
        pool_records=pool_records,
        selected_records=selected_records,
        exact_records=[],
        surrogate_records=[],
        train_df=train_df,
        test_df=test_df,
        schema_card=_schema_card(),
        column_order=["x", "target"],
        target_margin=0.05,
        max_swap_fraction=1.0,
    )

    assert len(final_df) == 4
    assert [record["candidate_id"] for record in final_records] == [0, 1, 2, 3]
    assert report["applied"] is False
    assert report["reason"] == "base_dcr_within_target_band"
    assert report["base_dcr_estimate"] == 0.5
