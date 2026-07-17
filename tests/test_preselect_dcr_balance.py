from __future__ import annotations

from post_selection_tool.preselect_dcr_balance import rebalance_preselected_for_dcr_surrogate


def _record(idx: int) -> dict:
    return {"candidate_id": idx, "row": {"x": idx, "target": "0"}}


def _surrogate(idx: int, *, holdout_gap: float, quality: float) -> dict:
    return {
        "candidate_id": idx,
        "holdout_gap": holdout_gap,
        "s_preselect_stage_b": quality,
        "s_pareto_fid_1d_sur": quality,
        "s_pareto_fid_2d_sur": quality,
        "s_preselect_support_tiebreak": quality,
        "s_preselect_priv_tiebreak": quality,
    }


def test_rebalance_preselected_for_dcr_surrogate_adds_minority_rows() -> None:
    pool_records = [_record(idx) for idx in range(10)]
    surrogate_records = [
        _surrogate(idx, holdout_gap=0.5, quality=0.9 - 0.01 * idx)
        for idx in range(7)
    ] + [
        _surrogate(7, holdout_gap=-0.5, quality=0.95),
        _surrogate(8, holdout_gap=-0.5, quality=0.94),
        _surrogate(9, holdout_gap=-0.5, quality=0.93),
    ]
    selected_records = pool_records[:6]
    selected_surrogates = surrogate_records[:6]

    final_records, final_surrogates, report = rebalance_preselected_for_dcr_surrogate(
        pool_records=pool_records,
        surrogate_records=surrogate_records,
        selected_records=selected_records,
        selected_surrogates=selected_surrogates,
        target_fraction=0.5,
        max_exchange_fraction=0.5,
    )

    final_ids = {record["candidate_id"] for record in final_records}
    assert len(final_records) == 6
    assert len(final_surrogates) == 6
    assert report["applied"] is True
    assert report["exchange_rows"] == 3
    assert final_ids == {0, 1, 2, 7, 8, 9}
    assert report["selected_after"]["minority_rows"] == 3


def test_rebalance_preselected_for_dcr_surrogate_keeps_already_balanced_selection() -> None:
    pool_records = [_record(idx) for idx in range(6)]
    surrogate_records = [
        _surrogate(0, holdout_gap=0.5, quality=0.9),
        _surrogate(1, holdout_gap=0.5, quality=0.8),
        _surrogate(2, holdout_gap=0.5, quality=0.7),
        _surrogate(3, holdout_gap=-0.5, quality=0.9),
        _surrogate(4, holdout_gap=-0.5, quality=0.8),
        _surrogate(5, holdout_gap=-0.5, quality=0.7),
    ]
    selected_records = [pool_records[idx] for idx in [0, 1, 3, 4]]
    selected_surrogates = [surrogate_records[idx] for idx in [0, 1, 3, 4]]

    final_records, final_surrogates, report = rebalance_preselected_for_dcr_surrogate(
        pool_records=pool_records,
        surrogate_records=surrogate_records,
        selected_records=selected_records,
        selected_surrogates=selected_surrogates,
        target_fraction=0.5,
        max_exchange_fraction=0.5,
    )

    assert [record["candidate_id"] for record in final_records] == [0, 1, 3, 4]
    assert [record["candidate_id"] for record in final_surrogates] == [0, 1, 3, 4]
    assert report["applied"] is False
    assert report["reason"] == "already_balanced_or_no_exchange"
