from __future__ import annotations

import pandas as pd
import pytest

from post_selection_tool.config import CoreSelectionConfig
from post_selection_tool.exact_score import compute_global_exact_scores
from post_selection_tool.fidelity_ceiling import build_fidelity_ceiling
from post_selection_tool.preselect import build_preselect_gate_report
from post_selection_tool.reward_candidate_v2 import refine_selection_for_reward_v2
from post_selection_tool.selector import ParetoSelector
from post_selection_tool.state import ArtifactPaths, SelectionState


def _toy_cards() -> tuple[dict, dict]:
    schema_card = {
        "dataset": "toy",
        "column_order": ["x", "cat", "target"],
        "target_column": "target",
        "columns": {
            "x": {"type": "numerical", "is_target": False},
            "cat": {"type": "categorical", "is_target": False, "legal_values": ["A", "B", "C"]},
            "target": {"type": "categorical", "is_target": True, "legal_values": ["0", "1"]},
        },
    }
    stats_card = {"numeric_bins": {"x": [0.0, 1.5, 3.0, 4.5, 6.0]}}
    return schema_card, stats_card


def _toy_selector() -> ParetoSelector:
    train_df = pd.DataFrame(
        {
            "x": [0.2, 0.8, 1.6, 2.4, 3.2, 3.8, 4.6, 5.2],
            "cat": ["A", "A", "B", "B", "C", "C", "A", "B"],
            "target": ["0", "0", "1", "1", "0", "1", "0", "1"],
        }
    )
    holdout_df = train_df.iloc[[1, 3, 5, 7]].reset_index(drop=True)
    schema_card, stats_card = _toy_cards()
    selector = ParetoSelector(
        train_df=train_df,
        holdout_df=holdout_df,
        schema_card=schema_card,
        stats_card=stats_card,
        seed=7,
        source="test",
        nn_device="cpu",
        density_reference_size=0,
        fidelity_1d_columns=["x", "cat"],
        fidelity_2d_anchor_columns=["x", "cat"],
        privacy_columns=["x", "cat"],
        max_theta_pairs=1,
        high_cardinality_enabled=False,
    )
    selector.progress_enabled = False
    return selector


def _candidate_records() -> list[dict]:
    rows = [
        {"x": 0.3, "cat": "A", "target": "0"},
        {"x": 1.9, "cat": "B", "target": "1"},
        {"x": 3.4, "cat": "C", "target": "0"},
        {"x": 4.8, "cat": "A", "target": "1"},
    ]
    return [{"candidate_id": idx, "row": row} for idx, row in enumerate(rows)]


def test_compute_exact_scores_keeps_raw_and_robust_calibrated_objectives() -> None:
    selector = _toy_selector()
    records = _candidate_records()
    d_cur_df = selector.train_df.iloc[:3].reset_index(drop=True)

    exact_records, baselines = selector.compute_exact_scores(d_cur_df, records)

    assert len(exact_records) == len(records)
    assert baselines["fidelity_objective_scaling"] == "candidate_pool_robust_quantile"
    assert baselines["privacy_objective_scaling"] == "candidate_pool_robust_quantile"
    assert baselines["objective_calibration"]["basis"] == "same_preselected_candidate_pool"
    for record in exact_records:
        assert "pareto_fid_1d_obj_raw" in record
        assert "pareto_priv_obj_raw" in record
        assert 0.0 <= record["pareto_fid_1d_obj"] <= 1.0
        assert 0.0 <= record["pareto_fid_2d_obj"] <= 1.0
        assert 0.0 <= record["pareto_priv_obj"] <= 1.0


def test_four_objective_scoring_records_complete_timing(tmp_path) -> None:
    selector = _toy_selector()
    records = _candidate_records()
    paths = ArtifactPaths(
        artifact_dir=tmp_path,
        input_dir=tmp_path / "input",
        cards_dir=tmp_path / "cards",
        validation_dir=tmp_path / "validation",
        selection_dir=tmp_path / "selection",
        versions_dir=tmp_path / "versions",
        report_dir=tmp_path / "report",
    )
    state = SelectionState(
        config=CoreSelectionConfig(seed=selector.seed),
        paths=paths,
        dataset_ctx=None,
        synthetic_csv=tmp_path / "synthetic.csv",
        train_df=selector.train_df.copy(),
        holdout_df=selector.holdout_df.copy(),
        test_df=selector.holdout_df.copy(),
        synthetic_df=pd.DataFrame([record["row"] for record in records]),
        selector=selector,
        d_cur_df=selector.train_df.iloc[:3].reset_index(drop=True),
        preselected_valid=records,
        effective_keep_k=2,
    )

    state = compute_global_exact_scores(state)
    state = build_fidelity_ceiling(state)

    timing = state.timing_report["objective_scoring"]
    assert timing["complete"] is True
    assert timing["candidate_rows"] == len(records)
    assert timing["exact_fidelity_privacy_seconds"] >= 0.0
    assert timing["utility_proxy_seconds"] >= 0.0
    assert timing["utility_postprocess_seconds"] >= 0.0
    assert timing["four_objective_total_seconds"] == pytest.approx(
        timing["exact_fidelity_privacy_seconds"]
        + timing["utility_proxy_seconds"]
        + timing["utility_postprocess_seconds"]
    )
    assert "fidelity_ceiling_subset_construction" in timing["excludes"]
    assert all("pareto_util_proxy_obj" in record for record in state.global_exact_records)


def test_select_keep_uses_exact_non_dominated_sort_report() -> None:
    selector = _toy_selector()
    records = _candidate_records()
    exact_records = [
        {
            "candidate_id": idx,
            "pareto_fid_1d_obj": fid1,
            "pareto_fid_2d_obj": fid2,
            "pareto_fid_obj": 0.5 * (fid1 + fid2),
            "pareto_priv_obj": priv,
            "pareto_util_proxy_obj": util,
            "privacy_score_selected": priv,
            "fid_marginal": 0.0,
        }
        for idx, (fid1, fid2, priv, util) in enumerate(
            [(0.9, 0.8, 0.2, 0.1), (0.4, 0.9, 0.8, 0.4), (0.7, 0.5, 0.5, 0.9), (0.2, 0.2, 1.0, 1.0)]
        )
    ]

    keep_df, keep_records, report = selector.select_keep(records, [], exact_records, keep_k=2)

    assert len(keep_df) == 2
    assert len(keep_records) == 2
    assert report["non_dominated_sort"]["exact"] is True
    assert report["non_dominated_sort"]["approximate"] is False
    assert report["front_component_mode"] == "deterministic_exact_nsga_front_rank"


def test_select_keep_large_4d_pool_uses_calibrated_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import post_selection_tool.pareto_selection as pareto_selection_module

    monkeypatch.setattr(pareto_selection_module, "PARETO_EXACT_NSGA_MAX_ROWS", 3)
    selector = _toy_selector()
    records = [
        {"candidate_id": idx, "row": {"x": float(idx), "cat": ["A", "B", "C"][idx % 3], "target": str(idx % 2)}}
        for idx in range(6)
    ]
    exact_records = [
        {
            "candidate_id": idx,
            "pareto_fid_1d_obj": fid1,
            "pareto_fid_2d_obj": fid2,
            "pareto_fid_obj": 0.5 * (fid1 + fid2),
            "pareto_priv_obj": priv,
            "pareto_util_proxy_obj": util,
            "privacy_score_selected": priv,
            "fid_marginal": 0.0,
        }
        for idx, (fid1, fid2, priv, util) in enumerate(
            [
                (0.10, 0.90, 0.20, 0.40),
                (0.40, 0.70, 0.50, 0.30),
                (0.60, 0.20, 0.80, 0.50),
                (0.90, 0.10, 0.30, 0.70),
                (0.30, 0.50, 0.90, 0.20),
                (0.70, 0.40, 0.40, 0.90),
            ]
        )
    ]

    keep_df, keep_records, report = selector.select_keep(records, [], exact_records, keep_k=4)

    assert len(keep_df) == 4
    assert len(keep_records) == 4
    assert report["non_dominated_sort"]["exact"] is False
    assert report["non_dominated_sort"]["approximate"] is True
    assert report["front_component_mode"] == "large_pool_calibrated_objective"


def test_preselect_gate_report_uses_configurable_thresholds() -> None:
    report = build_preselect_gate_report(
        raw_metrics={"fidelity": 1.0, "trend": 1.0, "dcr": 0.75, "privacy": 0.25},
        candidate_metrics={"fidelity": 0.995, "trend": 0.996, "dcr": 0.70, "privacy": 0.30},
        baseline_metrics={"fidelity": 0.994, "trend": 0.995, "dcr": 0.72, "privacy": 0.28},
        fidelity_max_drop=0.01,
        trend_max_drop=0.01,
        dcr_min_gain=0.02,
        candidate_vs_baseline_max_drop=0.002,
        candidate_vs_baseline_min_dcr_gain=0.01,
    )

    assert report["thresholds"]["candidate_vs_baseline_dcr_min_gain"] == pytest.approx(0.01)
    assert report["candidate"]["pass"] is True
    assert report["candidate"]["beats_baseline"] is True
    assert report["fallback_applied"] is False


def test_exact_floor_repair_reports_floor_status() -> None:
    selector = _toy_selector()
    records = _candidate_records()
    exact_records, _ = selector.compute_exact_scores(selector.train_df.iloc[:3].reset_index(drop=True), records)

    selected_indices, report = selector._apply_exact_floor_repair(
        preselected_records=records,
        exact_records=exact_records,
        selected_indices=[0, 1],
        keep_k=2,
        floor_reference={"name": "toy_floor", "fidelity_1d": 0.0, "fidelity_2d": 0.0},
    )

    assert len(selected_indices) == 2
    assert report["mode"] == "already_satisfied"
    assert report["reference_name"] == "toy_floor"
    assert report["satisfied"] is True


def test_disabled_fidelity_ceiling_keeps_utility_proxy_without_floor_reference(tmp_path) -> None:
    selector = _toy_selector()
    records = _candidate_records()
    exact_records, _ = selector.compute_exact_scores(selector.train_df.iloc[:3].reset_index(drop=True), records)
    paths = ArtifactPaths(
        artifact_dir=tmp_path,
        input_dir=tmp_path / "input",
        cards_dir=tmp_path / "cards",
        validation_dir=tmp_path / "validation",
        selection_dir=tmp_path / "selection",
        versions_dir=tmp_path / "versions",
        report_dir=tmp_path / "report",
    )
    state = SelectionState(
        config=CoreSelectionConfig(seed=selector.seed, fidelity_ceiling_enabled=False),
        paths=paths,
        dataset_ctx=None,
        synthetic_csv=tmp_path / "synthetic.csv",
        train_df=selector.train_df.copy(),
        holdout_df=selector.holdout_df.copy(),
        test_df=selector.holdout_df.copy(),
        synthetic_df=pd.DataFrame([record["row"] for record in records]),
        selector=selector,
        preselected_valid=records,
        global_exact_records=exact_records,
        effective_keep_k=2,
    )

    state = build_fidelity_ceiling(state)

    assert state.floor_reference is None
    assert state.fidelity_ceiling_records == []
    assert state.fidelity_ceiling_report["enabled"] is False
    assert state.fidelity_ceiling_report["reason"] == "disabled_by_config"
    assert state.utility_proxy_merge_report["matched_rows"] == len(exact_records)
    assert all("pareto_util_proxy_obj" in record for record in state.global_exact_records)
    assert all(record["utility_anchor_member"] is False for record in state.global_exact_records)


def _reward_v2_records(
    *,
    add_quality: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    preselected_records = [
        {"candidate_id": idx, "row": {"x": float(idx), "cat": "A", "target": "0"}}
        for idx in range(8)
    ]
    exact_records: list[dict] = []
    for idx in range(8):
        selected_like = idx < 4
        quality = 0.86 if selected_like else float(add_quality)
        exact_records.append(
            {
                "candidate_id": idx,
                "pareto_fid_1d_obj": quality,
                "pareto_fid_2d_obj": quality,
                "pareto_priv_obj": quality,
                "pareto_util_proxy_obj": quality,
                "holdout_gap": 0.35 if selected_like else -0.35,
            }
        )
    selected_records = preselected_records[:4]
    return preselected_records, exact_records, selected_records


def test_reward_candidate_v2_swaps_toward_balanced_dcr_proxy() -> None:
    preselected_records, exact_records, selected_records = _reward_v2_records(add_quality=0.96)

    keep_df, keep_records, report = refine_selection_for_reward_v2(
        preselected_records=preselected_records,
        exact_records=exact_records,
        selected_records=selected_records,
        keep_k=4,
        column_order=["x", "cat", "target"],
        max_swap_fraction=1.0,
        fidelity_floor_eps=0.02,
        utility_floor_eps=0.02,
    )

    kept_ids = {record["candidate_id"] for record in keep_records}
    assert len(keep_df) == 4
    assert len(keep_records) == 4
    assert report["applied"] is True
    assert report["best"]["stats"]["dcr_proxy"] == pytest.approx(0.5)
    assert kept_ids & {4, 5, 6, 7}
    assert kept_ids != {0, 1, 2, 3}


def test_reward_candidate_v2_keeps_selection_without_feasible_proxy_gain() -> None:
    preselected_records, exact_records, selected_records = _reward_v2_records(add_quality=0.10)

    keep_df, keep_records, report = refine_selection_for_reward_v2(
        preselected_records=preselected_records,
        exact_records=exact_records,
        selected_records=selected_records,
        keep_k=4,
        column_order=["x", "cat", "target"],
        max_swap_fraction=1.0,
        fidelity_floor_eps=0.0,
        utility_floor_eps=0.0,
    )

    assert len(keep_df) == 4
    assert [record["candidate_id"] for record in keep_records] == [0, 1, 2, 3]
    assert report["applied"] is False
    assert report["reason"] == "no_feasible_proxy_improvement"
