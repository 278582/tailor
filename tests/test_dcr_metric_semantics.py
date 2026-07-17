from __future__ import annotations

from pathlib import Path

import pytest

from llm_mcts_tool.strategy import StrategyTheta
from llm_mcts_tool.v2_pipeline import (
    DCR_PROMPT_SEMANTICS,
    SNode,
    SourceInfo,
    ThetaNode,
    V2MCTSConfig,
    _compact_dcr_for_prompt,
    _compact_feedback_for_prompt,
    _compact_source_profile_for_prompt,
    _diagnosis_theta_node_for_prompt,
    _metrics_4d_reward_summary,
    _render_prompt,
    _sanitize_llm_dcr_text,
    _summarize_s_nodes_for_prompt,
)
from metric_tool.evaluator import build_dcr_balance_summary, build_metric_directions


def test_dcr_balance_summary_uses_distance_from_half() -> None:
    summary = build_dcr_balance_summary(0.625)

    assert summary["raw_dcr_real_closer_rate"] == pytest.approx(0.625)
    assert summary["target_raw_dcr"] == 0.5
    assert summary["dcr_balance_error_abs"] == pytest.approx(0.125)
    assert summary["dcr_privacy_reward"] == pytest.approx(0.875)
    assert "best near 0.5" in summary["dcr_semantics"]


def test_metric_directions_distinguish_utility_metrics() -> None:
    assert build_metric_directions("roc_auc")["utility_exact_raw_direction"] == "higher_better"
    assert build_metric_directions("RMSE")["utility_exact_raw_direction"] == "lower_better"
    assert build_metric_directions("roc_auc")["shape"] == "higher_better"
    assert build_metric_directions("roc_auc")["trend"] == "higher_better"
    assert build_metric_directions("roc_auc")["raw_dcr_real_closer_rate"] == "target_0.5"


def _count_key(value, key: str) -> int:
    if isinstance(value, dict):
        return int(key in value) + sum(_count_key(item, key) for item in value.values())
    if isinstance(value, list):
        return sum(_count_key(item, key) for item in value)
    return 0


def _theta_node() -> ThetaNode:
    return ThetaNode(
        node_id="n_000001",
        s_id="s_000000",
        theta=StrategyTheta(
            col_1ds=("income", "age"),
            col_2ds=("income", "age"),
            col_ps=("income", "age"),
            col_u="age",
        ),
        theta_id="theta_1",
        parent_node_id=None,
        actions=[],
        proposal_action_validation={},
        llm_score=0.5,
        reason="fixture",
        exact_reward=0.9,
        exact_reward_available=True,
        search_reward=0.9,
        search_reward_available=True,
        reward_type="exact",
        status="success",
        audit_metrics={
            "shape_global": 0.97,
            "trend_global": 0.96,
            "dcr": 0.625,
            "dcr_privacy": 0.875,
            "utility_exact": 0.82,
            "metric_reward": 0.9,
        },
        feedback={
            "shape_weak_columns": [{"column": "age", "score": 0.8}],
            "trend_weak_pairs": [{"left": "age", "right": "income", "score": 0.7}],
            "privacy_summary": {"dcr": 0.625, "dcr_privacy": 0.875},
            "diagnostics": {
                "dcr_quantiles": {
                    "available": True,
                    "dcr_real": {"q50": 0.2},
                    "dcr_test": {"q50": 0.25},
                    "real_closer_rate": 0.625,
                    "dcr_privacy_reward": 0.875,
                },
                "utility_xgb_feature_importance": [{"feature": "age", "importance": 0.4}],
            },
        },
    )


def test_v2_prompt_metrics_use_dcr_privacy_reward_not_balance_block() -> None:
    compact = _metrics_4d_reward_summary(
        {
            "shape_global": 0.97,
            "trend_global": 0.96,
            "dcr": 0.625,
            "dcr_privacy": 0.875,
            "utility_exact": 0.82,
            "metric_reward": 0.9,
        }
    )

    assert "dcr" not in compact
    assert "dcr_privacy" not in compact
    assert "dcr_balance" not in compact
    assert compact["dcr_privacy_reward"] == pytest.approx(0.875)


def test_v2_prompt_keeps_dcr_balance_only_inside_quantile_diagnostics() -> None:
    node = _theta_node()

    feedback = _compact_feedback_for_prompt(node.feedback)
    diagnosis = _diagnosis_theta_node_for_prompt(node)

    assert "dcr_balance" not in feedback
    assert feedback["dcr_privacy_reward"] == pytest.approx(0.875)
    assert feedback["dcr_quantiles"]["dcr_balance"]["raw_dcr_real_closer_rate"] == pytest.approx(0.625)
    assert "semantics" not in feedback["dcr_quantiles"]
    assert "semantics" not in feedback["dcr_quantiles"]["dcr_balance"]
    assert _count_key(feedback, "dcr_balance") == 1

    assert "dcr_balance" not in diagnosis["metrics_4d_reward"]
    assert "dcr_balance" not in diagnosis["weakness"]
    assert diagnosis["metrics_4d_reward"]["dcr_privacy_reward"] == pytest.approx(0.875)
    assert diagnosis["weakness"]["dcr_privacy_reward"] == pytest.approx(0.875)
    assert _count_key(diagnosis, "dcr_balance") == 1


def test_v2_source_profile_metrics_use_dcr_privacy_reward_and_quantiles_keep_balance() -> None:
    profile = _compact_source_profile_for_prompt(
        "tabdiff",
        SourceInfo(source_id="tabdiff", path=Path("sample.csv"), rows=100, columns=["age"]),
        {
            "metrics_mean": {"shape": 0.9, "trend": 0.8, "dcr": 0.625, "dcr_privacy": 0.875},
            "dcr_quantiles_mean": {
                "available": True,
                "dcr_real": {"q50": 0.2},
                "dcr_test": {"q50": 0.25},
                "real_closer_rate": 0.625,
                "dcr_privacy_reward": 0.875,
            },
        },
    )

    assert "dcr_balance" not in profile["metrics_mean"]
    assert profile["metrics_mean"]["dcr_privacy_reward"] == pytest.approx(0.875)
    assert profile["dcr_quantiles"]["dcr_balance"]["raw_dcr_real_closer_rate"] == pytest.approx(0.625)
    assert _count_key(profile, "dcr_balance") == 1


def test_v2_existing_s_summary_avoids_aggregate_dcr_balance_duplication(tmp_path: Path) -> None:
    node = _theta_node()
    s_node = SNode(
        s_id="s_000000",
        pool_units=[{"source_id": "tabdiff", "multiplier": 2}, {"source_id": "tabsyn", "multiplier": 2}],
        synthetic_csv=tmp_path / "synthetic.csv",
        synthetic_row_map=tmp_path / "rows.jsonl",
        theta_node_ids=[node.node_id],
        best_reward=0.9,
    )

    summary = _summarize_s_nodes_for_prompt(s_nodes={s_node.s_id: s_node}, theta_nodes={node.node_id: node})[0]

    assert "dcr_balance" not in summary["aggregate_weakness"]
    assert "privacy_distribution" in summary["aggregate_weakness"]
    assert "dcr_balance" not in summary["best_theta"]["metrics_4d_reward"]
    assert summary["best_theta"]["metrics_4d_reward"]["dcr_privacy_reward"] == pytest.approx(0.875)


def test_v2_rendered_prompt_has_single_dcr_semantics_block() -> None:
    dcr_quantiles = _compact_dcr_for_prompt(
        {
            "available": True,
            "dcr_real": {"q50": 0.2},
            "dcr_test": {"q50": 0.25},
            "real_closer_rate": 0.625,
            "dcr_privacy_reward": 0.875,
        }
    )
    prompt = _render_prompt(
        V2MCTSConfig(),
        "v2_refine_node_prompt.j2",
        {
            "n_theta": 1,
            "dataset_brief": {},
            "dcr_balance_semantics": DCR_PROMPT_SEMANTICS,
            "real_utility_reference": {},
            "s_context": None,
            "current_node": {
                "metrics_4d_reward": {"dcr_privacy_reward": 0.875},
                "diagnostics": {"dcr_quantiles": dcr_quantiles},
            },
            "reference_nodes": [
                {
                    "metrics_4d_reward": {"dcr_privacy_reward": 0.875},
                    "diagnostics": {"dcr_quantiles": dcr_quantiles},
                }
            ],
        },
    )

    assert prompt.count('"raw_metric"') == 1
    assert prompt.count('"dcr_balance"') == 2
    assert prompt.count('"dcr_privacy_reward"') >= 3


def test_v2_current_prompt_templates_render_one_dcr_semantics_block() -> None:
    dcr_quantiles = _compact_dcr_for_prompt(
        {
            "available": True,
            "dcr_real": {"q50": 0.2},
            "dcr_test": {"q50": 0.25},
            "real_closer_rate": 0.625,
            "dcr_privacy_reward": 0.875,
        }
    )
    source_profile = _compact_source_profile_for_prompt(
        "tabdiff",
        SourceInfo(source_id="tabdiff", path=Path("sample.csv"), rows=100, columns=["age"]),
        {
            "metrics_mean": {"shape": 0.9, "trend": 0.8, "dcr": 0.625, "dcr_privacy": 0.875},
            "dcr_quantiles_mean": {
                "available": True,
                "dcr_real": {"q50": 0.2},
                "dcr_test": {"q50": 0.25},
                "real_closer_rate": 0.625,
                "dcr_privacy_reward": 0.875,
            },
        },
    )
    node_summary = {
        "metrics_4d_reward": {"shape_global": 0.9, "dcr_privacy_reward": 0.875},
        "diagnostics": {"dcr_quantiles": dcr_quantiles},
    }
    common = {"dcr_balance_semantics": DCR_PROMPT_SEMANTICS}
    cases = [
        (
            "v2_init_node_prompt.j2",
            {
                **common,
                "n_theta": 1,
                "dataset_brief": {},
                "real_utility_reference": {},
                "s_context": None,
                "seed_theta_examples": [],
            },
        ),
        (
            "v2_refine_node_prompt.j2",
            {
                **common,
                "n_theta": 1,
                "dataset_brief": {},
                "real_utility_reference": {},
                "s_context": None,
                "current_node": node_summary,
                "reference_nodes": [node_summary],
            },
        ),
        (
            "v2_init_node_diagnosis_prompt.j2",
            {
                **common,
                "real_utility_reference": {},
                "s_context": None,
                "theta_batch": [node_summary],
                "best_so_far": node_summary,
            },
        ),
        (
            "v2_refine_node_diagnosis_prompt.j2",
            {
                **common,
                "real_utility_reference": {},
                "s_context": None,
                "reference_nodes": [node_summary],
                "theta_batch": [node_summary],
            },
        ),
        (
            "v2_init_select_syn_prompt.j2",
            {
                **common,
                "n_pools": 1,
                "pool_multiplier": 4,
                "real_utility_reference": {},
                "source_models": [source_profile],
            },
        ),
        (
            "v2_refine_select_syn_prompt.j2",
            {
                **common,
                "pool_multiplier": 4,
                "real_utility_reference": {},
                "source_models": [source_profile],
                "search_summary": {"existing_s": []},
            },
        ),
        (
            "v2_source_profile_summary_prompt.j2",
            {
                **common,
                "dataset_brief": {},
                "source_profile": source_profile,
            },
        ),
    ]

    for template_name, payload in cases:
        prompt = _render_prompt(V2MCTSConfig(), template_name, payload)
        assert prompt.count('"raw_metric"') == 1, template_name


def test_llm_dcr_text_sanitizer_removes_monotonic_dcr_language() -> None:
    text = "Limiting objective is DCR (0.625); low DCR is the limiting factor."
    sanitized = _sanitize_llm_dcr_text(text, {"dcr": 0.625, "dcr_privacy": 0.875})

    assert "DCR (0.625)" not in sanitized
    assert "low DCR" not in sanitized
    assert "raw DCR real_closer_rate 0.625" in sanitized
    assert "DCR privacy reward 0.875" in sanitized


def test_prompt_templates_declare_metric_directions() -> None:
    template_dir = Path("prompt_pack/templates")
    required = [
        "v2_init_node_diagnosis_prompt.j2",
        "v2_refine_node_diagnosis_prompt.j2",
        "v2_select_syn_prompt.j2",
        "v2_source_profile_summary_prompt.j2",
    ]
    for name in required:
        text = (template_dir / name).read_text(encoding="utf-8")
        assert "shape and trend are higher-better" in text
        assert "roc_auc is higher-better, rmse is lower-better" in text
        assert "raw_dcr_real_closer_rate is best near 0.5" in text
        assert "DCR privacy near 0.5" not in text
