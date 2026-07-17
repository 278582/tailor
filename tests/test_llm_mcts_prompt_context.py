from __future__ import annotations

from llm_mcts_tool.prompt_context import build_init_prompt_context, build_refine_prompt_context
from llm_mcts_tool.prompt_templates import init_prompt, refine_prompt
from llm_mcts_tool.strategy import StrategyTheta
from llm_mcts_tool.strategy import theta_size_bounds
from llm_mcts_tool.tree import MCTSNode


def test_init_prompt_uses_dynamic_theta_size_guidance_for_adult_like_schema() -> None:
    features = [f"f{i}" for i in range(14)]
    schema_card = {
        "dataset": "adult",
        "target_column": "target",
        "column_order": [*features, "target"],
        "columns": {column: {"is_target": False, "type": "numerical"} for column in features}
        | {"target": {"is_target": True, "type": "categorical"}},
    }
    dataset_context = {
        "dataset": "adult",
        "target_column": "target",
        "columns": {
            "feature": features,
            "privacy_configured": features[:5],
            "privacy_domain": features[5:7],
        },
        "theta_guidance": {
            "shape_priority": features,
            "trend_priority": features,
            "privacy_priority": features,
            "utility_priority": features,
        },
        "pair_priors": [],
        "risks": {"shape": [], "trend": [], "privacy": [], "utility": []},
    }

    context = build_init_prompt_context(schema_card=schema_card, dataset_context=dataset_context)
    prompt = init_prompt(context, n_init=6)

    assert context["theta_size_guidance"]["col_1ds"]["min"] == 4
    assert context["theta_size_guidance"]["col_1ds"]["max"] == 6
    assert context["theta_size_guidance"]["col_ps"]["min"] == 7
    assert context["theta_size_guidance"]["col_ps"]["max"] == 12
    assert "col_1ds target: 4-6" in prompt
    assert "col_ps target: 7-12" in prompt
    assert "col_2ds target: 4-7" in prompt


def test_theta_size_bounds_scale_with_feature_count_without_using_all_privacy_columns() -> None:
    small = theta_size_bounds(4)
    adult = theta_size_bounds(14)
    wide = theta_size_bounds(100)

    assert small["col_1ds"]["max"] < adult["col_1ds"]["max"] < wide["col_1ds"]["max"]
    assert small["col_2ds"]["max"] < adult["col_2ds"]["max"] <= wide["col_2ds"]["max"]
    assert small["col_ps"]["max"] < adult["col_ps"]["max"] < wide["col_ps"]["max"]
    assert adult["col_ps"]["max"] == 12
    assert wide["col_ps"]["max"] == 64
    assert adult["col_ps"]["max"] < 14
    assert wide["col_ps"]["max"] < 100


def test_refine_prompt_compacts_historical_actions_without_null_fields() -> None:
    features = ["age", "workclass", "education", "hours.per.week"]
    schema_card = {
        "dataset": "adult",
        "target_column": "income",
        "column_order": [*features, "income"],
        "columns": {column: {"is_target": False, "type": "numerical"} for column in features}
        | {"income": {"is_target": True, "type": "categorical"}},
    }
    dataset_context = {
        "dataset": "adult",
        "target_column": "income",
        "columns": {
            "feature": features,
            "privacy_configured": ["age"],
            "privacy_domain": [],
        },
        "theta_guidance": {
            "shape_priority": features,
            "trend_priority": features,
            "privacy_priority": features,
            "utility_priority": features,
        },
        "pair_priors": [],
        "risks": {"shape": [], "trend": [], "privacy": [], "utility": []},
    }
    node = MCTSNode(
        node_id="n_000001",
        theta_id="theta_1",
        theta=StrategyTheta(
            col_1ds=("age", "workclass"),
            col_2ds=("age", "education"),
            col_ps=("age",),
            col_u="age",
        ),
        parent_id="root",
        children_ids=[],
        depth=1,
        Q_self=0.8917170126873918,
        Q=0.8917170126873918,
        N=1,
        p=0.79,
        max_child_Q=0.0,
        is_leaf=True,
        rollout_id="theta_1",
        reward_available=True,
        guard_pass=True,
        search_objectives={"F_1D_theta": 0.9921974321788092},
        audit_metrics={"metric_reward": 0.8917170126873918},
        actions=[
            {"type": "add_col_2d", "column": None, "old": None, "new": "education"},
            {"type": "replace_col_u", "column": None, "old": "hours.per.week", "new": "age"},
        ],
        action_validation={
            "ok": True,
            "errors": [],
            "warnings": [],
            "no_ops": [],
            "applied": [{"index": 0, "type": "add_col_2d", "old": None, "new": "education"}],
            "result_theta": {"col_2ds": ["age", "education"]},
            "theta_source": "payload",
            "theta_changed_after_repair": False,
            "actions_match_theta": True,
        },
    )

    context = build_refine_prompt_context(
        node=node,
        parent=None,
        siblings=[],
        archive=[],
        schema_card=schema_card,
        dataset_context=dataset_context,
    )
    prompt = refine_prompt(context, n_expand=4)

    assert set(context) >= {
        "dataset_brief",
        "current_state",
        "feedback_to_fix",
        "diversity_context",
        "constraints",
    }
    assert context["current_state"]["previous_actions"] == [
        {"type": "add_col_2d", "new": "education"},
        {"type": "replace_col_u", "old": "hours.per.week", "new": "age"},
    ]
    assert "result_theta" not in context["current_state"]["action_validation"]
    assert "current_node" not in context
    assert "sibling_nodes" not in context
    assert "archive_top" not in context
    assert '"column": null' not in prompt
    assert '"old": null' not in prompt
    assert '"F_1D_theta": 0.9922' in prompt
    assert "# 1. Dataset Brief" in prompt
    assert "# 5. Constraints" in prompt
    edit_policy = prompt.split("Edit policy:\n", 1)[1].split("\nPrior score calibration:", 1)[0]
    assert "\n\n" not in edit_policy


def test_refine_prompt_can_omit_dataset_priors_without_removing_schema_guidance() -> None:
    features = ["age", "workclass", "education", "hours.per.week"]
    schema_card = {
        "dataset": "adult",
        "target_column": "income",
        "column_order": [*features, "income"],
        "columns": {column: {"is_target": False, "type": "numerical"} for column in features}
        | {"income": {"is_target": True, "type": "categorical"}},
    }
    dataset_context = {
        "dataset": "adult",
        "target_column": "income",
        "columns": {
            "feature": features,
            "privacy_configured": ["age"],
            "privacy_domain": ["education"],
        },
        "theta_guidance": {
            "shape_priority": features,
            "trend_priority": features,
            "privacy_priority": features,
            "utility_priority": features,
        },
        "pair_priors": [["age", "education", "age and education are a high-value trend pair"]],
        "risks": {"shape": [], "trend": [], "privacy": [], "utility": []},
    }
    node = MCTSNode(
        node_id="n_000001",
        theta_id="theta_1",
        theta=StrategyTheta(
            col_1ds=("age", "workclass"),
            col_2ds=("age", "education"),
            col_ps=("age",),
            col_u="age",
        ),
        parent_id="root",
        children_ids=[],
        depth=1,
        Q_self=0.5,
        Q=0.5,
        N=1,
        p=0.5,
        max_child_Q=0.0,
        is_leaf=True,
        rollout_id="theta_1",
        reward_available=True,
        guard_pass=True,
    )

    default_context = build_refine_prompt_context(
        node=node,
        parent=None,
        siblings=[],
        archive=[],
        schema_card=schema_card,
        dataset_context=dataset_context,
    )
    default_prompt = refine_prompt(default_context, n_expand=4)
    disabled_context = build_refine_prompt_context(
        node=node,
        parent=None,
        siblings=[],
        archive=[],
        schema_card=schema_card,
        dataset_context=dataset_context,
        include_dataset_priors=False,
    )
    disabled_prompt = refine_prompt(disabled_context, n_expand=4)

    assert "theta_guidance" in default_context["dataset_brief"]
    assert "pair_priors" in default_context["dataset_brief"]
    assert "dataset_brief.pair_priors" in default_prompt
    assert "theta_guidance" not in disabled_context["dataset_brief"]
    assert "pair_priors" not in disabled_context["dataset_brief"]
    assert disabled_context["dataset_brief"]["columns"]["feature"] == features
    assert "theta_size_guidance" in disabled_context["dataset_brief"]
    assert '"theta_guidance"' not in disabled_prompt
    assert '"pair_priors"' not in disabled_prompt
    assert "dataset_brief.pair_priors" not in disabled_prompt
    assert "available feature columns" in disabled_prompt
    default_edit_policy = default_prompt.split("Edit policy:\n", 1)[1].split("\nPrior score calibration:", 1)[0]
    disabled_edit_policy = disabled_prompt.split("Edit policy:\n", 1)[1].split("\nPrior score calibration:", 1)[0]
    assert "\n\n" not in default_edit_policy
    assert "\n\n" not in disabled_edit_policy
