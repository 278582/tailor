from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from llm_mcts_tool.strategy import (
    StrategyTheta,
    repair_theta_target_inclusive,
    theta_size_bounds_target_inclusive,
    validate_theta_target_inclusive,
)
from llm_mcts_tool.v2_pipeline import (
    DCR_PROMPT_SEMANTICS,
    ThetaNode,
    V2MCTSConfig,
    _archive_theta_summary,
    _compact_theta_node_for_prompt,
    _dataset_brief_for_prompt,
    _diagnosis_theta_node_for_prompt,
    _proposal_from_payload,
    _render_prompt,
    _seed_theta_examples_for_prompt,
)


DATASETS = ("adult", "beijing", "default", "diabetes", "magic", "news", "shoppers")


def _dataset_context(dataset_name: str) -> dict[str, Any]:
    path = Path("prompt_pack/dataset_contexts") / f"{dataset_name}.prompt_context.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_card_for_context(dataset_name: str, dataset_context: dict[str, Any]) -> dict[str, Any]:
    features = list(dataset_context.get("columns", {}).get("feature", []))
    target = str(dataset_context.get("target_column") or dataset_context.get("target") or "target")
    return {
        "dataset": dataset_name,
        "target_column": target,
        "column_order": [*features, target],
        "columns": {
            **{column: {"is_target": False, "type": "numerical"} for column in features},
            target: {"is_target": True, "type": "categorical"},
        },
    }


def _count_key(value: Any, key: str) -> int:
    if isinstance(value, dict):
        return int(key in value) + sum(_count_key(item, key) for item in value.values())
    if isinstance(value, list):
        return sum(_count_key(item, key) for item in value)
    return 0


def _theta_node_with_search_scores() -> ThetaNode:
    return ThetaNode(
        node_id="n_000001",
        s_id="s_000000",
        theta=StrategyTheta(
            col_1ds=("income", "age", "workclass", "education", "hours.per.week", "capital.gain", "sex", "race"),
            col_2ds=("income", "age", "workclass", "education", "hours.per.week", "capital.gain", "sex", "race"),
            col_ps=(
                "age",
                "workclass",
                "education",
                "hours.per.week",
                "capital.gain",
                "capital.loss",
                "sex",
                "race",
                "occupation",
                "relationship",
                "native.country",
            ),
            col_u="age",
        ),
        theta_id="theta_1",
        parent_node_id=None,
        actions=[{"type": "replace_col_u", "old": "education.num", "new": "age"}],
        proposal_action_validation={},
        llm_score=0.8,
        reason="fixture",
        exact_reward=0.9,
        exact_reward_available=True,
        search_reward=0.85,
        search_reward_available=True,
        reward_type="exact",
        status="success",
        search_objectives={
            "F_1D_theta": 0.99,
            "F_2D_theta": 0.98,
            "P_theta": 0.91,
            "P_theta_raw": 0.52,
            "U_proxy_theta": 0.7,
        },
        audit_metrics={
            "shape_global": 0.99,
            "trend_global": 0.98,
            "dcr": 0.52,
            "dcr_privacy": 0.98,
            "utility_exact": 0.82,
            "metric_reward": 0.9,
        },
        feedback={"diagnostics": {"metrics_4d": {"metric_reward": 0.9}}},
    )


def test_v2_prompt_bounds_and_seed_examples_match_contract_for_all_dataset_contexts() -> None:
    for dataset_name in DATASETS:
        dataset_context = _dataset_context(dataset_name)
        schema_card = _schema_card_for_context(dataset_name, dataset_context)
        n_columns = len(schema_card["column_order"])
        n_features = n_columns - 1
        bounds = theta_size_bounds_target_inclusive(n_columns)

        assert bounds["col_1ds"] == {
            "min": math.ceil(0.50 * n_columns),
            "max": n_columns,
            "rule": "50-100% of all columns; target allowed but not required",
        }
        assert bounds["col_2ds"] == {
            "min": math.ceil(0.50 * n_columns),
            "max": n_columns,
            "rule": "50-100% of all columns; target allowed but not required",
        }
        assert bounds["col_ps"] == {
            "min": math.ceil(0.80 * n_features),
            "max": n_features,
            "rule": "80-100% of non-target feature columns; target forbidden",
        }
        assert _dataset_brief_for_prompt(schema_card, dataset_context)["theta_size_bounds"] == bounds

        features = set(dataset_context.get("columns", {}).get("feature", []))
        known_columns = set(schema_card["column_order"])
        target = schema_card["target_column"]
        for raw_item in dataset_context.get("seed_theta_examples", []):
            theta = raw_item["theta"]
            repaired = repair_theta_target_inclusive(theta, schema_card)
            assert validate_theta_target_inclusive(repaired, schema_card).ok, (dataset_name, raw_item)
            assert set(repaired.col_1ds).issubset(known_columns), dataset_name
            assert set(repaired.col_2ds).issubset(known_columns), dataset_name
            assert target not in repaired.col_ps, dataset_name
            assert set(repaired.col_ps).issubset(features), dataset_name

        seeds = _seed_theta_examples_for_prompt(schema_card=schema_card, dataset_context=dataset_context, limit=4)
        assert seeds, dataset_name
        for item in seeds:
            theta = item["theta"]
            assert validate_theta_target_inclusive(theta, schema_card).ok, (dataset_name, item)
            for field_name in ("col_1ds", "col_2ds"):
                size = len(theta[field_name])
                assert bounds[field_name]["min"] <= size <= bounds[field_name]["max"], (dataset_name, field_name, size)
                assert set(theta[field_name]).issubset(known_columns), (dataset_name, field_name)
            assert bounds["col_ps"]["min"] <= len(theta["col_ps"]) <= bounds["col_ps"]["max"], dataset_name
            assert target not in theta["col_ps"], dataset_name
            assert set(theta["col_ps"]).issubset(features), dataset_name
            assert theta["col_u"] in features, (dataset_name, theta["col_u"])


def test_v2_rendered_prompts_do_not_use_search_scores() -> None:
    node = _theta_node_with_search_scores()
    compact = _compact_theta_node_for_prompt(node)
    diagnosis = _diagnosis_theta_node_for_prompt(node)
    archive = _archive_theta_summary(node, depth=1)

    assert _count_key(compact, "search_scores") == 0
    assert _count_key(diagnosis, "search_scores") == 0
    assert _count_key(archive, "search_scores") == 0

    config = V2MCTSConfig()
    dataset_context = _dataset_context("adult")
    schema_card = _schema_card_for_context("adult", dataset_context)
    seed_theta_examples = _seed_theta_examples_for_prompt(schema_card=schema_card, dataset_context=dataset_context)
    prompts = [
        _render_prompt(
            config,
            "v2_init_node_prompt.j2",
            {
                "n_theta": 1,
                "dataset_brief": _dataset_brief_for_prompt(schema_card, dataset_context),
                "dcr_balance_semantics": DCR_PROMPT_SEMANTICS,
                "real_utility_reference": {},
                "s_context": None,
                "seed_theta_examples": seed_theta_examples,
            },
        ),
        _render_prompt(
            config,
            "v2_refine_node_prompt.j2",
            {
                "n_theta": 1,
                "dataset_brief": _dataset_brief_for_prompt(schema_card, dataset_context),
                "dcr_balance_semantics": DCR_PROMPT_SEMANTICS,
                "real_utility_reference": {},
                "s_context": None,
                "current_node": compact,
                "reference_nodes": [compact],
            },
        ),
        _render_prompt(
            config,
            "v2_refine_node_diagnosis_prompt.j2",
            {
                "dcr_balance_semantics": DCR_PROMPT_SEMANTICS,
                "real_utility_reference": {},
                "s_context": None,
                "reference_nodes": [diagnosis],
                "theta_batch": [diagnosis],
            },
        ),
        _render_prompt(
            config,
            "v2_refine_select_syn_prompt.j2",
            {
                "pool_multiplier": 4,
                "real_utility_reference": {},
                "dcr_balance_semantics": DCR_PROMPT_SEMANTICS,
                "source_models": [],
                "search_summary": {"archive": [archive], "existing_s": []},
            },
        ),
    ]

    assert "Seed theta examples" in prompts[0]
    assert "target may appear in col_1ds and col_2ds, but is not required." in prompts[0]
    assert "target may appear in col_1ds and col_2ds, but is not required." in prompts[1]
    assert "target must NOT appear in col_ps." in prompts[0]
    assert "target must NOT appear in col_ps." in prompts[1]
    for prompt in prompts:
        assert '"search_scores"' not in prompt


def test_v2_proposal_repair_removes_target_from_col_ps() -> None:
    dataset_context = _dataset_context("magic")
    schema_card = _schema_card_for_context("magic", dataset_context)
    target = schema_card["target_column"]
    features = list(dataset_context["columns"]["feature"])
    payload = {
        "theta": {
            "col_1ds": [*features[:5]],
            "col_2ds": [*features[:5]],
            "col_ps": [target, *features[:8]],
            "col_u": features[0],
        },
        "reason": "fixture",
    }

    proposal = _proposal_from_payload(payload, schema_card=schema_card, seed=7)

    assert proposal is not None
    assert validate_theta_target_inclusive(proposal.theta, schema_card).ok
    assert target not in proposal.theta.col_1ds
    assert target not in proposal.theta.col_2ds
    assert target not in proposal.theta.col_ps
    assert set(proposal.theta.col_ps).issubset(set(features))
    assert set(proposal.theta.col_1ds).issubset(set(schema_card["column_order"]))
    assert set(proposal.theta.col_2ds).issubset(set(schema_card["column_order"]))
