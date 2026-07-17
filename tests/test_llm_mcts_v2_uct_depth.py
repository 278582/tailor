from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_mcts_tool.strategy import StrategyTheta
from llm_mcts_tool.v2_pipeline import (
    SNode,
    ThetaNode,
    _select_s_for_refine,
    _select_uct_theta_from_s,
    _theta_depth,
    _uct_explore,
    _uct_base_score,
    _write_run_state,
)


def _theta_node(
    node_id: str,
    *,
    s_id: str = "s_000000",
    parent_node_id: str | None = None,
    llm_score: float = 0.5,
    best_reward: float = 0.8,
    visits: int = 0,
) -> ThetaNode:
    return ThetaNode(
        node_id=node_id,
        s_id=s_id,
        theta=StrategyTheta(
            col_1ds=("age",),
            col_2ds=("age",),
            col_ps=("age",),
            col_u="age",
        ),
        theta_id=f"theta_{node_id}",
        parent_node_id=parent_node_id,
        actions=[],
        proposal_action_validation={},
        llm_score=llm_score,
        reason="fixture",
        visits=visits,
        best_reward=best_reward,
        reward=best_reward,
        reward_available=True,
        exact_reward=best_reward,
        exact_reward_available=True,
        search_reward=best_reward,
        search_reward_available=True,
        reward_type="exact",
        guard_pass=True,
        status="success",
        audit_metrics={"utility_exact": best_reward},
        search_objectives={},
        feedback={},
    )


def _s_node(s_id: str, theta_ids: list[str], *, llm_score: float, best_reward: float) -> SNode:
    return SNode(
        s_id=s_id,
        pool_units=[{"source_id": "tabdiff", "multiplier": 2}],
        synthetic_csv=Path(f"/tmp/{s_id}.csv"),
        synthetic_row_map=Path(f"/tmp/{s_id}.jsonl"),
        llm_score=llm_score,
        reason="fixture",
        theta_node_ids=theta_ids,
        best_reward=best_reward,
    )


def test_uct_explore_scales_with_total_visits_and_visits() -> None:
    scale = 1.7
    low_visit = _uct_explore(scale, total_visits=10, visits=0)
    high_visit = _uct_explore(scale, total_visits=10, visits=5)
    later_total = _uct_explore(scale, total_visits=20, visits=0)

    assert low_visit > high_visit
    assert later_total > low_visit


def test_select_s_for_refine_uses_all_s_nodes_as_base_scores(tmp_path: Path) -> None:
    theta_nodes = {
        "n_000000": _theta_node("n_000000", s_id="s_000000", llm_score=0.1, best_reward=0.6),
        "n_000001": _theta_node("n_000001", s_id="s_000001", llm_score=0.2, best_reward=0.6),
        "n_000002": _theta_node("n_000002", s_id="s_000002", llm_score=0.9, best_reward=0.6),
    }
    s_nodes = {
        "s_000000": _s_node("s_000000", ["n_000000"], llm_score=0.1, best_reward=0.6),
        "s_000001": _s_node("s_000001", ["n_000001"], llm_score=0.2, best_reward=0.6),
        "s_000002": _s_node("s_000002", ["n_000002"], llm_score=0.9, best_reward=0.6),
    }

    selected, trace = _select_s_for_refine(s_nodes, theta_nodes=theta_nodes, total_visits=1, ucb_c=1.0)

    assert selected is not None
    assert selected.s_id == "s_000002"
    assert trace["base_scores_count"] == 3
    assert trace["total_s_nodes"] == 3
    assert trace["base_scores_source"] == "all_s_nodes"
    assert trace["explore_scale"] >= 0.01
    assert trace["candidates"][0]["s_id"] == "s_000002"
    assert trace["candidates"][0]["base_score"] == pytest.approx(_uct_base_score(0.6, 0.9))


def test_select_uct_theta_from_s_uses_theta_siblings_and_bounds_visits() -> None:
    theta_nodes = {
        "n_000000": _theta_node("n_000000", llm_score=0.1, best_reward=0.5),
        "n_000001": _theta_node("n_000001", llm_score=0.2, best_reward=0.5),
        "n_000002": _theta_node("n_000002", llm_score=0.3, best_reward=0.5),
        "n_000003": _theta_node("n_000003", llm_score=0.9, best_reward=0.5),
    }
    theta_nodes["n_000000"].status = "failed"
    s_node = _s_node("s_000000", list(theta_nodes), llm_score=0.5, best_reward=0.5)

    selected, path = _select_uct_theta_from_s(
        s_node=s_node,
        theta_nodes=theta_nodes,
        total_visits=1,
        ucb_c=1.0,
        theta_proposals_per_event=4,
    )

    assert selected is not None
    assert selected.node_id == "n_000003"
    assert selected.visits == 1
    assert len(path) == 1
    assert path[0]["base_scores_count"] == 4
    assert path[0]["actual_candidate_count"] == 3
    assert path[0]["expected_theta_proposals_per_event"] == 4
    assert "n_000000" not in path[0]["candidate_node_ids"]
    assert "n_000000" in path[0]["base_score_node_ids"]
    assert path[0]["visits_before"] == 0
    assert path[0]["visits_after"] == 1
    assert path[0]["theta_llm_score"] == pytest.approx(0.9)
    assert path[0]["explore_scale"] >= 0.01
    assert path[0]["total_visits"] == 1


def test_write_run_state_writes_theta_depth_everywhere(tmp_path: Path) -> None:
    root = _theta_node("n_000000", parent_node_id=None, llm_score=0.3, best_reward=0.7)
    child = _theta_node("n_000001", parent_node_id=root.node_id, llm_score=0.9, best_reward=0.8)
    theta_nodes = {root.node_id: root, child.node_id: child}
    s_nodes = {
        "s_000000": _s_node("s_000000", [root.node_id, child.node_id], llm_score=0.4, best_reward=0.8),
    }
    mcts_dir = tmp_path / "mcts"
    final_rollout_dir = tmp_path / "rollout"
    child.rollout_dir = final_rollout_dir

    _write_run_state(
        mcts_dir=mcts_dir,
        s_nodes=s_nodes,
        theta_nodes=theta_nodes,
        event_trace=[],
        final_node=child,
        final_status="guard_pass",
        include_source_context=False,
    )

    tree_records = [
        json.loads(line)
        for line in (mcts_dir / "tree" / "theta_nodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    archive_records = [
        json.loads(line)
        for line in (mcts_dir / "archive" / "all_theta_nodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    detail_root = json.loads((mcts_dir / "archive" / "theta_details" / "n_000000.json").read_text(encoding="utf-8"))
    detail_child = json.loads((mcts_dir / "archive" / "theta_details" / "n_000001.json").read_text(encoding="utf-8"))
    final_theta = json.loads((mcts_dir / "final" / "theta_star.json").read_text(encoding="utf-8"))
    schema = json.loads((mcts_dir / "archive" / "archive_schema.json").read_text(encoding="utf-8"))

    assert _theta_depth(root, theta_nodes) == 0
    assert _theta_depth(child, theta_nodes) == 1
    assert tree_records[0]["depth"] == 0
    assert tree_records[1]["depth"] == 1
    assert archive_records[0]["depth"] == 0
    assert archive_records[1]["depth"] == 1
    assert detail_root["compact"]["depth"] == 0
    assert detail_child["compact"]["depth"] == 1
    assert detail_root["full"]["depth"] == 0
    assert detail_child["full"]["depth"] == 1
    assert final_theta["depth"] == 1
    assert "depth" in schema
    assert "parent depth + 1" in schema["depth"]
