from __future__ import annotations

import json

from llm_mcts_tool.archive import MCTSArchive
from llm_mcts_tool.strategy import StrategyTheta
from llm_mcts_tool.tree import MCTSNode, MCTSTree


def test_archive_initializes_expected_jsonl_files(tmp_path) -> None:
    archive = MCTSArchive(tmp_path / "mcts")

    for filename in ("all_rollouts.jsonl", "valid_rollouts.jsonl", "failed_rollouts.jsonl", "duplicate_theta.jsonl"):
        path = archive.archive_dir / filename
        assert path.exists()
        assert path.read_text() == ""


def test_archive_snapshot_keeps_failed_rollouts_in_all_rollouts(tmp_path) -> None:
    archive = MCTSArchive(tmp_path / "mcts")
    tree = MCTSTree()
    root = MCTSNode(
        node_id="root",
        theta_id=None,
        theta=None,
        parent_id=None,
        children_ids=[],
        depth=0,
        Q_self=0.0,
        Q=0.0,
        N=0,
        p=1.0,
        max_child_Q=0.0,
        is_leaf=True,
        rollout_id=None,
        reward_available=False,
        guard_pass=False,
        rollout_status="root",
    )
    child = MCTSNode(
        node_id="n_000000",
        theta_id="theta_a",
        theta=StrategyTheta(col_1ds=("age",), col_2ds=("age",), col_ps=("age",), col_u="age"),
        parent_id="root",
        children_ids=[],
        depth=1,
        Q_self=0.0,
        Q=0.0,
        N=0,
        p=0.5,
        max_child_Q=0.0,
        is_leaf=True,
        rollout_id="theta_a",
        reward_available=False,
        guard_pass=False,
        rollout_status="failed",
        error="boom",
    )
    tree.add_node(root)
    tree.add_child("root", child)

    archive.write_archive_snapshot(tree)

    all_records = [json.loads(line) for line in (archive.archive_dir / "all_rollouts.jsonl").read_text().splitlines()]
    failed_records = [
        json.loads(line) for line in (archive.archive_dir / "failed_rollouts.jsonl").read_text().splitlines()
    ]
    assert all_records[0]["rollout_status"] == "failed"
    assert all_records[0]["error"] == "boom"
    assert failed_records[0]["error"] == "boom"
