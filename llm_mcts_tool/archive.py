from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from post_selection_tool.io import ensure_dir, save_json

from .tree import MCTSNode, MCTSTree


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, tuple):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")


class MCTSArchive:
    def __init__(self, mcts_dir: Path) -> None:
        self.mcts_dir = ensure_dir(Path(mcts_dir))
        self.context_dir = ensure_dir(self.mcts_dir / "context")
        self.tree_dir = ensure_dir(self.mcts_dir / "tree")
        self.prompts_dir = ensure_dir(self.mcts_dir / "prompts")
        self.rollouts_dir = ensure_dir(self.mcts_dir / "rollouts")
        self.archive_dir = ensure_dir(self.mcts_dir / "archive")
        self.final_dir = ensure_dir(self.mcts_dir / "final")
        for filename in ("all_rollouts.jsonl", "valid_rollouts.jsonl", "failed_rollouts.jsonl", "duplicate_theta.jsonl"):
            (self.archive_dir / filename).touch(exist_ok=True)

    def save_node(self, node: MCTSNode) -> None:
        _append_jsonl(self.tree_dir / "nodes.jsonl", node.to_dict())

    def save_edge(self, parent_id: str, child_id: str) -> None:
        _append_jsonl(self.tree_dir / "edges.jsonl", {"parent_id": parent_id, "child_id": child_id})

    def save_ucb_trace(self, trace: dict[str, Any]) -> None:
        _append_jsonl(self.tree_dir / "ucb_trace.jsonl", trace)

    def _rollout_record(self, node: MCTSNode) -> dict[str, Any]:
        return {
            "node_id": node.node_id,
            "theta_id": node.theta_id,
            "parent_id": node.parent_id,
            "depth": int(node.depth),
            "theta": node.to_dict().get("theta"),
            "prior_score": float(node.p),
            "Q_self": float(node.Q_self),
            "Q": float(node.Q),
            "N": int(node.N),
            "search_objectives": dict(node.search_objectives),
            "audit_metrics": dict(node.audit_metrics),
            "guard": dict(node.guard),
            "action_validation": dict(node.action_validation),
            "rollout_status": node.rollout_status,
            "rollout_dir": node.rollout_dir,
            "error": node.error,
        }

    def save_rollout_summary(self, result: Any, node: MCTSNode) -> None:
        record = self._rollout_record(node)
        _append_jsonl(self.archive_dir / "all_rollouts.jsonl", record)
        if node.reward_available and node.guard_pass:
            _append_jsonl(self.archive_dir / "valid_rollouts.jsonl", record)

    def save_failed_rollout(self, node: MCTSNode, error: str) -> None:
        record = node.to_dict()
        record["error"] = error
        _append_jsonl(self.archive_dir / "failed_rollouts.jsonl", record)

    def save_duplicate_theta(self, payload: dict[str, Any]) -> None:
        _append_jsonl(self.archive_dir / "duplicate_theta.jsonl", payload)

    def write_tree_snapshot(self, tree: MCTSTree, ucb_trace: list[dict[str, Any]]) -> None:
        _write_jsonl(self.tree_dir / "nodes.jsonl", [node.to_dict() for node in tree.nodes.values()])
        _write_jsonl(self.tree_dir / "edges.jsonl", list(tree.edges))
        _write_jsonl(self.tree_dir / "ucb_trace.jsonl", ucb_trace)

    def write_archive_snapshot(self, tree: MCTSTree) -> None:
        rollout_nodes = [node for node in tree.nodes.values() if node.theta is not None]
        all_records = [self._rollout_record(node) for node in rollout_nodes]
        _write_jsonl(self.archive_dir / "all_rollouts.jsonl", all_records)
        _write_jsonl(
            self.archive_dir / "valid_rollouts.jsonl",
            [record for record, node in zip(all_records, rollout_nodes) if node.reward_available and node.guard_pass],
        )
        failed = [node.to_dict() for node in rollout_nodes if node.rollout_status == "failed"]
        _write_jsonl(self.archive_dir / "failed_rollouts.jsonl", failed)
        duplicate_path = self.archive_dir / "duplicate_theta.jsonl"
        if not duplicate_path.exists():
            _write_jsonl(duplicate_path, [])


def select_final_node(nodes: list[MCTSNode]) -> tuple[MCTSNode, str]:
    successful = [node for node in nodes if node.rollout_status == "success"]
    if not successful:
        raise RuntimeError("No successful rollout nodes available for finalization")
    guarded = [node for node in successful if node.guard_pass and node.reward_available]
    if guarded:
        return max(guarded, key=lambda node: (node.Q_self, node.Q, node.node_id)), "guard_pass"
    return max(successful, key=lambda node: (node.Q_self, node.Q, node.node_id)), "guard_failed_best_effort"


def write_final_outputs(final_node: MCTSNode, archive: MCTSArchive, *, final_status: str) -> None:
    if final_node.rollout_dir is None:
        raise RuntimeError(f"Final node {final_node.node_id} has no rollout_dir")
    rollout_dir = Path(final_node.rollout_dir)
    final_pareto_src = rollout_dir / "selection_pareto.csv"
    final_metrics_src = rollout_dir / "metrics_summary.json"
    final_feedback_src = rollout_dir / "feedback.json"
    if not final_pareto_src.exists():
        raise FileNotFoundError(final_pareto_src)
    shutil.copyfile(final_pareto_src, archive.final_dir / "final_pareto.csv")
    if final_metrics_src.exists():
        shutil.copyfile(final_metrics_src, archive.final_dir / "final_metrics_summary.json")
    if final_feedback_src.exists():
        shutil.copyfile(final_feedback_src, archive.final_dir / "final_feedback.json")
    theta_star = {
        "final_status": final_status,
        "node_id": final_node.node_id,
        "theta_id": final_node.theta_id,
        "theta": final_node.to_dict().get("theta"),
        "parent_id": final_node.parent_id,
        "Q_self": float(final_node.Q_self),
        "Q": float(final_node.Q),
        "N": int(final_node.N),
        "p": float(final_node.p),
        "search_objectives": dict(final_node.search_objectives),
        "audit_metrics": dict(final_node.audit_metrics),
        "guard": dict(final_node.guard),
        "action_validation": dict(final_node.action_validation),
        "rollout_dir": str(rollout_dir),
        "final_pareto_source": str(final_pareto_src),
    }
    save_json(archive.final_dir / "theta_star.json", theta_star)
    save_json(
        archive.final_dir / "final_selection_manifest.json",
        {
            "final_status": final_status,
            "theta_star": theta_star,
            "final_pareto": "final_pareto.csv",
            "final_metrics_summary": "final_metrics_summary.json",
        },
    )
