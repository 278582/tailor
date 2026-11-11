from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .strategy import StrategyTheta


def theta_to_dict(theta: StrategyTheta | None) -> dict[str, Any] | None:
    if theta is None:
        return None
    return {
        "col_1ds": list(theta.col_1ds),
        "col_2ds": list(theta.col_2ds),
        "col_ps": list(theta.col_ps),
        "col_u": theta.col_u,
    }


@dataclass
class MCTSNode:
    node_id: str
    theta_id: str | None
    theta: StrategyTheta | None
    parent_id: str | None
    children_ids: list[str]
    depth: int
    Q_self: float
    Q: float
    N: int
    p: float
    max_child_Q: float
    is_leaf: bool
    rollout_id: str | None
    reward_available: bool
    guard_pass: bool
    rollout_status: str = "pending"
    rollout_dir: str | None = None
    search_objectives: dict[str, Any] = field(default_factory=dict)
    audit_metrics: dict[str, Any] = field(default_factory=dict)
    guard: dict[str, Any] = field(default_factory=dict)
    feedback: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    action_validation: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "theta_id": self.theta_id,
            "theta": theta_to_dict(self.theta),
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "depth": int(self.depth),
            "Q_self": float(self.Q_self),
            "Q": float(self.Q),
            "N": int(self.N),
            "p": float(self.p),
            "max_child_Q": float(self.max_child_Q),
            "is_leaf": bool(self.is_leaf),
            "rollout_id": self.rollout_id,
            "reward_available": bool(self.reward_available),
            "guard_pass": bool(self.guard_pass),
            "rollout_status": self.rollout_status,
            "rollout_dir": self.rollout_dir,
            "search_objectives": dict(self.search_objectives),
            "audit_metrics": dict(self.audit_metrics),
            "guard": dict(self.guard),
            "actions": list(self.actions),
            "action_validation": dict(self.action_validation),
            "reason": self.reason,
            "error": self.error,
        }


class MCTSTree:
    def __init__(self) -> None:
        self.nodes: dict[str, MCTSNode] = {}
        self.edges: list[dict[str, Any]] = []
        self._next_id = 0

    def next_node_id(self) -> str:
        node_id = f"n_{self._next_id:06d}"
        self._next_id += 1
        return node_id

    def add_node(self, node: MCTSNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node_id={node.node_id}")
        self.nodes[node.node_id] = node

    def add_child(self, parent_id: str, child: MCTSNode) -> None:
        parent = self.nodes[parent_id]
        child.parent_id = parent_id
        parent.children_ids.append(child.node_id)
        parent.is_leaf = False
        self.add_node(child)
        self.edges.append(
            {
                "parent_id": parent_id,
                "child_id": child.node_id,
                "theta_id": child.theta_id,
                "prior_score": float(child.p),
            }
        )

    def get(self, node_id: str) -> MCTSNode:
        return self.nodes[node_id]

    def children(self, node: MCTSNode) -> list[MCTSNode]:
        return [self.nodes[child_id] for child_id in node.children_ids]

    def successful_nodes(self) -> list[MCTSNode]:
        return [
            node
            for node in self.nodes.values()
            if node.theta is not None and node.rollout_status == "success"
        ]
