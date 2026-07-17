from __future__ import annotations

import statistics, math
from typing import Any

from .tree import MCTSNode, MCTSTree


def ucb(node: MCTSNode, parent: MCTSNode, c: float) -> float:
    parent_visits = max(int(parent.N), 1)
    exploit = float(node.Q)
    explore = float(c) * float(node.p) * math.sqrt(max(math.log(parent_visits + 1.0), 0.0)) / (1.0 + float(node.N))
    return float(exploit + explore)

def get_c(children: list[MCTSNode]):
    Qs = [float(node.Q) for node in children]
    
    if len(Qs) >= 2:
        c_eff = max(statistics.stdev(Qs), 0.002)
    else:
        c_eff = 0.002

    return c_eff


def selection(tree: MCTSTree, root_id: str, c: float) -> tuple[MCTSNode, list[dict[str, Any]]]:
    node = tree.get(root_id)
    trace: list[dict[str, Any]] = []
    node.N += 1
    while node.children_ids:
        children = tree.children(node)
        c_adapt = get_c(children)
        scored = [(ucb(child, node, c_adapt), child) for child in children]
        scored.sort(key=lambda item: (item[0], item[1].Q, item[1].p, item[1].node_id), reverse=True)
        chosen_score, chosen = scored[0]
        trace.append(
            {
                "parent_id": node.node_id,
                "chosen_id": chosen.node_id,
                "chosen_ucb": float(chosen_score),
                "children": [
                    {
                        "node_id": child.node_id,
                        "theta_id": child.theta_id,
                        "Q": float(child.Q),
                        "Q_self": float(child.Q_self),
                        "N": int(child.N),
                        "p": float(child.p),
                        "ucb": float(score),
                    }
                    for score, child in scored
                ],
            }
        )
        node = chosen
        node.N += 1
    return node, trace


def update_Q_upward(tree: MCTSTree, start_node_id: str) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    current_id: str | None = start_node_id
    while current_id is not None:
        node = tree.get(current_id)
        old_q = float(node.Q)
        old_max_child_q = float(node.max_child_Q)
        child_qs = [float(child.Q) for child in tree.children(node)]
        max_child_q = max(child_qs) if child_qs else float("-inf")
        if child_qs and max_child_q > old_max_child_q + 1e-12:
            node.max_child_Q = float(max_child_q)
            node.Q = max(float(node.Q_self), 0.5 * (old_q + max_child_q))
        elif not child_qs:
            node.Q = float(node.Q_self)
        updates.append(
            {
                "node_id": node.node_id,
                "old_Q": old_q,
                "new_Q": float(node.Q),
                "old_max_child_Q": old_max_child_q,
                "new_max_child_Q": float(node.max_child_Q),
            }
        )
        if abs(float(node.Q) - old_q) < 1e-9:
            break
        current_id = node.parent_id
    return updates
