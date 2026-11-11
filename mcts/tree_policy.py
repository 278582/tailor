"""Use the actual existing max-backup UCT implementation, with zero LLM priors."""
from llm_mcts_tool.v2_pipeline import (
    _node_selection_score,
    _refresh_s_best_reward,
    _refresh_theta_best_reward_path,
    _select_uct_theta_from_s,
    _uct_explore,
)


def select_parent(*, s_node, theta_nodes, total_visits, ucb_c=2.0, neighbors_per_step=4):
    if s_node.llm_score != 0 or any(node.llm_score != 0 for node in theta_nodes.values()):
        raise ValueError("Random MCTS requires zero LLM scores on every node")
    before = {key: node.visits for key, node in theta_nodes.items()}
    parent, path = _select_uct_theta_from_s(
        s_node=s_node, theta_nodes=theta_nodes, total_visits=total_visits,
        ucb_c=ucb_c, theta_proposals_per_event=neighbors_per_step)
    for step in path:
        step["candidates"] = [
            dict(node_id=key, own_score=_node_selection_score(theta_nodes[key]),
                 best_reward=theta_nodes[key].best_reward, visits_before=before[key],
                 prior=0.0,
                 explore=_uct_explore(step["explore_scale"], total_visits, before[key]),
                 uct_score=theta_nodes[key].best_reward + _uct_explore(
                     step["explore_scale"], total_visits, before[key]))
            for key in step["candidate_node_ids"]
        ]
        assert step["prior"] == step["theta_llm_score"] == step["s_llm_score"] == 0
    return parent, path


def backpropagate(*, theta_nodes, s_node, start_node_id):
    """Return an auditable ancestor path; preserve upstream maximum backup."""
    path = []
    current = theta_nodes.get(start_node_id)
    while current is not None:
        path.append(dict(node_id=current.node_id, before=current.best_reward,
                         own_score=_node_selection_score(current)))
        current = theta_nodes.get(current.parent_node_id)
    _refresh_theta_best_reward_path(theta_nodes, start_node_id)
    _refresh_s_best_reward(s_node, theta_nodes)
    for item in path:
        item["after"] = theta_nodes[item["node_id"]].best_reward
    return path
