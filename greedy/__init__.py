"""Greedy column-policy ablation: no LLM, no MCTS.

Search object
    Theta = (col_1ds, col_2ds, col_ps, col_u). Rows are never chosen here.

Algorithm
    Start from a compact schema-order prefix (not the strong all-column Theta0).
    Each step randomly adds or replaces one column. Set size may grow from
    greedy_focus_size up to greedy_max_size, but cannot fill the full schema.
    Commit only if the best neighbor strictly improves the current reward.
"""

from .config import GreedyColumnConfig, build_v2_config
from .random_edits import (
    apply_column_edit,
    default_column_theta,
    greedy_start_theta,
    is_single_column_edit,
    list_legal_column_edits,
    propose_random_neighbors,
)
from .search import choose_commit_child, run_greedy_column_search

__all__ = [
    "GreedyColumnConfig",
    "apply_column_edit",
    "build_v2_config",
    "choose_commit_child",
    "default_column_theta",
    "greedy_start_theta",
    "is_single_column_edit",
    "list_legal_column_edits",
    "propose_random_neighbors",
    "run_greedy_column_search",
]
