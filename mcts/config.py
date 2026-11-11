from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from greedy.config import GreedyColumnConfig, build_v2_config as build_greedy_runtime

SEARCH_BACKEND = "random_column_mcts"
PROPOSAL_SOURCE = "random_column_edit"


@dataclass(frozen=True)
class MCTSColumnConfig(GreedyColumnConfig):
    """Greedy-compatible settings; only the tree-selection policy differs."""

    exp_name: str = "tabdiff_sample_0"
    artifact_dir: Path = Path("artifacts_2/mcts/adult")
    sample_root: Path = Path("third_party/sample-2")
    greedy_max_fraction: float | None = 0.8
    evaluation_dcr_align_mode: str = "query_batches_full_train_v1"
    evaluation_dcr_repeats: int = 2
    evaluation_dcr_seed: int | None = 20261027
    evaluation_enable_hold_dcr: bool = True
    evaluation_seed: int | None = 20260915
    evaluation_density_reference_size: int | None = 5000
    ucb_c: float = 2.0


def build_v2_config(config: MCTSColumnConfig):
    if config.ucb_c < 0:
        raise ValueError("ucb_c must be nonnegative")
    if config.hill_climb_steps < 0 or config.neighbors_per_step < 1:
        raise ValueError("MCTS budget must be nonnegative and neighbor count positive")
    runtime = build_greedy_runtime(config)
    return replace(runtime, ucb_c=float(config.ucb_c))


def mcts_run_metadata(config: MCTSColumnConfig):
    return dict(
        search_backend=SEARCH_BACKEND, proposal_source=PROPOSAL_SOURCE,
        llm_used=False, llm_calls=0, mcts_used=True, uct_used=True,
        node_selection="llm_mcts_tool._select_uct_theta_from_s; zero LLM prior",
        tree_value="maximum reward among self and successful descendants",
        commit_rule="successful children retained regardless of parent improvement",
        start_policy="naive_schema_prefix_compact", greedy_focus_size=config.greedy_focus_size,
        greedy_max_size=config.greedy_max_size, greedy_max_fraction=config.greedy_max_fraction,
        column_cap_policy="floor(fraction * eligible_columns)" if config.greedy_max_fraction is not None else "fixed_size",
        expansion_budget=config.hill_climb_steps, neighbors_per_step=config.neighbors_per_step,
        early_stop_stagnation_events=config.early_stop_stagnation_events, ucb_c=config.ucb_c,
        single_source=config.single_source, evaluation_dcr_protocol=config.evaluation_dcr_align_mode,
        repair_signal_protocol="existing legacy_v19 full-train/hold per-row signals; unchanged column scopes",
        test_used_for_outer_search_reward=True,
    )
