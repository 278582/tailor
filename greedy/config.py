from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from post_selection_tool.census_profile import apply_census_runtime_profile

from llm_mcts_tool.v2_pipeline import V2MCTSConfig, materialize_v2_search_defaults

SEARCH_BACKEND = "greedy_column"
PROPOSAL_SOURCE = "random_column_edit"


@dataclass
class GreedyRuntimeConfig(V2MCTSConfig):
    """Greedy-only extensions; the shared V2 configuration stays unchanged."""

    reference_split_dir: Path | None = None
    evaluation_dcr_align_mode: str = "paired_nn_coreset"
    evaluation_dcr_repeats: int = 10
    evaluation_dcr_seed: int | None = None
    evaluation_enable_hold_dcr: bool = False
    evaluation_seed: int | None = None
    evaluation_density_reference_size: int | None = None


@dataclass(frozen=True)
class GreedyColumnConfig:
    """Search knobs for the no-LLM, no-MCTS column hill climb."""

    dataset_name: str = "adult"
    exp_name: str = "adult_greedy_column"
    artifact_dir: Path = Path("artifacts/greedy/adult")
    sample_root: Path = Path("third_party/sample")
    prompt_pack_dir: Path = Path("prompt_pack")
    single_source: str = "tabdiff"
    seed: int = 20260420
    keep_k: int | None = None
    preselect_target: int | None = None
    d_cur_size: int | None = None
    density_reference_size: int | None = None
    max_theta_pairs: int = 32
    eval_device: str = "auto"
    nn_device: str = "auto"
    utility_exact_evaluator: str = "torch_lightweight_mlp"
    utility_exact_torch_epochs: int = 6
    disable_progress: bool = True
    save_validation_records: bool = False
    save_rollout_internal_records: bool = False
    hill_climb_steps: int = 20
    neighbors_per_step: int = 4
    early_stop_stagnation_events: int = 6
    greedy_focus_size: int = 2
    greedy_max_size: int = 8
    greedy_max_fraction: float | None = None
    reference_split_dir: Path | None = None
    evaluation_dcr_align_mode: str = "paired_nn_coreset"
    evaluation_dcr_repeats: int = 10
    evaluation_dcr_seed: int | None = None
    evaluation_enable_hold_dcr: bool = False
    evaluation_seed: int | None = None
    evaluation_density_reference_size: int | None = None
    pool_multiplier: float = 1.0
    utility_diag_sample_size: int = 6000
    rollout_direct_dcr_repair_enabled: bool | None = None
    rollout_direct_dcr_target_margin: float | None = None
    rollout_direct_dcr_max_swap_fraction: float | None = None
    rollout_direct_dcr_candidate_neighbors: int | None = None
    rollout_direct_dcr_min_pair_utility_gain: float = -0.08
    rollout_direct_dcr_fallback_min_pair_utility_gain: float = -0.18
    rollout_reward_candidate_v2_enabled: bool = False
    rollout_reward_candidate_v2_max_swap_fraction: float = 0.16
    rollout_reward_candidate_v2_max_candidate_sizes: int = 10
    rollout_reward_candidate_v2_min_proxy_delta: float = 0.0
    rollout_reward_candidate_v2_fidelity_floor_eps: float = 0.015
    rollout_reward_candidate_v2_utility_floor_eps: float = 0.02


def build_v2_config(config: GreedyColumnConfig) -> V2MCTSConfig:
    """Map greedy knobs onto V2MCTSConfig so Algorithm 1 rollouts stay identical.

    LLM/MCTS fields are pinned off: mock provider, zero random-replace on
    LLM proposals (we never generate those), single-source pool, one S node.
    `mcts_budget` is reused as the number of hill-climb steps so the CLI
    budget flag matches Table 3's rollout-event counting.
    """
    v2 = GreedyRuntimeConfig(
        dataset_name=config.dataset_name,
        exp_name=config.exp_name,
        artifact_dir=config.artifact_dir,
        sample_root=config.sample_root,
        prompt_pack_dir=config.prompt_pack_dir,
        source_names=(config.single_source,),
        mode="single",
        single_source=config.single_source,
        seed=config.seed,
        keep_k=config.keep_k,
        preselect_target=config.preselect_target,
        d_cur_size=config.d_cur_size,
        density_reference_size=config.density_reference_size,
        max_theta_pairs=config.max_theta_pairs,
        eval_device=config.eval_device,
        nn_device=config.nn_device,
        utility_exact_evaluator=config.utility_exact_evaluator,
        utility_exact_torch_epochs=config.utility_exact_torch_epochs,
        disable_progress=bool(config.disable_progress),
        save_validation_records=bool(config.save_validation_records),
        save_rollout_internal_records=bool(config.save_rollout_internal_records),
        mcts_budget=int(config.hill_climb_steps),
        initial_s_pool_count=1,
        theta_proposals_per_event=int(config.neighbors_per_step),
        ucb_c=0.0,
        p_random_replace=0.0,
        pool_multiplier=float(config.pool_multiplier),
        provider="mock",
        refine_s_pool_count=1,
        source_profile_repeats=0,
        source_profile_sample_rows=None,
        utility_diag_sample_size=config.utility_diag_sample_size,
        rollout_direct_dcr_repair_enabled=config.rollout_direct_dcr_repair_enabled,
        rollout_direct_dcr_target_margin=config.rollout_direct_dcr_target_margin,
        rollout_direct_dcr_max_swap_fraction=config.rollout_direct_dcr_max_swap_fraction,
        rollout_direct_dcr_candidate_neighbors=config.rollout_direct_dcr_candidate_neighbors,
        rollout_direct_dcr_min_pair_utility_gain=config.rollout_direct_dcr_min_pair_utility_gain,
        rollout_direct_dcr_fallback_min_pair_utility_gain=config.rollout_direct_dcr_fallback_min_pair_utility_gain,
        rollout_reward_candidate_v2_enabled=bool(config.rollout_reward_candidate_v2_enabled),
        rollout_reward_candidate_v2_max_swap_fraction=config.rollout_reward_candidate_v2_max_swap_fraction,
        rollout_reward_candidate_v2_max_candidate_sizes=config.rollout_reward_candidate_v2_max_candidate_sizes,
        rollout_reward_candidate_v2_min_proxy_delta=config.rollout_reward_candidate_v2_min_proxy_delta,
        rollout_reward_candidate_v2_fidelity_floor_eps=config.rollout_reward_candidate_v2_fidelity_floor_eps,
        rollout_reward_candidate_v2_utility_floor_eps=config.rollout_reward_candidate_v2_utility_floor_eps,
        new_s_pool_stagnation_events=10**9,
        early_stop_stagnation_events=int(config.early_stop_stagnation_events),
        force_new_s_at_event=None,
        smoke=False,
        reference_split_dir=config.reference_split_dir,
        evaluation_dcr_align_mode=config.evaluation_dcr_align_mode,
        evaluation_dcr_repeats=config.evaluation_dcr_repeats,
        evaluation_dcr_seed=config.evaluation_dcr_seed,
        evaluation_enable_hold_dcr=config.evaluation_enable_hold_dcr,
        evaluation_seed=config.evaluation_seed,
        evaluation_density_reference_size=config.evaluation_density_reference_size,
    )
    v2 = materialize_v2_search_defaults(v2)
    return apply_census_runtime_profile(v2)


def greedy_run_metadata(config: GreedyColumnConfig) -> dict[str, object]:
    return {
        "search_backend": SEARCH_BACKEND,
        "proposal_source": PROPOSAL_SOURCE,
        "llm_used": False,
        "mcts_used": False,
        "uct_used": False,
        "commit_rule": "strict_improvement",
        "start_policy": "naive_schema_prefix_compact",
        "greedy_focus_size": int(config.greedy_focus_size),
        "greedy_max_size": int(config.greedy_max_size),
        "greedy_max_fraction": config.greedy_max_fraction,
        "column_cap_policy": "floor(fraction * eligible_columns)" if config.greedy_max_fraction is not None else "fixed_size",
        "evaluation_dcr_protocol": config.evaluation_dcr_align_mode,
        "repair_signal_protocol": "legacy full-train/hold per-row signals; existing column scopes",
        "hill_climb_steps": int(config.hill_climb_steps),
        "neighbors_per_step": int(config.neighbors_per_step),
        "early_stop_stagnation_events": int(config.early_stop_stagnation_events),
        "single_source": config.single_source,
    }
