from __future__ import annotations

import argparse
from pathlib import Path

from .config import MCTSColumnConfig
from .search import run_mcts_column_search


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random column-policy MCTS: no LLM. "
            "Neighbors are random single-column edits of Theta; "
            "each candidate is evaluated by the same Select+Floor+Repair operator."
        )
    )
    parser.add_argument("--dataset-name", type=str, default="adult")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--sample-root", type=Path, default=Path("third_party/sample-2"))
    parser.add_argument("--prompt-pack-dir", type=Path, default=Path("prompt_pack"))
    parser.add_argument("--single-source", type=str, default="tabdiff")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--keep-k", type=int, default=None)
    parser.add_argument("--preselect-target", type=int, default=None)
    parser.add_argument("--d-cur-size", type=int, default=None)
    parser.add_argument("--density-reference-size", type=int, default=None)
    parser.add_argument("--max-theta-pairs", type=int, default=32)
    parser.add_argument("--eval-device", type=str, default="auto")
    parser.add_argument("--nn-device", type=str, default="auto")
    parser.add_argument(
        "--utility-exact-evaluator",
        choices=["tabdiff_mle", "torch_lightweight_mlp"],
        default="torch_lightweight_mlp",
        help=(
            "Exact utility evaluator. torch_lightweight_mlp sets "
            "utility_exact_primary_model to TorchMLPClassifier on classification "
            "datasets such as Adult."
        ),
    )
    parser.add_argument("--utility-exact-torch-epochs", type=int, default=6)
    parser.add_argument(
        "--mcts-budget",
        type=int,
        default=20,
        help="MCTS expansion events. Each event evaluates --theta-proposals-per-event random column edits.",
    )
    parser.add_argument(
        "--theta-proposals-per-event",
        type=int,
        default=4,
        help="Random single-column neighbors evaluated at each greedy step.",
    )
    parser.add_argument(
        "--early-stop-stagnation-events",
        type=int,
        default=6,
        help="Stop after this many consecutive non-improving steps; negative disables early stop.",
    )
    parser.add_argument(
        "--greedy-focus-size",
        type=int,
        default=2,
        help="Start size for each column set. Schema-order prefixes of this size start the search.",
    )
    parser.add_argument(
        "--greedy-max-size",
        type=int,
        default=8,
        help="Hard cap on each column set. Greedy may add columns up to this size, but cannot grow back to all-column Theta0.",
    )
    parser.add_argument("--greedy-max-fraction", type=float, default=0.8,
                        help="Optional fraction of each eligible column universe; overrides fixed max size.")
    parser.add_argument("--reference-split-dir", type=Path, default=None)
    parser.add_argument("--dcr-align-mode", choices=["paired_nn_coreset", "query_batches_full_train_v1"], default="query_batches_full_train_v1")
    parser.add_argument("--dcr-repeats", type=int, default=2)
    parser.add_argument("--dcr-seed", type=int, default=20261027)
    parser.add_argument("--enable-hold-dcr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluation-seed", type=int, default=20260915)
    parser.add_argument("--evaluation-density-reference-size", type=int, default=5000)
    parser.add_argument("--ucb-c", type=float, default=2.0)
    parser.add_argument("--pool-multiplier", type=float, default=1.0)
    parser.add_argument("--utility-diag-sample-size", type=int, default=6000)
    parser.add_argument(
        "--rollout-direct-dcr-repair",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--rollout-direct-dcr-target-margin", type=float, default=None)
    parser.add_argument("--rollout-direct-dcr-max-swap-fraction", type=float, default=None)
    parser.add_argument("--rollout-direct-dcr-candidate-neighbors", type=int, default=None)
    parser.add_argument("--rollout-direct-dcr-min-pair-utility-gain", type=float, default=-0.08)
    parser.add_argument("--rollout-direct-dcr-fallback-min-pair-utility-gain", type=float, default=-0.18)
    parser.add_argument("--rollout-reward-candidate-v2", action="store_true")
    parser.add_argument("--rollout-reward-candidate-v2-max-swap-fraction", type=float, default=0.16)
    parser.add_argument("--rollout-reward-candidate-v2-max-candidate-sizes", type=int, default=10)
    parser.add_argument("--rollout-reward-candidate-v2-min-proxy-delta", type=float, default=0.0)
    parser.add_argument("--rollout-reward-candidate-v2-fidelity-floor-eps", type=float, default=0.015)
    parser.add_argument("--rollout-reward-candidate-v2-utility-floor-eps", type=float, default=0.02)
    parser.add_argument("--save-validation-records", action="store_true")
    parser.add_argument("--save-rollout-internal-records", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> MCTSColumnConfig:
    return MCTSColumnConfig(
        ucb_c=args.ucb_c,
        dataset_name=args.dataset_name,
        exp_name=args.exp_name or f"{args.single_source}_sample_0",
        artifact_dir=args.artifact_dir or Path("artifacts_2/mcts") / args.dataset_name,
        sample_root=args.sample_root,
        prompt_pack_dir=args.prompt_pack_dir,
        single_source=args.single_source,
        seed=args.seed,
        keep_k=args.keep_k,
        preselect_target=args.preselect_target,
        d_cur_size=args.d_cur_size,
        density_reference_size=args.density_reference_size,
        max_theta_pairs=args.max_theta_pairs,
        eval_device=args.eval_device,
        nn_device=args.nn_device,
        utility_exact_evaluator=args.utility_exact_evaluator,
        utility_exact_torch_epochs=args.utility_exact_torch_epochs,
        disable_progress=bool(args.disable_progress),
        save_validation_records=bool(args.save_validation_records),
        save_rollout_internal_records=bool(args.save_rollout_internal_records),
        hill_climb_steps=int(args.mcts_budget),
        neighbors_per_step=int(args.theta_proposals_per_event),
        early_stop_stagnation_events=int(args.early_stop_stagnation_events),
        greedy_focus_size=int(args.greedy_focus_size),
        greedy_max_size=int(args.greedy_max_size),
        greedy_max_fraction=args.greedy_max_fraction,
        reference_split_dir=args.reference_split_dir or Path(__file__).resolve().parents[1] / "synthetic" / args.dataset_name,
        evaluation_dcr_align_mode=args.dcr_align_mode,
        evaluation_dcr_repeats=args.dcr_repeats,
        evaluation_dcr_seed=args.dcr_seed,
        evaluation_enable_hold_dcr=bool(args.enable_hold_dcr),
        evaluation_seed=args.evaluation_seed,
        evaluation_density_reference_size=args.evaluation_density_reference_size,
        pool_multiplier=float(args.pool_multiplier),
        utility_diag_sample_size=args.utility_diag_sample_size,
        rollout_direct_dcr_repair_enabled=args.rollout_direct_dcr_repair,
        rollout_direct_dcr_target_margin=args.rollout_direct_dcr_target_margin,
        rollout_direct_dcr_max_swap_fraction=args.rollout_direct_dcr_max_swap_fraction,
        rollout_direct_dcr_candidate_neighbors=args.rollout_direct_dcr_candidate_neighbors,
        rollout_direct_dcr_min_pair_utility_gain=args.rollout_direct_dcr_min_pair_utility_gain,
        rollout_direct_dcr_fallback_min_pair_utility_gain=args.rollout_direct_dcr_fallback_min_pair_utility_gain,
        rollout_reward_candidate_v2_enabled=bool(args.rollout_reward_candidate_v2),
        rollout_reward_candidate_v2_max_swap_fraction=args.rollout_reward_candidate_v2_max_swap_fraction,
        rollout_reward_candidate_v2_max_candidate_sizes=args.rollout_reward_candidate_v2_max_candidate_sizes,
        rollout_reward_candidate_v2_min_proxy_delta=args.rollout_reward_candidate_v2_min_proxy_delta,
        rollout_reward_candidate_v2_fidelity_floor_eps=args.rollout_reward_candidate_v2_fidelity_floor_eps,
        rollout_reward_candidate_v2_utility_floor_eps=args.rollout_reward_candidate_v2_utility_floor_eps,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_mcts_column_search(config_from_args(args))
    print(f"random MCTS artifacts saved to {result.mcts_dir}")
    if result.final_node is None:
        print(f"Final node: none status={result.final_status}")
        return
    print(
        "Final node: "
        f"{result.final_node.node_id} s={result.final_node.s_id} "
        f"theta_id={result.final_node.theta_id} reward={result.final_node.reward:.12f} "
        f"status={result.final_status}"
    )


if __name__ == "__main__":
    main()
