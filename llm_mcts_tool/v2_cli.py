from __future__ import annotations

import argparse
import os
from pathlib import Path

from .llm_client import (
    DASHSCOPE_COMPAT_URLS,
    GLM_DEFAULT_API_KEY_ENV,
    GLM_DEFAULT_MAX_TOKENS,
    GLM_DEFAULT_MODEL,
    GLM_DEFAULT_TEMPERATURE,
    GLM_DEFAULT_TIMEOUT,
    GPT_LUNA_DEFAULT_API_KEY_ENV,
    GPT_LUNA_DEFAULT_BASE_URL,
    GPT_LUNA_DEFAULT_MODEL,
    GptLunaLLMClient,
    OpenAICompatibleLLMClient,
    VolcengineArkLLMClient,
    is_glm_model,
    is_gpt_luna_model,
    load_env_file,
    load_gpt_api_url_file,
)
from post_selection_tool.census_profile import apply_census_runtime_profile

from .v2_pipeline import V2MCTSConfig, run_v2_mcts


def _parse_sources(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2 target-aware LLM-MCTS for supported TabDiff datasets.")
    parser.add_argument("--dataset-name", type=str, default="adult")
    parser.add_argument("--exp-name", type=str, default="adult_llm_mcts_v2")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/llm_mcts_v2/adult"))
    parser.add_argument("--sample-root", type=Path, default=Path("third_party/sample"))
    parser.add_argument("--prompt-pack-dir", type=Path, default=Path("prompt_pack"))
    parser.add_argument("--mode", choices=["mixed", "single"], default="mixed")
    parser.add_argument("--sources", type=str, default="great,smote,tabdiff,tabsyn")
    parser.add_argument("--single-source", type=str, default="tabdiff")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument(
        "--keep-k",
        type=int,
        default=None,
        help="Rows to keep. Default: synthetic/{dataset}/train.csv size after the 90/10 train/hold split.",
    )
    parser.add_argument(
        "--preselect-target",
        type=int,
        default=None,
        help="Preselect pool size. Default: round(1.4 * keep_k) from the dataset search table.",
    )
    parser.add_argument(
        "--d-cur-size",
        type=int,
        default=None,
        help="Current-set size for density/DCR proxies. Default: 2000 from the dataset search table.",
    )
    parser.add_argument(
        "--density-reference-size",
        type=int,
        default=None,
        help="Density reference size. Default: 5000/8000/10000 by dataset from the search table.",
    )
    parser.add_argument("--max-theta-pairs", type=int, default=32)
    parser.add_argument("--eval-device", type=str, default="auto")
    parser.add_argument("--nn-device", type=str, default="auto")
    parser.add_argument(
        "--utility-exact-evaluator",
        choices=["tabdiff_mle", "torch_lightweight_mlp"],
        default="tabdiff_mle",
        help="Exact utility evaluator: tabdiff_mle keeps the existing TabDiff/XGBoost path; torch_lightweight_mlp uses the lightweight PyTorch evaluator.",
    )
    parser.add_argument("--mcts-budget", type=int, default=20)
    parser.add_argument(
        "--initial-s-pool-count",
        "--m1",
        dest="initial_s_pool_count",
        type=int,
        default=2,
        help="Number of initial S source pools to create; --m1 is a deprecated alias.",
    )
    parser.add_argument(
        "--theta-proposals-per-event",
        "--m2",
        dest="theta_proposals_per_event",
        type=int,
        default=4,
        help="Number of theta proposals evaluated for each initial/refine/transfer event; --m2 is a deprecated alias.",
    )
    parser.add_argument(
        "--ucb-c",
        type=float,
        default=2.0,
        help="UCT explore-scale coefficient: explore_scale=max(ucb_c*stdev(best_reward+0.05*llm_score), 0.01).",
    )
    parser.add_argument("--p-random-replace", type=float, default=0.1)
    parser.add_argument("--pool-multiplier", type=float, default=4.0)
    parser.add_argument(
        "--refine-s-pool-count",
        type=int,
        default=1,
        help="Number of new S source pools requested when mixed-mode stagnation triggers refine source selection.",
    )
    parser.add_argument("--source-profile-repeats", type=int, default=4)
    parser.add_argument("--source-profile-sample-rows", type=int, default=None)
    parser.add_argument("--utility-diag-sample-size", type=int, default=6000)
    parser.add_argument(
        "--utility-exact-torch-epochs",
        type=int,
        default=6,
        help="Epochs for the torch_lightweight_mlp exact utility evaluator; ignored by tabdiff_mle.",
    )
    parser.add_argument(
        "--rollout-direct-dcr-repair",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply direct DCR repair after each LLM-MCTS Pareto rollout and before exact final evaluation. "
            "Default: on for every dataset in the search table. Use --no-rollout-direct-dcr-repair to disable."
        ),
    )
    parser.add_argument("--rollout-direct-dcr-target-margin", type=float, default=None)
    parser.add_argument("--rollout-direct-dcr-max-swap-fraction", type=float, default=None)
    parser.add_argument("--rollout-direct-dcr-candidate-neighbors", type=int, default=None)
    parser.add_argument("--rollout-direct-dcr-min-pair-utility-gain", type=float, default=-0.08)
    parser.add_argument("--rollout-direct-dcr-fallback-min-pair-utility-gain", type=float, default=-0.18)
    parser.add_argument(
        "--rollout-reward-candidate-v2",
        action="store_true",
        help="Apply reward_candidate_v2 row-swap refinement after each LLM-MCTS Pareto rollout and before direct DCR repair.",
    )
    parser.add_argument("--rollout-reward-candidate-v2-max-swap-fraction", type=float, default=0.16)
    parser.add_argument("--rollout-reward-candidate-v2-max-candidate-sizes", type=int, default=10)
    parser.add_argument("--rollout-reward-candidate-v2-min-proxy-delta", type=float, default=0.0)
    parser.add_argument("--rollout-reward-candidate-v2-fidelity-floor-eps", type=float, default=0.015)
    parser.add_argument("--rollout-reward-candidate-v2-utility-floor-eps", type=float, default=0.02)
    parser.add_argument(
        "--new-s-pool-stagnation-events",
        "--mixed-stagnation-events",
        dest="new_s_pool_stagnation_events",
        type=int,
        default=2,
        help="Create a new S pool after the global best theta has not improved for more than this many expansion events; --mixed-stagnation-events is a deprecated alias.",
    )
    parser.add_argument(
        "--early-stop-stagnation-events",
        type=int,
        default=6,
        help="Stop search after the global best theta has not improved for more than this many expansion events; set a negative value to disable early stop.",
    )
    parser.add_argument("--force-new-s-at-event", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--provider",
        choices=["llm", "mock", "glm", "gpt"],
        default="llm",
        help=(
            "llm keeps DashScope/OpenAI-compatible calls; glm uses Volcengine Ark; "
            "gpt uses gpt-5.6-luna. Model names glm-* / gpt-5.6* also auto-route when --provider llm."
        ),
    )
    parser.add_argument("--llm-model", type=str, default="qwen3.7-plus")
    parser.add_argument("--llm-base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--llm-api-key-env", type=str, default="DASHSCOPE_API_KEY")
    parser.add_argument("--llm-timeout", type=int, default=300)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-retry-backoff", type=float, default=3.0)
    parser.add_argument(
        "--glm-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only used with --provider glm. Default: enabled, matching volcengine/glm-5-2-260617.py.",
    )
    parser.add_argument(
        "--glm-max-tokens",
        type=int,
        default=GLM_DEFAULT_MAX_TOKENS,
        help="Only used with --provider glm.",
    )
    parser.add_argument(
        "--glm-temperature",
        type=float,
        default=GLM_DEFAULT_TEMPERATURE,
        help="Only used with --provider glm.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--save-validation-records", action="store_true")
    parser.add_argument("--save-rollout-internal-records", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> V2MCTSConfig:
    return apply_census_runtime_profile(V2MCTSConfig(
        dataset_name=args.dataset_name,
        exp_name=args.exp_name,
        artifact_dir=args.artifact_dir,
        sample_root=args.sample_root,
        prompt_pack_dir=args.prompt_pack_dir,
        source_names=_parse_sources(args.sources),
        mode=args.mode,
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
        disable_progress=bool(args.disable_progress),
        save_validation_records=bool(args.save_validation_records),
        save_rollout_internal_records=bool(args.save_rollout_internal_records),
        mcts_budget=args.mcts_budget,
        initial_s_pool_count=args.initial_s_pool_count,
        theta_proposals_per_event=args.theta_proposals_per_event,
        ucb_c=args.ucb_c,
        p_random_replace=args.p_random_replace,
        pool_multiplier=args.pool_multiplier,
        refine_s_pool_count=args.refine_s_pool_count,
        source_profile_repeats=args.source_profile_repeats,
        source_profile_sample_rows=args.source_profile_sample_rows,
        utility_diag_sample_size=args.utility_diag_sample_size,
        utility_exact_torch_epochs=args.utility_exact_torch_epochs,
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
        new_s_pool_stagnation_events=args.new_s_pool_stagnation_events,
        early_stop_stagnation_events=args.early_stop_stagnation_events,
        force_new_s_at_event=args.force_new_s_at_event,
        smoke=bool(args.smoke),
        provider=args.provider,
    ))


def client_from_args(args: argparse.Namespace):
    if args.provider == "mock":
        return None
    load_env_file(args.env_file)
    load_gpt_api_url_file()
    use_glm = args.provider == "glm" or is_glm_model(args.llm_model)
    if use_glm:
        model = args.llm_model
        api_key_env = args.llm_api_key_env
        timeout = args.llm_timeout
        if model == "qwen3.7-plus":
            model = GLM_DEFAULT_MODEL
        if api_key_env == "DASHSCOPE_API_KEY":
            api_key_env = GLM_DEFAULT_API_KEY_ENV
        if timeout == 300:
            timeout = GLM_DEFAULT_TIMEOUT
        return VolcengineArkLLMClient(
            model=model,
            api_key_env=api_key_env,
            timeout=timeout,
            max_retries=args.llm_max_retries,
            retry_backoff=args.llm_retry_backoff,
            thinking=bool(args.glm_thinking),
            max_tokens=int(args.glm_max_tokens),
            temperature=float(args.glm_temperature),
        )
    use_gpt_luna = args.provider == "gpt" or is_gpt_luna_model(args.llm_model)
    if use_gpt_luna:
        model = args.llm_model
        api_key_env = args.llm_api_key_env
        base_url = str(args.llm_base_url).rstrip("/")
        if model == "qwen3.7-plus":
            model = GPT_LUNA_DEFAULT_MODEL
        if api_key_env == "DASHSCOPE_API_KEY":
            api_key_env = GPT_LUNA_DEFAULT_API_KEY_ENV
        if base_url in DASHSCOPE_COMPAT_URLS:
            base_url = os.environ.get("OPENAI_BASE_URL", GPT_LUNA_DEFAULT_BASE_URL).rstrip("/")
        return GptLunaLLMClient(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            retry_backoff=args.llm_retry_backoff,
        )
    return OpenAICompatibleLLMClient(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key_env=args.llm_api_key_env,
        timeout=args.llm_timeout,
        max_retries=args.llm_max_retries,
        retry_backoff=args.llm_retry_backoff,
    )


def main() -> None:
    args = parse_args()
    result = run_v2_mcts(config_from_args(args), client_from_args(args))
    print(f"v2 MCTS artifacts saved to {result.mcts_dir}")
    if result.final_node is None:
        print(f"Final node: none status={result.final_status}")
    else:
        print(
            "Final node: "
            f"{result.final_node.node_id} s={result.final_node.s_id} "
            f"theta_id={result.final_node.theta_id} reward={result.final_node.reward:.12f} "
            f"status={result.final_status}"
        )


if __name__ == "__main__":
    main()
