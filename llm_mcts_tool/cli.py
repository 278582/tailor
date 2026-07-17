from __future__ import annotations

import argparse
from pathlib import Path

from .config import MCTSGuideConfig
from .llm_client import OpenAICompatibleLLMClient, load_env_file
from .pipeline import run_mcts_with_provider
from .proposal import JsonlProposalProvider, LLMProposalProvider, MockProposalProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM/MCTS-guided Pareto theta search.")
    parser.add_argument("--dataset-name", type=str, default="adult")
    parser.add_argument("--exp-name", type=str, default="adult_mcts")
    parser.add_argument("--synthetic-csv", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/llm_mcts"))
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--keep-k", type=int, default=50)
    parser.add_argument("--preselect-target", type=int, default=300)
    parser.add_argument("--d-cur-size", type=int, default=200)
    parser.add_argument("--d-cur-source", choices=["synthetic", "train"], default="synthetic")
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--n-init", type=int, default=6)
    parser.add_argument("--n-expand", type=int, default=4)
    parser.add_argument("--mcts-budget", type=int, default=20)
    parser.add_argument("--ucb-c", type=float, default=2.0)
    parser.add_argument("--p-random-replace", type=float, default=0.2)
    parser.add_argument("--max-theta-pairs", type=int, default=32)
    parser.add_argument("--provider", choices=["mock", "jsonl", "llm"], default="mock")
    parser.add_argument("--provider-jsonl", type=Path, default=None)
    parser.add_argument("--llm-model", type=str, default="qwen3.6-plus")
    parser.add_argument("--llm-base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--llm-api-key-env", type=str, default="DASHSCOPE_API_KEY")
    parser.add_argument("--llm-timeout", type=int, default=300)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-retry-backoff", type=float, default=3.0)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--prompt-pack-dir", type=Path, default=Path("prompt_pack"))
    parser.add_argument(
        "--disable-refine-dataset-priors",
        action="store_true",
        help="Omit theta_guidance and pair_priors from refine prompt dataset_brief.",
    )
    parser.add_argument("--eval-device", type=str, default="auto")
    parser.add_argument("--nn-device", type=str, default="auto")
    parser.add_argument(
        "--utility-exact-evaluator",
        choices=["tabdiff_mle", "torch_lightweight_mlp"],
        default="tabdiff_mle",
        help="Exact utility evaluator: tabdiff_mle keeps the existing TabDiff/XGBoost path; torch_lightweight_mlp uses the lightweight PyTorch evaluator.",
    )
    parser.add_argument("--density-reference-size", type=int, default=5000)
    parser.add_argument("--save-validation-records", action="store_true")
    parser.add_argument("--save-rollout-internal-records", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> MCTSGuideConfig:
    return MCTSGuideConfig(
        dataset_name=args.dataset_name,
        exp_name=args.exp_name,
        synthetic_csv=args.synthetic_csv,
        artifact_dir=args.artifact_dir,
        seed=args.seed,
        keep_k=args.keep_k,
        preselect_target=args.preselect_target,
        d_cur_size=args.d_cur_size,
        n_init=args.n_init,
        n_expand=args.n_expand,
        mcts_budget=args.mcts_budget,
        ucb_c=args.ucb_c,
        p_random_replace=args.p_random_replace,
        max_theta_pairs=args.max_theta_pairs,
        disable_progress=args.disable_progress,
        d_cur_source=args.d_cur_source,
        holdout_fraction=args.holdout_fraction,
        eval_device=args.eval_device,
        nn_device=args.nn_device,
        utility_exact_evaluator=args.utility_exact_evaluator,
        density_reference_size=args.density_reference_size,
        save_validation_records=bool(args.save_validation_records),
        save_rollout_internal_records=bool(args.save_rollout_internal_records),
        provider=args.provider,
        provider_jsonl=args.provider_jsonl,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key_env=args.llm_api_key_env,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff=args.llm_retry_backoff,
        prompt_pack_dir=args.prompt_pack_dir,
        refine_prompt_use_dataset_priors=not bool(args.disable_refine_dataset_priors),
    )


def provider_from_args(args: argparse.Namespace):
    if args.provider == "mock":
        return MockProposalProvider(seed=args.seed)
    if args.provider == "jsonl":
        if args.provider_jsonl is None:
            raise ValueError("--provider jsonl requires --provider-jsonl")
        return JsonlProposalProvider(args.provider_jsonl)
    load_env_file(args.env_file)
    client = OpenAICompatibleLLMClient(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key_env=args.llm_api_key_env,
        timeout=args.llm_timeout,
        max_retries=args.llm_max_retries,
        retry_backoff=args.llm_retry_backoff,
    )
    return LLMProposalProvider(client)


def main() -> None:
    args = parse_args()
    result = run_mcts_with_provider(config_from_args(args), provider_from_args(args))
    print(f"MCTS artifacts saved to {result.mcts_dir}")
    if result.final_node is None:
        print(f"Final node: none status={result.final_status}")
    else:
        print(f"Final node: {result.final_node.node_id} theta_id={result.final_node.theta_id} status={result.final_status}")


if __name__ == "__main__":
    main()
