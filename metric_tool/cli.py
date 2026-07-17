from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import MetricConfig
from .pipeline import evaluate_single_selection, run_core_metrics


def _selection_name_from_csv(path: Path, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    stem = path.stem
    return stem[len("selection_") :] if stem.startswith("selection_") else stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate post-selection tabular selections.")
    parser.add_argument("--dataset-name", type=str, default=MetricConfig.dataset_name)
    parser.add_argument("--exp-name", type=str, default=MetricConfig.exp_name)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=MetricConfig.seed)
    parser.add_argument("--holdout-fraction", type=float, default=MetricConfig.holdout_fraction)
    parser.add_argument("--eval-device", type=str, default=MetricConfig.eval_device)
    parser.add_argument("--privacy-version", choices=["v1", "v2", "v3"], default=MetricConfig.privacy_version)
    parser.add_argument("--density-reference-size", type=int, default=MetricConfig.density_reference_size)
    parser.add_argument("--nn-device", type=str, default=MetricConfig.nn_device)
    parser.add_argument("--nn-query-batch-size", type=int, default=MetricConfig.nn_query_batch_size)
    parser.add_argument("--nn-reference-chunk-size", type=int, default=MetricConfig.nn_reference_chunk_size)
    parser.add_argument(
        "--utility-exact-evaluator",
        choices=["tabdiff_mle", "torch_lightweight_mlp"],
        default=MetricConfig.utility_exact_evaluator,
        help="Exact utility evaluator: tabdiff_mle keeps the existing TabDiff/XGBoost path; torch_lightweight_mlp uses the lightweight PyTorch evaluator.",
    )
    parser.add_argument("--versions-dir", type=Path, default=None)
    parser.add_argument("--selection-csv", type=Path, default=None)
    parser.add_argument("--selection-name", type=str, default=None)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> MetricConfig:
    return MetricConfig(
        dataset_name=args.dataset_name,
        exp_name=args.exp_name,
        artifact_dir=args.artifact_dir,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        eval_device=args.eval_device,
        privacy_version=args.privacy_version,
        density_reference_size=args.density_reference_size,
        nn_device=args.nn_device,
        nn_query_batch_size=args.nn_query_batch_size,
        nn_reference_chunk_size=args.nn_reference_chunk_size,
        utility_exact_evaluator=args.utility_exact_evaluator,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    selection_csv_arg = args.selection_csv
    if selection_csv_arg is None and args.versions_dir is not None and Path(args.versions_dir).is_file():
        selection_csv_arg = Path(args.versions_dir)
    if selection_csv_arg is not None:
        selection_csv = Path(selection_csv_arg)
        selection_name = _selection_name_from_csv(selection_csv, args.selection_name)
        eval_dir = Path(args.eval_dir) if args.eval_dir is not None else selection_csv.parent.parent / "eval"
        metrics = {
            selection_name: evaluate_single_selection(
                config=config,
                selection_name=selection_name,
                df=pd.read_csv(selection_csv),
                eval_dir=eval_dir,
            )
        }
        if args.metrics_output is not None:
            from .io import save_json

            save_json(Path(args.metrics_output), metrics)
    else:
        metrics = run_core_metrics(
            config,
            versions_dir=args.versions_dir,
            eval_dir=args.eval_dir,
            metrics_output=args.metrics_output,
        )
    print(f"Evaluated {len(metrics)} core selections")


if __name__ == "__main__":
    main()
