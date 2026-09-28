from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from post_selection_tool.census_profile import apply_census_runtime_profile, bind_selector_utility_runtime
from post_selection_tool.context import build_artifact_paths, resolve_eval_device, resolve_nn_device
from post_selection_tool.selector import ParetoSelector
from postprocess.tabdiff_protocol import resolve_tabdiff_selection_context

from .config import MetricConfig
from .evaluator import build_audit_metrics, evaluate_one_selection
from .io import load_core_selection_frames, load_json, save_eval_timing, save_json
from .tabdiff_density import TabDiffMetricRunner
from .timing import add_summary_elapsed


def prepare_metric_objects(config: MetricConfig) -> dict[str, Any]:
    config = apply_census_runtime_profile(config)
    dataset_ctx = resolve_tabdiff_selection_context(
        dataset_name=config.dataset_name,
        seed=config.seed,
        holdout_fraction=config.holdout_fraction,
    )
    artifact_root = dataset_ctx.artifact_root if config.artifact_dir is None else config.artifact_dir
    paths = build_artifact_paths(config, artifact_root)
    schema_card = load_json(paths.cards_dir / "schema_card.json")
    stats_card = load_json(paths.cards_dir / "stats_card.json")
    eval_device = resolve_eval_device(config.eval_device)
    nn_device = resolve_nn_device(config.nn_device, eval_device)
    selector = ParetoSelector(
        train_df=dataset_ctx.train_df.copy(),
        holdout_df=dataset_ctx.holdout_df.copy(),
        schema_card=schema_card,
        stats_card=stats_card,
        seed=config.seed,
        source="tabdiff",
        privacy_version=config.privacy_version,
        density_reference_size=config.density_reference_size,
        nn_device=nn_device,
        nn_query_batch_size=config.nn_query_batch_size,
        nn_reference_chunk_size=config.nn_reference_chunk_size,
        high_cardinality_enabled=False,
    )
    bind_selector_utility_runtime(selector, config)
    runner = TabDiffMetricRunner(
        dataset_name=config.dataset_name,
        device=eval_device,
        metric_list=["density", "dcr"],
        real_data_path=dataset_ctx.train_source_path,
        test_data_path=dataset_ctx.test_source_path,
        val_data_path=dataset_ctx.holdout_source_path,
        nn_query_batch_size=config.nn_query_batch_size,
        nn_reference_chunk_size=config.nn_reference_chunk_size,
        dcr_repeats=int(getattr(config, "dcr_repeats", 10)),
        dcr_cap=int(getattr(config, "dcr_cap", 0) or 0),
        dcr_seed=int(getattr(config, "dcr_seed", config.seed)),
    )
    return {
        "dataset_ctx": dataset_ctx,
        "paths": paths,
        "selector": selector,
        "runner": runner,
    }


def run_core_metrics(
    config: MetricConfig,
    *,
    versions_dir: Path | None = None,
    eval_dir: Path | None = None,
    metrics_output: Path | None = None,
) -> dict[str, dict[str, Any]]:
    objects = prepare_metric_objects(config)
    paths = objects["paths"]
    source_versions_dir = paths.versions_dir if versions_dir is None else Path(versions_dir)
    frames = load_core_selection_frames(source_versions_dir)
    if not frames:
        raise FileNotFoundError(f"No core selection CSVs found in {source_versions_dir}")

    selection_metrics: dict[str, dict[str, Any]] = {}
    target_eval_dir = paths.artifact_dir / "eval" if eval_dir is None else Path(eval_dir)
    for selection_name, df in frames.items():
        selection_metrics[selection_name] = evaluate_one_selection(
            selection_name=selection_name,
            df=df,
            selector=objects["selector"],
            runner=objects["runner"],
            eval_dir=target_eval_dir,
            test_df=objects["dataset_ctx"].test_df.copy(),
        )
    output_path = paths.report_dir / "core_metrics_summary.json" if metrics_output is None else Path(metrics_output)
    save_json(output_path, selection_metrics)
    return selection_metrics


def _load_single_metric_extras(eval_dir, selection_name: str) -> dict[str, Any]:
    target_dir = eval_dir / selection_name
    extras: dict[str, Any] = {}
    for key, filename in (("shape_details", "shapes.csv"), ("trend_details", "trends.csv")):
        path = target_dir / filename
        if path.exists():
            try:
                extras[key] = pd.read_csv(path).to_dict(orient="records")
            except Exception as exc:
                extras[key] = {"available": False, "reason": f"failed_to_read_{filename}", "error": str(exc)}
        else:
            extras[key] = {"available": False, "reason": f"{filename}_not_found"}
    utility_path = target_dir / "utility_metrics_summary.json"
    if utility_path.exists():
        extras["utility_exact_report"] = load_json(utility_path)
    return extras


def evaluate_single_selection(
    *,
    config: MetricConfig,
    selection_name: str,
    df,
    eval_dir,
) -> dict[str, Any]:
    objects = prepare_metric_objects(config)
    summary = evaluate_one_selection(
        selection_name=selection_name,
        df=df,
        selector=objects["selector"],
        runner=objects["runner"],
        eval_dir=eval_dir,
        test_df=objects["dataset_ctx"].test_df.copy(),
    )
    started = time.perf_counter()
    summary["audit_metrics"] = build_audit_metrics(summary)
    summary["metric_extras"] = _load_single_metric_extras(eval_dir, selection_name)
    summary["eval_timing"] = add_summary_elapsed(
        summary.get("eval_timing"),
        time.perf_counter() - started,
    )
    save_eval_timing(eval_dir, selection_name, summary)
    return summary
