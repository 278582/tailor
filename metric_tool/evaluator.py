from __future__ import annotations

import time
from typing import Any

import pandas as pd

from .dcr import evaluate_dcr, evaluate_hold_dcr
from .io import save_eval_extras, save_eval_timing, save_json
from .mle import evaluate_mle
from .reward import compute_metric_reward
from .tabdiff_density import TabDiffMetricRunner
from .timing import build_eval_timing


def build_dcr_balance_summary(raw_dcr: Any) -> dict[str, Any]:
    if raw_dcr is None:
        return {
            "raw_dcr_real_closer_rate": None,
            "target_raw_dcr": 0.5,
            "dcr_balance_error_abs": None,
            "dcr_privacy_reward": None,
            "dcr_semantics": "raw DCR is mean(distance_to_train < distance_to_test); best near 0.5; dcr_privacy_reward is higher-better",
        }
    parsed = float(raw_dcr)
    error = abs(parsed - 0.5)
    return {
        "raw_dcr_real_closer_rate": parsed,
        "target_raw_dcr": 0.5,
        "dcr_balance_error_abs": error,
        "dcr_privacy_reward": 1.0 - error,
        "dcr_semantics": "raw DCR is mean(distance_to_train < distance_to_test); best near 0.5; dcr_privacy_reward is higher-better",
    }


def utility_metric_direction(metric: Any) -> str:
    normalized = str(metric or "").strip().lower()
    if normalized in {"rmse", "mse", "mae"}:
        return "lower_better"
    if normalized in {"roc_auc", "auc", "accuracy", "f1", "macro_f1", "balanced_accuracy"}:
        return "higher_better"
    return "unknown"


def build_metric_directions(utility_metric: Any, *, include_hold_dcr: bool = False) -> dict[str, Any]:
    directions = {
        "shape": "higher_better",
        "trend": "higher_better",
        "dcr": "legacy_raw_dcr_real_closer_rate; target is 0.5, not higher/lower better",
        "raw_dcr_real_closer_rate": "target_0.5",
        "dcr_balance_error_abs": "lower_better",
        "dcr_privacy": "legacy_alias_of_dcr_privacy_reward; higher_better",
        "dcr_privacy_reward": "higher_better",
        "privacy_mean_nn_distance": "higher_better",
        "utility_exact_metric": utility_metric,
        "utility_exact_raw_direction": utility_metric_direction(utility_metric),
        "utility_exact_overall": "raw_metric; direction=utility_exact_raw_direction",
        "utility_exact_overall_semantics": "raw_metric",
        "metric_reward_score": "normalized_higher_better",
        "metric_reward_score_direction": "higher_better",
        "metric_reward_score_semantics": "normalized_reward",
    }
    if include_hold_dcr:
        directions.update(
            {
                "dcr_hold": "mean(distance_to_train < distance_to_hold); target is 0.5",
                "dcr_hold_balance_error_abs": "lower_better",
                "dcr_hold_privacy_reward": "higher_better",
            }
        )
    return directions


def build_audit_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    reward_report = summary.get("metric_reward")
    if not isinstance(reward_report, dict):
        reward_report = {}
    dcr_balance = build_dcr_balance_summary(summary.get("dcr"))
    audit = {
        "shape_global": float(summary.get("shape", 0.0)),
        "trend_global": float(summary.get("trend", 0.0)),
        "dcr": summary.get("dcr"),
        "dcr_privacy": summary.get("dcr_privacy"),
        "raw_dcr_real_closer_rate": dcr_balance.get("raw_dcr_real_closer_rate"),
        "dcr_balance_error_abs": dcr_balance.get("dcr_balance_error_abs"),
        "dcr_privacy_reward": dcr_balance.get("dcr_privacy_reward"),
        "utility_exact": summary.get("utility_exact_overall"),
        "utility_exact_available": bool(summary.get("utility_exact_available", False)),
        "utility_exact_metric": summary.get("utility_exact_metric"),
        "utility_exact_raw_direction": summary.get(
            "utility_exact_raw_direction",
            utility_metric_direction(summary.get("utility_exact_metric")),
        ),
        "utility_exact_overall_semantics": summary.get("utility_exact_overall_semantics", "raw_metric"),
        "metric_reward": float(summary.get("metric_reward_score", 0.0)),
        "metric_reward_available": bool(reward_report.get("available", False)),
        "metric_reward_score_direction": summary.get("metric_reward_score_direction", "higher_better"),
        "metric_reward_score_semantics": summary.get("metric_reward_score_semantics", "normalized_reward"),
    }
    if "dcr_hold" in summary:
        audit.update(
            {
                "dcr_hold": summary.get("dcr_hold"),
                "dcr_hold_balance_error_abs": summary.get("dcr_hold_balance_error_abs"),
                "dcr_hold_privacy_reward": summary.get("dcr_hold_privacy_reward"),
            }
        )
    return audit


def _run_selection_metrics(runner: TabDiffMetricRunner, df: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    sdmetrics_seconds = 0.0
    aligned_dcr_seconds = 0.0
    aligned_dcr_timed_repeats = 0
    aligned_dcr_computed_repeats = 0
    aligned_dcr_hold_seconds = 0.0
    aligned_dcr_hold_timed_repeats = 0
    aligned_dcr_hold_computed_repeats = 0
    for metric in runner.metric_list:
        if metric == "density":
            started = time.perf_counter()
            out_metrics, out_extras = runner.evaluate_density(df.copy())
            sdmetrics_seconds = float(time.perf_counter() - started)
        elif metric == "dcr":
            out_metrics, out_extras = evaluate_dcr(runner, df.copy())
            aligned_dcr_seconds = float(out_extras.pop("aligned_dcr_seconds", 0.0) or 0.0)
            aligned_dcr_timed_repeats = int(out_extras.pop("aligned_dcr_timed_repeats", 0) or 0)
            aligned_dcr_computed_repeats = int(out_extras.get("dcr_repeats", 0) or 0)
            if bool(getattr(runner, "enable_hold_dcr", False)):
                hold_metrics, hold_extras = evaluate_hold_dcr(runner, df.copy())
                out_metrics["dcr_hold"] = float(hold_metrics["dcr"])
                if "dcr_target_pool_protocol" in hold_extras:
                    out_extras["dcr_hold_target_pool_protocol"] = hold_extras["dcr_target_pool_protocol"]
                out_extras.update(
                    {
                        "dcr_hold": pd.DataFrame(
                            {
                                "dcr_train": hold_extras["dcr_real"],
                                "dcr_hold": hold_extras["dcr_test"],
                            }
                        ),
                        "dcr_hold_std": float(hold_extras["dcr_std"]),
                        "dcr_hold_n": int(hold_extras["dcr_n"]),
                        "dcr_hold_repeats": int(hold_extras["dcr_repeats"]),
                        "dcr_hold_requested_repeats": int(hold_extras["dcr_requested_repeats"]),
                        "dcr_hold_aligned": bool(hold_extras["dcr_aligned"]),
                        "dcr_hold_align_mode": str(hold_extras["dcr_align_mode"]),
                        "dcr_hold_repeat_rates": list(hold_extras["dcr_repeat_rates"]),
                    }
                )
                aligned_dcr_hold_seconds = float(hold_extras.get("aligned_dcr_seconds", 0.0) or 0.0)
                aligned_dcr_hold_timed_repeats = int(hold_extras.get("aligned_dcr_timed_repeats", 0) or 0)
                aligned_dcr_hold_computed_repeats = int(hold_extras.get("dcr_repeats", 0) or 0)
        else:
            func = getattr(runner.metrics, f"evaluate_{metric}")
            out_metrics, out_extras = func(df.copy())
        metrics.update(out_metrics)
        extras.update(out_extras)
    return {
        "metrics": metrics,
        "extras": extras,
        "sdmetrics_seconds": sdmetrics_seconds,
        "aligned_dcr_seconds": aligned_dcr_seconds,
        "aligned_dcr_timed_repeats": aligned_dcr_timed_repeats,
        "aligned_dcr_computed_repeats": aligned_dcr_computed_repeats,
        "aligned_dcr_hold_seconds": aligned_dcr_hold_seconds,
        "aligned_dcr_hold_timed_repeats": aligned_dcr_hold_timed_repeats,
        "aligned_dcr_hold_computed_repeats": aligned_dcr_hold_computed_repeats,
    }


def evaluate_one_selection(
    *,
    selection_name: str,
    df: pd.DataFrame,
    selector: Any,
    runner: TabDiffMetricRunner,
    eval_dir,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    quality_rng_before = None
    if getattr(runner,"dcr_align_mode",None)=="query_batches_full_train_v1":
        import numpy as np
        quality_rng_before=np.random.get_state()
    measured = _run_selection_metrics(runner, df)
    if quality_rng_before is not None:
        import pickle
        folder=eval_dir/selection_name;folder.mkdir(parents=True,exist_ok=True)
        (folder/'quality_numpy_before.pickle').write_bytes(pickle.dumps(quality_rng_before))
        (folder/'quality_numpy_after.pickle').write_bytes(pickle.dumps(np.random.get_state()))
        np.save(folder/'quality_ordered_query_hashes.npy',pd.util.hash_pandas_object(df,index=False).to_numpy())
        save_json(folder/'quality_sampling.json',dict(
            SDMetrics_contingency_default_subsample_rows=50000,query_rows=len(df),train_path=str(runner.real_data_path),
            replay='Saved initial NumPy state, exact ordered query hashes and original SDMetrics version; no RNG reset during this evaluation',
            selection_name=selection_name))
    metrics = measured["metrics"]
    extras = measured["extras"]

    started = time.perf_counter()
    utility_exact_report = evaluate_mle(selector, df, test_df)
    utility_exact_seconds = float(time.perf_counter() - started)
    extras = {
        **extras,
        "utility_exact_report": utility_exact_report,
    }

    raw_dcr = float(metrics.get("dcr", 0.0)) if "dcr" in metrics else None
    dcr_balance = build_dcr_balance_summary(raw_dcr)
    raw_dcr_hold = float(metrics["dcr_hold"]) if "dcr_hold" in metrics else None
    dcr_hold_balance = build_dcr_balance_summary(raw_dcr_hold) if raw_dcr_hold is not None else None
    utility_metric = utility_exact_report.get("metric")
    utility_direction = utility_metric_direction(utility_metric)
    summary = {
        "rows": int(len(df)),
        "fidelity": selector.compute_dataset_fidelity(df),
        "shape": float(metrics.get("density/Shape", 0.0)),
        "trend": float(metrics.get("density/Trend", 0.0)),
        "density_overall": float(metrics.get("density/Overall", 0.0)),
        "privacy": selector.compute_dataset_privacy(df),
        "privacy_mean_nn_distance": selector.compute_dataset_mean_nn_distance(df),
        "dcr": raw_dcr,
        "dcr_privacy": dcr_balance.get("dcr_privacy_reward"),
        **dcr_balance,
        "metric_directions": build_metric_directions(utility_metric, include_hold_dcr=raw_dcr_hold is not None),
        "utility_exact_metric": utility_exact_report.get("metric"),
        "utility_exact_raw_direction": utility_direction,
        "utility_exact_available": bool(utility_exact_report.get("available", False)),
        "utility_exact_overall": utility_exact_report.get("overall"),
        "utility_exact_overall_semantics": "raw_metric",
        "utility_exact_mode": utility_exact_report.get("mode"),
        "utility_exact_middle": utility_exact_report.get("middle"),
        "utility_exact_tail": utility_exact_report.get("tail"),
        "utility_exact_task_type": utility_exact_report.get("task_type"),
        "utility_exact_tabdiff_task_type": utility_exact_report.get("tabdiff_task_type"),
        "utility_exact_primary_score_group": utility_exact_report.get("primary_score_group"),
        "utility_exact_primary_model": utility_exact_report.get("primary_model"),
        "utility_exact_regression_target_transform": utility_exact_report.get("regression_target_transform"),
        "utility_exact_regression_target_clip_min": utility_exact_report.get("regression_target_clip_min"),
        "utility_exact_regression_target_clip_max": utility_exact_report.get("regression_target_clip_max"),
        "utility_exact_regression_target_raw_std": utility_exact_report.get("regression_target_raw_std"),
        "utility_exact_regression_target_eval_std": utility_exact_report.get("regression_target_eval_std"),
        "metric_reward_score_direction": "higher_better",
        "metric_reward_score_semantics": "normalized_reward",
    }
    if raw_dcr_hold is not None and dcr_hold_balance is not None:
        summary.update(
            {
                "dcr_hold": raw_dcr_hold,
                "dcr_hold_raw_train_closer_rate": dcr_hold_balance["raw_dcr_real_closer_rate"],
                "dcr_hold_target_raw_dcr": 0.5,
                "dcr_hold_balance_error_abs": dcr_hold_balance["dcr_balance_error_abs"],
                "dcr_hold_privacy_reward": dcr_hold_balance["dcr_privacy_reward"],
                "dcr_hold_semantics": "raw DCR is mean(distance_to_train < distance_to_hold); best near 0.5",
                "dcr_hold_std": extras.get("dcr_hold_std"),
                "dcr_hold_n": extras.get("dcr_hold_n"),
                "dcr_hold_repeats": extras.get("dcr_hold_repeats"),
            }
        )

    if getattr(runner, "dcr_align_mode", None) == "target_pool_mc":
        summary["dcr_align_mode"] = "target_pool_mc"
        summary["dcr_semantics"] = "Random equal train target pools; full query; exact ties half; mean round shares; 0.5 conditional reference"
        summary["dcr_hold_semantics"] = summary["dcr_semantics"]
        summary["dcr_target_pool_protocol"] = extras.get("dcr_target_pool_protocol")
        summary["dcr_hold_target_pool_protocol"] = extras.get("dcr_hold_target_pool_protocol")
    if getattr(runner,"dcr_align_mode",None)=="query_batches_full_train_v1":
        from .query_batches import save_details
        save_details(runner,summary,eval_dir/selection_name)
    started = time.perf_counter()
    save_eval_extras(eval_dir=eval_dir, selection_name=selection_name, extras=extras)
    save_json(eval_dir / selection_name / "utility_metrics_summary.json", utility_exact_report)
    reward_report = compute_metric_reward(
        summary=summary,
        selector=selector,
        test_df=test_df,
    )
    summary["metric_reward"] = reward_report
    summary["metric_reward_score"] = float(reward_report.get("reward", 0.0))
    summary["audit_metrics"] = build_audit_metrics(summary)
    summary["eval_timing"] = build_eval_timing(
        sdmetrics_seconds=measured["sdmetrics_seconds"],
        aligned_dcr_seconds=measured["aligned_dcr_seconds"],
        aligned_dcr_timed_repeats=measured["aligned_dcr_timed_repeats"],
        aligned_dcr_computed_repeats=measured["aligned_dcr_computed_repeats"],
        aligned_dcr_hold_seconds=measured["aligned_dcr_hold_seconds"],
        aligned_dcr_hold_timed_repeats=measured["aligned_dcr_hold_timed_repeats"],
        aligned_dcr_hold_computed_repeats=measured["aligned_dcr_hold_computed_repeats"],
        utility_exact_seconds=utility_exact_seconds,
        summary_seconds=0.0,
    )
    save_eval_timing(eval_dir, selection_name, summary)
    summary["eval_timing"] = build_eval_timing(
        sdmetrics_seconds=measured["sdmetrics_seconds"],
        aligned_dcr_seconds=measured["aligned_dcr_seconds"],
        aligned_dcr_timed_repeats=measured["aligned_dcr_timed_repeats"],
        aligned_dcr_computed_repeats=measured["aligned_dcr_computed_repeats"],
        aligned_dcr_hold_seconds=measured["aligned_dcr_hold_seconds"],
        aligned_dcr_hold_timed_repeats=measured["aligned_dcr_hold_timed_repeats"],
        aligned_dcr_hold_computed_repeats=measured["aligned_dcr_hold_computed_repeats"],
        utility_exact_seconds=utility_exact_seconds,
        summary_seconds=float(time.perf_counter() - started),
    )
    save_eval_timing(eval_dir, selection_name, summary)
    return summary
