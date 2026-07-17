from __future__ import annotations

from typing import Any

import pandas as pd

from .io import save_eval_extras, save_json
from .mle import evaluate_mle
from .reward import compute_metric_reward
from .tabdiff_density import TabDiffMetricRunner


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


def build_metric_directions(utility_metric: Any) -> dict[str, Any]:
    return {
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


def build_audit_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    reward_report = summary.get("metric_reward")
    if not isinstance(reward_report, dict):
        reward_report = {}
    dcr_balance = build_dcr_balance_summary(summary.get("dcr"))
    return {
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


def evaluate_one_selection(
    *,
    selection_name: str,
    df: pd.DataFrame,
    selector: Any,
    runner: TabDiffMetricRunner,
    eval_dir,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    metrics, extras = runner.evaluate(df)
    utility_exact_report = evaluate_mle(selector, df, test_df)
    extras = {
        **extras,
        "utility_exact_report": utility_exact_report,
    }
    save_eval_extras(eval_dir=eval_dir, selection_name=selection_name, extras=extras)
    save_json(eval_dir / selection_name / "utility_metrics_summary.json", utility_exact_report)

    raw_dcr = float(metrics.get("dcr", 0.0)) if "dcr" in metrics else None
    dcr_balance = build_dcr_balance_summary(raw_dcr)
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
        "metric_directions": build_metric_directions(utility_metric),
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
    reward_report = compute_metric_reward(
        summary=summary,
        selector=selector,
        test_df=test_df,
    )
    summary["metric_reward"] = reward_report
    summary["metric_reward_score"] = float(reward_report.get("reward", 0.0))
    summary["audit_metrics"] = build_audit_metrics(summary)
    save_json(eval_dir / selection_name / "metrics_summary.json", summary)
    return summary
