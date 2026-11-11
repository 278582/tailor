from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd


DEFAULT_METRIC_REWARD_RHO = 0.05
DEFAULT_METRIC_REWARD_WEIGHTS = {
    "shape": 0.25,
    "trend": 0.25,
    "privacy": 0.25,
    "utility": 0.25,
}
REGRESSION_UTILITY_TARGET_TRANSFORM = "log_clip_1_20000"
REGRESSION_UTILITY_TARGET_CLIP_MIN = 1.0
REGRESSION_UTILITY_TARGET_CLIP_MAX = 20000.0
REGRESSION_UTILITY_RELATIVE_RMSE_SIGMOID_SLOPE = 4.0


def _as_finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _regression_target_eval_scale(values: Any) -> np.ndarray:
    return np.log(
        np.clip(
            np.asarray(values, dtype=float),
            REGRESSION_UTILITY_TARGET_CLIP_MIN,
            REGRESSION_UTILITY_TARGET_CLIP_MAX,
        )
    )


def _score_unit_interval(value: Any, *, component: str) -> dict[str, Any]:
    raw = _as_finite_float(value)
    if raw is None:
        return {
            "available": False,
            "component": component,
            "reason": "missing_or_non_finite_value",
            "raw": None,
            "score": 0.0,
            "direction": "higher_better",
            "normalization": "clip(raw, 0, 1)",
        }
    return {
        "available": True,
        "component": component,
        "raw": raw,
        "score": float(np.clip(raw, 0.0, 1.0)),
        "direction": "higher_better",
        "normalization": "clip(raw, 0, 1)",
    }


def _score_dcr_privacy(summary: dict[str, Any]) -> dict[str, Any]:
    raw_dcr = _as_finite_float(summary.get("dcr"))
    if raw_dcr is None:
        return {
            "available": False,
            "component": "privacy",
            "reason": "missing_or_non_finite_dcr",
            "raw_dcr": raw_dcr,
            "raw_dcr_privacy": _as_finite_float(summary.get("dcr_privacy")),
            "score": 0.0,
            "direction": "higher_better",
            "ideal_dcr": 0.5,
            "normalization": "clip(1 - abs(dcr - 0.5), 0, 1)",
        }
    raw_dcr_privacy = 1.0 - abs(raw_dcr - 0.5)
    return {
        "available": True,
        "component": "privacy",
        "raw_dcr": raw_dcr,
        "raw_dcr_privacy": raw_dcr_privacy,
        "score": float(np.clip(raw_dcr_privacy, 0.0, 1.0)),
        "direction": "higher_better",
        "ideal_dcr": 0.5,
        "normalization": "clip(1 - abs(dcr - 0.5), 0, 1)",
    }


def _score_classification_utility(
    *,
    metric: str | None,
    raw_value: float | None,
    task_type: str | None,
    tabdiff_task_type: str | None,
    primary_score_group: str | None,
    primary_model: str | None,
) -> dict[str, Any]:
    metric_name = str(metric or "").strip()
    metric_key = metric_name.lower()
    base = {
        "component": "utility",
        "task_type": task_type,
        "tabdiff_task_type": tabdiff_task_type,
        "metric": metric_name or None,
        "raw": raw_value,
        "primary_score_group": primary_score_group,
        "primary_model": primary_model,
    }
    if raw_value is None:
        return {
            **base,
            "available": False,
            "reason": "missing_or_non_finite_utility_value",
            "score": 0.0,
            "direction": "higher_better",
        }
    if metric_key != "roc_auc":
        return {
            **base,
            "available": False,
            "reason": "unsupported_classification_utility_metric",
            "score": 0.0,
            "direction": "higher_better",
            "supported_metrics": ["roc_auc"],
        }
    score = float(np.clip((raw_value - 0.5) / 0.5, 0.0, 1.0))
    return {
        **base,
        "available": True,
        "score": score,
        "direction": "higher_better",
        "random_baseline": 0.5,
        "perfect_score": 1.0,
        "normalization": "clip((roc_auc - 0.5) / 0.5, 0, 1)",
    }


def _score_regression_utility(
    *,
    metric: str | None,
    raw_value: float | None,
    task_type: str | None,
    tabdiff_task_type: str | None,
    primary_score_group: str | None,
    primary_model: str | None,
    selector: Any,
    test_df: pd.DataFrame,
    eps: float = 1e-12,
) -> dict[str, Any]:
    metric_name = str(metric or "").strip()
    metric_key = metric_name.lower()
    base = {
        "component": "utility",
        "task_type": task_type,
        "tabdiff_task_type": tabdiff_task_type,
        "metric": metric_name or None,
        "raw": raw_value,
        "primary_score_group": primary_score_group,
        "primary_model": primary_model,
    }
    if raw_value is None:
        return {
            **base,
            "available": False,
            "reason": "missing_or_non_finite_utility_value",
            "score": 0.0,
            "direction": "lower_better",
        }
    if metric_key != "rmse":
        return {
            **base,
            "available": False,
            "reason": "unsupported_regression_utility_metric",
            "score": 0.0,
            "direction": "lower_better",
            "supported_metrics": ["RMSE"],
        }

    target_column = getattr(selector, "target_column", None)
    if not target_column or target_column not in test_df.columns:
        return {
            **base,
            "available": False,
            "reason": "missing_test_target_column",
            "score": 0.0,
            "direction": "lower_better",
            "target_column": target_column,
        }

    y_test = pd.to_numeric(test_df[target_column], errors="coerce").dropna().to_numpy(dtype=float)
    if y_test.size < 2:
        return {
            **base,
            "available": False,
            "reason": "insufficient_test_target_values",
            "score": 0.0,
            "direction": "lower_better",
            "target_column": target_column,
            "test_target_rows": int(y_test.size),
        }

    target_raw_std = float(np.std(y_test, ddof=0))
    y_test_eval_scale = _regression_target_eval_scale(y_test)
    target_std = float(np.std(y_test_eval_scale, ddof=0))
    if not math.isfinite(target_std) or target_std <= eps:
        return {
            **base,
            "available": False,
            "reason": "near_constant_test_target",
            "score": 0.0,
            "direction": "lower_better",
            "target_column": target_column,
            "target_std": target_std,
            "target_raw_std": target_raw_std,
            "target_transform": REGRESSION_UTILITY_TARGET_TRANSFORM,
            "target_clip_min": REGRESSION_UTILITY_TARGET_CLIP_MIN,
            "target_clip_max": REGRESSION_UTILITY_TARGET_CLIP_MAX,
        }

    relative_rmse = float(raw_value / (target_std + eps))
    improvement_vs_mean = float(1.0 - relative_rmse)
    sigmoid_logit = float(REGRESSION_UTILITY_RELATIVE_RMSE_SIGMOID_SLOPE * improvement_vs_mean)
    score = float(1.0 / (1.0 + math.exp(-float(np.clip(sigmoid_logit, -60.0, 60.0)))))
    return {
        **base,
        "available": True,
        "score": score,
        "direction": "lower_better",
        "target_column": target_column,
        "target_std": target_std,
        "target_raw_std": target_raw_std,
        "target_transform": REGRESSION_UTILITY_TARGET_TRANSFORM,
        "target_clip_min": REGRESSION_UTILITY_TARGET_CLIP_MIN,
        "target_clip_max": REGRESSION_UTILITY_TARGET_CLIP_MAX,
        "relative_rmse": relative_rmse,
        "improvement_vs_mean_baseline": improvement_vs_mean,
        "baseline": "log_clipped_test_target_mean_predictor",
        "sigmoid_slope": REGRESSION_UTILITY_RELATIVE_RMSE_SIGMOID_SLOPE,
        "normalization": "sigmoid(4 * (1 - rmse / std(log(clip(y_test, 1, 20000)))))",
    }


def _score_utility(
    *,
    summary: dict[str, Any],
    selector: Any,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    if not bool(summary.get("utility_exact_available", False)):
        return {
            "available": False,
            "component": "utility",
            "reason": "utility_exact_unavailable",
            "score": 0.0,
            "metric": summary.get("utility_exact_metric"),
            "raw": summary.get("utility_exact_overall"),
        }

    task_type = summary.get("utility_exact_task_type")
    tabdiff_task_type = summary.get("utility_exact_tabdiff_task_type")
    metric = summary.get("utility_exact_metric")
    raw_value = _as_finite_float(summary.get("utility_exact_overall"))
    primary_score_group = summary.get("utility_exact_primary_score_group")
    primary_model = summary.get("utility_exact_primary_model")

    normalized_task = str(task_type or "").strip().lower()
    normalized_tabdiff_task = str(tabdiff_task_type or "").strip().lower()
    if normalized_task == "classification" or normalized_tabdiff_task in {"binclass", "multiclass"}:
        return _score_classification_utility(
            metric=metric,
            raw_value=raw_value,
            task_type=task_type,
            tabdiff_task_type=tabdiff_task_type,
            primary_score_group=primary_score_group,
            primary_model=primary_model,
        )
    if normalized_task == "regression" or normalized_tabdiff_task == "regression":
        return _score_regression_utility(
            metric=metric,
            raw_value=raw_value,
            task_type=task_type,
            tabdiff_task_type=tabdiff_task_type,
            primary_score_group=primary_score_group,
            primary_model=primary_model,
            selector=selector,
            test_df=test_df,
        )
    return {
        "available": False,
        "component": "utility",
        "reason": "unknown_utility_task_type",
        "score": 0.0,
        "task_type": task_type,
        "tabdiff_task_type": tabdiff_task_type,
        "metric": metric,
        "raw": raw_value,
    }


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    use_weights = dict(DEFAULT_METRIC_REWARD_WEIGHTS if weights is None else weights)
    normalized: dict[str, float] = {}
    for key in DEFAULT_METRIC_REWARD_WEIGHTS:
        value = _as_finite_float(use_weights.get(key))
        normalized[key] = max(0.0, float(value)) if value is not None else 0.0
    total = float(sum(normalized.values()))
    if total <= 1e-12:
        return dict(DEFAULT_METRIC_REWARD_WEIGHTS)
    return {key: float(value / total) for key, value in normalized.items()}


def _shifted_weighted_geomean(
    *,
    scores: dict[str, float],
    weights: dict[str, float],
    rho: float,
) -> dict[str, Any]:
    safe_rho = max(0.0, float(rho))
    q_min = safe_rho / (1.0 + safe_rho)
    shifted_scores = {
        key: float((np.clip(value, 0.0, 1.0) + safe_rho) / (1.0 + safe_rho))
        for key, value in scores.items()
    }
    log_sum = 0.0
    for key, shifted_score in shifted_scores.items():
        log_sum += float(weights[key]) * math.log(max(shifted_score, 1e-300))
    raw_geomean = float(math.exp(log_sum))
    denom = 1.0 - q_min
    reward = 0.0 if denom <= 1e-12 else float((raw_geomean - q_min) / denom)
    reward = float(np.clip(reward, 0.0, 1.0))
    return {
        "type": "shifted_weighted_geomean",
        "shift_rho": safe_rho,
        "shifted_scores": shifted_scores,
        "raw_geomean": raw_geomean,
        "rebased_from": q_min,
        "reward": reward,
        "formula": "clip((prod(((z+rho)/(1+rho))^w)-rho/(1+rho))/(1-rho/(1+rho)),0,1)",
    }


def compute_metric_reward(
    *,
    summary: dict[str, Any],
    selector: Any,
    test_df: pd.DataFrame,
    rho: float = DEFAULT_METRIC_REWARD_RHO,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    normalized_weights = _normalize_weights(weights)
    components = {
        "shape": _score_unit_interval(summary.get("shape"), component="shape"),
        "trend": _score_unit_interval(summary.get("trend"), component="trend"),
        "privacy": _score_dcr_privacy(summary),
        "utility": _score_utility(summary=summary, selector=selector, test_df=test_df),
    }
    unavailable = [name for name, report in components.items() if not bool(report.get("available", False))]
    if unavailable:
        return {
            "available": False,
            "method": "metric_tool_shifted_4d_geomean_reward",
            "reward": 0.0,
            "reason": "component_unavailable",
            "unavailable_components": unavailable,
            "rho": float(rho),
            "weights": normalized_weights,
            "components": components,
        }

    scores = {name: float(report["score"]) for name, report in components.items()}
    aggregation = _shifted_weighted_geomean(scores=scores, weights=normalized_weights, rho=rho)
    return {
        "available": True,
        "method": "metric_tool_shifted_4d_geomean_reward",
        "reward": float(aggregation["reward"]),
        "rho": float(rho),
        "weights": normalized_weights,
        "components": components,
        "aggregation": aggregation,
    }
