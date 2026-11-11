from __future__ import annotations

import math
from typing import Any

from .strategy import StrategyTheta


def _finite(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed)


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _theta_to_dict(theta: StrategyTheta) -> dict[str, Any]:
    return {
        "col_1ds": list(theta.col_1ds),
        "col_2ds": list(theta.col_2ds),
        "col_ps": list(theta.col_ps),
        "col_u": theta.col_u,
    }


def _row_score(row: dict[str, Any]) -> float | None:
    for key in ("Score", "score", "Quality Score", "QualityScore"):
        if key in row and _finite(row[key]):
            return float(row[key])
    return None


def _column_name(value: Any, column_lookup: dict[str, str | None]) -> str | None:
    if value is None:
        return None
    raw = str(value)
    return column_lookup.get(raw)


def _weak_shape_columns(
    metric_extras: dict[str, Any],
    column_lookup: dict[str, str | None],
    limit: int = 5,
) -> list[dict[str, Any]]:
    details = metric_extras.get("shape_details")
    if not isinstance(details, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in details:
        if not isinstance(row, dict):
            continue
        score = _row_score(row)
        if score is None:
            continue
        column = row.get("Column") or row.get("column") or row.get("Column Name") or row.get("ColumnName")
        mapped_column = _column_name(column, column_lookup)
        if mapped_column is None:
            continue
        rows.append(
            {
                "column": mapped_column,
                "score": float(score),
                "metric": row.get("Metric") or row.get("metric"),
            }
        )
    rows.sort(key=lambda item: (item["score"], str(item.get("column"))))
    return rows[:limit]


def _weak_trend_pairs(
    metric_extras: dict[str, Any],
    column_lookup: dict[str, str | None],
    limit: int = 5,
) -> list[dict[str, Any]]:
    details = metric_extras.get("trend_details")
    if not isinstance(details, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in details:
        if not isinstance(row, dict):
            continue
        score = _row_score(row)
        if score is None:
            continue
        left = row.get("Column 1") or row.get("Column1") or row.get("column_1") or row.get("left")
        right = row.get("Column 2") or row.get("Column2") or row.get("column_2") or row.get("right")
        mapped_left = _column_name(left, column_lookup)
        mapped_right = _column_name(right, column_lookup)
        if mapped_left is None or mapped_right is None:
            continue
        rows.append(
            {
                "left": mapped_left,
                "right": mapped_right,
                "score": float(score),
                "metric": row.get("Metric") or row.get("metric"),
            }
        )
    rows.sort(key=lambda item: (item["score"], str(item.get("left")), str(item.get("right"))))
    return rows[:limit]


def build_guard(
    audit_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shape_finite = _finite(audit_metrics.get("shape_global"))
    trend_finite = _finite(audit_metrics.get("trend_global"))
    dcr_finite = _finite(audit_metrics.get("dcr"))
    reward_available = bool(audit_metrics.get("metric_reward_available", _finite(audit_metrics.get("metric_reward"))))
    utility_available = bool(audit_metrics.get("utility_exact_available", False))
    checks = {
        "shape_finite": bool(shape_finite),
        "trend_finite": bool(trend_finite),
        "dcr_finite": bool(dcr_finite),
        "metric_reward_available": bool(reward_available),
        "utility_exact_available": bool(utility_available),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "pass": not failed,
        **checks,
        "failed_checks": failed,
        "baseline_checked": baseline_metrics is not None,
    }


def build_rollout_feedback(
    theta: StrategyTheta,
    search_objectives: dict[str, Any],
    audit_metrics: dict[str, Any],
    metric_extras: dict[str, Any],
    internal_reports: dict[str, Any],
    column_lookup: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    lookup = {} if column_lookup is None else dict(column_lookup)
    shape_weak_columns = _weak_shape_columns(metric_extras, lookup)
    trend_weak_pairs = _weak_trend_pairs(metric_extras, lookup)
    raw_dcr = _finite_float(audit_metrics.get("dcr"))
    dcr_privacy = _finite_float(audit_metrics.get("dcr_privacy"))
    dcr_balance_error = abs(raw_dcr - 0.5) if raw_dcr is not None else None
    feedback = {
        "theta": _theta_to_dict(theta),
        "shape_weak_columns": shape_weak_columns,
        "trend_weak_pairs": trend_weak_pairs,
        "privacy_summary": {
            "search_privacy": search_objectives.get("P_theta"),
            "search_privacy_raw": search_objectives.get("P_theta_raw"),
            "dcr": audit_metrics.get("dcr"),
            "dcr_privacy": audit_metrics.get("dcr_privacy"),
            "raw_dcr_real_closer_rate": raw_dcr,
            "target_raw_dcr": 0.5,
            "balance_error_abs": dcr_balance_error,
            "dcr_privacy_reward": dcr_privacy,
            "dcr_semantics": "raw DCR is best near 0.5; dcr_privacy_reward is higher-better",
        },
        "utility_summary": {
            "search_utility_proxy": search_objectives.get("U_proxy_theta"),
            "utility_exact": audit_metrics.get("utility_exact"),
            "utility_exact_available": audit_metrics.get("utility_exact_available"),
        },
        "search_objective_summary": dict(search_objectives),
        "audit_summary": dict(audit_metrics),
        "internal_summary": {
            "preselect_status": internal_reports.get("preselect_status", {}),
            "preselect_report": internal_reports.get("preselect_report", {}),
            "fidelity_ceiling_mode": internal_reports.get("fidelity_ceiling_report", {}).get("mode"),
            "utility_proxy_manifest": internal_reports.get("utility_proxy_manifest", {}),
        },
    }
    if not shape_weak_columns:
        feedback["shape_weak_columns_reason"] = "shape details unavailable or no scored rows"
    if not trend_weak_pairs:
        feedback["trend_weak_pairs_reason"] = "trend details unavailable or no scored rows"
    return feedback
