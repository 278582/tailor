from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .strategy import theta_size_bounds
from .tree import MCTSNode, theta_to_dict


REFINE_ALLOWED_ACTIONS = [
    "add_col_1d",
    "replace_col_1d",
    "add_col_2d",
    "replace_col_2d",
    "add_col_p",
    "replace_col_p",
    "replace_col_u",
]


def schema_summary(schema_card: dict[str, Any]) -> dict[str, Any]:
    columns = schema_card.get("columns", {})
    return {
        "dataset": schema_card.get("dataset"),
        "target_column": schema_card.get("target_column"),
        "feature_columns": [
            column for column in schema_card.get("column_order", []) if not bool(columns.get(column, {}).get("is_target", False))
        ],
        "column_types": {column: info.get("type") for column, info in columns.items()},
    }


def dataset_context_from_schema(schema_card: dict[str, Any]) -> dict[str, Any]:
    columns = schema_card.get("columns", {})
    feature_columns = [
        column for column in schema_card.get("column_order", []) if not bool(columns.get(column, {}).get("is_target", False))
    ]
    numerical_columns = [column for column in feature_columns if columns.get(column, {}).get("type") == "numerical"]
    categorical_columns = [column for column in feature_columns if columns.get(column, {}).get("type") == "categorical"]
    discrete_columns = [column for column in feature_columns if columns.get(column, {}).get("type") == "discrete_numerical"]
    return {
        "schema_version": "llm_mcts_dataset_context_v2_compact_fallback",
        "dataset": schema_card.get("dataset"),
        "task_type": "unknown",
        "target_column": schema_card.get("target_column"),
        "columns": {
            "feature": feature_columns,
            "num": numerical_columns,
            "cat": categorical_columns,
            "discrete_num": discrete_columns,
            "privacy_configured": [],
            "privacy_domain": [],
        },
        "theta_guidance": {
            "shape_priority": feature_columns,
            "trend_priority": feature_columns,
            "privacy_priority": feature_columns[: min(5, len(feature_columns))],
            "utility_priority": feature_columns,
        },
        "pair_priors": [],
        "risks": {"shape": [], "trend": [], "privacy": [], "utility": []},
    }


def load_dataset_prompt_context(
    *,
    dataset_name: str | None,
    schema_card: dict[str, Any],
    prompt_pack_dir: Path | str | None = None,
) -> dict[str, Any]:
    if prompt_pack_dir is not None and dataset_name:
        path = Path(prompt_pack_dir) / "dataset_contexts" / f"{dataset_name}.prompt_context.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
    return dataset_context_from_schema(schema_card)


def _compact_dataset_context(context: dict[str, Any], *, include_seeds: bool) -> dict[str, Any]:
    compact = dict(context)
    if not include_seeds:
        compact.pop("seed_theta_examples", None)
    return compact


def _theta_size_guidance(context: dict[str, Any]) -> dict[str, Any]:
    columns = context.get("columns", {}) if isinstance(context.get("columns"), dict) else {}
    feature_columns = list(columns.get("feature", []) or [])
    feature_count = len(feature_columns)
    return {"feature_count": feature_count, **theta_size_bounds(feature_count)}


def _prompt_number(value: Any, digits: int = 4) -> Any:
    if isinstance(value, bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _dcr_balance_summary(raw_dcr: Any, dcr_privacy: Any = None) -> dict[str, Any]:
    raw = _finite_float(raw_dcr)
    privacy = _finite_float(dcr_privacy)
    if raw is None and privacy is None:
        return {"available": False}
    error = abs(raw - 0.5) if raw is not None else None
    if privacy is None and error is not None:
        privacy = 1.0 - error
    return {
        "available": True,
        "raw_dcr_real_closer_rate": _prompt_number(raw),
        "target_raw_dcr": 0.5,
        "dcr_balance_error_abs": _prompt_number(error),
        "dcr_privacy_reward": _prompt_number(privacy),
        "semantics": "raw DCR is best near 0.5; DCR privacy reward is higher-better",
    }


def _refine_dataset_context(context: dict[str, Any], *, include_dataset_priors: bool = True) -> dict[str, Any]:
    columns = context.get("columns", {}) if isinstance(context.get("columns"), dict) else {}
    theta_guidance = context.get("theta_guidance", {}) if isinstance(context.get("theta_guidance"), dict) else {}
    brief = {
        "dataset": context.get("dataset"),
        "target_column": context.get("target_column"),
        "columns": {
            key: columns.get(key, [])
            for key in ("feature", "privacy_configured", "privacy_domain")
            if key in columns
        },
    }
    if include_dataset_priors:
        brief["theta_guidance"] = {
            key: list(value or [])[:6]
            for key, value in theta_guidance.items()
        }
        brief["pair_priors"] = list(context.get("pair_priors", []) or [])[:6]
    return brief


def _dataset_brief(context: dict[str, Any], *, include_dataset_priors: bool = True) -> dict[str, Any]:
    brief = _refine_dataset_context(context, include_dataset_priors=include_dataset_priors)
    brief["theta_size_guidance"] = _theta_size_guidance(brief)
    return brief


def _compact_node(node: MCTSNode | None) -> dict[str, Any] | None:
    if node is None:
        return None
    return {
        "node_id": node.node_id,
        "theta_id": node.theta_id,
        "theta": theta_to_dict(node.theta),
        "depth": int(node.depth),
        "Q_self": float(node.Q_self),
        "Q": float(node.Q),
        "N": int(node.N),
        "prior_score": float(node.p),
        "reward_available": bool(node.reward_available),
        "guard_pass": bool(node.guard_pass),
        "rollout_status": node.rollout_status,
        "search_objectives": dict(node.search_objectives),
        "audit_metrics": dict(node.audit_metrics),
        "guard_failed_checks": list(node.guard.get("failed_checks", [])) if isinstance(node.guard, dict) else [],
        "actions": list(node.actions),
        "action_validation": dict(node.action_validation),
        "reason": node.reason,
    }


def _metric_summary(search_objectives: dict[str, Any] | None, audit_metrics: dict[str, Any] | None) -> dict[str, Any]:
    search = search_objectives if isinstance(search_objectives, dict) else {}
    audit = audit_metrics if isinstance(audit_metrics, dict) else {}
    output: dict[str, Any] = {}
    for key in ("F_1D_theta", "F_2D_theta", "P_theta", "P_theta_raw", "U_proxy_theta"):
        if key in search:
            output[key] = _prompt_number(search.get(key))
    for key in ("shape_global", "trend_global", "utility_exact", "metric_reward"):
        if key in audit:
            output[key] = _prompt_number(audit.get(key))
    if "dcr" in audit or "dcr_privacy" in audit:
        output["dcr_balance"] = _dcr_balance_summary(audit.get("dcr"), audit.get("dcr_privacy"))
    return output


def _diversity_metric_summary(search_objectives: dict[str, Any] | None, audit_metrics: dict[str, Any] | None) -> dict[str, Any]:
    search = search_objectives if isinstance(search_objectives, dict) else {}
    audit = audit_metrics if isinstance(audit_metrics, dict) else {}
    output: dict[str, Any] = {}
    for key in ("shape_global", "trend_global", "utility_exact", "metric_reward"):
        if key in audit:
            output[key] = _prompt_number(audit.get(key))
    if "dcr" in audit or "dcr_privacy" in audit:
        output["dcr_balance"] = _dcr_balance_summary(audit.get("dcr"), audit.get("dcr_privacy"))
    if "P_theta" in search:
        output["P_theta"] = _prompt_number(search.get("P_theta"))
    return output


def _compact_action(action: dict[str, Any]) -> dict[str, Any] | None:
    action_type = str(action.get("type", "") or "").strip()
    if not action_type:
        return None
    new_column = action.get("new") or action.get("column")
    output: dict[str, Any] = {"type": action_type}
    if action_type.startswith("replace_") and action.get("old"):
        output["old"] = action.get("old")
    if new_column:
        output["new"] = new_column
    return output


def _compact_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        item = _compact_action(action)
        if item is not None:
            compact.append(item)
    return compact


def _compact_action_validation(report: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if "ok" in report:
        output["ok"] = bool(report.get("ok"))
    for key in ("errors", "warnings", "no_ops"):
        values = report.get(key)
        if values:
            output[key] = values
    for key in ("theta_source", "theta_changed_after_repair", "actions_match_theta"):
        if key in report:
            output[key] = report.get(key)
    return output


def _current_state(node: MCTSNode) -> dict[str, Any]:
    metrics = _metric_summary(node.search_objectives, node.audit_metrics)
    output = {
        "node_id": node.node_id,
        "theta_id": node.theta_id,
        "theta": theta_to_dict(node.theta),
        "Q_self": _prompt_number(node.Q_self),
        "Q": _prompt_number(node.Q),
        "reward": metrics.get("metric_reward"),
        "N": int(node.N),
        "prior_score": _prompt_number(node.p),
        "guard_pass": bool(node.guard_pass),
        "metrics": metrics,
        "previous_actions": _compact_actions(list(node.actions)),
    }
    if node.action_validation:
        output["action_validation"] = _compact_action_validation(dict(node.action_validation))
    return output


def _refine_node(node: MCTSNode | None, *, include_actions: bool = True) -> dict[str, Any] | None:
    if node is None or node.theta is None:
        return None
    output = {
        "node_id": node.node_id,
        "theta_id": node.theta_id,
        "theta": theta_to_dict(node.theta),
        "Q_self": _prompt_number(node.Q_self),
        "Q": _prompt_number(node.Q),
        "N": int(node.N),
        "prior_score": _prompt_number(node.p),
        "guard_pass": bool(node.guard_pass),
        "metrics": _metric_summary(node.search_objectives, node.audit_metrics),
    }
    if include_actions:
        output["actions"] = _compact_actions(list(node.actions))
        if node.action_validation:
            output["action_validation"] = _compact_action_validation(dict(node.action_validation))
    return output


def _refine_reference_node(node: MCTSNode) -> dict[str, Any] | None:
    if node.theta is None:
        return None
    return {
        "node_id": node.node_id,
        "theta_id": node.theta_id,
        "theta": theta_to_dict(node.theta),
        "Q_self": _prompt_number(node.Q_self),
        "metrics": _diversity_metric_summary(node.search_objectives, node.audit_metrics),
        "guard_pass": bool(node.guard_pass),
    }


def _column_allowed(column: Any, feature_columns: set[str]) -> bool:
    return isinstance(column, str) and bool(column) and (not feature_columns or column in feature_columns)


def _compact_feedback(
    feedback: dict[str, Any] | None,
    *,
    feature_columns: set[str] | None = None,
    limit: int = 4,
) -> dict[str, Any]:
    if not isinstance(feedback, dict):
        return {}
    features = set(feature_columns or [])
    shape_weak_columns: list[dict[str, Any]] = []
    for item in feedback.get("shape_weak_columns", []) or []:
        if not isinstance(item, dict):
            continue
        column = item.get("column")
        if not _column_allowed(column, features):
            continue
        shape_weak_columns.append(
            {
                "column": column,
                "score": _prompt_number(item.get("score")),
                "metric": item.get("metric"),
            }
        )
        if len(shape_weak_columns) >= limit:
            break

    trend_weak_pairs: list[dict[str, Any]] = []
    for item in feedback.get("trend_weak_pairs", []) or []:
        if not isinstance(item, dict):
            continue
        left = item.get("left")
        right = item.get("right")
        if not _column_allowed(left, features) or not _column_allowed(right, features):
            continue
        trend_weak_pairs.append(
            {
                "left": left,
                "right": right,
                "score": _prompt_number(item.get("score")),
                "metric": item.get("metric"),
            }
        )
        if len(trend_weak_pairs) >= limit:
            break

    privacy_summary = feedback.get("privacy_summary", {}) if isinstance(feedback.get("privacy_summary"), dict) else {}
    utility_summary = feedback.get("utility_summary", {}) if isinstance(feedback.get("utility_summary"), dict) else {}
    return {
        "shape_weak_columns": shape_weak_columns,
        "trend_weak_pairs": trend_weak_pairs,
        "privacy": {
            key: _prompt_number(privacy_summary.get(key))
            for key in ("search_privacy", "search_privacy_raw")
            if key in privacy_summary
        },
        "dcr_balance": _dcr_balance_summary(privacy_summary.get("dcr"), privacy_summary.get("dcr_privacy")),
        "utility": {
            key: _prompt_number(utility_summary.get(key))
            for key in ("search_utility_proxy", "utility_exact", "utility_exact_available")
            if key in utility_summary
        },
    }


def _compact_archive(records: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    compact_records: list[dict[str, Any]] = []
    for record in records[:limit]:
        compact_records.append(
            {
                "node_id": record.get("node_id"),
                "theta_id": record.get("theta_id"),
                "theta": record.get("theta"),
                "Q_self": record.get("Q_self"),
                "reward_available": record.get("reward_available"),
                "guard_pass": record.get("guard_pass"),
                "search_objectives": record.get("search_objectives", {}),
                "audit_metrics": record.get("audit_metrics", {}),
            }
        )
    return compact_records


def _refine_archive(
    records: list[dict[str, Any]],
    *,
    current_theta_id: str | None = None,
    exclude_theta_ids: set[str] | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    compact_records: list[dict[str, Any]] = []
    seen: set[str] = set(exclude_theta_ids or set())
    for record in records:
        theta_id = record.get("theta_id")
        if theta_id == current_theta_id or theta_id in seen:
            continue
        theta = record.get("theta")
        if not isinstance(theta, dict):
            continue
        seen.add(str(theta_id))
        item = {
            "node_id": record.get("node_id"),
            "theta_id": theta_id,
            "theta": theta,
            "Q_self": _prompt_number(record.get("Q_self")),
            "metrics": _diversity_metric_summary(
                record.get("search_objectives", {}),
                record.get("audit_metrics", {}),
            ),
        }
        if record.get("guard_pass") is not None:
            item["guard_pass"] = record.get("guard_pass")
        compact_records.append(item)
        if len(compact_records) >= limit:
            break
    return compact_records


def build_init_prompt_context(
    *,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any] | None = None,
    existing_theta_keys: list[str] | None = None,
    baseline_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    use_dataset_context = dataset_context or dataset_context_from_schema(schema_card)
    return {
        "dataset_context": _compact_dataset_context(use_dataset_context, include_seeds=True),
        "theta_size_guidance": _theta_size_guidance(use_dataset_context),
        "schema": schema_summary(schema_card),
        "existing_theta_keys": list(existing_theta_keys or []),
        "baseline_diagnostics": baseline_diagnostics or {},
        "task": "propose diverse initial theta strategies",
    }


def build_refine_prompt_context(
    *,
    node: MCTSNode,
    parent: MCTSNode | None,
    siblings: list[MCTSNode],
    archive: list[dict[str, Any]],
    feedback: dict[str, Any] | None = None,
    schema_card: dict[str, Any] | None = None,
    dataset_context: dict[str, Any] | None = None,
    existing_theta_keys: list[str] | None = None,
    include_dataset_priors: bool = True,
) -> dict[str, Any]:
    schema = schema_card or {}
    use_feedback = feedback or node.feedback
    brief = _dataset_brief(
        dataset_context or (dataset_context_from_schema(schema) if schema else {}),
        include_dataset_priors=include_dataset_priors,
    )
    feature_columns = set(brief.get("columns", {}).get("feature", []))
    sibling_refs = [
        ref
        for ref in (_refine_reference_node(sibling) for sibling in siblings[:4])
        if ref is not None
    ]
    archive_refs = _refine_archive(
        archive,
        current_theta_id=node.theta_id,
        exclude_theta_ids={str(ref.get("theta_id")) for ref in sibling_refs if ref.get("theta_id")},
        limit=3,
    )
    return {
        "dataset_brief": brief,
        "current_state": _current_state(node),
        "feedback_to_fix": _compact_feedback(use_feedback, feature_columns=feature_columns),
        "diversity_context": {
            "siblings": sibling_refs,
            "archive": archive_refs,
            "existing_theta_count": len(existing_theta_keys or []),
        },
        "constraints": {
            "allowed_actions": REFINE_ALLOWED_ACTIONS,
            "add_format": {"type": "add_col_1d", "new": "<feature>"},
            "replace_format": {"type": "replace_col_1d", "old": "<existing feature>", "new": "<feature>"},
            "existing_theta_count": len(existing_theta_keys or []),
        },
        "task": "refine current theta using allowed actions",
    }


def build_prior_scoring_context(
    *,
    theta: dict[str, Any],
    parent_context: dict[str, Any] | None = None,
    schema_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": schema_summary(schema_card) if schema_card else {},
        "theta": theta,
        "parent_context": parent_context or {},
        "task": "score theta prior before rollout",
    }
