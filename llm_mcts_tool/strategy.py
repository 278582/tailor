from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class StrategyTheta:
    col_1ds: tuple[str, ...]
    col_2ds: tuple[str, ...]
    col_ps: tuple[str, ...]
    col_u: str


@dataclass(frozen=True)
class ThetaAction:
    type: str
    column: str | None = None
    old: str | None = None
    new: str | None = None


@dataclass
class StrategyProposal:
    theta: StrategyTheta
    actions: list[ThetaAction]
    prior_score: float
    reason: str
    action_validation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def theta_size_bounds(feature_count: int) -> dict[str, dict[str, int | str]]:
    n_features = max(0, int(feature_count))
    if n_features <= 0:
        return {
            "col_1ds": {"min": 1, "max": 8, "rule": "fallback default"},
            "col_2ds": {"min": 2, "max": 8, "rule": "fallback default"},
            "col_ps": {"min": 1, "max": 6, "rule": "fallback default"},
        }
    if n_features <= 4:
        privacy_max = max(1, n_features - 1)
        return {
            "col_1ds": {"min": 1, "max": n_features, "rule": "small schema: allow compact-to-full 1D scope"},
            "col_2ds": {"min": min(2, n_features), "max": n_features, "rule": "small schema: allow compact-to-full 2D anchors"},
            "col_ps": {"min": 1, "max": privacy_max, "rule": "small schema: privacy scope leaves at least one feature out when possible"},
        }

    col_1d_min = min(max(3, math.ceil(0.25 * n_features)), 8)
    col_1d_max = max(col_1d_min, min(n_features, _round_half_up(0.45 * n_features), 16))
    col_2d_min = min(max(3, math.ceil(0.25 * n_features)), 8)
    col_2d_max = max(col_2d_min, min(n_features, _round_half_up(0.50 * n_features), 12))
    privacy_hard_max = max(1, n_features - 1)
    col_p_min = min(max(2, math.ceil(0.50 * n_features)), privacy_hard_max, 48)
    col_p_max = max(col_p_min, min(privacy_hard_max, math.ceil(0.85 * n_features), 64))
    return {
        "col_1ds": {
            "min": col_1d_min,
            "max": col_1d_max,
            "rule": "about 25-45% of feature columns, capped for wide schemas",
        },
        "col_2ds": {
            "min": col_2d_min,
            "max": col_2d_max,
            "rule": "about 25-50% of feature columns as pair anchors, capped for wide schemas",
        },
        "col_ps": {
            "min": col_p_min,
            "max": col_p_max,
            "rule": "about 50-85% of feature columns for privacy/DCR, capped for wide schemas and never all features",
        },
    }


def theta_size_bounds_target_inclusive(column_count: int) -> dict[str, dict[str, int | str]]:
    n_columns = max(1, int(column_count))
    n_features = max(1, n_columns - 1)
    return {
        "col_1ds": {
            "min": max(1, math.ceil(0.50 * n_columns)),
            "max": n_columns,
            "rule": "50-100% of all columns; target allowed but not required",
        },
        "col_2ds": {
            "min": max(1, math.ceil(0.50 * n_columns)),
            "max": n_columns,
            "rule": "50-100% of all columns; target allowed but not required",
        },
        "col_ps": {
            "min": max(1, math.ceil(0.80 * n_features)),
            "max": n_features,
            "rule": "80-100% of non-target feature columns; target forbidden",
        },
    }


def _schema_columns(schema_card: Mapping[str, Any]) -> tuple[list[str], str | None, list[str]]:
    columns_obj = schema_card.get("columns", {})
    column_order = list(schema_card.get("column_order") or columns_obj.keys())
    target_column = schema_card.get("target_column")
    feature_columns = [
        column
        for column in column_order
        if column in columns_obj
        and column != target_column
        and not bool(columns_obj.get(column, {}).get("is_target", False))
    ]
    return column_order, None if target_column is None else str(target_column), feature_columns


def _as_column_tuple(values: Iterable[Any] | Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        iterable: Iterable[Any] = [values]
    else:
        try:
            iterable = list(values)
        except TypeError:
            iterable = [values]
    normalized = [str(value) for value in iterable if value is not None and str(value).strip()]
    return tuple(sorted(dict.fromkeys(normalized)))


def _coerce_theta(theta: StrategyTheta | Mapping[str, Any]) -> StrategyTheta:
    if isinstance(theta, StrategyTheta):
        return theta
    return StrategyTheta(
        col_1ds=_as_column_tuple(theta.get("col_1ds", ())),
        col_2ds=_as_column_tuple(theta.get("col_2ds", ())),
        col_ps=_as_column_tuple(theta.get("col_ps", ())),
        col_u=str(theta.get("col_u", "") or "") if str(theta.get("col_u", "") or "").strip() else "",
    )


def _coerce_action(action: ThetaAction | Mapping[str, Any]) -> ThetaAction:
    if isinstance(action, ThetaAction):
        return action
    return ThetaAction(
        type=str(action.get("type", "") or "").strip(),
        column=None
        if action.get("column") is None or not str(action.get("column")).strip()
        else str(action.get("column")),
        old=None if action.get("old") is None or not str(action.get("old")).strip() else str(action.get("old")),
        new=None if action.get("new") is None or not str(action.get("new")).strip() else str(action.get("new")),
    )


def normalize_theta(theta: StrategyTheta | Mapping[str, Any], schema_card: Mapping[str, Any] | None = None) -> StrategyTheta:
    coerced = _coerce_theta(theta)
    return StrategyTheta(
        col_1ds=_as_column_tuple(coerced.col_1ds),
        col_2ds=_as_column_tuple(coerced.col_2ds),
        col_ps=_as_column_tuple(coerced.col_ps),
        col_u=str(coerced.col_u or "") if str(coerced.col_u or "").strip() else "",
    )


def _theta_dict(theta: StrategyTheta) -> dict[str, Any]:
    return {
        "col_1ds": list(theta.col_1ds),
        "col_2ds": list(theta.col_2ds),
        "col_ps": list(theta.col_ps),
        "col_u": theta.col_u,
    }


def validate_theta_actions(
    parent_theta: StrategyTheta | Mapping[str, Any],
    actions: Sequence[ThetaAction | Mapping[str, Any]],
    schema_card: Mapping[str, Any],
) -> dict[str, Any]:
    parent = normalize_theta(parent_theta, schema_card)
    _, target_column, feature_columns = _schema_columns(schema_card)
    feature_set = set(feature_columns)
    values = {
        "col_1ds": list(parent.col_1ds),
        "col_2ds": list(parent.col_2ds),
        "col_ps": list(parent.col_ps),
    }
    col_u = parent.col_u
    errors: list[str] = []
    warnings: list[str] = []
    no_ops: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []

    scope_by_action = {
        "add_col_1d": "col_1ds",
        "replace_col_1d": "col_1ds",
        "add_col_2d": "col_2ds",
        "replace_col_2d": "col_2ds",
        "add_col_p": "col_ps",
        "replace_col_p": "col_ps",
    }

    def check_feature(column: str | None, *, idx: int, role: str) -> bool:
        if not column:
            errors.append(f"action[{idx}] missing {role}")
            return False
        if column == target_column:
            errors.append(f"action[{idx}] {role} is target column: {column}")
            return False
        if column not in feature_set:
            errors.append(f"action[{idx}] {role} is not a feature column: {column}")
            return False
        return True

    for idx, raw_action in enumerate(actions):
        action = _coerce_action(raw_action)
        action_type = action.type.strip().lower()
        new_column = action.new or action.column
        applied_record = {
            "index": idx,
            "type": action_type,
            "old": action.old,
            "new": new_column,
        }

        if action_type not in {
            "add_col_1d",
            "replace_col_1d",
            "add_col_2d",
            "replace_col_2d",
            "add_col_p",
            "replace_col_p",
            "replace_col_u",
        }:
            errors.append(f"action[{idx}] has unsupported type: {action.type}")
            continue

        if action_type == "replace_col_u":
            if not check_feature(new_column, idx=idx, role="new"):
                continue
            if action.old and action.old != col_u:
                errors.append(f"action[{idx}] old does not match current col_u: old={action.old}, current={col_u}")
                continue
            if new_column == col_u:
                no_ops.append({**applied_record, "reason": "replace_col_u_same_value"})
                continue
            col_u = str(new_column)
            applied.append(applied_record)
            continue

        scope = scope_by_action[action_type]
        scope_values = values[scope]
        if not check_feature(new_column, idx=idx, role="new"):
            continue

        if action_type.startswith("add_"):
            if new_column in scope_values:
                no_ops.append({**applied_record, "scope": scope, "reason": "add_existing_column"})
                continue
            scope_values.append(str(new_column))
            applied.append({**applied_record, "scope": scope})
            continue

        if not action.old:
            errors.append(f"action[{idx}] missing old for {action_type}")
            continue
        if action.old not in scope_values:
            errors.append(f"action[{idx}] old not present in {scope}: {action.old}")
            continue
        if new_column == action.old:
            no_ops.append({**applied_record, "scope": scope, "reason": "replace_same_column"})
            continue
        if new_column in scope_values:
            errors.append(f"action[{idx}] new already present in {scope}: {new_column}")
            continue
        scope_values[scope_values.index(action.old)] = str(new_column)
        applied.append({**applied_record, "scope": scope})

    result_theta = StrategyTheta(
        col_1ds=tuple(sorted(dict.fromkeys(values["col_1ds"]))),
        col_2ds=tuple(sorted(dict.fromkeys(values["col_2ds"]))),
        col_ps=tuple(sorted(dict.fromkeys(values["col_ps"]))),
        col_u=col_u,
    )
    bounds = theta_size_bounds(len(feature_columns))
    for field_name in ("col_1ds", "col_2ds", "col_ps"):
        size = len(getattr(result_theta, field_name))
        min_size = int(bounds[field_name]["min"])
        max_size = int(bounds[field_name]["max"])
        if size < min_size:
            errors.append(f"{field_name} would have {size} columns; hard minimum is {min_size}")
        if size > max_size:
            errors.append(f"{field_name} would have {size} columns; hard maximum is {max_size}")
    if not actions:
        warnings.append("no actions supplied")
    return {
        "ok": not errors and not no_ops,
        "errors": errors,
        "warnings": warnings,
        "no_ops": no_ops,
        "applied": applied,
        "result_theta": _theta_dict(result_theta),
    }


def validate_theta(
    theta: StrategyTheta | Mapping[str, Any],
    schema_card: Mapping[str, Any],
    *,
    min_col_1ds: int | None = None,
    max_col_1ds: int | None = None,
    min_col_2ds: int | None = None,
    max_col_2ds: int | None = None,
    min_col_ps: int | None = None,
    max_col_ps: int | None = None,
) -> ValidationReport:
    normalized = normalize_theta(theta, schema_card)
    column_order, target_column, feature_columns = _schema_columns(schema_card)
    bounds = theta_size_bounds(len(feature_columns))
    min_col_1ds = int(bounds["col_1ds"]["min"] if min_col_1ds is None else min_col_1ds)
    max_col_1ds = int(bounds["col_1ds"]["max"] if max_col_1ds is None else max_col_1ds)
    min_col_2ds = int(bounds["col_2ds"]["min"] if min_col_2ds is None else min_col_2ds)
    max_col_2ds = int(bounds["col_2ds"]["max"] if max_col_2ds is None else max_col_2ds)
    min_col_ps = int(bounds["col_ps"]["min"] if min_col_ps is None else min_col_ps)
    max_col_ps = int(bounds["col_ps"]["max"] if max_col_ps is None else max_col_ps)
    known = set(column_order)
    features = set(feature_columns)
    errors: list[str] = []
    warnings: list[str] = []

    for field_name in ("col_1ds", "col_2ds", "col_ps"):
        values = getattr(normalized, field_name)
        if len(values) != len(set(values)):
            errors.append(f"{field_name} contains duplicate columns")
        unknown = [column for column in values if column not in known]
        if unknown:
            errors.append(f"{field_name} contains unknown columns: {unknown}")
        target_hits = [column for column in values if column == target_column or column not in features]
        target_hits = [column for column in target_hits if column in known]
        if target_hits:
            errors.append(f"{field_name} contains non-feature or target columns: {target_hits}")

    if not normalized.col_u:
        errors.append("col_u is required")
    elif normalized.col_u not in known:
        errors.append(f"col_u is unknown: {normalized.col_u}")
    elif normalized.col_u not in features:
        errors.append(f"col_u must be a non-target feature column: {normalized.col_u}")

    size_rules = (
        ("col_1ds", len(normalized.col_1ds), int(min_col_1ds), max_col_1ds),
        ("col_2ds", len(normalized.col_2ds), int(min_col_2ds), max_col_2ds),
        ("col_ps", len(normalized.col_ps), int(min_col_ps), max_col_ps),
    )
    for field_name, size, min_size, max_size in size_rules:
        if size < min_size:
            errors.append(f"{field_name} requires at least {min_size} columns")
        if max_size is not None and size > int(max_size):
            errors.append(f"{field_name} allows at most {int(max_size)} columns")

    if len(feature_columns) < 2:
        warnings.append("schema has fewer than two feature columns; col_2ds repair may be limited")

    return ValidationReport(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def validate_theta_target_inclusive(
    theta: StrategyTheta | Mapping[str, Any],
    schema_card: Mapping[str, Any],
) -> ValidationReport:
    normalized = normalize_theta(theta, schema_card)
    column_order, target_column, feature_columns = _schema_columns(schema_card)
    bounds = theta_size_bounds_target_inclusive(len(column_order))
    known = set(column_order)
    feature_set = set(feature_columns)
    errors: list[str] = []
    warnings: list[str] = []

    if not target_column:
        errors.append("schema_card.target_column is required for target-aware theta")

    for field_name in ("col_1ds", "col_2ds", "col_ps"):
        values = getattr(normalized, field_name)
        if len(values) != len(set(values)):
            errors.append(f"{field_name} contains duplicate columns")
        unknown = [column for column in values if column not in known]
        if unknown:
            errors.append(f"{field_name} contains unknown columns: {unknown}")
        if field_name == "col_ps":
            non_features = [column for column in values if column in known and column not in feature_set]
            if non_features:
                errors.append(f"col_ps contains non-feature or target columns: {non_features}")
        size = len(values)
        min_size = int(bounds[field_name]["min"])
        max_size = int(bounds[field_name]["max"])
        if size < min_size:
            errors.append(f"{field_name} requires at least {min_size} columns")
        if size > max_size:
            errors.append(f"{field_name} allows at most {max_size} columns")

    if not normalized.col_u:
        errors.append("col_u is required")
    elif normalized.col_u not in known:
        errors.append(f"col_u is unknown: {normalized.col_u}")
    elif normalized.col_u not in feature_set:
        errors.append(f"col_u must be a non-target feature column: {normalized.col_u}")

    return ValidationReport(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def canonical_key(theta: StrategyTheta | Mapping[str, Any]) -> str:
    normalized = normalize_theta(theta)
    payload = {
        "col_1ds": list(normalized.col_1ds),
        "col_2ds": list(normalized.col_2ds),
        "col_ps": list(normalized.col_ps),
        "col_u": normalized.col_u,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def theta_id(theta: StrategyTheta | Mapping[str, Any], length: int = 12) -> str:
    digest = hashlib.sha1(canonical_key(theta).encode("utf-8")).hexdigest()
    return digest[: max(1, int(length))]


def _replace_column(values: list[str], old: str | None, new: str | None) -> list[str]:
    if not new:
        return values
    if old and old in values:
        values[values.index(old)] = new
    elif new not in values:
        values.append(new)
    return values


def apply_actions(
    parent_theta: StrategyTheta | Mapping[str, Any],
    actions: Sequence[ThetaAction | Mapping[str, Any]],
    schema_card: Mapping[str, Any],
) -> StrategyTheta:
    parent = normalize_theta(parent_theta, schema_card)
    values = {
        "col_1ds": list(parent.col_1ds),
        "col_2ds": list(parent.col_2ds),
        "col_ps": list(parent.col_ps),
    }
    col_u = parent.col_u

    for raw_action in actions:
        action = _coerce_action(raw_action)
        action_type = action.type.strip().lower()
        new_column = action.new or action.column
        if action_type == "add_col_1d":
            if new_column and new_column not in values["col_1ds"]:
                values["col_1ds"].append(new_column)
        elif action_type == "replace_col_1d":
            values["col_1ds"] = _replace_column(values["col_1ds"], action.old, new_column)
        elif action_type == "add_col_2d":
            if new_column and new_column not in values["col_2ds"]:
                values["col_2ds"].append(new_column)
        elif action_type == "replace_col_2d":
            values["col_2ds"] = _replace_column(values["col_2ds"], action.old, new_column)
        elif action_type == "add_col_p":
            if new_column and new_column not in values["col_ps"]:
                values["col_ps"].append(new_column)
        elif action_type == "replace_col_p":
            values["col_ps"] = _replace_column(values["col_ps"], action.old, new_column)
        elif action_type == "replace_col_u":
            if new_column:
                col_u = new_column

    return repair_theta(
        StrategyTheta(
            col_1ds=tuple(values["col_1ds"]),
            col_2ds=tuple(values["col_2ds"]),
            col_ps=tuple(values["col_ps"]),
            col_u=col_u,
        ),
        schema_card,
        rng=random.Random(0),
    )


def _fill_columns(values: list[str], feature_columns: Sequence[str], min_size: int, max_size: int | None, rng: random.Random) -> list[str]:
    feature_set = set(feature_columns)
    output = [column for column in dict.fromkeys(values) if column in feature_set]
    rng.shuffle(output)
    for column in feature_columns:
        if len(output) >= min_size:
            break
        if column not in output:
            output.append(column)
    if max_size is not None and len(output) > int(max_size):
        output = output[: int(max_size)]
    return sorted(output)


def _fill_columns_target_inclusive(
    values: list[str],
    column_order: Sequence[str],
    target_column: str,
    min_size: int,
    max_size: int,
    rng: random.Random,
) -> list[str]:
    known = set(column_order)
    output = [column for column in dict.fromkeys(values) if column in known]
    if target_column not in output:
        output.insert(0, target_column)
    rng.shuffle(output)
    if target_column in output:
        output = [target_column] + [column for column in output if column != target_column]
    for column in column_order:
        if len(output) >= min_size:
            break
        if column not in output:
            output.append(column)
    if len(output) > max_size:
        kept = [target_column]
        for column in output:
            if column == target_column:
                continue
            if len(kept) >= max_size:
                break
            kept.append(column)
        output = kept
    return sorted(output)


def repair_theta_target_inclusive(
    theta: StrategyTheta | Mapping[str, Any],
    schema_card: Mapping[str, Any],
    rng: random.Random | Any | None = None,
) -> StrategyTheta:
    normalized = normalize_theta(theta, schema_card)
    column_order, target_column, feature_columns = _schema_columns(schema_card)
    if not target_column:
        return repair_theta(theta, schema_card, rng=rng)
    bounds = theta_size_bounds_target_inclusive(len(column_order))
    random_like = rng if rng is not None else random.Random()
    col_1ds = _fill_columns(
        list(normalized.col_1ds),
        column_order,
        int(bounds["col_1ds"]["min"]),
        int(bounds["col_1ds"]["max"]),
        random_like,
    )
    col_2ds = _fill_columns(
        list(normalized.col_2ds),
        column_order,
        int(bounds["col_2ds"]["min"]),
        int(bounds["col_2ds"]["max"]),
        random_like,
    )
    col_ps = _fill_columns(
        list(normalized.col_ps),
        feature_columns,
        int(bounds["col_ps"]["min"]),
        int(bounds["col_ps"]["max"]),
        random_like,
    )
    features = list(feature_columns)
    col_u = normalized.col_u if normalized.col_u in set(features) else (features[0] if features else "")
    return StrategyTheta(
        col_1ds=tuple(col_1ds),
        col_2ds=tuple(col_2ds),
        col_ps=tuple(col_ps),
        col_u=col_u,
    )


def repair_theta(
    theta: StrategyTheta | Mapping[str, Any],
    schema_card: Mapping[str, Any],
    rng: random.Random | Any | None = None,
    *,
    min_col_1ds: int | None = None,
    max_col_1ds: int | None = None,
    min_col_2ds: int | None = None,
    max_col_2ds: int | None = None,
    min_col_ps: int | None = None,
    max_col_ps: int | None = None,
) -> StrategyTheta:
    normalized = normalize_theta(theta, schema_card)
    _, _, feature_columns = _schema_columns(schema_card)
    bounds = theta_size_bounds(len(feature_columns))
    min_col_1ds = int(bounds["col_1ds"]["min"] if min_col_1ds is None else min_col_1ds)
    max_col_1ds = int(bounds["col_1ds"]["max"] if max_col_1ds is None else max_col_1ds)
    min_col_2ds = int(bounds["col_2ds"]["min"] if min_col_2ds is None else min_col_2ds)
    max_col_2ds = int(bounds["col_2ds"]["max"] if max_col_2ds is None else max_col_2ds)
    min_col_ps = int(bounds["col_ps"]["min"] if min_col_ps is None else min_col_ps)
    max_col_ps = int(bounds["col_ps"]["max"] if max_col_ps is None else max_col_ps)
    random_like = rng if rng is not None else random.Random()
    features = list(feature_columns)

    col_1ds = _fill_columns(list(normalized.col_1ds), features, min_col_1ds, max_col_1ds, random_like)
    col_2ds = _fill_columns(list(normalized.col_2ds), features, min_col_2ds, max_col_2ds, random_like)
    col_ps = _fill_columns(list(normalized.col_ps), features, min_col_ps, max_col_ps, random_like)
    col_u = normalized.col_u if normalized.col_u in set(features) else (features[0] if features else "")

    return StrategyTheta(
        col_1ds=tuple(col_1ds),
        col_2ds=tuple(col_2ds),
        col_ps=tuple(col_ps),
        col_u=col_u,
    )


def dedupe_proposals(
    proposals: Sequence[StrategyProposal],
    visited_keys: set[str] | Sequence[str],
) -> list[StrategyProposal]:
    seen = set(visited_keys)
    output: list[StrategyProposal] = []
    for proposal in proposals:
        key = canonical_key(proposal.theta)
        if key in seen:
            continue
        seen.add(key)
        output.append(proposal)
    return output
