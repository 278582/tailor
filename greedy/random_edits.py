from __future__ import annotations

import random
import math
from typing import Any, Iterable

from llm_mcts_tool.strategy import StrategyProposal, StrategyTheta, ThetaAction, canonical_key
from llm_mcts_tool.v2_pipeline import _schema_feature_columns, theta_to_dict

SET_FIELDS = ("col_1ds", "col_2ds", "col_ps")
DEFAULT_GREEDY_FOCUS_SIZE = 2
DEFAULT_GREEDY_MAX_SIZE = 8

_ADD_TYPE = {
    "col_1ds": "add_col_1d",
    "col_2ds": "add_col_2d",
    "col_ps": "add_col_p",
}
_REPLACE_TYPE = {
    "col_1ds": "replace_col_1d",
    "col_2ds": "replace_col_2d",
    "col_ps": "replace_col_p",
    "col_u": "replace_col_u",
}
_FIELD_BY_ACTION = {
    "add_col_1d": "col_1ds",
    "replace_col_1d": "col_1ds",
    "add_col_2d": "col_2ds",
    "replace_col_2d": "col_2ds",
    "add_col_p": "col_ps",
    "replace_col_p": "col_ps",
    "replace_col_u": "col_u",
}


def _column_order(schema_card: dict[str, Any]) -> list[str]:
    return [str(column) for column in schema_card.get("column_order", [])]


def _universe(field_name: str, schema_card: dict[str, Any]) -> list[str]:
    if field_name in {"col_ps", "col_u"}:
        return list(_schema_feature_columns(schema_card))
    return _column_order(schema_card)


def greedy_set_bounds(
    universe: list[str],
    *,
    focus_size: int = DEFAULT_GREEDY_FOCUS_SIZE,
    max_size: int = DEFAULT_GREEDY_MAX_SIZE,
    max_fraction: float | None = None,
) -> tuple[int, int]:
    """Return (min_size, cap) for one Theta set. Min is the compact start; cap is the hard max."""
    if max_fraction is not None and not 0 < float(max_fraction) <= 1:
        raise ValueError("greedy max fraction must be in (0, 1]")
    if not universe:
        return 0, 0
    if max_fraction is None:
        min_size = max(1, min(int(focus_size), len(universe)))
        cap = max(min_size, min(int(max_size), len(universe)))
    else:
        cap = max(1, min(len(universe), int(math.floor(float(max_fraction) * len(universe)))))
        min_size = max(1, min(int(focus_size), cap))
    return min_size, cap


def _clamp_set(
    values: Iterable[str],
    universe: list[str],
    *,
    min_size: int,
    max_size: int,
) -> tuple[str, ...]:
    allowed = set(universe)
    output: list[str] = []
    for column in values:
        name = str(column)
        if name in allowed and name not in output:
            output.append(name)
    if max_size > 0:
        output = output[:max_size]
    for column in universe:
        if len(output) >= min_size:
            break
        if column not in output:
            output.append(column)
    return tuple(output)


def repair_greedy_theta(
    payload: dict[str, Any] | StrategyTheta,
    schema_card: dict[str, Any],
    *,
    focus_size: int = DEFAULT_GREEDY_FOCUS_SIZE,
    max_size: int = DEFAULT_GREEDY_MAX_SIZE,
    max_fraction: float | None = None,
) -> StrategyTheta:
    """Keep each set in [start_size, max_size]. Do not pad to the TAILOR 50%/80% floors."""
    raw = theta_to_dict(payload) if isinstance(payload, StrategyTheta) else dict(payload)
    features = _universe("col_ps", schema_card)
    columns = _universe("col_1ds", schema_card)
    col_1d_min, col_1d_max = greedy_set_bounds(columns, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)
    col_2d_min, col_2d_max = greedy_set_bounds(columns, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)
    col_p_min, col_p_max = greedy_set_bounds(features, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)
    col_1ds = _clamp_set(raw.get("col_1ds") or (), columns, min_size=col_1d_min, max_size=col_1d_max)
    col_2ds = _clamp_set(raw.get("col_2ds") or (), columns, min_size=col_2d_min, max_size=col_2d_max)
    col_ps = _clamp_set(raw.get("col_ps") or (), features, min_size=col_p_min, max_size=col_p_max)
    col_u = str(raw.get("col_u") or "")
    if col_u not in set(features):
        col_u = features[0] if features else ""
    return StrategyTheta(col_1ds=col_1ds, col_2ds=col_2ds, col_ps=col_ps, col_u=col_u)


def greedy_theta_ok(
    theta: StrategyTheta,
    schema_card: dict[str, Any],
    *,
    focus_size: int = DEFAULT_GREEDY_FOCUS_SIZE,
    max_size: int = DEFAULT_GREEDY_MAX_SIZE,
    max_fraction: float | None = None,
) -> bool:
    columns = set(_universe("col_1ds", schema_card))
    features = set(_universe("col_ps", schema_card))
    repaired = repair_greedy_theta(theta, schema_card, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)
    col_1d_min, col_1d_max = greedy_set_bounds(list(columns), focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)
    col_2d_min, col_2d_max = greedy_set_bounds(list(columns), focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)
    col_p_min, col_p_max = greedy_set_bounds(list(features), focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)
    return (
        canonical_key(repaired) == canonical_key(theta)
        and set(theta.col_1ds) <= columns
        and set(theta.col_2ds) <= columns
        and set(theta.col_ps) <= features
        and theta.col_u in features
        and col_1d_min <= len(theta.col_1ds) <= col_1d_max
        and col_2d_min <= len(theta.col_2ds) <= col_2d_max
        and col_p_min <= len(theta.col_ps) <= col_p_max
    )


def greedy_start_theta(
    schema_card: dict[str, Any],
    *,
    focus_size: int = DEFAULT_GREEDY_FOCUS_SIZE,
    max_size: int = DEFAULT_GREEDY_MAX_SIZE,
    max_fraction: float | None = None,
) -> StrategyTheta:
    """Naive greedy start: schema-order prefixes, not the strong all-column Theta0."""
    columns = _universe("col_1ds", schema_card)
    features = _universe("col_ps", schema_card)
    start_1d, _ = greedy_set_bounds(columns, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)
    start_2d, _ = greedy_set_bounds(columns, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)
    start_p, _ = greedy_set_bounds(features, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)
    payload = {
        "col_1ds": columns[:start_1d],
        "col_2ds": columns[:start_2d],
        "col_ps": features[:start_p],
        "col_u": features[0] if features else "",
    }
    theta = repair_greedy_theta(payload, schema_card, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)
    if not greedy_theta_ok(theta, schema_card, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction):
        raise ValueError("greedy start theta is invalid")
    return theta


def default_column_theta(
    schema_card: dict[str, Any],
    *,
    focus_size: int = DEFAULT_GREEDY_FOCUS_SIZE,
    max_size: int = DEFAULT_GREEDY_MAX_SIZE,
    max_fraction: float | None = None,
) -> StrategyTheta:
    return greedy_start_theta(schema_card, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)


def list_legal_column_edits(
    theta: StrategyTheta,
    schema_card: dict[str, Any],
    *,
    focus_size: int = DEFAULT_GREEDY_FOCUS_SIZE,
    max_size: int = DEFAULT_GREEDY_MAX_SIZE,
    max_fraction: float | None = None,
) -> list[ThetaAction]:
    """Add until the greedy cap, otherwise replace. Size cannot grow past max_size or back to Theta0."""
    features = _universe("col_u", schema_card)
    actions: list[ThetaAction] = []
    for feature in features:
        if feature != theta.col_u:
            actions.append(ThetaAction(type="replace_col_u", old=theta.col_u, new=feature))
    for field_name in SET_FIELDS:
        current = [str(column) for column in getattr(theta, field_name)]
        current_set = set(current)
        universe = _universe(field_name, schema_card)
        _, cap = greedy_set_bounds(universe, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)
        missing = [column for column in universe if column not in current_set]
        if len(current) < cap:
            for new_column in missing:
                actions.append(ThetaAction(type=_ADD_TYPE[field_name], new=new_column))
        for old_column in current:
            for new_column in missing:
                actions.append(
                    ThetaAction(type=_REPLACE_TYPE[field_name], old=old_column, new=new_column)
                )
    return actions


def apply_column_edit(
    parent: StrategyTheta,
    action: ThetaAction,
    schema_card: dict[str, Any],
    *,
    focus_size: int = DEFAULT_GREEDY_FOCUS_SIZE,
    max_size: int = DEFAULT_GREEDY_MAX_SIZE,
    max_fraction: float | None = None,
) -> StrategyTheta:
    payload = theta_to_dict(parent)
    action_type = str(action.type or "").strip()
    field_name = _FIELD_BY_ACTION.get(action_type)
    if field_name is None:
        raise ValueError(f"unsupported greedy column action: {action_type}")
    if action_type == "replace_col_u":
        if not action.new:
            raise ValueError("replace_col_u requires new")
        payload["col_u"] = str(action.new)
    elif action_type.startswith("add_"):
        if not action.new:
            raise ValueError(f"{action_type} requires new")
        payload[field_name] = [*(payload[field_name] or []), str(action.new)]
    elif action_type.startswith("replace_"):
        old_column = str(action.old or "")
        new_column = str(action.new or "")
        payload[field_name] = [
            new_column if column == old_column else column for column in payload[field_name]
        ]
    else:
        raise ValueError(f"unsupported greedy column action: {action_type}")
    return repair_greedy_theta(payload, schema_card, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction)


def is_single_column_edit(parent: StrategyTheta, child: StrategyTheta) -> bool:
    changed: list[str] = []
    if parent.col_u != child.col_u:
        changed.append("col_u")
    for field_name in SET_FIELDS:
        before = set(getattr(parent, field_name))
        after = set(getattr(child, field_name))
        if before == after:
            continue
        added = after - before
        removed = before - after
        add_ok = len(added) == 1 and len(removed) == 0
        replace_ok = len(added) == 1 and len(removed) == 1
        if not (add_ok or replace_ok):
            return False
        changed.append(field_name)
    return len(changed) == 1


def _interleave_add_and_replace(actions: list[ThetaAction], rng: random.Random) -> list[ThetaAction]:
    adds: list[ThetaAction] = []
    others: list[ThetaAction] = []
    for action in actions:
        if str(action.type).startswith("add_"):
            adds.append(action)
        else:
            others.append(action)
    rng.shuffle(adds)
    rng.shuffle(others)
    merged: list[ThetaAction] = []
    take_add_first = bool(rng.random() < 0.5)
    i = j = 0
    while i < len(adds) or j < len(others):
        if take_add_first:
            if i < len(adds):
                merged.append(adds[i])
                i += 1
            if j < len(others):
                merged.append(others[j])
                j += 1
        else:
            if j < len(others):
                merged.append(others[j])
                j += 1
            if i < len(adds):
                merged.append(adds[i])
                i += 1
    return merged


def propose_random_neighbors(
    parent: StrategyTheta,
    *,
    schema_card: dict[str, Any],
    n: int,
    rng: random.Random,
    exclude_keys: Iterable[str] = (),
    focus_size: int = DEFAULT_GREEDY_FOCUS_SIZE,
    max_size: int = DEFAULT_GREEDY_MAX_SIZE,
    max_fraction: float | None = None,
) -> list[StrategyProposal]:
    actions = _interleave_add_and_replace(
        list_legal_column_edits(parent, schema_card, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction),
        rng,
    )
    seen = {canonical_key(parent), *exclude_keys}
    proposals: list[StrategyProposal] = []
    for action in actions:
        child = apply_column_edit(
            parent, action, schema_card, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction
        )
        key = canonical_key(child)
        if key in seen:
            continue
        if not is_single_column_edit(parent, child):
            continue
        if not greedy_theta_ok(child, schema_card, focus_size=focus_size, max_size=max_size, max_fraction=max_fraction):
            continue
        seen.add(key)
        proposals.append(
            StrategyProposal(
                theta=child,
                actions=[action],
                prior_score=0.0,
                reason="random_column_edit",
                action_validation={
                    "proposal_source": "random_column_edit",
                    "llm": False,
                    "mcts": False,
                    "greedy_column": True,
                    "greedy_focus_size": int(focus_size),
                    "greedy_max_size": int(max_size),
                    "edit_type": str(action.type),
                },
            )
        )
        if len(proposals) >= max(0, int(n)):
            break
    return proposals
