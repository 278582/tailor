from __future__ import annotations

import json
import math
import random
import re
import statistics
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

try:
    from sklearn.exceptions import UndefinedMetricWarning

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", message=r".*old version of glibc.*", category=FutureWarning)

from post_selection_tool.context import resolve_eval_device, resolve_nn_device
from post_selection_tool.io import ensure_dir, save_csv, save_json, save_jsonl
from post_selection_tool.selector import ParetoSelector
from post_selection_tool.utility_proxy import compute_utility_exact_metrics
from metric_tool.evaluator import build_audit_metrics, evaluate_one_selection, utility_metric_direction
from metric_tool.tabdiff_density import TabDiffMetricRunner, load_tabdiff_info
from postprocess.cards import build_and_save_cards
from postprocess.tabdiff_protocol import normalize_tabdiff_dataframe_columns, resolve_tabdiff_selection_context
from postprocess.validator import TabularValidator

from .feedback import build_guard
from .llm_client import LLMClient
from .rollout import GuidedRolloutConfig, GuidedRolloutResult, run_guided_pareto_rollout
from .strategy import (
    StrategyProposal,
    StrategyTheta,
    ThetaAction,
    canonical_key,
    repair_theta_target_inclusive,
    theta_id,
    theta_size_bounds_target_inclusive,
    validate_theta_target_inclusive,
)


SUPPORTED_V2_DATASETS = {"adult", "beijing", "default", "diabetes", "magic", "news", "shoppers"}
SOURCE_ALIASES = {
    "tansyn": "tabsyn",
    "tabsyn": "tabsyn",
    "tabdiff": "tabdiff",
    "great": "great",
    "smote": "smote",
}
SOURCE_PROFILE_VERSION = "v2_source_profiles_complete_eval_v8_log_target_rmse"
UTILITY_IMPORTANCE_PROMPT_LIMIT = 8
UTILITY_IMPORTANCE_POOL_LIMIT = 4
UCT_EXPLORE_SCALE_MIN = 0.01
DCR_PROMPT_SEMANTICS = {
    "balance_signal": "raw_dcr_real_closer_rate targets 0.5; privacy_reward = 1 - abs(raw_dcr_real_closer_rate - 0.5). Treat them as one DCR balance signal.",
    "balance_direction": "privacy_reward is higher-better; raw_dcr_real_closer_rate is not monotonic higher-better or lower-better.",
    "distance_quantiles": "dcr_real/dcr_test are nearest-neighbor distances; larger distances reduce exact-neighbor risk while raw balance stays near 0.5.",
    "reporting_rule": "Do not count raw DCR movement and privacy_reward movement as two independent DCR changes.",
}


@dataclass
class V2MCTSConfig:
    dataset_name: str = "adult"
    exp_name: str = "adult_llm_mcts_v2"
    artifact_dir: Path = Path("artifacts/llm_mcts_v2/adult")
    sample_root: Path = Path("third_party/sample")
    prompt_pack_dir: Path = Path("prompt_pack")
    source_names: tuple[str, ...] = ("great", "smote", "tabdiff", "tabsyn")
    mode: str = "mixed"
    single_source: str = "tabdiff"
    seed: int = 20260420
    keep_k: int = 32561
    preselect_target: int = 45585
    d_cur_size: int = 1000
    density_reference_size: int = 5000
    max_theta_pairs: int = 32
    eval_device: str = "auto"
    nn_device: str = "auto"
    utility_exact_evaluator: str = "tabdiff_mle"
    utility_exact_torch_epochs: int = 6
    disable_progress: bool = True
    save_validation_records: bool = False
    save_rollout_internal_records: bool = False
    holdout_fraction: float = 0.1
    mcts_budget: int = 20
    initial_s_pool_count: int = 2
    theta_proposals_per_event: int = 4
    ucb_c: float = 2.0
    p_random_replace: float = 0.1
    pool_multiplier: float = 4.0
    provider: str = "llm"
    refine_s_pool_count: int = 1
    source_profile_repeats: int = 4
    source_profile_sample_rows: int | None = None
    utility_diag_sample_size: int = 6000
    rollout_direct_dcr_repair_enabled: bool = False
    rollout_direct_dcr_target_margin: float = 0.03
    rollout_direct_dcr_max_swap_fraction: float = 0.30
    rollout_direct_dcr_candidate_neighbors: int = 64
    rollout_direct_dcr_min_pair_utility_gain: float = -0.08
    rollout_direct_dcr_fallback_min_pair_utility_gain: float = -0.18
    rollout_reward_candidate_v2_enabled: bool = False
    rollout_reward_candidate_v2_max_swap_fraction: float = 0.16
    rollout_reward_candidate_v2_max_candidate_sizes: int = 10
    rollout_reward_candidate_v2_min_proxy_delta: float = 0.0
    rollout_reward_candidate_v2_fidelity_floor_eps: float = 0.015
    rollout_reward_candidate_v2_utility_floor_eps: float = 0.02
    new_s_pool_stagnation_events: int = 2
    early_stop_stagnation_events: int = 6
    force_new_s_at_event: int | None = None
    smoke: bool = False


@dataclass
class SourceInfo:
    source_id: str
    path: Path
    rows: int
    columns: list[str]


@dataclass
class SNode:
    s_id: str
    pool_units: list[dict[str, Any]]
    synthetic_csv: Path
    synthetic_row_map: Path
    llm_score: float = 0.5
    reason: str = ""
    semantic_summary: str = ""
    visits: int = 0
    theta_node_ids: list[str] = field(default_factory=list)
    best_reward: float = 0.0


@dataclass
class ThetaNode:
    node_id: str
    s_id: str
    theta: StrategyTheta
    theta_id: str
    parent_node_id: str | None
    actions: list[dict[str, Any]]
    proposal_action_validation: dict[str, Any]
    llm_score: float
    reason: str
    visits: int = 0
    best_reward: float = 0.0
    rollout_dir: Path | None = None
    reward: float = 0.0
    reward_available: bool = False
    exact_reward: float = 0.0
    exact_reward_available: bool = False
    exact_reward_failure_reason: str | None = None
    search_reward: float = 0.0
    search_reward_available: bool = False
    reward_type: str = "unavailable"
    guard_pass: bool = False
    status: str = "pending"
    search_objectives: dict[str, Any] = field(default_factory=dict)
    audit_metrics: dict[str, Any] = field(default_factory=dict)
    guard: dict[str, Any] = field(default_factory=dict)
    feedback: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "s_id": self.s_id,
            "theta_id": self.theta_id,
            "parent_node_id": self.parent_node_id,
            "theta": theta_to_dict(self.theta),
            "actions": list(self.actions),
            "proposal_action_validation": dict(self.proposal_action_validation),
            "llm_score": float(self.llm_score),
            "reason": self.reason,
            "visits": int(self.visits),
            "best_reward": float(self.best_reward),
            "semantic_summary": self.feedback.get("llm_semantic_summary") or _fallback_node_summary(self),
            "rollout_dir": None if self.rollout_dir is None else str(self.rollout_dir),
            "reward": float(self.reward),
            "reward_available": bool(self.reward_available),
            "exact_reward": float(self.exact_reward),
            "exact_reward_available": bool(self.exact_reward_available),
            "exact_reward_failure_reason": self.exact_reward_failure_reason,
            "search_reward": float(self.search_reward),
            "search_reward_available": bool(self.search_reward_available),
            "reward_type": self.reward_type,
            "guard_pass": bool(self.guard_pass),
            "status": self.status,
            "search_objectives": dict(self.search_objectives),
            "audit_metrics": dict(self.audit_metrics),
            "guard": dict(self.guard),
            "feedback": dict(self.feedback),
            "error": self.error,
        }


@dataclass
class V2RunResult:
    mcts_dir: Path
    final_node: ThetaNode | None
    final_status: str
    baseline_reward: float | None


def theta_to_dict(theta: StrategyTheta) -> dict[str, Any]:
    return {
        "col_1ds": list(theta.col_1ds),
        "col_2ds": list(theta.col_2ds),
        "col_ps": list(theta.col_ps),
        "col_u": theta.col_u,
    }


def _compact_action_dict(action: Any) -> dict[str, Any]:
    if isinstance(action, ThetaAction):
        raw = {"type": action.type, "column": action.column, "old": action.old, "new": action.new}
    elif isinstance(action, dict):
        raw = {
            "type": action.get("type"),
            "column": action.get("column"),
            "old": action.get("old"),
            "new": action.get("new"),
        }
    else:
        return {}
    action_type = str(raw.get("type") or "").strip()
    if not action_type:
        return {}

    def pick(*keys: str) -> Any:
        for key in keys:
            value = raw.get(key)
            if value is not None and str(value).strip() != "":
                return value
        return None

    if action_type.startswith("add_"):
        output = {"type": action_type}
        new = pick("new", "column")
        if new is not None:
            output["new"] = new
        return output
    if action_type.startswith("replace_"):
        output = {"type": action_type}
        old = pick("old", "column")
        new = pick("new")
        if old is not None:
            output["old"] = old
        if new is not None:
            output["new"] = new
        return output
    output = {"type": action_type}
    for key in ("old", "new"):
        value = pick(key)
        if value is not None:
            output[key] = value
    return output


def _compact_action_validation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keep_keys = (
        "fallback",
        "theta_target_mode",
        "llm_actions_match_theta",
        "actions_repaired_from_theta",
        "actions_derived_from_theta",
        "action_derivation_failed",
        "random_replace",
        "locked_copy",
        "transfer_source_node_id",
        "transfer_source_theta_id",
        "transfer_source_score",
    )
    output: dict[str, Any] = {}
    for key in keep_keys:
        item = value.get(key)
        if item is None:
            continue
        output[key] = _prompt_number(item) if isinstance(item, (int, float)) and not isinstance(item, bool) else item
    return output


def _apply_actions_target_inclusive(
    parent_theta: StrategyTheta,
    actions: list[ThetaAction],
    schema_card: dict[str, Any],
) -> StrategyTheta:
    payload = theta_to_dict(parent_theta)
    for action in actions:
        action_type = str(action.type or "").strip()
        new_value = action.new if action.new is not None else action.column
        if action_type == "replace_col_u":
            if new_value:
                payload["col_u"] = str(new_value)
            continue
        scope = {
            "add_col_1d": "col_1ds",
            "replace_col_1d": "col_1ds",
            "add_col_2d": "col_2ds",
            "replace_col_2d": "col_2ds",
            "add_col_p": "col_ps",
            "replace_col_p": "col_ps",
        }.get(action_type)
        if scope is None or not new_value:
            continue
        values = list(payload.get(scope, []))
        if action_type.startswith("add_"):
            if new_value not in values:
                values.append(str(new_value))
        else:
            old_value = action.old if action.old is not None else action.column
            if old_value in values:
                values = [str(new_value) if column == old_value else column for column in values]
            elif new_value not in values:
                values.append(str(new_value))
        payload[scope] = values
    return repair_theta_target_inclusive(payload, schema_card, random.Random(0))


def _derive_actions_between_thetas(
    parent_theta: StrategyTheta,
    child_theta: StrategyTheta,
    schema_card: dict[str, Any],
) -> list[ThetaAction]:
    if canonical_key(parent_theta) == canonical_key(child_theta):
        return []
    target = str(schema_card.get("target_column") or "")
    actions: list[ThetaAction] = []
    scope_specs = [
        ("col_1ds", "add_col_1d", "replace_col_1d"),
        ("col_2ds", "add_col_2d", "replace_col_2d"),
        ("col_ps", "add_col_p", "replace_col_p"),
    ]
    for field_name, add_type, replace_type in scope_specs:
        parent_values = [column for column in getattr(parent_theta, field_name) if column != target]
        child_values = [column for column in getattr(child_theta, field_name) if column != target]
        removed = [column for column in parent_values if column not in child_values]
        added = [column for column in child_values if column not in parent_values]
        for old_value, new_value in zip(removed, added):
            actions.append(ThetaAction(type=replace_type, old=old_value, new=new_value))
        for new_value in added[len(removed) :]:
            actions.append(ThetaAction(type=add_type, new=new_value))
        if len(removed) > len(added):
            return []
    if parent_theta.col_u != child_theta.col_u:
        actions.append(ThetaAction(type="replace_col_u", old=parent_theta.col_u, new=child_theta.col_u))
    applied = _apply_actions_target_inclusive(parent_theta, actions, schema_card)
    if canonical_key(applied) != canonical_key(child_theta):
        return []
    return actions


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, tuple):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _render_prompt(config: V2MCTSConfig, template_name: str, payload: dict[str, Any]) -> str:
    template_dir = Path(config.prompt_pack_dir) / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["tojson"] = lambda value, indent=None: json.dumps(value, ensure_ascii=False, indent=indent)
    return env.get_template(template_name).render(**payload)


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


def _dcr_privacy_reward_for_prompt(raw_dcr: Any, dcr_privacy: Any = None) -> Any:
    privacy = _finite_float(dcr_privacy)
    raw = _finite_float(raw_dcr)
    if privacy is None and raw is not None:
        privacy = 1.0 - abs(raw - 0.5)
    return _prompt_number(privacy)


def _dcr_balance_for_prompt(raw_dcr: Any, dcr_privacy: Any = None) -> dict[str, Any]:
    raw = _finite_float(raw_dcr)
    privacy = _finite_float(dcr_privacy)
    if raw is None and privacy is None:
        return {"available": False}
    balance_error = abs(raw - 0.5) if raw is not None else None
    if privacy is None and balance_error is not None:
        privacy = 1.0 - balance_error
    if raw is None:
        balance_direction = "raw_dcr_unavailable"
    elif raw > 0.5:
        balance_direction = "above_0.5_more_synthetic_rows_closer_to_train_than_test"
    elif raw < 0.5:
        balance_direction = "below_0.5_more_synthetic_rows_closer_to_test_than_train"
    else:
        balance_direction = "balanced_at_0.5"
    return {
        "available": True,
        "raw_dcr_real_closer_rate": _prompt_number(raw),
        "target_raw_dcr": 0.5,
        "balance_error_abs": _prompt_number(balance_error),
        "privacy_reward": _prompt_number(privacy),
        "balance_direction": balance_direction,
    }


def _short_text(value: Any, limit: int = 220) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = " ".join(text.split())
    return text[:limit]


def _compact_target_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, Any] = {}
    for key in ("target", "minority", "positive_class", "task_type"):
        if key in value:
            output[key] = value[key]
    notes = value.get("utility_notes")
    if isinstance(notes, list):
        output["utility_notes"] = [_short_text(item, 120) for item in notes[:2] if _short_text(item, 120)]
    return output


def _compact_risks(value: Any, *, per_key: int = 1) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, list[str]] = {}
    for key in ("shape", "trend", "privacy", "utility"):
        items = value.get(key)
        if isinstance(items, list):
            compact = [_short_text(item, 120) for item in items[:per_key]]
            output[key] = [item for item in compact if item]
    return output


def _compact_quantile_block_for_prompt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"available": False}
    keys = ("available", "min", "q05", "q25", "q50", "q75", "q95", "mean")
    output = {
        key: _prompt_number(value.get(key))
        for key in keys
        if value.get(key) is not None
    }
    return output or {"available": False}


def _compact_dcr_for_prompt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"available": False}
    nested_keys = ("dcr_real", "dcr_test", "margin_real_minus_test")
    if any(isinstance(value.get(key), dict) for key in nested_keys):
        output: dict[str, Any] = {
            "available": bool(value.get("available", True)),
        }
        for key in nested_keys:
            if isinstance(value.get(key), dict):
                output[key] = _compact_quantile_block_for_prompt(value.get(key))
        if value.get("real_closer_rate") is not None:
            output["dcr_balance"] = _dcr_balance_for_prompt(
                value.get("real_closer_rate"),
                value.get("dcr_privacy_reward"),
            )
        if value.get("reason"):
            output["reason"] = _short_text(value.get("reason"), 120)
        return output
    keys = ("available", "q05", "q25", "q50", "q75", "q95", "mean")
    output = {key: _prompt_number(value.get(key)) for key in keys if key in value}
    if value.get("real_closer_rate") is not None:
        output["dcr_balance"] = _dcr_balance_for_prompt(value.get("real_closer_rate"))
    return output


def _compact_metrics_for_prompt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, Any] = {}
    for key in ("shape", "trend", "utility_exact_overall"):
        if key in value:
            output[key] = _prompt_number(value.get(key))
    metric = value.get("utility_exact_metric")
    if metric is not None:
        output["utility_metric"] = metric
        output["utility_metric_direction"] = utility_metric_direction(metric)
    target_scale = value.get("utility_exact_regression_target_transform")
    if target_scale is not None:
        output["utility_target_scale"] = target_scale
    if "dcr" in value or "dcr_privacy" in value:
        output["dcr_balance"] = _dcr_balance_for_prompt(value.get("dcr"), value.get("dcr_privacy"))
    if "metric_reward_score" in value:
        output["metric_reward_score"] = _prompt_number(value.get("metric_reward_score"))
        output["metric_reward_direction"] = "higher_better"
    return output


def _compact_weak_columns_for_prompt(items: Any, *, limit: int = 2) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in list(items or [])[:limit]:
        if not isinstance(item, dict):
            continue
        output.append(
            {
                "column": item.get("column"),
                "score": _prompt_number(item.get("score", item.get("score_mean", item.get("mean_score")))),
            }
        )
    return output


def _compact_weak_pairs_for_prompt(items: Any, *, limit: int = 2) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in list(items or [])[:limit]:
        if not isinstance(item, dict):
            continue
        output.append(
            {
                "left": item.get("left"),
                "right": item.get("right"),
                "score": _prompt_number(item.get("score", item.get("score_mean", item.get("mean_score")))),
            }
        )
    return output


def _compact_utility_feature_for_prompt(item: dict[str, Any]) -> dict[str, Any]:
    rank = item.get("rank_mean")
    if rank is None:
        rank = item.get("rank")
    output = {
        "feature": item.get("feature"),
        "importance": _prompt_number(item.get("importance", item.get("importance_mean"))),
    }
    if rank is not None:
        output["rank"] = _prompt_number(rank)
    return output


def _compact_utility_importance_for_prompt(value: Any, *, limit: int = 4) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(value, dict):
        top_features = list(value.get("top_features", []) or [])[:limit]
        output = {
            key: _prompt_number(value.get(key)) if key in {"test_score"} else value.get(key)
            for key in ("backend", "metric", "test_score", "test_score_direction", "feature_importance_method", "reason")
            if value.get(key) is not None
        }
        if output.get("metric") is not None:
            output.setdefault("test_score_direction", utility_metric_direction(output.get("metric")))
        target_scale = value.get("regression_target_transform")
        if target_scale is not None:
            output["target_scale"] = target_scale
        return output | {
            "top_features": [
                _compact_utility_feature_for_prompt(item)
                for item in top_features
                if isinstance(item, dict)
            ]
        }
    output: list[dict[str, Any]] = []
    for item in list(value or [])[:limit]:
        if not isinstance(item, dict):
            continue
        output.append(_compact_utility_feature_for_prompt(item))
    return output


def _compact_source_contribution_for_prompt(value: Any, *, limit: int = 4) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value.get("available", True):
        return None
    rows: list[dict[str, Any]] = []
    for item in list(value.get("sources", []) or [])[:limit]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "source_id": item.get("source_id"),
                "pool_frac": _prompt_number(item.get("source_pool_fraction")),
                "selected_frac": _prompt_number(item.get("source_selected_fraction")),
                "gain": _prompt_number(item.get("source_contribution_gain")),
            }
        )
    return {"sources": rows}


def _compact_source_profile_for_prompt(
    source_id: str,
    info: SourceInfo | None,
    item: dict[str, Any],
    *,
    include_rows: bool = True,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "source_id": source_id,
        "metrics_mean": _compact_metrics_for_prompt(item.get("metrics_mean", {})),
        "shape_worst": _compact_weak_columns_for_prompt(item.get("shape_worst_columns_mean", []), limit=2),
        "trend_worst": _compact_weak_pairs_for_prompt(item.get("trend_worst_pairs_mean", []), limit=2),
        "dcr_quantiles": _compact_dcr_for_prompt(item.get("dcr_quantiles_mean", {})),
        "utility_top": _compact_utility_importance_for_prompt(
            item.get("utility_xgb_feature_importance_mean", []),
            limit=UTILITY_IMPORTANCE_PROMPT_LIMIT,
        ),
        "semantic_summary": _short_text(item.get("semantic_summary") or _source_semantic_summary(source_id), 260),
    }
    if include_rows and info is not None:
        output["rows"] = int(info.rows)
    if item.get("best_use") is not None:
        output["best_use"] = item.get("best_use")
    return output


def _source_profile_sample_for_storage(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_name": report.get("sample_name"),
        "sample_path": report.get("sample_path"),
        "sample_rows": report.get("sample_rows"),
        "replace": bool(report.get("replace", False)),
        "metrics": _compact_metrics_for_prompt(report.get("metrics", {})),
        "shape_worst": _compact_weak_columns_for_prompt(report.get("shape_worst_columns", []), limit=2),
        "trend_worst": _compact_weak_pairs_for_prompt(report.get("trend_worst_pairs", []), limit=2),
        "dcr_quantiles": _compact_dcr_for_prompt(report.get("dcr_quantiles", {})),
        "utility_top": _compact_utility_importance_for_prompt(
            report.get("utility_xgb_feature_importance", []),
            limit=UTILITY_IMPORTANCE_PROMPT_LIMIT,
        ),
    }


def _sample_frame(df: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    if int(n) <= 0 or len(df) <= int(n):
        return df.reset_index(drop=True)
    return df.sample(n=int(n), random_state=int(seed)).reset_index(drop=True)


def _normalize_source_frame(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_tabdiff_dataframe_columns(dataset_name, df).copy()
    for column in normalized.columns:
        if normalized[column].dtype == object:
            normalized[column] = normalized[column].astype(str).str.strip()
    return normalized


def _load_valid_source_frame(
    *,
    dataset_name: str,
    source_id: str,
    path: Path,
    column_order: list[str],
    schema_card: dict[str, Any],
    stats_card: dict[str, Any],
    validation_dir: Path | None = None,
    save_records: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_df = _normalize_source_frame(dataset_name, pd.read_csv(path))[column_order].copy()
    records = [
        {"candidate_id": int(idx), "row": row}
        for idx, row in zip(source_df.index.tolist(), source_df.to_dict(orient="records"))
    ]
    bundle = TabularValidator(schema_card, stats_card).validate(records)
    valid_source_ids = [int(record["candidate_id"]) for record in bundle.valid_records]
    valid_df = pd.DataFrame(
        [record["row"] for record in bundle.valid_records],
        columns=column_order,
        index=pd.Index(valid_source_ids, name="source_row_id"),
    )
    reject_reason_histogram: dict[str, int] = {}
    for record in bundle.rejected_records:
        reason = str(record.get("reason", "unknown"))
        reject_reason_histogram[reason] = reject_reason_histogram.get(reason, 0) + 1
    report = _json_safe({
        **bundle.report,
        "source_id": source_id,
        "source_path": str(path),
        "rows_before": int(len(source_df)),
        "rows_after": int(len(valid_df)),
        "reject_reason_histogram": reject_reason_histogram,
        "rejected_preview": bundle.rejected_records[:10],
    })
    if validation_dir is not None:
        ensure_dir(validation_dir)
        save_json(validation_dir / "source_validation_summary.json", report)
        if save_records:
            save_jsonl(validation_dir / "source_valid_records.jsonl", _json_safe(bundle.valid_records))
            save_jsonl(validation_dir / "source_rejected_records.jsonl", _json_safe(bundle.rejected_records))
    if len(valid_df) == 0:
        raise ValueError(f"All rows rejected for source={source_id} at {path}")
    return valid_df, report


def _schema_column_types(schema_card: dict[str, Any]) -> dict[str, str]:
    columns = schema_card.get("columns", {}) if isinstance(schema_card.get("columns"), dict) else {}
    return {str(column): str(info.get("type", "categorical")) for column, info in columns.items() if isinstance(info, dict)}


def _is_numeric_column(column: str, column_types: dict[str, str]) -> bool:
    return column_types.get(column) in {"numerical", "discrete_numerical"}


def _quantile_dict(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"available": False}
    return {
        "available": True,
        "min": _prompt_number(np.min(finite)),
        "q05": _prompt_number(np.quantile(finite, 0.05)),
        "q25": _prompt_number(np.quantile(finite, 0.25)),
        "q50": _prompt_number(np.quantile(finite, 0.50)),
        "q75": _prompt_number(np.quantile(finite, 0.75)),
        "q95": _prompt_number(np.quantile(finite, 0.95)),
        "mean": _prompt_number(np.mean(finite)),
    }


def _mean_quantile_dict(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"available": False}
    keys = ("min", "q05", "q25", "q50", "q75", "q95", "mean")
    output: dict[str, Any] = {"available": True}
    for key in keys:
        values = [float(item[key]) for item in items if item.get("available") and item.get(key) is not None]
        output[key] = _prompt_number(float(np.mean(values))) if values else None
    return output


def _encode_target(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    labels = sorted(str(value) for value in series.dropna().astype(str).unique())
    mapping = {label: idx for idx, label in enumerate(labels)}
    return series.astype(str).map(mapping), mapping


def _feature_from_encoded_column(encoded_column: str, feature_columns: list[str], categorical_columns: list[str]) -> str:
    if encoded_column in feature_columns:
        return encoded_column
    for feature in categorical_columns:
        prefix = f"{feature}="
        if encoded_column.startswith(prefix):
            return feature
    return encoded_column.split("=", 1)[0]


def _utility_feature_importance(
    *,
    train_like_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
    seed: int,
    sample_size: int,
) -> dict[str, Any]:
    target = str(schema_card.get("target_column"))
    feature_columns = _schema_feature_columns(schema_card)
    column_types = _schema_column_types(schema_card)
    if target not in train_like_df.columns or not feature_columns:
        return _static_utility_importance(dataset_context, reason="target_or_features_unavailable")

    train_df = _sample_frame(train_like_df[[*feature_columns, target]].dropna(subset=[target]), n=int(sample_size), seed=seed)
    if train_df[target].astype(str).nunique(dropna=True) < 2:
        return _static_utility_importance(dataset_context, reason="single_target_class")
    test_sample = _sample_frame(test_df[[*feature_columns, target]].dropna(subset=[target]), n=int(sample_size), seed=seed + 23)
    categorical_columns = [column for column in feature_columns if not _is_numeric_column(column, column_types)]
    x_train = pd.get_dummies(train_df[feature_columns], columns=categorical_columns, dummy_na=True, prefix_sep="=")
    x_test = pd.get_dummies(test_sample[feature_columns], columns=categorical_columns, dummy_na=True, prefix_sep="=")
    x_train, x_test = x_train.align(x_test, join="outer", axis=1, fill_value=0)
    y_train, mapping = _encode_target(train_df[target])
    if len(mapping) < 2:
        return _static_utility_importance(dataset_context, reason="encoded_single_target_class")
    y_test = test_sample[target].astype(str).map(mapping)
    valid_test = y_test.notna()
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic" if len(mapping) == 2 else "multi:softprob",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=int(seed),
            verbosity=0,
        )
        backend = "xgboost"
        model.fit(x_train.to_numpy(dtype=float), y_train.to_numpy(dtype=int))
        importances = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
        metric_name = None
        test_score = None
        if valid_test.any():
            from sklearn.metrics import accuracy_score, roc_auc_score

            x_eval = x_test.loc[valid_test].to_numpy(dtype=float)
            y_eval = y_test.loc[valid_test].to_numpy(dtype=int)
            if len(mapping) == 2 and hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_eval)[:, 1]
                test_score = float(roc_auc_score(y_eval, proba))
                metric_name = "roc_auc"
            else:
                pred = model.predict(x_eval)
                test_score = float(accuracy_score(y_eval, pred))
                metric_name = "accuracy"
    except Exception as exc:
        return _static_utility_importance(dataset_context, reason=f"xgboost_unavailable_or_failed: {exc}")

    if importances.size != len(x_train.columns) or float(importances.sum()) <= 0:
        return _static_utility_importance(dataset_context, reason="feature_importances_unavailable")

    grouped = {feature: 0.0 for feature in feature_columns}
    for encoded_column, importance in zip(x_train.columns, importances):
        feature = _feature_from_encoded_column(str(encoded_column), feature_columns, categorical_columns)
        grouped[feature] = grouped.get(feature, 0.0) + float(importance)
    total = sum(grouped.values()) or 1.0
    top_features = [
        {"feature": feature, "importance": _prompt_number(value / total), "rank": int(rank)}
        for rank, (feature, value) in enumerate(
            sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:8],
            start=1,
        )
    ]
    return {
        "backend": backend,
        "metric": metric_name,
        "test_score": _prompt_number(test_score) if test_score is not None else None,
        "top_features": top_features,
    }


def _real_utility_feature_importance_for_selected_evaluator(
    *,
    train_like_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_card: dict[str, Any],
    stats_card: dict[str, Any],
    dataset_context: dict[str, Any],
    config: V2MCTSConfig,
    seed: int,
    sample_size: int,
) -> dict[str, Any]:
    if str(config.utility_exact_evaluator) != "torch_lightweight_mlp":
        return _utility_feature_importance(
            train_like_df=train_like_df,
            test_df=test_df,
            schema_card=schema_card,
            dataset_context=dataset_context,
            seed=seed,
            sample_size=0,
        )

    column_order = list(schema_card.get("column_order", []))
    target = str(schema_card.get("target_column") or "")
    if not column_order or target not in train_like_df.columns or target not in test_df.columns:
        return _static_utility_importance(dataset_context, reason="torch_lightweight_mlp_target_or_columns_unavailable")
    if any(column not in train_like_df.columns or column not in test_df.columns for column in column_order):
        return _static_utility_importance(dataset_context, reason="torch_lightweight_mlp_columns_unavailable")

    train_eval = train_like_df[column_order].dropna(subset=[target]).copy().reset_index(drop=True)
    test_eval = test_df[column_order].dropna(subset=[target]).copy().reset_index(drop=True)
    if train_eval.empty or test_eval.empty:
        return _static_utility_importance(dataset_context, reason="torch_lightweight_mlp_empty_sample")
    if train_eval[target].astype(str).nunique(dropna=True) < 2:
        return _static_utility_importance(dataset_context, reason="torch_lightweight_mlp_single_target_class")

    eval_device = resolve_eval_device(config.eval_device)
    nn_device = resolve_nn_device(config.nn_device, eval_device)
    selector = ParetoSelector(
        train_df=train_like_df[column_order].copy(),
        holdout_df=test_df[column_order].copy(),
        schema_card=schema_card,
        stats_card=stats_card,
        seed=config.seed,
        source="real_utility_profile",
        privacy_version="v2",
        density_reference_size=config.density_reference_size,
        nn_device=nn_device,
        high_cardinality_enabled=False,
    )
    selector.utility_exact_evaluator = config.utility_exact_evaluator
    selector.utility_exact_torch_epochs = config.utility_exact_torch_epochs
    selector.utility_exact_torch_importance_sample_size = 0

    report = compute_utility_exact_metrics(
        selector,
        train_eval,
        test_eval,
        evaluator=config.utility_exact_evaluator,
        random_state=seed,
    )
    if not isinstance(report, dict) or not report.get("available", False):
        reason = "torch_lightweight_mlp_unavailable"
        if isinstance(report, dict) and report.get("reason"):
            reason = f"{reason}: {report.get('reason')}"
        return _static_utility_importance(dataset_context, reason=reason)

    top_features = _utility_importance_rows_from_exact_report(report, limit=10)
    if not top_features:
        return _static_utility_importance(dataset_context, reason="torch_lightweight_mlp_feature_importances_unavailable")

    metric = report.get("metric")
    raw_direction = utility_metric_direction(metric)
    return {
        "backend": str(report.get("protocol") or "torch_lightweight_mlp"),
        "metric": metric,
        "test_score": _prompt_number(report.get("overall")),
        "test_score_direction": raw_direction,
        "test_score_semantics": "raw_metric",
        "feature_importance_method": report.get("feature_importance_method"),
        "runtime_model_device": report.get("runtime_model_device"),
        "regression_target_transform": report.get("regression_target_transform"),
        "regression_target_clip_min": report.get("regression_target_clip_min"),
        "regression_target_clip_max": report.get("regression_target_clip_max"),
        "utility_full_eval": True,
        "train_rows_used": int(len(train_eval)),
        "test_rows_used": int(len(test_eval)),
        "feature_importance_test_rows_used": int(len(test_eval)),
        "utility_exact_overall": _prompt_number(report.get("overall")),
        "utility_exact_raw_direction": raw_direction,
        "utility_exact_overall_semantics": "raw_metric",
        "top_features": top_features,
    }


def _static_utility_importance(dataset_context: dict[str, Any], *, reason: str) -> dict[str, Any]:
    priorities = []
    if isinstance(dataset_context.get("theta_guidance"), dict):
        priorities = list(dataset_context["theta_guidance"].get("utility_priority", []) or [])
    return {
        "backend": "static_dataset_context",
        "reason": reason,
        "test_score_semantics": "raw_metric",
        "utility_exact_overall_semantics": "raw_metric",
        "top_features": [
            {"feature": str(feature), "importance": _prompt_number(1.0 / (idx + 1)), "rank": int(idx + 1)}
            for idx, feature in enumerate(priorities[:8])
        ],
    }


def _dataset_brief_for_prompt(schema_card: dict[str, Any], dataset_context: dict[str, Any]) -> dict[str, Any]:
    columns = schema_card.get("columns", {}) if isinstance(schema_card.get("columns"), dict) else {}
    feature_columns = _schema_feature_columns(schema_card)
    guidance = dataset_context.get("theta_guidance", {}) if isinstance(dataset_context.get("theta_guidance"), dict) else {}
    bounds = theta_size_bounds_target_inclusive(len(schema_card.get("column_order", [])))
    return {
        "dataset": schema_card.get("dataset", "adult"),
        "target": schema_card.get("target_column"),
        "feature_columns": feature_columns,
        "column_types": {
            column: columns.get(column, {}).get("type")
            for column in schema_card.get("column_order", [])
            if isinstance(columns.get(column), dict)
        },
        "theta_size_bounds": bounds,
        "shape_priority_columns": list(guidance.get("shape_priority", []) or [])[:6],
        "trend_priority_columns": list(guidance.get("trend_priority", []) or [])[:6],
        "privacy_priority_columns": list(guidance.get("privacy_priority", []) or [])[:6],
        "utility_priority_columns": list(guidance.get("utility_priority", []) or [])[:6],
        "target_summary": _compact_target_summary(dataset_context.get("target_summary", {})),
    }


def _prompt_column_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        values = list(value)
    except TypeError:
        return [str(value)]
    output: list[str] = []
    for item in values:
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, (list, tuple, set)):
            output.extend(str(part) for part in item)
        else:
            output.append(str(item))
    return output


def _unique_known_columns(values: list[Any], known: set[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        column = str(raw).strip()
        if not column or column not in known or column in seen:
            continue
        seen.add(column)
        output.append(column)
    return output


def _seed_priority_order(
    *,
    field_name: str,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
) -> list[str]:
    column_order = [str(column) for column in schema_card.get("column_order", [])]
    feature_columns = _schema_feature_columns(schema_card)
    known = set(feature_columns if field_name == "col_ps" else column_order)
    guidance = dataset_context.get("theta_guidance", {}) if isinstance(dataset_context.get("theta_guidance"), dict) else {}
    if field_name == "col_1ds":
        keys = ("shape_priority", "utility_priority", "trend_priority", "privacy_priority")
    elif field_name == "col_2ds":
        keys = ("trend_priority", "shape_priority", "utility_priority", "privacy_priority")
    else:
        keys = ("privacy_priority", "shape_priority", "trend_priority", "utility_priority")

    values: list[Any] = []
    for key in keys:
        values.extend(_prompt_column_list(guidance.get(key, [])))
    values.extend(feature_columns if field_name == "col_ps" else column_order)
    return _unique_known_columns(values, known)


def _expand_seed_theta_field_to_bounds(
    *,
    field_name: str,
    values: Any,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
) -> list[str]:
    column_order = [str(column) for column in schema_card.get("column_order", [])]
    feature_columns = _schema_feature_columns(schema_card)
    valid_columns = feature_columns if field_name == "col_ps" else column_order
    known = set(valid_columns)
    bounds = theta_size_bounds_target_inclusive(len(column_order))
    min_size = int(bounds[field_name]["min"])
    max_size = int(bounds[field_name]["max"])

    output: list[str] = []
    for column in _unique_known_columns(_prompt_column_list(values), known):
        if column not in output:
            output.append(column)
    for column in _seed_priority_order(field_name=field_name, schema_card=schema_card, dataset_context=dataset_context):
        if len(output) >= min_size:
            break
        if column not in output:
            output.append(column)
    if len(output) > max_size:
        kept: list[str] = []
        for column in output:
            if column in kept:
                continue
            if len(kept) >= max_size:
                break
            kept.append(column)
        output = kept
    return output


def _seed_col_u_for_prompt(value: Any, schema_card: dict[str, Any], dataset_context: dict[str, Any]) -> str:
    features = _schema_feature_columns(schema_card)
    feature_set = set(features)
    candidate = str(value or "").strip()
    if candidate in feature_set:
        return candidate
    guidance = dataset_context.get("theta_guidance", {}) if isinstance(dataset_context.get("theta_guidance"), dict) else {}
    for column in _prompt_column_list(guidance.get("utility_priority", [])) + features:
        candidate = str(column).strip()
        if candidate in feature_set:
            return candidate
    return ""


def _seed_theta_examples_for_prompt(
    *,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
    limit: int = 4,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in dataset_context.get("seed_theta_examples", []) or []:
        if not isinstance(item, dict) or not isinstance(item.get("theta"), dict):
            continue
        theta_payload = item["theta"]
        theta = StrategyTheta(
            col_1ds=tuple(
                _expand_seed_theta_field_to_bounds(
                    field_name="col_1ds",
                    values=theta_payload.get("col_1ds"),
                    schema_card=schema_card,
                    dataset_context=dataset_context,
                )
            ),
            col_2ds=tuple(
                _expand_seed_theta_field_to_bounds(
                    field_name="col_2ds",
                    values=theta_payload.get("col_2ds"),
                    schema_card=schema_card,
                    dataset_context=dataset_context,
                )
            ),
            col_ps=tuple(
                _expand_seed_theta_field_to_bounds(
                    field_name="col_ps",
                    values=theta_payload.get("col_ps"),
                    schema_card=schema_card,
                    dataset_context=dataset_context,
                )
            ),
            col_u=_seed_col_u_for_prompt(theta_payload.get("col_u"), schema_card, dataset_context),
        )
        if not validate_theta_target_inclusive(theta, schema_card).ok:
            theta = repair_theta_target_inclusive(theta, schema_card, random.Random(len(output) + 9173))
        if not validate_theta_target_inclusive(theta, schema_card).ok:
            continue
        output.append(
            {
                "family": item.get("family"),
                "theta": theta_to_dict(theta),
                "reason": item.get("reason"),
            }
        )
        if len(output) >= int(limit):
            break
    return output


def _file_fingerprint(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _stable_json_hash(payload: dict[str, Any]) -> str:
    import hashlib

    text = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean_finite(values: list[Any]) -> float | None:
    finite = [_finite_float(value) for value in values]
    cleaned = [value for value in finite if value is not None]
    if not cleaned:
        return None
    return float(np.mean(cleaned))


def _search_reward_from_proxy(
    *,
    audit_metrics: dict[str, Any],
    search_objectives: dict[str, Any],
    feedback: dict[str, Any],
) -> tuple[float, bool]:
    utility_summary = feedback.get("utility_summary", {}) if isinstance(feedback.get("utility_summary"), dict) else {}
    utility_proxy = _finite_float(search_objectives.get("U_proxy_theta"))
    if utility_proxy is None:
        utility_proxy = _finite_float(utility_summary.get("search_utility_proxy"))
    if utility_proxy is None:
        utility_proxy = _finite_float(audit_metrics.get("utility_exact"))
    score = _mean_finite(
        [
            audit_metrics.get("shape_global"),
            audit_metrics.get("trend_global"),
            audit_metrics.get("dcr_privacy"),
            utility_proxy,
        ]
    )
    if score is None:
        return 0.0, False
    return max(0.0, min(1.0, float(score))), True


def _node_selection_score(node: "ThetaNode") -> float:
    if node.search_reward_available:
        return float(node.search_reward)
    if node.exact_reward_available:
        return float(node.exact_reward)
    return float(node.reward)


def _exact_reward_failure_reason(node: "ThetaNode") -> str | None:
    if node.exact_reward_available:
        return None
    if node.rollout_dir is not None:
        for rel in (
            "eval/selection_pareto/utility_exact_report.json",
            "eval/selection_pareto/utility_metrics_summary.json",
            "eval/pareto/utility_exact_report.json",
            "eval/pareto/utility_metrics_summary.json",
            "utility_exact_report.json",
            "utility_metrics_summary.json",
        ):
            path = Path(node.rollout_dir) / rel
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            reason = payload.get("reason")
            if reason:
                return str(reason)
    failed_checks = node.guard.get("failed_checks") if isinstance(node.guard, dict) else None
    if failed_checks:
        return ",".join(str(item) for item in failed_checks)
    return None


def build_real_utility_reference(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_card: dict[str, Any],
    stats_card: dict[str, Any],
    dataset_context: dict[str, Any],
    config: V2MCTSConfig,
) -> dict[str, Any]:
    return {
        "dataset": config.dataset_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "utility_feature_importance": _real_utility_feature_importance_for_selected_evaluator(
            train_like_df=train_df,
            test_df=test_df,
            schema_card=schema_card,
            stats_card=stats_card,
            dataset_context=dataset_context,
            config=config,
            seed=config.seed + 701,
            sample_size=config.utility_diag_sample_size,
        ),
        "target_summary": dataset_context.get("target_summary", {}),
    }


def _default_real_utility_semantic_summary(
    *,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
    utility_reference: dict[str, Any],
) -> str:
    target = str(schema_card.get("target_column") or "target")
    utility = utility_reference.get("utility_feature_importance", {})
    top_features = []
    if isinstance(utility, dict):
        top_features = [
            str(item.get("feature"))
            for item in utility.get("top_features", []) or []
            if isinstance(item, dict) and item.get("feature") is not None
        ]
    target_summary = dataset_context.get("target_summary", {}) if isinstance(dataset_context, dict) else {}
    task_hint = str(target_summary.get("task_type") or schema_card.get("task_type") or "prediction")
    if top_features:
        return f"Real utility anchors {target} {task_hint} on {', '.join(top_features[:5])}."
    return f"Real utility anchors {target} {task_hint}; preserve target coverage and feature-label relationships."


def build_real_utility_profile(
    *,
    config: V2MCTSConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_card: dict[str, Any],
    stats_card: dict[str, Any],
    dataset_context: dict[str, Any],
    context_dir: Path,
    client: LLMClient | None,
    trace_dir: Path,
) -> dict[str, Any]:
    cache_path = Path(context_dir) / "real_utility_profile.json"
    cache_key_payload = {
        "version": "v2_real_utility_profile_v5_log_target_rmse",
        "dataset_name": config.dataset_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "columns": list(schema_card.get("column_order", [])),
        "utility_diag_sample_size": int(config.utility_diag_sample_size),
        "utility_exact_evaluator": config.utility_exact_evaluator,
        "utility_exact_torch_epochs": int(config.utility_exact_torch_epochs),
        "utility_full_eval": True,
        "utility_exact_torch_importance_sample_size": 0,
        "eval_device": config.eval_device,
        "nn_device": config.nn_device,
    }
    cache_key = _stable_json_hash(cache_key_payload)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
        if cached.get("cache_key") == cache_key:
            return cached

    profile = build_real_utility_reference(
        train_df=train_df,
        test_df=test_df,
        schema_card=schema_card,
        stats_card=stats_card,
        dataset_context=dataset_context,
        config=config,
    )
    profile["cache_key"] = cache_key
    profile["cache_key_payload"] = cache_key_payload
    profile["semantic_summary"] = _default_real_utility_semantic_summary(
        schema_card=schema_card,
        dataset_context=dataset_context,
        utility_reference=profile,
    )
    prompt = _render_prompt(
        config,
        "v2_real_utility_profile_summary_prompt.j2",
        {
            "dataset_brief": _dataset_brief_for_prompt(schema_card, dataset_context),
            "real_utility_profile": _real_utility_for_prompt(profile),
        },
    )
    payload = _call_llm_json(
        client=client,
        prompt=prompt,
        schema_name="context_real_utility_profile_summary",
        trace_dir=trace_dir,
    )
    if isinstance(payload, dict) and str(payload.get("semantic_summary", "")).strip():
        profile["semantic_summary"] = str(payload.get("semantic_summary", ""))[:600]
    save_json(cache_path, profile)
    return profile


def _tabdiff_metric_column_lookup(dataset_name: str) -> dict[str, str | None]:
    info = load_tabdiff_info(dataset_name)
    names = [str(name) for name in info.get("column_names", [])]
    num_idx = [int(idx) for idx in info.get("num_col_idx", [])]
    cat_idx = [int(idx) for idx in info.get("cat_col_idx", [])]
    target_idx = [int(idx) for idx in info.get("target_col_idx", [])]
    if str(info.get("task_type", "binclass")) == "regression":
        ordered = num_idx + target_idx + cat_idx
    else:
        ordered = num_idx + cat_idx + target_idx
    lookup: dict[str, str | None] = {}
    for out_idx, raw_idx in enumerate(ordered):
        name = names[raw_idx] if raw_idx < len(names) else None
        lookup[str(out_idx)] = name
        if name is not None:
            lookup[name] = name
    return lookup


def _score_from_detail(row: dict[str, Any]) -> float | None:
    for key in ("Score", "score", "Quality Score", "QualityScore"):
        if key in row:
            try:
                parsed = float(row[key])
            except Exception:
                return None
            return parsed if math.isfinite(parsed) else None
    return None


def _read_metric_details(eval_dir: Path, selection_name: str) -> dict[str, Any]:
    target_dir = Path(eval_dir) / selection_name
    output: dict[str, Any] = {}
    for key, filename in (("shape_details", "shapes.csv"), ("trend_details", "trends.csv")):
        path = target_dir / filename
        if path.exists():
            output[key] = pd.read_csv(path).to_dict(orient="records")
    dcr_path = target_dir / "dcr.csv"
    if dcr_path.exists():
        dcr_df = pd.read_csv(dcr_path)
        output["dcr_quantiles"] = _dcr_quantiles_from_frame(dcr_df)
    return output


def _dcr_quantiles_from_frame(dcr_df: pd.DataFrame) -> dict[str, Any]:
    if "dcr_real" not in dcr_df.columns or "dcr_test" not in dcr_df.columns:
        return {"available": False}
    real = pd.to_numeric(dcr_df["dcr_real"], errors="coerce").to_numpy(dtype=float)
    test = pd.to_numeric(dcr_df["dcr_test"], errors="coerce").to_numpy(dtype=float)
    margin = real - test
    finite = np.isfinite(real) & np.isfinite(test)
    real_closer_rate = float(np.mean(real[finite] < test[finite])) if finite.any() else None
    balance_error = abs(real_closer_rate - 0.5) if real_closer_rate is not None else None
    return {
        "available": True,
        "dcr_real": _quantile_dict(real),
        "dcr_test": _quantile_dict(test),
        "margin_real_minus_test": _quantile_dict(margin),
        "real_closer_rate": _prompt_number(real_closer_rate) if real_closer_rate is not None else None,
        "balance_error_abs": _prompt_number(balance_error) if balance_error is not None else None,
        "dcr_privacy_reward": _prompt_number(1.0 - balance_error) if balance_error is not None else None,
    }


def _weak_shape_from_details(
    details: list[dict[str, Any]],
    column_lookup: dict[str, str | None],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in details:
        if not isinstance(row, dict):
            continue
        score = _score_from_detail(row)
        column = row.get("Column") or row.get("column") or row.get("Column Name") or row.get("ColumnName")
        mapped = column_lookup.get(str(column))
        if score is None or mapped is None:
            continue
        rows.append({"column": mapped, "score": _prompt_number(score), "metric": row.get("Metric") or row.get("metric")})
    rows.sort(key=lambda item: (float(item.get("score") or 0.0), str(item.get("column"))))
    return rows[:limit]


def _weak_trend_from_details(
    details: list[dict[str, Any]],
    column_lookup: dict[str, str | None],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in details:
        if not isinstance(row, dict):
            continue
        score = _score_from_detail(row)
        left = row.get("Column 1") or row.get("Column1") or row.get("column_1") or row.get("left")
        right = row.get("Column 2") or row.get("Column2") or row.get("column_2") or row.get("right")
        mapped_left = column_lookup.get(str(left))
        mapped_right = column_lookup.get(str(right))
        if score is None or mapped_left is None or mapped_right is None:
            continue
        rows.append(
            {
                "left": mapped_left,
                "right": mapped_right,
                "score": _prompt_number(score),
                "metric": row.get("Metric") or row.get("metric"),
            }
        )
    rows.sort(key=lambda item: (float(item.get("score") or 0.0), str(item.get("left")), str(item.get("right"))))
    return rows[:limit]


def _mean_metric_dict(reports: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("shape", "trend", "dcr", "dcr_privacy", "utility_exact_overall", "metric_reward_score")
    output: dict[str, Any] = {}
    for key in keys:
        values = [float(report[key]) for report in reports if report.get(key) is not None]
        if values:
            output[key] = _prompt_number(float(np.mean(values)))
    for key in (
        "utility_exact_metric",
        "utility_exact_raw_direction",
        "utility_exact_overall_semantics",
        "metric_reward_score_direction",
        "metric_reward_score_semantics",
        "utility_exact_regression_target_transform",
        "utility_exact_regression_target_clip_min",
        "utility_exact_regression_target_clip_max",
    ):
        for report in reports:
            if report.get(key) is not None:
                output[key] = report.get(key)
                break
    if output.get("utility_exact_metric") is not None and output.get("utility_exact_raw_direction") is None:
        output["utility_exact_raw_direction"] = utility_metric_direction(output.get("utility_exact_metric"))
    if "utility_exact_overall" in output:
        output.setdefault("utility_exact_overall_semantics", "raw_metric")
    if "metric_reward_score" in output:
        output.setdefault("metric_reward_score_direction", "higher_better")
        output.setdefault("metric_reward_score_semantics", "normalized_reward")
    return output


def _std_metric_dict(reports: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("shape", "trend", "dcr", "dcr_privacy", "utility_exact_overall", "metric_reward_score")
    output: dict[str, Any] = {}
    for key in keys:
        values = [float(report[key]) for report in reports if report.get(key) is not None]
        if values:
            output[key] = _prompt_number(float(np.std(values)))
    return output


def _aggregate_weak_columns(reports: list[dict[str, Any]], key: str, *, limit: int = 5) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    meta: dict[tuple[Any, ...], dict[str, Any]] = {}
    for report in reports:
        for item in report.get(key, []) or []:
            if not isinstance(item, dict) or item.get("score") is None:
                continue
            if key == "shape_worst_columns":
                group_key = (item.get("column"),)
            else:
                group_key = (item.get("left"), item.get("right"))
            grouped.setdefault(group_key, []).append(float(item["score"]))
            meta[group_key] = dict(item)
    rows: list[dict[str, Any]] = []
    for group_key, values in grouped.items():
        item = dict(meta[group_key])
        item.pop("score", None)
        item["score_mean"] = _prompt_number(float(np.mean(values)))
        item["score_std"] = _prompt_number(float(np.std(values)))
        item["count"] = len(values)
        rows.append(item)
    rows.sort(key=lambda item: (float(item.get("score_mean") or 1.0), -int(item.get("count") or 0)))
    return rows[:limit]


def _aggregate_dcr_quantiles(reports: list[dict[str, Any]]) -> dict[str, Any]:
    output = {"available": False}
    blocks = [report.get("dcr_quantiles", {}) for report in reports if isinstance(report.get("dcr_quantiles"), dict)]
    if not blocks:
        return output
    result: dict[str, Any] = {"available": True}
    for key in ("dcr_real", "dcr_test", "margin_real_minus_test"):
        result[key] = _mean_quantile_dict([block.get(key, {}) for block in blocks if isinstance(block.get(key), dict)])
    rates = [float(block["real_closer_rate"]) for block in blocks if block.get("real_closer_rate") is not None]
    result["real_closer_rate"] = _prompt_number(float(np.mean(rates))) if rates else None
    if rates:
        errors = [abs(rate - 0.5) for rate in rates]
        result["balance_error_abs"] = _prompt_number(float(np.mean(errors)))
        result["dcr_privacy_reward"] = _prompt_number(float(np.mean([1.0 - error for error in errors])))
    return result


def _aggregate_utility_importance(reports: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    ranks: dict[str, list[float]] = {}
    for report in reports:
        rows = report.get("utility_feature_importance") or report.get("utility_xgb_feature_importance") or []
        for rank, item in enumerate(rows, start=1):
            if not isinstance(item, dict) or item.get("feature") is None:
                continue
            feature = str(item["feature"])
            importance = item.get("importance", item.get("importance_mean"))
            if importance is None:
                continue
            grouped.setdefault(feature, []).append(float(importance))
            ranks.setdefault(feature, []).append(float(item.get("rank", rank)))
    rows = [
        {
            "feature": feature,
            "importance_mean": _prompt_number(float(np.mean(values))),
            "importance_std": _prompt_number(float(np.std(values))),
            "rank_mean": _prompt_number(float(np.mean(ranks.get(feature, [limit + 1])))),
        }
        for feature, values in grouped.items()
    ]
    rows.sort(key=lambda item: (-float(item.get("importance_mean") or 0.0), float(item.get("rank_mean") or 999.0)))
    return rows[:limit]


def _utility_importance_rows_from_exact_report(report: Any, *, limit: int = 10) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(report.get("feature_importance", []) or [], start=1):
        if not isinstance(item, dict) or item.get("feature") is None:
            continue
        importance = item.get("importance", item.get("importance_mean"))
        if importance is None:
            continue
        rows.append(
            {
                "feature": str(item["feature"]),
                "importance": _prompt_number(importance),
                "rank": int(item.get("rank", rank)),
            }
        )
    rows.sort(key=lambda item: (int(item.get("rank", 999)), -float(item.get("importance") or 0.0)))
    return rows[:limit]


def _load_utility_exact_report(path_or_dir: Path | str | None) -> dict[str, Any] | None:
    if path_or_dir is None:
        return None
    base = Path(path_or_dir)
    candidates = [base] if base.name == "utility_metrics_summary.json" else [
        base / "eval" / "selection_pareto" / "utility_metrics_summary.json",
        base / "eval" / "pareto" / "utility_metrics_summary.json",
        base / "utility_metrics_summary.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _utility_importance_for_selected_evaluator(
    *,
    config: V2MCTSConfig,
    train_like_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
    seed: int,
    sample_size: int,
    utility_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if str(config.utility_exact_evaluator) == "torch_lightweight_mlp":
        rows = _utility_importance_rows_from_exact_report(utility_report, limit=10)
        if rows:
            return rows
    return _utility_importance_for_profile(
        train_like_df=train_like_df,
        test_df=test_df,
        schema_card=schema_card,
        dataset_context=dataset_context,
        seed=seed,
        sample_size=sample_size,
    )


def _utility_importance_for_profile(
    *,
    train_like_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
    seed: int,
    sample_size: int,
) -> list[dict[str, Any]]:
    report = _utility_feature_importance(
        train_like_df=train_like_df,
        test_df=test_df,
        schema_card=schema_card,
        dataset_context=dataset_context,
        seed=seed,
        sample_size=sample_size,
    )
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(report.get("top_features", []) or [], start=1):
        if not isinstance(item, dict) or item.get("feature") is None:
            continue
        rows.append(
            {
                "feature": str(item["feature"]),
                "importance": _prompt_number(item.get("importance")),
                "rank": int(rank),
            }
        )
    return rows


def build_source_profiles(
    *,
    config: V2MCTSConfig,
    sources: dict[str, SourceInfo],
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_card: dict[str, Any],
    stats_card: dict[str, Any],
    dataset_context: dict[str, Any],
    context_dir: Path,
    client: LLMClient | None,
    trace_dir: Path,
) -> dict[str, Any]:
    profile_root = ensure_dir(Path(context_dir) / "source_profiles")
    cache_path = Path(context_dir) / "source_profiles.json"
    column_order = list(schema_card["column_order"])
    train_rows = int(len(train_df))
    if config.mode == "single":
        single_profile_cap = None if config.source_profile_sample_rows is None else max(1, int(config.source_profile_sample_rows))
        profile_mode = (
            "single_deterministic_head_profile_no_random_sampling"
            if single_profile_cap is not None
            else "single_full_source_no_random_sampling"
        )
        effective_repeats = 1
        sample_rows: int | str = "full_source" if single_profile_cap is None else int(single_profile_cap)
    else:
        single_profile_cap = None
        profile_mode = "mixed_repeated_random_sampling"
        effective_repeats = max(1, int(config.source_profile_repeats))
        requested_sample_rows = train_rows if config.source_profile_sample_rows is None else int(config.source_profile_sample_rows)
        sample_rows = max(1, min(requested_sample_rows, train_rows))
    cache_key_payload = {
        "version": SOURCE_PROFILE_VERSION,
        "dataset_name": config.dataset_name,
        "mode": config.mode,
        "smoke": bool(config.smoke),
        "profile_mode": profile_mode,
        "profile_repeats": int(effective_repeats),
        "profile_sample_rows": sample_rows,
        "density_reference_size": int(config.density_reference_size),
        "eval_device": config.eval_device,
        "nn_device": config.nn_device,
        "utility_exact_evaluator": config.utility_exact_evaluator,
        "utility_exact_torch_epochs": int(config.utility_exact_torch_epochs),
        "utility_full_eval": True,
        "utility_exact_torch_importance_sample_size": 0,
        "schema_columns": column_order,
        "schema_card_hash": _stable_json_hash(schema_card),
        "stats_card_hash": _stable_json_hash(stats_card),
        "sources": {source_id: _file_fingerprint(info.path) for source_id, info in sources.items()},
    }
    cache_key = _stable_json_hash(cache_key_payload)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
        if cached.get("cache_key") == cache_key:
            return cached

    profile_train_path = Path(context_dir) / "profile_eval_train.csv"
    profile_holdout_path = Path(context_dir) / "profile_eval_holdout.csv"
    profile_test_path = Path(context_dir) / "profile_eval_test.csv"
    save_csv(profile_train_path, train_df[column_order])
    save_csv(profile_holdout_path, holdout_df[column_order])
    save_csv(profile_test_path, test_df[column_order])

    eval_device = resolve_eval_device(config.eval_device)
    nn_device = resolve_nn_device(config.nn_device, eval_device)
    selector = ParetoSelector(
        train_df=train_df[column_order].copy(),
        holdout_df=holdout_df[column_order].copy(),
        schema_card=schema_card,
        stats_card=stats_card,
        seed=config.seed,
        source="source_profile",
        privacy_version="v2",
        density_reference_size=config.density_reference_size,
        nn_device=nn_device,
        high_cardinality_enabled=False,
    )
    selector.utility_exact_evaluator = config.utility_exact_evaluator
    selector.utility_exact_torch_epochs = config.utility_exact_torch_epochs
    selector.utility_exact_torch_importance_sample_size = 0
    runner = TabDiffMetricRunner(
        dataset_name=config.dataset_name,
        device=eval_device,
        metric_list=["density", "dcr"],
        real_data_path=profile_train_path,
        test_data_path=profile_test_path,
        val_data_path=profile_holdout_path,
    )
    column_lookup = _tabdiff_metric_column_lookup(config.dataset_name)
    source_profiles: dict[str, Any] = {}
    for source_id, info in sources.items():
        source_dir = ensure_dir(profile_root / source_id)
        sample_dir = ensure_dir(source_dir / "samples")
        eval_dir = ensure_dir(source_dir / "eval")
        source_df, source_validation = _load_valid_source_frame(
            dataset_name=config.dataset_name,
            source_id=source_id,
            path=info.path,
            column_order=column_order,
            schema_card=schema_card,
            stats_card=stats_card,
            validation_dir=source_dir / "validation",
            save_records=config.save_validation_records,
        )
        repeat_reports: list[dict[str, Any]] = []
        sample_profiles: list[dict[str, Any]] = []
        for repeat_idx in range(effective_repeats):
            seed = int(config.seed + 1777 * (repeat_idx + 1) + sum(ord(ch) for ch in source_id))
            if config.mode == "single":
                replace = False
                if single_profile_cap is None:
                    sample_df = source_df.reset_index(drop=True)
                else:
                    sample_df = source_df.head(int(single_profile_cap)).reset_index(drop=True)
            else:
                sample_n = int(sample_rows)
                replace = sample_n > len(source_df)
                sample_df = source_df.sample(n=sample_n, replace=replace, random_state=seed).reset_index(drop=True)
            sample_name = f"profile_{repeat_idx:03d}"
            sample_path = sample_dir / f"{sample_name}.csv"
            save_csv(sample_path, sample_df)
            summary = evaluate_one_selection(
                selection_name=sample_name,
                df=sample_df,
                selector=selector,
                runner=runner,
                eval_dir=eval_dir,
                test_df=test_df[column_order].copy(),
            )
            summary["audit_metrics"] = build_audit_metrics(summary)
            details = _read_metric_details(eval_dir, sample_name)
            utility_exact_report = _load_utility_exact_report(eval_dir / sample_name)
            utility_feature_importance = _utility_importance_for_selected_evaluator(
                config=config,
                train_like_df=sample_df,
                test_df=test_df[column_order],
                schema_card=schema_card,
                dataset_context=dataset_context,
                seed=seed + 97,
                sample_size=0,
                utility_report=utility_exact_report,
            )
            report = {
                "sample_name": sample_name,
                "sample_path": str(sample_path),
                "sample_rows": int(len(sample_df)),
                "replace": bool(replace),
                "metrics": {
                    "shape": summary.get("shape"),
                    "trend": summary.get("trend"),
                    "dcr": summary.get("dcr"),
                    "dcr_privacy": summary.get("dcr_privacy"),
                    "utility_exact_metric": summary.get("utility_exact_metric"),
                    "utility_exact_raw_direction": summary.get("utility_exact_raw_direction"),
                    "utility_exact_overall": summary.get("utility_exact_overall"),
                    "utility_exact_overall_semantics": summary.get("utility_exact_overall_semantics"),
                    "utility_exact_regression_target_transform": summary.get("utility_exact_regression_target_transform"),
                    "utility_exact_regression_target_clip_min": summary.get("utility_exact_regression_target_clip_min"),
                    "utility_exact_regression_target_clip_max": summary.get("utility_exact_regression_target_clip_max"),
                    "metric_reward_score": summary.get("metric_reward_score"),
                    "metric_reward_score_direction": summary.get("metric_reward_score_direction"),
                    "metric_reward_score_semantics": summary.get("metric_reward_score_semantics"),
                },
                "shape_worst_columns": _weak_shape_from_details(details.get("shape_details", []), column_lookup),
                "trend_worst_pairs": _weak_trend_from_details(details.get("trend_details", []), column_lookup),
                "dcr_quantiles": details.get("dcr_quantiles", {"available": False}),
                "utility_feature_importance": utility_feature_importance,
                "utility_xgb_feature_importance": utility_feature_importance,
            }
            save_json(eval_dir / sample_name / "diagnostics.json", report)
            sample_profiles.append(_source_profile_sample_for_storage(report))
            repeat_reports.append({**report["metrics"], **report})
        profile = {
            "source_id": source_id,
            "source_path": str(info.path),
            "rows": int(len(source_df)),
            "raw_rows": int(info.rows),
            "source_validation": source_validation,
            "profile_mode": profile_mode,
            "profile_repeats": int(effective_repeats),
            "sample_rows": int(len(source_df) if config.mode == "single" and single_profile_cap is None else int(sample_rows)),
            "metrics_mean": _mean_metric_dict(repeat_reports),
            "metrics_std": _std_metric_dict(repeat_reports),
            "shape_worst_columns_mean": _aggregate_weak_columns(repeat_reports, "shape_worst_columns"),
            "trend_worst_pairs_mean": _aggregate_weak_columns(repeat_reports, "trend_worst_pairs"),
            "dcr_quantiles_mean": _aggregate_dcr_quantiles(repeat_reports),
            "utility_xgb_feature_importance_mean": _aggregate_utility_importance(repeat_reports),
            "sample_profiles": sample_profiles,
            "semantic_summary": _source_semantic_summary(source_id),
        }
        prompt = _render_prompt(
            config,
            "v2_source_profile_summary_prompt.j2",
            {
                "dataset_brief": _dataset_brief_for_prompt(schema_card, dataset_context),
                "dcr_balance_semantics": DCR_PROMPT_SEMANTICS,
                "source_profile": _compact_source_profile_for_prompt(source_id, info, profile),
            },
        )
        payload = _call_llm_json(
            client=client,
            prompt=prompt,
            schema_name=f"{source_id}_source_profile_summary",
            trace_dir=trace_dir,
        )
        if isinstance(payload, dict) and str(payload.get("semantic_summary", "")).strip():
            profile["semantic_summary"] = str(payload.get("semantic_summary", ""))[:600]
            profile["semantic_strengths"] = list(payload.get("strengths", []) or [])[:5]
            profile["semantic_weaknesses"] = list(payload.get("weaknesses", []) or [])[:5]
            profile["best_use"] = payload.get("best_use")
        save_json(source_dir / "profile_manifest.json", profile)
        save_jsonl(source_dir / "sample_profiles.jsonl", sample_profiles)
        source_profiles[source_id] = profile
    output = {
        "cache_key": cache_key,
        "cache_key_payload": cache_key_payload,
        "profile_version": SOURCE_PROFILE_VERSION,
        "profile_mode": profile_mode,
        "sources": source_profiles,
    }
    save_json(cache_path, output)
    return output


def _mcts_dir(config: V2MCTSConfig) -> Path:
    return ensure_dir(Path(config.artifact_dir) / config.exp_name / "mcts_v2")


def _normalize_source_name(source_name: str) -> str:
    key = str(source_name).strip().lower()
    if key not in SOURCE_ALIASES:
        raise ValueError(f"Unsupported source={source_name}; expected great/smote/tabdiff/tabsyn")
    return SOURCE_ALIASES[key]


def _resolve_source_path(config: V2MCTSConfig, source_id: str) -> Path:
    source = _normalize_source_name(source_id)
    preferred = Path(config.sample_root) / source / config.dataset_name / "sample_0.csv"
    if preferred.exists():
        return preferred
    legacy = Path(config.sample_root) / source / config.dataset_name / "samples_0.csv"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Cannot find synthetic sample for source={source}: {preferred}")


def resolve_sources(config: V2MCTSConfig) -> dict[str, SourceInfo]:
    source_names = (
        (_normalize_source_name(config.single_source),)
        if config.mode == "single"
        else tuple(_normalize_source_name(source) for source in config.source_names)
    )
    sources: dict[str, SourceInfo] = {}
    for source_id in dict.fromkeys(source_names):
        path = _resolve_source_path(config, source_id)
        header = _normalize_source_frame(config.dataset_name, pd.read_csv(path, nrows=0))
        rows = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
        sources[source_id] = SourceInfo(
            source_id=source_id,
            path=path,
            rows=int(rows),
            columns=[str(column) for column in header.columns],
        )
    return sources


def _canonical_s_key(pool_units: list[dict[str, Any]]) -> str:
    normalized = [
        {"source_id": str(unit["source_id"]), "multiplier": int(unit["multiplier"]) if float(unit["multiplier"]).is_integer() else round(float(unit["multiplier"]), 6)}
        for unit in pool_units
    ]
    normalized.sort(key=lambda item: (item["source_id"], item["multiplier"]))
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _integer_pool_total(pool_multiplier: float) -> int:
    total = int(round(float(pool_multiplier)))
    if total <= 0 or not math.isclose(float(pool_multiplier), float(total), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("mixed source pool_multiplier must be a positive integer when integer multipliers are required")
    return total


def _allocate_integer_pool_units(
    weights_by_source: dict[str, float],
    *,
    sources: dict[str, SourceInfo],
    pool_multiplier: float,
    min_sources: int,
) -> list[dict[str, Any]] | None:
    total = _integer_pool_total(pool_multiplier)
    valid_items = [
        (source_id, float(weight))
        for source_id, weight in weights_by_source.items()
        if source_id in sources and float(weight) > 0
    ]
    valid_items.sort(key=lambda item: (-item[1], list(sources).index(item[0]) if item[0] in sources else 10**6, item[0]))
    if len(valid_items) < int(min_sources):
        return None
    if total < int(min_sources):
        return None
    selected = valid_items[: min(len(valid_items), total)]
    if len(selected) < int(min_sources):
        return None

    base_units = {source_id: 1 for source_id, _ in selected}
    remaining = total - len(selected)
    if remaining > 0:
        selected_weight_sum = sum(weight for _, weight in selected)
        raw_extras = [
            (source_id, (weight / selected_weight_sum) * remaining if selected_weight_sum > 0 else 0.0)
            for source_id, weight in selected
        ]
        extras = {source_id: int(math.floor(extra)) for source_id, extra in raw_extras}
        leftover = remaining - sum(extras.values())
        raw_extras.sort(key=lambda item: (-(item[1] - math.floor(item[1])), list(sources).index(item[0]), item[0]))
        for source_id, _ in raw_extras[:leftover]:
            extras[source_id] += 1
        for source_id, extra in extras.items():
            base_units[source_id] += int(extra)

    return [
        {"source_id": source_id, "multiplier": int(base_units[source_id])}
        for source_id in sources
        if source_id in base_units
    ]


def _balanced_integer_pool_units(source_ids: list[str], *, pool_multiplier: float, min_sources: int) -> list[dict[str, Any]]:
    total = _integer_pool_total(pool_multiplier)
    if len(source_ids) < int(min_sources) or total < int(min_sources):
        raise ValueError("mixed mode requires at least two sources and pool_multiplier >= 2")
    chosen = source_ids[: min(len(source_ids), total)]
    if len(chosen) < int(min_sources):
        raise ValueError("mixed mode requires at least two sources in every synthetic pool")
    units = {source_id: 1 for source_id in chosen}
    for idx in range(total - len(chosen)):
        units[chosen[idx % len(chosen)]] += 1
    return [{"source_id": source_id, "multiplier": int(units[source_id])} for source_id in chosen]


def _pool_source_count(pool_units: list[dict[str, Any]]) -> int:
    return len({str(unit.get("source_id")) for unit in pool_units if float(unit.get("multiplier", 0.0)) > 0})


def _candidate_source_count_targets(config: V2MCTSConfig, sources: dict[str, SourceInfo]) -> list[int]:
    if config.mode != "mixed":
        return [1]
    total = _integer_pool_total(config.pool_multiplier)
    max_count = min(len(sources), total)
    if max_count < 2:
        return []
    return list(range(2, max_count + 1))


def _fallback_s_unit_candidates(config: V2MCTSConfig, sources: dict[str, SourceInfo]) -> list[dict[str, Any]]:
    if config.mode == "single":
        return [
            {
                "pool_units": [{"source_id": _normalize_source_name(config.single_source), "multiplier": config.pool_multiplier}],
                "llm_score": 0.5,
                "reason": "fixed single-source mode",
            }
        ]
    source_ids = list(sources)
    outputs: list[dict[str, Any]] = []
    for source_count in reversed(_candidate_source_count_targets(config, sources)):
        for subset in combinations(source_ids, source_count):
            balanced_units = _balanced_integer_pool_units(list(subset), pool_multiplier=config.pool_multiplier, min_sources=source_count)
            outputs.append(
                {
                    "pool_units": balanced_units,
                    "llm_score": 0.6 if source_count > 2 else 0.54,
                    "reason": f"balanced {source_count}-source integer pool",
                }
            )
    if len(source_ids) >= 2:
        total = _integer_pool_total(config.pool_multiplier)
        heavy_weights = {source_ids[0]: float(max(1, total - 1)), source_ids[1]: 1.0}
        heavy = _allocate_integer_pool_units(
            heavy_weights,
            sources=sources,
            pool_multiplier=config.pool_multiplier,
            min_sources=2,
        )
        if heavy is not None:
            outputs.append({"pool_units": heavy, "llm_score": 0.55, "reason": "source-skewed integer exploration pool"})
        for offset in range(1, len(source_ids)):
            first = source_ids[offset % len(source_ids)]
            second = source_ids[(offset + 1) % len(source_ids)]
            rotated_weights = {first: float(max(1, total - 1)), second: 1.0}
            rotated = _allocate_integer_pool_units(
                rotated_weights,
                sources=sources,
                pool_multiplier=config.pool_multiplier,
                min_sources=2,
            )
            if rotated is not None:
                outputs.append(
                    {
                        "pool_units": rotated,
                        "llm_score": 0.52,
                        "reason": "rotated integer exploration pool",
                    }
                )
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in outputs:
        key = _canonical_s_key(item["pool_units"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _fallback_initial_s_units(config: V2MCTSConfig, sources: dict[str, SourceInfo]) -> list[dict[str, Any]]:
    return _fallback_s_unit_candidates(config, sources)[: max(1, int(config.initial_s_pool_count))]


def _source_select_prompt(
    *,
    config: V2MCTSConfig,
    sources: dict[str, SourceInfo],
    n: int,
    phase: str,
    source_profiles: dict[str, Any],
    real_utility_reference: dict[str, Any],
    s_nodes: dict[str, SNode] | None = None,
    theta_nodes: dict[str, ThetaNode] | None = None,
) -> str:
    source_brief = _source_profiles_for_prompt(sources=sources, source_profiles=source_profiles)
    search_summary = {"existing_s": []}
    if phase == "refine":
        search_summary["existing_s"] = _summarize_s_nodes_for_prompt(
            s_nodes=s_nodes or {},
            theta_nodes=theta_nodes or {},
        )
    template_name = "v2_init_select_syn_prompt.j2" if phase == "init" else "v2_refine_select_syn_prompt.j2"
    return _render_prompt(
        config,
        template_name,
        {
            "n_pools": int(n),
            "pool_multiplier": _integer_pool_total(config.pool_multiplier),
            "dcr_balance_semantics": DCR_PROMPT_SEMANTICS,
            "source_models": source_brief,
            "real_utility_reference": _real_utility_for_prompt(real_utility_reference),
            "search_summary": search_summary,
        },
    )


def _source_profiles_for_prompt(
    *,
    sources: dict[str, SourceInfo],
    source_profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    profiles = source_profiles.get("sources", {}) if isinstance(source_profiles.get("sources"), dict) else {}
    brief: list[dict[str, Any]] = []
    for source_id, info in sources.items():
        item = profiles.get(source_id, {}) if isinstance(profiles.get(source_id), dict) else {}
        brief.append(_compact_source_profile_for_prompt(source_id, info, item))
    return brief


def _single_source_profiles_placeholder(config: V2MCTSConfig, sources: dict[str, SourceInfo]) -> dict[str, Any]:
    source_id = _normalize_source_name(config.single_source)
    info = sources[source_id]
    return {
        "profile_version": "single_source_profile_skipped_v1",
        "profile_mode": "single_fixed_source_no_source_selection",
        "sources": {},
        "fixed_source": {
            "source_id": source_id,
            "source_path": str(info.path),
            "rows": int(info.rows),
            "reason": "single mode fixes the synthetic source and searches theta only",
        },
    }


def _real_utility_for_prompt(real_utility_reference: dict[str, Any]) -> dict[str, Any]:
    utility = real_utility_reference.get("utility_feature_importance", {})
    output: dict[str, Any] = {
        "dataset": real_utility_reference.get("dataset"),
        "train_rows": real_utility_reference.get("train_rows"),
        "test_rows": real_utility_reference.get("test_rows"),
        "utility_feature_importance": _compact_utility_importance_for_prompt(
            utility,
            limit=UTILITY_IMPORTANCE_PROMPT_LIMIT,
        ),
        "target_summary": _compact_target_summary(real_utility_reference.get("target_summary", {})),
    }
    semantic_summary = _short_text(real_utility_reference.get("semantic_summary"), 260)
    if semantic_summary:
        output["semantic_summary"] = semantic_summary
    return output


def _search_scores_summary(search_objectives: dict[str, Any] | None) -> dict[str, Any]:
    search = search_objectives if isinstance(search_objectives, dict) else {}
    return {
        key: _prompt_number(search.get(key))
        for key in ("F_1D_theta", "F_2D_theta", "P_theta", "P_theta_raw", "U_proxy_theta")
        if key in search
    }


def _metrics_4d_reward_summary(audit_metrics: dict[str, Any] | None) -> dict[str, Any]:
    audit = audit_metrics if isinstance(audit_metrics, dict) else {}
    output = {
        key: _prompt_number(audit.get(key))
        for key in ("shape_global", "trend_global", "utility_exact", "metric_reward")
        if key in audit
    }
    for key in (
        "utility_exact_metric",
        "utility_exact_raw_direction",
        "utility_exact_overall_semantics",
        "metric_reward_score_direction",
        "metric_reward_score_semantics",
    ):
        if audit.get(key) is not None:
            output[key] = audit.get(key)
    if output.get("utility_exact_metric") is not None and output.get("utility_exact_raw_direction") is None:
        output["utility_exact_raw_direction"] = utility_metric_direction(output.get("utility_exact_metric"))
    if "utility_exact" in output:
        output.setdefault("utility_exact_overall_semantics", "raw_metric")
    if "metric_reward" in output:
        output.setdefault("metric_reward_score_direction", "higher_better")
        output.setdefault("metric_reward_score_semantics", "normalized_reward")
    if "dcr" in audit or "dcr_privacy" in audit:
        output["dcr_privacy_reward"] = _dcr_privacy_reward_for_prompt(audit.get("dcr"), audit.get("dcr_privacy"))
    return output


def _metrics_4d_reward_for_prompt(audit_metrics: dict[str, Any] | None) -> dict[str, Any]:
    audit = audit_metrics if isinstance(audit_metrics, dict) else {}
    output: dict[str, Any] = {}
    for key in ("shape_global", "trend_global", "utility_exact"):
        if key in audit:
            output[key] = _prompt_number(audit.get(key))
    metric = audit.get("utility_exact_metric")
    if metric is not None:
        output["utility_metric"] = metric
        output["utility_metric_direction"] = utility_metric_direction(metric)
    target_scale = audit.get("utility_exact_regression_target_transform")
    if target_scale is not None:
        output["utility_target_scale"] = target_scale
    if "dcr" in audit or "dcr_privacy" in audit:
        output["dcr_balance"] = _dcr_balance_for_prompt(audit.get("dcr"), audit.get("dcr_privacy"))
    if "metric_reward" in audit:
        output["metric_reward"] = _prompt_number(audit.get("metric_reward"))
        output["metric_reward_direction"] = "higher_better"
    return output


def _compact_feedback_for_prompt(feedback: dict[str, Any] | None, *, limit: int = 2) -> dict[str, Any]:
    if not isinstance(feedback, dict):
        return {}
    diagnostics = feedback.get("diagnostics", {}) if isinstance(feedback.get("diagnostics"), dict) else {}
    privacy = feedback.get("privacy_summary", {}) if isinstance(feedback.get("privacy_summary"), dict) else {}
    utility = feedback.get("utility_summary", {}) if isinstance(feedback.get("utility_summary"), dict) else {}
    dcr_summary = {
        key: _prompt_number(privacy.get(key))
        for key in ("search_privacy", "search_privacy_raw")
        if key in privacy
    }
    dcr_balance = _dcr_balance_for_prompt(privacy.get("dcr"), privacy.get("dcr_privacy"))
    if dcr_balance.get("available"):
        dcr_summary["balance"] = dcr_balance
    utility_summary = {
        key: _prompt_number(utility.get(key))
        for key in ("search_utility_proxy", "utility_exact", "utility_exact_available")
        if key in utility
    }
    metric = utility.get("utility_exact_metric")
    if metric is not None:
        utility_summary["utility_metric"] = metric
        utility_summary["utility_metric_direction"] = utility_metric_direction(metric)
    target_scale = utility.get("utility_exact_regression_target_transform")
    if target_scale is not None:
        utility_summary["utility_target_scale"] = target_scale
    return {
        "shape_bad_columns": _compact_weak_columns_for_prompt(feedback.get("shape_weak_columns", []), limit=limit),
        "trend_bad_pairs": _compact_weak_pairs_for_prompt(feedback.get("trend_weak_pairs", []), limit=limit),
        "dcr_summary": dcr_summary,
        "dcr_quantiles": _compact_dcr_for_prompt(diagnostics.get("dcr_quantiles", {"available": False})),
        "utility_summary": utility_summary,
        "utility_top": _compact_utility_importance_for_prompt(
            diagnostics.get("utility_xgb_feature_importance", {}),
            limit=UTILITY_IMPORTANCE_PROMPT_LIMIT,
        ),
    }


def _compact_theta_node_for_prompt(
    node: ThetaNode,
    *,
    include_actions: bool = True,
    include_feedback: bool = True,
    include_source_context: bool = True,
) -> dict[str, Any]:
    output = {
        "node_id": node.node_id,
        "theta": theta_to_dict(node.theta),
        "metrics_4d_reward": _metrics_4d_reward_for_prompt(node.audit_metrics),
        "exact_reward": _prompt_number(node.exact_reward) if node.exact_reward_available else None,
        "exact_reward_available": bool(node.exact_reward_available),
        "search_reward": _prompt_number(node.search_reward) if node.search_reward_available else None,
        "search_reward_available": bool(node.search_reward_available),
        "reward_type": node.reward_type,
        "semantic_summary": _short_text(node.feedback.get("llm_semantic_summary") or _fallback_node_summary(node), 260),
        "reason": _short_text(node.reason, 180),
    }
    if include_source_context:
        output["s_id"] = node.s_id
        output["source_contribution"] = _compact_source_contribution_for_prompt(_load_source_contribution_summary(node.rollout_dir))
    if include_actions and node.actions:
        output["actions"] = [
            action
            for action in (_compact_action_dict(item) for item in list(node.actions or [])[:6])
            if action
        ]
    if include_feedback:
        output["diagnostics"] = _compact_feedback_for_prompt(node.feedback)
    return output


def _diagnosis_theta_node_for_prompt(node: ThetaNode, *, include_source_context: bool = True) -> dict[str, Any]:
    feedback = _compact_feedback_for_prompt(node.feedback, limit=1)
    output = {
        "node_id": node.node_id,
        "theta": theta_to_dict(node.theta),
        "actions": [
            action
            for action in (_compact_action_dict(item) for item in list(node.actions or [])[:3])
            if action
        ],
        "metrics_4d_reward": _metrics_4d_reward_for_prompt(node.audit_metrics),
        "exact_reward": _prompt_number(node.exact_reward) if node.exact_reward_available else None,
        "search_reward": _prompt_number(node.search_reward) if node.search_reward_available else None,
        "reward_type": node.reward_type,
        "weakness": {
            "shape": feedback.get("shape_bad_columns", []),
            "trend": feedback.get("trend_bad_pairs", []),
            "dcr_distance_diagnostics": feedback.get("dcr_quantiles", {"available": False}),
            "utility_top": (
                feedback.get("utility_top", {}).get("top_features", [])
                if isinstance(feedback.get("utility_top"), dict)
                else feedback.get("utility_top", [])
            )[:2],
        },
        "semantic_summary": _short_text(node.feedback.get("llm_semantic_summary") or _fallback_node_summary(node), 180),
    }
    if include_source_context:
        output["s_id"] = node.s_id
        output["source_contribution"] = _compact_source_contribution_for_prompt(_load_source_contribution_summary(node.rollout_dir))
    return output


def _diagnosis_reference_nodes_for_prompt(
    *,
    theta_nodes: dict[str, ThetaNode] | None,
    exclude_node_ids: set[str],
    s_id: str,
    include_source_context: bool,
) -> list[dict[str, Any]]:
    if not theta_nodes:
        return []
    candidates = [
        node
        for node in theta_nodes.values()
        if node.s_id == s_id and node.node_id not in exclude_node_ids and node.status == "success"
    ]
    if not candidates:
        return []
    selected: dict[str, tuple[ThetaNode, list[str]]] = {}

    reward_candidates = [
        node
        for node in candidates
        if node.reward_available or node.search_reward_available or node.exact_reward_available
    ]
    if reward_candidates:
        best_reward = max(reward_candidates, key=lambda node: (_node_selection_score(node), node.node_id))
        selected[best_reward.node_id] = (best_reward, ["best_reward"])

    best_llm = max(candidates, key=lambda node: (float(node.llm_score), _node_selection_score(node), node.node_id))
    if best_llm.node_id in selected:
        selected[best_llm.node_id][1].append("best_llm_score")
    else:
        selected[best_llm.node_id] = (best_llm, ["best_llm_score"])

    output: list[dict[str, Any]] = []
    for node, roles in selected.values():
        item = _diagnosis_theta_node_for_prompt(node, include_source_context=include_source_context)
        item["roles"] = roles
        item["role"] = roles[0]
        item["reference_reason"] = " and ".join(roles)
        output.append(item)
    return output[:2]


def _top_unique_transfer_proposals(theta_nodes: dict[str, ThetaNode], *, n: int) -> list[StrategyProposal]:
    ranked = sorted(
        [node for node in theta_nodes.values() if node.status == "success"],
        key=lambda node: (_node_selection_score(node), node.node_id),
        reverse=True,
    )
    proposals: list[StrategyProposal] = []
    seen: set[str] = set()
    for node in ranked:
        key = canonical_key(node.theta)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(
            StrategyProposal(
                theta=node.theta,
                actions=[],
                prior_score=node.llm_score,
                reason=f"transfer_existing_theta_from_{node.node_id}",
                action_validation={
                    "locked_copy": True,
                    "transfer_source_node_id": node.node_id,
                    "transfer_source_theta_id": node.theta_id,
                    "transfer_source_score": _prompt_number(_node_selection_score(node)),
                },
            )
        )
        if len(proposals) >= int(n):
            break
    return proposals


def _refine_reference_nodes_for_prompt(
    *,
    theta_nodes: dict[str, ThetaNode] | None,
    parent: ThetaNode,
    include_source_context: bool,
    limit: int = 4,
    sibling_limit: int = 3,
) -> list[dict[str, Any]]:
    if not theta_nodes:
        return []
    candidates = [
        node
        for node in theta_nodes.values()
        if node.s_id == parent.s_id and node.node_id != parent.node_id and node.status == "success"
    ]
    if not candidates:
        return []
    selected: dict[str, tuple[ThetaNode, list[str]]] = {}

    def add_node(node: ThetaNode | None, role: str) -> None:
        if node is None or node.node_id == parent.node_id:
            return
        if node.node_id in selected:
            if role not in selected[node.node_id][1]:
                selected[node.node_id][1].append(role)
        else:
            selected[node.node_id] = (node, [role])

    sibling_nodes = [
        node
        for node in candidates
        if node.parent_node_id == parent.parent_node_id
    ]
    sibling_nodes = sorted(
        sibling_nodes,
        key=lambda node: (_node_selection_score(node), float(node.llm_score), node.node_id),
        reverse=True,
    )
    for node in sibling_nodes[: max(0, int(sibling_limit))]:
        add_node(node, "sibling")

    reward_candidates = [
        node
        for node in candidates
        if node.reward_available or node.search_reward_available or node.exact_reward_available
    ]
    ranked_existing = sorted(
        reward_candidates or candidates,
        key=lambda node: (_node_selection_score(node), float(node.llm_score), node.node_id),
        reverse=True,
    )
    for idx, node in enumerate(ranked_existing):
        role = "best_reward" if idx == 0 else "next_best_reward"
        add_node(node, role)
        if len(selected) >= int(limit):
            break

    output: list[dict[str, Any]] = []
    for node, roles in selected.values():
        item = _compact_theta_node_for_prompt(
            node,
            include_actions=True,
            include_feedback=False,
            include_source_context=include_source_context,
        )
        item["roles"] = roles
        item["role"] = roles[0]
        item["reference_reason"] = " and ".join(roles)
        item["llm_score"] = _prompt_number(node.llm_score)
        output.append(item)
        if len(output) >= int(limit):
            break
    return output


def _best_node_for_s(
    theta_nodes: dict[str, ThetaNode] | None,
    *,
    s_id: str,
    exclude_node_ids: set[str] | None = None,
) -> ThetaNode | None:
    if not theta_nodes:
        return None
    excluded = exclude_node_ids or set()
    candidates = [
        node
        for node in theta_nodes.values()
        if node.s_id == s_id and node.node_id not in excluded and node.status == "success"
    ]
    if not candidates:
        return None
    return max(candidates, key=_node_selection_score)


def _aggregate_source_contribution(nodes: list[ThetaNode]) -> dict[str, Any]:
    by_source: dict[str, list[float]] = {}
    for node in nodes:
        summary = _load_source_contribution_summary(node.rollout_dir)
        if not isinstance(summary, dict):
            continue
        for item in summary.get("sources", []) or []:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id"))
            by_source.setdefault(source_id, []).append(float(item.get("source_selected_fraction") or 0.0))
    return {
        source_id: {
            "selected_fraction_mean": _prompt_number(float(np.mean(values))),
            "selected_fraction_var": _prompt_number(float(np.var(values))),
        }
        for source_id, values in sorted(by_source.items())
        if values
    }


def _aggregate_weak_feedback(nodes: list[ThetaNode], *, limit: int = 2) -> dict[str, Any]:
    shape_scores: dict[str, list[float]] = {}
    trend_scores: dict[str, list[float]] = {}
    dcr_values: list[float] = []
    dcr_privacy_values: list[float] = []
    dcr_error_values: list[float] = []
    for node in nodes:
        for item in node.feedback.get("shape_weak_columns", []) or []:
            if isinstance(item, dict) and item.get("column") is not None and item.get("score") is not None:
                shape_scores.setdefault(str(item["column"]), []).append(float(item["score"]))
        for item in node.feedback.get("trend_weak_pairs", []) or []:
            if isinstance(item, dict) and item.get("left") is not None and item.get("right") is not None and item.get("score") is not None:
                key = f"{item['left']}|{item['right']}"
                trend_scores.setdefault(key, []).append(float(item["score"]))
        if node.audit_metrics.get("dcr") is not None:
            raw_dcr = float(node.audit_metrics["dcr"])
            dcr_values.append(raw_dcr)
            dcr_error_values.append(abs(raw_dcr - 0.5))
        if node.audit_metrics.get("dcr_privacy") is not None:
            dcr_privacy_values.append(float(node.audit_metrics["dcr_privacy"]))
    shape = [
        {"column": column, "mean_score": _prompt_number(float(np.mean(values))), "count": len(values)}
        for column, values in sorted(shape_scores.items(), key=lambda item: float(np.mean(item[1])))[:limit]
    ]
    trend = []
    for pair_key, values in sorted(trend_scores.items(), key=lambda item: float(np.mean(item[1])))[:limit]:
        left, right = pair_key.split("|", 1)
        trend.append({"left": left, "right": right, "mean_score": _prompt_number(float(np.mean(values))), "count": len(values)})
    return {
        "shape_bad_columns": shape,
        "trend_bad_pairs": trend,
        "privacy_distribution": {
            "raw_dcr_real_closer_rate_distribution": (
                _quantile_dict(np.asarray(dcr_values, dtype=float)) if dcr_values else {"available": False}
            ),
            "dcr_privacy_reward_distribution": (
                _quantile_dict(np.asarray(dcr_privacy_values, dtype=float)) if dcr_privacy_values else {"available": False}
            ),
            "balance_error_abs_distribution": (
                _quantile_dict(np.asarray(dcr_error_values, dtype=float)) if dcr_error_values else {"available": False}
            ),
        },
    }


def _summarize_s_nodes_for_prompt(
    *,
    s_nodes: dict[str, SNode],
    theta_nodes: dict[str, ThetaNode],
    limit: int = 3,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for s_node in sorted(s_nodes.values(), key=lambda item: float(item.best_reward), reverse=True)[:limit]:
        nodes = [theta_nodes[node_id] for node_id in s_node.theta_node_ids if node_id in theta_nodes and theta_nodes[node_id].status == "success"]
        rewards = [float(node.exact_reward) for node in nodes if node.exact_reward_available]
        search_rewards = [float(node.search_reward) for node in nodes if node.search_reward_available]
        best = max(nodes, key=_node_selection_score) if nodes else None
        metric_keys = {
            "shape_global": "shape_global",
            "trend_global": "trend_global",
            "utility_exact": "utility_exact",
            "dcr_privacy": "dcr_balance_reward",
            "metric_reward": "metric_reward",
        }
        means = {}
        for metric_key, prompt_key in metric_keys.items():
            values = [float(node.audit_metrics[metric_key]) for node in nodes if node.audit_metrics.get(metric_key) is not None]
            if values:
                means[prompt_key] = _prompt_number(float(np.mean(values)))
        summaries.append(
            {
                "s_id": s_node.s_id,
                "strategy": s_node.pool_units,
                "theta_count": len(nodes),
                "mean_metrics_4d_reward": means,
                "reward_q50": _prompt_number(float(np.quantile(np.asarray(rewards, dtype=float), 0.5))) if rewards else None,
                "search_reward_q50": _prompt_number(float(np.quantile(np.asarray(search_rewards, dtype=float), 0.5))) if search_rewards else None,
                "aggregate_weakness": _aggregate_weak_feedback(nodes),
                "best_theta": None if best is None else _compact_theta_node_for_prompt(best, include_actions=False, include_feedback=False),
                "source_contribution_mean_var": _aggregate_source_contribution(nodes),
                "semantic_summary": _short_text(s_node.semantic_summary or s_node.reason, 220),
            }
        )
    return summaries


def _source_semantic_summary(source_id: str) -> str:
    summaries = {
        "great": "GAN-like synthetic table; useful for smooth shapes but can blur rare categories.",
        "smote": "Interpolation-heavy source; can preserve local neighborhoods but may hurt categorical realism.",
        "tabdiff": "Diffusion source; often strong density/trend base and useful for balanced repair.",
        "tabsyn": "VAE/diffusion-style source; useful diversity and may preserve minority utility patterns.",
    }
    return summaries.get(source_id, "synthetic source")


def _llm_call_purpose(schema_name: str) -> tuple[str, str]:
    if "real_utility_profile_summary" in schema_name:
        return "context", "Summarize real-data utility anchors and feature importance."
    if "source_profile_summary" in schema_name:
        return "context", "Summarize one synthetic source profile from complete-eval diagnostics."
    if "init_select_syn" in schema_name:
        return "select_syn", "Choose initial synthetic source pool strategy."
    if "refine_select_syn" in schema_name:
        return "select_syn", "Refine synthetic source pool strategy after search feedback."
    if "init_node_diagnosis" in schema_name:
        return "diagnosis", "Diagnose an initial sibling theta batch and assign future-search LLM scores."
    if "refine_node_diagnosis" in schema_name:
        return "diagnosis", "Diagnose a refined sibling theta batch and assign future-search LLM scores."
    if "init_node" in schema_name:
        return "theta", "Generate initial theta proposals for the current search context."
    if "refine_node" in schema_name:
        return "theta", "Generate refined theta proposals for the current search context."
    if "diagnosis" in schema_name:
        return "diagnosis", "Diagnose a sibling theta batch and assign future-search LLM scores."
    return "other", "LLM JSON call."


def _mock_fallback_description(schema_name: str) -> str:
    if "real_utility_profile_summary" in schema_name:
        return "Fallback keeps the computed configured/static utility feature ranking and omits empty semantic anchors."
    if "source_profile_summary" in schema_name:
        return "Fallback uses the built-in source semantic summary while retaining repeated profile metrics, DCR balance/distance diagnostics, and utility ranking."
    if "init_select_syn" in schema_name:
        return "Fallback creates deterministic initial S-pool candidates from configured sources."
    if "refine_select_syn" in schema_name:
        return "Fallback creates the next deterministic non-duplicate S-pool candidate after mixed-source stagnation."
    if "init_node_diagnosis" in schema_name or "refine_node_diagnosis" in schema_name:
        return "Fallback writes readable node semantic summaries from evaluated objectives and keeps existing prior scores unchanged."
    if "init_node" in schema_name:
        return "Fallback generates deterministic target-aware initial theta proposals from dataset guidance."
    if "refine_node" in schema_name:
        return "Fallback generates deterministic target-aware child theta proposals by editing the selected leaf node."
    return "Fallback returns no parsed JSON; the caller uses its deterministic local recovery path."


def _call_llm_json(
    *,
    client: LLMClient | None,
    prompt: str,
    schema_name: str,
    trace_dir: Path,
) -> dict[str, Any] | None:
    ensure_dir(trace_dir)
    readme = Path(trace_dir) / "README.md"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    "# LLM calls",
                    "",
                    "Each numbered directory is one LLM call. Directory names follow `000000_node-or-context_prompt-type`.",
                    "",
                    "- `prompt.md`: readable prompt text sent to the LLM, or the prompt that would be sent in mock mode.",
                    "- `response.md`: readable response/status for the same call. This file always exists for every `prompt.md`.",
                    "- `input_summary.json`: compact metadata explaining the call phase, purpose, and prompt size.",
                    "- `status.json`: machine-readable provider/status metadata.",
                    "- `response.parsed.json`: parsed JSON response when a real LLM call succeeds.",
                    "- `response.raw.txt`: raw model response when available.",
                    "",
                    "`manifest.jsonl` indexes every call with phase, status, prompt, and response paths.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    existing = [
        path
        for path in Path(trace_dir).iterdir()
        if path.is_dir() and len(path.name) >= 6 and path.name[:6].isdigit()
    ]
    call_idx = len(existing)
    safe_schema = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in schema_name)[:120]
    stem = f"{call_idx:06d}_{safe_schema}"
    call_dir = ensure_dir(Path(trace_dir) / stem)
    prompt_path = call_dir / "prompt.md"
    input_summary_path = call_dir / "input_summary.json"
    response_path = call_dir / "response.md"
    status_path = call_dir / "status.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    phase, purpose = _llm_call_purpose(schema_name)
    save_json(
        input_summary_path,
        {
            "schema_name": schema_name,
            "phase": phase,
            "purpose": purpose,
            "prompt_chars": int(len(prompt)),
            "prompt_lines": int(prompt.count("\n") + 1),
            "prompt_estimated_tokens": int(math.ceil(len(prompt) / 4)),
        },
    )
    manifest_record = {
        "call_id": f"{call_idx:06d}",
        "schema_name": schema_name,
        "phase": phase,
        "purpose": purpose,
        "call_dir": str(call_dir),
        "prompt_file": str(prompt_path),
        "input_summary_file": str(input_summary_path),
        "response_file": str(response_path),
        "provider": "mock" if client is None else "llm",
    }
    pending_status = {"available": False, "status": "pending", "reason": "llm_call_started"}
    save_json(status_path, pending_status)
    response_path.write_text(
        "\n".join(
            [
                "# LLM response",
                "",
                "Status: pending",
                "",
                "The prompt has been written and the LLM call has started.",
                "If this status remains, the process stopped before a final response was written.",
                "",
                f"Prompt type: `{schema_name}`",
                f"Phase: {phase}",
                f"Purpose: {purpose}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if client is None:
        status = {"available": False, "status": "mock", "reason": "mock_provider_no_llm_call"}
        mock_payload = {
            "available": False,
            "status": "mock",
            "schema_name": schema_name,
            "phase": phase,
            "purpose": purpose,
            "fallback_description": _mock_fallback_description(schema_name),
        }
        save_json(status_path, status)
        save_json(call_dir / "response.parsed.json", mock_payload)
        fallback_description = _mock_fallback_description(schema_name)
        response_path.write_text(
            "\n".join(
                [
                    "# LLM response",
                    "",
                    "Status: mock",
                    "",
                    "No real LLM call was made because the provider is `mock`.",
                    "",
                    f"Prompt type: `{schema_name}`",
                    f"Phase: {phase}",
                    f"Purpose: {purpose}",
                    "",
                    "Deterministic fallback:",
                    fallback_description,
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        _append_jsonl_record(Path(trace_dir) / "manifest.jsonl", {**manifest_record, **status})
        return None
    try:
        payload = client.complete_json(prompt, schema_name)
    except Exception as exc:
        status = {"available": False, "status": "error", "error": str(exc)}
        save_json(status_path, status)
        response_path.write_text(
            f"# LLM response\n\nStatus: error\n\nError:\n\n```text\n{exc}\n```\n",
            encoding="utf-8",
        )
        _append_jsonl_record(Path(trace_dir) / "manifest.jsonl", {**manifest_record, **status})
        return None
    call = getattr(client, "last_call", None) or {}
    save_json(call_dir / "response.parsed.json", payload)
    status = {"available": True, "status": "success"}
    save_json(status_path, status)
    response_path.write_text(
        "# LLM response\n\nStatus: success\n\nParsed JSON:\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n",
        encoding="utf-8",
    )
    _append_jsonl_record(Path(trace_dir) / "manifest.jsonl", {**manifest_record, **status})
    return payload


def _validate_pool_units(
    raw_units: Any,
    *,
    sources: dict[str, SourceInfo],
    pool_multiplier: float,
    require_integer: bool = False,
    min_sources: int = 1,
) -> list[dict[str, Any]] | None:
    if not isinstance(raw_units, list):
        return None
    weights_by_source: dict[str, float] = {}
    for item in raw_units:
        if not isinstance(item, dict):
            continue
        try:
            source_id = _normalize_source_name(str(item.get("source_id", "")))
            multiplier = float(item.get("multiplier", 0.0))
        except Exception:
            continue
        if source_id not in sources or multiplier <= 0:
            continue
        weights_by_source[source_id] = weights_by_source.get(source_id, 0.0) + multiplier
    if len(weights_by_source) < int(min_sources):
        return None
    if require_integer:
        total = _integer_pool_total(pool_multiplier)
        integer_units = {
            source_id: int(round(weight))
            for source_id, weight in weights_by_source.items()
            if math.isclose(float(weight), float(round(weight)), rel_tol=0.0, abs_tol=1e-6)
        }
        if len(integer_units) == len(weights_by_source) and sum(integer_units.values()) == total:
            return [
                {"source_id": source_id, "multiplier": int(integer_units[source_id])}
                for source_id in sources
                if source_id in integer_units
            ]
        return _allocate_integer_pool_units(
            weights_by_source,
            sources=sources,
            pool_multiplier=pool_multiplier,
            min_sources=min_sources,
        )
    units = [{"source_id": source_id, "multiplier": multiplier} for source_id, multiplier in weights_by_source.items()]
    if not units:
        return None
    total = sum(float(unit["multiplier"]) for unit in units)
    if total <= 0:
        return None
    scale = float(pool_multiplier) / total
    return [
        {"source_id": unit["source_id"], "multiplier": float(unit["multiplier"]) * scale}
        for unit in units
    ]


def _select_source_count_diverse_pool_candidates(
    candidates: list[dict[str, Any]],
    *,
    config: V2MCTSConfig,
    sources: dict[str, SourceInfo],
    n: int,
    phase: str,
    s_nodes: dict[str, SNode] | None,
) -> list[dict[str, Any]]:
    target_n = max(0, int(n))
    if target_n <= 0 or config.mode != "mixed":
        return candidates[:target_n]
    source_count_targets = _candidate_source_count_targets(config, sources)
    if not source_count_targets:
        return candidates[:target_n]
    existing_counts = {
        _pool_source_count(node.pool_units)
        for node in (s_nodes or {}).values()
    }
    if phase != "init":
        selected: list[dict[str, Any]] = []
        selected_keys: set[str] = set()
        missing_counts = [count for count in source_count_targets if count not in existing_counts]

        def take_first_with_source_count(source_count: int) -> None:
            if len(selected) >= target_n:
                return
            for item in candidates:
                key = _canonical_s_key(item["pool_units"])
                if key in selected_keys:
                    continue
                if _pool_source_count(item["pool_units"]) != int(source_count):
                    continue
                selected.append(item)
                selected_keys.add(key)
                return

        if target_n > 1 and 2 in source_count_targets:
            take_first_with_source_count(2)

        for source_count in missing_counts:
            if target_n > 1 and int(source_count) == 2:
                continue
            take_first_with_source_count(source_count)
            if len(selected) >= target_n:
                break

        for item in candidates:
            if len(selected) >= target_n:
                break
            key = _canonical_s_key(item["pool_units"])
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)
        return selected

    if phase == "init":
        preferred_counts = list(source_count_targets)
    else:
        preferred_counts = [count for count in source_count_targets if count not in existing_counts]
        if not preferred_counts:
            preferred_counts = list(source_count_targets)

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    def take_first_with_source_count(source_count: int) -> None:
        if len(selected) >= target_n:
            return
        for item in candidates:
            key = _canonical_s_key(item["pool_units"])
            if key in selected_keys:
                continue
            if _pool_source_count(item["pool_units"]) != int(source_count):
                continue
            selected.append(item)
            selected_keys.add(key)
            return

    for source_count in preferred_counts:
        take_first_with_source_count(source_count)
        if len(selected) >= target_n:
            break

    for item in candidates:
        if len(selected) >= target_n:
            break
        key = _canonical_s_key(item["pool_units"])
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
    return selected


def _should_stop_for_hard_no_improve(hard_no_improve: int, early_stop_stagnation_events: int) -> bool:
    threshold = int(early_stop_stagnation_events)
    return threshold >= 0 and int(hard_no_improve) > threshold


def select_s_pools(
    *,
    config: V2MCTSConfig,
    sources: dict[str, SourceInfo],
    client: LLMClient | None,
    n: int,
    phase: str,
    source_profiles: dict[str, Any],
    real_utility_reference: dict[str, Any],
    s_nodes: dict[str, SNode] | None,
    theta_nodes: dict[str, ThetaNode] | None,
    trace_dir: Path,
) -> list[dict[str, Any]]:
    if config.mode == "single":
        return _fallback_initial_s_units(config, sources)
    prompt = _source_select_prompt(
        config=config,
        sources=sources,
        n=n,
        phase=phase,
        source_profiles=source_profiles,
        real_utility_reference=real_utility_reference,
        s_nodes=s_nodes,
        theta_nodes=theta_nodes,
    )
    payload = _call_llm_json(client=client, prompt=prompt, schema_name=f"global_{phase}_select_syn", trace_dir=trace_dir)
    outputs: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        raw_items: list[Any] = []
        if isinstance(payload.get("syn_pools"), list):
            raw_items.extend(payload.get("syn_pools", []))
        if isinstance(payload.get("syn_pool"), dict):
            raw_items.append(payload.get("syn_pool"))
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            units = _validate_pool_units(
                item.get("pool_units"),
                sources=sources,
                pool_multiplier=config.pool_multiplier,
                require_integer=True,
                min_sources=2,
            )
            if units is None:
                continue
            outputs.append(
                {
                    "pool_units": units,
                    "llm_score": _clip01(item.get("llm_score", 0.5)),
                    "reason": str(item.get("reason", ""))[:240],
                    "family": str(item.get("family", ""))[:80],
                }
            )
    seen = {_canonical_s_key(node.pool_units) for node in (s_nodes or {}).values()}
    candidates: list[dict[str, Any]] = []
    for item in outputs + _fallback_s_unit_candidates(config, sources):
        key = _canonical_s_key(item["pool_units"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(item)
    return _select_source_count_diverse_pool_candidates(
        candidates,
        config=config,
        sources=sources,
        n=n,
        phase=phase,
        s_nodes=s_nodes,
    )


def build_s_pool(
    *,
    s_id: str,
    pool_units: list[dict[str, Any]],
    config: V2MCTSConfig,
    sources: dict[str, SourceInfo],
    train_rows: int,
    output_dir: Path,
    column_order: list[str],
    schema_card: dict[str, Any],
    stats_card: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    ensure_dir(output_dir)
    if config.mode == "single" and len(pool_units) == 1:
        source_id = _normalize_source_name(str(pool_units[0]["source_id"]))
        info = sources[source_id]
        source_df, source_validation = _load_valid_source_frame(
            dataset_name=config.dataset_name,
            source_id=source_id,
            path=info.path,
            column_order=column_order,
            schema_card=schema_card,
            stats_card=stats_card,
            validation_dir=output_dir / "source_validation" / source_id,
            save_records=config.save_validation_records,
        )
        source_indices = [int(idx) for idx in source_df.index.tolist()]
        source_df = source_df.reset_index(drop=True)
        synthetic_csv = output_dir / "synthetic_pool.csv"
        row_map_path = output_dir / "synthetic_pool_rows.jsonl"
        save_csv(synthetic_csv, source_df)
        save_jsonl(
            row_map_path,
            [
                {
                    "pool_row_id": int(idx),
                    "source_id": source_id,
                    "source_row_id": int(source_indices[idx]),
                    "draw_index": 0,
                    "draw_local_index": int(idx),
                    "sample_seed": None,
                    "with_replacement": False,
                    "sampling_mode": "single_full_source_no_random_sampling",
                }
                for idx in range(len(source_df))
            ],
        )
        report = {
            "s_id": s_id,
            "rows": int(len(source_df)),
            "target_rows": int(len(source_df)),
            "pool_units": pool_units,
            "source_counts": {source_id: int(len(source_df))},
            "sampling_mode": "single_full_source_no_random_sampling",
            "source_path": str(info.path),
            "source_validation": {source_id: source_validation},
        }
        save_json(output_dir / "synthetic_pool_manifest.json", report)
        return synthetic_csv, row_map_path, report

    frames: list[pd.DataFrame] = []
    row_map: list[dict[str, Any]] = []
    source_validations: dict[str, Any] = {}
    pool_row_id = 0
    for draw_index, unit in enumerate(pool_units):
        source_id = str(unit["source_id"])
        multiplier = int(unit["multiplier"])
        rows_to_draw = max(1, int(multiplier) * int(train_rows))
        info = sources[source_id]
        source_df, source_validation = _load_valid_source_frame(
            dataset_name=config.dataset_name,
            source_id=source_id,
            path=info.path,
            column_order=column_order,
            schema_card=schema_card,
            stats_card=stats_card,
            validation_dir=output_dir / "source_validation" / source_id,
            save_records=config.save_validation_records,
        )
        source_validations[source_id] = source_validation
        replace = rows_to_draw > len(source_df)
        seed = int(config.seed + 997 * (draw_index + 1) + sum(ord(ch) for ch in source_id))
        sampled = source_df.sample(n=rows_to_draw, replace=replace, random_state=seed)
        source_indices = list(sampled.index)
        sampled = sampled.reset_index(drop=True)
        frames.append(sampled)
        for local_idx, source_row_id in enumerate(source_indices):
            row_map.append(
                {
                    "pool_row_id": int(pool_row_id),
                    "source_id": source_id,
                    "source_row_id": int(source_row_id),
                    "draw_index": int(draw_index),
                    "draw_local_index": int(local_idx),
                    "sample_seed": int(seed),
                    "with_replacement": bool(replace),
                }
            )
            pool_row_id += 1
    pool_df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame(columns=column_order)
    target_multiplier = sum(int(unit["multiplier"]) for unit in pool_units)
    target_rows = max(1, int(target_multiplier) * int(train_rows))
    if len(pool_df) > target_rows:
        pool_df = pool_df.iloc[:target_rows].reset_index(drop=True)
        row_map = row_map[:target_rows]
    synthetic_csv = output_dir / "synthetic_pool.csv"
    row_map_path = output_dir / "synthetic_pool_rows.jsonl"
    save_csv(synthetic_csv, pool_df)
    save_jsonl(row_map_path, row_map)
    report = {
        "s_id": s_id,
        "rows": int(len(pool_df)),
        "target_rows": int(target_rows),
        "pool_units": pool_units,
        "source_counts": {
            source_id: int(sum(1 for row in row_map if row["source_id"] == source_id))
            for source_id in sorted({row["source_id"] for row in row_map})
        },
        "source_validation": source_validations,
    }
    save_json(output_dir / "synthetic_pool_manifest.json", report)
    return synthetic_csv, row_map_path, report


def _clip01(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = 0.5
    return max(0.0, min(1.0, parsed))


def _sanitize_llm_dcr_text(text: Any, audit_metrics: dict[str, Any] | None = None) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    if "dcr" not in cleaned.lower():
        return cleaned
    metrics = audit_metrics if isinstance(audit_metrics, dict) else {}
    raw = _finite_float(metrics.get("dcr"))
    privacy = _finite_float(metrics.get("dcr_privacy"))
    if raw is not None:
        replacement = f"raw DCR real_closer_rate {_prompt_number(raw)} (target 0.5"
        if privacy is not None:
            replacement += f"; DCR balance privacy_reward {_prompt_number(privacy)}"
        replacement += ")"
        cleaned = re.sub(r"\bDCR\s*\(\s*[0-9.]+\s*\)", replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\blow\s+DCR\b", "DCR imbalance", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhigh\s+DCR\b", "DCR imbalance", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\bDCR privacy\s+(?:near|close to)\s+0\.5\b",
        "raw DCR near 0.5 and DCR balance privacy_reward high",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def _schema_feature_columns(schema_card: dict[str, Any]) -> list[str]:
    return [
        column
        for column in schema_card.get("column_order", [])
        if not bool(schema_card.get("columns", {}).get(column, {}).get("is_target", False))
    ]


def _source_profile_mix_for_prompt(
    *,
    pool_units: list[dict[str, Any]],
    source_profiles: dict[str, Any],
    include_dcr_quantiles: bool = True,
) -> list[dict[str, Any]]:
    profiles = source_profiles.get("sources", {}) if isinstance(source_profiles.get("sources"), dict) else {}
    output: list[dict[str, Any]] = []
    for unit in pool_units:
        source_id = str(unit.get("source_id"))
        profile = profiles.get(source_id, {}) if isinstance(profiles.get(source_id), dict) else {}
        item = {
            "source_id": source_id,
            "multiplier": _prompt_number(unit.get("multiplier")),
            "metrics_mean": _compact_metrics_for_prompt(profile.get("metrics_mean", {})),
            "shape_worst": _compact_weak_columns_for_prompt(profile.get("shape_worst_columns_mean", []), limit=1),
            "trend_worst": _compact_weak_pairs_for_prompt(profile.get("trend_worst_pairs_mean", []), limit=1),
            "utility_top": _compact_utility_importance_for_prompt(
                profile.get("utility_xgb_feature_importance_mean", []),
                limit=UTILITY_IMPORTANCE_POOL_LIMIT,
            ),
            "summary": _short_text(profile.get("semantic_summary") or _source_semantic_summary(source_id), 120),
        }
        if include_dcr_quantiles:
            item["dcr_quantiles"] = _compact_dcr_for_prompt(profile.get("dcr_quantiles_mean", {}))
        output.append(item)
    return output


def _s_context_for_prompt(
    *,
    s_node: SNode,
    source_profiles: dict[str, Any],
    include_source_profile_mix: bool = True,
    include_dcr_quantiles: bool = True,
) -> dict[str, Any]:
    output = {
        "s_id": s_node.s_id,
        "pool_units": s_node.pool_units,
        "reason": _short_text(s_node.reason, 160),
        "semantic_summary": _short_text(s_node.semantic_summary, 220),
    }
    if include_source_profile_mix:
        profiles = source_profiles.get("sources", {}) if isinstance(source_profiles, dict) else {}
        if profiles:
            output["source_profile_mix"] = _source_profile_mix_for_prompt(
                pool_units=s_node.pool_units,
                source_profiles=source_profiles,
                include_dcr_quantiles=include_dcr_quantiles,
            )
    return output


def _theta_prompt(
    *,
    config: V2MCTSConfig,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
    real_utility_reference: dict[str, Any],
    s_node: SNode,
    source_profiles: dict[str, Any],
    n: int,
    parent: ThetaNode | None,
    archive: list[dict[str, Any]],
    theta_nodes: dict[str, ThetaNode] | None = None,
) -> str:
    dataset_brief = _dataset_brief_for_prompt(schema_card, dataset_context)
    seed_theta_examples = _seed_theta_examples_for_prompt(
        schema_card=schema_card,
        dataset_context=dataset_context,
        limit=4,
    )
    s_context = (
        _s_context_for_prompt(
            s_node=s_node,
            source_profiles=source_profiles,
            include_source_profile_mix=True,
            include_dcr_quantiles=False,
        )
        if config.mode == "mixed"
        else None
    )
    if parent is None:
        return _render_prompt(
            config,
            "v2_init_node_prompt.j2",
            {
                "n_theta": int(n),
                "dataset_brief": dataset_brief,
                "dcr_balance_semantics": DCR_PROMPT_SEMANTICS,
                "real_utility_reference": _real_utility_for_prompt(real_utility_reference),
                "s_context": s_context,
                "seed_theta_examples": seed_theta_examples,
            },
        )

    if parent.s_id != s_node.s_id:
        raise ValueError(f"refine_node parent {parent.node_id} belongs to {parent.s_id}, not current S {s_node.s_id}")
    include_source_context = config.mode == "mixed"
    current_node = _compact_theta_node_for_prompt(
        parent,
        include_actions=True,
        include_feedback=True,
        include_source_context=include_source_context,
    )
    reference_nodes = _refine_reference_nodes_for_prompt(
        theta_nodes=theta_nodes,
        parent=parent,
        include_source_context=include_source_context,
        limit=4,
    )
    return _render_prompt(
        config,
        "v2_refine_node_prompt.j2",
        {
            "n_theta": int(n),
            "dataset_brief": dataset_brief,
            "dcr_balance_semantics": DCR_PROMPT_SEMANTICS,
            "real_utility_reference": _real_utility_for_prompt(real_utility_reference),
            "s_context": s_context,
            "current_node": current_node,
            "reference_nodes": reference_nodes,
        },
    )


def _proposal_from_payload(
    item: dict[str, Any],
    *,
    schema_card: dict[str, Any],
    seed: int,
    parent: ThetaNode | None = None,
) -> StrategyProposal | None:
    if not isinstance(item, dict) or not isinstance(item.get("theta"), dict):
        return None
    theta = repair_theta_target_inclusive(item["theta"], schema_card, random.Random(seed))
    if not validate_theta_target_inclusive(theta, schema_card).ok:
        return None
    actions = []
    for action in item.get("actions", []):
        if isinstance(action, dict):
            actions.append(
                ThetaAction(
                    type=str(action.get("type", "")),
                    column=action.get("column"),
                    old=action.get("old"),
                    new=action.get("new"),
                )
            )
    action_validation: dict[str, Any] = {"theta_target_mode": "target_inclusive"}
    if parent is not None:
        actions_match = bool(actions) and canonical_key(
            _apply_actions_target_inclusive(parent.theta, actions, schema_card)
        ) == canonical_key(theta)
        action_validation["llm_actions_match_theta"] = actions_match
        if not actions_match:
            derived_actions = _derive_actions_between_thetas(parent.theta, theta, schema_card)
            if derived_actions or canonical_key(parent.theta) == canonical_key(theta):
                actions = derived_actions
                action_validation["actions_repaired_from_theta"] = True
            else:
                action_validation["action_derivation_failed"] = True
    return StrategyProposal(
        theta=theta,
        actions=actions,
        prior_score=_clip01(item.get("prior_score", 0.5)),
        reason=str(item.get("reason", ""))[:240],
        action_validation=action_validation,
    )


def _deterministic_theta_candidates(
    *,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
    n: int,
    seed: int,
    parent: ThetaNode | None = None,
) -> list[StrategyProposal]:
    rng = random.Random(seed)
    features = _schema_feature_columns(schema_card)
    target = schema_card["target_column"]
    guidance = dataset_context.get("theta_guidance", {})
    seeds = [
        item.get("theta")
        for item in _seed_theta_examples_for_prompt(
            schema_card=schema_card,
            dataset_context=dataset_context,
            limit=4,
        )
        if isinstance(item, dict)
    ]
    proposals: list[StrategyProposal] = []
    base_payloads: list[dict[str, Any]] = []
    if parent is None:
        base_payloads.extend([payload for payload in seeds if isinstance(payload, dict)])
    else:
        base = theta_to_dict(parent.theta)
        priorities = []
        for key in ("shape_priority", "trend_priority", "privacy_priority", "utility_priority"):
            priorities.extend(guidance.get(key, []))
        priorities = [column for column in dict.fromkeys(priorities) if column in features]
        for idx in range(max(1, int(n))):
            payload = {field: list(base[field]) for field in ("col_1ds", "col_2ds", "col_ps")}
            payload["col_u"] = base["col_u"]
            field = rng.choice(["col_1ds", "col_2ds", "col_ps", "col_u"])
            if field == "col_u":
                payload["col_u"] = rng.choice(priorities or features)
            else:
                choices = (priorities + features) if field == "col_ps" else ([target] + priorities + features)
                current = [column for column in payload[field] if column != target]
                if current and rng.random() < 0.65:
                    old = rng.choice(current)
                    new = rng.choice([column for column in choices if column not in payload[field]] or choices)
                    payload[field] = [new if column == old else column for column in payload[field]]
                else:
                    new = rng.choice([column for column in choices if column not in payload[field]] or choices)
                    payload[field].append(new)
            base_payloads.append(payload)
    while len(base_payloads) < int(n):
        priorities = []
        for key in ("shape_priority", "trend_priority", "privacy_priority", "utility_priority"):
            priorities.extend(guidance.get(key, []))
        ordered = [column for column in dict.fromkeys([target, *priorities, *features]) if column in set([target, *features])]
        privacy_ordered = [column for column in dict.fromkeys([*priorities, *features]) if column in set(features)]
        rng.shuffle(ordered)
        rng.shuffle(privacy_ordered)
        base_payloads.append(
            {
                "col_1ds": ordered[:7],
                "col_2ds": ordered[:6],
                "col_ps": privacy_ordered[:10],
                "col_u": rng.choice(features),
            }
        )
    seen: set[str] = set()
    for idx, payload in enumerate(base_payloads):
        theta = repair_theta_target_inclusive(payload, schema_card, random.Random(seed + idx))
        key = canonical_key(theta)
        if key in seen:
            continue
        actions: list[ThetaAction] = []
        action_validation: dict[str, Any] = {"theta_target_mode": "target_inclusive", "fallback": True}
        if parent is not None and canonical_key(parent.theta) != key:
            actions = _derive_actions_between_thetas(parent.theta, theta, schema_card)
            if actions:
                action_validation["actions_derived_from_theta"] = True
            else:
                continue
        seen.add(key)
        proposals.append(
            StrategyProposal(
                theta=theta,
                actions=actions,
                prior_score=0.55 + 0.02 * min(idx, 10),
                reason="deterministic target-aware fallback",
                action_validation=action_validation,
            )
        )
        if len(proposals) >= int(n):
            break
    return proposals


def propose_thetas(
    *,
    config: V2MCTSConfig,
    client: LLMClient | None,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
    real_utility_reference: dict[str, Any],
    s_node: SNode,
    source_profiles: dict[str, Any],
    n: int,
    parent: ThetaNode | None,
    archive: list[dict[str, Any]],
    trace_dir: Path,
    theta_nodes: dict[str, ThetaNode] | None = None,
) -> list[StrategyProposal]:
    if parent is not None and parent.s_id != s_node.s_id:
        raise ValueError(f"cannot refine parent {parent.node_id} from {parent.s_id} under current S {s_node.s_id}")
    prompt = _theta_prompt(
        config=config,
        schema_card=schema_card,
        dataset_context=dataset_context,
        real_utility_reference=real_utility_reference,
        s_node=s_node,
        source_profiles=source_profiles,
        n=n,
        parent=parent,
        archive=archive,
        theta_nodes=theta_nodes,
    )
    if parent is None:
        schema_name = "root_init_node" if config.mode == "single" else f"{s_node.s_id}_init_node"
    else:
        schema_name = f"{parent.node_id}_refine_node"
    payload = _call_llm_json(
        client=client,
        prompt=prompt,
        schema_name=schema_name,
        trace_dir=trace_dir,
    )
    proposals: list[StrategyProposal] = []
    if isinstance(payload, dict):
        raw_items = payload.get("theta_proposals", payload.get("proposals", []))
        for item in raw_items if isinstance(raw_items, list) else []:
            proposal = _proposal_from_payload(
                item,
                schema_card=schema_card,
                seed=config.seed + len(proposals),
                parent=parent,
            )
            if proposal is not None:
                proposals.append(proposal)
    proposals.extend(
        _deterministic_theta_candidates(
            schema_card=schema_card,
            dataset_context=dataset_context,
            n=int(n),
            seed=config.seed + 313 + 1009 * len(archive) + (0 if parent is None else sum(ord(ch) for ch in parent.node_id)),
            parent=parent,
        )
    )
    seen: set[str] = set()
    deduped: list[StrategyProposal] = []
    for proposal in proposals:
        key = canonical_key(proposal.theta)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(proposal)
        if len(deduped) >= int(n):
            break
    return deduped


def _maybe_random_replace(
    proposal: StrategyProposal,
    *,
    config: V2MCTSConfig,
    schema_card: dict[str, Any],
    rng: random.Random,
) -> StrategyProposal:
    validation = proposal.action_validation if isinstance(proposal.action_validation, dict) else {}
    if validation.get("locked_copy") or str(proposal.reason or "").startswith("transfer_existing_theta_from_"):
        return proposal
    if rng.random() >= float(config.p_random_replace):
        return proposal
    target = schema_card["target_column"]
    columns = list(schema_card["column_order"])
    features = _schema_feature_columns(schema_card)
    payload = theta_to_dict(proposal.theta)
    field = rng.choice(["col_1ds", "col_2ds", "col_ps", "col_u"])
    random_action: ThetaAction | None = None
    if field == "col_u":
        old = payload.get("col_u")
        choices = [column for column in features if column != old]
        if not choices:
            return proposal
        new = rng.choice(choices)
        payload["col_u"] = new
        random_action = ThetaAction(type="replace_col_u", old=old, new=new)
    else:
        current = [column for column in payload[field] if column != target]
        action_prefix = {"col_1ds": "col_1d", "col_2ds": "col_2d", "col_ps": "col_p"}[field]
        scope_choices = features if field == "col_ps" else columns
        if current:
            old = rng.choice(current)
            replacement_choices = [column for column in scope_choices if column != old and column not in payload[field]]
            if not replacement_choices:
                replacement_choices = [column for column in scope_choices if column != old]
            replacement = rng.choice(replacement_choices or scope_choices)
            payload[field] = [replacement if column == old else column for column in payload[field]]
            random_action = ThetaAction(type=f"replace_{action_prefix}", old=old, new=replacement)
        else:
            choices = [column for column in scope_choices if column not in payload[field]]
            if not choices:
                return proposal
            replacement = rng.choice(choices)
            payload[field].append(replacement)
            random_action = ThetaAction(type=f"add_{action_prefix}", new=replacement)
    theta = repair_theta_target_inclusive(payload, schema_card, rng)
    return StrategyProposal(
        theta=theta,
        actions=[*proposal.actions, *([random_action] if random_action is not None else [])],
        prior_score=proposal.prior_score,
        reason=f"random_replace_applied: {proposal.reason}",
        action_validation={**proposal.action_validation, "random_replace": True},
    )


def _rollout_node(
    *,
    config: V2MCTSConfig,
    s_node: SNode,
    theta_node: ThetaNode,
    rollouts_dir: Path,
) -> GuidedRolloutResult:
    rollout_dir = rollouts_dir / f"{s_node.s_id}_{theta_node.theta_id}"
    rollout_config = GuidedRolloutConfig(
        theta_id=theta_node.theta_id,
        dataset_name=config.dataset_name,
        exp_name=f"{config.exp_name}_{s_node.s_id}",
        artifact_dir=rollouts_dir,
        synthetic_csv=s_node.synthetic_csv,
        seed=config.seed,
        keep_k=config.keep_k,
        preselect_target=config.preselect_target,
        d_cur_size=config.d_cur_size,
        max_theta_pairs=config.max_theta_pairs,
        rollout_dir=rollout_dir,
        d_cur_source="synthetic",
        holdout_fraction=config.holdout_fraction,
        source="tabdiff",
        eval_device=config.eval_device,
        nn_device=config.nn_device,
        utility_exact_evaluator=config.utility_exact_evaluator,
        utility_exact_torch_epochs=config.utility_exact_torch_epochs,
        utility_diag_sample_size=config.utility_diag_sample_size,
        density_reference_size=config.density_reference_size,
        save_validation_records=config.save_validation_records,
        save_internal_records=config.save_rollout_internal_records,
        disable_progress=config.disable_progress,
        allow_target_in_fidelity_columns=True,
        allow_target_in_privacy_columns=True,
        privacy_encoding_column_mode="privacy_columns",
        synthetic_row_map=s_node.synthetic_row_map,
        rollout_direct_dcr_repair_enabled=config.rollout_direct_dcr_repair_enabled,
        rollout_direct_dcr_target_margin=config.rollout_direct_dcr_target_margin,
        rollout_direct_dcr_max_swap_fraction=config.rollout_direct_dcr_max_swap_fraction,
        rollout_direct_dcr_candidate_neighbors=config.rollout_direct_dcr_candidate_neighbors,
        rollout_direct_dcr_min_pair_utility_gain=config.rollout_direct_dcr_min_pair_utility_gain,
        rollout_direct_dcr_fallback_min_pair_utility_gain=config.rollout_direct_dcr_fallback_min_pair_utility_gain,
        rollout_reward_candidate_v2_enabled=config.rollout_reward_candidate_v2_enabled,
        rollout_reward_candidate_v2_max_swap_fraction=config.rollout_reward_candidate_v2_max_swap_fraction,
        rollout_reward_candidate_v2_max_candidate_sizes=config.rollout_reward_candidate_v2_max_candidate_sizes,
        rollout_reward_candidate_v2_min_proxy_delta=config.rollout_reward_candidate_v2_min_proxy_delta,
        rollout_reward_candidate_v2_fidelity_floor_eps=config.rollout_reward_candidate_v2_fidelity_floor_eps,
        rollout_reward_candidate_v2_utility_floor_eps=config.rollout_reward_candidate_v2_utility_floor_eps,
    )
    return run_guided_pareto_rollout(rollout_config, theta_node.theta)


def _attach_result(
    theta_node: ThetaNode,
    result: GuidedRolloutResult,
    *,
    config: V2MCTSConfig,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
    test_df: pd.DataFrame,
) -> None:
    theta_node.rollout_dir = Path(result.artifact_dir)
    theta_node.guard_pass = bool(result.guard.get("pass", False))
    theta_node.status = "success"
    theta_node.search_objectives = dict(result.search_objectives)
    theta_node.audit_metrics = dict(result.audit_metrics)
    theta_node.guard = dict(result.guard)
    theta_node.feedback = dict(result.feedback)
    theta_node.exact_reward = float(result.reward) if bool(result.reward_available) else 0.0
    theta_node.exact_reward_available = bool(result.reward_available)
    theta_node.reward_available = bool(result.reward_available)
    search_reward, search_available = _search_reward_from_proxy(
        audit_metrics=theta_node.audit_metrics,
        search_objectives=theta_node.search_objectives,
        feedback=theta_node.feedback,
    )
    if theta_node.exact_reward_available:
        theta_node.search_reward = float(theta_node.exact_reward)
        theta_node.search_reward_available = True
        theta_node.reward = float(theta_node.exact_reward)
        theta_node.reward_type = "exact"
    else:
        theta_node.search_reward = float(search_reward)
        theta_node.search_reward_available = bool(search_available)
        theta_node.reward = float(search_reward)
        theta_node.reward_type = "proxy_search"
    theta_node.exact_reward_failure_reason = _exact_reward_failure_reason(theta_node)
    diagnostics = _theta_diagnostics_from_rollout(
        theta_node,
        config=config,
        schema_card=schema_card,
        dataset_context=dataset_context,
        test_df=test_df,
        pareto_df=result.pareto_df,
    )
    theta_node.feedback["diagnostics"] = diagnostics
    if theta_node.rollout_dir is not None:
        save_json(Path(theta_node.rollout_dir) / "diagnostics.json", diagnostics)


def _theta_diagnostics_from_rollout(
    theta_node: ThetaNode,
    *,
    config: V2MCTSConfig,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
    test_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
) -> dict[str, Any]:
    dcr_quantiles = {"available": False}
    if theta_node.rollout_dir is not None:
        dcr_candidates = [
            Path(theta_node.rollout_dir) / "eval" / "selection_pareto" / "dcr.csv",
            Path(theta_node.rollout_dir) / "eval" / "pareto" / "dcr.csv",
            Path(theta_node.rollout_dir) / "dcr.csv",
        ]
        for dcr_path in dcr_candidates:
            if not dcr_path.exists():
                continue
            try:
                dcr_quantiles = _dcr_quantiles_from_frame(pd.read_csv(dcr_path))
                dcr_quantiles["dcr_csv_path"] = str(dcr_path)
            except Exception:
                dcr_quantiles = {"available": False, "reason": "failed_to_read_dcr_csv", "dcr_csv_path": str(dcr_path)}
            break
    utility_exact_report = _load_utility_exact_report(theta_node.rollout_dir)
    utility_feature_importance = _utility_importance_for_selected_evaluator(
        config=config,
        train_like_df=pareto_df,
        test_df=test_df,
        schema_card=schema_card,
        dataset_context=dataset_context,
        seed=config.seed + sum(ord(ch) for ch in theta_node.theta_id),
        sample_size=0,
        utility_report=utility_exact_report,
    )
    metrics_4d_reward = _metrics_4d_reward_summary(theta_node.audit_metrics)
    return {
        "shape_bad_columns": list(theta_node.feedback.get("shape_weak_columns", []) or [])[:5],
        "trend_bad_pairs": list(theta_node.feedback.get("trend_weak_pairs", []) or [])[:5],
        "dcr_quantiles": dcr_quantiles,
        "utility_feature_importance": utility_feature_importance,
        "utility_xgb_feature_importance": utility_feature_importance,
        "source_contribution": _load_source_contribution_summary(theta_node.rollout_dir),
        "search_scores": _search_scores_summary(theta_node.search_objectives),
        "exact_reward": theta_node.exact_reward if theta_node.exact_reward_available else None,
        "exact_reward_available": bool(theta_node.exact_reward_available),
        "exact_reward_failure_reason": theta_node.exact_reward_failure_reason,
        "search_reward": theta_node.search_reward if theta_node.search_reward_available else None,
        "search_reward_available": bool(theta_node.search_reward_available),
        "reward_type": theta_node.reward_type,
        "metrics_4d": {
            "shape": metrics_4d_reward.get("shape_global"),
            "trend": metrics_4d_reward.get("trend_global"),
            "dcr_privacy_reward": metrics_4d_reward.get("dcr_privacy_reward"),
            "utility_exact": metrics_4d_reward.get("utility_exact"),
        },
        "reward": theta_node.reward,
        "metrics_4d_reward": metrics_4d_reward,
    }


def diagnose_theta_batch(
    *,
    config: V2MCTSConfig,
    client: LLMClient | None,
    nodes: list[ThetaNode],
    theta_nodes: dict[str, ThetaNode] | None,
    best_node: ThetaNode | None,
    real_utility_reference: dict[str, Any],
    s_node: SNode,
    source_profiles: dict[str, Any],
    trace_dir: Path,
    event_name: str,
    phase: str,
) -> dict[str, Any]:
    nodes = [node for node in nodes if node.s_id == s_node.s_id]
    if not nodes:
        return {"available": False, "reason": "empty_batch_or_s_id_mismatch", "s_id": s_node.s_id}
    include_source_context = config.mode == "mixed"
    theta_batch = [_diagnosis_theta_node_for_prompt(node, include_source_context=include_source_context) for node in nodes]
    batch_node_ids = {node.node_id for node in nodes}
    if best_node is not None and best_node.s_id == s_node.s_id and best_node.node_id not in batch_node_ids:
        scoped_best_node = best_node
    else:
        scoped_best_node = _best_node_for_s(
            theta_nodes,
            s_id=s_node.s_id,
            exclude_node_ids=batch_node_ids,
        )
    best_summary = (
        None
        if scoped_best_node is None or scoped_best_node.node_id in batch_node_ids
        else _diagnosis_theta_node_for_prompt(scoped_best_node, include_source_context=include_source_context)
    )
    s_context = (
        _s_context_for_prompt(s_node=s_node, source_profiles=source_profiles, include_source_profile_mix=False)
        if config.mode == "mixed"
        else None
    )
    template_name = "v2_init_node_diagnosis_prompt.j2" if phase == "init" else "v2_refine_node_diagnosis_prompt.j2"
    payload_for_template: dict[str, Any] = {
        "dcr_balance_semantics": DCR_PROMPT_SEMANTICS,
        "real_utility_reference": _real_utility_for_prompt(real_utility_reference),
        "s_context": s_context,
        "theta_batch": theta_batch,
    }
    if phase == "init":
        payload_for_template["best_so_far"] = best_summary
    else:
        payload_for_template["reference_nodes"] = _diagnosis_reference_nodes_for_prompt(
            theta_nodes=theta_nodes,
            exclude_node_ids=batch_node_ids,
            s_id=s_node.s_id,
            include_source_context=include_source_context,
        )
    prompt = _render_prompt(
        config,
        template_name,
        payload_for_template,
    )
    batch_node_name = nodes[0].node_id if len(nodes) == 1 else f"{nodes[0].node_id}-{nodes[-1].node_id}"
    diagnosis_prompt_type = "init_node_diagnosis" if phase == "init" else "refine_node_diagnosis"
    payload = _call_llm_json(
        client=client,
        prompt=prompt,
        schema_name=f"batch_{batch_node_name}_{diagnosis_prompt_type}",
        trace_dir=trace_dir,
    )
    if not isinstance(payload, dict):
        for node in nodes:
            node.feedback["llm_semantic_summary"] = _fallback_node_summary(node)
        return {"available": False, "reason": "llm_unavailable", "batch_size": len(nodes)}

    by_id = {node.node_id: node for node in nodes}
    for item in payload.get("theta_diagnoses", payload.get("diagnosis", [])):
        if not isinstance(item, dict):
            continue
        node = by_id.get(str(item.get("node_id")))
        if node is None:
            continue
        node.feedback["llm_semantic_summary"] = _sanitize_llm_dcr_text(
            item.get("semantic_summary", ""),
            node.audit_metrics,
        )[:500]
        if str(item.get("score_reason", "")).strip():
            node.feedback["llm_score_reason"] = _sanitize_llm_dcr_text(
                item.get("score_reason", ""),
                node.audit_metrics,
            )[:500]
        node.llm_score = _clip01(item.get("llm_score", node.llm_score))
    if str(payload.get("s_semantic_summary", "")).strip():
        s_node.semantic_summary = _sanitize_llm_dcr_text(payload.get("s_semantic_summary", ""))[:600]
    report = {
        "available": True,
        "event_name": event_name,
        "batch_semantic_summary": _sanitize_llm_dcr_text(
            payload.get("batch_summary", payload.get("batch_semantic_summary", ""))
        )[:800],
        "node_ids": [node.node_id for node in nodes],
    }
    if config.mode == "mixed":
        report["s_semantic_summary"] = s_node.semantic_summary
    return report


def _fallback_node_summary(node: ThetaNode) -> str:
    metrics = node.audit_metrics
    objective_values = {
        "shape": metrics.get("shape_global"),
        "trend": metrics.get("trend_global"),
        "DCR balance": metrics.get("dcr_privacy"),
        "utility": metrics.get("utility_exact"),
    }
    valid = {
        key: parsed
        for key, value in objective_values.items()
        if (parsed := _finite_float(value)) is not None
    }
    strongest = max(valid, key=valid.get) if valid else "unknown"
    limiting = min(valid, key=valid.get) if valid else "unknown"
    reward_label = "exact reward" if node.exact_reward_available else "search reward"
    reward_value = node.exact_reward if node.exact_reward_available else node.search_reward
    parts = [
        f"{reward_label} {_prompt_number(reward_value)} candidate",
        f"strongest objective is {strongest}",
        f"limiting objective is {limiting}",
    ]
    if metrics.get("dcr_privacy") is not None:
        parts.append(f"DCR balance privacy_reward {_prompt_number(metrics.get('dcr_privacy'))}")
    if node.theta.col_u:
        parts.append(f"utility focus uses {node.theta.col_u}")
    return "; ".join(parts)


def _load_source_contribution_summary(rollout_dir: Path | None) -> dict[str, Any] | None:
    if rollout_dir is None:
        return None
    path = Path(rollout_dir) / "source_contribution.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        "available": data.get("available"),
        "sources": data.get("sources", [])[:6],
    }


def _archive_theta_summary(
    node: ThetaNode,
    *,
    include_source_context: bool = True,
    depth: int | None = None,
) -> dict[str, Any]:
    diagnostics = node.feedback.get("diagnostics", {}) if isinstance(node.feedback.get("diagnostics"), dict) else {}
    metrics = diagnostics.get("metrics_4d", {}) if isinstance(diagnostics.get("metrics_4d"), dict) else {}
    utility = diagnostics.get("utility_feature_importance") or diagnostics.get("utility_xgb_feature_importance") or {}
    output = {
        "node_id": node.node_id,
        "parent_node_id": node.parent_node_id,
        "depth": None if depth is None else int(depth),
        "s_id": node.s_id,
        "theta_id": node.theta_id,
        "status": node.status,
        "guard_pass": bool(node.guard_pass),
        "exact_reward": _prompt_number(node.exact_reward) if node.exact_reward_available else None,
        "exact_reward_available": bool(node.exact_reward_available),
        "exact_reward_failure_reason": node.exact_reward_failure_reason,
        "search_reward": _prompt_number(node.search_reward) if node.search_reward_available else None,
        "search_reward_available": bool(node.search_reward_available),
        "reward_type": node.reward_type,
        "reward": float(node.reward),
        "reward_available": bool(node.reward_available),
        "selection_score": _prompt_number(_node_selection_score(node)),
        "best_reward": _prompt_number(node.best_reward),
        "visits": int(node.visits),
        "metrics_4d": metrics,
        "llm_score": float(node.llm_score),
        "actions": [
            compact
            for compact in (_compact_action_dict(action) for action in list(node.actions or [])[:8])
            if compact
        ],
        "proposal_action_validation": dict(node.proposal_action_validation),
        "reason": _short_text(node.reason, 240),
        "shape_bad_columns_top": list(diagnostics.get("shape_bad_columns", []) or [])[:5],
        "trend_bad_pairs_top": list(diagnostics.get("trend_bad_pairs", []) or [])[:5],
        "dcr_quantiles": diagnostics.get("dcr_quantiles", {"available": False}),
        "utility_top_features": list(utility if isinstance(utility, list) else utility.get("top_features", []) if isinstance(utility, dict) else [])[:8],
        "theta_summary": theta_to_dict(node.theta),
        "semantic_summary": node.feedback.get("llm_semantic_summary") or _fallback_node_summary(node),
        "rollout_dir": None if node.rollout_dir is None else str(node.rollout_dir),
    }
    if node.proposal_action_validation.get("locked_copy") is True:
        output["locked_copy"] = True
        output["transfer_source_node_id"] = node.proposal_action_validation.get("transfer_source_node_id")
        output["transfer_source_theta_id"] = node.proposal_action_validation.get("transfer_source_theta_id")
        output["transfer_source_score"] = node.proposal_action_validation.get("transfer_source_score")
    if node.proposal_action_validation.get("random_replace") is True:
        output["random_replace"] = True
    if include_source_context:
        output["source_contribution_summary"] = diagnostics.get("source_contribution") or _load_source_contribution_summary(
            node.rollout_dir
        )
    return output


def _theta_record_with_depth(node: ThetaNode, depth: int | None) -> dict[str, Any]:
    output = node.to_dict()
    if depth is not None:
        output["depth"] = int(depth)
    return output


def _theta_detail_record(
    node: ThetaNode,
    *,
    include_source_context: bool = True,
    depth: int | None = None,
) -> dict[str, Any]:
    return {
        "compact": _archive_theta_summary(
            node,
            include_source_context=include_source_context,
            depth=depth,
        ),
        "full": _theta_record_with_depth(node, depth),
    }


def _best_node(theta_nodes: dict[str, ThetaNode]) -> ThetaNode | None:
    successful = [node for node in theta_nodes.values() if node.status == "success"]
    if not successful:
        return None
    guarded = [node for node in successful if node.guard_pass and node.exact_reward_available]
    exact_available = [node for node in successful if node.exact_reward_available]
    search_available = [node for node in successful if node.search_reward_available]
    pool = guarded if guarded else (exact_available if exact_available else (search_available if search_available else successful))
    return max(
        pool,
        key=lambda node: (
            bool(node.guard_pass and node.exact_reward_available),
            bool(node.exact_reward_available),
            _node_selection_score(node),
            float(node.audit_metrics.get("utility_exact") or 0.0),
            node.node_id,
        ),
    )


def _theta_child_counts(theta_nodes: dict[str, ThetaNode]) -> dict[str, int]:
    counts = {node_id: 0 for node_id in theta_nodes}
    for node in theta_nodes.values():
        if node.parent_node_id in counts:
            counts[str(node.parent_node_id)] += 1
    return counts


def _theta_children_by_parent(theta_nodes: dict[str, ThetaNode]) -> dict[str | None, list[ThetaNode]]:
    children: dict[str | None, list[ThetaNode]] = {}
    for node in theta_nodes.values():
        children.setdefault(node.parent_node_id, []).append(node)
    for bucket in children.values():
        bucket.sort(key=lambda node: node.node_id)
    return children


def _stdev_or_zero(values: list[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 2:
        return 0.0
    try:
        return float(statistics.stdev(clean))
    except statistics.StatisticsError:
        return 0.0


def _uct_base_score(best_reward: float, llm_score: float) -> float:
    return float(best_reward) + 0.05 * float(llm_score)


def _uct_explore_scale(base_scores: list[float], ucb_c: float) -> float:
    return max(float(ucb_c) * _stdev_or_zero(base_scores), UCT_EXPLORE_SCALE_MIN)


def _uct_explore(explore_scale: float, total_visits: int, visits: int) -> float:
    total = max(0, int(total_visits))
    node_visits = max(0, int(visits))
    return float(explore_scale) * math.sqrt(math.log(float(total) + 1.0) / (float(node_visits) + 1.0))


def _bounded_visit_increment(visits: int, total_visits: int) -> int:
    total = max(0, int(total_visits))
    return min(max(0, int(visits)) + 1, total)


def _theta_depth(node: ThetaNode, theta_nodes: dict[str, ThetaNode]) -> int:
    depth = 0
    current = node
    seen: set[str] = set()
    while current.parent_node_id is not None:
        parent_id = current.parent_node_id
        if parent_id in seen:
            break
        parent = theta_nodes.get(parent_id)
        if parent is None:
            break
        seen.add(parent_id)
        depth += 1
        current = parent
    return depth


def _theta_own_reward_for_tree(node: ThetaNode) -> float:
    if node.status != "success":
        return 0.0
    return _node_selection_score(node)


def _refresh_theta_best_reward_path(theta_nodes: dict[str, ThetaNode], start_node_id: str) -> None:
    children = _theta_children_by_parent(theta_nodes)
    current = theta_nodes.get(start_node_id)
    while current is not None:
        child_best = [
            float(child.best_reward)
            for child in children.get(current.node_id, [])
            if child.status == "success"
        ]
        current.best_reward = max([_theta_own_reward_for_tree(current), *child_best])
        current = theta_nodes.get(current.parent_node_id) if current.parent_node_id is not None else None


def _refresh_s_best_reward(s_node: SNode, theta_nodes: dict[str, ThetaNode]) -> None:
    rewards = [
        _node_selection_score(theta_nodes[node_id])
        for node_id in s_node.theta_node_ids
        if node_id in theta_nodes and theta_nodes[node_id].status == "success"
    ]
    s_node.best_reward = max([0.0, *rewards])


def _has_expandable_success_leaf(
    node: ThetaNode,
    *,
    children_by_parent: dict[str | None, list[ThetaNode]],
    child_counts: dict[str, int],
    memo: dict[str, bool],
) -> bool:
    if node.node_id in memo:
        return memo[node.node_id]
    if node.status != "success":
        memo[node.node_id] = False
        return False
    if int(child_counts.get(node.node_id, 0)) == 0:
        memo[node.node_id] = True
        return True
    memo[node.node_id] = any(
        _has_expandable_success_leaf(
            child,
            children_by_parent=children_by_parent,
            child_counts=child_counts,
            memo=memo,
        )
        for child in children_by_parent.get(node.node_id, [])
        if child.status == "success"
    )
    return memo[node.node_id]


def _root_successful_nodes_for_s(s_node: SNode, theta_nodes: dict[str, ThetaNode]) -> list[ThetaNode]:
    return [
        theta_nodes[node_id]
        for node_id in s_node.theta_node_ids
        if node_id in theta_nodes
        and theta_nodes[node_id].status == "success"
        and theta_nodes[node_id].parent_node_id is None
    ]


def _leaf_successful_nodes_for_s(s_node: SNode, theta_nodes: dict[str, ThetaNode]) -> list[ThetaNode]:
    child_counts = _theta_child_counts(theta_nodes)
    return [
        theta_nodes[node_id]
        for node_id in s_node.theta_node_ids
        if node_id in theta_nodes
        and theta_nodes[node_id].status == "success"
        and int(child_counts.get(node_id, 0)) == 0
    ]


def _select_uct_theta_from_s(
    *,
    s_node: SNode,
    theta_nodes: dict[str, ThetaNode],
    total_visits: int,
    ucb_c: float = 2.0,
    theta_proposals_per_event: int | None = None,
) -> tuple[ThetaNode | None, list[dict[str, Any]]]:
    children_by_parent = _theta_children_by_parent(theta_nodes)
    child_counts = _theta_child_counts(theta_nodes)
    expandable_memo: dict[str, bool] = {}
    roots = [
        node
        for node in _root_successful_nodes_for_s(s_node, theta_nodes)
        if _has_expandable_success_leaf(
            node,
            children_by_parent=children_by_parent,
            child_counts=child_counts,
            memo=expandable_memo,
        )
    ]
    if not roots:
        return None, []

    path: list[dict[str, Any]] = []

    def base_nodes_for_candidates(candidates: list[ThetaNode]) -> list[ThetaNode]:
        if not candidates:
            return []
        parent_id = candidates[0].parent_node_id
        if parent_id is None:
            base_nodes = [
                theta_nodes[node_id]
                for node_id in s_node.theta_node_ids
                if node_id in theta_nodes
                and theta_nodes[node_id].s_id == s_node.s_id
                and theta_nodes[node_id].parent_node_id is None
            ]
        else:
            base_nodes = [
                node
                for node in children_by_parent.get(parent_id, [])
                if node.s_id == s_node.s_id
            ]
        return base_nodes or candidates

    def choose(candidates: list[ThetaNode], depth: int) -> ThetaNode:
        base_nodes = base_nodes_for_candidates(candidates)
        base_scores = [_uct_base_score(node.best_reward, node.llm_score) for node in base_nodes]
        explore_scale = _uct_explore_scale(base_scores, ucb_c)
        t_value = max(0, int(total_visits))

        def score_components(node: ThetaNode) -> tuple[float, dict[str, float | int]]:
            visits = max(0, int(node.visits))
            prior = 0.05 * float(node.llm_score)
            base_score = _uct_base_score(node.best_reward, node.llm_score)
            explore = _uct_explore(explore_scale, t_value, visits)
            value = base_score + explore
            return (
                value,
                {
                    "base_score": float(base_score),
                    "best_reward": float(node.best_reward),
                    "prior": float(prior),
                    "explore": float(explore),
                    "visits_denominator": int(visits + 1),
                },
            )

        selected = max(candidates, key=lambda node: (score_components(node)[0], node.node_id))
        selected_score, selected_components = score_components(selected)
        visits_after = _bounded_visit_increment(selected.visits, t_value)
        path.append(
            {
                "depth": int(depth),
                "node_id": selected.node_id,
                "candidate_node_ids": [node.node_id for node in candidates],
                "base_score_node_ids": [node.node_id for node in base_nodes],
                "base_scores_count": int(len(base_scores)),
                "expected_theta_proposals_per_event": (
                    None if theta_proposals_per_event is None else int(theta_proposals_per_event)
                ),
                "actual_candidate_count": int(len(candidates)),
                "best_reward": float(selected.best_reward),
                "base_score": float(selected_components["base_score"]),
                "visits_before": int(selected.visits),
                "visits_after": int(visits_after),
                "visits_denominator": int(selected_components["visits_denominator"]),
                "dynamic_c": float(explore_scale),
                "explore_scale": float(explore_scale),
                "explore_scale_min": float(UCT_EXPLORE_SCALE_MIN),
                "ucb_c": float(ucb_c),
                "t_value": int(t_value),
                "total_visits": int(t_value),
                "uct_score": float(selected_score),
                "prior": float(selected_components["prior"]),
                "explore": float(selected_components["explore"]),
                "s_llm_score": float(s_node.llm_score),
                "theta_llm_score": float(selected.llm_score),
            }
        )
        selected.visits = visits_after
        return selected

    current = choose(roots, depth=0)
    depth = 1
    while int(child_counts.get(current.node_id, 0)) > 0:
        children = [
            child
            for child in children_by_parent.get(current.node_id, [])
            if child.status == "success"
            and _has_expandable_success_leaf(
                child,
                children_by_parent=children_by_parent,
                child_counts=child_counts,
                memo=expandable_memo,
            )
        ]
        if not children:
            return None, path
        current = choose(children, depth=depth)
        depth += 1
    return current, path


def _select_s_for_refine(
    s_nodes: dict[str, SNode],
    theta_nodes: dict[str, ThetaNode],
    total_visits: int,
    *,
    ucb_c: float = 2.0,
    trace_limit: int = 8,
) -> tuple[SNode | None, dict[str, Any]]:
    eligible = [s_node for s_node in s_nodes.values() if _leaf_successful_nodes_for_s(s_node, theta_nodes)]
    if not eligible:
        return None, {"available": False, "reason": "no_successful_leaf_theta_nodes"}
    base_score_by_s = {
        s_node.s_id: _uct_base_score(s_node.best_reward, s_node.llm_score)
        for s_node in s_nodes.values()
    }
    base_scores = list(base_score_by_s.values())
    explore_scale = _uct_explore_scale(base_scores, ucb_c)
    t_value = max(0, int(total_visits))
    scored: list[dict[str, Any]] = []

    def score_value(s_node: SNode) -> float:
        visits = max(0, int(s_node.visits))
        explore = _uct_explore(explore_scale, t_value, visits)
        return float(base_score_by_s[s_node.s_id]) + explore

    for s_node in eligible:
        visits = max(0, int(s_node.visits))
        prior = 0.05 * float(s_node.llm_score)
        explore = _uct_explore(explore_scale, t_value, visits)
        base_score = float(base_score_by_s[s_node.s_id])
        scored.append(
            {
                "s_id": s_node.s_id,
                "score": float(base_score + explore),
                "base_score": float(base_score),
                "best_reward": float(s_node.best_reward),
                "visits_before": int(s_node.visits),
                "visits_denominator": int(visits + 1),
                "llm_score": float(s_node.llm_score),
                "prior": float(prior),
                "explore": float(explore),
            }
        )

    selected = max(eligible, key=lambda node: (score_value(node), node.s_id))
    scored = sorted(scored, key=lambda item: (float(item["score"]), str(item["s_id"])), reverse=True)
    return selected, {
        "available": True,
        "selected_s_id": selected.s_id,
        "dynamic_c": float(explore_scale),
        "explore_scale": float(explore_scale),
        "explore_scale_min": float(UCT_EXPLORE_SCALE_MIN),
        "ucb_c": float(ucb_c),
        "t_value": int(t_value),
        "total_visits": int(t_value),
        "base_scores_count": int(len(base_scores)),
        "base_scores_source": "all_s_nodes",
        "total_s_nodes": int(len(s_nodes)),
        "eligible_s_count": int(len(eligible)),
        "candidates": scored[: max(1, int(trace_limit))],
    }


def _write_run_state(
    *,
    mcts_dir: Path,
    s_nodes: dict[str, SNode],
    theta_nodes: dict[str, ThetaNode],
    event_trace: list[dict[str, Any]],
    final_node: ThetaNode | None,
    final_status: str,
    include_source_context: bool = True,
) -> None:
    archive_dir = ensure_dir(mcts_dir / "archive")
    theta_depths = {
        node_id: _theta_depth(node, theta_nodes)
        for node_id, node in theta_nodes.items()
    }
    s_node_records = [
        {
            "s_id": node.s_id,
            "pool_units": node.pool_units,
            "synthetic_csv": str(node.synthetic_csv),
            "synthetic_row_map": str(node.synthetic_row_map),
            "llm_score": node.llm_score,
            "reason": node.reason,
            "semantic_summary": node.semantic_summary,
            "visits": node.visits,
            "theta_node_ids": node.theta_node_ids,
            "best_reward": node.best_reward,
        }
        for node in s_nodes.values()
    ]
    save_json(
        mcts_dir / "tree" / "s_nodes.json",
        s_node_records,
    )
    save_jsonl(mcts_dir / "tree" / "s_nodes.jsonl", s_node_records)
    save_jsonl(
        mcts_dir / "tree" / "theta_nodes.jsonl",
        [_theta_record_with_depth(node, theta_depths.get(node.node_id)) for node in theta_nodes.values()],
    )
    save_jsonl(mcts_dir / "tree" / "event_trace.jsonl", event_trace)
    detail_dir = ensure_dir(archive_dir / "theta_details")
    for node in theta_nodes.values():
        save_json(
            detail_dir / f"{node.node_id}.json",
            _theta_detail_record(
                node,
                include_source_context=include_source_context,
                depth=theta_depths.get(node.node_id),
            ),
        )
    all_nodes = [
        _archive_theta_summary(
            node,
            include_source_context=include_source_context,
            depth=theta_depths.get(node.node_id),
        )
        for node in theta_nodes.values()
    ]
    strategy_generation_events = [
        event
        for event in event_trace
        if event.get("event")
        in {"duplicate_strategy", "topup_strategy_candidates", "topup_exhausted", "transfer_top_theta_to_new_s"}
    ]
    save_jsonl(archive_dir / "all_theta_nodes.jsonl", all_nodes)
    save_jsonl(archive_dir / "successful_theta_nodes.jsonl", [node for node in all_nodes if node.get("status") == "success"])
    save_jsonl(archive_dir / "failed_theta_nodes.jsonl", [node for node in all_nodes if node.get("status") == "failed"])
    save_jsonl(archive_dir / "duplicate_strategy.jsonl", [event for event in event_trace if event.get("event") == "duplicate_strategy"])
    save_jsonl(archive_dir / "strategy_generation_events.jsonl", strategy_generation_events)
    save_json(
        archive_dir / "archive_schema.json",
        {
            "all_theta_nodes.jsonl": "Compact theta records for direct reading, including strict actions, proposal reason, and depth.",
            "theta_details/{node_id}.json": "Full theta node with complete feedback/debug payload and depth.",
            "successful_theta_nodes.jsonl": "Compact records with status=success and depth.",
            "failed_theta_nodes.jsonl": "Compact records with status=failed and depth.",
            "duplicate_strategy.jsonl": "Duplicate strategy events.",
            "strategy_generation_events.jsonl": "Transfer, duplicate, and top-up strategy generation events.",
            "s_nodes.jsonl": "Compact S-pool records.",
            "depth": "Theta tree depth: root theta nodes are 0; child theta nodes are parent depth + 1.",
        },
    )
    save_json(
        archive_dir / "run_index.json",
        {
            "final_node_id": None if final_node is None else final_node.node_id,
            "final_theta_id": None if final_node is None else final_node.theta_id,
            "total_theta_nodes": int(len(theta_nodes)),
            "total_s_nodes": int(len(s_nodes)),
            "theta_detail_dir": str(detail_dir),
        },
    )
    save_jsonl(archive_dir / "s_nodes.jsonl", [
        {
            "s_id": node.s_id,
            "pool_units": node.pool_units,
            "llm_score": node.llm_score,
            "semantic_summary": node.semantic_summary,
            "theta_node_ids": node.theta_node_ids,
            "best_reward": node.best_reward,
        }
        for node in s_nodes.values()
    ])
    if final_node is not None and final_node.rollout_dir is not None:
        final_dir = ensure_dir(mcts_dir / "final")
        src = final_node.rollout_dir / "selection_pareto.csv"
        metrics = final_node.rollout_dir / "metrics_summary.json"
        feedback = final_node.rollout_dir / "feedback.json"
        if src.exists():
            save_csv(final_dir / "final_pareto.csv", pd.read_csv(src))
        if metrics.exists():
            save_json(final_dir / "final_metrics_summary.json", json.loads(metrics.read_text(encoding="utf-8")))
        if feedback.exists():
            save_json(final_dir / "final_feedback.json", json.loads(feedback.read_text(encoding="utf-8")))
        save_json(final_dir / "theta_star.json", _theta_record_with_depth(final_node, theta_depths.get(final_node.node_id)))
    save_json(
        mcts_dir / "run_summary.json",
        {
            "final_status": final_status,
            "final_node_id": None if final_node is None else final_node.node_id,
            "final_theta_id": None if final_node is None else final_node.theta_id,
            "final_reward": None if final_node is None else final_node.reward,
            "final_exact_reward": None if final_node is None or not final_node.exact_reward_available else final_node.exact_reward,
            "final_search_reward": None if final_node is None or not final_node.search_reward_available else final_node.search_reward,
            "final_reward_type": None if final_node is None else final_node.reward_type,
            "successful_rollouts": int(sum(1 for node in theta_nodes.values() if node.status == "success")),
            "failed_rollouts": int(sum(1 for node in theta_nodes.values() if node.status == "failed")),
            "total_theta_nodes": int(len(theta_nodes)),
            "total_s_nodes": int(len(s_nodes)),
        },
    )


def _load_dataset_prompt_context(config: V2MCTSConfig, schema_card: dict[str, Any]) -> dict[str, Any]:
    path = Path(config.prompt_pack_dir) / "dataset_contexts" / f"{config.dataset_name}.prompt_context.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    columns = schema_card.get("columns", {})
    return {
        "dataset": config.dataset_name,
        "target_column": schema_card.get("target_column"),
        "columns": {
            "feature": [
                column
                for column in schema_card.get("column_order", [])
                if not bool(columns.get(column, {}).get("is_target", False))
            ]
        },
    }


def _evaluate_proposals(
    *,
    proposals: list[StrategyProposal],
    config: V2MCTSConfig,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
    test_df: pd.DataFrame,
    s_node: SNode,
    parent: ThetaNode | None,
    theta_nodes: dict[str, ThetaNode],
    visited: set[str],
    rollouts_dir: Path,
    rng: random.Random,
    event_trace: list[dict[str, Any]],
    checkpoint: Callable[[], None] | None = None,
    target_count: int | None = None,
    fallback_seed: int = 0,
) -> list[ThetaNode]:
    outputs: list[ThetaNode] = []
    queue = list(proposals)
    fallback_round = 0
    target = None if target_count is None else max(0, int(target_count))
    max_fallback_rounds = max(2, int(config.theta_proposals_per_event) * 3)

    def refill_queue_if_needed() -> None:
        nonlocal fallback_round
        if target is None or len(outputs) >= target or queue:
            return
        if fallback_round >= max_fallback_rounds:
            event_trace.append(
                {
                    "event": "topup_exhausted",
                    "s_id": s_node.s_id,
                    "parent_node_id": None if parent is None else parent.node_id,
                    "target_count": int(target),
                    "created_so_far": int(len(outputs)),
                    "fallback_rounds": int(fallback_round),
                }
            )
            return
        fallback_round += 1
        needed = max(1, target - len(outputs))
        fallback = _deterministic_theta_candidates(
            schema_card=schema_card,
            dataset_context=dataset_context,
            n=max(int(config.theta_proposals_per_event), needed * 3),
            seed=int(config.seed) + int(fallback_seed) + 7919 * fallback_round + 101 * len(theta_nodes),
            parent=parent,
        )
        if fallback:
            event_trace.append(
                {
                    "event": "topup_strategy_candidates",
                    "s_id": s_node.s_id,
                    "parent_node_id": None if parent is None else parent.node_id,
                    "target_count": int(target),
                    "created_so_far": int(len(outputs)),
                    "fallback_round": int(fallback_round),
                    "candidate_count": int(len(fallback)),
                }
            )
            queue.extend(fallback)

    while queue:
        if target is not None and len(outputs) >= target:
            break
        proposal = queue.pop(0)
        proposal = _maybe_random_replace(proposal, config=config, schema_card=schema_card, rng=rng)
        key = json.dumps({"s": _canonical_s_key(s_node.pool_units), "theta": canonical_key(proposal.theta)}, sort_keys=True)
        if key in visited:
            event_trace.append({"event": "duplicate_strategy", "s_id": s_node.s_id, "theta_id": theta_id(proposal.theta)})
            if checkpoint is not None:
                checkpoint()
            refill_queue_if_needed()
            continue
        visited.add(key)
        tid = theta_id(proposal.theta)
        node_id = f"n_{len(theta_nodes):06d}"
        theta_node = ThetaNode(
            node_id=node_id,
            s_id=s_node.s_id,
            theta=proposal.theta,
            theta_id=tid,
            parent_node_id=None if parent is None else parent.node_id,
            actions=[
                compact
                for compact in (_compact_action_dict(action) for action in proposal.actions)
                if compact
            ],
            proposal_action_validation=_compact_action_validation(proposal.action_validation),
            llm_score=float(proposal.prior_score),
            reason=proposal.reason,
        )
        theta_nodes[node_id] = theta_node
        s_node.theta_node_ids.append(node_id)
        if checkpoint is not None:
            checkpoint()
        try:
            result = _rollout_node(config=config, s_node=s_node, theta_node=theta_node, rollouts_dir=rollouts_dir)
        except Exception as exc:
            theta_node.status = "failed"
            theta_node.error = str(exc)
            theta_node.guard = build_guard({})
            event_trace.append({"event": "rollout_failed", "node_id": node_id, "s_id": s_node.s_id, "error": str(exc)})
        else:
            _attach_result(
                theta_node,
                result,
                config=config,
                schema_card=schema_card,
                dataset_context=dataset_context,
                test_df=test_df,
            )
            _refresh_theta_best_reward_path(theta_nodes, node_id)
            _refresh_s_best_reward(s_node, theta_nodes)
            event_trace.append(
                {
                    "event": "rollout_success",
                    "node_id": node_id,
                    "s_id": s_node.s_id,
                    "theta_id": tid,
                    "reward": float(theta_node.reward),
                    "selection_score": float(_node_selection_score(theta_node)),
                    "node_best_reward": float(theta_node.best_reward),
                    "s_best_reward": float(s_node.best_reward),
                    "exact_reward": theta_node.exact_reward if theta_node.exact_reward_available else None,
                    "search_reward": theta_node.search_reward if theta_node.search_reward_available else None,
                    "reward_type": theta_node.reward_type,
                    "guard_pass": bool(theta_node.guard_pass),
                    "proposal_reason": theta_node.reason,
                    "proposal_action_validation": dict(proposal.action_validation)
                    if isinstance(proposal.action_validation, dict)
                    else {},
                }
            )
        outputs.append(theta_node)
        if checkpoint is not None:
            checkpoint()
        refill_queue_if_needed()
    return outputs


def run_v2_mcts(config: V2MCTSConfig, client: LLMClient | None = None) -> V2RunResult:
    if config.dataset_name not in SUPPORTED_V2_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_V2_DATASETS))
        raise ValueError(f"v2 runner currently supports dataset_name in {{{supported}}}")
    if config.mode not in {"mixed", "single"}:
        raise ValueError("--mode must be mixed or single")

    mcts_dir = _mcts_dir(config)
    context_dir = ensure_dir(mcts_dir / "context")
    s_dir = ensure_dir(mcts_dir / "s_nodes")
    rollouts_dir = ensure_dir(mcts_dir / "rollouts")
    trace_dir = ensure_dir(mcts_dir / "llm_calls")
    ensure_dir(mcts_dir / "tree")
    ensure_dir(mcts_dir / "archive")

    dataset_ctx = resolve_tabdiff_selection_context(
        dataset_name=config.dataset_name,
        seed=config.seed,
        holdout_fraction=config.holdout_fraction,
    )
    cards = build_and_save_cards(
        train_df=dataset_ctx.train_df.copy(),
        output_dir=context_dir / "cards",
        seed=config.seed,
        dataset_name=config.dataset_name,
        target_column=dataset_ctx.target_column,
        categorical_columns=dataset_ctx.categorical_columns,
        numerical_columns=dataset_ctx.numerical_columns,
        discrete_numerical_columns=dataset_ctx.discrete_numerical_columns,
        privacy_sensitive_columns=dataset_ctx.privacy_sensitive_columns,
    )
    schema_card = cards.schema_card
    stats_card = cards.stats_card
    dataset_prompt_context = _load_dataset_prompt_context(config, schema_card)
    sources = resolve_sources(config)
    save_json(context_dir / "source_registry.json", _json_safe({key: info.__dict__ for key, info in sources.items()}))
    save_json(context_dir / "dataset_prompt_context.json", dataset_prompt_context)
    save_json(context_dir / "run_config.json", _json_safe(config.__dict__))
    real_utility_reference = build_real_utility_profile(
        config=config,
        train_df=dataset_ctx.train_df.copy(),
        test_df=dataset_ctx.test_df.copy(),
        schema_card=schema_card,
        stats_card=stats_card,
        dataset_context=dataset_prompt_context,
        context_dir=context_dir,
        client=None if config.provider == "mock" else client,
        trace_dir=trace_dir,
    )
    if config.mode == "single":
        source_profiles = _single_source_profiles_placeholder(config, sources)
        save_json(context_dir / "source_profiles.json", source_profiles)
    else:
        source_profiles = build_source_profiles(
            config=config,
            sources=sources,
            train_df=dataset_ctx.train_df.copy(),
            holdout_df=dataset_ctx.holdout_df.copy(),
            test_df=dataset_ctx.test_df.copy(),
            schema_card=schema_card,
            stats_card=stats_card,
            dataset_context=dataset_prompt_context,
            context_dir=context_dir,
            client=None if config.provider == "mock" else client,
            trace_dir=trace_dir,
        )

    rng = random.Random(config.seed)
    s_nodes: dict[str, SNode] = {}
    theta_nodes: dict[str, ThetaNode] = {}
    visited: set[str] = set()
    event_trace: list[dict[str, Any]] = []
    archive: list[dict[str, Any]] = []
    include_source_context = config.mode == "mixed"
    theta_star: ThetaNode | None = None
    no_improve = 0
    hard_no_improve = 0

    def checkpoint_run_state(final_status: str = "running") -> None:
        _write_run_state(
            mcts_dir=mcts_dir,
            s_nodes=s_nodes,
            theta_nodes=theta_nodes,
            event_trace=event_trace,
            final_node=_best_node(theta_nodes),
            final_status=final_status,
            include_source_context=include_source_context,
        )

    checkpoint_run_state("initialized")

    def register_s(pool_payload: dict[str, Any]) -> SNode:
        s_id = f"s_{len(s_nodes):06d}"
        synthetic_csv, row_map, report = build_s_pool(
            s_id=s_id,
            pool_units=pool_payload["pool_units"],
            config=config,
            sources=sources,
            train_rows=len(dataset_ctx.train_df),
            output_dir=s_dir / s_id,
            column_order=list(schema_card["column_order"]),
            schema_card=schema_card,
            stats_card=stats_card,
        )
        s_node = SNode(
            s_id=s_id,
            pool_units=pool_payload["pool_units"],
            synthetic_csv=synthetic_csv,
            synthetic_row_map=row_map,
            llm_score=float(pool_payload.get("llm_score", 0.5)),
            reason=str(pool_payload.get("reason", "")),
        )
        s_nodes[s_id] = s_node
        event_trace.append(
            {
                "event": "create_s",
                "s_id": s_id,
                "pool_units": s_node.pool_units,
                "manifest": report,
                "family": pool_payload.get("family"),
                "reason": s_node.reason,
            }
        )
        return s_node

    initial_s = select_s_pools(
        config=config,
        sources=sources,
        client=None if config.provider == "mock" else client,
        n=1 if config.mode == "single" else int(config.initial_s_pool_count),
        phase="init",
        source_profiles=source_profiles,
        real_utility_reference=real_utility_reference,
        s_nodes=s_nodes,
        theta_nodes=theta_nodes,
        trace_dir=trace_dir,
    )
    for pool_payload in initial_s:
        s_node = register_s(pool_payload)
        proposals = propose_thetas(
            config=config,
            client=None if config.provider == "mock" else client,
            schema_card=schema_card,
            dataset_context=dataset_prompt_context,
            real_utility_reference=real_utility_reference,
            s_node=s_node,
            source_profiles=source_profiles,
            n=config.theta_proposals_per_event,
            parent=None,
            archive=archive,
            trace_dir=trace_dir,
            theta_nodes=theta_nodes,
        )
        created = _evaluate_proposals(
            proposals=proposals,
            config=config,
            schema_card=schema_card,
            dataset_context=dataset_prompt_context,
            test_df=dataset_ctx.test_df.copy(),
            s_node=s_node,
            parent=None,
            theta_nodes=theta_nodes,
            visited=visited,
            rollouts_dir=rollouts_dir,
            rng=rng,
            event_trace=event_trace,
            checkpoint=lambda: checkpoint_run_state("running"),
            target_count=config.theta_proposals_per_event,
            fallback_seed=10000 + len(theta_nodes),
        )
        theta_star = _best_node(theta_nodes)
        diagnosis = diagnose_theta_batch(
            config=config,
            client=None if config.provider == "mock" else client,
            nodes=created,
            theta_nodes=theta_nodes,
            best_node=theta_star,
            real_utility_reference=real_utility_reference,
            s_node=s_node,
            source_profiles=source_profiles,
            trace_dir=trace_dir,
            event_name=f"init_{s_node.s_id}",
            phase="init",
        )
        event_trace.append({"event": "diagnosis", "s_id": s_node.s_id, "report": diagnosis})
        for node in created:
            if node.status == "success":
                archive.insert(
                    0,
                    _archive_theta_summary(
                        node,
                        include_source_context=include_source_context,
                        depth=_theta_depth(node, theta_nodes),
                    ),
                )
        theta_star = _best_node(theta_nodes)
        checkpoint_run_state("running")

    forced_new_s_used = False
    for event_idx in range(int(config.mcts_budget)):
        before = _node_selection_score(theta_star) if theta_star is not None else -1.0
        new_s_pool_stagnation_threshold = max(0, int(config.new_s_pool_stagnation_events))
        early_stop_stagnation_threshold = int(config.early_stop_stagnation_events)
        parent: ThetaNode | None = None
        s_ucb_selection: dict[str, Any] = {}
        theta_uct_path: list[dict[str, Any]] = []
        force_new_s_now = (
            config.mode == "mixed"
            and config.force_new_s_at_event is not None
            and not forced_new_s_used
            and event_idx >= max(0, int(config.force_new_s_at_event))
        )
        transfer_new_s_event = config.mode == "mixed" and (force_new_s_now or no_improve > new_s_pool_stagnation_threshold)
        event_batches: list[dict[str, Any]] = []
        if transfer_new_s_event:
            refine_s_pool_count = max(1, int(config.refine_s_pool_count))
            pool_payloads = select_s_pools(
                config=config,
                sources=sources,
                client=None if config.provider == "mock" else client,
                n=refine_s_pool_count,
                phase="refine",
                source_profiles=source_profiles,
                real_utility_reference=real_utility_reference,
                s_nodes=s_nodes,
                theta_nodes=theta_nodes,
                trace_dir=trace_dir,
            )
            if not pool_payloads:
                event_trace.append(
                    {
                        "event": "stop_no_new_s_pool_for_transfer",
                        "event_idx": int(event_idx),
                        "trigger": "forced_validation" if force_new_s_now else "stagnation",
                        "requested_refine_s_pool_count": int(refine_s_pool_count),
                        "existing_s": list(s_nodes),
                        "reason": "source selector returned no unseen integer multi-source pool",
                    }
                )
                break
            for pool_idx, pool_payload in enumerate(pool_payloads):
                s_node = register_s(pool_payload)
                proposals = _top_unique_transfer_proposals(theta_nodes, n=config.theta_proposals_per_event)
                event_trace.append(
                    {
                        "event": "transfer_top_theta_to_new_s",
                        "event_idx": int(event_idx),
                        "s_id": s_node.s_id,
                        "pool_idx": int(pool_idx),
                        "requested_refine_s_pool_count": int(refine_s_pool_count),
                        "actual_refine_s_pool_count": int(len(pool_payloads)),
                        "target_count": int(config.theta_proposals_per_event),
                        "trigger": "forced_validation" if force_new_s_now else "stagnation",
                        "force_new_s_at_event": config.force_new_s_at_event,
                        "no_improve_expand_events": int(no_improve),
                        "new_s_pool_stagnation_threshold": int(new_s_pool_stagnation_threshold),
                        "transfers": [
                            {
                                "source_node_id": proposal.action_validation.get("transfer_source_node_id"),
                                "source_theta_id": proposal.action_validation.get("transfer_source_theta_id"),
                                "source_score": proposal.action_validation.get("transfer_source_score"),
                                "theta_id": theta_id(proposal.theta),
                                "locked_copy": bool(proposal.action_validation.get("locked_copy")),
                            }
                            for proposal in proposals
                        ],
                    }
                )
                event_batches.append(
                    {
                        "s_node": s_node,
                        "parent": None,
                        "proposals": proposals,
                        "event_name": "transfer_top_theta_to_new_s_event",
                        "transfer_new_s_event": True,
                        "s_ucb_selection": {},
                        "theta_uct_path": [],
                    }
                )
            no_improve = 0
            forced_new_s_used = forced_new_s_used or bool(force_new_s_now)
        else:
            total_visits = int(event_idx) + 1
            s_node, s_ucb_selection = _select_s_for_refine(
                s_nodes,
                theta_nodes=theta_nodes,
                total_visits=total_visits,
                ucb_c=config.ucb_c,
            )
            if s_node is None:
                event_trace.append(
                    {
                        "event": "stop_no_leaf_nodes",
                        "event_idx": int(event_idx),
                        "reason": "no successful leaf theta nodes available for expansion",
                    }
                )
                break
            s_node.visits = _bounded_visit_increment(s_node.visits, total_visits)
            parent, theta_uct_path = _select_uct_theta_from_s(
                s_node=s_node,
                theta_nodes=theta_nodes,
                total_visits=total_visits,
                ucb_c=config.ucb_c,
                theta_proposals_per_event=config.theta_proposals_per_event,
            )
            if parent is None:
                event_trace.append(
                    {
                        "event": "stop_no_leaf_nodes",
                        "event_idx": int(event_idx),
                        "s_id": s_node.s_id,
                        "reason": "selected S has no successful leaf theta nodes",
                    }
                )
                break
            proposals = propose_thetas(
                config=config,
                client=None if config.provider == "mock" else client,
                schema_card=schema_card,
                dataset_context=dataset_prompt_context,
                real_utility_reference=real_utility_reference,
                s_node=s_node,
                source_profiles=source_profiles,
                n=config.theta_proposals_per_event,
                parent=parent,
                archive=archive,
                trace_dir=trace_dir,
                theta_nodes=theta_nodes,
            )
            event_name = "refine_node_event"
            event_batches.append(
                {
                    "s_node": s_node,
                    "parent": parent,
                    "proposals": proposals,
                    "event_name": event_name,
                    "transfer_new_s_event": False,
                    "s_ucb_selection": s_ucb_selection,
                    "theta_uct_path": theta_uct_path,
                }
            )

        all_created: list[ThetaNode] = []
        batch_summaries: list[dict[str, Any]] = []
        for batch_idx, batch in enumerate(event_batches):
            batch_s_node = batch["s_node"]
            batch_parent = batch["parent"]
            batch_transfer = bool(batch["transfer_new_s_event"])
            batch_event_name = str(batch["event_name"])
            created = _evaluate_proposals(
                proposals=batch["proposals"],
                config=config,
                schema_card=schema_card,
                dataset_context=dataset_prompt_context,
                test_df=dataset_ctx.test_df.copy(),
                s_node=batch_s_node,
                parent=None if batch_transfer else batch_parent,
                theta_nodes=theta_nodes,
                visited=visited,
                rollouts_dir=rollouts_dir,
                rng=rng,
                event_trace=event_trace,
                checkpoint=lambda: checkpoint_run_state("running"),
                target_count=config.theta_proposals_per_event,
                fallback_seed=20000 + event_idx * 1000 + len(theta_nodes) + batch_idx * 101,
            )
            all_created.extend(created)
            theta_star = _best_node(theta_nodes)
            diagnosis = diagnose_theta_batch(
                config=config,
                client=None if config.provider == "mock" else client,
                nodes=created,
                theta_nodes=theta_nodes,
                best_node=theta_star,
                real_utility_reference=real_utility_reference,
                s_node=batch_s_node,
                source_profiles=source_profiles,
                trace_dir=trace_dir,
                event_name=f"{batch_event_name}_{event_idx}_{batch_idx}",
                phase="init" if batch_transfer else "refine",
            )
            event_trace.append({"event": "diagnosis", "event_idx": int(event_idx), "s_id": batch_s_node.s_id, "report": diagnosis})
            for node in created:
                if node.status == "success":
                    archive.insert(
                        0,
                        _archive_theta_summary(
                            node,
                            include_source_context=include_source_context,
                            depth=_theta_depth(node, theta_nodes),
                        ),
                    )
            batch_summaries.append(
                {
                    "s_id": batch_s_node.s_id,
                    "parent_node_id": None if batch_transfer else (None if batch_parent is None else batch_parent.node_id),
                    "created_nodes": [node.node_id for node in created],
                    "generation_mode": "transfer_top_theta_to_new_s" if batch_transfer else "refine_leaf_theta",
                    "s_ucb_selection": None if batch_transfer else batch["s_ucb_selection"],
                    "theta_uct_path": [] if batch_transfer else batch["theta_uct_path"],
                }
            )
        archive = archive[:32]
        after = _node_selection_score(theta_star) if theta_star is not None else -1.0
        improved = after > before + 1e-12
        if improved:
            no_improve = 0
            hard_no_improve = 0
        else:
            no_improve += 1
            hard_no_improve += 1
        event_trace.append(
            {
                "event": "multi_transfer_top_theta_to_new_s_event"
                if transfer_new_s_event and len(event_batches) > 1
                else str(event_batches[0]["event_name"]),
                "event_idx": int(event_idx),
                "s_id": batch_summaries[0]["s_id"] if len(batch_summaries) == 1 else None,
                "parent_node_id": batch_summaries[0]["parent_node_id"] if len(batch_summaries) == 1 else None,
                "created_nodes": [node.node_id for node in all_created],
                "generation_mode": "multi_transfer_top_theta_to_new_s"
                if transfer_new_s_event and len(event_batches) > 1
                else batch_summaries[0]["generation_mode"],
                "s_ucb_selection": None if transfer_new_s_event else batch_summaries[0]["s_ucb_selection"],
                "theta_uct_path": [] if transfer_new_s_event else batch_summaries[0]["theta_uct_path"],
                "batches": batch_summaries,
                "best_selection_score_before": float(before),
                "best_selection_score_after": float(after),
                "best_exact_reward_after": None if theta_star is None or not theta_star.exact_reward_available else theta_star.exact_reward,
                "best_search_reward_after": None if theta_star is None or not theta_star.search_reward_available else theta_star.search_reward,
                "improved": bool(improved),
                "no_improve_expand_events": int(no_improve),
                "new_s_pool_stagnation_threshold": int(new_s_pool_stagnation_threshold),
                "trigger_requires_no_improve_gt_threshold": True,
                "hard_no_improve_expand_events": int(hard_no_improve),
                "early_stop_stagnation_threshold": int(early_stop_stagnation_threshold),
            }
        )
        checkpoint_run_state("running")
        if _should_stop_for_hard_no_improve(hard_no_improve, early_stop_stagnation_threshold):
            event_trace.append(
                {
                    "event": "stop_hard_no_improve",
                    "event_idx": int(event_idx),
                    "hard_no_improve_expand_events": int(hard_no_improve),
                    "early_stop_stagnation_threshold": int(early_stop_stagnation_threshold),
                    "reason": "global best theta did not improve for more than early_stop_stagnation_events expansion events",
                }
            )
            checkpoint_run_state("running")
            break

    final_node = _best_node(theta_nodes)
    if final_node is None:
        final_status = "failed_no_successful_rollouts"
    elif final_node.guard_pass and final_node.reward_available:
        final_status = "guard_pass"
    else:
        final_status = "guard_failed_best_effort"
    _write_run_state(
        mcts_dir=mcts_dir,
        s_nodes=s_nodes,
        theta_nodes=theta_nodes,
        event_trace=event_trace,
        final_node=final_node,
        final_status=final_status,
        include_source_context=include_source_context,
    )
    return V2RunResult(
        mcts_dir=mcts_dir,
        final_node=final_node,
        final_status=final_status,
        baseline_reward=None,
    )
