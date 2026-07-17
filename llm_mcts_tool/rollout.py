from __future__ import annotations

import math
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metric_tool.config import MetricConfig
from metric_tool.pipeline import evaluate_single_selection
from post_selection_tool.config import CoreSelectionConfig, progress_enabled
from post_selection_tool.context import prepare_context
from post_selection_tool.exact_score import compute_global_exact_scores
from post_selection_tool.fidelity_ceiling import build_fidelity_ceiling
from post_selection_tool.io import ensure_dir, save_csv, save_json, save_jsonl
from post_selection_tool.pareto_repair import apply_pareto_post_selection_repairs
from post_selection_tool.preselect import build_preselected_valid
from post_selection_tool.reward_candidate_v2 import refine_selection_for_reward_v2
from post_selection_tool.validation import build_cards_and_validate, initialize_selector_and_pool
from postprocess.tabdiff_utils import find_latest_tabdiff_sample

from .feedback import build_guard, build_rollout_feedback
from .strategy import StrategyTheta, canonical_key, theta_id as make_theta_id


@dataclass
class GuidedRolloutConfig:
    theta_id: str
    dataset_name: str
    exp_name: str
    artifact_dir: Path | None
    synthetic_csv: Path | None
    seed: int
    keep_k: int
    preselect_target: int
    d_cur_size: int
    max_theta_pairs: int
    rollout_dir: Path
    d_cur_source: str = "synthetic"
    holdout_fraction: float = 0.1
    source: str = "tabdiff"
    eval_device: str = "auto"
    nn_device: str = "auto"
    utility_exact_evaluator: str = "tabdiff_mle"
    utility_exact_torch_epochs: int = 6
    utility_diag_sample_size: int = 6000
    density_reference_size: int = 5000
    save_validation_records: bool = False
    save_internal_records: bool = False
    disable_progress: bool = True
    allow_target_in_fidelity_columns: bool = False
    allow_target_in_privacy_columns: bool = False
    privacy_encoding_column_mode: str = "privacy_columns"
    synthetic_row_map: Path | None = None
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


@dataclass
class GuidedRolloutResult:
    theta_id: str
    theta: StrategyTheta
    pareto_df: pd.DataFrame
    pareto_records: list[dict[str, Any]]
    search_objectives: dict[str, float]
    internal_reports: dict[str, Any]
    artifact_dir: Path
    reward: float
    reward_available: bool
    audit_metrics: dict[str, Any]
    feedback: dict[str, Any]
    guard: dict[str, Any]
    metrics_summary: dict[str, Any]


def _theta_to_dict(theta: StrategyTheta) -> dict[str, Any]:
    return {
        "col_1ds": list(theta.col_1ds),
        "col_2ds": list(theta.col_2ds),
        "col_ps": list(theta.col_ps),
        "col_u": theta.col_u,
    }


def _column_lookup(columns: list[str], feature_columns: list[str] | None = None) -> dict[str, str | None]:
    feature_set = set(columns if feature_columns is None else feature_columns)
    lookup: dict[str, str | None] = {}
    for idx, column in enumerate(columns):
        lookup[str(idx)] = str(column) if column in feature_set else None
        lookup[column] = str(column) if column in feature_set else None
    return lookup


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, tuple):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(inner) for inner in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return [_json_safe(inner) for inner in value.tolist()]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _resolve_rollout_synthetic_csv(config: GuidedRolloutConfig) -> Path:
    if config.synthetic_csv is not None:
        return Path(config.synthetic_csv)
    return find_latest_tabdiff_sample(dataset_name=config.dataset_name, exp_name=config.exp_name)


def _build_core_config(config: GuidedRolloutConfig, theta: StrategyTheta) -> CoreSelectionConfig:
    rollout_dir = Path(config.rollout_dir)
    mcts_dir = rollout_dir.parent.parent
    return CoreSelectionConfig(
        synthetic_csv=_resolve_rollout_synthetic_csv(config),
        dataset_name=config.dataset_name,
        exp_name=rollout_dir.name,
        artifact_dir=rollout_dir.parent,
        shared_artifact_dir=mcts_dir / "shared",
        seed=config.seed,
        source=config.source,
        keep_k=config.keep_k,
        preselect_target=config.preselect_target,
        d_cur_size=config.d_cur_size,
        d_cur_source=config.d_cur_source,
        holdout_fraction=config.holdout_fraction,
        fidelity_1d_columns=list(theta.col_1ds),
        fidelity_2d_anchor_columns=list(theta.col_2ds),
        privacy_columns=list(theta.col_ps),
        utility_balance_column=theta.col_u,
        allow_target_in_fidelity_columns=config.allow_target_in_fidelity_columns,
        allow_target_in_privacy_columns=config.allow_target_in_privacy_columns,
        privacy_encoding_column_mode=config.privacy_encoding_column_mode,
        max_theta_pairs=config.max_theta_pairs,
        direct_dcr_repair_v19_enabled=config.rollout_direct_dcr_repair_enabled,
        direct_dcr_repair_v19_target_margin=config.rollout_direct_dcr_target_margin,
        direct_dcr_repair_v19_max_swap_fraction=config.rollout_direct_dcr_max_swap_fraction,
        direct_dcr_repair_v19_candidate_neighbors=config.rollout_direct_dcr_candidate_neighbors,
        direct_dcr_repair_v19_min_pair_utility_gain=config.rollout_direct_dcr_min_pair_utility_gain,
        direct_dcr_repair_v19_fallback_min_pair_utility_gain=config.rollout_direct_dcr_fallback_min_pair_utility_gain,
        reward_candidate_v2_enabled=config.rollout_reward_candidate_v2_enabled,
        reward_candidate_v2_max_swap_fraction=config.rollout_reward_candidate_v2_max_swap_fraction,
        reward_candidate_v2_max_candidate_sizes=config.rollout_reward_candidate_v2_max_candidate_sizes,
        reward_candidate_v2_min_proxy_delta=config.rollout_reward_candidate_v2_min_proxy_delta,
        reward_candidate_v2_fidelity_floor_eps=config.rollout_reward_candidate_v2_fidelity_floor_eps,
        reward_candidate_v2_utility_floor_eps=config.rollout_reward_candidate_v2_utility_floor_eps,
        density_reference_size=config.density_reference_size,
        save_validation_records=config.save_validation_records,
        nn_device=config.nn_device,
        eval_device=config.eval_device,
        disable_progress=config.disable_progress,
    )


def _select_guided_pareto(state: Any) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    if state.selector is None:
        raise RuntimeError("selector is required before guided Pareto selection")
    return state.selector.select_keep(
        preselected_records=state.preselected_valid,
        surrogate_records=[],
        exact_records=state.global_exact_records,
        keep_k=state.effective_keep_k,
        floor_reference=state.floor_reference,
        constraint_reference_records=state.fidelity_ceiling_records,
        floor_mode=state.config.pareto_floor_mode,
        soft_fidelity_floor_eps=state.config.pareto_soft_fidelity_floor_eps,
        soft_trend_floor_eps=state.config.pareto_soft_trend_floor_eps,
        soft_privacy_floor_eps=state.config.pareto_soft_privacy_floor_eps,
        soft_utility_floor_eps=state.config.pareto_soft_utility_floor_eps,
        soft_min_score_delta=state.config.pareto_soft_min_score_delta,
    )


def _apply_rollout_reward_candidate_v2(
    *,
    config: GuidedRolloutConfig,
    state: Any,
    pareto_df: pd.DataFrame,
    pareto_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    if config.rollout_reward_candidate_v2_enabled:
        out_df, out_records, report = refine_selection_for_reward_v2(
            preselected_records=state.preselected_valid,
            exact_records=state.global_exact_records,
            selected_records=pareto_records,
            keep_k=state.effective_keep_k,
            column_order=state.selector.column_order,
            max_swap_fraction=config.rollout_reward_candidate_v2_max_swap_fraction,
            max_candidate_sizes=config.rollout_reward_candidate_v2_max_candidate_sizes,
            min_proxy_delta=config.rollout_reward_candidate_v2_min_proxy_delta,
            fidelity_floor_eps=config.rollout_reward_candidate_v2_fidelity_floor_eps,
            utility_floor_eps=config.rollout_reward_candidate_v2_utility_floor_eps,
        )
        return out_df, out_records, {**report, "stage": "pre_direct_dcr_repair"}
    return (
        pareto_df,
        pareto_records,
        {
            "enabled": False,
            "version": "reward_candidate_v2",
            "applied": False,
            "reason": "disabled",
            "stage": "pre_direct_dcr_repair",
        },
    )


def _mean_record_value(records: list[dict[str, Any]], key: str) -> float:
    values = []
    for record in records:
        try:
            value = float(record.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def _finite_report_value(report: dict[str, Any], key: str) -> float | None:
    try:
        value = float(report.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _compute_search_objectives(
    *,
    selector: Any,
    pareto_df: pd.DataFrame,
    pareto_records: list[dict[str, Any]],
    pareto_report: dict[str, Any] | None = None,
) -> dict[str, float]:
    report = pareto_report or {}
    p_theta = _finite_report_value(report, "selected_privacy_component_mean")
    if p_theta is None:
        p_theta = _mean_record_value(pareto_records, "pareto_priv_obj")
    u_proxy_theta = _finite_report_value(report, "selected_utility_mean")
    if u_proxy_theta is None:
        u_proxy_theta = _mean_record_value(pareto_records, "pareto_util_proxy_obj")
    return {
        "F_1D_theta": float(selector.compute_dataset_fidelity(pareto_df)),
        "F_2D_theta": float(selector.compute_dataset_pair_fidelity(pareto_df)),
        "P_theta": float(p_theta),
        "P_theta_raw": float(selector.compute_dataset_privacy(pareto_df)),
        "U_proxy_theta": float(u_proxy_theta),
    }


def _save_rollout_artifacts(
    *,
    config: GuidedRolloutConfig,
    theta: StrategyTheta,
    pareto_df: pd.DataFrame,
    pareto_records: list[dict[str, Any]],
    pareto_report: dict[str, Any],
    state: Any,
    search_objectives: dict[str, float],
) -> dict[str, Any]:
    rollout_dir = ensure_dir(Path(config.rollout_dir))
    internal_dir = ensure_dir(rollout_dir / "internal")
    save_json(
        rollout_dir / "theta.json",
        {
            "theta_id": config.theta_id,
            "canonical_key": canonical_key(theta),
            "theta": _theta_to_dict(theta),
        },
    )
    save_csv(rollout_dir / "selection_pareto.csv", pareto_df)
    selected_source_summary = _save_selected_source_map(
        rollout_dir=rollout_dir,
        synthetic_row_map=config.synthetic_row_map,
        pareto_records=pareto_records,
        pool_records=getattr(state, "pool_records", []),
    )
    if config.save_internal_records:
        save_jsonl(internal_dir / "pareto_records.jsonl", pareto_records)
        save_jsonl(internal_dir / "exact_scores.jsonl", state.global_exact_records)
        save_jsonl(internal_dir / "utility_proxy_scores.jsonl", state.utility_proxy_bundle.get("proxy_scores", []))
    save_json(
        internal_dir / "record_artifacts.json",
        {
            "save_internal_records": bool(config.save_internal_records),
            "pareto_records_rows": int(len(pareto_records)),
            "exact_scores_rows": int(len(state.global_exact_records)),
            "utility_proxy_scores_rows": int(len(state.utility_proxy_bundle.get("proxy_scores", []))),
            "saved_files": (
                ["pareto_records.jsonl", "exact_scores.jsonl", "utility_proxy_scores.jsonl"]
                if config.save_internal_records
                else []
            ),
        },
    )
    save_json(internal_dir / "preselect_report.json", state.preselect_gate)
    save_json(internal_dir / "preselect_status.json", state.preselect_status)
    save_json(internal_dir / "fidelity_ceiling_report.json", state.fidelity_ceiling_report)
    save_json(internal_dir / "baselines.json", state.global_baselines)
    save_json(
        internal_dir / "utility_proxy_manifest.json",
        {
            **state.utility_proxy_bundle.get("manifest", {}),
            "pre_ceiling_static": state.utility_proxy_bundle.get("pre_ceiling_static", {}).get("manifest", {}),
            "merge_report": state.utility_proxy_merge_report,
        },
    )
    internal_reports = {
        "preselect_report": state.preselect_gate,
        "preselect_status": state.preselect_status,
        "selector_preselect_report": getattr(state.selector, "last_preselect_report", {}),
        "fidelity_ceiling_report": state.fidelity_ceiling_report,
        "pareto_report": pareto_report,
        "global_baselines": state.global_baselines,
        "utility_proxy_manifest": state.utility_proxy_bundle.get("manifest", {}),
    }
    save_json(
        rollout_dir / "rollout_report.json",
        _json_safe(
            {
                "theta_id": config.theta_id,
                "rows": int(len(pareto_df)),
                "requested_keep_k": int(config.keep_k),
                "effective_keep_k": int(state.effective_keep_k),
                "guided_scopes": {
                    "fidelity_1d_columns": list(state.selector.fidelity_1d_columns),
                    "fidelity_2d_anchor_columns": list(state.selector.fidelity_2d_anchor_columns),
                    "privacy_columns": list(state.selector.privacy_columns),
                    "utility_balance_column": state.selector.utility_balance_column,
                    "max_theta_pairs": int(state.selector.max_pair_marginal_edges),
                    "pair_edges": [
                        {"left": edge["left"], "right": edge["right"], "mi": float(edge.get("mi", 0.0))}
                        for edge in state.selector.pair_marginal_edges
                    ],
                },
                "reports": internal_reports,
                "source_contribution": selected_source_summary,
            }
        ),
    )
    save_json(rollout_dir / "theta_objectives.json", _json_safe(search_objectives))
    return internal_reports


def _load_pool_row_map(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    map_path = Path(path)
    if not map_path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for line in map_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            pool_row_id = int(record.get("pool_row_id"))
        except Exception:
            continue
        rows[pool_row_id] = record
    return rows


def _save_selected_source_map(
    *,
    rollout_dir: Path,
    synthetic_row_map: Path | None,
    pareto_records: list[dict[str, Any]],
    pool_records: list[dict[str, Any]],
) -> dict[str, Any]:
    pool_row_map = _load_pool_row_map(synthetic_row_map)
    if not pool_row_map:
        summary = {"available": False, "reason": "synthetic_row_map_missing_or_empty"}
        save_json(rollout_dir / "source_contribution.json", summary)
        return summary

    candidate_to_pool_row: dict[int, int] = {}
    for record in pool_records:
        try:
            candidate_id = int(record.get("candidate_id"))
        except Exception:
            continue
        candidate_to_pool_row[candidate_id] = candidate_id

    selected_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for pos, record in enumerate(pareto_records):
        try:
            candidate_id = int(record.get("candidate_id", pos))
        except Exception:
            candidate_id = int(pos)
        pool_row_id = candidate_to_pool_row.get(candidate_id, candidate_id)
        source_record = pool_row_map.get(pool_row_id, {})
        source_id = str(source_record.get("source_id", "unknown"))
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
        selected_rows.append(
            {
                "selected_pos": int(pos),
                "candidate_id": int(candidate_id),
                "pool_row_id": int(pool_row_id),
                "source_id": source_id,
                "source_row_id": source_record.get("source_row_id"),
                "draw_index": source_record.get("draw_index"),
                "sample_seed": source_record.get("sample_seed"),
                "with_replacement": bool(source_record.get("with_replacement", False)),
            }
        )

    save_jsonl(rollout_dir / "selected_row_map.jsonl", selected_rows)
    total_pool = max(len(pool_row_map), 1)
    total_selected = max(len(selected_rows), 1)
    pool_counts: dict[str, int] = {}
    for record in pool_row_map.values():
        source_id = str(record.get("source_id", "unknown"))
        pool_counts[source_id] = pool_counts.get(source_id, 0) + 1
    sources = sorted(set(pool_counts) | set(source_counts))
    contribution = [
        {
            "source_id": source_id,
            "source_pool_count": int(pool_counts.get(source_id, 0)),
            "source_pool_fraction": float(pool_counts.get(source_id, 0) / total_pool),
            "source_selected_count": int(source_counts.get(source_id, 0)),
            "source_selected_fraction": float(source_counts.get(source_id, 0) / total_selected),
            "source_contribution_gain": float(
                source_counts.get(source_id, 0) / total_selected - pool_counts.get(source_id, 0) / total_pool
            ),
        }
        for source_id in sources
    ]
    summary = {
        "available": True,
        "selected_rows": int(len(selected_rows)),
        "pool_rows": int(len(pool_row_map)),
        "sources": contribution,
    }
    save_json(rollout_dir / "source_contribution.json", summary)
    return summary


def _evaluate_rollout_selection(config: GuidedRolloutConfig, pareto_df: pd.DataFrame) -> dict[str, Any]:
    rollout_dir = Path(config.rollout_dir)
    metric_config = MetricConfig(
        dataset_name=config.dataset_name,
        exp_name=rollout_dir.name,
        artifact_dir=rollout_dir.parent,
        shared_artifact_dir=rollout_dir.parent.parent / "shared",
        seed=config.seed,
        holdout_fraction=config.holdout_fraction,
        eval_device=config.eval_device,
        privacy_version="v2",
        density_reference_size=config.density_reference_size,
        nn_device=config.nn_device,
        utility_exact_evaluator=config.utility_exact_evaluator,
        utility_exact_torch_epochs=config.utility_exact_torch_epochs,
        utility_exact_torch_importance_sample_size=0,
    )
    return evaluate_single_selection(
        config=metric_config,
        selection_name="selection_pareto",
        df=pareto_df,
        eval_dir=rollout_dir / "eval",
    )


def run_guided_pareto_rollout(
    config: GuidedRolloutConfig,
    theta: StrategyTheta,
) -> GuidedRolloutResult:
    rollout_dir = ensure_dir(Path(config.rollout_dir))
    use_theta_id = config.theta_id or make_theta_id(theta)
    if use_theta_id != config.theta_id:
        config = GuidedRolloutConfig(**{**asdict(config), "theta_id": use_theta_id})

    core_config = _build_core_config(config, theta)
    state = prepare_context(core_config)
    state = build_cards_and_validate(state, show_progress=progress_enabled(core_config))
    state = initialize_selector_and_pool(state)
    state = build_preselected_valid(state)
    state = compute_global_exact_scores(state)
    state = build_fidelity_ceiling(state)
    pareto_df, pareto_records, pareto_report = _select_guided_pareto(state)
    pareto_df, pareto_records, reward_v2_report = _apply_rollout_reward_candidate_v2(
        config=config,
        state=state,
        pareto_df=pareto_df,
        pareto_records=pareto_records,
    )
    pareto_df, pareto_records, pareto_report = apply_pareto_post_selection_repairs(
        state=state,
        pareto_df=pareto_df,
        pareto_records=pareto_records,
        pareto_report=pareto_report,
    )
    pareto_report = {**pareto_report, "reward_candidate_v2": reward_v2_report}
    search_objectives = _compute_search_objectives(
        selector=state.selector,
        pareto_df=pareto_df,
        pareto_records=pareto_records,
        pareto_report=pareto_report,
    )
    internal_reports = _save_rollout_artifacts(
        config=config,
        theta=theta,
        pareto_df=pareto_df,
        pareto_records=pareto_records,
        pareto_report=pareto_report,
        state=state,
        search_objectives=search_objectives,
    )

    metrics_summary = _evaluate_rollout_selection(config, pareto_df)
    audit_metrics = dict(metrics_summary.get("audit_metrics", {}))
    reward_available = bool(audit_metrics.get("metric_reward_available", False))
    reward = float(audit_metrics.get("metric_reward", 0.0)) if reward_available else 0.0
    guard = build_guard(audit_metrics)
    feedback = build_rollout_feedback(
        theta=theta,
        search_objectives=search_objectives,
        audit_metrics=audit_metrics,
        metric_extras=dict(metrics_summary.get("metric_extras", {})),
        internal_reports=internal_reports,
        column_lookup=_column_lookup(list(pareto_df.columns), list(state.selector.feature_columns)),
    )
    save_json(rollout_dir / "metrics_summary.json", _json_safe(metrics_summary))
    save_json(rollout_dir / "feedback.json", _json_safe(feedback))
    save_json(rollout_dir / "guard.json", _json_safe(guard))
    save_json(
        rollout_dir / "reward.json",
        _json_safe(
            {
                "Q_self": reward,
                "reward": reward,
                "reward_available": reward_available,
                "guard_pass": bool(guard.get("pass", False)),
            }
        ),
    )

    return GuidedRolloutResult(
        theta_id=config.theta_id,
        theta=theta,
        pareto_df=pareto_df,
        pareto_records=pareto_records,
        search_objectives=search_objectives,
        internal_reports=internal_reports,
        artifact_dir=rollout_dir,
        reward=reward,
        reward_available=reward_available,
        audit_metrics=audit_metrics,
        feedback=feedback,
        guard=guard,
        metrics_summary=metrics_summary,
    )
