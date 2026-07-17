from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from postprocess.tabdiff_eval import TabDiffSelectionEvaluator

from .config import progress_enabled
from .context import resolve_eval_device
from .io import ensure_dir, records_to_df, save_csv
from .preselect_dcr_balance import rebalance_preselected_for_dcr_surrogate
from .state import SelectionState


def selection_delta(base_metrics: dict[str, Any], target_metrics: dict[str, Any]) -> dict[str, Any]:
    if not base_metrics or not target_metrics:
        return {
            "available": False,
            "fidelity_drop": None,
            "trend_drop": None,
            "dcr_gain": None,
            "privacy_gain": None,
        }

    base_dcr = base_metrics.get("dcr")
    target_dcr = target_metrics.get("dcr")
    if base_dcr is not None and target_dcr is not None:
        dcr_gain = float(base_dcr) - float(target_dcr)
    elif base_metrics.get("dcr_privacy") is not None and target_metrics.get("dcr_privacy") is not None:
        dcr_gain = float(target_metrics["dcr_privacy"]) - float(base_metrics["dcr_privacy"])
    else:
        dcr_gain = None

    privacy_gain = None
    if base_metrics.get("privacy") is not None and target_metrics.get("privacy") is not None:
        privacy_gain = float(target_metrics["privacy"]) - float(base_metrics["privacy"])

    return {
        "available": True,
        "fidelity_drop": float(base_metrics.get("fidelity", 0.0) - target_metrics.get("fidelity", 0.0)),
        "trend_drop": float(base_metrics.get("trend", 0.0) - target_metrics.get("trend", 0.0)),
        "dcr_gain": dcr_gain,
        "privacy_gain": privacy_gain,
    }


def _build_gate_evaluator(
    state: SelectionState,
    eval_device: str,
) -> tuple[TabDiffSelectionEvaluator, bool]:
    selector = state.selector
    if selector is None:
        raise RuntimeError("selector is required for gate evaluator")
    compressor = getattr(selector, "high_cardinality_compressor", None)
    use_compressed = bool(compressor is not None and getattr(compressor, "active_columns", []))
    if not use_compressed:
        return TabDiffSelectionEvaluator(
            dataset_name=state.config.dataset_name,
            device=eval_device,
            metric_list=["density", "dcr"],
            real_data_path=state.paths.input_dir / "eval_train.csv",
            test_data_path=state.paths.input_dir / "eval_test.csv",
            val_data_path=state.paths.input_dir / "eval_holdout.csv",
        ), False

    shared_root = getattr(state.config, "shared_artifact_dir", None)
    compressed_dir = ensure_dir(
        Path(shared_root) / "input" / "preselect_gate_compressed"
        if shared_root is not None
        else state.paths.selection_dir / "preselect_gate_compressed"
    )
    real_df = compressor.transform_df(pd.read_csv(state.paths.input_dir / "eval_train.csv"))
    test_df = compressor.transform_df(pd.read_csv(state.paths.input_dir / "eval_test.csv"))
    holdout_df = compressor.transform_df(pd.read_csv(state.paths.input_dir / "eval_holdout.csv"))
    real_path = compressed_dir / "eval_train.csv"
    test_path = compressed_dir / "eval_test.csv"
    holdout_path = compressed_dir / "eval_holdout.csv"
    save_csv(real_path, real_df)
    save_csv(test_path, test_df)
    save_csv(holdout_path, holdout_df)
    return TabDiffSelectionEvaluator(
        dataset_name=state.config.dataset_name,
        device=eval_device,
        metric_list=["density", "dcr"],
        real_data_path=real_path,
        test_data_path=test_path,
        val_data_path=holdout_path,
    ), True


def subset_gate_metrics(
    selector: Any,
    evaluator: TabDiffSelectionEvaluator,
    df: pd.DataFrame,
    *,
    use_compressed_view: bool = False,
) -> dict[str, Any]:
    eval_df = selector.high_cardinality_compressor.transform_df(df) if use_compressed_view else df
    metrics, _ = evaluator.evaluate(eval_df)
    return {
        "rows": int(len(df)),
        "fidelity": selector.compute_dataset_fidelity(df),
        "trend": float(metrics.get("density/Trend", 0.0)),
        "dcr": float(metrics.get("dcr", 0.0)) if "dcr" in metrics else None,
        "dcr_privacy": (1.0 - float(metrics["dcr"])) if "dcr" in metrics else None,
        "metric_view": "high_cardinality_compressed" if use_compressed_view else "raw",
        "privacy": selector.compute_dataset_privacy(df),
    }


def build_preselect_gate_report(
    raw_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    *,
    candidate_mode: str = "candidate_preselect",
    baseline_mode: str = "baseline_preselect",
    fidelity_max_drop: float = 0.01,
    trend_max_drop: float = 0.01,
    dcr_min_gain: float = 0.02,
    candidate_vs_baseline_max_drop: float = 0.001,
    candidate_vs_baseline_min_dcr_gain: float = 0.002,
) -> dict[str, Any]:
    candidate_delta = selection_delta(raw_metrics, candidate_metrics)
    baseline_delta = selection_delta(raw_metrics, baseline_metrics)
    candidate_vs_baseline = selection_delta(baseline_metrics, candidate_metrics)

    def _passes(delta: dict[str, Any]) -> bool:
        return bool(
            delta.get("available")
            and delta.get("fidelity_drop") is not None
            and delta.get("trend_drop") is not None
            and delta.get("dcr_gain") is not None
            and float(delta["fidelity_drop"]) <= float(fidelity_max_drop)
            and float(delta["trend_drop"]) <= float(trend_max_drop)
            and float(delta["dcr_gain"]) >= float(dcr_min_gain)
        )

    candidate_pass = _passes(candidate_delta)
    baseline_pass = _passes(baseline_delta)
    candidate_beats_baseline = bool(
        candidate_vs_baseline.get("available")
        and candidate_vs_baseline.get("fidelity_drop") is not None
        and candidate_vs_baseline.get("trend_drop") is not None
        and candidate_vs_baseline.get("dcr_gain") is not None
        and float(candidate_vs_baseline["fidelity_drop"]) <= float(candidate_vs_baseline_max_drop)
        and float(candidate_vs_baseline["trend_drop"]) <= float(candidate_vs_baseline_max_drop)
        and float(candidate_vs_baseline["dcr_gain"]) >= float(candidate_vs_baseline_min_dcr_gain)
    )

    if candidate_pass and candidate_beats_baseline:
        selected_source = "candidate_preselect"
        selected_mode = candidate_mode
    else:
        selected_source = "baseline_preselect"
        selected_mode = baseline_mode

    return {
        "thresholds": {
            "fidelity_max_drop": float(fidelity_max_drop),
            "trend_max_drop": float(trend_max_drop),
            "dcr_min_gain": float(dcr_min_gain),
            "candidate_vs_baseline_fidelity_max_drop": float(candidate_vs_baseline_max_drop),
            "candidate_vs_baseline_trend_max_drop": float(candidate_vs_baseline_max_drop),
            "candidate_vs_baseline_dcr_min_gain": float(candidate_vs_baseline_min_dcr_gain),
        },
        "raw_reference": raw_metrics,
        "candidate": {
            "mode": candidate_mode,
            "metrics": candidate_metrics,
            "delta_vs_raw": candidate_delta,
            "delta_vs_baseline": candidate_vs_baseline,
            "pass": bool(candidate_pass),
            "beats_baseline": bool(candidate_beats_baseline),
        },
        "baseline": {
            "mode": baseline_mode,
            "metrics": baseline_metrics,
            "delta_vs_raw": baseline_delta,
            "pass": bool(baseline_pass),
        },
        "selected_source": selected_source,
        "selected_mode": selected_mode,
        "selected_pass": bool(candidate_pass if selected_source == "candidate_preselect" else baseline_pass),
        "fallback_applied": bool(selected_source != "candidate_preselect"),
        "candidate_vs_baseline_thresholds": {
            "fidelity_max_drop": float(candidate_vs_baseline_max_drop),
            "trend_max_drop": float(candidate_vs_baseline_max_drop),
            "dcr_min_gain": float(candidate_vs_baseline_min_dcr_gain),
        },
    }


def build_preselected_valid(state: SelectionState) -> SelectionState:
    if state.selector is None or state.pool_df is None:
        raise RuntimeError("initialize_selector_and_pool must run before build_preselected_valid")

    config = state.config
    selector = state.selector
    progress = progress_enabled(config)
    state.desired_keep_k = min(config.keep_k, len(state.pool_records))
    state.requested_preselect_target = min(len(state.pool_records), max(config.preselect_target, state.desired_keep_k))
    state.surrogate_records_all = selector.compute_surrogates(
        state.pool_df,
        show_progress=progress,
        progress_desc="surrogate scoring",
        candidate_ids=np.fromiter(
            (int(record.get("candidate_id", idx)) for idx, record in enumerate(state.pool_records)),
            dtype=int,
            count=len(state.pool_records),
        ),
    )

    preselect_should_run = (
        state.requested_preselect_target < len(state.pool_records)
        and len(state.pool_records) > state.desired_keep_k
    )
    if preselect_should_run:
        baseline_surrogates = [dict(record) for record in state.surrogate_records_all]
        candidate_surrogates = [dict(record) for record in state.surrogate_records_all]
        baseline_valid, baseline_sur = selector.dual_median_filter_baseline(
            valid_records=state.pool_records,
            surrogate_records=baseline_surrogates,
            target_preselect=state.requested_preselect_target,
            show_progress=progress,
            progress_desc="preselect baseline",
        )
        baseline_preselect_report = dict(selector.last_preselect_report)
        baseline_candidate_ids = np.asarray(
            [int(record.get("candidate_id", idx)) for idx, record in enumerate(baseline_sur)],
            dtype=int,
        )
        preselected_valid_candidate, preselected_sur_candidate = selector.dual_median_filter(
            valid_records=state.pool_records,
            surrogate_records=candidate_surrogates,
            target_preselect=state.requested_preselect_target,
            anchor_candidate_ids=baseline_candidate_ids,
            show_progress=progress,
            progress_desc="preselect candidate",
        )
        candidate_preselect_report = dict(selector.last_preselect_report)

        eval_device = resolve_eval_device(config.eval_device)
        gate_evaluator, use_compressed_view = _build_gate_evaluator(state, eval_device)
        state.evaluator = gate_evaluator
        raw_reference_df, _, _ = selector.select_keep_random(
            candidate_records=state.pool_records,
            keep_k=state.requested_preselect_target,
            rng_seed=config.seed,
        )
        candidate_preselected_df = records_to_df(preselected_valid_candidate, selector.column_order)
        baseline_preselected_df = records_to_df(baseline_valid, selector.column_order)
        preselect_gate = build_preselect_gate_report(
            raw_metrics=subset_gate_metrics(
                selector,
                gate_evaluator,
                raw_reference_df,
                use_compressed_view=use_compressed_view,
            ),
            candidate_metrics=subset_gate_metrics(
                selector,
                gate_evaluator,
                candidate_preselected_df,
                use_compressed_view=use_compressed_view,
            ),
            baseline_metrics=subset_gate_metrics(
                selector,
                gate_evaluator,
                baseline_preselected_df,
                use_compressed_view=use_compressed_view,
            ),
            candidate_mode=str(candidate_preselect_report.get("mode", "candidate_preselect")),
            baseline_mode=str(baseline_preselect_report.get("mode", "baseline_preselect")),
            fidelity_max_drop=config.preselect_gate_fidelity_max_drop,
            trend_max_drop=config.preselect_gate_trend_max_drop,
            dcr_min_gain=config.preselect_gate_dcr_min_gain,
            candidate_vs_baseline_max_drop=config.preselect_gate_candidate_vs_baseline_max_drop,
            candidate_vs_baseline_min_dcr_gain=config.preselect_gate_candidate_vs_baseline_min_dcr_gain,
        )
        preselect_gate["metric_view"] = "high_cardinality_compressed" if use_compressed_view else "raw"
        preselect_gate["candidate"]["preselect_report"] = candidate_preselect_report
        preselect_gate["baseline"]["preselect_report"] = baseline_preselect_report

        if preselect_gate["selected_source"] == "candidate_preselect":
            state.preselected_valid = preselected_valid_candidate
            state.preselected_surrogates = preselected_sur_candidate
        else:
            state.preselected_valid = baseline_valid
            state.preselected_surrogates = baseline_sur
        state.preselect_gate = preselect_gate
        state.preselect_status = {
            "applied": True,
            "mode": str(preselect_gate["selected_mode"]),
            "reason": None if not preselect_gate["fallback_applied"] else "preselect_gate_fallback_to_baseline",
            "rows_before": len(state.pool_records),
            "rows_after": len(state.preselected_valid),
            "selected_source": preselect_gate["selected_source"],
            "fallback_applied": bool(preselect_gate["fallback_applied"]),
            "selected_pass": bool(preselect_gate["selected_pass"]),
        }
    else:
        state.preselected_valid = state.pool_records.copy()
        state.preselected_surrogates = state.surrogate_records_all.copy()
        state.preselect_gate = {
            "skipped": True,
            "reason": (
                "target_not_reductive"
                if state.requested_preselect_target >= len(state.pool_records)
                else "pool_too_close_to_keep_k"
            ),
            "selected_source": "full_pool",
            "selected_mode": "skipped_full_pool",
            "selected_pass": None,
            "fallback_applied": False,
        }
        state.preselect_status = {
            "applied": False,
            "mode": "skipped_full_pool",
            "reason": state.preselect_gate["reason"],
            "rows_before": len(state.pool_records),
            "rows_after": len(state.preselected_valid),
        }

    if len(state.preselected_valid) < state.desired_keep_k:
        state.preselected_valid = state.pool_records.copy()
        state.preselected_surrogates = state.surrogate_records_all.copy()
        state.preselect_gate = {
            **state.preselect_gate,
            "selected_source": "full_pool",
            "selected_mode": "fallback_full_pool",
            "selected_pass": False,
            "fallback_applied": True,
            "reason": "selected_preselect_below_keep_k",
        }
        state.preselect_status = {
            "applied": False,
            "mode": "fallback_full_pool",
            "reason": "dual_median_filter_below_keep_k",
            "rows_before": len(state.pool_records),
            "rows_after": len(state.preselected_valid),
            "selected_source": "full_pool",
            "fallback_applied": True,
            "selected_pass": False,
        }

    if config.preselect_dcr_balance_enabled and len(state.preselected_valid) > state.desired_keep_k:
        state.preselected_valid, state.preselected_surrogates, dcr_balance_report = (
            rebalance_preselected_for_dcr_surrogate(
                pool_records=state.pool_records,
                surrogate_records=state.surrogate_records_all,
                selected_records=state.preselected_valid,
                selected_surrogates=state.preselected_surrogates,
                target_fraction=config.preselect_dcr_balance_target_fraction,
                max_exchange_fraction=config.preselect_dcr_balance_max_exchange_fraction,
            )
        )
    else:
        dcr_balance_report = {
            "enabled": bool(config.preselect_dcr_balance_enabled),
            "version": "preselect_dcr_balance_v3",
            "applied": False,
            "reason": "disabled_or_not_reductive",
        }
    state.preselect_gate = {
        **state.preselect_gate,
        "dcr_balance_repair": dcr_balance_report,
    }
    state.preselect_status = {
        **state.preselect_status,
        "dcr_balance_repair_applied": bool(dcr_balance_report.get("applied", False)),
    }

    state.effective_preselect_target = len(state.preselected_valid)
    state.effective_keep_k = min(state.desired_keep_k, len(state.preselected_valid))
    if state.effective_keep_k <= 0:
        raise RuntimeError("effective_keep_k <= 0. Increase sample size or reduce d_cur_size / keep_k.")
    return state
