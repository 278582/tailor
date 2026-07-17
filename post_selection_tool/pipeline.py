from __future__ import annotations

from typing import Any

from .config import CoreSelectionConfig, progress_enabled
from .context import prepare_context
from .core_select import build_core_selections
from .exact_score import compute_global_exact_scores
from .fidelity_ceiling import build_fidelity_ceiling
from .io import save_csv, save_json, save_jsonl
from .logging_utils import configure_logging
from .preselect import build_preselected_valid
from .state import CoreSelectionOutputs, SelectionState
from .validation import build_cards_and_validate, initialize_selector_and_pool


def save_core_outputs(state: SelectionState, outputs: CoreSelectionOutputs) -> None:
    versions_dir = state.paths.versions_dir
    selection_dir = state.paths.selection_dir
    report_dir = state.paths.report_dir

    save_csv(versions_dir / "selection_preselected_valid.csv", outputs.preselected_valid_df)
    save_csv(versions_dir / "selection_preselected_fidelity_ceiling_keep_k.csv", outputs.fidelity_ceiling_df)
    save_csv(versions_dir / "selection_random_full.csv", outputs.random_full_df)
    save_csv(versions_dir / "selection_scalar.csv", outputs.scalar_df)
    save_csv(versions_dir / "selection_pareto.csv", outputs.pareto_df)
    for legacy_filename in (
        "preselected_valid_keep.csv",
        "preselected_fidelity_ceiling_keep_k.csv",
        "random_full_keep.csv",
        "scalarization_keep.csv",
        "pareto_keep.csv",
    ):
        legacy_path = versions_dir / legacy_filename
        if legacy_path.exists():
            legacy_path.unlink()

    save_jsonl(selection_dir / "surrogate_scores.jsonl", state.surrogate_records_all)
    save_jsonl(selection_dir / "preselected_surrogates.jsonl", state.preselected_surrogates)
    save_jsonl(selection_dir / "exact_scores.jsonl", state.global_exact_records)
    save_jsonl(selection_dir / "utility_static_scores.jsonl", state.utility_proxy_bundle.get("static_scores", []))
    save_jsonl(selection_dir / "utility_proxy_scores.jsonl", state.utility_proxy_bundle.get("proxy_scores", []))

    save_json(selection_dir / "baselines.json", state.global_baselines)
    save_json(selection_dir / "preselect_gate.json", state.preselect_gate)
    save_json(selection_dir / "fidelity_ceiling_initial_report.json", state.initial_fidelity_ceiling_report)
    save_json(selection_dir / "fidelity_ceiling_report.json", state.fidelity_ceiling_report)
    save_json(selection_dir / "theta_guidance.json", state.config.theta_guidance_report or {"enabled": False})
    save_json(
        selection_dir / "high_cardinality_manifest.json",
        state.selector.high_cardinality_compressor.to_manifest() if state.selector is not None else {},
    )
    save_json(selection_dir / "random_full_report.json", outputs.reports.get("random_full", {}))
    save_json(selection_dir / "scalarization_report.json", outputs.reports.get("scalar", {}))
    save_json(selection_dir / "pareto_report.json", outputs.reports.get("pareto", {}))
    save_json(report_dir / "timing_report.json", state.timing_report)
    for legacy_filename in (
        "dataset_hv_reward_reports.json",
        "scalar_hv_reward_report.json",
        "pareto_hv_reward_report.json",
        "dataset_hv_reports.json",
        "dataset_mo_reward_reports.json",
        "random_full_hv_report.json",
        "scalar_hv_report.json",
        "pareto_hv_report.json",
        "global_hv_report.json",
        "random_full_mo_reward_report.json",
        "scalar_mo_reward_report.json",
        "pareto_mo_reward_report.json",
        "global_mo_reward_report.json",
        "pareto_hvc_scores.csv",
        "pareto_hvc_report.json",
    ):
        legacy_path = selection_dir / legacy_filename
        if legacy_path.exists():
            legacy_path.unlink()
    save_json(
        selection_dir / "utility_proxy_manifest.json",
        {
            **state.utility_proxy_bundle.get("manifest", {}),
            "pre_ceiling_static": state.utility_proxy_bundle.get("pre_ceiling_static", {}).get("manifest", {}),
            "merge_report": state.utility_proxy_merge_report,
        },
    )
    save_json(report_dir / "core_selection_summary.json", build_core_summary(state, outputs))


def build_core_summary(state: SelectionState, outputs: CoreSelectionOutputs) -> dict[str, Any]:
    config = state.config
    return {
        "source": config.source,
        "synthetic_csv": str(state.synthetic_csv),
        "dataset_name": config.dataset_name,
        "train_rows": int(len(state.train_df)),
        "holdout_rows": int(len(state.holdout_df)),
        "test_rows": int(len(state.test_df)),
        "eval_holdout_policy": {
            "selection_eval_holdout": "eval_holdout.csv",
            "holdout_strategy": getattr(state.dataset_ctx, "holdout_strategy", None),
            "post_selection_proxy": True,
            "method_note": (
                "Selection uses the file-backed eval_holdout split for proxy privacy/utility calibration; "
                "metric_tool reports final post-selection metrics on eval_test.csv."
            ),
        },
        "raw_rows": int(len(state.synthetic_df)),
        "valid_rows": int(len(state.valid_df)) if state.valid_df is not None else 0,
        "candidate_pool_rows": int(len(state.pool_records)),
        "preselected_rows": int(len(state.preselected_valid)),
        "requested_keep_k": int(config.keep_k),
        "effective_keep_k": int(state.effective_keep_k),
        "requested_preselect_target": int(state.requested_preselect_target),
        "effective_preselect_target": int(state.effective_preselect_target),
        "d_cur_rows": int(len(state.d_cur_df)) if state.d_cur_df is not None else 0,
        "fidelity_ceiling_rows": int(len(state.fidelity_ceiling_records)),
        "random_full_rows": int(len(outputs.random_full_df)),
        "scalar_rows": int(len(outputs.scalar_df)),
        "pareto_rows": int(len(outputs.pareto_df)),
        "preselect_status": state.preselect_status,
        "preselect_gate_thresholds": {
            "fidelity_max_drop": float(config.preselect_gate_fidelity_max_drop),
            "trend_max_drop": float(config.preselect_gate_trend_max_drop),
            "dcr_min_gain": float(config.preselect_gate_dcr_min_gain),
            "candidate_vs_baseline_max_drop": float(config.preselect_gate_candidate_vs_baseline_max_drop),
            "candidate_vs_baseline_min_dcr_gain": float(config.preselect_gate_candidate_vs_baseline_min_dcr_gain),
        },
        "floor_reference": state.floor_reference,
        "theta_guidance": config.theta_guidance_report or {"enabled": False},
        "llm_mcts_theta_interface": {
            "available": True,
            "enabled": bool((config.theta_guidance_report or {}).get("enabled", False)),
            "inputs": ["--theta-json", "--theta-source", "--theta-mcts-dir"],
        },
        "utility_proxy": {
            **state.utility_proxy_bundle.get("manifest", {}),
            "dynamic_utility": {"available": False, "reason": "removed_from_main_chain"},
        },
        "timing": state.timing_report,
        "core_outputs": {
            "preselected_valid": "versions/selection_preselected_valid.csv",
            "preselected_fidelity_ceiling_keep_k": "versions/selection_preselected_fidelity_ceiling_keep_k.csv",
            "random_full": "versions/selection_random_full.csv",
            "scalar": "versions/selection_scalar.csv",
            "pareto": "versions/selection_pareto.csv",
        },
    }


def run_core_selection(config: CoreSelectionConfig) -> tuple[SelectionState, CoreSelectionOutputs]:
    state = prepare_context(config)
    configure_logging(
        log_file=config.log_file if config.log_file is not None else state.paths.report_dir / "post_selection_tool.log",
        quiet=config.quiet,
    )
    state = build_cards_and_validate(state, show_progress=progress_enabled(config))
    state = initialize_selector_and_pool(state)
    state = build_preselected_valid(state)
    state = compute_global_exact_scores(state)
    state = build_fidelity_ceiling(state)
    outputs = build_core_selections(state)
    save_core_outputs(state, outputs)
    return state, outputs
