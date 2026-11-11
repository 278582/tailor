"""Opt-in Greedy rollout using frozen references and explicit evaluation settings.

The existing llm_mcts_tool rollout and all selection/repair implementations are
unchanged. Only Greedy requests with extended runtime settings use this adapter.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from llm_mcts_tool import rollout as legacy
from metric_tool.config import MetricConfig
from metric_tool.pipeline import evaluate_single_selection
from post_selection_tool.io import save_json


def rollout_config(config, s_node, theta_node, rollouts_dir):
    """Build exactly the existing guided-selection options for one Theta."""
    return legacy.GuidedRolloutConfig(
        theta_id=theta_node.theta_id, dataset_name=config.dataset_name,
        exp_name=f"{config.exp_name}_{s_node.s_id}", artifact_dir=rollouts_dir,
        synthetic_csv=s_node.synthetic_csv, seed=config.seed, keep_k=config.keep_k,
        preselect_target=config.preselect_target, d_cur_size=config.d_cur_size,
        max_theta_pairs=config.max_theta_pairs,
        rollout_dir=Path(rollouts_dir) / f"{s_node.s_id}_{theta_node.theta_id}",
        d_cur_source="synthetic", holdout_fraction=config.holdout_fraction,
        source=config.single_source, eval_device=config.eval_device,
        nn_device=config.nn_device, utility_exact_evaluator=config.utility_exact_evaluator,
        utility_exact_torch_epochs=config.utility_exact_torch_epochs,
        utility_diag_sample_size=config.utility_diag_sample_size,
        density_reference_size=config.density_reference_size,
        save_validation_records=config.save_validation_records,
        save_internal_records=config.save_rollout_internal_records,
        disable_progress=config.disable_progress, allow_target_in_fidelity_columns=True,
        allow_target_in_privacy_columns=True, privacy_encoding_column_mode="privacy_columns",
        synthetic_row_map=s_node.synthetic_row_map,
        **{name: getattr(config, name) for name in (
            "rollout_direct_dcr_repair_enabled", "rollout_direct_dcr_target_margin",
            "rollout_direct_dcr_max_swap_fraction", "rollout_direct_dcr_candidate_neighbors",
            "rollout_direct_dcr_min_pair_utility_gain", "rollout_direct_dcr_fallback_min_pair_utility_gain",
            "rollout_reward_candidate_v2_enabled", "rollout_reward_candidate_v2_max_swap_fraction",
            "rollout_reward_candidate_v2_max_candidate_sizes", "rollout_reward_candidate_v2_min_proxy_delta",
            "rollout_reward_candidate_v2_fidelity_floor_eps", "rollout_reward_candidate_v2_utility_floor_eps",
        )},
    )


def core_config(runtime, guided, theta):
    # Existing full-train/hold row signals and column scopes remain intact.
    return replace(legacy._build_core_config(guided, theta),
                   reference_split_dir=runtime.reference_split_dir)


def metric_config(runtime, guided):
    folder = Path(guided.rollout_dir)
    return MetricConfig(
        dataset_name=guided.dataset_name, exp_name=folder.name,
        artifact_dir=folder.parent, shared_artifact_dir=folder.parent.parent / "shared",
        reference_split_dir=runtime.reference_split_dir,
        seed=guided.seed if runtime.evaluation_seed is None else runtime.evaluation_seed,
        holdout_fraction=guided.holdout_fraction, eval_device=guided.eval_device,
        privacy_version="v2", nn_device=guided.nn_device,
        density_reference_size=(guided.density_reference_size if runtime.evaluation_density_reference_size is None
                                else runtime.evaluation_density_reference_size),
        utility_exact_evaluator=guided.utility_exact_evaluator,
        utility_exact_torch_epochs=guided.utility_exact_torch_epochs,
        utility_exact_torch_importance_sample_size=0,
        dcr_align_mode=runtime.evaluation_dcr_align_mode,
        dcr_repeats=runtime.evaluation_dcr_repeats,
        dcr_seed=runtime.evaluation_dcr_seed,
        enable_hold_dcr=runtime.evaluation_enable_hold_dcr,
    )


def rollout_node(*, config, s_node, theta_node, rollouts_dir):
    guided = rollout_config(config, s_node, theta_node, rollouts_dir)
    folder = legacy.ensure_dir(Path(guided.rollout_dir))
    core = core_config(config, guided, theta_node.theta)
    state = legacy.prepare_context(core)
    state = legacy.build_cards_and_validate(state, show_progress=legacy.progress_enabled(core))
    state = legacy.initialize_selector_and_pool(state)
    state = legacy.build_preselected_valid(state)
    state = legacy.compute_global_exact_scores(state)
    state = legacy.build_fidelity_ceiling(state)
    frame, records, report = legacy._select_guided_pareto(state)
    frame, records, reward_v2 = legacy._apply_rollout_reward_candidate_v2(
        config=guided, state=state, pareto_df=frame, pareto_records=records)
    frame, records, report = legacy.apply_pareto_post_selection_repairs(
        state=state, pareto_df=frame, pareto_records=records, pareto_report=report)
    report = {**report, "reward_candidate_v2": reward_v2}
    legacy.log_pipeline_stages(state.timing_report)
    objectives = legacy._compute_search_objectives(
        selector=state.selector, pareto_df=frame, pareto_records=records, pareto_report=report)
    internal = legacy._save_rollout_artifacts(
        config=guided, theta=theta_node.theta, pareto_df=frame, pareto_records=records,
        pareto_report=report, state=state, search_objectives=objectives)
    metric = metric_config(config, guided)
    save_json(folder / "protocol_manifest.json", {
        "reference_split_dir": str(config.reference_split_dir),
        "evaluation": legacy._json_safe(metric.__dict__),
        "selection_dcr_align_mode": core.dcr_align_mode,
        "repair_signal": "existing legacy_v19 full-train/hold row signals in configured column scopes",
        "repair_reference": str(state.dataset_ctx.holdout_source_path),
        "test_used_for_outer_search_reward": True,
        "shared_code_modified": False,
    })
    metrics = evaluate_single_selection(config=metric, selection_name="selection_pareto",
                                        df=frame, eval_dir=folder / "eval")
    if metric.dcr_align_mode == "query_batches_full_train_v1":
        if metrics.get("protocol_id") != metric.dcr_align_mode:
            raise RuntimeError("Greedy evaluation did not execute the requested query-batch protocol")
        if metrics["dcr_protocol"]["rounds"] != metric.dcr_repeats:
            raise RuntimeError("Greedy DCR repeat mismatch")
    audit = dict(metrics.get("audit_metrics", {}))
    available = bool(audit.get("metric_reward_available", False))
    reward = float(audit.get("metric_reward", 0.0)) if available else 0.0
    guard = legacy.build_guard(audit)
    feedback = legacy.build_rollout_feedback(
        theta=theta_node.theta, search_objectives=objectives, audit_metrics=audit,
        metric_extras=dict(metrics.get("metric_extras", {})), internal_reports=internal,
        column_lookup=legacy._column_lookup(list(frame.columns), list(state.selector.feature_columns)))
    for name, value in {
        "metrics_summary": metrics, "feedback": feedback, "guard": guard,
        "reward": {"Q_self": reward, "reward": reward, "reward_available": available,
                   "guard_pass": bool(guard.get("pass", False))},
    }.items():
        save_json(folder / f"{name}.json", legacy._json_safe(value))
    return legacy.GuidedRolloutResult(
        theta_id=guided.theta_id, theta=theta_node.theta, pareto_df=frame,
        pareto_records=records, search_objectives=objectives, internal_reports=internal,
        artifact_dir=folder, reward=reward, reward_available=available,
        audit_metrics=audit, feedback=feedback, guard=guard, metrics_summary=metrics)
