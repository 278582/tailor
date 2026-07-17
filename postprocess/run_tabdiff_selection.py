from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data_io import ensure_dir, load_csv, save_csv, save_json, save_jsonl, set_seed
from .selection_runtime import (
    _build_preselect_gate_report,
    _build_selection_gate_report,
    _build_direction_family,
    _compare_families,
    _df_to_candidate_records,
    _evaluate_selection,
    _progress,
    _progress_write,
    _records_to_df,
    _resolve_eval_device,
    _resolve_nn_device,
    _rerank_pareto_finalists_on_search_holdout,
    _run_streaming_archive,
    _save_family_csvs,
    _save_selection_csvs,
    _selection_name,
    _subset_gate_metrics,
    _subset_metrics,
)
from .tabdiff_utils import find_latest_tabdiff_sample
from .utility_proxy import build_static_balanced_utility_scores, build_utility_proxy_scores

from .core import ParetoSelector, TabDiffEvaluator, TabularValidator, build_cards_bundle
from .tabdiff_protocol import normalize_tabdiff_dataframe_columns, resolve_tabdiff_selection_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generator-agnostic postprocessing on TabDiff synthetic samples using the clean postprocess path."
    )
    parser.add_argument("--synthetic-csv", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--source", type=str, default="tabdiff")
    parser.add_argument("--keep-k", type=int, required=True)
    parser.add_argument("--preselect-target", type=int, required=True)
    parser.add_argument("--d-cur-size", type=int, default=1000)
    parser.add_argument("--selection-chunk-size", type=int, default=4096)
    parser.add_argument("--archive-budget-scale", type=float, default=1.2)
    parser.add_argument("--local-keep-factor", type=float, default=3.0)
    parser.add_argument("--lambda-penalty", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--privacy-version", choices=["v1", "v2", "v3"], default="v2")
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--jsd-epsilon", type=float, default=0.15)
    parser.add_argument("--rare-threshold", type=float, default=0.05)
    parser.add_argument("--final-fidelity-floor-eps", type=float, default=0.01)
    parser.add_argument("--final-trend-floor-eps", type=float, default=0.01)
    parser.add_argument("--fidelity-ceiling-utility-weight", type=float, default=0.04)
    parser.add_argument("--fidelity-ceiling-refine-utility-weight", type=float, default=0.15)
    parser.add_argument("--fidelity-ceiling-second-pass-utility-weight", type=float, default=0.08)
    parser.add_argument("--fidelity-ceiling-second-pass-refine-utility-weight", type=float, default=0.20)
    parser.add_argument("--pareto-rerank-utility-switch-min", type=float, default=0.002)
    parser.add_argument("--pareto-rerank-privacy-switch-min", type=float, default=0.005)
    parser.add_argument("--pareto-floor-mode", choices=["hard", "soft"], default="soft")
    parser.add_argument("--pareto-soft-fidelity-floor-eps", type=float, default=0.02)
    parser.add_argument("--pareto-soft-trend-floor-eps", type=float, default=0.02)
    parser.add_argument("--pareto-soft-privacy-floor-eps", type=float, default=0.005)
    parser.add_argument("--pareto-soft-utility-floor-eps", type=float, default=0.005)
    parser.add_argument("--pareto-soft-min-score-delta", type=float, default=0.0)
    parser.add_argument("--density-reference-size", type=int, default=5000)
    parser.add_argument("--nn-device", type=str, default="auto")
    parser.add_argument("--nn-query-batch-size", type=int, default=2048)
    parser.add_argument("--nn-reference-chunk-size", type=int, default=8192)
    parser.add_argument("--eval-device", type=str, default="auto")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument(
        "--skip-family-eval",
        action="store_true",
        help="Skip 4D direction family evaluation after the core selection metrics are written.",
    )
    return parser.parse_args()


def _resolve_synthetic_csv(args: argparse.Namespace) -> Path:
    if args.synthetic_csv is not None:
        return args.synthetic_csv
    return find_latest_tabdiff_sample(dataset_name=args.dataset_name, exp_name=args.exp_name)


def _preselect_objective_manifest(selected_mode: str) -> tuple[str, dict[str, object]]:
    if selected_mode == "two_stage_band_quota_v2":
        return (
            "density_normalized_nn_distance_v2_band_limited_tiebreak",
            {
                "type": "two_stage_band_quota_v2",
                "stage_a": {
                    "components": [
                        "1d_train_clipped_quota_alignment",
                        "2d_train_clipped_quota_alignment",
                        "fidelity_safe_band_score",
                    ],
                    "privacy_in_primary_score": False,
                    "target_mode": "train_clipped_by_availability",
                },
                "stage_b": {
                    "components": [
                        "1d_band_empirical_quota_alignment",
                        "2d_band_empirical_quota_alignment",
                        "fidelity_safe_stage_b_score",
                        "weak_privacy_tiebreak",
                    ],
                    "target_source": "fidelity_safe_band_empirical",
                    "band_target_scale": 1.4,
                    "privacy_weight_max": 0.05,
                },
                "support_diagnostics": ["1d_train_support", "2d_graph_support", "density_normalized_nn_distance"],
            },
        )
    if selected_mode == "three_objective_preselect_v3":
        return (
            "density_normalized_nn_distance_v2_secondary",
            {
                "type": "three_objective_preselect_v3",
                "components": ["1d_empirical_quota_alignment", "2d_empirical_quota_alignment", "privacy_tiebreak"],
                "support_diagnostics": ["1d_train_support", "2d_graph_support", "density_normalized_nn_distance"],
            },
        )
    return (
        "not_applied",
        {
            "type": selected_mode,
            "components": [],
            "support_diagnostics": [],
        },
    )


def _attach_utility_proxy_fields(
    exact_records: list[dict[str, object]],
    proxy_scores: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    proxy_by_id = {int(record["candidate_id"]): record for record in proxy_scores}
    merged_records: list[dict[str, object]] = []
    matched_rows = 0
    missing_rows = 0
    for idx, record in enumerate(exact_records):
        candidate_id = int(record.get("candidate_id", idx))
        proxy = proxy_by_id.get(candidate_id)
        merged = dict(record)
        if proxy is None:
            missing_rows += 1
            merged["pareto_util_proxy_obj"] = 0.0
            merged["utility_proxy_static"] = 0.0
            merged["utility_proxy_dynamic"] = 0.0
            merged["utility_proxy_total"] = 0.0
            merged["utility_proxy_static_norm"] = 0.0
            merged["utility_proxy_dynamic_norm"] = 0.0
            merged["utility_proxy_static_raw"] = 0.0
            merged["utility_proxy_static_group_rank"] = 0.0
            merged["utility_proxy_density_weight"] = 0.0
            merged["utility_proxy_coverage_gain"] = 0.0
            merged["utility_proxy_gate_stratum"] = -1
            merged["utility_proxy_target_label"] = None
            merged["utility_anchor_member"] = False
        else:
            matched_rows += 1
            merged["pareto_util_proxy_obj"] = float(proxy.get("u_proxy", 0.0))
            merged["utility_proxy_static"] = float(proxy.get("u_static", 0.0))
            merged["utility_proxy_dynamic"] = float(proxy.get("u_dynamic", 0.0))
            merged["utility_proxy_total"] = float(proxy.get("u_proxy", 0.0))
            merged["utility_proxy_static_norm"] = float(proxy.get("u_static_norm", 0.0))
            merged["utility_proxy_dynamic_norm"] = float(proxy.get("u_dynamic_norm", 0.0))
            merged["utility_proxy_static_raw"] = float(proxy.get("u_static_raw", proxy.get("u_static", 0.0)))
            merged["utility_proxy_static_group_rank"] = float(proxy.get("u_static_group_rank", 0.0))
            merged["utility_proxy_density_weight"] = float(proxy.get("density_weight", 0.0))
            merged["utility_proxy_coverage_gain"] = float(proxy.get("coverage_gain", 0.0))
            merged["utility_proxy_gate_stratum"] = int(proxy.get("gate_stratum", -1))
            merged["utility_proxy_target_label"] = proxy.get("target_label")
            merged["utility_anchor_member"] = bool(proxy.get("is_anchor_member", False))
        merged_records.append(merged)
    return merged_records, {
        "matched_rows": matched_rows,
        "missing_rows": missing_rows,
        "proxy_rows": len(proxy_scores),
        "exact_rows": len(exact_records),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    progress_enabled = not args.disable_progress
    overall = _progress(total=13, desc="run_tabdiff_selection", dynamic_ncols=True, disable=not progress_enabled)
    eval_device = _resolve_eval_device(args.eval_device)
    nn_device = _resolve_nn_device(args.nn_device, eval_device)

    dataset_ctx = resolve_tabdiff_selection_context(
        dataset_name=args.dataset_name,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
    )
    synthetic_csv = _resolve_synthetic_csv(args)
    artifact_root = dataset_ctx.artifact_root if args.artifact_dir is None else args.artifact_dir
    artifact_dir = ensure_dir(Path(artifact_root) / args.exp_name)
    input_dir = ensure_dir(artifact_dir / "input")
    cards_dir = ensure_dir(artifact_dir / "cards")
    validation_dir = ensure_dir(artifact_dir / "validation")
    selection_dir = ensure_dir(artifact_dir / "selection")
    versions_dir = ensure_dir(artifact_dir / "versions")
    eval_dir = ensure_dir(artifact_dir / "eval")
    report_dir = ensure_dir(artifact_dir / "report")

    _progress_write("[1/13] load canonical TabDiff splits and synthetic samples")
    train_df = dataset_ctx.train_df.copy()
    holdout_df = dataset_ctx.holdout_df.copy()
    test_df = dataset_ctx.test_df.copy()
    synthetic_df = normalize_tabdiff_dataframe_columns(args.dataset_name, load_csv(synthetic_csv))
    save_csv(input_dir / "synthetic_raw.csv", synthetic_df)
    save_csv(input_dir / "eval_train.csv", train_df)
    save_csv(input_dir / "eval_holdout.csv", holdout_df)
    save_csv(input_dir / "eval_test.csv", test_df)
    save_json(input_dir / "selection_context.json", dataset_ctx.to_manifest())
    overall.update(1)

    _progress_write("[2/13] build cards from canonical train split")
    cards = build_cards_bundle(
        train_df=train_df,
        output_dir=cards_dir,
        seed=args.seed,
        dataset_name=args.dataset_name,
        target_column=dataset_ctx.target_column,
        categorical_columns=dataset_ctx.categorical_columns,
        numerical_columns=dataset_ctx.numerical_columns,
        discrete_numerical_columns=dataset_ctx.discrete_numerical_columns,
        privacy_sensitive_columns=dataset_ctx.privacy_sensitive_columns,
    )
    overall.update(1)

    _progress_write("[3/13] validate synthetic candidates")
    validator = TabularValidator(cards.schema_card, cards.stats_card)
    validation_bundle = validator.validate(
        _df_to_candidate_records(synthetic_df),
        show_progress=progress_enabled,
        progress_desc="validate candidates",
    )
    valid_df = validation_bundle.valid_df.reset_index(drop=True)
    repaired_records = [record for record in validation_bundle.valid_records if record.get("repair_actions")]
    save_json(validation_dir / "validator_report.json", validation_bundle.report)
    save_jsonl(validation_dir / "candidates_valid.jsonl", validation_bundle.valid_records)
    save_jsonl(validation_dir / "candidates_rejected.jsonl", validation_bundle.rejected_records)
    save_jsonl(validation_dir / "candidates_repaired.jsonl", repaired_records)
    save_csv(versions_dir / "raw_valid.csv", valid_df)
    overall.update(1)

    _progress_write("[4/13] initialize unified Pareto selector")
    selector = ParetoSelector(
        train_df=train_df,
        holdout_df=holdout_df,
        schema_card=cards.schema_card,
        stats_card=cards.stats_card,
        seed=args.seed,
        source=args.source,
        lambda_penalty=args.lambda_penalty,
        gamma=args.gamma,
        privacy_version=args.privacy_version,
        density_reference_size=args.density_reference_size,
        nn_device=nn_device,
        nn_query_batch_size=args.nn_query_batch_size,
        nn_reference_chunk_size=args.nn_reference_chunk_size,
        final_fidelity_floor_eps=args.final_fidelity_floor_eps,
        final_trend_floor_eps=args.final_trend_floor_eps,
    )
    d_cur_df = selector.initialize_d_cur(size=args.d_cur_size)
    pool_df = valid_df.copy()
    pool_records = validation_bundle.valid_records.copy()
    save_csv(selection_dir / "d_cur_init.csv", d_cur_df)
    save_csv(selection_dir / "candidate_pool.csv", pool_df)
    evaluator = TabDiffEvaluator(
        dataset_name=args.dataset_name,
        device=eval_device,
        metric_list=["density", "dcr"],
        real_data_path=input_dir / "eval_train.csv",
        test_data_path=input_dir / "eval_test.csv",
        val_data_path=input_dir / "eval_holdout.csv",
    )
    overall.update(1)

    _progress_write("[5/13] surrogate scoring and preselect")
    desired_keep_k = min(args.keep_k, len(pool_records))
    requested_preselect_target = min(len(pool_records), max(args.preselect_target, desired_keep_k))
    surrogate_records_all = selector.compute_surrogates(
        pool_df,
        show_progress=progress_enabled,
        progress_desc="surrogate scoring",
        candidate_ids=np.fromiter(
            (int(record.get("candidate_id", idx)) for idx, record in enumerate(pool_records)),
            dtype=int,
            count=len(pool_records),
        ),
    )
    preselect_should_run = requested_preselect_target < len(pool_records) and len(pool_records) > desired_keep_k
    preselect_gate: dict[str, object]
    candidate_preselect_report: dict[str, object] = {}
    baseline_preselect_report: dict[str, object] = {}
    if preselect_should_run:
        baseline_surrogates = [dict(record) for record in surrogate_records_all]
        candidate_surrogates = [dict(record) for record in surrogate_records_all]
        baseline_valid, baseline_sur = selector.dual_median_filter_baseline(
            valid_records=pool_records,
            surrogate_records=baseline_surrogates,
            target_preselect=requested_preselect_target,
            show_progress=progress_enabled,
            progress_desc="preselect baseline",
        )
        baseline_preselect_report = dict(selector.last_preselect_report)
        baseline_candidate_ids = np.asarray(
            [int(record.get("candidate_id", idx)) for idx, record in enumerate(baseline_sur)],
            dtype=int,
        )
        preselected_valid_candidate, preselected_sur_candidate = selector.dual_median_filter(
            valid_records=pool_records,
            surrogate_records=candidate_surrogates,
            target_preselect=requested_preselect_target,
            anchor_candidate_ids=baseline_candidate_ids,
            show_progress=progress_enabled,
            progress_desc="preselect candidate",
        )
        candidate_preselect_report = dict(selector.last_preselect_report)

        raw_reference_df, _, _ = selector.select_keep_random(
            candidate_records=pool_records,
            keep_k=requested_preselect_target,
            rng_seed=args.seed,
        )
        candidate_preselected_df = _records_to_df(preselected_valid_candidate, selector.column_order)
        baseline_preselected_df = _records_to_df(baseline_valid, selector.column_order)
        preselect_gate = _build_preselect_gate_report(
            raw_metrics=_subset_gate_metrics(selector, evaluator, raw_reference_df),
            candidate_metrics=_subset_gate_metrics(selector, evaluator, candidate_preselected_df),
            baseline_metrics=_subset_gate_metrics(selector, evaluator, baseline_preselected_df),
            candidate_mode=str(candidate_preselect_report.get("mode", "candidate_preselect")),
            baseline_mode=str(baseline_preselect_report.get("mode", "baseline_preselect")),
        )
        preselect_gate["candidate"]["preselect_report"] = candidate_preselect_report
        preselect_gate["baseline"]["preselect_report"] = baseline_preselect_report

        if preselect_gate["selected_source"] == "candidate_preselect":
            preselected_valid = preselected_valid_candidate
            preselected_sur = preselected_sur_candidate
        else:
            preselected_valid = baseline_valid
            preselected_sur = baseline_sur
        preselect_status = {
            "applied": True,
            "mode": str(preselect_gate["selected_mode"]),
            "reason": None if not preselect_gate["fallback_applied"] else "preselect_gate_fallback_to_baseline",
            "rows_before": len(pool_records),
            "rows_after": len(preselected_valid),
            "selected_source": preselect_gate["selected_source"],
            "fallback_applied": bool(preselect_gate["fallback_applied"]),
            "selected_pass": bool(preselect_gate["selected_pass"]),
        }
    else:
        preselected_valid = pool_records.copy()
        preselected_sur = surrogate_records_all.copy()
        preselect_gate = {
            "skipped": True,
            "reason": (
                "target_not_reductive" if requested_preselect_target >= len(pool_records) else "pool_too_close_to_keep_k"
            ),
            "selected_source": "full_pool",
            "selected_mode": "skipped_full_pool",
            "selected_pass": None,
            "fallback_applied": False,
        }
        preselect_status = {
            "applied": False,
            "mode": "skipped_full_pool",
            "reason": (
                "target_not_reductive" if requested_preselect_target >= len(pool_records) else "pool_too_close_to_keep_k"
            ),
            "rows_before": len(pool_records),
            "rows_after": len(preselected_valid),
        }
    if len(preselected_valid) < desired_keep_k:
        preselected_valid = pool_records.copy()
        preselected_sur = surrogate_records_all.copy()
        preselect_gate = {
            **preselect_gate,
            "selected_source": "full_pool",
            "selected_mode": "fallback_full_pool",
            "selected_pass": False,
            "fallback_applied": True,
            "reason": "selected_preselect_below_keep_k",
        }
        preselect_status = {
            "applied": False,
            "mode": "fallback_full_pool",
            "reason": "dual_median_filter_below_keep_k",
            "rows_before": len(pool_records),
            "rows_after": len(preselected_valid),
            "selected_source": "full_pool",
            "fallback_applied": True,
            "selected_pass": False,
        }
    effective_preselect_target = len(preselected_valid)
    effective_keep_k = min(desired_keep_k, len(preselected_valid))
    if effective_keep_k <= 0:
        raise RuntimeError("effective_keep_k <= 0. Increase generator sample size or reduce keep_k.")
    preselect_privacy_objective, preselect_fidelity_objective = _preselect_objective_manifest(preselect_status["mode"])
    overall.update(1)

    _progress_write("[6/13] global exact scoring")
    global_exact_records, global_baselines = selector.compute_exact_scores(
        d_cur_df,
        preselected_valid,
        show_progress=progress_enabled,
        progress_desc="global exact scoring",
    )
    overall.update(1)

    _progress_write("[7/13] fidelity ceiling and utility proxy")
    selection_records = preselected_valid
    preselected_valid_df = _records_to_df(preselected_valid, selector.column_order)
    pre_ceiling_static_utility_bundle = build_static_balanced_utility_scores(
        selector=selector,
        preselected_records=selection_records,
        random_state=args.seed,
    )
    (
        initial_fidelity_ceiling_df,
        initial_fidelity_ceiling_records,
        initial_fidelity_ceiling_report,
    ) = selector.construct_fidelity_ceiling_subset(
        preselected_records=selection_records,
        exact_records=global_exact_records,
        keep_k=effective_keep_k,
        utility_scores_by_id=pre_ceiling_static_utility_bundle["score_by_id"],
        utility_weight=args.fidelity_ceiling_utility_weight,
        refine_utility_weight=args.fidelity_ceiling_refine_utility_weight,
        utility_score_name="utility_static_balanced",
        show_progress=progress_enabled,
        progress_desc="initial fidelity ceiling",
    )
    utility_proxy_bundle = build_utility_proxy_scores(
        selector=selector,
        preselected_records=selection_records,
        anchor_records=initial_fidelity_ceiling_records,
        random_state=args.seed,
        show_progress=progress_enabled,
    )
    global_exact_records, utility_proxy_merge_report = _attach_utility_proxy_fields(
        global_exact_records,
        utility_proxy_bundle["proxy_scores"],
    )
    second_pass_utility_scores_by_id = {
        int(record["candidate_id"]): float(record.get("u_proxy", 0.0))
        for record in utility_proxy_bundle["proxy_scores"]
    }
    second_pass_enabled = bool(
        second_pass_utility_scores_by_id and float(args.fidelity_ceiling_second_pass_utility_weight) > 0.0
    )
    if second_pass_enabled:
        fidelity_ceiling_df, fidelity_ceiling_records, fidelity_ceiling_report = selector.construct_fidelity_ceiling_subset(
            preselected_records=selection_records,
            exact_records=global_exact_records,
            keep_k=effective_keep_k,
            utility_scores_by_id=second_pass_utility_scores_by_id,
            utility_weight=args.fidelity_ceiling_second_pass_utility_weight,
            refine_utility_weight=args.fidelity_ceiling_second_pass_refine_utility_weight,
            utility_score_name="utility_proxy_second_pass",
            show_progress=progress_enabled,
            progress_desc="dynamic utility ceiling",
        )
        fidelity_ceiling_report["initial_anchor"] = initial_fidelity_ceiling_report.get("reference", {})
        fidelity_ceiling_report["second_pass"] = {
            "applied": True,
            "utility_source": "utility_proxy_scores.u_proxy",
            "dynamic_anchor": "initial_static_fidelity_ceiling",
            "initial_rows": int(len(initial_fidelity_ceiling_records)),
            "final_rows": int(len(fidelity_ceiling_records)),
            "utility_weight": float(args.fidelity_ceiling_second_pass_utility_weight),
            "refine_utility_weight": float(args.fidelity_ceiling_second_pass_refine_utility_weight),
        }
    else:
        fidelity_ceiling_df = initial_fidelity_ceiling_df
        fidelity_ceiling_records = initial_fidelity_ceiling_records
        fidelity_ceiling_report = initial_fidelity_ceiling_report
        fidelity_ceiling_report["second_pass"] = {
            "applied": False,
            "reason": "disabled_or_empty_utility_scores",
        }
    floor_reference = fidelity_ceiling_report.get(
        "reference",
        {
            "name": "preselected_fidelity_ceiling_keep_k",
            "rows": len(fidelity_ceiling_records),
            "fidelity_1d": selector.compute_dataset_fidelity(fidelity_ceiling_df),
            "fidelity_2d": selector.compute_dataset_pair_fidelity(fidelity_ceiling_df),
            "privacy_mean": selector.compute_dataset_privacy(fidelity_ceiling_df),
        },
    )
    overall.update(1)

    _progress_write("[8/13] streaming archive diagnostic")
    archive_budget = max(effective_keep_k, int(round(args.archive_budget_scale * effective_keep_k)))
    archive_should_run = archive_budget < len(preselected_valid) and len(preselected_valid) > effective_keep_k
    if archive_should_run:
        archive_records, archive_exact_records, streaming_report = _run_streaming_archive(
            selector=selector,
            pool_records=preselected_valid,
            d_cur_df=d_cur_df,
            keep_k=effective_keep_k,
            preselect_target=effective_preselect_target,
            chunk_size=args.selection_chunk_size,
            archive_budget=archive_budget,
            local_keep_factor=args.local_keep_factor,
            show_progress=progress_enabled,
        )
        archive_status = {
            "applied": True,
            "mode": "streaming_archive",
            "reason": None,
            "rows_before": len(preselected_valid),
            "rows_after": len(archive_records),
            "fixed_reference_baseline": True,
        }
    else:
        archive_records = preselected_valid.copy()
        archive_exact_records = global_exact_records.copy()
        streaming_report = {
            "mode": "skipped",
            "reason": "archive_budget_not_binding",
            "chunk_size": args.selection_chunk_size,
            "archive_budget": archive_budget,
            "archive_rows_final": len(archive_records),
            "fixed_reference_baseline": True,
            "chunks": [],
        }
        archive_status = {
            "applied": False,
            "mode": "skipped_full_preselected_pool",
            "reason": "archive_budget_not_binding",
            "rows_before": len(preselected_valid),
            "rows_after": len(archive_records),
            "fixed_reference_baseline": True,
        }
    archive_df = _records_to_df(archive_records, selector.column_order)
    save_csv(selection_dir / "archive_pool.csv", archive_df)
    overall.update(1)

    _progress_write("[9/13] core selections on unified comparison pool")
    if archive_should_run:
        archive_rescored_exact_records, archive_rescore_baselines = selector.compute_exact_scores(
            d_cur_df,
            archive_records,
            show_progress=progress_enabled,
            progress_desc="archive exact scoring",
        )
        archive_rescored_exact_records, archive_utility_proxy_merge_report = _attach_utility_proxy_fields(
            archive_rescored_exact_records,
            utility_proxy_bundle["proxy_scores"],
        )
    else:
        archive_rescored_exact_records = global_exact_records.copy()
        archive_rescore_baselines = dict(global_baselines)
        archive_utility_proxy_merge_report = dict(utility_proxy_merge_report)

    baseline_full_records = pool_records.copy()
    selection_exact_records = global_exact_records
    raw_baseline_pool_name = "pool_records"
    selection_pool_name = "preselected_valid"

    raw_full_keep_records = baseline_full_records[:effective_keep_k]
    raw_full_selection_df = _records_to_df(raw_full_keep_records, selector.column_order)
    random_full_keep_df, random_full_keep_records, random_full_report = selector.select_keep_random(
        candidate_records=baseline_full_records,
        keep_k=effective_keep_k,
        rng_seed=args.seed,
    )
    scalar_keep_df, scalar_keep_records, scalar_report = selector.select_keep_scalarization(
        preselected_records=selection_records,
        exact_records=selection_exact_records,
        keep_k=effective_keep_k,
        fidelity_1d_weight=0.25,
        fidelity_2d_weight=0.25,
        privacy_weight=0.30,
        utility_weight=0.20,
        mode="matched",
        floor_reference=floor_reference,
    )
    scalar_naive_keep_df, scalar_naive_keep_records, scalar_naive_report = selector.select_keep_scalarization(
        preselected_records=selection_records,
        exact_records=selection_exact_records,
        keep_k=effective_keep_k,
        fidelity_1d_weight=0.25,
        fidelity_2d_weight=0.25,
        privacy_weight=0.30,
        utility_weight=0.0,
        mode="naive",
    )
    pareto_keep_df, pareto_keep_records, pareto_report = selector.select_keep(
        preselected_records=selection_records,
        surrogate_records=[],
        exact_records=selection_exact_records,
        keep_k=effective_keep_k,
        floor_reference=floor_reference,
        constraint_reference_records=fidelity_ceiling_records,
        floor_mode=args.pareto_floor_mode,
        soft_fidelity_floor_eps=args.pareto_soft_fidelity_floor_eps,
        soft_trend_floor_eps=args.pareto_soft_trend_floor_eps,
        soft_privacy_floor_eps=args.pareto_soft_privacy_floor_eps,
        soft_utility_floor_eps=args.pareto_soft_utility_floor_eps,
        soft_min_score_delta=args.pareto_soft_min_score_delta,
    )
    overall.update(1)

    _progress_write("[10/13] build 4D direction families")
    scalar_naive_family_df, scalar_naive_family_records, scalar_naive_family_reports = _build_direction_family(
        selector=selector,
        records=selection_records,
        exact_records=selection_exact_records,
        keep_k=effective_keep_k,
        family_type="scalar_naive",
        show_progress=progress_enabled,
    )
    scalar_matched_family_df, scalar_matched_family_records, scalar_matched_family_reports = _build_direction_family(
        selector=selector,
        records=selection_records,
        exact_records=selection_exact_records,
        keep_k=effective_keep_k,
        family_type="scalar_matched",
        floor_reference=floor_reference,
        show_progress=progress_enabled,
    )
    pareto_family_df, pareto_family_records, pareto_family_reports = _build_direction_family(
        selector=selector,
        records=selection_records,
        exact_records=selection_exact_records,
        keep_k=effective_keep_k,
        family_type="pareto",
        floor_reference=floor_reference,
        constraint_reference_records=fidelity_ceiling_records,
        floor_mode=args.pareto_floor_mode,
        soft_fidelity_floor_eps=args.pareto_soft_fidelity_floor_eps,
        soft_trend_floor_eps=args.pareto_soft_trend_floor_eps,
        soft_privacy_floor_eps=args.pareto_soft_privacy_floor_eps,
        soft_utility_floor_eps=args.pareto_soft_utility_floor_eps,
        soft_min_score_delta=args.pareto_soft_min_score_delta,
        show_progress=progress_enabled,
    )
    pareto_keep_df, pareto_keep_records, pareto_report, pareto_finalist_rerank = _rerank_pareto_finalists_on_search_holdout(
        selector=selector,
        exact_records=selection_exact_records,
        pareto_keep_df=pareto_keep_df,
        pareto_keep_records=pareto_keep_records,
        pareto_report=pareto_report,
        pareto_family_df=pareto_family_df,
        pareto_family_records=pareto_family_records,
        pareto_family_reports=pareto_family_reports,
        floor_reference=floor_reference,
        fidelity_ceiling_df=fidelity_ceiling_df,
        fidelity_ceiling_records=fidelity_ceiling_records,
        utility_switch_min=args.pareto_rerank_utility_switch_min,
        privacy_switch_min=args.pareto_rerank_privacy_switch_min,
        floor_mode=args.pareto_floor_mode,
        soft_fidelity_floor_eps=args.pareto_soft_fidelity_floor_eps,
        soft_trend_floor_eps=args.pareto_soft_trend_floor_eps,
    )
    overall.update(1)

    save_jsonl(selection_dir / "surrogate_scores.jsonl", surrogate_records_all)
    save_jsonl(selection_dir / "preselected_surrogates.jsonl", preselected_sur)
    save_jsonl(selection_dir / "utility_pre_ceiling_static_scores.jsonl", pre_ceiling_static_utility_bundle["static_scores"])
    save_jsonl(selection_dir / "utility_static_scores.jsonl", utility_proxy_bundle["static_scores"])
    save_jsonl(selection_dir / "utility_dynamic_scores.jsonl", utility_proxy_bundle["dynamic_scores"])
    save_jsonl(selection_dir / "utility_proxy_scores.jsonl", utility_proxy_bundle["proxy_scores"])
    save_jsonl(selection_dir / "exact_scores.jsonl", global_exact_records)
    save_jsonl(selection_dir / "archive_exact_scores.jsonl", archive_rescored_exact_records)
    save_json(selection_dir / "baselines.json", global_baselines)
    save_json(selection_dir / "archive_rescore_baselines.json", archive_rescore_baselines)
    save_json(selection_dir / "utility_dynamic_blocks.json", utility_proxy_bundle["dynamic_blocks"])
    save_json(
        selection_dir / "utility_proxy_manifest.json",
        {
            **utility_proxy_bundle["manifest"],
            "teacher_manifest": utility_proxy_bundle.get("teacher_manifest", {}),
            "pre_ceiling_static": pre_ceiling_static_utility_bundle["manifest"],
            "fidelity_ceiling_second_pass": fidelity_ceiling_report.get("second_pass", {}),
            "merge_report": utility_proxy_merge_report,
            "archive_merge_report": archive_utility_proxy_merge_report,
        },
    )
    save_json(selection_dir / "random_full_report.json", random_full_report)
    save_json(selection_dir / "scalarization_report.json", scalar_report)
    save_json(selection_dir / "scalarization_naive_report.json", scalar_naive_report)
    save_json(selection_dir / "pareto_report.json", pareto_report)
    save_json(selection_dir / "pareto_finalist_rerank.json", pareto_finalist_rerank)
    save_json(selection_dir / "fidelity_ceiling_initial_report.json", initial_fidelity_ceiling_report)
    save_json(selection_dir / "fidelity_ceiling_report.json", fidelity_ceiling_report)
    save_json(selection_dir / "streaming_archive_report.json", streaming_report)
    save_json(selection_dir / "exact_chunk_report.json", streaming_report)
    save_json(selection_dir / "preselect_gate.json", preselect_gate)
    save_json(selection_dir / "scalar_family_naive_reports.json", scalar_naive_family_reports)
    save_json(selection_dir / "scalar_family_matched_reports.json", scalar_matched_family_reports)
    save_json(selection_dir / "pareto_family_reports.json", pareto_family_reports)
    save_json(
        selection_dir / "selection_manifest.json",
        {
            "dataset_name": args.dataset_name,
            "synthetic_csv": str(synthetic_csv),
            "protocol": "tabdiff_canonical_train_with_derived_holdout",
            "requested_keep_k": args.keep_k,
            "requested_preselect_target": requested_preselect_target,
            "effective_preselect_target": effective_preselect_target,
            "effective_keep_k": effective_keep_k,
            "candidate_pool_rows": len(pool_records),
            "preselected_rows": len(preselected_valid),
            "raw_baseline_pool_name": raw_baseline_pool_name,
            "raw_baseline_pool_rows": len(baseline_full_records),
            "selection_pool_name": selection_pool_name,
            "selection_pool_rows": len(selection_records),
            "comparison_pool_name": selection_pool_name,
            "comparison_pool_rows": len(selection_records),
            "final_floor_reference_name": floor_reference.get("name", "preselected_fidelity_ceiling_keep_k"),
            "final_floor_reference_rows": len(fidelity_ceiling_records),
            "archive_budget": archive_budget,
            "archive_rows": len(archive_records),
            "d_cur_rows": len(d_cur_df),
            "selection_chunk_size": args.selection_chunk_size,
            "lambda_penalty": args.lambda_penalty,
            "gamma": args.gamma,
            "privacy_version": args.privacy_version,
            "scalar_fidelity_1d_weight": 0.25,
            "scalar_fidelity_2d_weight": 0.25,
            "scalar_privacy_weight": 0.30,
            "scalar_utility_weight": 0.20,
            "three_objective_enabled": True,
            "final_fidelity_floor_eps": args.final_fidelity_floor_eps,
            "final_trend_floor_eps": args.final_trend_floor_eps,
            "fidelity_ceiling_utility_weight": args.fidelity_ceiling_utility_weight,
            "fidelity_ceiling_refine_utility_weight": args.fidelity_ceiling_refine_utility_weight,
            "fidelity_ceiling_second_pass_utility_weight": args.fidelity_ceiling_second_pass_utility_weight,
            "fidelity_ceiling_second_pass_refine_utility_weight": args.fidelity_ceiling_second_pass_refine_utility_weight,
            "pareto_rerank_utility_switch_min": args.pareto_rerank_utility_switch_min,
            "pareto_rerank_privacy_switch_min": args.pareto_rerank_privacy_switch_min,
            "pareto_floor_mode": args.pareto_floor_mode,
            "pareto_soft_fidelity_floor_eps": args.pareto_soft_fidelity_floor_eps,
            "pareto_soft_trend_floor_eps": args.pareto_soft_trend_floor_eps,
            "pareto_soft_privacy_floor_eps": args.pareto_soft_privacy_floor_eps,
            "pareto_soft_utility_floor_eps": args.pareto_soft_utility_floor_eps,
            "pareto_soft_min_score_delta": args.pareto_soft_min_score_delta,
            "preselect_privacy_objective": preselect_privacy_objective,
            "preselect_fidelity_objective": {
                **preselect_fidelity_objective,
                "pair_edges": len(selector.pair_marginal_edges),
            },
            "final_selection_floor_proxy": {
                "fidelity": "exact_1d_marginal_similarity",
                "trend": "exact_2d_pair_similarity",
            },
            "nn_device": nn_device,
            "nn_query_batch_size": args.nn_query_batch_size,
            "nn_reference_chunk_size": args.nn_reference_chunk_size,
            "density_reference_size": args.density_reference_size,
            "holdout_strategy": dataset_ctx.holdout_strategy,
            "holdout_fraction": dataset_ctx.holdout_fraction,
            "preselect_fallback": preselect_status,
            "preselect_status": preselect_status,
            "preselect_gate": preselect_gate,
            "archive_status": archive_status,
            "fidelity_ceiling_second_pass": fidelity_ceiling_report.get("second_pass", {}),
            "utility_proxy": {
                **utility_proxy_bundle["manifest"],
                "pre_ceiling_static": pre_ceiling_static_utility_bundle["manifest"],
                "merge_report": utility_proxy_merge_report,
                "archive_merge_report": archive_utility_proxy_merge_report,
            },
            "pareto_finalist_rerank": pareto_finalist_rerank,
            "validator_num_repaired": validation_bundle.report.get("num_repaired", 0),
            "validator_repair_rate": validation_bundle.report.get("repair_rate", 0.0),
        },
    )
    _save_selection_csvs(
        versions_dir=versions_dir,
        raw_df=raw_full_selection_df,
        random_df=random_full_keep_df,
        scalar_df=scalar_keep_df,
        pareto_df=pareto_keep_df,
        raw_tag="raw_full",
        random_tag="random_full",
    )
    save_csv(versions_dir / "selection_scalar_naive.csv", scalar_naive_keep_df)
    save_csv(versions_dir / "selection_archive_pool.csv", archive_df)
    save_csv(versions_dir / "selection_preselected_valid.csv", preselected_valid_df)
    save_csv(versions_dir / "selection_preselected_fidelity_ceiling_initial_keep_k.csv", initial_fidelity_ceiling_df)
    save_csv(versions_dir / "selection_preselected_fidelity_ceiling_keep_k.csv", fidelity_ceiling_df)
    save_csv(versions_dir / "preselected_valid_keep.csv", preselected_valid_df)
    save_csv(versions_dir / "preselected_fidelity_ceiling_keep_k.csv", fidelity_ceiling_df)
    _save_family_csvs(versions_dir, "selection_scalar_family_naive", scalar_naive_family_df)
    _save_family_csvs(versions_dir, "selection_scalar_family_matched", scalar_matched_family_df)
    _save_family_csvs(versions_dir, "selection_endpoint", pareto_family_df)

    _progress_write("[11/13] evaluate core selections")
    selection_inputs = [
        ("raw_full", raw_full_selection_df, raw_full_keep_records, baseline_full_records),
        ("random_full", random_full_keep_df, random_full_keep_records, baseline_full_records),
        ("preselected_valid", preselected_valid_df, preselected_valid, baseline_full_records),
        ("preselected_fidelity_ceiling_keep_k", fidelity_ceiling_df, fidelity_ceiling_records, selection_records),
        ("scalar", scalar_keep_df, scalar_keep_records, selection_records),
        ("scalar_naive", scalar_naive_keep_df, scalar_naive_keep_records, selection_records),
        ("pareto", pareto_keep_df, pareto_keep_records, selection_records),
    ]
    selection_metrics: dict[str, dict[str, object]] = {}
    selection_iter = _progress(
        selection_inputs,
        total=len(selection_inputs),
        desc="evaluate selections",
        dynamic_ncols=True,
        disable=not progress_enabled,
    )
    for selection_name, selection_df, keep_records, source_records in selection_iter:
        selection_metrics[selection_name] = _evaluate_selection(
            selection_name,
            selection_df,
            keep_records,
            source_records,
            selector,
            evaluator,
            eval_dir,
            test_df,
            args.jsd_epsilon,
            args.rare_threshold,
        )
        selection_iter.set_postfix(selection=selection_name, rows=len(selection_df))
    overall.update(1)

    scalar_naive_family_metrics: dict[str, dict[str, object]] = {}
    scalar_matched_family_metrics: dict[str, dict[str, object]] = {}
    pareto_family_metrics: dict[str, dict[str, object]] = {}

    if args.skip_family_eval:
        _progress_write("[12/13] skip 4D direction family evaluation")
    else:
        _progress_write("[12/13] evaluate 4D direction families")
        family_eval_iter = _progress(
            [tag for tag in scalar_naive_family_df.keys()],
            total=len(scalar_naive_family_df),
            desc="evaluate families",
            dynamic_ncols=True,
            disable=not progress_enabled,
        )
        for tag in family_eval_iter:
            scalar_naive_family_metrics[tag] = _evaluate_selection(
                _selection_name("scalar_family_naive", tag),
                scalar_naive_family_df[tag],
                scalar_naive_family_records[tag],
                selection_records,
                selector,
                evaluator,
                eval_dir,
                test_df,
                args.jsd_epsilon,
                args.rare_threshold,
            )
            scalar_matched_family_metrics[tag] = _evaluate_selection(
                _selection_name("scalar_family_matched", tag),
                scalar_matched_family_df[tag],
                scalar_matched_family_records[tag],
                selection_records,
                selector,
                evaluator,
                eval_dir,
                test_df,
                args.jsd_epsilon,
                args.rare_threshold,
            )
            pareto_family_metrics[tag] = _evaluate_selection(
                _selection_name("endpoint", tag),
                pareto_family_df[tag],
                pareto_family_records[tag],
                selection_records,
                selector,
                evaluator,
                eval_dir,
                test_df,
                args.jsd_epsilon,
                args.rare_threshold,
            )
            family_eval_iter.set_postfix(direction=tag)

    family_comparison = _compare_families(
        pareto_family_metrics=pareto_family_metrics,
        scalar_family_metrics=scalar_matched_family_metrics,
    )
    utility_family_comparison = dict(family_comparison.get("utility_space", {}))
    selection_gate = _build_selection_gate_report(
        selection_metrics=selection_metrics,
        family_comparison=family_comparison,
    )
    save_json(report_dir / "family_comparison.json", family_comparison)
    save_json(report_dir / "utility_family_comparison.json", utility_family_comparison)
    save_json(report_dir / "selection_gate.json", selection_gate)
    save_json(report_dir / "preselect_gate.json", preselect_gate)
    overall.update(1)

    _progress_write("[13/13] write summary")
    summary = {
        "source": args.source,
        "synthetic_csv": str(synthetic_csv),
        "protocol": "tabdiff_canonical_train_with_derived_holdout",
        "eval_device": eval_device,
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "test_rows": int(len(test_df)),
        "raw_rows": int(len(synthetic_df)),
        "valid_rows": int(len(valid_df)),
        "rejected_rows": int(len(validation_bundle.rejected_records)),
        "validator_reject_rate": float(validation_bundle.report.get("reject_rate", 0.0)),
        "validator_repaired_rows": int(validation_bundle.report.get("num_repaired", 0)),
        "validator_repair_rate": float(validation_bundle.report.get("repair_rate", 0.0)),
        "d_cur_rows": int(len(d_cur_df)),
        "candidate_pool_rows": int(len(pool_df)),
        "preselected_rows": int(len(preselected_valid)),
        "raw_baseline_pool_name": raw_baseline_pool_name,
        "raw_baseline_pool_rows": int(len(baseline_full_records)),
        "selection_pool_name": selection_pool_name,
        "selection_pool_rows": int(len(selection_records)),
        "comparison_pool_name": selection_pool_name,
        "comparison_pool_rows": int(len(selection_records)),
        "final_floor_reference_name": floor_reference.get("name", "preselected_fidelity_ceiling_keep_k"),
        "final_floor_reference_rows": int(len(fidelity_ceiling_records)),
        "archive_rows": int(len(archive_records)),
        "requested_preselect_target": int(requested_preselect_target),
        "effective_preselect_target": int(effective_preselect_target),
        "requested_keep_k": int(args.keep_k),
        "effective_keep_k": int(effective_keep_k),
        "selection_chunk_size": int(args.selection_chunk_size),
        "archive_budget": int(archive_budget),
        "lambda_penalty": float(args.lambda_penalty),
        "gamma": float(args.gamma),
        "privacy_version": args.privacy_version,
        "final_fidelity_floor_eps": float(args.final_fidelity_floor_eps),
        "final_trend_floor_eps": float(args.final_trend_floor_eps),
        "preselect_privacy_objective": preselect_privacy_objective,
        "preselect_fidelity_objective": {
            **preselect_fidelity_objective,
            "pair_edges": int(len(selector.pair_marginal_edges)),
        },
        "final_selection_floor_proxy": {
            "fidelity": "exact_1d_marginal_similarity",
            "trend": "exact_2d_pair_similarity",
        },
        "nn_device": nn_device,
        "nn_query_batch_size": int(args.nn_query_batch_size),
        "nn_reference_chunk_size": int(args.nn_reference_chunk_size),
        "density_reference_size": int(args.density_reference_size),
        "holdout_fraction": float(args.holdout_fraction),
        "preselect_fallback": preselect_status,
        "preselect_status": preselect_status,
        "preselect_gate": preselect_gate,
        "archive_status": archive_status,
        "pareto_finalist_rerank": pareto_finalist_rerank,
        "raw_valid": _subset_metrics(selector, valid_df),
        "selection_metrics": selection_metrics,
        "scalar_family_naive_metrics": scalar_naive_family_metrics,
        "scalar_family_matched_metrics": scalar_matched_family_metrics,
        "pareto_family_metrics": pareto_family_metrics,
        "family_comparison": family_comparison,
        "family_eval_skipped": bool(args.skip_family_eval),
        "utility_family_comparison": utility_family_comparison,
        "selection_gate": selection_gate,
        "streaming_archive_report": streaming_report,
        "random_full_smoke": selector.compute_smoke_metrics(
            surrogate_records_all=surrogate_records_all,
            keep_records=random_full_keep_records,
            keep_df=random_full_keep_df,
        ),
        "preselected_valid_smoke": selector.compute_smoke_metrics(
            surrogate_records_all=surrogate_records_all,
            keep_records=preselected_valid,
            keep_df=preselected_valid_df,
        ),
        "preselected_fidelity_ceiling_smoke": selector.compute_smoke_metrics(
            surrogate_records_all=surrogate_records_all,
            keep_records=fidelity_ceiling_records,
            keep_df=fidelity_ceiling_df,
        ),
        "scalarization_smoke": selector.compute_smoke_metrics(
            surrogate_records_all=surrogate_records_all,
            keep_records=scalar_keep_records,
            keep_df=scalar_keep_df,
        ),
        "pareto_smoke": selector.compute_smoke_metrics(
            surrogate_records_all=surrogate_records_all,
            keep_records=pareto_keep_records,
            keep_df=pareto_keep_df,
        ),
    }
    save_json(report_dir / "summary.json", summary)
    overall.update(1)
    overall.close()
    print(f"Postprocess summary saved to {report_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
