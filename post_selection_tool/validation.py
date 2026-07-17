from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from postprocess.cards import build_and_save_cards
from postprocess.types import CardsBundle
from postprocess.validator import TabularValidator

from .config import progress_enabled
from .context import resolve_eval_device, resolve_nn_device
from .io import (
    df_to_candidate_records,
    load_json,
    load_jsonl,
    remove_known_mirror,
    save_csv,
    save_json,
    save_jsonl,
)
from .selector import ParetoSelector
from .state import SelectionState
from .theta_guidance import mark_report_default_fidelity_columns, override_report_col_ps_with_all_features


CARD_FILENAMES = ("schema_card.json", "stats_card.json", "prototype_card.jsonl", "residual_card.json")


def _cards_complete(cards_dir: Path) -> bool:
    return all((cards_dir / filename).exists() for filename in CARD_FILENAMES)


def _load_cards(cards_dir: Path) -> CardsBundle:
    return CardsBundle(
        schema_card=dict(load_json(cards_dir / "schema_card.json")),
        stats_card=dict(load_json(cards_dir / "stats_card.json")),
        prototype_entries=load_jsonl(cards_dir / "prototype_card.jsonl"),
        residual_card=dict(load_json(cards_dir / "residual_card.json")),
    )


def _build_or_share_cards(state: SelectionState) -> CardsBundle:
    config = state.config
    ctx = state.dataset_ctx
    shared_root = config.shared_artifact_dir
    if shared_root is None:
        return build_and_save_cards(
            train_df=state.train_df,
            output_dir=state.paths.cards_dir,
            seed=config.seed,
            dataset_name=config.dataset_name,
            target_column=ctx.target_column,
            categorical_columns=ctx.categorical_columns,
            numerical_columns=ctx.numerical_columns,
            discrete_numerical_columns=ctx.discrete_numerical_columns,
            privacy_sensitive_columns=ctx.privacy_sensitive_columns,
        )

    shared_cards_dir = Path(shared_root) / "cards"
    if _cards_complete(shared_cards_dir):
        cards = _load_cards(shared_cards_dir)
    else:
        cards = build_and_save_cards(
            train_df=state.train_df,
            output_dir=shared_cards_dir,
            seed=config.seed,
            dataset_name=config.dataset_name,
            target_column=ctx.target_column,
            categorical_columns=ctx.categorical_columns,
            numerical_columns=ctx.numerical_columns,
            discrete_numerical_columns=ctx.discrete_numerical_columns,
            privacy_sensitive_columns=ctx.privacy_sensitive_columns,
        )
    artifact_dir = getattr(state.paths, "artifact_dir", None)
    rollout_cards_dir = (Path(artifact_dir) / "cards") if artifact_dir is not None else Path(state.paths.cards_dir)
    if rollout_cards_dir != shared_cards_dir:
        remove_known_mirror(rollout_cards_dir, CARD_FILENAMES)
    return cards


def _feature_columns_from_schema(schema_card: dict[str, Any]) -> list[str]:
    return [
        str(column)
        for column in schema_card.get("column_order", [])
        if not bool(schema_card.get("columns", {}).get(column, {}).get("is_target", False))
    ]


def _filter_target_column_list(
    columns: list[str] | None,
    *,
    field_name: str,
    schema_card: dict[str, Any],
    fallback_when_empty: bool,
) -> tuple[list[str] | None, dict[str, Any]]:
    target_column = str(schema_card.get("target_column", ""))
    feature_columns = _feature_columns_from_schema(schema_card)
    if columns is None:
        return None, {
            "field": field_name,
            "active": False,
            "removed_target_columns": [],
            "before_count": None,
            "after_count": None,
            "fallback_applied": False,
        }
    normalized = [str(column).strip() for column in columns if column is not None and str(column).strip()]
    deduped = list(dict.fromkeys(normalized))
    filtered = [column for column in deduped if column != target_column]
    removed = [column for column in deduped if column == target_column]
    fallback_applied = False
    if not filtered and fallback_when_empty:
        filtered = list(feature_columns)
        fallback_applied = True
    return filtered, {
        "field": field_name,
        "active": True,
        "removed_target_columns": removed,
        "before_count": len(deduped),
        "after_count": len(filtered),
        "fallback_applied": fallback_applied,
    }


def _remove_target_from_guided_columns(config: Any, schema_card: dict[str, Any]) -> None:
    report = dict(config.theta_guidance_report or {"enabled": False})
    if not bool(report.get("enabled", False)):
        return
    field_reports: dict[str, Any] = {}
    config.fidelity_1d_columns, field_reports["col_1ds"] = _filter_target_column_list(
        config.fidelity_1d_columns,
        field_name="col_1ds",
        schema_card=schema_card,
        fallback_when_empty=True,
    )
    config.fidelity_2d_anchor_columns, field_reports["col_2ds"] = _filter_target_column_list(
        config.fidelity_2d_anchor_columns,
        field_name="col_2ds",
        schema_card=schema_card,
        fallback_when_empty=False,
    )
    config.privacy_columns, field_reports["col_ps"] = _filter_target_column_list(
        config.privacy_columns,
        field_name="col_ps",
        schema_card=schema_card,
        fallback_when_empty=True,
    )
    target_column = str(schema_card.get("target_column", ""))
    utility_removed = config.utility_balance_column == target_column
    if utility_removed:
        config.utility_balance_column = None
    theta = report.get("theta")
    if isinstance(theta, dict):
        updated_theta = dict(theta)
        if config.fidelity_1d_columns is not None:
            updated_theta["col_1ds"] = list(config.fidelity_1d_columns)
        if config.fidelity_2d_anchor_columns is not None:
            updated_theta["col_2ds"] = list(config.fidelity_2d_anchor_columns)
        if config.privacy_columns is not None:
            updated_theta["col_ps"] = list(config.privacy_columns)
        if utility_removed:
            updated_theta["col_u"] = None
        report["theta"] = updated_theta
    report["target_column_filter"] = {
        "enabled": True,
        "target_column": target_column,
        "mode": "remove_target_from_theta_guided_columns",
        "fields": field_reports,
        "col_u_removed": bool(utility_removed),
    }
    config.theta_guidance_report = report


def _reject_required_missing_rows(
    *,
    valid_df: pd.DataFrame,
    valid_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
    schema_card: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, object]]:
    required_columns = [
        column
        for column in schema_card["column_order"]
        if not bool(schema_card["columns"][column].get("missing_allowed", False))
    ]
    if valid_df.empty or not required_columns:
        return valid_df, valid_records, rejected_records, {
            "required_columns": required_columns,
            "rejected_rows": 0,
            "rejected_candidate_ids": [],
        }

    missing_mask = valid_df[required_columns].isna().any(axis=1)
    if not bool(missing_mask.any()):
        return valid_df, valid_records, rejected_records, {
            "required_columns": required_columns,
            "rejected_rows": 0,
            "rejected_candidate_ids": [],
        }

    keep_mask = ~missing_mask
    rejected_candidate_ids: list[int] = []
    filtered_valid_records: list[dict] = []
    for row_idx, record in enumerate(valid_records):
        if bool(missing_mask.iloc[row_idx]):
            row = record.get("row", {})
            missing_columns = [column for column in required_columns if pd.isna(valid_df.iloc[row_idx][column])]
            candidate_id = int(record.get("candidate_id", row_idx))
            rejected_candidate_ids.append(candidate_id)
            rejected_records.append(
                {
                    "candidate_id": candidate_id,
                    "reason": f"required_missing_values:{missing_columns}",
                    "row": row,
                }
            )
        else:
            filtered_valid_records.append(record)

    filtered_valid_df = valid_df.loc[keep_mask].reset_index(drop=True)
    return filtered_valid_df, filtered_valid_records, rejected_records, {
        "required_columns": required_columns,
        "rejected_rows": int(missing_mask.sum()),
        "rejected_candidate_ids": rejected_candidate_ids,
    }


def _deduplicate_valid_rows(
    *,
    valid_df: pd.DataFrame,
    valid_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, object]]:
    if valid_df.empty:
        return valid_df, valid_records, {
            "rows_before": 0,
            "rows_after": 0,
            "duplicate_rows_removed": 0,
            "removed_candidate_ids": [],
        }

    duplicate_mask = valid_df.duplicated(keep="first")
    if not bool(duplicate_mask.any()):
        return valid_df.reset_index(drop=True), valid_records, {
            "rows_before": int(len(valid_df)),
            "rows_after": int(len(valid_df)),
            "duplicate_rows_removed": 0,
            "removed_candidate_ids": [],
        }

    keep_mask = ~duplicate_mask
    removed_candidate_ids = [
        int(record.get("candidate_id", row_idx))
        for row_idx, record in enumerate(valid_records)
        if bool(duplicate_mask.iloc[row_idx])
    ]
    deduped_records = [
        record
        for row_idx, record in enumerate(valid_records)
        if bool(keep_mask.iloc[row_idx])
    ]
    deduped_df = valid_df.loc[keep_mask].reset_index(drop=True)
    return deduped_df, deduped_records, {
        "rows_before": int(len(valid_df)),
        "rows_after": int(len(deduped_df)),
        "duplicate_rows_removed": int(duplicate_mask.sum()),
        "removed_candidate_ids": removed_candidate_ids,
    }


def _attach_candidate_source_metadata(
    records: list[dict[str, Any]],
    source_by_id: dict[int, str],
) -> list[dict[str, Any]]:
    if not source_by_id:
        return records
    for row_idx, record in enumerate(records):
        try:
            candidate_id = int(record.get("candidate_id", row_idx))
        except (TypeError, ValueError):
            candidate_id = row_idx
        source_id = source_by_id.get(candidate_id)
        if source_id:
            record["_source_id"] = source_id
    return records


def build_cards_and_validate(state: SelectionState, *, show_progress: bool = False) -> SelectionState:
    config = state.config
    cards = _build_or_share_cards(state)
    validator = TabularValidator(cards.schema_card, cards.stats_card)
    validation_bundle = validator.validate(
        df_to_candidate_records(state.synthetic_df),
        show_progress=show_progress,
        progress_desc="validate candidates",
    )
    valid_df, valid_records, rejected_records, required_missing_report = _reject_required_missing_rows(
        valid_df=validation_bundle.valid_df.reset_index(drop=True),
        valid_records=validation_bundle.valid_records,
        rejected_records=validation_bundle.rejected_records,
        schema_card=cards.schema_card,
    )
    num_valid_before_dedup = len(valid_records)
    valid_df, valid_records, duplicate_valid_report = _deduplicate_valid_rows(
        valid_df=valid_df,
        valid_records=valid_records,
    )
    valid_records = _attach_candidate_source_metadata(valid_records, state.candidate_source_by_id)
    rejected_records = _attach_candidate_source_metadata(rejected_records, state.candidate_source_by_id)
    validation_report = dict(validation_bundle.report)
    validation_report["required_missing_filter"] = required_missing_report
    validation_report["duplicate_valid_filter"] = duplicate_valid_report
    validation_report["num_valid_before_dedup"] = num_valid_before_dedup
    validation_report["num_valid"] = len(valid_records)
    validation_report["num_rejected"] = len(rejected_records)
    total_candidates = int(validation_report.get("total_candidates", len(state.synthetic_df)))
    validation_report["reject_rate"] = (len(rejected_records) / total_candidates) if total_candidates else 0.0
    validation_report["records_saved"] = bool(config.save_validation_records)
    validation_report["records_save_policy"] = (
        "full_valid_and_rejected_jsonl" if config.save_validation_records else "skipped_large_jsonl"
    )

    state.cards = cards
    state.valid_df = valid_df
    state.valid_records = valid_records
    state.rejected_records = rejected_records
    state.validation_report = validation_report

    save_json(state.paths.validation_dir / "validator_report.json", validation_report)
    if config.save_validation_records:
        save_jsonl(state.paths.validation_dir / "candidates_valid.jsonl", valid_records)
        save_jsonl(state.paths.validation_dir / "candidates_rejected.jsonl", rejected_records)
    shared_root = state.config.shared_artifact_dir
    raw_valid_dir = state.paths.versions_dir if shared_root is None else Path(shared_root) / "versions"
    save_csv(raw_valid_dir / "raw_valid.csv", state.valid_df)
    return state


def initialize_selector_and_pool(state: SelectionState) -> SelectionState:
    if state.cards is None or state.valid_df is None:
        raise RuntimeError("build_cards_and_validate must run before initialize_selector_and_pool")

    config = state.config
    eval_device = resolve_eval_device(config.eval_device)
    nn_device = resolve_nn_device(config.nn_device, eval_device)
    high_cardinality_enabled = (
        str(config.dataset_name).lower() == "diabetes"
        if config.high_cardinality_enabled is None
        else bool(config.high_cardinality_enabled)
    )
    privacy_columns = config.privacy_columns
    fidelity_1d_columns = config.fidelity_1d_columns
    fidelity_2d_anchor_columns = config.fidelity_2d_anchor_columns
    _remove_target_from_guided_columns(config, state.cards.schema_card)
    privacy_columns = config.privacy_columns
    fidelity_1d_columns = config.fidelity_1d_columns
    fidelity_2d_anchor_columns = config.fidelity_2d_anchor_columns
    if config.theta_default_fidelity_columns:
        fidelity_1d_columns = None
        fidelity_2d_anchor_columns = None
        config.fidelity_1d_columns = None
        config.fidelity_2d_anchor_columns = None
        config.theta_guidance_report = mark_report_default_fidelity_columns(
            config.theta_guidance_report,
            state.cards.schema_card,
        )
    if config.theta_col_ps_all_columns:
        privacy_columns, config.theta_guidance_report = override_report_col_ps_with_all_features(
            config.theta_guidance_report,
            state.cards.schema_card,
        )
        config.privacy_columns = list(privacy_columns)

    selector = ParetoSelector(
        train_df=state.train_df,
        holdout_df=state.holdout_df,
        schema_card=state.cards.schema_card,
        stats_card=state.cards.stats_card,
        seed=config.seed,
        source=config.source,
        lambda_penalty=config.lambda_penalty,
        gamma=config.gamma,
        privacy_version=config.privacy_version,
        density_reference_size=config.density_reference_size,
        nn_device=nn_device,
        nn_query_batch_size=config.nn_query_batch_size,
        nn_reference_chunk_size=config.nn_reference_chunk_size,
        fidelity_1d_columns=fidelity_1d_columns,
        fidelity_2d_anchor_columns=fidelity_2d_anchor_columns,
        privacy_columns=privacy_columns,
        utility_balance_column=config.utility_balance_column,
        allow_target_in_fidelity_columns=config.allow_target_in_fidelity_columns,
        allow_target_in_privacy_columns=config.allow_target_in_privacy_columns,
        privacy_encoding_column_mode=config.privacy_encoding_column_mode,
        max_theta_pairs=config.max_theta_pairs,
        final_fidelity_floor_eps=config.final_fidelity_floor_eps,
        final_trend_floor_eps=config.final_trend_floor_eps,
        high_cardinality_enabled=high_cardinality_enabled,
        high_cardinality_threshold=config.high_cardinality_threshold,
        high_cardinality_top_k=config.high_cardinality_top_k,
        high_cardinality_tail_clusters=config.high_cardinality_tail_clusters,
    )
    selector.progress_enabled = progress_enabled(config)

    pool_df = state.valid_df.copy()
    pool_records = state.valid_records.copy()
    if config.d_cur_source == "synthetic" and not state.valid_df.empty:
        max_d_cur = max(1, len(state.valid_df) - min(config.keep_k, len(state.valid_df)))
        d_cur_size = min(config.d_cur_size, max_d_cur)
        d_cur_indices = state.valid_df.sample(n=d_cur_size, random_state=config.seed, replace=False).index.to_list()
        d_cur_index_set = set(d_cur_indices)
        d_cur_df = state.valid_df.loc[d_cur_indices].reset_index(drop=True)
        pool_df = state.valid_df.drop(index=d_cur_indices).reset_index(drop=True)
        pool_records = [record for idx, record in enumerate(state.valid_records) if idx not in d_cur_index_set]
    else:
        d_cur_df = selector.initialize_d_cur(size=config.d_cur_size)

    state.selector = selector
    state.pool_df = pool_df
    state.pool_records = pool_records
    state.d_cur_df = d_cur_df
    shared_root = state.config.shared_artifact_dir
    selection_dir = state.paths.selection_dir if shared_root is None else Path(shared_root) / "selection"
    save_csv(selection_dir / "d_cur_init.csv", d_cur_df)
    save_csv(selection_dir / "candidate_pool.csv", pool_df)
    return state
