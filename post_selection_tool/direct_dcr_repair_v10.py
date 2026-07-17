from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from .direct_dcr_repair_v4 import (
    _build_pairs,
    _candidate_id,
    _dcr_signal_column_order,
    _feature_matrix,
    _row_dcr_signal,
    _utility_scores,
)
from .direct_dcr_repair_v6 import apply_direct_dcr_repair_v6
from .direct_dcr_repair_v9 import (
    _candidate_pool_indices_v9,
    _progressive_pair_filter,
    _signal_prior_arrays,
    _truncate_ids,
)
from .io import records_to_df
from .logging_utils import get_logger


def _log(message: str) -> None:
    get_logger().info("[direct_dcr_repair_v10] %s", message)


def _resolve_signal_device(nn_device: str) -> torch.device:
    if nn_device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if nn_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"nn_device={nn_device} requested but CUDA is unavailable.")
        return torch.device(nn_device)
    return torch.device("cpu")


def _auto_query_batch_size(feature_count: int, requested: int) -> int:
    if requested > 0:
        return int(requested)
    return max(1, min(512, 50_000 // max(1, int(feature_count))))


def _min_l1_distances(
    *,
    query_matrix: np.ndarray,
    reference_tensor: torch.Tensor,
    device: torch.device,
    query_batch_size: int,
    reference_chunk_size: int,
) -> np.ndarray:
    query_count = int(query_matrix.shape[0])
    reference_count = int(reference_tensor.shape[0])
    if query_count == 0:
        return np.zeros(0, dtype=np.float32)
    if reference_count == 0:
        return np.full(query_count, np.inf, dtype=np.float32)

    query_batch_size = max(1, int(query_batch_size))
    reference_chunk_size = max(1, int(reference_chunk_size))
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for q_start in range(0, query_count, query_batch_size):
            q_end = min(q_start + query_batch_size, query_count)
            query = torch.as_tensor(
                np.asarray(query_matrix[q_start:q_end], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            best: torch.Tensor | None = None
            for r_start in range(0, reference_count, reference_chunk_size):
                r_end = min(r_start + reference_chunk_size, reference_count)
                distances = torch.cdist(query, reference_tensor[r_start:r_end], p=1)
                chunk_best = distances.min(dim=1).values
                best = chunk_best if best is None else torch.minimum(best, chunk_best)
            assert best is not None
            outputs.append(best.detach().cpu())
    return torch.cat(outputs).numpy().astype(np.float32, copy=False)


def _row_dcr_signal_batched_l1(
    *,
    pool_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_card: dict[str, Any],
    column_order: list[str],
    cat_weight: float,
    nn_device: str,
    query_batch_size: int,
    reference_chunk_size: int,
) -> dict[str, np.ndarray | str | int]:
    signal_column_order = _dcr_signal_column_order(schema_card, column_order)
    train_matrix, encoder = _feature_matrix(
        df=train_df[column_order],
        train_df=train_df[column_order],
        test_df=test_df[column_order],
        schema_card=schema_card,
        column_order=column_order,
        cat_weight=cat_weight,
    )
    test_matrix, _ = _feature_matrix(
        df=test_df[column_order],
        train_df=train_df[column_order],
        test_df=test_df[column_order],
        schema_card=schema_card,
        column_order=column_order,
        cat_weight=cat_weight,
        encoder=encoder,
    )
    pool_matrix, _ = _feature_matrix(
        df=pool_df[column_order],
        train_df=train_df[column_order],
        test_df=test_df[column_order],
        schema_card=schema_card,
        column_order=column_order,
        cat_weight=cat_weight,
        encoder=encoder,
    )

    device = _resolve_signal_device(str(nn_device))
    query_batch_size = _auto_query_batch_size(pool_matrix.shape[1], int(query_batch_size))
    reference_chunk_size = max(1, int(reference_chunk_size))
    train_tensor = torch.as_tensor(train_matrix, dtype=torch.float32, device=device)
    test_tensor = torch.as_tensor(test_matrix, dtype=torch.float32, device=device)
    dcr_real = _min_l1_distances(
        query_matrix=pool_matrix,
        reference_tensor=train_tensor,
        device=device,
        query_batch_size=query_batch_size,
        reference_chunk_size=reference_chunk_size,
    )
    dcr_test = _min_l1_distances(
        query_matrix=pool_matrix,
        reference_tensor=test_tensor,
        device=device,
        query_batch_size=query_batch_size,
        reference_chunk_size=reference_chunk_size,
    )
    return {
        "features": pool_matrix,
        "dcr_real": np.asarray(dcr_real, dtype=float),
        "dcr_test": np.asarray(dcr_test, dtype=float),
        "is_real_closer": np.asarray(dcr_real < dcr_test, dtype=bool),
        "margin": np.asarray(dcr_test - dcr_real, dtype=float),
        "signal_backend": f"torch_{device.type}",
        "signal_query_batch_size": int(query_batch_size),
        "signal_reference_chunk_size": int(reference_chunk_size),
        "signal_feature_count": int(pool_matrix.shape[1]),
        "signal_column_order": signal_column_order,
        "signal_column_count": int(len(signal_column_order)),
        "signal_column_source": str(schema_card.get("dcr_signal_column_source", "full_column_order")),
    }


def apply_direct_dcr_repair_v10(
    *,
    pool_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    exact_records: list[dict[str, Any]],
    surrogate_records: list[dict[str, Any]] | None,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_card: dict[str, Any],
    column_order: list[str],
    target_margin: float = 0.05,
    max_swap_fraction: float = 0.30,
    candidate_neighbors: int = 64,
    margin_weight: float = 0.10,
    utility_weight: float = 0.55,
    cat_weight: float = 1.0,
    large_keep_k_threshold: int = 50_000,
    large_pool_rows_threshold: int = 180_000,
    large_candidate_rows: int = 72_000,
    large_reference_rows: int = 0,
    large_max_swaps: int = 20_000,
    large_candidate_neighbors: int = 28,
    min_pair_utility_gain: float = -0.10,
    fallback_min_pair_utility_gain: float = -0.25,
    signal_query_batch_size: int = 0,
    signal_reference_chunk_size: int = 65536,
    signal_device: str = "auto",
    report_id_limit: int = 64,
    _report_version: str = "direct_dcr_repair_v10",
    _selection_signal: str = "tabdiff_full_reference_l1_dcr_with_surrogate_biased_candidates",
    _base_strategy: str = "full_reference_surrogate_biased_candidates_v10",
    _pair_builder: Any | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    report_base: dict[str, Any] = {
        "enabled": True,
        "version": _report_version,
        "candidate_full_eval_used": False,
        "intermediate_candidate_count": 0,
        "selection_signal": _selection_signal,
    }
    if not pool_records or not selected_records:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {**report_base, "applied": False, "reason": "empty_inputs"},
        )

    use_bounded = (
        len(selected_records) > int(large_keep_k_threshold)
        or len(pool_records) > int(large_pool_rows_threshold)
    )
    if not use_bounded:
        final_df, final_records, v6_report = apply_direct_dcr_repair_v6(
            pool_records=pool_records,
            selected_records=selected_records,
            exact_records=exact_records,
            surrogate_records=surrogate_records,
            train_df=train_df,
            test_df=test_df,
            schema_card=schema_card,
            column_order=column_order,
            target_margin=target_margin,
            max_swap_fraction=max_swap_fraction,
            candidate_neighbors=candidate_neighbors,
            margin_weight=margin_weight,
            utility_weight=utility_weight,
            cat_weight=cat_weight,
        )
        return final_df, final_records, {
            **v6_report,
            "version": _report_version,
            "base_strategy": "exact_v6_small",
            "bounded_mode": False,
        }

    t_all = time.perf_counter()
    _log(
        f"bounded start pool_rows={len(pool_records)} selected_rows={len(selected_records)} "
        f"candidate_budget={int(large_candidate_rows)} reference_budget={int(large_reference_rows)}"
    )
    pool_id_to_index = {
        _candidate_id(record, idx): idx
        for idx, record in enumerate(pool_records)
    }
    selected_pool_indices_list: list[int] = []
    for pos, record in enumerate(selected_records):
        idx = pool_id_to_index.get(_candidate_id(record, pos))
        if idx is None:
            return (
                records_to_df(selected_records, column_order),
                selected_records,
                {**report_base, "applied": False, "reason": "selected_record_not_in_pool"},
            )
        selected_pool_indices_list.append(int(idx))

    selected_pool_indices = np.asarray(selected_pool_indices_list, dtype=np.int64)
    selected_mask = np.zeros(len(pool_records), dtype=bool)
    selected_mask[selected_pool_indices] = True
    if int(selected_mask.sum()) != len(selected_records):
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {**report_base, "applied": False, "reason": "duplicate_selected_pool_indices"},
        )

    utility, utility_report = _utility_scores(
        pool_records=pool_records,
        exact_records=exact_records,
        surrogate_records=surrogate_records,
    )
    exact_quality, holdout_gap, surrogate_quality = _signal_prior_arrays(
        pool_records=pool_records,
        exact_records=exact_records,
        surrogate_records=surrogate_records,
        pool_id_to_index=pool_id_to_index,
    )
    add_pool_indices, candidate_report = _candidate_pool_indices_v9(
        selected_mask=selected_mask,
        selected_pool_indices=selected_pool_indices,
        utility=utility,
        exact_quality=exact_quality,
        holdout_gap=holdout_gap,
        surrogate_quality=surrogate_quality,
        max_rows=int(large_candidate_rows),
    )
    if not add_pool_indices:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "empty_bounded_candidate_pool",
                "bounded_mode": True,
                "pool_rows": int(len(pool_records)),
                "keep_k": int(len(selected_records)),
                "candidate_pool": candidate_report,
            },
        )
    _log(f"candidate pool ready rows={len(add_pool_indices)} report={candidate_report}")

    local_pool_indices = np.concatenate(
        [
            selected_pool_indices,
            np.asarray(add_pool_indices, dtype=np.int64),
        ]
    )
    local_selected_count = int(selected_pool_indices.size)
    local_records = [pool_records[int(idx)] for idx in local_pool_indices.tolist()]
    local_df = records_to_df(local_records, column_order)
    train_ref = train_df[column_order].reset_index(drop=True)
    test_ref = test_df[column_order].reset_index(drop=True)
    sampled_estimate = False
    if int(large_reference_rows) > 0:
        # Kept as an explicit opt-in for quick experiments; default v10 uses full references.
        positions_train = np.linspace(0, len(train_ref) - 1, num=min(int(large_reference_rows), len(train_ref)), dtype=np.int64)
        positions_test = np.linspace(0, len(test_ref) - 1, num=min(int(large_reference_rows), len(test_ref)), dtype=np.int64)
        train_ref = train_ref.iloc[positions_train].reset_index(drop=True)
        test_ref = test_ref.iloc[positions_test].reset_index(drop=True)
        sampled_estimate = True

    t_signal = time.perf_counter()
    _log(
        f"signal start local_rows={len(local_records)} train_ref={len(train_ref)} "
        f"test_ref={len(test_ref)} device={signal_device}"
    )
    signal: dict[str, Any]
    try:
        signal = _row_dcr_signal_batched_l1(
            pool_df=local_df[column_order],
            train_df=train_ref[column_order],
            test_df=test_ref[column_order],
            schema_card=schema_card,
            column_order=column_order,
            cat_weight=cat_weight,
            nn_device=signal_device,
            query_batch_size=signal_query_batch_size,
            reference_chunk_size=signal_reference_chunk_size,
        )
    except RuntimeError as exc:
        if str(signal_device).startswith("cuda"):
            raise
        _log(f"batched signal failed; falling back to sklearn reason={exc}")
        signal = _row_dcr_signal(
            pool_df=local_df[column_order],
            train_df=train_ref[column_order],
            test_df=test_ref[column_order],
            schema_card=schema_card,
            column_order=column_order,
            cat_weight=cat_weight,
        )
        signal["signal_backend"] = "sklearn_fallback"
        signal["signal_query_batch_size"] = None
        signal["signal_reference_chunk_size"] = None
        signal["signal_feature_count"] = int(np.asarray(signal["features"]).shape[1])
    signal_elapsed = time.perf_counter() - t_signal
    _log(f"signal done elapsed={signal_elapsed:.2f}s backend={signal.get('signal_backend')}")
    signal_report = {
        "signal_backend": signal.get("signal_backend"),
        "signal_query_batch_size": signal.get("signal_query_batch_size"),
        "signal_reference_chunk_size": signal.get("signal_reference_chunk_size"),
        "signal_feature_count": signal.get("signal_feature_count"),
        "signal_column_source": signal.get("signal_column_source"),
        "signal_column_order": list(signal.get("signal_column_order") or []),
        "signal_column_count": int(signal.get("signal_column_count") or 0),
    }

    is_real_closer = np.asarray(signal["is_real_closer"], dtype=bool)
    base_dcr = float(np.mean(is_real_closer[:local_selected_count]))
    target_margin = abs(float(target_margin))
    lower_target = 0.5 - target_margin
    upper_target = 0.5 + target_margin
    if base_dcr > upper_target:
        reduce_dcr = True
        target_dcr = upper_target
        desired_swaps = int(round((base_dcr - upper_target) * float(len(selected_records))))
    elif base_dcr < lower_target:
        reduce_dcr = False
        target_dcr = lower_target
        desired_swaps = int(round((lower_target - base_dcr) * float(len(selected_records))))
    else:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "base_dcr_within_target_band",
                "bounded_mode": True,
                "sampled_estimate": bool(sampled_estimate),
                "base_dcr_estimate": base_dcr,
                "target_band": [float(lower_target), float(upper_target)],
                "base_dcr_privacy_estimate": float(1.0 - abs(base_dcr - 0.5)),
                "pool_rows": int(len(pool_records)),
                "keep_k": int(len(selected_records)),
                "local_selected_rows": int(local_selected_count),
                "candidate_rows": int(len(add_pool_indices)),
                "reference_rows": [int(len(train_ref)), int(len(test_ref))],
                "candidate_pool": candidate_report,
                **signal_report,
                "signal_elapsed_seconds": float(signal_elapsed),
                "elapsed_seconds": float(time.perf_counter() - t_all),
            },
        )
    max_swaps = max(0, int(round(float(len(selected_records)) * max(0.0, float(max_swap_fraction)))))
    max_swaps = min(max_swaps, max(0, int(large_max_swaps)))
    if desired_swaps <= 0 or max_swaps <= 0:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "no_required_bounded_swaps",
                "bounded_mode": True,
                "sampled_estimate": bool(sampled_estimate),
                "base_dcr_estimate": base_dcr,
                "target_dcr": float(target_dcr),
                "target_band": [float(lower_target), float(upper_target)],
                "desired_swaps": int(desired_swaps),
                "max_swaps": int(max_swaps),
                **signal_report,
            },
        )

    local_selected_rows = np.arange(local_selected_count, dtype=np.int64)
    local_selected_mask = np.zeros(len(local_pool_indices), dtype=bool)
    local_selected_mask[:local_selected_count] = True
    local_utility = utility[local_pool_indices]
    t_pairs = time.perf_counter()
    _log(f"pairs start desired_swaps={desired_swaps} max_swaps={max_swaps} reduce_dcr={reduce_dcr}")
    pair_builder_report: dict[str, Any] = {}
    if _pair_builder is None:
        raw_pairs = _build_pairs(
            selected_pool_indices=local_selected_rows,
            selected_mask=local_selected_mask,
            is_real_closer=is_real_closer,
            margin=np.asarray(signal["margin"], dtype=float),
            features=np.asarray(signal["features"], dtype=np.float32),
            utility_scores=local_utility,
            reduce_dcr=reduce_dcr,
            candidate_neighbors=min(int(candidate_neighbors), int(large_candidate_neighbors)),
            margin_weight=margin_weight,
            utility_weight=utility_weight,
        )
    else:
        quality = np.where(np.isfinite(exact_quality), exact_quality, surrogate_quality)
        built_pairs = _pair_builder(
            selected_pool_indices=local_selected_rows,
            selected_mask=local_selected_mask,
            is_real_closer=is_real_closer,
            margin=np.asarray(signal["margin"], dtype=float),
            features=np.asarray(signal["features"], dtype=np.float32),
            utility_scores=local_utility,
            quality_scores=quality[local_pool_indices],
            local_records=local_records,
            schema_card=schema_card,
            reduce_dcr=reduce_dcr,
            candidate_neighbors=min(int(candidate_neighbors), int(large_candidate_neighbors)),
            margin_weight=margin_weight,
            utility_weight=utility_weight,
            desired_swaps=desired_swaps,
            max_swaps=max_swaps,
        )
        if isinstance(built_pairs, tuple):
            raw_pairs = built_pairs[0]
            pair_builder_report = dict(built_pairs[1])
        else:
            raw_pairs = built_pairs
    pairs, pair_filter_report = _progressive_pair_filter(
        raw_pairs,
        desired_swaps=desired_swaps,
        max_swaps=max_swaps,
        min_pair_utility_gain=min_pair_utility_gain,
        fallback_min_pair_utility_gain=fallback_min_pair_utility_gain,
    )
    pair_elapsed = time.perf_counter() - t_pairs
    selection_swap_budget = int(desired_swaps)
    selected_swaps = min(int(selection_swap_budget), int(max_swaps), int(len(pairs)))
    _log(
        f"pairs done raw={len(raw_pairs)} filtered={len(pairs)} selected_swaps={selected_swaps} "
        f"elapsed={pair_elapsed:.2f}s filter={pair_filter_report}"
    )
    if selected_swaps <= 0:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "no_feasible_bounded_pairs",
                "bounded_mode": True,
                "sampled_estimate": bool(sampled_estimate),
                "base_dcr_estimate": base_dcr,
                "target_dcr": float(target_dcr),
                "target_band": [float(lower_target), float(upper_target)],
                "desired_swaps": int(desired_swaps),
                "max_swaps": int(max_swaps),
                "pair_count": int(len(pairs)),
                "raw_pair_count": int(len(raw_pairs)),
                **signal_report,
                **pair_builder_report,
                **pair_filter_report,
            },
        )

    final_records = [dict(record) for record in selected_records]
    local_selected_flags = is_real_closer[:local_selected_count].copy()
    removed_ids: list[int] = []
    added_ids: list[int] = []
    prefix = pairs[:selected_swaps]
    prefix_extra_report: dict[str, Any] = {}
    if prefix and len(prefix[0]) > 5:
        prefix_extra_report["mean_pair_quality_gain"] = float(np.mean([item[5] for item in prefix]))
    if prefix and len(prefix[0]) > 6:
        prefix_extra_report["selected_target_match_fraction"] = float(np.mean([item[6] for item in prefix]))
    for remove_local_pos, add_local_idx, *_ in prefix:
        actual_selected_pos = int(remove_local_pos)
        actual_add_pool_idx = int(local_pool_indices[int(add_local_idx)])
        final_records[actual_selected_pos] = dict(pool_records[actual_add_pool_idx])
        local_selected_flags[int(remove_local_pos)] = bool(is_real_closer[int(add_local_idx)])
        removed_ids.append(_candidate_id(selected_records[actual_selected_pos], actual_selected_pos))
        added_ids.append(_candidate_id(pool_records[actual_add_pool_idx], actual_add_pool_idx))

    final_dcr = float(np.mean(local_selected_flags))
    final_df = records_to_df(final_records, column_order)
    return (
        final_df,
        final_records,
        {
            **report_base,
            "applied": True,
            "bounded_mode": True,
            "sampled_estimate": bool(sampled_estimate),
            "base_strategy": _base_strategy,
            "base_dcr_estimate": base_dcr,
            "target_dcr": float(target_dcr),
            "target_band": [float(lower_target), float(upper_target)],
            "final_dcr_estimate": final_dcr,
            "base_dcr_privacy_estimate": float(1.0 - abs(base_dcr - 0.5)),
            "final_dcr_privacy_estimate": float(1.0 - abs(final_dcr - 0.5)),
            "desired_swaps": int(desired_swaps),
            "max_swaps": int(max_swaps),
            "selection_swap_budget": int(selection_swap_budget),
            "selected_swaps": int(selected_swaps),
            "pair_count": int(len(pairs)),
            "raw_pair_count": int(len(raw_pairs)),
            **pair_builder_report,
            "reduce_dcr": bool(reduce_dcr),
            "candidate_neighbors": int(min(int(candidate_neighbors), int(large_candidate_neighbors))),
            "margin_weight": float(margin_weight),
            "utility_weight": float(utility_weight),
            "min_pair_utility_gain": float(min_pair_utility_gain),
            "fallback_min_pair_utility_gain": float(fallback_min_pair_utility_gain),
            **pair_filter_report,
            "cat_weight": float(cat_weight),
            "mean_pair_distance": float(np.mean([item[2] for item in prefix])),
            "mean_pair_utility_gain": float(np.mean([item[4] for item in prefix])),
            "sum_pair_utility_gain": float(np.sum([item[4] for item in prefix])),
            **prefix_extra_report,
            "utility_scores": utility_report,
            "pool_rows": int(len(pool_records)),
            "keep_k": int(len(selected_records)),
            "local_selected_rows": int(local_selected_count),
            "candidate_rows": int(len(add_pool_indices)),
            "reference_rows": [int(len(train_ref)), int(len(test_ref))],
            "candidate_pool": candidate_report,
            **signal_report,
            "signal_elapsed_seconds": float(signal_elapsed),
            "pair_elapsed_seconds": float(pair_elapsed),
            "elapsed_seconds": float(time.perf_counter() - t_all),
            "report_id_limit": int(report_id_limit),
            "removed_candidate_ids": _truncate_ids(removed_ids, report_id_limit),
            "added_candidate_ids": _truncate_ids(added_ids, report_id_limit),
            "removed_candidate_id_count": int(len(removed_ids)),
            "added_candidate_id_count": int(len(added_ids)),
        },
    )
