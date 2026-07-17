from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .direct_dcr_repair_v10 import apply_direct_dcr_repair_v10
from .direct_dcr_repair_v11 import _build_target_aware_pairs, _target_group_labels


def _build_limited_generic_fill_pairs(
    *,
    selected_pool_indices: np.ndarray,
    selected_mask: np.ndarray,
    is_real_closer: np.ndarray,
    margin: np.ndarray,
    features: np.ndarray,
    utility_scores: np.ndarray,
    quality_scores: np.ndarray,
    target_labels: np.ndarray,
    reduce_dcr: bool,
    candidate_neighbors: int,
    margin_weight: float,
    utility_weight: float,
    quality_weight: float,
    target_mismatch_penalty: float,
    active_floor: float,
    fill_needed: int,
    used_remove: set[int],
    used_add: set[int],
    generic_remove_budget: int,
) -> tuple[list[tuple[int, int, float, float, float, float, int]], dict[str, Any]]:
    if fill_needed <= 0:
        return [], {
            "generic_fill_reason": "no_fill_needed",
            "generic_remove_rows": 0,
            "generic_add_rows": 0,
        }

    selected_set = set(int(idx) for idx in selected_pool_indices.tolist())
    remove_positions = np.asarray(
        [
            pos
            for pos, pool_idx in enumerate(selected_pool_indices.tolist())
            if pos not in used_remove and bool(is_real_closer[int(pool_idx)]) == bool(reduce_dcr)
        ],
        dtype=np.int64,
    )
    add_pool_indices = np.asarray(
        [
            idx
            for idx in range(len(is_real_closer))
            if idx not in selected_set
            and idx not in used_add
            and bool(is_real_closer[idx]) != bool(reduce_dcr)
        ],
        dtype=np.int64,
    )
    if remove_positions.size == 0 or add_pool_indices.size == 0:
        return [], {
            "generic_fill_reason": "empty_remaining_remove_or_add",
            "generic_remove_rows": int(remove_positions.size),
            "generic_add_rows": int(add_pool_indices.size),
        }

    remove_pool_indices = selected_pool_indices[remove_positions]
    priority = np.abs(margin[remove_pool_indices])
    order = np.argsort(-priority)
    budget = min(int(remove_positions.size), max(int(fill_needed), int(generic_remove_budget)))
    budget = max(0, int(budget))
    budget_remove_positions = remove_positions[order[:budget]]
    budget_remove_pool_indices = selected_pool_indices[budget_remove_positions]
    n_neighbors = min(max(1, int(candidate_neighbors)), int(add_pool_indices.size))

    model = NearestNeighbors(n_neighbors=n_neighbors, metric="manhattan", algorithm="auto")
    model.fit(features[add_pool_indices])
    distances, neighbor_positions = model.kneighbors(features[budget_remove_pool_indices])

    candidates: list[tuple[int, int, float, float, float, float, int]] = []
    for row_pos, remove_base_pos in enumerate(budget_remove_positions.tolist()):
        remove_pool_idx = int(selected_pool_indices[int(remove_base_pos)])
        remove_margin = abs(float(margin[remove_pool_idx]))
        remove_utility = float(utility_scores[remove_pool_idx])
        remove_quality = float(quality_scores[remove_pool_idx])
        for neighbor_pos in range(n_neighbors):
            add_idx = int(add_pool_indices[int(neighbor_positions[row_pos, neighbor_pos])])
            add_margin = abs(float(margin[add_idx]))
            utility_gain = float(utility_scores[add_idx] - remove_utility)
            if utility_gain < float(active_floor):
                continue
            quality_gain = float(quality_scores[add_idx] - remove_quality)
            target_match = int(str(target_labels[remove_pool_idx]) == str(target_labels[add_idx]))
            distance = float(distances[row_pos, neighbor_pos])
            score = distance
            if not target_match:
                score += float(target_mismatch_penalty)
            score -= float(margin_weight) * (remove_margin + add_margin)
            score -= float(utility_weight) * utility_gain
            score -= float(quality_weight) * quality_gain
            candidates.append(
                (
                    int(remove_base_pos),
                    int(add_idx),
                    distance,
                    float(score),
                    utility_gain,
                    quality_gain,
                    target_match,
                )
            )

    candidates.sort(key=lambda item: (0 if int(item[6]) == 1 else 1, item[3], item[2]))
    pairs: list[tuple[int, int, float, float, float, float, int]] = []
    local_used_remove = set(used_remove)
    local_used_add = set(used_add)
    for pair in candidates:
        remove_pos = int(pair[0])
        add_idx = int(pair[1])
        if remove_pos in local_used_remove or add_idx in local_used_add:
            continue
        local_used_remove.add(remove_pos)
        local_used_add.add(add_idx)
        pairs.append(pair)
        if len(pairs) >= int(fill_needed):
            break

    return pairs, {
        "generic_fill_reason": "limited_remaining_nearest",
        "generic_remove_rows": int(remove_positions.size),
        "generic_add_rows": int(add_pool_indices.size),
        "generic_remove_budget": int(budget),
        "generic_candidate_edges": int(len(candidates)),
    }


def _build_target_then_limited_generic_pairs(
    *,
    selected_pool_indices: np.ndarray,
    selected_mask: np.ndarray,
    is_real_closer: np.ndarray,
    margin: np.ndarray,
    features: np.ndarray,
    utility_scores: np.ndarray,
    quality_scores: np.ndarray,
    local_records: list[dict[str, Any]],
    schema_card: dict[str, Any],
    reduce_dcr: bool,
    candidate_neighbors: int,
    margin_weight: float,
    utility_weight: float,
    desired_swaps: int,
    max_swaps: int,
    target_bins: int,
    quality_weight: float,
    target_mismatch_penalty: float,
    min_pair_utility_gain: float,
    fallback_min_pair_utility_gain: float,
    generic_remove_budget: int,
) -> tuple[list[tuple[int, int, float, float, float, float, int]], dict[str, Any]]:
    target_limit = max(0, min(int(desired_swaps), int(max_swaps)))
    active_floor = min(float(min_pair_utility_gain), float(fallback_min_pair_utility_gain))
    same_pairs_raw, same_report = _build_target_aware_pairs(
        selected_pool_indices=selected_pool_indices,
        selected_mask=selected_mask,
        is_real_closer=is_real_closer,
        margin=margin,
        features=features,
        utility_scores=utility_scores,
        quality_scores=quality_scores,
        local_records=local_records,
        schema_card=schema_card,
        reduce_dcr=reduce_dcr,
        candidate_neighbors=candidate_neighbors,
        margin_weight=margin_weight,
        utility_weight=utility_weight,
        desired_swaps=desired_swaps,
        max_swaps=max_swaps,
        target_bins=target_bins,
        quality_weight=quality_weight,
        target_mismatch_penalty=target_mismatch_penalty,
        allow_cross_target_fallback=False,
    )
    same_pairs = [
        pair
        for pair in same_pairs_raw
        if float(pair[4]) >= active_floor
    ][:target_limit]
    used_remove = {int(pair[0]) for pair in same_pairs}
    used_add = {int(pair[1]) for pair in same_pairs}
    target_labels, target_report = _target_group_labels(local_records, schema_card, target_bins)

    fill_needed = max(0, target_limit - len(same_pairs))
    generic_fill_pairs, generic_report = _build_limited_generic_fill_pairs(
        selected_pool_indices=selected_pool_indices,
        selected_mask=selected_mask,
        is_real_closer=is_real_closer,
        margin=margin,
        features=features,
        utility_scores=utility_scores,
        quality_scores=quality_scores,
        target_labels=target_labels,
        reduce_dcr=reduce_dcr,
        candidate_neighbors=candidate_neighbors,
        margin_weight=margin_weight,
        utility_weight=utility_weight,
        quality_weight=quality_weight,
        target_mismatch_penalty=target_mismatch_penalty,
        active_floor=active_floor,
        fill_needed=fill_needed,
        used_remove=used_remove,
        used_add=used_add,
        generic_remove_budget=generic_remove_budget,
    )

    pairs = [*same_pairs, *generic_fill_pairs]
    same_selected_count = int(sum(1 for pair in pairs if int(pair[6]) == 1))
    generic_target_match_count = int(sum(1 for pair in generic_fill_pairs if int(pair[6]) == 1))
    return pairs, {
        "pair_builder_mode": "target_then_limited_generic_fill",
        **target_report,
        "target_aware_stage_mode": same_report.get("pair_builder_mode"),
        "target_aware_raw_pair_count": int(len(same_pairs_raw)),
        "target_aware_floor_pair_count": int(len(same_pairs)),
        "target_aware_selected_pair_count": int(len(same_pairs)),
        "generic_fill_pair_count": int(len(generic_fill_pairs)),
        "generic_fill_target_match_count": generic_target_match_count,
        **generic_report,
        "same_target_pair_count": same_selected_count,
        "cross_target_pair_count": int(len(pairs) - same_selected_count),
        "target_aware_target_pairs": int(target_limit),
        "target_mismatch_penalty": float(target_mismatch_penalty),
        "quality_weight": float(quality_weight),
        "active_pair_utility_floor_for_builder": float(active_floor),
        "same_target_candidate_edges": int(same_report.get("same_target_candidate_edges", 0)),
        "cross_target_candidate_edges": int(generic_report.get("generic_candidate_edges", 0)),
    }


def apply_direct_dcr_repair_v13(
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
    utility_weight: float = 0.65,
    cat_weight: float = 1.0,
    large_keep_k_threshold: int = 50_000,
    large_pool_rows_threshold: int = 180_000,
    large_candidate_rows: int = 72_000,
    large_reference_rows: int = 0,
    large_max_swaps: int = 20_000,
    large_candidate_neighbors: int = 28,
    min_pair_utility_gain: float = -0.08,
    fallback_min_pair_utility_gain: float = -0.18,
    signal_query_batch_size: int = 0,
    signal_reference_chunk_size: int = 65536,
    signal_device: str = "auto",
    report_id_limit: int = 64,
    target_bins: int = 12,
    quality_weight: float = 0.20,
    target_mismatch_penalty: float = 4.0,
    generic_remove_budget: int = 20_000,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    def pair_builder(**kwargs: Any) -> tuple[list[tuple[int, int, float, float, float, float, int]], dict[str, Any]]:
        return _build_target_then_limited_generic_pairs(
            **kwargs,
            target_bins=target_bins,
            quality_weight=quality_weight,
            target_mismatch_penalty=target_mismatch_penalty,
            min_pair_utility_gain=min_pair_utility_gain,
            fallback_min_pair_utility_gain=fallback_min_pair_utility_gain,
            generic_remove_budget=generic_remove_budget,
        )

    return apply_direct_dcr_repair_v10(
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
        large_keep_k_threshold=large_keep_k_threshold,
        large_pool_rows_threshold=large_pool_rows_threshold,
        large_candidate_rows=large_candidate_rows,
        large_reference_rows=large_reference_rows,
        large_max_swaps=large_max_swaps,
        large_candidate_neighbors=large_candidate_neighbors,
        min_pair_utility_gain=min_pair_utility_gain,
        fallback_min_pair_utility_gain=fallback_min_pair_utility_gain,
        signal_query_batch_size=signal_query_batch_size,
        signal_reference_chunk_size=signal_reference_chunk_size,
        signal_device=signal_device,
        report_id_limit=report_id_limit,
        _report_version="direct_dcr_repair_v13",
        _selection_signal="tabdiff_full_reference_l1_dcr_with_target_then_limited_generic_fill",
        _base_strategy="full_reference_target_then_limited_generic_fill_v13",
        _pair_builder=pair_builder,
    )
