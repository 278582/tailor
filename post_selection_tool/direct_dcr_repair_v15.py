from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .direct_dcr_repair_v10 import apply_direct_dcr_repair_v10
from .direct_dcr_repair_v13 import _build_target_then_limited_generic_pairs
from .direct_dcr_repair_v11 import _target_group_labels


def _build_duplicate_fill_pairs(
    *,
    selected_pool_indices: np.ndarray,
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
    quality_floor: float,
    require_target_match: bool,
    gain_quantile: float | None,
    fill_needed: int,
    used_remove: set[int],
) -> tuple[list[tuple[int, int, float, float, float, float, int]], dict[str, Any]]:
    if fill_needed <= 0:
        return [], {"duplicate_fill_reason": "no_fill_needed"}

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
            if idx not in selected_set and bool(is_real_closer[idx]) != bool(reduce_dcr)
        ],
        dtype=np.int64,
    )
    if remove_positions.size == 0 or add_pool_indices.size == 0:
        return [], {
            "duplicate_fill_reason": "empty_remaining_remove_or_add",
            "duplicate_fill_remove_rows": int(remove_positions.size),
            "duplicate_fill_add_rows": int(add_pool_indices.size),
        }

    n_neighbors = min(max(1, int(candidate_neighbors)), int(add_pool_indices.size))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="manhattan", algorithm="auto")
    model.fit(features[add_pool_indices])
    distances, neighbor_positions = model.kneighbors(features[selected_pool_indices[remove_positions]])

    candidates: list[tuple[int, int, float, float, float, float, int]] = []
    for row_pos, remove_base_pos in enumerate(remove_positions.tolist()):
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
            if quality_gain < float(quality_floor):
                continue
            target_match = int(str(target_labels[remove_pool_idx]) == str(target_labels[add_idx]))
            if bool(require_target_match) and not target_match:
                continue
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

    gain_filter_report: dict[str, Any] = {
        "duplicate_fill_gain_quantile": None,
        "duplicate_fill_utility_gain_floor": None,
        "duplicate_fill_quality_gain_floor": None,
        "duplicate_fill_candidate_edges_before_gain_filter": int(len(candidates)),
    }
    if gain_quantile is not None and candidates:
        quantile = min(max(float(gain_quantile), 0.0), 1.0)
        utility_floor = float(np.quantile([pair[4] for pair in candidates], quantile))
        quality_gain_floor = float(np.quantile([pair[5] for pair in candidates], quantile))
        candidates = [
            pair
            for pair in candidates
            if float(pair[4]) >= utility_floor and float(pair[5]) >= quality_gain_floor
        ]
        gain_filter_report = {
            "duplicate_fill_gain_quantile": float(quantile),
            "duplicate_fill_utility_gain_floor": utility_floor,
            "duplicate_fill_quality_gain_floor": quality_gain_floor,
            "duplicate_fill_candidate_edges_before_gain_filter": int(
                gain_filter_report["duplicate_fill_candidate_edges_before_gain_filter"]
            ),
        }

    candidates.sort(key=lambda item: (0 if int(item[6]) == 1 else 1, item[3], item[2]))
    pairs: list[tuple[int, int, float, float, float, float, int]] = []
    local_used_remove = set(used_remove)
    reused_adds: set[int] = set()
    for pair in candidates:
        remove_pos = int(pair[0])
        add_idx = int(pair[1])
        if remove_pos in local_used_remove:
            continue
        local_used_remove.add(remove_pos)
        reused_adds.add(add_idx)
        pairs.append(pair)
        if len(pairs) >= int(fill_needed):
            break

    return pairs, {
        "duplicate_fill_reason": "nearest_reuse_adds",
        "duplicate_fill_remove_rows": int(remove_positions.size),
        "duplicate_fill_add_rows": int(add_pool_indices.size),
        "duplicate_fill_candidate_edges": int(len(candidates)),
        "duplicate_fill_unique_reused_add_rows": int(len(reused_adds)),
        "duplicate_fill_allows_duplicate_adds": True,
        **gain_filter_report,
    }


def apply_direct_dcr_repair_v15(
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
    originally_large = (
        len(selected_records) > int(large_keep_k_threshold)
        or len(pool_records) > int(large_pool_rows_threshold)
    )
    if originally_large:
        effective_candidate_rows = int(large_candidate_rows)
        effective_candidate_neighbors = int(large_candidate_neighbors)
        effective_generic_remove_budget = int(generic_remove_budget)
    else:
        effective_candidate_rows = max(0, int(len(pool_records)))
        effective_candidate_neighbors = max(int(candidate_neighbors), int(large_candidate_neighbors))
        effective_generic_remove_budget = max(int(generic_remove_budget), int(len(selected_records)))

    def pair_builder(**kwargs: Any) -> tuple[list[tuple[int, int, float, float, float, float, int]], dict[str, Any]]:
        theta_guidance_enabled = bool(kwargs["schema_card"].get("theta_guidance_enabled", False))
        target_limit = max(0, min(int(kwargs["desired_swaps"]), int(kwargs["max_swaps"])))
        unique_pairs, unique_report = _build_target_then_limited_generic_pairs(
            **kwargs,
            target_bins=target_bins,
            quality_weight=quality_weight,
            target_mismatch_penalty=target_mismatch_penalty,
            min_pair_utility_gain=min_pair_utility_gain,
            fallback_min_pair_utility_gain=fallback_min_pair_utility_gain,
            generic_remove_budget=effective_generic_remove_budget,
        )
        fill_needed = max(0, target_limit - len(unique_pairs))
        duplicate_fill_limit = int(fill_needed)
        duplicate_fill_budget_report = {
            "duplicate_fill_budget_mode": "shared_full_fill",
            "duplicate_fill_support_ratio": None,
            "duplicate_fill_support_surplus": None,
        }
        if duplicate_fill_limit > 0:
            active_floor = min(float(min_pair_utility_gain), float(fallback_min_pair_utility_gain))
            target_labels, _ = _target_group_labels(kwargs["local_records"], kwargs["schema_card"], target_bins)
            duplicate_pairs, duplicate_report = _build_duplicate_fill_pairs(
                selected_pool_indices=kwargs["selected_pool_indices"],
                is_real_closer=kwargs["is_real_closer"],
                margin=kwargs["margin"],
                features=kwargs["features"],
                utility_scores=kwargs["utility_scores"],
                quality_scores=kwargs["quality_scores"],
                target_labels=target_labels,
                reduce_dcr=kwargs["reduce_dcr"],
                candidate_neighbors=kwargs["candidate_neighbors"],
                margin_weight=kwargs["margin_weight"],
                utility_weight=kwargs["utility_weight"],
                quality_weight=quality_weight,
                target_mismatch_penalty=target_mismatch_penalty,
                active_floor=active_floor,
                quality_floor=float("-inf"),
                require_target_match=False,
                gain_quantile=None,
                fill_needed=duplicate_fill_limit,
                used_remove={int(pair[0]) for pair in unique_pairs},
            )
        else:
            duplicate_pairs = []
            duplicate_report = {
                "duplicate_fill_reason": "no_duplicate_fill_budget",
                "duplicate_fill_remove_rows": 0,
                "duplicate_fill_add_rows": 0,
                "duplicate_fill_candidate_edges": 0,
                "duplicate_fill_unique_reused_add_rows": 0,
                "duplicate_fill_allows_duplicate_adds": False,
            }
        pairs = [*unique_pairs, *duplicate_pairs]
        same_target_count = int(sum(1 for pair in pairs if int(pair[6]) == 1))
        return pairs, {
            **unique_report,
            **duplicate_report,
            "pair_builder_mode": "target_then_limited_generic_duplicate_fill",
            "theta_guidance_enabled": bool(theta_guidance_enabled),
            **duplicate_fill_budget_report,
            "duplicate_fill_limit": int(duplicate_fill_limit),
            "duplicate_fill_needed": int(fill_needed),
            "unique_pair_count_before_duplicate_fill": int(len(unique_pairs)),
            "duplicate_fill_pair_count": int(len(duplicate_pairs)),
            "same_target_pair_count": same_target_count,
            "cross_target_pair_count": int(len(pairs) - same_target_count),
            "force_bounded_small_mode": not originally_large,
            "effective_candidate_rows": int(effective_candidate_rows),
            "effective_large_candidate_neighbors": int(effective_candidate_neighbors),
        }

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
        large_keep_k_threshold=0,
        large_pool_rows_threshold=0,
        large_candidate_rows=effective_candidate_rows,
        large_reference_rows=large_reference_rows,
        large_max_swaps=large_max_swaps,
        large_candidate_neighbors=effective_candidate_neighbors,
        min_pair_utility_gain=min_pair_utility_gain,
        fallback_min_pair_utility_gain=fallback_min_pair_utility_gain,
        signal_query_batch_size=signal_query_batch_size,
        signal_reference_chunk_size=signal_reference_chunk_size,
        signal_device=signal_device,
        report_id_limit=report_id_limit,
        _report_version="direct_dcr_repair_v15",
        _selection_signal="tabdiff_full_reference_l1_dcr_force_bounded_duplicate_fill",
        _base_strategy="full_reference_force_bounded_duplicate_fill_v15",
        _pair_builder=pair_builder,
    )
