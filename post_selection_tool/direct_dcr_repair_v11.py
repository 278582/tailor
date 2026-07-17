from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .direct_dcr_repair_v10 import apply_direct_dcr_repair_v10


def _target_group_labels(
    local_records: list[dict[str, Any]],
    schema_card: dict[str, Any],
    target_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_column = str(schema_card.get("target_column") or "")
    if not target_column:
        return np.full(len(local_records), "__all__", dtype=object), {
            "target_group_mode": "missing_target_column",
            "target_group_count": 1,
        }

    values = [record.get("row", {}).get(target_column) for record in local_records]
    series = pd.Series(values)
    column_info = dict(schema_card.get("columns", {}).get(target_column, {}))
    column_type = str(column_info.get("type", ""))
    unique_raw = int(series.dropna().astype(str).nunique())
    target_bins = max(2, int(target_bins))

    if column_type in {"numerical", "discrete_numerical"} and unique_raw > target_bins:
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        finite_unique = int(finite.nunique())
        if finite_unique >= 2:
            bin_count = min(target_bins, finite_unique)
            edges = np.unique(np.quantile(finite.to_numpy(dtype=float), np.linspace(0.0, 1.0, bin_count + 1)))
            if edges.size >= 2:
                labels: list[str] = []
                for value in numeric.to_numpy(dtype=float):
                    if not np.isfinite(value):
                        labels.append("__target_missing__")
                    else:
                        bin_idx = int(np.searchsorted(edges[1:-1], float(value), side="right"))
                        labels.append(f"target_bin_{bin_idx}")
                return np.asarray(labels, dtype=object), {
                    "target_group_mode": "numeric_quantile_bins",
                    "target_group_count": int(len(set(labels))),
                    "target_bins": int(bin_count),
                    "target_column": target_column,
                }

    labels = series.fillna("__target_missing__").astype(str).to_numpy(dtype=object)
    return labels, {
        "target_group_mode": "exact_target_labels",
        "target_group_count": int(len(set(labels.tolist()))),
        "target_column": target_column,
    }


def _collect_nearest_candidates(
    *,
    selected_pool_indices: np.ndarray,
    remove_base_positions: np.ndarray,
    add_pool_indices: np.ndarray,
    margin: np.ndarray,
    features: np.ndarray,
    utility_scores: np.ndarray,
    quality_scores: np.ndarray,
    candidate_neighbors: int,
    margin_weight: float,
    utility_weight: float,
    quality_weight: float,
    target_mismatch_penalty: float,
    target_match: bool,
) -> list[tuple[int, int, float, float, float, float, int]]:
    if remove_base_positions.size == 0 or add_pool_indices.size == 0:
        return []

    n_neighbors = min(max(1, int(candidate_neighbors)), int(add_pool_indices.size))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="manhattan", algorithm="auto")
    model.fit(features[add_pool_indices])
    remove_pool_indices = selected_pool_indices[remove_base_positions]
    distances, neighbor_positions = model.kneighbors(features[remove_pool_indices])

    candidates: list[tuple[int, int, float, float, float, float, int]] = []
    target_penalty = 0.0 if target_match else float(target_mismatch_penalty)
    target_flag = 1 if target_match else 0
    for row_pos, remove_base_pos in enumerate(remove_base_positions.tolist()):
        remove_pool_idx = int(selected_pool_indices[int(remove_base_pos)])
        remove_margin = abs(float(margin[remove_pool_idx]))
        remove_utility = float(utility_scores[remove_pool_idx])
        remove_quality = float(quality_scores[remove_pool_idx])
        for neighbor_pos in range(n_neighbors):
            add_idx = int(add_pool_indices[int(neighbor_positions[row_pos, neighbor_pos])])
            add_margin = abs(float(margin[add_idx]))
            utility_gain = float(utility_scores[add_idx] - remove_utility)
            quality_gain = float(quality_scores[add_idx] - remove_quality)
            distance = float(distances[row_pos, neighbor_pos])
            score = distance + target_penalty
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
                    target_flag,
                )
            )
    return candidates


def _greedy_unique_pairs(
    candidates: list[tuple[int, int, float, float, float, float, int]]
) -> list[tuple[int, int, float, float, float, float, int]]:
    candidates.sort(key=lambda item: (0 if int(item[6]) == 1 else 1, item[3], item[2]))
    used_remove: set[int] = set()
    used_add: set[int] = set()
    pairs: list[tuple[int, int, float, float, float, float, int]] = []
    for pair in candidates:
        remove_base_pos = int(pair[0])
        add_idx = int(pair[1])
        if remove_base_pos in used_remove or add_idx in used_add:
            continue
        used_remove.add(remove_base_pos)
        used_add.add(add_idx)
        pairs.append(pair)
    return pairs


def _build_target_aware_pairs(
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
    allow_cross_target_fallback: bool,
) -> tuple[list[tuple[int, int, float, float, float, float, int]], dict[str, Any]]:
    selected_set = set(int(idx) for idx in selected_pool_indices.tolist())
    remove_base_positions = np.asarray(
        [
            pos
            for pos, pool_idx in enumerate(selected_pool_indices.tolist())
            if bool(is_real_closer[int(pool_idx)]) == bool(reduce_dcr)
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
    if remove_base_positions.size == 0 or add_pool_indices.size == 0:
        return [], {
            "pair_builder_mode": "target_aware_grouped",
            "target_aware_reason": "empty_remove_or_add_pool",
        }

    target_labels, target_report = _target_group_labels(local_records, schema_card, target_bins)
    remove_pool_indices = selected_pool_indices[remove_base_positions]
    same_candidates: list[tuple[int, int, float, float, float, float, int]] = []
    cross_candidates: list[tuple[int, int, float, float, float, float, int]] = []

    remove_groups = sorted(set(target_labels[remove_pool_indices].tolist()))
    add_labels = target_labels[add_pool_indices]
    target_limit = max(0, min(int(desired_swaps), int(max_swaps)))
    for group in remove_groups:
        group_remove_positions = remove_base_positions[target_labels[remove_pool_indices] == group]
        group_add_indices = add_pool_indices[add_labels == group]
        same_candidates.extend(
            _collect_nearest_candidates(
                selected_pool_indices=selected_pool_indices,
                remove_base_positions=group_remove_positions,
                add_pool_indices=group_add_indices,
                margin=margin,
                features=features,
                utility_scores=utility_scores,
                quality_scores=quality_scores,
                candidate_neighbors=candidate_neighbors,
                margin_weight=margin_weight,
                utility_weight=utility_weight,
                quality_weight=quality_weight,
                target_mismatch_penalty=target_mismatch_penalty,
                target_match=True,
            )
        )

    same_pairs = _greedy_unique_pairs(same_candidates)
    if allow_cross_target_fallback and len(same_pairs) < target_limit:
        for group in remove_groups:
            group_remove_positions = remove_base_positions[target_labels[remove_pool_indices] == group]
            group_add_indices = add_pool_indices[add_labels != group]
            cross_candidates.extend(
                _collect_nearest_candidates(
                    selected_pool_indices=selected_pool_indices,
                    remove_base_positions=group_remove_positions,
                    add_pool_indices=group_add_indices,
                    margin=margin,
                    features=features,
                    utility_scores=utility_scores,
                    quality_scores=quality_scores,
                    candidate_neighbors=candidate_neighbors,
                    margin_weight=margin_weight,
                    utility_weight=utility_weight,
                    quality_weight=quality_weight,
                    target_mismatch_penalty=target_mismatch_penalty,
                    target_match=False,
                )
            )

    pairs = _greedy_unique_pairs([*same_candidates, *cross_candidates])
    same_pair_count = int(sum(1 for pair in pairs if int(pair[6]) == 1))
    return pairs, {
        "pair_builder_mode": "target_aware_grouped",
        **target_report,
        "target_aware_remove_rows": int(remove_base_positions.size),
        "target_aware_add_rows": int(add_pool_indices.size),
        "same_target_candidate_edges": int(len(same_candidates)),
        "cross_target_candidate_edges": int(len(cross_candidates)),
        "same_target_pair_count": same_pair_count,
        "cross_target_pair_count": int(len(pairs) - same_pair_count),
        "target_aware_target_pairs": int(target_limit),
        "target_mismatch_penalty": float(target_mismatch_penalty),
        "quality_weight": float(quality_weight),
        "allow_cross_target_fallback": bool(allow_cross_target_fallback),
    }


def apply_direct_dcr_repair_v11(
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
    target_mismatch_penalty: float = 8.0,
    allow_cross_target_fallback: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    def pair_builder(**kwargs: Any) -> tuple[list[tuple[int, int, float, float, float, float, int]], dict[str, Any]]:
        return _build_target_aware_pairs(
            **kwargs,
            target_bins=target_bins,
            quality_weight=quality_weight,
            target_mismatch_penalty=target_mismatch_penalty,
            allow_cross_target_fallback=allow_cross_target_fallback,
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
        _report_version="direct_dcr_repair_v11",
        _selection_signal="tabdiff_full_reference_l1_dcr_with_target_aware_pairs",
        _base_strategy="full_reference_target_aware_pairs_v11",
        _pair_builder=pair_builder,
    )
