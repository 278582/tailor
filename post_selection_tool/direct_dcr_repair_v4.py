from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from .io import records_to_df


def _candidate_id(record: dict[str, Any], fallback: int) -> int:
    try:
        return int(record.get("candidate_id", fallback))
    except (TypeError, ValueError):
        return int(fallback)


def _finite_float(record: dict[str, Any], key: str, default: float = 0.5) -> float:
    try:
        value = float(record.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _dcr_signal_column_order(schema_card: dict[str, Any], column_order: list[str]) -> list[str]:
    raw_columns = schema_card.get("dcr_signal_column_order")
    if raw_columns is None:
        return list(column_order)
    if isinstance(raw_columns, str):
        requested = [raw_columns]
    else:
        requested = list(raw_columns)
    resolved = list(dict.fromkeys(str(column).strip() for column in requested if str(column).strip()))
    if not resolved:
        raise ValueError("dcr_signal_column_order must not be empty when provided.")
    known_columns = set(column_order)
    unknown = [column for column in resolved if column not in known_columns]
    if unknown:
        raise ValueError(f"dcr_signal_column_order contains unknown columns: {unknown}")
    return resolved


def _feature_matrix(
    *,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_card: dict[str, Any],
    column_order: list[str],
    cat_weight: float,
    encoder: OneHotEncoder | None = None,
) -> tuple[np.ndarray, OneHotEncoder | None]:
    columns = schema_card["columns"]
    feature_column_order = _dcr_signal_column_order(schema_card, column_order)
    numeric_columns = [
        column
        for column in feature_column_order
        if columns[column]["type"] in {"numerical", "discrete_numerical"}
    ]
    categorical_columns = [
        column
        for column in feature_column_order
        if columns[column]["type"] == "categorical"
    ]

    parts: list[np.ndarray] = []
    if numeric_columns:
        ranges = (
            train_df[numeric_columns].astype(float).max(axis=0).to_numpy(dtype=float)
            - train_df[numeric_columns].astype(float).min(axis=0).to_numpy(dtype=float)
        )
        ranges = np.where(np.abs(ranges) < 1e-12, 1.0, ranges)
        numeric = df[numeric_columns].astype(float).to_numpy(dtype=float) / ranges
        parts.append(np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0))

    fitted_encoder = encoder
    if categorical_columns:
        if fitted_encoder is None:
            fitted_encoder = _make_one_hot_encoder()
            fit_values = pd.concat(
                [
                    train_df[categorical_columns].astype(str),
                    test_df[categorical_columns].astype(str),
                ],
                axis=0,
            ).to_numpy()
            fitted_encoder.fit(fit_values)
        categorical = fitted_encoder.transform(df[categorical_columns].astype(str).to_numpy())
        parts.append(np.asarray(categorical, dtype=float) * float(cat_weight))

    if not parts:
        return np.zeros((len(df), 0), dtype=np.float32), fitted_encoder
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False), fitted_encoder


def _row_dcr_signal(
    *,
    pool_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema_card: dict[str, Any],
    column_order: list[str],
    cat_weight: float,
) -> dict[str, np.ndarray]:
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
    train_nn = NearestNeighbors(n_neighbors=1, metric="manhattan", algorithm="auto")
    test_nn = NearestNeighbors(n_neighbors=1, metric="manhattan", algorithm="auto")
    train_nn.fit(train_matrix)
    test_nn.fit(test_matrix)
    dcr_real = train_nn.kneighbors(pool_matrix, return_distance=True)[0][:, 0]
    dcr_test = test_nn.kneighbors(pool_matrix, return_distance=True)[0][:, 0]
    return {
        "features": pool_matrix,
        "dcr_real": np.asarray(dcr_real, dtype=float),
        "dcr_test": np.asarray(dcr_test, dtype=float),
        "is_real_closer": np.asarray(dcr_real < dcr_test, dtype=bool),
        "margin": np.asarray(dcr_test - dcr_real, dtype=float),
        "signal_column_order": signal_column_order,
        "signal_column_count": int(len(signal_column_order)),
        "signal_column_source": str(schema_card.get("dcr_signal_column_source", "full_column_order")),
    }


def _utility_scores(
    *,
    pool_records: list[dict[str, Any]],
    exact_records: list[dict[str, Any]],
    surrogate_records: list[dict[str, Any]] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.full(len(pool_records), 0.5, dtype=float)
    pool_id_to_index = {
        _candidate_id(record, idx): idx
        for idx, record in enumerate(pool_records)
    }
    exact_filled = 0
    for record in exact_records:
        idx = pool_id_to_index.get(_candidate_id(record, -1))
        if idx is None:
            continue
        scores[idx] = np.clip(_finite_float(record, "pareto_util_proxy_obj", 0.5), 0.0, 1.0)
        exact_filled += 1

    surrogate_filled = 0
    if surrogate_records is not None:
        for record in surrogate_records:
            idx = pool_id_to_index.get(_candidate_id(record, -1))
            if idx is None or scores[idx] != 0.5:
                continue
            for key in ("s_preselect_stage_b", "s_preselect_band", "s_fid_sur"):
                if key in record:
                    scores[idx] = np.clip(_finite_float(record, key, 0.5), 0.0, 1.0)
                    surrogate_filled += 1
                    break
    return scores, {
        "exact_filled_rows": int(exact_filled),
        "surrogate_filled_rows": int(surrogate_filled),
        "default_rows": int(len(scores) - exact_filled - surrogate_filled),
    }


def _build_pairs(
    *,
    selected_pool_indices: np.ndarray,
    selected_mask: np.ndarray,
    is_real_closer: np.ndarray,
    margin: np.ndarray,
    features: np.ndarray,
    utility_scores: np.ndarray,
    reduce_dcr: bool,
    candidate_neighbors: int,
    margin_weight: float,
    utility_weight: float,
) -> list[tuple[int, int, float, float, float]]:
    selected_set = set(int(idx) for idx in selected_pool_indices.tolist())
    remove_base_positions = np.asarray(
        [
            pos
            for pos, pool_idx in enumerate(selected_pool_indices.tolist())
            if bool(is_real_closer[int(pool_idx)]) == bool(reduce_dcr)
        ],
        dtype=np.int64,
    )
    remove_pool_indices = selected_pool_indices[remove_base_positions]
    add_pool_indices = np.asarray(
        [
            idx
            for idx in range(len(is_real_closer))
            if idx not in selected_set and bool(is_real_closer[idx]) != bool(reduce_dcr)
        ],
        dtype=np.int64,
    )
    if remove_base_positions.size == 0 or add_pool_indices.size == 0:
        return []

    n_neighbors = min(max(1, int(candidate_neighbors)), int(add_pool_indices.size))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="manhattan", algorithm="auto")
    model.fit(features[add_pool_indices])
    distances, neighbor_positions = model.kneighbors(features[remove_pool_indices])

    candidates: list[tuple[float, int, int, float, float]] = []
    for row_pos, remove_base_pos in enumerate(remove_base_positions.tolist()):
        remove_pool_idx = int(selected_pool_indices[int(remove_base_pos)])
        remove_margin = abs(float(margin[remove_pool_idx]))
        remove_utility = float(utility_scores[remove_pool_idx])
        for neighbor_pos in range(n_neighbors):
            add_idx = int(add_pool_indices[int(neighbor_positions[row_pos, neighbor_pos])])
            add_margin = abs(float(margin[add_idx]))
            utility_gain = float(utility_scores[add_idx] - remove_utility)
            distance = float(distances[row_pos, neighbor_pos])
            score = distance - float(margin_weight) * (remove_margin + add_margin)
            score -= float(utility_weight) * utility_gain
            candidates.append((score, int(remove_base_pos), add_idx, distance, utility_gain))

    candidates.sort(key=lambda item: (item[0], item[3]))
    used_remove: set[int] = set()
    used_add: set[int] = set()
    pairs: list[tuple[int, int, float, float, float]] = []
    for score, remove_base_pos, add_idx, distance, utility_gain in candidates:
        if remove_base_pos in used_remove or add_idx in used_add:
            continue
        used_remove.add(remove_base_pos)
        used_add.add(add_idx)
        pairs.append((remove_base_pos, add_idx, distance, score, utility_gain))
    return pairs


def apply_direct_dcr_repair_v4(
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
    max_swap_fraction: float = 0.22,
    candidate_neighbors: int = 64,
    margin_weight: float = 0.05,
    utility_weight: float = 0.35,
    cat_weight: float = 1.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    report_base: dict[str, Any] = {
        "enabled": True,
        "version": "direct_dcr_repair_v4",
        "candidate_full_eval_used": False,
        "intermediate_candidate_count": 0,
        "selection_signal": "row_level_train_vs_test_dcr_direction",
    }
    if not pool_records or not selected_records:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {**report_base, "applied": False, "reason": "empty_inputs"},
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

    pool_df = records_to_df(pool_records, column_order)
    signal = _row_dcr_signal(
        pool_df=pool_df,
        train_df=train_df[column_order],
        test_df=test_df[column_order],
        schema_card=schema_card,
        column_order=column_order,
        cat_weight=cat_weight,
    )
    is_real_closer = np.asarray(signal["is_real_closer"], dtype=bool)
    base_dcr = float(np.mean(is_real_closer[selected_pool_indices]))
    target_margin = abs(float(target_margin))
    if base_dcr >= 0.5:
        target_dcr = 0.5 + target_margin
        reduce_dcr = True
    else:
        target_dcr = 0.5 - target_margin
        reduce_dcr = False

    keep_k = len(selected_records)
    desired_swaps = int(round(abs(base_dcr - target_dcr) * float(keep_k)))
    max_swaps = max(0, int(round(float(keep_k) * max(0.0, float(max_swap_fraction)))))
    if desired_swaps <= 0 or max_swaps <= 0:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "already_near_target",
                "base_dcr_estimate": base_dcr,
                "target_dcr": target_dcr,
            },
        )

    utility, utility_report = _utility_scores(
        pool_records=pool_records,
        exact_records=exact_records,
        surrogate_records=surrogate_records,
    )
    pairs = _build_pairs(
        selected_pool_indices=selected_pool_indices,
        selected_mask=selected_mask,
        is_real_closer=is_real_closer,
        margin=np.asarray(signal["margin"], dtype=float),
        features=np.asarray(signal["features"], dtype=np.float32),
        utility_scores=utility,
        reduce_dcr=reduce_dcr,
        candidate_neighbors=candidate_neighbors,
        margin_weight=margin_weight,
        utility_weight=utility_weight,
    )
    selected_swaps = min(int(desired_swaps), int(max_swaps), int(len(pairs)))
    if selected_swaps <= 0:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "no_feasible_pairs",
                "base_dcr_estimate": base_dcr,
                "target_dcr": target_dcr,
                "desired_swaps": desired_swaps,
                "pair_count": int(len(pairs)),
            },
        )

    final_records = [dict(record) for record in selected_records]
    final_pool_indices = selected_pool_indices.copy()
    for remove_base_pos, add_pool_idx, _, _, _ in pairs[:selected_swaps]:
        final_records[int(remove_base_pos)] = dict(pool_records[int(add_pool_idx)])
        final_pool_indices[int(remove_base_pos)] = int(add_pool_idx)

    final_dcr = float(np.mean(is_real_closer[final_pool_indices]))
    final_df = records_to_df(final_records, column_order)
    prefix = pairs[:selected_swaps]
    return (
        final_df,
        final_records,
        {
            **report_base,
            "applied": True,
            "base_dcr_estimate": base_dcr,
            "target_dcr": target_dcr,
            "final_dcr_estimate": final_dcr,
            "base_dcr_privacy_estimate": float(1.0 - abs(base_dcr - 0.5)),
            "final_dcr_privacy_estimate": float(1.0 - abs(final_dcr - 0.5)),
            "desired_swaps": int(desired_swaps),
            "max_swaps": int(max_swaps),
            "selected_swaps": int(selected_swaps),
            "pair_count": int(len(pairs)),
            "reduce_dcr": bool(reduce_dcr),
            "candidate_neighbors": int(candidate_neighbors),
            "margin_weight": float(margin_weight),
            "utility_weight": float(utility_weight),
            "cat_weight": float(cat_weight),
            "mean_pair_distance": float(np.mean([item[2] for item in prefix])),
            "mean_pair_utility_gain": float(np.mean([item[4] for item in prefix])),
            "sum_pair_utility_gain": float(np.sum([item[4] for item in prefix])),
            "utility_scores": utility_report,
            "removed_candidate_ids": [
                _candidate_id(selected_records[int(item[0])], int(item[0]))
                for item in prefix
            ],
            "added_candidate_ids": [
                _candidate_id(pool_records[int(item[1])], int(item[1]))
                for item in prefix
            ],
        },
    )
