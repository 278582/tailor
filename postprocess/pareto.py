from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


DEFAULT_D_CUR_SIZE = 200


def _progress(iterable: Any, *, total: int, desc: str, disable: bool) -> Any:
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, disable=disable)


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _digitize_value(value: float, edges: list[float]) -> int:
    if len(edges) <= 2:
        return 0
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return len(edges) - 2
    return int(np.digitize([value], edges[1:-1], right=False)[0])


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a >= b) and np.any(a > b)


def _non_dominated_sort_generic_iterative(points: np.ndarray) -> list[list[int]]:
    n = len(points)
    if n == 0:
        return []

    remaining = np.arange(n, dtype=int)
    fronts: list[list[int]] = []
    block_size = 1024

    while remaining.size > 0:
        current_points = points[remaining]
        dominated = np.zeros(current_points.shape[0], dtype=bool)

        for start in range(0, current_points.shape[0], block_size):
            end = min(start + block_size, current_points.shape[0])
            block = current_points[start:end]
            ge = np.all(current_points[None, :, :] >= block[:, None, :], axis=2)
            gt = np.any(current_points[None, :, :] > block[:, None, :], axis=2)
            dominated[start:end] = np.any(ge & gt, axis=1)

        front_mask = ~dominated
        if not np.any(front_mask):
            fronts.append(remaining.tolist())
            break

        fronts.append(remaining[front_mask].tolist())
        remaining = remaining[~front_mask]

    return fronts


def _non_dominated_sort_generic(points: np.ndarray) -> list[list[int]]:
    n = len(points)
    if n == 0:
        return []

    packed_cols = (n + 7) // 8
    packed_bytes = n * packed_cols
    max_packed_bytes = 512 * 1024 * 1024
    if packed_bytes > max_packed_bytes:
        return _non_dominated_sort_generic_iterative(points)

    block_size = 128 if packed_bytes > 256 * 1024 * 1024 else 256
    dominates_packed = np.zeros((n, packed_cols), dtype=np.uint8)
    dominated_count = np.zeros(n, dtype=np.int32)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = points[start:end]
        ge = np.all(block[:, None, :] >= points[None, :, :], axis=2)
        gt = np.any(block[:, None, :] > points[None, :, :], axis=2)
        dominates = ge & gt
        dominated_count += dominates.sum(axis=0, dtype=np.int32)
        dominates_packed[start:end] = np.packbits(dominates, axis=1, bitorder="little")

    fronts: list[list[int]] = []
    remaining_count = dominated_count.copy()
    assigned = np.zeros(n, dtype=bool)
    current = np.flatnonzero(remaining_count == 0)

    while current.size > 0:
        fronts.append(current.tolist())
        assigned[current] = True
        dominated_rows = np.unpackbits(dominates_packed[current], axis=1, bitorder="little")[:, :n]
        remaining_count -= dominated_rows.sum(axis=0, dtype=np.int32)
        current = np.flatnonzero((remaining_count == 0) & ~assigned)

    if not np.all(assigned):
        remaining = np.flatnonzero(~assigned)
        if remaining.size > 0:
            fronts.append(remaining.tolist())
    return fronts


def _non_dominated_sort_2d(points: np.ndarray) -> list[list[int]]:
    n = len(points)
    if n == 0:
        return []

    x = points[:, 0]
    y = points[:, 1]
    order = np.lexsort((-y, -x))

    unique_y_desc = np.unique(y)[::-1]
    y_to_fenwick = {float(value): idx + 1 for idx, value in enumerate(unique_y_desc.tolist())}
    tree = np.full(len(unique_y_desc) + 1, -1, dtype=int)

    def _query(prefix_idx: int) -> int:
        best = -1
        while prefix_idx > 0:
            best = max(best, int(tree[prefix_idx]))
            prefix_idx -= prefix_idx & -prefix_idx
        return best

    def _update(prefix_idx: int, value: int) -> None:
        while prefix_idx < len(tree):
            if value > tree[prefix_idx]:
                tree[prefix_idx] = value
            prefix_idx += prefix_idx & -prefix_idx

    front_rank = np.zeros(n, dtype=int)
    group_start = 0
    while group_start < n:
        group_end = group_start + 1
        x_value = x[order[group_start]]
        while group_end < n and x[order[group_end]] == x_value:
            group_end += 1

        group_indices = order[group_start:group_end]
        local_start = 0
        while local_start < len(group_indices):
            local_end = local_start + 1
            y_value = y[group_indices[local_start]]
            while local_end < len(group_indices) and y[group_indices[local_end]] == y_value:
                local_end += 1

            batch_indices = group_indices[local_start:local_end]
            fenwick_idx = y_to_fenwick[float(y_value)]
            rank = _query(fenwick_idx) + 1
            front_rank[batch_indices] = rank
            _update(fenwick_idx, rank)
            local_start = local_end

        group_start = group_end

    fronts: list[list[int]] = [[] for _ in range(int(front_rank.max()) + 1)]
    for idx, rank in enumerate(front_rank.tolist()):
        fronts[rank].append(idx)
    return [front for front in fronts if front]


def _non_dominated_sort_3d(points: np.ndarray) -> list[list[int]]:
    n = len(points)
    if n == 0:
        return []

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    order = np.lexsort((-z, -y, -x))

    unique_y_desc = np.unique(y)[::-1]
    unique_z_desc = np.unique(z)[::-1]
    y_to_fenwick = {float(value): idx + 1 for idx, value in enumerate(unique_y_desc.tolist())}
    z_to_fenwick = {float(value): idx + 1 for idx, value in enumerate(unique_z_desc.tolist())}
    y_size = len(unique_y_desc)
    z_size = len(unique_z_desc)
    tree: dict[tuple[int, int], int] = {}

    def _query(y_idx: int, z_idx: int) -> int:
        best = -1
        i = y_idx
        while i > 0:
            j = z_idx
            while j > 0:
                best = max(best, tree.get((i, j), -1))
                j -= j & -j
            i -= i & -i
        return best

    def _update(y_idx: int, z_idx: int, value: int) -> None:
        i = y_idx
        while i <= y_size:
            j = z_idx
            while j <= z_size:
                key = (i, j)
                if value > tree.get(key, -1):
                    tree[key] = value
                j += j & -j
            i += i & -i

    front_rank = np.zeros(n, dtype=int)
    group_start = 0
    while group_start < n:
        group_end = group_start + 1
        first_idx = order[group_start]
        x_value = x[first_idx]
        y_value = y[first_idx]
        z_value = z[first_idx]
        while group_end < n:
            next_idx = order[group_end]
            if x[next_idx] != x_value or y[next_idx] != y_value or z[next_idx] != z_value:
                break
            group_end += 1

        group_indices = order[group_start:group_end]
        fenwick_y = y_to_fenwick[float(y_value)]
        fenwick_z = z_to_fenwick[float(z_value)]
        rank = _query(fenwick_y, fenwick_z) + 1
        front_rank[group_indices] = rank
        _update(fenwick_y, fenwick_z, rank)
        group_start = group_end

    fronts: list[list[int]] = [[] for _ in range(int(front_rank.max()) + 1)]
    for idx, rank in enumerate(front_rank.tolist()):
        fronts[rank].append(idx)
    return [front for front in fronts if front]


def _drop_constant_objectives(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or len(points) == 0:
        return points
    variable_mask = np.any(points != points[0:1, :], axis=0)
    return points[:, variable_mask]


def _non_dominated_sort(points: np.ndarray) -> list[list[int]]:
    sort_points = _drop_constant_objectives(points)
    if sort_points.ndim == 2 and sort_points.shape[1] == 0:
        return [list(range(len(sort_points)))] if len(sort_points) else []
    if sort_points.ndim == 2 and len(sort_points) > 1:
        unique_points, inverse = np.unique(sort_points, axis=0, return_inverse=True)
        if len(unique_points) < len(sort_points):
            unique_fronts = (
                _non_dominated_sort_2d(unique_points)
                if unique_points.shape[1] == 2
                else _non_dominated_sort_3d(unique_points)
                if unique_points.shape[1] == 3
                else _non_dominated_sort_generic(unique_points)
            )
            inverse_order = np.argsort(inverse, kind="mergesort")
            inverse_sorted = inverse[inverse_order]
            split_points = np.r_[
                0,
                np.flatnonzero(np.diff(inverse_sorted)) + 1,
                inverse_order.size,
            ]
            unique_to_original = [
                inverse_order[split_points[idx] : split_points[idx + 1]]
                for idx in range(len(split_points) - 1)
            ]
            expanded_fronts: list[list[int]] = []
            for front in unique_fronts:
                if len(front) == 1:
                    expanded_fronts.append(unique_to_original[int(front[0])].tolist())
                else:
                    expanded_fronts.append(
                        np.sort(np.concatenate([unique_to_original[int(unique_idx)] for unique_idx in front])).tolist()
                    )
            return expanded_fronts
    if sort_points.ndim == 2 and sort_points.shape[1] == 2:
        return _non_dominated_sort_2d(sort_points)
    if sort_points.ndim == 2 and sort_points.shape[1] == 3:
        return _non_dominated_sort_3d(sort_points)
    return _non_dominated_sort_generic(sort_points)


def _crowding_distance(points: np.ndarray, indices: list[int]) -> dict[int, float]:
    index_array = np.asarray(indices, dtype=int)
    distance_array = _crowding_distance_array(points, index_array)
    return {int(idx): float(distance_array[pos]) for pos, idx in enumerate(index_array.tolist())}


def _crowding_distance_array(points: np.ndarray, indices: np.ndarray | list[int]) -> np.ndarray:
    index_array = np.asarray(indices, dtype=int)
    if index_array.size <= 2:
        return np.full(index_array.size, float("inf"), dtype=float)

    front_points = points[index_array]
    num_obj = front_points.shape[1]
    distances = np.zeros(index_array.size, dtype=float)
    infinite_mask = np.zeros(index_array.size, dtype=bool)

    for m in range(num_obj):
        order = np.lexsort((index_array, front_points[:, m]))
        values = front_points[order, m]
        infinite_mask[order[0]] = True
        infinite_mask[order[-1]] = True
        denom = values[-1] - values[0]
        if denom <= 1e-12:
            continue
        contributions = np.zeros(index_array.size, dtype=float)
        contributions[order[1:-1]] = (values[2:] - values[:-2]) / denom
        distances += contributions

    distances[infinite_mask] = float("inf")
    return distances


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lower = float(values.min())
    upper = float(values.max())
    if upper - lower <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lower) / (upper - lower)


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    if values.size == 1:
        return np.ones_like(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, values.size)
    return ranks


def _same_scale_objective_components(
    benefit: np.ndarray,
    penalty: np.ndarray,
    *,
    penalty_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    benefit_norm = _minmax_normalize(np.asarray(benefit, dtype=float))
    penalty_norm = _minmax_normalize(np.asarray(penalty, dtype=float))
    objective = benefit_norm - float(penalty_weight) * penalty_norm
    return benefit_norm, penalty_norm, objective


def _safe_geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    clipped = np.clip(np.asarray(values, dtype=float), 1e-12, 1.0)
    return float(np.exp(np.mean(np.log(clipped))))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(float(p.sum()), 1.0)
    q = q / max(float(q.sum()), 1.0)
    m = 0.5 * (p + q)
    p_term = np.where(p > 0, p * np.log(np.clip(p / np.clip(m, 1e-12, None), 1e-12, None)), 0.0)
    q_term = np.where(q > 0, q * np.log(np.clip(q / np.clip(m, 1e-12, None), 1e-12, None)), 0.0)
    return float(0.5 * np.sum(p_term) + 0.5 * np.sum(q_term))


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges.astype(float))
    if edges.size < 2:
        value = float(values[0])
        return np.array([value - 0.5, value + 0.5], dtype=float)
    return edges


class ParetoSelector:
    def __init__(
        self,
        train_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        schema_card: dict[str, Any],
        stats_card: dict[str, Any],
        seed: int,
        source: str = "llm",
        lambda_penalty: float = 1.0,
        gamma: float = 0.5,
        privacy_version: str = "v2",
        density_k: int = 10,
        density_reference_size: int = 5000,
        nn_device: str = "auto",
        nn_query_batch_size: int = 2048,
        nn_reference_chunk_size: int = 8192,
        rarity_strata: int = 5,
        max_pair_marginal_edges: int = 32,
        final_fidelity_floor_eps: float = 0.01,
        final_trend_floor_eps: float = 0.01,
    ) -> None:
        self.train_df = train_df.reset_index(drop=True)
        self.holdout_df = holdout_df.reset_index(drop=True)
        self.schema_card = schema_card
        self.stats_card = stats_card
        self.seed = seed
        self.source = source
        self.lambda_penalty = float(lambda_penalty)
        self.gamma = float(gamma)
        self.privacy_version = privacy_version
        self.density_k = max(1, int(density_k))
        self.density_reference_size = max(0, int(density_reference_size))
        self.nn_device_arg = nn_device
        self.nn_query_batch_size = max(1, int(nn_query_batch_size))
        self.nn_reference_chunk_size = max(1, int(nn_reference_chunk_size))
        self.rarity_strata = max(3, int(rarity_strata))
        self.max_pair_marginal_edges = max(0, int(max_pair_marginal_edges))
        self.final_fidelity_floor_eps = max(0.0, float(final_fidelity_floor_eps))
        self.final_trend_floor_eps = max(0.0, float(final_trend_floor_eps))
        self.nn_backend, self.nn_device = self._resolve_nn_backend(nn_device)

        self.column_order = schema_card["column_order"]
        self.target_column = schema_card["target_column"]
        self.fidelity_columns = list(self.column_order)
        self.feature_columns = [c for c in self.column_order if not schema_card["columns"][c]["is_target"]]
        self.numeric_columns = [
            c
            for c in self.column_order
            if schema_card["columns"][c]["type"] in {"numerical", "discrete_numerical"}
        ]
        self.categorical_columns = [c for c in self.column_order if schema_card["columns"][c]["type"] == "categorical"]

        self.train_distributions = self._build_train_distributions()
        self.pair_marginal_edges = self._build_pair_marginals()
        self.num_fidelity_columns = max(len(self.fidelity_columns), 1)
        self.pair_weights = np.asarray(
            [max(float(edge.get("mi", 0.0)), 1e-6) for edge in self.pair_marginal_edges],
            dtype=float,
        )
        self.total_pair_weight = max(float(self.pair_weights.sum()), 1.0)
        self._fit_privacy_encoder()
        self._fit_density_reference()
        self._fit_gate_rarity_reference()
        self.last_preselect_report: dict[str, Any] = {}
        self._record_bucket_pair_cache: dict[tuple[int, int], tuple[dict[str, np.ndarray], list[np.ndarray]]] = {}
        self._record_target_counts_cache: dict[tuple[int, int], tuple[dict[str, np.ndarray], list[np.ndarray]]] = {}

    def _column_bucket_indices_from_series(self, series: pd.Series, column: str) -> np.ndarray:
        info = self.schema_card["columns"][column]
        train_dist = self.train_distributions[column]
        if info["type"] == "numerical":
            values = series.astype(float).to_numpy()
            edges = np.asarray(train_dist["edges"], dtype=float)
            if edges.size <= 2:
                return np.zeros(len(values), dtype=int)
            clipped = np.clip(values, float(edges[0]), float(edges[-1]))
            return np.digitize(clipped, edges[1:-1], right=False).astype(int)
        if info["type"] == "discrete_numerical":
            values = series.astype(float).to_numpy()
            legal_values = np.asarray(train_dist["values"], dtype=float)
            distances = np.abs(values[:, None] - legal_values[None, :])
            return np.argmin(distances, axis=1).astype(int)

        categorical = pd.Categorical(series.astype(str), categories=train_dist["values"])
        return categorical.codes.astype(int, copy=False)

    def _column_bucket_indices_for_df(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        use_columns = self.column_order if columns is None else columns
        return {column: self._column_bucket_indices_from_series(df[column], column) for column in use_columns}

    def _column_probabilities_from_indices(self, column: str, bucket_indices: np.ndarray) -> np.ndarray:
        probs = np.asarray(self.train_distributions[column]["probs"], dtype=float)
        if bucket_indices.size == 0:
            return np.zeros(0, dtype=float)
        if np.any(bucket_indices < 0):
            padded = np.concatenate([probs, np.asarray([1e-12], dtype=float)])
            safe_indices = bucket_indices.copy()
            safe_indices[safe_indices < 0] = len(probs)
            return padded[safe_indices]
        return probs[bucket_indices]

    def _prob_geomean_from_bucket_indices(
        self,
        bucket_indices: dict[str, np.ndarray],
        columns: list[str],
    ) -> np.ndarray:
        if not columns:
            return np.zeros(0, dtype=float)
        probs = np.stack(
            [np.clip(self._column_probabilities_from_indices(column, bucket_indices[column]), 1e-12, 1.0) for column in columns],
            axis=1,
        )
        return np.exp(np.mean(np.log(probs), axis=1))

    def _prob_geomean_for_df(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        bucket_indices: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        use_columns = self.column_order if columns is None else columns
        if df.empty:
            return np.zeros(0, dtype=float)
        if bucket_indices is None:
            bucket_indices = self._column_bucket_indices_for_df(df, use_columns)
        return self._prob_geomean_from_bucket_indices(bucket_indices, use_columns)

    def _pair_prob_geomean_from_bucket_indices(self, bucket_indices: dict[str, np.ndarray]) -> np.ndarray:
        if not self.pair_marginal_edges:
            first_column = self.fidelity_columns[0] if self.fidelity_columns else None
            size = len(bucket_indices[first_column]) if first_column is not None else 0
            return np.ones(size, dtype=float)

        first_edge = self.pair_marginal_edges[0]
        size = len(bucket_indices[first_edge["left"]])
        log_sum = np.zeros(size, dtype=float)
        edge_count = 0
        for edge in self.pair_marginal_edges:
            left_idx = bucket_indices[edge["left"]]
            right_idx = bucket_indices[edge["right"]]
            probs = edge["probs"]
            right_bins = int(edge["right_bins"])
            pair_probs = np.full(size, 1e-12, dtype=float)
            valid = (left_idx >= 0) & (right_idx >= 0)
            if valid.any():
                flat_indices = left_idx[valid] * right_bins + right_idx[valid]
                pair_probs[valid] = probs[flat_indices]
            log_sum += np.log(np.clip(pair_probs, 1e-12, 1.0))
            edge_count += 1

        if edge_count == 0:
            return np.ones(size, dtype=float)
        return np.exp(log_sum / float(edge_count))

    def _pool_balance_1d_from_bucket_indices(
        self,
        bucket_indices: dict[str, np.ndarray],
        columns: list[str],
    ) -> np.ndarray:
        if not columns:
            return np.zeros(0, dtype=float)

        row_scores: list[np.ndarray] = []
        for column in columns:
            indices = bucket_indices[column]
            train_probs = np.asarray(self.train_distributions[column]["probs"], dtype=float)
            pool_counts = self._column_counts_from_bucket_indices(column, indices)
            pool_probs = pool_counts / max(float(pool_counts.sum()), 1.0)
            keep_weights = np.minimum(
                train_probs / np.clip(pool_probs, 1e-12, None),
                1.0,
            )
            keep_weights = np.clip(keep_weights, 0.0, 1.0)
            padded = np.concatenate([keep_weights, np.asarray([0.0], dtype=float)])
            safe_indices = indices.copy()
            safe_indices[safe_indices < 0] = len(keep_weights)
            row_scores.append(padded[safe_indices])

        return np.mean(np.stack(row_scores, axis=1), axis=1)

    def _pool_balance_2d_from_bucket_indices(self, bucket_indices: dict[str, np.ndarray]) -> np.ndarray:
        if not self.pair_marginal_edges:
            first_column = self.fidelity_columns[0] if self.fidelity_columns else None
            size = len(bucket_indices[first_column]) if first_column is not None else 0
            return np.ones(size, dtype=float)

        first_edge = self.pair_marginal_edges[0]
        size = len(bucket_indices[first_edge["left"]])
        score_sum = np.zeros(size, dtype=float)
        edge_count = 0

        for edge in self.pair_marginal_edges:
            left_idx = bucket_indices[edge["left"]]
            right_idx = bucket_indices[edge["right"]]
            right_bins = int(edge["right_bins"])
            valid = (left_idx >= 0) & (right_idx >= 0)
            edge_scores = np.zeros(size, dtype=float)
            if valid.any():
                flat_indices = left_idx[valid] * right_bins + right_idx[valid]
                pool_counts = np.bincount(
                    flat_indices,
                    minlength=int(edge["left_bins"]) * int(edge["right_bins"]),
                ).astype(float)
                pool_probs = pool_counts / max(float(pool_counts.sum()), 1.0)
                train_probs = np.asarray(edge["probs"], dtype=float)
                keep_weights = np.minimum(
                    train_probs / np.clip(pool_probs, 1e-12, None),
                    1.0,
                )
                keep_weights = np.clip(keep_weights, 0.0, 1.0)
                edge_scores[valid] = keep_weights[flat_indices]
            score_sum += edge_scores
            edge_count += 1

        if edge_count == 0:
            return np.ones(size, dtype=float)
        return score_sum / float(edge_count)

    def _pair_codes_from_bucket_indices(self, bucket_indices: dict[str, np.ndarray]) -> list[np.ndarray]:
        pair_codes: list[np.ndarray] = []
        if not self.pair_marginal_edges:
            return pair_codes
        for edge in self.pair_marginal_edges:
            left_idx = bucket_indices[edge["left"]]
            right_idx = bucket_indices[edge["right"]]
            flat_codes = np.full(len(left_idx), -1, dtype=int)
            valid = (left_idx >= 0) & (right_idx >= 0)
            if valid.any():
                flat_codes[valid] = left_idx[valid] * int(edge["right_bins"]) + right_idx[valid]
            pair_codes.append(flat_codes)
        return pair_codes

    def _record_cache_key(self, records: list[dict[str, Any]]) -> tuple[int, int]:
        return (id(records), len(records))

    def _bucket_pair_state_for_records(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
        key = self._record_cache_key(records)
        cached = self._record_bucket_pair_cache.get(key)
        if cached is not None:
            return cached
        candidate_df = pd.DataFrame([record["row"] for record in records], columns=self.column_order)
        bucket_indices = self._column_bucket_indices_for_df(candidate_df, self.fidelity_columns)
        pair_codes = self._pair_codes_from_bucket_indices(bucket_indices)
        if len(self._record_bucket_pair_cache) >= 6:
            self._record_bucket_pair_cache.pop(next(iter(self._record_bucket_pair_cache)))
        self._record_bucket_pair_cache[key] = (bucket_indices, pair_codes)
        return bucket_indices, pair_codes

    def _lookup_code_scores(self, codes: np.ndarray, score_vector: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        vector = np.asarray(score_vector, dtype=float)
        code_array = np.asarray(codes, dtype=int)
        if fill_value == 0.0:
            scores = np.zeros(code_array.shape, dtype=float)
        else:
            scores = np.full(code_array.shape, float(fill_value), dtype=float)
        valid = (code_array >= 0) & (code_array < vector.size)
        if np.any(valid):
            scores[valid] = vector[code_array[valid]]
        return scores

    @staticmethod
    def _add_code_count_delta(counts: np.ndarray, codes: np.ndarray, delta: int) -> None:
        valid_codes = np.asarray(codes, dtype=int)
        valid_codes = valid_codes[valid_codes >= 0]
        if valid_codes.size > 0:
            np.add.at(counts, valid_codes, int(delta))

    def _target_count_support_scores_1d(
        self,
        bucket_indices: dict[str, np.ndarray],
        selected_counts_1d: dict[str, np.ndarray],
        target_counts_1d: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not bucket_indices or not self.fidelity_columns:
            empty = np.zeros(0, dtype=float)
            return empty, empty
        first_column = self.fidelity_columns[0]
        num_rows = len(bucket_indices[first_column])
        remove_support = np.zeros(num_rows, dtype=float)
        add_support = np.zeros(num_rows, dtype=float)
        for column in self.fidelity_columns:
            target = np.asarray(target_counts_1d[column], dtype=float)
            current = np.asarray(selected_counts_1d[column], dtype=float)
            denom = np.clip(target, 1.0, None)
            oversupply = np.maximum(current - target, 0.0) / denom
            deficit = np.maximum(target - current, 0.0) / denom
            codes = bucket_indices[column]
            remove_support += self._lookup_code_scores(codes, oversupply)
            add_support += self._lookup_code_scores(codes, deficit)
        scale = float(self.num_fidelity_columns)
        return remove_support / scale, add_support / scale

    def _target_count_support_scores_2d(
        self,
        pair_codes: list[np.ndarray],
        selected_counts_2d: list[np.ndarray],
        target_counts_2d: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not pair_codes:
            empty = np.zeros(0, dtype=float)
            return empty, empty
        num_rows = len(pair_codes[0])
        remove_support = np.zeros(num_rows, dtype=float)
        add_support = np.zeros(num_rows, dtype=float)
        for pair_idx, codes in enumerate(pair_codes):
            if pair_idx >= len(target_counts_2d):
                continue
            target = np.asarray(target_counts_2d[pair_idx], dtype=float)
            current = np.asarray(selected_counts_2d[pair_idx], dtype=float)
            denom = np.clip(target, 1.0, None)
            oversupply = np.maximum(current - target, 0.0) / denom
            deficit = np.maximum(target - current, 0.0) / denom
            weight = float(self.pair_weights[pair_idx]) if pair_idx < len(self.pair_weights) else 1.0
            remove_support += weight * self._lookup_code_scores(codes, oversupply)
            add_support += weight * self._lookup_code_scores(codes, deficit)
        return remove_support / self.total_pair_weight, add_support / self.total_pair_weight

    def _train_prob_support_scores_1d(
        self,
        bucket_indices: dict[str, np.ndarray],
        selected_counts_1d: dict[str, np.ndarray],
        subset_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not bucket_indices or not self.fidelity_columns:
            empty = np.zeros(0, dtype=float)
            return empty, empty
        first_column = self.fidelity_columns[0]
        num_rows = len(bucket_indices[first_column])
        remove_support = np.zeros(num_rows, dtype=float)
        add_support = np.zeros(num_rows, dtype=float)
        denom = max(float(subset_size), 1.0)
        for column in self.fidelity_columns:
            train_probs = np.asarray(self.train_distributions[column]["probs"], dtype=float)
            selected_probs = np.asarray(selected_counts_1d[column], dtype=float) / denom
            oversupply = np.maximum(selected_probs - train_probs, 0.0)
            deficit = np.maximum(train_probs - selected_probs, 0.0)
            codes = bucket_indices[column]
            remove_support += self._lookup_code_scores(codes, oversupply)
            add_support += self._lookup_code_scores(codes, deficit)
        scale = float(self.num_fidelity_columns)
        return remove_support / scale, add_support / scale

    def _train_prob_support_scores_2d(
        self,
        pair_codes: list[np.ndarray],
        selected_counts_2d: list[np.ndarray],
        subset_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not pair_codes:
            empty = np.zeros(0, dtype=float)
            return empty, empty
        num_rows = len(pair_codes[0])
        remove_support = np.zeros(num_rows, dtype=float)
        add_support = np.zeros(num_rows, dtype=float)
        denom = max(float(subset_size), 1.0)
        for pair_idx, codes in enumerate(pair_codes):
            if pair_idx >= len(self.pair_marginal_edges):
                continue
            train_probs = np.asarray(self.pair_marginal_edges[pair_idx]["probs"], dtype=float)
            selected_probs = np.asarray(selected_counts_2d[pair_idx], dtype=float) / denom
            oversupply = np.maximum(selected_probs - train_probs, 0.0)
            deficit = np.maximum(train_probs - selected_probs, 0.0)
            weight = float(self.pair_weights[pair_idx]) if pair_idx < len(self.pair_weights) else 1.0
            remove_support += weight * self._lookup_code_scores(codes, oversupply)
            add_support += weight * self._lookup_code_scores(codes, deficit)
        return remove_support / self.total_pair_weight, add_support / self.total_pair_weight

    def _weighted_mean(self, matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.zeros(matrix.shape[0] if matrix.ndim > 0 else 0, dtype=float)
        weights = np.asarray(weights, dtype=float)
        denom = max(float(weights.sum()), 1e-12)
        return (matrix * weights[None, :]).sum(axis=1) / denom

    def _exact_fidelity_after_and_penalty(
        self,
        candidate_df: pd.DataFrame,
        baseline_counts: dict[str, np.ndarray],
        subset_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if candidate_df.empty:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        bucket_indices = self._column_bucket_indices_for_df(candidate_df, self.fidelity_columns)
        similarities: list[np.ndarray] = []
        penalties: list[np.ndarray] = []

        for column in self.fidelity_columns:
            train_dist = self.train_distributions[column]
            counts = np.asarray(baseline_counts[column], dtype=float)
            indices = bucket_indices[column]
            if np.any(indices < 0):
                raise ValueError(f"Unknown bucket encountered in exact fidelity scoring for column={column}.")

            train_probs = np.asarray(train_dist["probs"], dtype=float)
            denom = max(float(counts.sum()) + 1.0, 1.0)
            abs_noadd = np.abs(counts / denom - train_probs)
            abs_add = np.abs((counts + 1.0) / denom - train_probs)
            delta = abs_add - abs_noadd
            tvd = 0.5 * (float(abs_noadd.sum()) + delta[indices])
            similarities.append(1.0 - tvd)

            expected = train_probs[indices]
            new_freq = (counts[indices] + 1.0) / max(float(subset_size + 1), 1.0)
            penalties.append(np.maximum(0.0, new_freq - expected))

        similarity_matrix = np.stack(similarities, axis=1)
        penalty_matrix = np.stack(penalties, axis=1)
        return similarity_matrix.mean(axis=1), penalty_matrix.mean(axis=1)

    def _compute_pair_fidelity_baseline(self, d_cur_df: pd.DataFrame) -> tuple[float, list[np.ndarray]]:
        if not self.pair_marginal_edges:
            return 1.0, []
        bucket_indices = self._column_bucket_indices_for_df(d_cur_df, self.fidelity_columns)
        pair_codes = self._pair_codes_from_bucket_indices(bucket_indices)
        scores: list[float] = []
        count_list: list[np.ndarray] = []
        weights: list[float] = []
        for edge, flat_codes in zip(self.pair_marginal_edges, pair_codes):
            probs = np.asarray(edge["probs"], dtype=float)
            counts = np.bincount(flat_codes[flat_codes >= 0], minlength=len(probs)).astype(float)
            count_list.append(counts)
            scores.append(self._column_similarity(counts, probs))
            weights.append(max(float(edge.get("mi", 0.0)), 1e-6))
        if not scores:
            return 1.0, count_list
        score_array = np.asarray(scores, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        return float(np.dot(score_array, weight_array) / max(float(weight_array.sum()), 1e-12)), count_list

    def _exact_pair_fidelity_after_and_penalty(
        self,
        candidate_df: pd.DataFrame,
        baseline_pair_counts: list[np.ndarray],
        subset_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if candidate_df.empty:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        if not self.pair_marginal_edges:
            rows = len(candidate_df)
            return np.ones(rows, dtype=float), np.zeros(rows, dtype=float)

        bucket_indices = self._column_bucket_indices_for_df(candidate_df, self.fidelity_columns)
        pair_codes = self._pair_codes_from_bucket_indices(bucket_indices)
        similarities: list[np.ndarray] = []
        penalties: list[np.ndarray] = []
        weights: list[float] = []

        for edge, counts, codes in zip(self.pair_marginal_edges, baseline_pair_counts, pair_codes):
            counts = np.asarray(counts, dtype=float)
            probs = np.asarray(edge["probs"], dtype=float)
            if np.any(codes < 0):
                raise ValueError(
                    f"Unknown pair bucket encountered in exact pair fidelity scoring for edge={edge['left']}->{edge['right']}."
                )
            denom = max(float(counts.sum()) + 1.0, 1.0)
            abs_noadd = np.abs(counts / denom - probs)
            abs_add = np.abs((counts + 1.0) / denom - probs)
            delta = abs_add - abs_noadd
            tvd = 0.5 * (float(abs_noadd.sum()) + delta[codes])
            similarities.append(1.0 - tvd)

            expected = probs[codes]
            new_freq = (counts[codes] + 1.0) / max(float(subset_size + 1), 1.0)
            penalties.append(np.maximum(0.0, new_freq - expected))
            weights.append(max(float(edge.get("mi", 0.0)), 1e-6))

        similarity_matrix = np.stack(similarities, axis=1)
        penalty_matrix = np.stack(penalties, axis=1)
        weight_array = np.asarray(weights, dtype=float)
        return self._weighted_mean(similarity_matrix, weight_array), self._weighted_mean(penalty_matrix, weight_array)

    def _resolve_nn_backend(self, nn_device: str) -> tuple[str, torch.device | None]:
        if nn_device == "auto":
            if torch.cuda.is_available():
                return "torch", torch.device("cuda:0")
            return "sklearn", None
        if nn_device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"nn_device={nn_device} requested but torch.cuda.is_available() is False.")
            return "torch", torch.device(nn_device)
        if nn_device == "cpu":
            return "sklearn", None
        raise ValueError(f"Unsupported nn_device={nn_device}. Use auto, cpu, or cuda:<idx>.")

    def _to_device_tensor(self, matrix: np.ndarray) -> torch.Tensor:
        assert self.nn_device is not None
        array = np.asarray(matrix, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D matrix for tensor conversion, got shape={array.shape}")
        return torch.as_tensor(array, dtype=torch.float32, device=self.nn_device)

    def _torch_knn_topk(
        self,
        query_matrix: np.ndarray,
        reference_tensor: torch.Tensor,
        k: int,
        *,
        exclude_self: bool = False,
    ) -> np.ndarray:
        if self.nn_device is None:
            raise RuntimeError("torch kNN backend requested without a resolved torch device")
        query_count = int(query_matrix.shape[0])
        reference_count = int(reference_tensor.shape[0])
        k = max(1, min(int(k), reference_count))
        if query_count == 0:
            return np.zeros((0, k), dtype=np.float32)

        outputs: list[torch.Tensor] = []
        same_reference = exclude_self and query_count == reference_count
        for q_start in range(0, query_count, self.nn_query_batch_size):
            q_end = min(q_start + self.nn_query_batch_size, query_count)
            query_batch = self._to_device_tensor(query_matrix[q_start:q_end])
            best_values: torch.Tensor | None = None

            for r_start in range(0, reference_count, self.nn_reference_chunk_size):
                r_end = min(r_start + self.nn_reference_chunk_size, reference_count)
                reference_chunk = reference_tensor[r_start:r_end]
                distances = torch.cdist(query_batch, reference_chunk, p=2)

                if same_reference:
                    overlap_start = max(q_start, r_start)
                    overlap_end = min(q_end, r_end)
                    if overlap_start < overlap_end:
                        diag_rows = torch.arange(
                            overlap_start - q_start,
                            overlap_end - q_start,
                            device=self.nn_device,
                            dtype=torch.long,
                        )
                        diag_cols = torch.arange(
                            overlap_start - r_start,
                            overlap_end - r_start,
                            device=self.nn_device,
                            dtype=torch.long,
                        )
                        distances[diag_rows, diag_cols] = float("inf")

                chunk_k = min(k, int(distances.shape[1]))
                chunk_values = torch.topk(distances, k=chunk_k, largest=False, dim=1).values
                if best_values is None:
                    best_values = chunk_values
                else:
                    combined = torch.cat([best_values, chunk_values], dim=1)
                    best_values = torch.topk(combined, k=k, largest=False, dim=1).values

            assert best_values is not None
            outputs.append(best_values)

        return torch.cat(outputs, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)

    def _torch_knn_min(self, query_matrix: np.ndarray, reference_tensor: torch.Tensor) -> np.ndarray:
        return self._torch_knn_topk(query_matrix, reference_tensor, k=1)[:, 0]

    def initialize_d_cur(self, size: int = DEFAULT_D_CUR_SIZE) -> pd.DataFrame:
        sample_size = min(size, len(self.train_df))
        return self.train_df.sample(n=sample_size, random_state=self.seed, replace=False).reset_index(drop=True)

    def _fit_privacy_encoder(self) -> None:
        if self.numeric_columns:
            self.scaler: StandardScaler | None = StandardScaler()
            self.scaler.fit(self.train_df[self.numeric_columns].astype(float))
        else:
            self.scaler = None

        if self.categorical_columns:
            self.ohe: OneHotEncoder | None = _make_ohe()
            cat_fit = pd.concat(
                [
                    self.train_df[self.categorical_columns].astype(str),
                    self.holdout_df[self.categorical_columns].astype(str),
                ],
                axis=0,
                ignore_index=True,
            )
            self.ohe.fit(cat_fit)
        else:
            self.ohe = None

        self.train_matrix = self._encode_df(self.train_df)
        self.holdout_matrix = self._encode_df(self.holdout_df)

        if 0 < self.density_reference_size < len(self.train_df):
            rng = np.random.default_rng(self.seed)
            self.density_reference_indices = np.sort(
                rng.choice(len(self.train_df), size=self.density_reference_size, replace=False)
            )
        else:
            self.density_reference_indices = np.arange(len(self.train_df))
        self.density_reference_matrix = self.train_matrix[self.density_reference_indices]

        if self.nn_backend == "torch":
            self.train_tensor = self._to_device_tensor(self.train_matrix)
            self.holdout_tensor = self._to_device_tensor(self.holdout_matrix)
            self.density_reference_tensor = self._to_device_tensor(self.density_reference_matrix)
            self.nn_train = None
            self.nn_holdout = None
            self.nn_density = None
        else:
            self.train_tensor = None
            self.holdout_tensor = None
            self.density_reference_tensor = None
            self.nn_train = NearestNeighbors(n_neighbors=1, metric="euclidean")
            self.nn_train.fit(self.train_matrix)
            self.nn_holdout = NearestNeighbors(n_neighbors=1, metric="euclidean")
            self.nn_holdout.fit(self.holdout_matrix)
            self.nn_density = NearestNeighbors(
                n_neighbors=min(self.density_k, len(self.density_reference_matrix)),
                metric="euclidean",
            )
            self.nn_density.fit(self.density_reference_matrix)

    def _fit_density_reference(self) -> None:
        if len(self.density_reference_matrix) <= 1:
            self.train_density_edges = np.array([0.0, 1.0], dtype=float)
            self.train_density_expected_nn = np.array([1.0], dtype=float)
            self.train_density_values = np.ones(len(self.density_reference_matrix), dtype=float)
            return

        k_for_self = min(len(self.density_reference_matrix), self.density_k + 1)
        if self.nn_backend == "torch":
            assert self.density_reference_tensor is not None
            distances = self._torch_knn_topk(
                self.density_reference_matrix,
                self.density_reference_tensor,
                k=k_for_self,
                exclude_self=True,
            )
        else:
            nn_self = NearestNeighbors(n_neighbors=k_for_self, metric="euclidean")
            nn_self.fit(self.density_reference_matrix)
            distances, _ = nn_self.kneighbors(self.density_reference_matrix, n_neighbors=k_for_self)
        local_mean_dist = distances[:, 1:].mean(axis=1)
        local_ref_dist = distances[:, 1]
        density_values = 1.0 / np.clip(local_mean_dist, 1e-12, None)

        edges = _quantile_edges(density_values, n_bins=self.rarity_strata)
        density_bins = self._assign_bins_from_edges(density_values, edges)
        bucket_count = np.bincount(density_bins, minlength=len(edges) - 1).astype(float)
        bucket_sum = np.bincount(density_bins, weights=local_ref_dist, minlength=len(edges) - 1).astype(float)
        global_mean = float(np.mean(local_ref_dist))
        expected_nn = np.divide(
            bucket_sum,
            np.clip(bucket_count, 1.0, None),
            out=np.full(len(edges) - 1, global_mean, dtype=float),
            where=bucket_count > 0,
        )
        self.train_density_edges = edges
        self.train_density_expected_nn = np.asarray(expected_nn, dtype=float)
        self.train_density_values = density_values

    def _fit_gate_rarity_reference(self) -> None:
        gate_probs = self._prob_geomean_for_df(self.train_df, columns=self.feature_columns)
        self.train_gate_probs = gate_probs
        self.train_gate_edges = _quantile_edges(gate_probs, n_bins=self.rarity_strata)
        gate_bins = self._assign_bins_from_edges(gate_probs, self.train_gate_edges)
        counts = np.bincount(gate_bins, minlength=len(self.train_gate_edges) - 1).astype(float)
        probs = counts / max(counts.sum(), 1.0)
        self.train_gate_strata_probs = probs

    def _encode_df(self, df: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.numeric_columns:
            assert self.scaler is not None
            parts.append(self.scaler.transform(df[self.numeric_columns].astype(float)))
        if self.categorical_columns:
            assert self.ohe is not None
            parts.append(self.ohe.transform(df[self.categorical_columns].astype(str)))
        if not parts:
            return np.zeros((len(df), 0), dtype=float)
        return np.concatenate(parts, axis=1)

    def _encode_row(self, row: dict[str, Any]) -> np.ndarray:
        df = pd.DataFrame([row], columns=self.column_order)
        return self._encode_df(df)

    def _match_discrete_value(self, value: Any, legal_values: list[Any]) -> int:
        legal_array = np.asarray([float(v) for v in legal_values], dtype=float)
        return int(np.argmin(np.abs(legal_array - float(value))))

    def _build_train_distributions(self) -> dict[str, Any]:
        distributions: dict[str, Any] = {}
        for column in self.column_order:
            info = self.schema_card["columns"][column]
            if info["type"] == "numerical":
                edges = self.stats_card["numeric_bins"][column]
                values = self.train_df[column].astype(float).to_numpy()
                edge_array = np.asarray(edges, dtype=float)
                if edge_array.size <= 2:
                    bucket_indices = np.zeros(len(values), dtype=int)
                else:
                    clipped = np.clip(values, float(edge_array[0]), float(edge_array[-1]))
                    bucket_indices = np.digitize(clipped, edge_array[1:-1], right=False).astype(int)
                counts = np.bincount(bucket_indices, minlength=len(edges) - 1).astype(float)
                probs = counts / max(counts.sum(), 1.0)
                distributions[column] = {"edges": edges, "counts": counts, "probs": probs}
            elif info["type"] == "discrete_numerical":
                legal_values = info["legal_values"]
                values = self.train_df[column].astype(float).to_numpy()
                legal_array = np.asarray(legal_values, dtype=float)
                distances = np.abs(values[:, None] - legal_array[None, :])
                matched_indices = np.argmin(distances, axis=1).astype(int)
                counts = np.bincount(matched_indices, minlength=len(legal_values)).astype(float)
                probs = counts / max(counts.sum(), 1.0)
                distributions[column] = {"values": legal_values, "counts": counts, "probs": probs}
            else:
                legal_values = info["legal_values"]
                categorical = pd.Categorical(self.train_df[column].astype(str), categories=legal_values)
                valid_codes = categorical.codes[categorical.codes >= 0]
                counts = np.bincount(valid_codes, minlength=len(legal_values)).astype(float)
                probs = counts / max(counts.sum(), 1.0)
                distributions[column] = {"values": legal_values, "counts": counts, "probs": probs}
        return distributions

    def _normalized_mutual_information_from_joint(self, joint: np.ndarray) -> float:
        if joint.size == 0:
            return 0.0
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)
        denom = np.clip(px * py, 1e-12, None)
        mask = joint > 0
        mi = float(np.sum(joint[mask] * np.log(np.clip(joint[mask] / denom[mask], 1e-12, None))))
        hx = float(-np.sum(px[px > 0] * np.log(np.clip(px[px > 0], 1e-12, None))))
        hy = float(-np.sum(py[py > 0] * np.log(np.clip(py[py > 0], 1e-12, None))))
        return mi / max(np.sqrt(max(hx, 1e-12) * max(hy, 1e-12)), 1e-12)

    def _build_pair_marginals(self) -> list[dict[str, Any]]:
        if self.max_pair_marginal_edges <= 0 or len(self.fidelity_columns) < 2:
            return []

        train_bucket_indices = self._column_bucket_indices_for_df(self.train_df, self.fidelity_columns)
        candidates: list[dict[str, Any]] = []
        for left_pos, left in enumerate(self.fidelity_columns):
            left_probs = np.asarray(self.train_distributions[left]["probs"], dtype=float)
            left_idx = train_bucket_indices[left]
            left_bins = int(len(left_probs))
            if left_bins <= 0:
                continue
            for right in self.fidelity_columns[left_pos + 1 :]:
                right_probs = np.asarray(self.train_distributions[right]["probs"], dtype=float)
                right_idx = train_bucket_indices[right]
                right_bins = int(len(right_probs))
                if right_bins <= 0:
                    continue

                valid = (left_idx >= 0) & (right_idx >= 0)
                if not valid.any():
                    continue

                flat = left_idx[valid] * right_bins + right_idx[valid]
                counts = np.bincount(flat, minlength=left_bins * right_bins).astype(float)
                probs = counts / max(float(counts.sum()), 1.0)
                mi = self._normalized_mutual_information_from_joint(probs.reshape(left_bins, right_bins))
                candidates.append(
                    {
                        "left": left,
                        "right": right,
                        "left_bins": left_bins,
                        "right_bins": right_bins,
                        "mi": float(mi),
                        "probs": probs,
                    }
                )

        if not candidates:
            return []

        candidates.sort(key=lambda item: (float(item["mi"]), item["left"], item["right"]), reverse=True)
        return candidates[: self.max_pair_marginal_edges]

    def _row_column_probability(self, row: dict[str, Any] | pd.Series, column: str) -> float:
        info = self.schema_card["columns"][column]
        train_dist = self.train_distributions[column]
        value = row[column]
        if info["type"] == "numerical":
            bucket = _digitize_value(float(value), train_dist["edges"])
            return float(train_dist["probs"][bucket])
        if info["type"] == "discrete_numerical":
            idx = self._match_discrete_value(value, train_dist["values"])
            return float(train_dist["probs"][idx])
        values = train_dist["values"]
        try:
            idx = values.index(str(value))
        except ValueError:
            return 1e-12
        return float(train_dist["probs"][idx])

    def _row_marginal_prob_geomean(
        self,
        row: dict[str, Any] | pd.Series,
        columns: list[str] | None = None,
    ) -> float:
        use_columns = self.fidelity_columns if columns is None else columns
        probs = [self._row_column_probability(row, column) for column in use_columns]
        return _safe_geometric_mean(probs)

    def _row_gate_probability(self, row: dict[str, Any] | pd.Series) -> float:
        return self._row_marginal_prob_geomean(row, columns=self.feature_columns)

    def _assign_bins_from_edges(self, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        if len(edges) <= 2:
            return np.zeros(len(values), dtype=int)
        clipped = np.clip(values, float(edges[0]), float(edges[-1]))
        bins = np.digitize(clipped, edges[1:-1], right=False)
        return bins.astype(int)

    def _density_normalized_distance(self, encoded_row: np.ndarray) -> tuple[float, float, float, float]:
        if self.nn_backend == "torch":
            assert self.train_tensor is not None
            assert self.holdout_tensor is not None
            assert self.density_reference_tensor is not None
            nn_train = float(self._torch_knn_min(encoded_row, self.train_tensor)[0])
            nn_holdout = float(self._torch_knn_min(encoded_row, self.holdout_tensor)[0])
            k = min(len(self.density_reference_matrix), self.density_k)
            density_distances = self._torch_knn_topk(encoded_row, self.density_reference_tensor, k=max(k, 1))
            local_density = 1.0 / max(float(density_distances[0].mean()), 1e-12)
        else:
            assert self.nn_train is not None
            assert self.nn_holdout is not None
            assert self.nn_density is not None
            nn_train_distance, _ = self.nn_train.kneighbors(encoded_row, n_neighbors=1)
            nn_holdout_distance, _ = self.nn_holdout.kneighbors(encoded_row, n_neighbors=1)
            nn_train = float(nn_train_distance[0][0])
            nn_holdout = float(nn_holdout_distance[0][0])
            k = min(len(self.density_reference_matrix), self.density_k)
            density_distances, _ = self.nn_density.kneighbors(encoded_row, n_neighbors=max(k, 1))
            local_density = 1.0 / max(float(density_distances[0].mean()), 1e-12)

        density_bucket = int(self._assign_bins_from_edges(np.asarray([local_density]), self.train_density_edges)[0])
        expected_nn = float(self.train_density_expected_nn[density_bucket])
        normalized = nn_train / max(expected_nn, 1e-12)
        return nn_train, nn_holdout, local_density, normalized

    def _privacy_components_frame_for_df(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "nn_distance_train",
            "nn_distance_holdout",
            "holdout_gap",
            "local_density",
            "density_normalized_nn_distance",
            "p_marginal_geomean",
            "gate_stratum",
            "privacy_score_v1",
            "privacy_score_v2",
            "privacy_score_v3",
            "privacy_score_selected",
        ]
        if df.empty:
            return pd.DataFrame(columns=columns)

        normalized_df = df.reset_index(drop=True)
        gate_bucket_indices = self._column_bucket_indices_for_df(normalized_df, self.feature_columns)
        gate_probs = self._prob_geomean_from_bucket_indices(gate_bucket_indices, self.feature_columns)
        encoded = self._encode_df(normalized_df)
        if self.nn_backend == "torch":
            assert self.train_tensor is not None
            assert self.holdout_tensor is not None
            assert self.density_reference_tensor is not None
            nn_train = self._torch_knn_min(encoded, self.train_tensor).astype(float)
            nn_holdout = self._torch_knn_min(encoded, self.holdout_tensor).astype(float)
            k = min(len(self.density_reference_matrix), self.density_k)
            density_distances = self._torch_knn_topk(encoded, self.density_reference_tensor, k=max(k, 1)).astype(float)
            local_density = 1.0 / np.clip(density_distances.mean(axis=1), 1e-12, None)
        else:
            assert self.nn_train is not None
            assert self.nn_holdout is not None
            assert self.nn_density is not None
            nn_train_distances, _ = self.nn_train.kneighbors(encoded, n_neighbors=1)
            nn_holdout_distances, _ = self.nn_holdout.kneighbors(encoded, n_neighbors=1)
            k = min(len(self.density_reference_matrix), self.density_k)
            density_distances, _ = self.nn_density.kneighbors(encoded, n_neighbors=max(k, 1))
            local_density = 1.0 / np.clip(density_distances.mean(axis=1), 1e-12, None)
            nn_train = nn_train_distances[:, 0].astype(float)
            nn_holdout = nn_holdout_distances[:, 0].astype(float)

        density_buckets = self._assign_bins_from_edges(local_density, self.train_density_edges)
        density_buckets = np.clip(density_buckets, 0, len(self.train_density_expected_nn) - 1)
        expected_nn = self.train_density_expected_nn[density_buckets]
        gate_strata = self._assign_bins_from_edges(gate_probs, self.train_gate_edges)
        normalized = nn_train / np.clip(expected_nn.astype(float), 1e-12, None)
        gamma_penalty = self.gamma * np.maximum(0.0, nn_holdout - nn_train)
        v1 = np.log1p(nn_train)
        v2 = np.maximum(0.0, normalized - gamma_penalty)
        v3 = np.maximum(0.0, nn_train * (1.0 - gate_probs) - gamma_penalty)
        selected_values = {"v1": v1, "v2": v2, "v3": v3}.get(self.privacy_version, v2)
        return pd.DataFrame(
            {
                "nn_distance_train": nn_train.astype(float, copy=False),
                "nn_distance_holdout": nn_holdout.astype(float, copy=False),
                "holdout_gap": (nn_holdout - nn_train).astype(float, copy=False),
                "local_density": local_density.astype(float, copy=False),
                "density_normalized_nn_distance": normalized.astype(float, copy=False),
                "p_marginal_geomean": gate_probs.astype(float, copy=False),
                "gate_stratum": gate_strata.astype(int, copy=False),
                "privacy_score_v1": v1.astype(float, copy=False),
                "privacy_score_v2": v2.astype(float, copy=False),
                "privacy_score_v3": v3.astype(float, copy=False),
                "privacy_score_selected": selected_values.astype(float, copy=False),
            }
        )

    def _privacy_components_for_df(self, df: pd.DataFrame) -> list[dict[str, float]]:
        return self._privacy_components_frame_for_df(df).to_dict(orient="records")

    def _row_privacy_components(self, row: dict[str, Any] | pd.Series) -> dict[str, float]:
        df = pd.DataFrame([dict(row)], columns=self.column_order)
        return self._privacy_components_for_df(df)[0]

    def compute_surrogates(
        self,
        valid_df: pd.DataFrame,
        show_progress: bool = False,
        progress_desc: str = "surrogate scoring",
        candidate_ids: np.ndarray | list[int] | None = None,
    ) -> list[dict[str, Any]]:
        if valid_df.empty:
            return []
        normalized_df = valid_df.reset_index(drop=True)
        if candidate_ids is None:
            candidate_ids_array = np.arange(len(normalized_df), dtype=int)
        else:
            candidate_ids_array = np.asarray(candidate_ids, dtype=int)
            if len(candidate_ids_array) != len(normalized_df):
                raise ValueError(
                    f"candidate_ids length mismatch: got {len(candidate_ids_array)} ids for {len(normalized_df)} rows."
                )
        fidelity_bucket_indices = self._column_bucket_indices_for_df(normalized_df, self.fidelity_columns)
        fidelity_1d_support = self._prob_geomean_from_bucket_indices(fidelity_bucket_indices, self.fidelity_columns)
        fidelity_2d_support = self._pair_prob_geomean_from_bucket_indices(fidelity_bucket_indices)
        fidelity_1d_balance = self._pool_balance_1d_from_bucket_indices(fidelity_bucket_indices, self.fidelity_columns)
        fidelity_2d_balance = self._pool_balance_2d_from_bucket_indices(fidelity_bucket_indices)
        fidelity_1d_rank = _rank_normalize(fidelity_1d_balance)
        fidelity_2d_rank = _rank_normalize(fidelity_2d_balance)
        fidelity_support_1d_rank = _rank_normalize(fidelity_1d_support)
        fidelity_support_2d_rank = _rank_normalize(fidelity_2d_support)
        fidelity_surrogates = 0.5 * fidelity_1d_rank + 0.5 * fidelity_2d_rank
        privacy_df = self._privacy_components_frame_for_df(normalized_df)
        if show_progress:
            for _ in _progress(range(1), total=1, desc=progress_desc, disable=False):
                pass

        privacy_rank = _rank_normalize(privacy_df["privacy_score_v2"].to_numpy(dtype=float, copy=False))
        support_tiebreak = 0.4 * fidelity_support_1d_rank + 0.6 * fidelity_support_2d_rank
        preselect_band = 0.45 * fidelity_1d_rank + 0.45 * fidelity_2d_rank + 0.10 * support_tiebreak
        preselect_fidelity_safe = (
            0.35 * fidelity_1d_rank
            + 0.35 * fidelity_2d_rank
            + 0.15 * fidelity_support_1d_rank
            + 0.15 * fidelity_support_2d_rank
        )
        preselect_stage_b = 0.40 * fidelity_1d_rank + 0.40 * fidelity_2d_rank + 0.20 * support_tiebreak
        surrogate_df = pd.DataFrame(
            {
                "candidate_index": np.arange(len(normalized_df), dtype=int),
                "candidate_id": candidate_ids_array.astype(int, copy=False),
                "s_fid_sur": fidelity_surrogates.astype(float, copy=False),
                "s_pareto_fid_1d_sur": fidelity_1d_rank.astype(float, copy=False),
                "s_pareto_fid_2d_sur": fidelity_2d_rank.astype(float, copy=False),
                "s_pareto_priv_sur": privacy_df["privacy_score_v2"].to_numpy(dtype=float, copy=False),
                "s_fid_sur_1d": fidelity_1d_balance.astype(float, copy=False),
                "s_fid_sur_2d": fidelity_2d_balance.astype(float, copy=False),
                "s_fid_sur_1d_rank": fidelity_1d_rank.astype(float, copy=False),
                "s_fid_sur_2d_rank": fidelity_2d_rank.astype(float, copy=False),
                "s_fid_support_1d": fidelity_1d_support.astype(float, copy=False),
                "s_fid_support_2d": fidelity_2d_support.astype(float, copy=False),
                "s_fid_support_1d_rank": fidelity_support_1d_rank.astype(float, copy=False),
                "s_fid_support_2d_rank": fidelity_support_2d_rank.astype(float, copy=False),
                "s_preselect_band": preselect_band.astype(float, copy=False),
                "s_preselect_fidelity_safe": preselect_fidelity_safe.astype(float, copy=False),
                "s_preselect_stage_b": preselect_stage_b.astype(float, copy=False),
                "s_preselect_support_tiebreak": support_tiebreak.astype(float, copy=False),
                "s_preselect_priv_tiebreak": privacy_rank.astype(float, copy=False),
                "s_priv_sur": privacy_df["privacy_score_v2"].to_numpy(dtype=float, copy=False),
                "s_priv_sur_preselect": privacy_df["privacy_score_v2"].to_numpy(dtype=float, copy=False),
                "s_priv_sur_selected": privacy_df["privacy_score_selected"].to_numpy(dtype=float, copy=False),
                "s_priv_sur_v1": privacy_df["privacy_score_v1"].to_numpy(dtype=float, copy=False),
                "s_priv_sur_v2": privacy_df["privacy_score_v2"].to_numpy(dtype=float, copy=False),
                "s_priv_sur_v3": privacy_df["privacy_score_v3"].to_numpy(dtype=float, copy=False),
                "p_marginal_geomean": privacy_df["p_marginal_geomean"].to_numpy(dtype=float, copy=False),
                "gate_stratum": privacy_df["gate_stratum"].to_numpy(dtype=int, copy=False),
                "rarity_score": 1.0 - privacy_df["p_marginal_geomean"].to_numpy(dtype=float, copy=False),
                "nn_distance_train": privacy_df["nn_distance_train"].to_numpy(dtype=float, copy=False),
                "nn_distance_holdout": privacy_df["nn_distance_holdout"].to_numpy(dtype=float, copy=False),
                "density_normalized_nn_distance": privacy_df["density_normalized_nn_distance"].to_numpy(
                    dtype=float, copy=False
                ),
            }
        )
        return surrogate_df.to_dict(orient="records")

    def _build_preselect_quota_targets(
        self,
        bucket_indices: dict[str, np.ndarray],
        pair_codes: list[np.ndarray],
        budget: int,
        *,
        target_mode: str = "available_empirical",
    ) -> tuple[dict[str, np.ndarray], list[np.ndarray], dict[str, np.ndarray], list[np.ndarray]]:
        quota_targets_1d: dict[str, np.ndarray] = {}
        selected_counts_1d: dict[str, np.ndarray] = {}
        for column in self.fidelity_columns:
            codes = np.asarray(bucket_indices[column], dtype=int)
            num_bins = len(self.train_distributions[column]["probs"])
            available_counts = np.bincount(codes[codes >= 0], minlength=num_bins).astype(int)
            if target_mode == "train_clipped_by_availability":
                target_probs = np.asarray(self.train_distributions[column]["probs"], dtype=float)
            elif target_mode == "available_empirical":
                target_probs = available_counts.astype(float) / max(float(available_counts.sum()), 1.0)
            else:
                raise ValueError(f"Unsupported target_mode={target_mode}")
            quota_targets_1d[column] = self._allocate_counts_from_probs(
                target_probs,
                available_counts,
                budget,
            )
            selected_counts_1d[column] = np.zeros(num_bins, dtype=int)

        quota_targets_2d: list[np.ndarray] = []
        selected_counts_2d: list[np.ndarray] = []
        for edge, codes in zip(self.pair_marginal_edges, pair_codes):
            num_bins = int(edge["left_bins"]) * int(edge["right_bins"])
            available_counts = np.bincount(codes[codes >= 0], minlength=num_bins).astype(int)
            if target_mode == "train_clipped_by_availability":
                target_probs = np.asarray(edge["probs"], dtype=float)
            elif target_mode == "available_empirical":
                target_probs = available_counts.astype(float) / max(float(available_counts.sum()), 1.0)
            else:
                raise ValueError(f"Unsupported target_mode={target_mode}")
            quotas = self._allocate_counts_from_probs(
                target_probs,
                available_counts,
                budget,
            )
            quota_targets_2d.append(quotas)
            selected_counts_2d.append(np.zeros_like(quotas, dtype=int))

        return quota_targets_1d, quota_targets_2d, selected_counts_1d, selected_counts_2d

    def _preselect_quota_fill(
        self,
        *,
        bucket_indices: dict[str, np.ndarray],
        pair_codes: list[np.ndarray],
        target_preselect: int,
        base_score: np.ndarray,
        support_tiebreak: np.ndarray,
        privacy_tiebreak: np.ndarray,
        privacy_weight: float,
        refine_privacy_weight: float,
        target_mode: str,
        show_progress: bool,
        progress_desc: str,
    ) -> dict[str, Any]:
        total_rows = int(len(base_score))
        target_preselect = min(max(1, int(target_preselect)), total_rows)
        quota_targets_1d, quota_targets_2d, selected_counts_1d, selected_counts_2d = self._build_preselect_quota_targets(
            bucket_indices=bucket_indices,
            pair_codes=pair_codes,
            budget=target_preselect,
            target_mode=target_mode,
        )

        selected_mask = np.zeros(total_rows, dtype=bool)
        batch_id = np.full(total_rows, -1, dtype=int)
        batch_score_1d = np.zeros(total_rows, dtype=float)
        batch_score_2d = np.zeros(total_rows, dtype=float)
        batch_score_priv = np.zeros(total_rows, dtype=float)
        batch_score_support = np.zeros(total_rows, dtype=float)
        batch_score_static = np.zeros(total_rows, dtype=float)
        batch_score_final = np.zeros(total_rows, dtype=float)

        batch_size = max(128, min(1536, int(round(target_preselect / 24.0))))
        remaining_target = int(target_preselect)
        num_batches = int(np.ceil(target_preselect / max(batch_size, 1)))

        w_priv = float(np.clip(privacy_weight, 0.0, 0.10))
        w_support = 0.08
        w_static = 0.10
        w_quota = max(0.0, 1.0 - w_priv - w_support - w_static)
        w_quota_1d = 0.5 * w_quota
        w_quota_2d = 0.5 * w_quota

        batch_iter = _progress(
            range(num_batches),
            total=num_batches,
            desc=progress_desc,
            disable=not show_progress,
        )
        for current_batch in batch_iter:
            if remaining_target <= 0:
                break

            _, add_support_1d = self._target_count_support_scores_1d(
                bucket_indices,
                selected_counts_1d,
                quota_targets_1d,
            )
            _, add_support_2d = self._target_count_support_scores_2d(
                pair_codes,
                selected_counts_2d,
                quota_targets_2d,
            )
            if add_support_1d.size == 0:
                add_support_1d = np.zeros(total_rows, dtype=float)
            if add_support_2d.size == 0:
                add_support_2d = np.zeros(total_rows, dtype=float)

            final_score = (
                w_quota_1d * add_support_1d
                + w_quota_2d * add_support_2d
                + w_static * base_score
                + w_support * support_tiebreak
                + w_priv * privacy_tiebreak
            )
            final_score[selected_mask] = -np.inf

            available_indices = np.flatnonzero(~selected_mask)
            if available_indices.size == 0:
                break
            take_k = min(int(remaining_target), int(batch_size), int(available_indices.size))
            if take_k <= 0:
                break
            if available_indices.size <= take_k:
                chosen = available_indices
            else:
                local_scores = final_score[available_indices]
                top_local = np.argpartition(-local_scores, take_k - 1)[:take_k]
                chosen = available_indices[top_local]
            chosen = chosen[
                np.lexsort(
                    (
                        chosen,
                        -support_tiebreak[chosen],
                        -privacy_tiebreak[chosen],
                        -base_score[chosen],
                        -add_support_2d[chosen],
                        -add_support_1d[chosen],
                        -final_score[chosen],
                    )
                )
            ]

            selected_mask[chosen] = True
            batch_id[chosen] = int(current_batch)
            batch_score_1d[chosen] = add_support_1d[chosen]
            batch_score_2d[chosen] = add_support_2d[chosen]
            batch_score_priv[chosen] = privacy_tiebreak[chosen]
            batch_score_support[chosen] = support_tiebreak[chosen]
            batch_score_static[chosen] = base_score[chosen]
            batch_score_final[chosen] = final_score[chosen]

            for column in self.fidelity_columns:
                self._add_code_count_delta(selected_counts_1d[column], bucket_indices[column][chosen], 1)

            for pair_idx, codes in enumerate(pair_codes):
                self._add_code_count_delta(selected_counts_2d[pair_idx], codes[chosen], 1)

            remaining_target -= int(len(chosen))
            if hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(batch=current_batch, remaining=remaining_target)

        refine_utility = 0.60 * base_score + 0.40 * support_tiebreak
        privacy_component = None
        if refine_privacy_weight > 1e-8:
            privacy_component = float(refine_privacy_weight) * privacy_tiebreak
        selected_mask, refine_report = self._refine_subset_to_target_counts(
            bucket_indices=bucket_indices,
            pair_codes=pair_codes,
            selected_mask=selected_mask,
            target_counts_1d=quota_targets_1d,
            target_counts_2d=quota_targets_2d,
            utility=refine_utility,
            privacy_component=privacy_component,
            max_rounds=8,
            batch_scale=0.003,
        )

        return {
            "selected_mask": selected_mask,
            "keep_indices": np.sort(np.flatnonzero(selected_mask).astype(int, copy=False)),
            "quota_targets_1d": quota_targets_1d,
            "quota_targets_2d": quota_targets_2d,
            "batch_id": batch_id,
            "batch_score_1d": batch_score_1d,
            "batch_score_2d": batch_score_2d,
            "batch_score_priv": batch_score_priv,
            "batch_score_support": batch_score_support,
            "batch_score_static": batch_score_static,
            "batch_score_final": batch_score_final,
            "batch_size": int(batch_size),
            "num_batches": int(num_batches),
            "privacy_weight": float(w_priv),
            "target_mode": target_mode,
            "refine_report": refine_report,
        }

    def _build_blended_preselect_targets(
        self,
        *,
        bucket_indices: dict[str, np.ndarray],
        pair_codes: list[np.ndarray],
        selected_mask: np.ndarray,
        budget: int,
        blend_alpha: float,
    ) -> tuple[dict[str, np.ndarray], list[np.ndarray], dict[str, np.ndarray], list[np.ndarray]]:
        blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        anchor_counts_1d, anchor_counts_2d = self._subset_count_state_from_mask(
            bucket_indices=bucket_indices,
            pair_codes=pair_codes,
            selected_mask=selected_mask,
        )
        train_targets_1d, train_targets_2d, _, _ = self._build_preselect_quota_targets(
            bucket_indices=bucket_indices,
            pair_codes=pair_codes,
            budget=budget,
            target_mode="train_clipped_by_availability",
        )

        blended_1d: dict[str, np.ndarray] = {}
        for column in self.fidelity_columns:
            available_counts = self._column_counts_from_bucket_indices(column, bucket_indices[column]).astype(int)
            anchor_probs = np.asarray(anchor_counts_1d[column], dtype=float) / max(float(budget), 1.0)
            train_probs = np.asarray(train_targets_1d[column], dtype=float) / max(float(budget), 1.0)
            blended_probs = (1.0 - blend_alpha) * anchor_probs + blend_alpha * train_probs
            blended_1d[column] = self._allocate_counts_from_probs(
                blended_probs,
                available_counts,
                budget,
            )

        blended_2d: list[np.ndarray] = []
        for edge_idx, (edge, codes) in enumerate(zip(self.pair_marginal_edges, pair_codes)):
            available_counts = np.bincount(codes[codes >= 0], minlength=len(edge["probs"])).astype(int)
            anchor_probs = np.asarray(anchor_counts_2d[edge_idx], dtype=float) / max(float(budget), 1.0)
            train_probs = np.asarray(train_targets_2d[edge_idx], dtype=float) / max(float(budget), 1.0)
            blended_probs = (1.0 - blend_alpha) * anchor_probs + blend_alpha * train_probs
            blended_2d.append(
                self._allocate_counts_from_probs(
                    blended_probs,
                    available_counts,
                    budget,
                )
            )
        return blended_1d, blended_2d, anchor_counts_1d, anchor_counts_2d

    def _annotate_preselect_surrogates(
        self,
        surrogate_records: list[dict[str, Any]],
        *,
        selected_mask: np.ndarray,
        band_mask: np.ndarray,
        batch_id: np.ndarray,
        batch_score_1d: np.ndarray,
        batch_score_2d: np.ndarray,
        batch_score_priv: np.ndarray,
        batch_score_support: np.ndarray,
        batch_score_static: np.ndarray,
        batch_score_final: np.ndarray,
        mode: str,
        target_source: str,
        band_target: int,
        band_rows: int,
        refine_applied: bool,
    ) -> None:
        for idx, record in enumerate(surrogate_records):
            record["preselect_batch_id"] = int(batch_id[idx])
            record["preselect_batch_score_1d"] = float(batch_score_1d[idx])
            record["preselect_batch_score_2d"] = float(batch_score_2d[idx])
            record["preselect_batch_score_priv"] = float(batch_score_priv[idx])
            record["preselect_batch_score_support"] = float(batch_score_support[idx])
            record["preselect_batch_score_static"] = float(batch_score_static[idx])
            record["preselect_batch_score"] = float(batch_score_final[idx])
            record["preselect_selected"] = bool(selected_mask[idx])
            record["preselect_band_selected"] = bool(band_mask[idx])
            record["preselect_band_target"] = int(band_target)
            record["preselect_band_rows"] = int(band_rows)
            record["preselect_mode"] = mode
            record["preselect_target_source"] = target_source
            record["preselect_refine_applied"] = bool(refine_applied)

    def dual_median_filter_baseline(
        self,
        valid_records: list[dict[str, Any]],
        surrogate_records: list[dict[str, Any]],
        target_preselect: int,
        *,
        show_progress: bool = False,
        progress_desc: str = "preselect baseline",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not valid_records:
            self.last_preselect_report = {"mode": "three_objective_preselect_v3", "rows": 0, "selected_rows": 0}
            return [], []

        total_rows = len(valid_records)
        target_preselect = min(max(1, int(target_preselect)), total_rows)
        if target_preselect >= total_rows:
            self.last_preselect_report = {
                "mode": "three_objective_preselect_v3",
                "target_source": "candidate_pool_empirical_scaled",
                "rows": int(total_rows),
                "selected_rows": int(total_rows),
                "privacy_weight": 0.10,
                "refine_applied": False,
            }
            return valid_records.copy(), surrogate_records.copy()

        candidate_df = pd.DataFrame([record["row"] for record in valid_records], columns=self.column_order)
        fidelity_bucket_indices = self._column_bucket_indices_for_df(candidate_df, self.fidelity_columns)
        pair_codes = self._pair_codes_from_bucket_indices(fidelity_bucket_indices)

        base_score = np.asarray(
            [float(record.get("s_preselect_band", record.get("s_fid_sur", 0.0))) for record in surrogate_records],
            dtype=float,
        )
        support_tiebreak = np.asarray(
            [float(record.get("s_preselect_support_tiebreak", 0.0)) for record in surrogate_records],
            dtype=float,
        )
        privacy_tiebreak = np.asarray(
            [float(record.get("s_preselect_priv_tiebreak", 0.0)) for record in surrogate_records],
            dtype=float,
        )

        fill_report = self._preselect_quota_fill(
            bucket_indices=fidelity_bucket_indices,
            pair_codes=pair_codes,
            target_preselect=target_preselect,
            base_score=base_score,
            support_tiebreak=support_tiebreak,
            privacy_tiebreak=privacy_tiebreak,
            privacy_weight=0.10,
            refine_privacy_weight=0.10,
            target_mode="available_empirical",
            show_progress=show_progress,
            progress_desc=progress_desc,
        )

        selected_mask = np.asarray(fill_report["selected_mask"], dtype=bool)
        band_mask = np.ones(total_rows, dtype=bool)
        self._annotate_preselect_surrogates(
            surrogate_records,
            selected_mask=selected_mask,
            band_mask=band_mask,
            batch_id=np.asarray(fill_report["batch_id"], dtype=int),
            batch_score_1d=np.asarray(fill_report["batch_score_1d"], dtype=float),
            batch_score_2d=np.asarray(fill_report["batch_score_2d"], dtype=float),
            batch_score_priv=np.asarray(fill_report["batch_score_priv"], dtype=float),
            batch_score_support=np.asarray(fill_report["batch_score_support"], dtype=float),
            batch_score_static=np.asarray(fill_report["batch_score_static"], dtype=float),
            batch_score_final=np.asarray(fill_report["batch_score_final"], dtype=float),
            mode="three_objective_preselect_v3",
            target_source="candidate_pool_empirical_scaled",
            band_target=total_rows,
            band_rows=total_rows,
            refine_applied=bool(fill_report["refine_report"].get("applied", False)),
        )

        keep_indices = np.asarray(fill_report["keep_indices"], dtype=int)
        self.last_preselect_report = {
            "mode": "three_objective_preselect_v3",
            "target_source": "candidate_pool_empirical_scaled",
            "rows": int(total_rows),
            "selected_rows": int(len(keep_indices)),
            "band_rows": int(total_rows),
            "band_target": int(total_rows),
            "privacy_weight": float(fill_report["privacy_weight"]),
            "batch_size": int(fill_report["batch_size"]),
            "num_batches": int(fill_report["num_batches"]),
            "refine_applied": bool(fill_report["refine_report"].get("applied", False)),
            "refine_report": fill_report["refine_report"],
        }
        kept_valid = [valid_records[int(idx)] for idx in keep_indices.tolist()]
        kept_sur = [surrogate_records[int(idx)] for idx in keep_indices.tolist()]
        return kept_valid, kept_sur

    def dual_median_filter(
        self,
        valid_records: list[dict[str, Any]],
        surrogate_records: list[dict[str, Any]],
        target_preselect: int,
        *,
        anchor_candidate_ids: np.ndarray | list[int] | None = None,
        show_progress: bool = False,
        progress_desc: str = "preselect construction",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not valid_records:
            self.last_preselect_report = {"mode": "two_stage_band_quota_v2", "rows": 0, "selected_rows": 0}
            return [], []

        total_rows = len(valid_records)
        target_preselect = min(max(1, int(target_preselect)), total_rows)
        if target_preselect >= total_rows:
            self.last_preselect_report = {
                "mode": "two_stage_band_quota_v2",
                "target_source": "full_pool_passthrough",
                "rows": int(total_rows),
                "selected_rows": int(total_rows),
                "refine_applied": False,
            }
            return valid_records.copy(), surrogate_records.copy()

        candidate_df = pd.DataFrame([record["row"] for record in valid_records], columns=self.column_order)
        fidelity_bucket_indices = self._column_bucket_indices_for_df(candidate_df, self.fidelity_columns)
        pair_codes = self._pair_codes_from_bucket_indices(fidelity_bucket_indices)

        stage_a_base = np.asarray(
            [
                float(record.get("s_preselect_band", record.get("s_preselect_fidelity_safe", record.get("s_fid_sur", 0.0))))
                for record in surrogate_records
            ],
            dtype=float,
        )
        stage_b_base = np.asarray(
            [
                float(
                    record.get(
                        "s_preselect_stage_b",
                        record.get("s_preselect_fidelity_safe", record.get("s_preselect_band", record.get("s_fid_sur", 0.0))),
                    )
                )
                for record in surrogate_records
            ],
            dtype=float,
        )
        support_tiebreak = np.asarray(
            [float(record.get("s_preselect_support_tiebreak", 0.0)) for record in surrogate_records],
            dtype=float,
        )
        privacy_tiebreak = np.asarray(
            [float(record.get("s_preselect_priv_tiebreak", 0.0)) for record in surrogate_records],
            dtype=float,
        )
        candidate_id_array = np.asarray(
            [int(record.get("candidate_id", idx)) for idx, record in enumerate(surrogate_records)],
            dtype=int,
        )
        band_scale = 1.40
        band_target = min(total_rows, max(target_preselect, int(np.ceil(float(target_preselect) * band_scale))))
        stage_a_report = self._preselect_quota_fill(
            bucket_indices=fidelity_bucket_indices,
            pair_codes=pair_codes,
            target_preselect=band_target,
            base_score=stage_a_base,
            support_tiebreak=support_tiebreak,
            privacy_tiebreak=np.zeros(total_rows, dtype=float),
            privacy_weight=0.0,
            refine_privacy_weight=0.0,
            target_mode="train_clipped_by_availability",
            show_progress=show_progress,
            progress_desc=f"{progress_desc} stage_a",
        )
        band_mask = np.asarray(stage_a_report["selected_mask"], dtype=bool)
        band_indices = np.flatnonzero(band_mask)
        if band_indices.size == 0:
            baseline_valid, baseline_sur = self.dual_median_filter_baseline(
                valid_records=valid_records,
                surrogate_records=surrogate_records,
                target_preselect=target_preselect,
                show_progress=show_progress,
                progress_desc=f"{progress_desc} candidate_empty_band_fallback",
            )
            self.last_preselect_report = {
                "mode": "two_stage_band_quota_v2",
                "target_source": "candidate_empty_band_fallback",
                "rows": int(total_rows),
                "selected_rows": int(len(baseline_valid)),
                "band_rows": 0,
                "refine_applied": False,
            }
            return baseline_valid, baseline_sur

        band_bucket_indices = {column: codes[band_indices] for column, codes in fidelity_bucket_indices.items()}
        band_pair_codes = [codes[band_indices] for codes in pair_codes]
        stage_b_report = self._preselect_quota_fill(
            bucket_indices=band_bucket_indices,
            pair_codes=band_pair_codes,
            target_preselect=target_preselect,
            base_score=stage_b_base[band_indices],
            support_tiebreak=support_tiebreak[band_indices],
            privacy_tiebreak=privacy_tiebreak[band_indices],
            privacy_weight=0.05,
            refine_privacy_weight=0.05,
            target_mode="available_empirical",
            show_progress=show_progress,
            progress_desc=f"{progress_desc} stage_b",
        )

        stage_b_mask_local = np.asarray(stage_b_report["selected_mask"], dtype=bool)
        selected_mask = np.zeros(total_rows, dtype=bool)
        selected_mask[band_indices] = stage_b_mask_local
        keep_indices = np.sort(np.flatnonzero(selected_mask).astype(int, copy=False))

        batch_id = np.full(total_rows, -1, dtype=int)
        batch_score_1d = np.zeros(total_rows, dtype=float)
        batch_score_2d = np.zeros(total_rows, dtype=float)
        batch_score_priv = np.zeros(total_rows, dtype=float)
        batch_score_support = np.zeros(total_rows, dtype=float)
        batch_score_static = np.zeros(total_rows, dtype=float)
        batch_score_final = np.zeros(total_rows, dtype=float)
        batch_id[band_indices] = np.asarray(stage_b_report["batch_id"], dtype=int)
        batch_score_1d[band_indices] = np.asarray(stage_b_report["batch_score_1d"], dtype=float)
        batch_score_2d[band_indices] = np.asarray(stage_b_report["batch_score_2d"], dtype=float)
        batch_score_priv[band_indices] = np.asarray(stage_b_report["batch_score_priv"], dtype=float)
        batch_score_support[band_indices] = np.asarray(stage_b_report["batch_score_support"], dtype=float)
        batch_score_static[band_indices] = np.asarray(stage_b_report["batch_score_static"], dtype=float)
        batch_score_final[band_indices] = np.asarray(stage_b_report["batch_score_final"], dtype=float)

        selected_counts_1d, selected_counts_2d = self._subset_count_state_from_mask(
            bucket_indices=fidelity_bucket_indices,
            pair_codes=pair_codes,
            selected_mask=selected_mask,
        )
        anchor_overlap = None
        if anchor_candidate_ids is not None:
            anchor_mask = np.isin(candidate_id_array, np.asarray(anchor_candidate_ids, dtype=int))
            union = np.count_nonzero(anchor_mask | selected_mask)
            anchor_overlap = {
                "rows": int(np.count_nonzero(anchor_mask & selected_mask)),
                "ratio_vs_selected": float(np.count_nonzero(anchor_mask & selected_mask) / max(int(np.count_nonzero(selected_mask)), 1)),
                "jaccard": float(np.count_nonzero(anchor_mask & selected_mask) / max(int(union), 1)),
            }

        self._annotate_preselect_surrogates(
            surrogate_records,
            selected_mask=selected_mask,
            band_mask=band_mask,
            batch_id=batch_id,
            batch_score_1d=batch_score_1d,
            batch_score_2d=batch_score_2d,
            batch_score_priv=batch_score_priv,
            batch_score_support=batch_score_support,
            batch_score_static=batch_score_static,
            batch_score_final=batch_score_final,
            mode="two_stage_band_quota_v2",
            target_source="stage_a_train_clipped_band_then_stage_b_empirical_keep",
            band_target=band_target,
            band_rows=int(band_mask.sum()),
            refine_applied=bool(stage_b_report["refine_report"].get("applied", False)),
        )

        self.last_preselect_report = {
            "mode": "two_stage_band_quota_v2",
            "target_source": "stage_a_train_clipped_band_then_stage_b_empirical_keep",
            "rows": int(total_rows),
            "selected_rows": int(len(keep_indices)),
            "band_rows": int(band_mask.sum()),
            "band_target": int(band_target),
            "privacy_weight": float(stage_b_report["privacy_weight"]),
            "refine_applied": bool(stage_b_report["refine_report"].get("applied", False)),
            "refine_report": stage_b_report["refine_report"],
            "alignment_to_stage_b_targets": {
                "alignment_1d": float(
                    self._subset_alignment_from_target_counts_1d(
                        selected_counts_1d,
                        stage_b_report["quota_targets_1d"],
                    )
                ),
                "alignment_2d": float(
                    self._subset_alignment_from_target_counts_2d(
                        selected_counts_2d,
                        stage_b_report["quota_targets_2d"],
                    )
                ),
            },
            "anchor_overlap": anchor_overlap,
            "stage_a": {
                "mode": "fidelity_safe_band_quota_fill",
                "selected_rows": int(band_mask.sum()),
                "target_mode": "train_clipped_by_availability",
                "privacy_weight": float(stage_a_report["privacy_weight"]),
                "batch_size": int(stage_a_report["batch_size"]),
                "num_batches": int(stage_a_report["num_batches"]),
                "components": [
                    "1d_train_clipped_quota_alignment",
                    "2d_train_clipped_quota_alignment",
                    "fidelity_safe_band_score",
                ],
                "refine_report": stage_a_report["refine_report"],
            },
            "stage_b": {
                "mode": "empirical_keep_within_fidelity_safe_band",
                "selected_rows": int(len(keep_indices)),
                "target_mode": "available_empirical",
                "privacy_weight": float(stage_b_report["privacy_weight"]),
                "batch_size": int(stage_b_report["batch_size"]),
                "num_batches": int(stage_b_report["num_batches"]),
                "components": [
                    "1d_band_empirical_quota_alignment",
                    "2d_band_empirical_quota_alignment",
                    "fidelity_safe_stage_b_score",
                    "weak_privacy_tiebreak",
                ],
            },
        }
        kept_valid = [valid_records[int(idx)] for idx in keep_indices.tolist()]
        kept_sur = [surrogate_records[int(idx)] for idx in keep_indices.tolist()]
        return kept_valid, kept_sur

    def _column_similarity(self, counts: np.ndarray, probs_train: np.ndarray) -> float:
        probs = counts / max(counts.sum(), 1.0)
        tvd = 0.5 * np.abs(probs - probs_train).sum()
        return float(1.0 - tvd)

    def _column_counts_for_df(self, df: pd.DataFrame, column: str) -> np.ndarray:
        train_dist = self.train_distributions[column]
        bucket_indices = self._column_bucket_indices_from_series(df[column], column)
        if bucket_indices.size == 0:
            return np.zeros(len(train_dist["probs"]), dtype=float)
        valid_indices = bucket_indices[bucket_indices >= 0]
        return np.bincount(valid_indices, minlength=len(train_dist["probs"])).astype(float)

    def _column_counts_from_bucket_indices(self, column: str, bucket_indices: np.ndarray) -> np.ndarray:
        train_dist = self.train_distributions[column]
        if bucket_indices.size == 0:
            return np.zeros(len(train_dist["probs"]), dtype=float)
        valid_indices = bucket_indices[bucket_indices >= 0]
        return np.bincount(valid_indices, minlength=len(train_dist["probs"])).astype(float)

    def compute_column_jsd(self, df: pd.DataFrame) -> dict[str, float]:
        if df.empty:
            return {column: 0.0 for column in self.fidelity_columns}
        bucket_indices_map = self._column_bucket_indices_for_df(df, self.fidelity_columns)
        report: dict[str, float] = {}
        for column in self.fidelity_columns:
            train_dist = self.train_distributions[column]
            counts = self._column_counts_from_bucket_indices(column, bucket_indices_map[column])
            report[column] = _js_divergence(counts, train_dist["probs"])
        return report

    def _compute_fidelity_baseline(self, d_cur_df: pd.DataFrame) -> tuple[float, dict[str, np.ndarray]]:
        count_map: dict[str, np.ndarray] = {}
        per_column_scores = []
        for column in self.fidelity_columns:
            train_dist = self.train_distributions[column]
            counts = self._column_counts_for_df(d_cur_df, column)
            count_map[column] = counts
            per_column_scores.append(self._column_similarity(counts, train_dist["probs"]))
        return float(np.mean(per_column_scores)), count_map

    def _candidate_fidelity_after(self, candidate_row: dict[str, Any], baseline_counts: dict[str, np.ndarray]) -> float:
        candidate_df = pd.DataFrame([candidate_row], columns=self.column_order)
        fidelity_after, _ = self._exact_fidelity_after_and_penalty(
            candidate_df=candidate_df,
            baseline_counts=baseline_counts,
            subset_size=int(next(iter(baseline_counts.values())).sum()) if baseline_counts else 0,
        )
        return float(fidelity_after[0]) if fidelity_after.size else 0.0

    def _frequency_penalty(
        self,
        candidate_row: dict[str, Any],
        baseline_counts: dict[str, np.ndarray],
        subset_size: int,
    ) -> float:
        if subset_size < 0:
            return 0.0
        candidate_df = pd.DataFrame([candidate_row], columns=self.column_order)
        _, penalties = self._exact_fidelity_after_and_penalty(
            candidate_df=candidate_df,
            baseline_counts=baseline_counts,
            subset_size=subset_size,
        )
        return float(penalties[0]) if penalties.size else 0.0

    def compute_dataset_fidelity(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        fidelity_score, _ = self._compute_fidelity_baseline(df.reset_index(drop=True))
        return fidelity_score

    def compute_dataset_pair_fidelity(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        pair_score, _ = self._compute_pair_fidelity_baseline(df.reset_index(drop=True))
        return pair_score

    def compute_dataset_privacy(
        self,
        df: pd.DataFrame,
        show_progress: bool = False,
        progress_desc: str = "dataset privacy",
    ) -> float:
        if df.empty:
            return 0.0
        privacy_df = self._privacy_components_frame_for_df(df)
        scores = privacy_df["privacy_score_selected"].to_numpy(dtype=float, copy=False)
        for _ in _progress(range(len(df)), total=len(df), desc=progress_desc, disable=not show_progress):
            pass
        return float(np.mean(scores)) if scores.size else 0.0

    def compute_dataset_mean_nn_distance(
        self,
        df: pd.DataFrame,
        show_progress: bool = False,
        progress_desc: str = "dataset nn distance",
    ) -> float:
        if df.empty:
            return 0.0
        privacy_df = self._privacy_components_frame_for_df(df)
        scores = privacy_df["nn_distance_train"].to_numpy(dtype=float, copy=False)
        for _ in _progress(range(len(df)), total=len(df), desc=progress_desc, disable=not show_progress):
            pass
        return float(np.mean(scores)) if scores.size else 0.0

    def compute_exact_scores(
        self,
        d_cur_df: pd.DataFrame,
        preselected_records: list[dict[str, Any]],
        show_progress: bool = False,
        progress_desc: str = "exact scoring",
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if not preselected_records:
            return [], {"baseline_fidelity": 0.0, "baseline_privacy": 0.0}

        baseline_fid_1d, baseline_counts = self._compute_fidelity_baseline(d_cur_df)
        baseline_fid_2d, baseline_pair_counts = self._compute_pair_fidelity_baseline(d_cur_df)
        baseline_priv = self.compute_dataset_privacy(d_cur_df)
        subset_size = len(d_cur_df)
        preselected_df = pd.DataFrame([record["row"] for record in preselected_records], columns=self.column_order)
        privacy_df = self._privacy_components_frame_for_df(preselected_df)
        fidelity_after_1d, fidelity_penalty_1d = self._exact_fidelity_after_and_penalty(
            candidate_df=preselected_df,
            baseline_counts=baseline_counts,
            subset_size=subset_size,
        )
        fidelity_after_2d, fidelity_penalty_2d = self._exact_pair_fidelity_after_and_penalty(
            candidate_df=preselected_df,
            baseline_pair_counts=baseline_pair_counts,
            subset_size=subset_size,
        )
        candidate_ids = np.fromiter(
            (int(record.get("candidate_id", idx)) for idx, record in enumerate(preselected_records)),
            dtype=int,
            count=len(preselected_records),
        )
        marginal_fidelity_1d = fidelity_after_1d.astype(float, copy=False) - float(baseline_fid_1d)
        marginal_fidelity_2d = fidelity_after_2d.astype(float, copy=False) - float(baseline_fid_2d)
        fidelity_penalty_1d = fidelity_penalty_1d.astype(float, copy=False)
        fidelity_penalty_2d = fidelity_penalty_2d.astype(float, copy=False)

        benefit_1d_norm = _minmax_normalize(marginal_fidelity_1d)
        benefit_2d_norm = _minmax_normalize(marginal_fidelity_2d)
        penalty_1d_norm = _minmax_normalize(fidelity_penalty_1d)
        penalty_2d_norm = _minmax_normalize(fidelity_penalty_2d)

        if self.pair_marginal_edges:
            marginal_fidelity = 0.5 * marginal_fidelity_1d + 0.5 * marginal_fidelity_2d
            fidelity_penalty = 0.5 * fidelity_penalty_1d + 0.5 * fidelity_penalty_2d
            marginal_fidelity_norm = 0.5 * benefit_1d_norm + 0.5 * benefit_2d_norm
            fidelity_penalty_norm = 0.5 * penalty_1d_norm + 0.5 * penalty_2d_norm
        else:
            marginal_fidelity = marginal_fidelity_1d
            fidelity_penalty = fidelity_penalty_1d
            marginal_fidelity_norm = benefit_1d_norm
            fidelity_penalty_norm = penalty_1d_norm

        raw_pareto_fid_1d_obj = marginal_fidelity_1d - self.lambda_penalty * fidelity_penalty_1d
        raw_pareto_fid_2d_obj = marginal_fidelity_2d - self.lambda_penalty * fidelity_penalty_2d
        pareto_fid_1d_obj = benefit_1d_norm - self.lambda_penalty * penalty_1d_norm
        pareto_fid_2d_obj = benefit_2d_norm - self.lambda_penalty * penalty_2d_norm
        raw_pareto_fid_obj = 0.5 * raw_pareto_fid_1d_obj + 0.5 * raw_pareto_fid_2d_obj
        pareto_fid_obj = 0.5 * pareto_fid_1d_obj + 0.5 * pareto_fid_2d_obj

        if show_progress:
            for _ in _progress(range(1), total=1, desc=progress_desc, disable=False):
                pass

        exact_df = pd.DataFrame(
            {
                "candidate_index": np.arange(len(preselected_records), dtype=int),
                "candidate_id": candidate_ids,
                "fidelity_after": (0.5 * fidelity_after_1d + 0.5 * fidelity_after_2d).astype(float, copy=False)
                if self.pair_marginal_edges
                else fidelity_after_1d.astype(float, copy=False),
                "fidelity_after_1d": fidelity_after_1d.astype(float, copy=False),
                "fidelity_after_2d": fidelity_after_2d.astype(float, copy=False),
                "baseline_fidelity": float(0.5 * baseline_fid_1d + 0.5 * baseline_fid_2d)
                if self.pair_marginal_edges
                else float(baseline_fid_1d),
                "baseline_fidelity_1d": float(baseline_fid_1d),
                "baseline_fidelity_2d": float(baseline_fid_2d),
                "baseline_privacy": float(baseline_priv),
                "fid_marginal": marginal_fidelity,
                "fid_marginal_1d": marginal_fidelity_1d,
                "fid_marginal_2d": marginal_fidelity_2d,
                "fid_penalty": fidelity_penalty,
                "fid_penalty_1d": fidelity_penalty_1d,
                "fid_penalty_2d": fidelity_penalty_2d,
                "fid_marginal_norm": marginal_fidelity_norm,
                "fid_marginal_1d_norm": benefit_1d_norm,
                "fid_marginal_2d_norm": benefit_2d_norm,
                "fid_penalty_norm": fidelity_penalty_norm,
                "fid_penalty_1d_norm": penalty_1d_norm,
                "fid_penalty_2d_norm": penalty_2d_norm,
                "pareto_fid_1d_obj_raw": raw_pareto_fid_1d_obj,
                "pareto_fid_2d_obj_raw": raw_pareto_fid_2d_obj,
                "pareto_fid_1d_obj": pareto_fid_1d_obj,
                "pareto_fid_2d_obj": pareto_fid_2d_obj,
                "pareto_fid_obj_raw": raw_pareto_fid_obj,
                "pareto_fid_obj": pareto_fid_obj,
                "privacy_score_v1": privacy_df["privacy_score_v1"].to_numpy(dtype=float, copy=False),
                "privacy_score_v2": privacy_df["privacy_score_v2"].to_numpy(dtype=float, copy=False),
                "privacy_score_v3": privacy_df["privacy_score_v3"].to_numpy(dtype=float, copy=False),
                "privacy_score_selected": privacy_df["privacy_score_selected"].to_numpy(dtype=float, copy=False),
                "pareto_priv_obj": privacy_df["privacy_score_selected"].to_numpy(dtype=float, copy=False),
                "nn_distance_train": privacy_df["nn_distance_train"].to_numpy(dtype=float, copy=False),
                "nn_distance_holdout": privacy_df["nn_distance_holdout"].to_numpy(dtype=float, copy=False),
                "holdout_gap": privacy_df["holdout_gap"].to_numpy(dtype=float, copy=False),
                "density_normalized_nn_distance": privacy_df["density_normalized_nn_distance"].to_numpy(
                    dtype=float, copy=False
                ),
                "local_density": privacy_df["local_density"].to_numpy(dtype=float, copy=False),
                "p_marginal_geomean": privacy_df["p_marginal_geomean"].to_numpy(dtype=float, copy=False),
                "gate_stratum": privacy_df["gate_stratum"].to_numpy(dtype=int, copy=False),
                "rarity_score": 1.0 - privacy_df["p_marginal_geomean"].to_numpy(dtype=float, copy=False),
            }
        )

        return exact_df.to_dict(orient="records"), {
            "baseline_fidelity": float(0.5 * baseline_fid_1d + 0.5 * baseline_fid_2d)
            if self.pair_marginal_edges
            else float(baseline_fid_1d),
            "baseline_fidelity_1d": float(baseline_fid_1d),
            "baseline_fidelity_2d": float(baseline_fid_2d),
            "baseline_privacy": baseline_priv,
            "lambda_penalty": self.lambda_penalty,
            "fidelity_objective_scaling": "minmax_per_batch",
            "fidelity_components": ["1d_exact", "2d_exact", "over_frequency_penalty"],
            "gamma": self.gamma,
            "privacy_version": self.privacy_version,
        }

    def _build_front_rank_map(self, fronts: list[list[int]]) -> dict[int, int]:
        return {idx: front_rank for front_rank, front in enumerate(fronts) for idx in front}

    def _front_rank_and_crowding(
        self,
        points: np.ndarray,
        fronts: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        num_records = len(points)
        front_rank = np.full(num_records, float(len(fronts) + 1), dtype=float)
        crowding = np.zeros(num_records, dtype=float)
        for rank, front in enumerate(fronts):
            if not front:
                continue
            front_indices = np.asarray(front, dtype=int)
            front_rank[front_indices] = float(rank)
            crowding[front_indices] = _crowding_distance_array(points, front_indices)
        return front_rank, crowding

    def _pareto_util_values(
        self,
        exact_records: list[dict[str, Any]],
    ) -> np.ndarray:
        return np.asarray(
            [float(record.get("pareto_util_proxy_obj", 0.0)) for record in exact_records],
            dtype=float,
        )

    def _pareto_points(
        self,
        exact_records: list[dict[str, Any]],
    ) -> np.ndarray:
        if not exact_records:
            return np.zeros((0, 4), dtype=float)
        return np.column_stack(
            [
                np.asarray([float(record.get("pareto_fid_1d_obj", 0.0)) for record in exact_records], dtype=float),
                np.asarray([float(record.get("pareto_fid_2d_obj", 0.0)) for record in exact_records], dtype=float),
                np.asarray([float(record.get("pareto_priv_obj", 0.0)) for record in exact_records], dtype=float),
                self._pareto_util_values(exact_records),
            ]
        )

    def _candidate_priority(
        self,
        points: np.ndarray,
        exact_records: list[dict[str, Any]],
        fronts: list[list[int]],
        *,
        front_rank: np.ndarray | None = None,
        crowding: np.ndarray | None = None,
    ) -> np.ndarray:
        num_records = len(exact_records)
        if front_rank is None or crowding is None:
            front_rank, crowding = self._front_rank_and_crowding(points, fronts)
        pareto_fid = np.fromiter(
            (float(record.get("pareto_fid_obj", 0.0)) for record in exact_records),
            dtype=float,
            count=num_records,
        )
        pareto_priv = np.fromiter(
            (float(record.get("pareto_priv_obj", 0.0)) for record in exact_records),
            dtype=float,
            count=num_records,
        )
        pareto_util = self._pareto_util_values(exact_records)
        return np.column_stack(
            [
                front_rank,
                -points.sum(axis=1),
                -crowding,
                -pareto_util,
                -pareto_priv,
                -pareto_fid,
            ]
        )

    def _priority_order(self, priority: np.ndarray, indices: np.ndarray | list[int]) -> np.ndarray:
        index_array = np.asarray(indices, dtype=int)
        if index_array.size == 0:
            return index_array
        local_priority = priority[index_array]
        order = np.lexsort(
            (
                index_array,
                local_priority[:, 5],
                local_priority[:, 4],
                local_priority[:, 3],
                local_priority[:, 2],
                local_priority[:, 1],
                local_priority[:, 0],
            )
        )
        return index_array[order]

    def _select_indices_by_nsga(
        self,
        points: np.ndarray,
        exact_records: list[dict[str, Any]],
        keep_k: int,
        *,
        fronts: list[list[int]] | None = None,
        front_rank: np.ndarray | None = None,
        crowding: np.ndarray | None = None,
    ) -> tuple[list[int], list[dict[str, Any]], list[list[int]]]:
        if keep_k <= 0 or len(points) == 0:
            return [], [], []
        if fronts is None:
            fronts = _non_dominated_sort(points)
        selected_indices: list[int] = []
        front_summaries: list[dict[str, Any]] = []
        pareto_fid = np.fromiter(
            (float(record.get("pareto_fid_obj", 0.0)) for record in exact_records),
            dtype=float,
            count=len(exact_records),
        )
        pareto_priv = np.fromiter(
            (float(record.get("pareto_priv_obj", 0.0)) for record in exact_records),
            dtype=float,
            count=len(exact_records),
        )
        pareto_util = self._pareto_util_values(exact_records)
        for front in fronts:
            if not front:
                continue
            if len(selected_indices) + len(front) <= keep_k:
                selected_indices.extend(front)
                front_summaries.append({"front_size": len(front), "mode": "all"})
            else:
                front_indices = np.asarray(front, dtype=int)
                if crowding is None:
                    front_crowding = _crowding_distance_array(points, front_indices)
                else:
                    front_crowding = crowding[front_indices]
                front_sum = points[front_indices].sum(axis=1)
                order = np.lexsort(
                    (
                        front_indices,
                        -pareto_util[front_indices],
                        -pareto_priv[front_indices],
                        -pareto_fid[front_indices],
                        -front_sum,
                        -front_crowding,
                    )
                )
                ordered = front_indices[order]
                need = keep_k - len(selected_indices)
                selected_indices.extend(ordered[:need].tolist())
                front_summaries.append({"front_size": len(front), "mode": "crowding_cut", "selected": need})
                break
        return selected_indices, front_summaries, fronts

    def _allocate_stratum_caps(self, budget: int) -> np.ndarray:
        probs = self.train_gate_strata_probs
        raw = probs * budget
        caps = np.floor(raw).astype(int)
        remainder = budget - int(caps.sum())
        if remainder > 0:
            frac_order = np.argsort(-(raw - caps))
            for idx in frac_order[:remainder]:
                caps[idx] += 1
        caps = np.maximum(caps, 1)
        while int(caps.sum()) > budget:
            idx = int(np.argmax(caps))
            if caps[idx] <= 1:
                break
            caps[idx] -= 1
        return caps

    def _secondary_rarity_reduce(
        self,
        selected_indices: list[int],
        exact_records: list[dict[str, Any]],
        candidate_records: list[dict[str, Any]],
        points: np.ndarray,
        fronts: list[list[int]],
        budget: int,
        *,
        priority: np.ndarray | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        if not selected_indices or budget <= 0:
            return selected_indices, {"applied": False}

        if priority is None:
            priority = self._candidate_priority(points, exact_records, fronts)
        caps = self._allocate_stratum_caps(budget)
        num_records = len(exact_records)
        gate_strata = np.fromiter(
            (int(record.get("gate_stratum", 0)) for record in exact_records),
            dtype=int,
            count=num_records,
        )
        gate_strata = np.clip(gate_strata, 0, len(caps) - 1)

        selected_array = np.asarray(selected_indices, dtype=int)
        selected_order = self._priority_order(priority, selected_array)
        kept_parts: list[np.ndarray] = []
        removed_count = 0
        for stratum in range(len(caps)):
            stratum_selected = selected_order[gate_strata[selected_order] == stratum]
            cap = int(caps[stratum])
            kept_parts.append(stratum_selected[:cap])
            removed_count += max(0, int(stratum_selected.size - cap))
        kept = np.concatenate(kept_parts) if kept_parts else np.zeros(0, dtype=int)

        need_fill = int(budget - kept.size)
        replacements = np.zeros(0, dtype=int)
        if need_fill > 0:
            current_counts = np.bincount(gate_strata[kept], minlength=len(caps)).astype(int)
            remaining_caps = np.maximum(caps.astype(int) - current_counts, 0)
            selected_mask = np.zeros(num_records, dtype=bool)
            selected_mask[selected_array] = True
            available_indices = np.flatnonzero(~selected_mask)
            if available_indices.size > 0:
                available_order = self._priority_order(priority, available_indices)
                available_parts: list[np.ndarray] = []
                for stratum in range(len(caps)):
                    cap = int(remaining_caps[stratum])
                    if cap <= 0:
                        continue
                    stratum_available = available_order[gate_strata[available_order] == stratum]
                    if stratum_available.size > 0:
                        available_parts.append(stratum_available[:cap])
                if available_parts:
                    replacements = np.concatenate(available_parts)
                    replacements = self._priority_order(priority, replacements)[:need_fill]

        final_indices = np.concatenate([kept, replacements]) if replacements.size else kept
        final_indices = self._priority_order(priority, final_indices)[:budget]
        report = {
            "applied": True,
            "budget": budget,
            "caps": caps.tolist(),
            "removed": int(removed_count),
            "replacements": int(replacements.size),
            "selected_before_secondary": len(selected_indices),
            "selected_after_secondary": int(len(final_indices)),
        }
        return final_indices.tolist(), report

    def _subset_fidelity_from_counts(self, selected_counts: dict[str, np.ndarray]) -> float:
        if not selected_counts:
            return 0.0
        scores = [
            self._column_similarity(selected_counts[column], np.asarray(self.train_distributions[column]["probs"], dtype=float))
            for column in self.fidelity_columns
        ]
        return float(np.mean(scores)) if scores else 0.0

    def _subset_pair_fidelity_from_counts(self, selected_pair_counts: list[np.ndarray]) -> float:
        if not self.pair_marginal_edges:
            return 1.0
        if not selected_pair_counts:
            return 1.0
        scores = []
        weights = []
        for edge, counts in zip(self.pair_marginal_edges, selected_pair_counts):
            scores.append(self._column_similarity(np.asarray(counts, dtype=float), np.asarray(edge["probs"], dtype=float)))
            weights.append(max(float(edge.get("mi", 0.0)), 1e-6))
        score_array = np.asarray(scores, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        return float(np.dot(score_array, weight_array) / max(float(weight_array.sum()), 1e-12))

    def _allocate_counts_from_probs(
        self,
        probs: np.ndarray,
        available_counts: np.ndarray,
        budget: int,
    ) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        available_counts = np.asarray(available_counts, dtype=int)
        if probs.size == 0 or budget <= 0 or available_counts.size == 0:
            return np.zeros_like(probs, dtype=int)
        budget = min(int(budget), int(available_counts.sum()))
        if budget <= 0:
            return np.zeros_like(probs, dtype=int)

        allowed = available_counts > 0
        weights = np.where(allowed, probs, 0.0)
        if float(weights.sum()) <= 0.0:
            weights = available_counts.astype(float)

        raw = weights / max(float(weights.sum()), 1e-12) * float(budget)
        counts = np.minimum(np.floor(raw).astype(int), available_counts)
        remaining = int(budget - counts.sum())
        while remaining > 0:
            capacity = np.maximum(available_counts - counts, 0)
            if not np.any(capacity > 0):
                break
            active_weights = np.where(capacity > 0, weights, 0.0)
            if float(active_weights.sum()) <= 0.0:
                active_weights = capacity.astype(float)
            raw_extra = active_weights / max(float(active_weights.sum()), 1e-12) * float(remaining)
            extra = np.minimum(np.floor(raw_extra).astype(int), capacity)
            extra_sum = int(extra.sum())
            if extra_sum <= 0:
                frac = raw_extra - np.floor(raw_extra)
                frac = np.where(capacity > 0, frac, -1.0)
                order = np.argsort(-frac, kind="mergesort")
                for idx in order:
                    if remaining <= 0:
                        break
                    if capacity[idx] <= 0:
                        continue
                    extra[idx] += 1
                    remaining -= 1
            else:
                counts += extra
                remaining -= extra_sum
                continue
            counts += extra
        return counts

    def _subset_alignment_from_target_counts_1d(
        self,
        selected_counts: dict[str, np.ndarray],
        target_counts: dict[str, np.ndarray],
    ) -> float:
        if not selected_counts or not target_counts:
            return 0.0
        scores = []
        for column in self.fidelity_columns:
            selected = np.asarray(selected_counts.get(column, np.zeros(0, dtype=float)), dtype=float)
            target = np.asarray(target_counts.get(column, np.zeros_like(selected)), dtype=float)
            if selected.size == 0 or target.size == 0:
                continue
            target_probs = target / max(float(target.sum()), 1.0)
            scores.append(self._column_similarity(selected, target_probs))
        return float(np.mean(scores)) if scores else 0.0

    def _subset_alignment_from_target_counts_2d(
        self,
        selected_counts: list[np.ndarray],
        target_counts: list[np.ndarray],
    ) -> float:
        if not self.pair_marginal_edges:
            return 1.0
        if not selected_counts or not target_counts:
            return 1.0
        scores = []
        weights = []
        for edge, selected, target in zip(self.pair_marginal_edges, selected_counts, target_counts):
            selected_array = np.asarray(selected, dtype=float)
            target_array = np.asarray(target, dtype=float)
            if selected_array.size == 0 or target_array.size == 0:
                continue
            target_probs = target_array / max(float(target_array.sum()), 1.0)
            scores.append(self._column_similarity(selected_array, target_probs))
            weights.append(max(float(edge.get("mi", 0.0)), 1e-6))
        if not scores:
            return 1.0
        score_array = np.asarray(scores, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        return float(np.dot(score_array, weight_array) / max(float(weight_array.sum()), 1e-12))

    def _refine_subset_to_target_counts(
        self,
        *,
        bucket_indices: dict[str, np.ndarray],
        pair_codes: list[np.ndarray],
        selected_mask: np.ndarray,
        target_counts_1d: dict[str, np.ndarray],
        target_counts_2d: list[np.ndarray],
        utility: np.ndarray,
        privacy_component: np.ndarray | None = None,
        max_rounds: int = 8,
        batch_scale: float = 0.003,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if selected_mask.size == 0 or not np.any(selected_mask):
            return selected_mask, {"applied": False, "mode": "empty"}

        selected_counts_1d, selected_counts_2d = self._subset_count_state_from_mask(
            bucket_indices=bucket_indices,
            pair_codes=pair_codes,
            selected_mask=selected_mask,
        )
        current_align_1d = self._subset_alignment_from_target_counts_1d(selected_counts_1d, target_counts_1d)
        current_align_2d = self._subset_alignment_from_target_counts_2d(selected_counts_2d, target_counts_2d)
        current_utility = float(np.asarray(utility, dtype=float)[selected_mask].mean())
        if privacy_component is None:
            privacy_values = np.zeros_like(np.asarray(utility, dtype=float))
        else:
            privacy_values = np.asarray(privacy_component, dtype=float)
        current_privacy = float(privacy_values[selected_mask].mean()) if np.any(selected_mask) else 0.0
        keep_k = int(selected_mask.sum())
        batch_size = max(4, min(128, int(round(max(keep_k, 1) * batch_scale))))
        accepted_rounds = 0
        accepted_swaps = 0

        for _ in range(max_rounds):
            selected_idx = np.flatnonzero(selected_mask)
            available_idx = np.flatnonzero(~selected_mask)
            if selected_idx.size == 0 or available_idx.size == 0:
                break

            remove_support_1d, add_support_1d = self._target_count_support_scores_1d(
                bucket_indices,
                selected_counts_1d,
                target_counts_1d,
            )
            remove_support_2d, add_support_2d = self._target_count_support_scores_2d(
                pair_codes,
                selected_counts_2d,
                target_counts_2d,
            )

            remove_priority = 0.45 * remove_support_1d + 0.35 * remove_support_2d + 0.20 * (1.0 - utility)
            add_priority = 0.40 * add_support_1d + 0.35 * add_support_2d + 0.15 * utility + 0.10 * privacy_values

            remove_order = selected_idx[np.argsort(-remove_priority[selected_idx], kind="mergesort")]
            add_order = available_idx[np.argsort(-add_priority[available_idx], kind="mergesort")]
            local_batch = min(int(batch_size), int(remove_order.size), int(add_order.size))
            accepted = False

            while local_batch > 0:
                remove_batch = remove_order[:local_batch]
                add_batch = add_order[:local_batch]
                trial_counts_1d = {column: counts.copy() for column, counts in selected_counts_1d.items()}
                trial_counts_2d = [counts.copy() for counts in selected_counts_2d]
                self._update_subset_count_state(
                    selected_counts_1d=trial_counts_1d,
                    selected_counts_2d=trial_counts_2d,
                    bucket_indices=bucket_indices,
                    pair_codes=pair_codes,
                    remove_indices=remove_batch,
                    add_indices=add_batch,
                )
                trial_align_1d = self._subset_alignment_from_target_counts_1d(trial_counts_1d, target_counts_1d)
                trial_align_2d = self._subset_alignment_from_target_counts_2d(trial_counts_2d, target_counts_2d)
                delta_utility = float((utility[add_batch].sum() - utility[remove_batch].sum()) / max(float(keep_k), 1.0))
                delta_privacy = float(
                    (privacy_values[add_batch].sum() - privacy_values[remove_batch].sum()) / max(float(keep_k), 1.0)
                )
                trial_utility = float(current_utility + delta_utility)
                trial_privacy = float(current_privacy + delta_privacy)
                improve_primary = (trial_align_1d + trial_align_2d) > (current_align_1d + current_align_2d + 1e-9)
                improve_secondary = (
                    abs((trial_align_1d + trial_align_2d) - (current_align_1d + current_align_2d)) <= 1e-9
                    and trial_privacy > current_privacy + 1e-9
                )
                improve_tertiary = (
                    abs((trial_align_1d + trial_align_2d) - (current_align_1d + current_align_2d)) <= 1e-9
                    and abs(trial_privacy - current_privacy) <= 1e-9
                    and trial_utility > current_utility + 1e-9
                )
                if improve_primary or improve_secondary or improve_tertiary:
                    selected_mask[remove_batch] = False
                    selected_mask[add_batch] = True
                    selected_counts_1d = trial_counts_1d
                    selected_counts_2d = trial_counts_2d
                    current_align_1d = trial_align_1d
                    current_align_2d = trial_align_2d
                    current_utility = trial_utility
                    current_privacy = trial_privacy
                    accepted_rounds += 1
                    accepted_swaps += int(remove_batch.size)
                    accepted = True
                    break
                local_batch //= 2

            if not accepted:
                break

        return selected_mask, {
            "applied": accepted_rounds > 0,
            "accepted_rounds": int(accepted_rounds),
            "accepted_swaps": int(accepted_swaps),
            "alignment_1d": float(current_align_1d),
            "alignment_2d": float(current_align_2d),
            "privacy_mean": float(current_privacy),
            "utility_mean": float(current_utility),
            "batch_size": int(batch_size),
        }

    def _subset_count_state_from_mask(
        self,
        bucket_indices: dict[str, np.ndarray],
        pair_codes: list[np.ndarray],
        selected_mask: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
        selected_counts_1d = {
            column: self._column_counts_from_bucket_indices(column, bucket_indices[column][selected_mask]).astype(int)
            for column in self.fidelity_columns
        }
        selected_counts_2d = [
            np.bincount(codes[selected_mask & (codes >= 0)], minlength=len(edge["probs"])).astype(int)
            for edge, codes in zip(self.pair_marginal_edges, pair_codes)
        ]
        return selected_counts_1d, selected_counts_2d

    def _update_subset_count_state(
        self,
        selected_counts_1d: dict[str, np.ndarray],
        selected_counts_2d: list[np.ndarray],
        bucket_indices: dict[str, np.ndarray],
        pair_codes: list[np.ndarray],
        remove_indices: np.ndarray,
        add_indices: np.ndarray,
    ) -> None:
        for column in self.fidelity_columns:
            remove_codes = bucket_indices[column][remove_indices]
            add_codes = bucket_indices[column][add_indices]
            self._add_code_count_delta(selected_counts_1d[column], remove_codes, -1)
            self._add_code_count_delta(selected_counts_1d[column], add_codes, 1)

        for pair_idx, codes in enumerate(pair_codes):
            remove_codes = codes[remove_indices]
            add_codes = codes[add_indices]
            self._add_code_count_delta(selected_counts_2d[pair_idx], remove_codes, -1)
            self._add_code_count_delta(selected_counts_2d[pair_idx], add_codes, 1)

    def _refine_subset_with_signature_swaps(
        self,
        *,
        selected_mask: np.ndarray,
        bucket_indices: dict[str, np.ndarray],
        privacy_component: np.ndarray,
        utility_component: np.ndarray,
        max_swaps: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.fidelity_columns or not np.any(selected_mask):
            return np.array(selected_mask, dtype=bool, copy=True), {
                "applied": False,
                "reason": "empty_or_no_fidelity_columns",
                "swaps": 0,
            }

        selected_mask = np.array(selected_mask, dtype=bool, copy=True)
        num_rows = len(selected_mask)
        if num_rows == 0:
            return selected_mask, {
                "applied": False,
                "reason": "empty_selection",
                "swaps": 0,
            }

        privacy_values = np.asarray(privacy_component, dtype=float)
        utility_values = np.asarray(utility_component, dtype=float)
        swap_score = 0.55 * privacy_values + 0.45 * utility_values
        signature_matrix = np.column_stack(
            [np.asarray(bucket_indices[column], dtype=int) for column in self.fidelity_columns]
        )
        valid_signature_mask = np.all(signature_matrix >= 0, axis=1)
        skipped_invalid = int(num_rows - np.count_nonzero(valid_signature_mask))
        valid_indices = np.flatnonzero(valid_signature_mask)
        if valid_indices.size == 0:
            return selected_mask, {
                "applied": False,
                "reason": "no_valid_signature_rows",
                "swaps": 0,
                "groups": 0,
                "groups_with_opportunity": 0,
                "score_gain_total": 0.0,
                "privacy_component_gain_total": 0.0,
                "utility_component_gain_total": 0.0,
                "max_swaps": 0,
                "skipped_invalid_rows": int(skipped_invalid),
                "score_weights": {
                    "privacy": 0.55,
                    "utility": 0.45,
                },
            }
        valid_signatures = np.ascontiguousarray(signature_matrix[valid_indices])
        signature_row_dtype = np.dtype((np.void, valid_signatures.dtype.itemsize * valid_signatures.shape[1]))
        _, inverse = np.unique(valid_signatures.view(signature_row_dtype).ravel(), return_inverse=True)
        inverse_order = np.argsort(inverse, kind="mergesort")
        inverse_sorted = inverse[inverse_order]
        group_bounds = np.r_[
            0,
            np.flatnonzero(np.diff(inverse_sorted)) + 1,
            inverse_order.size,
        ]

        max_swap_budget = max_swaps
        if max_swap_budget is None:
            max_swap_budget = max(16, min(4096, int(round(0.08 * float(selected_mask.sum())))))
        max_swap_budget = max(0, int(max_swap_budget))

        applied_swaps = 0
        privacy_gain_total = 0.0
        utility_gain_total = 0.0
        score_gain_total = 0.0
        groups_with_opportunity = 0

        for group_pos in range(len(group_bounds) - 1):
            group_indices = valid_indices[inverse_order[group_bounds[group_pos] : group_bounds[group_pos + 1]]]
            selected_indices = group_indices[selected_mask[group_indices]]
            available_indices = group_indices[~selected_mask[group_indices]]
            if selected_indices.size == 0 or available_indices.size == 0:
                continue

            selected_order = selected_indices[
                np.lexsort(
                    (
                        selected_indices,
                        utility_values[selected_indices],
                        privacy_values[selected_indices],
                        swap_score[selected_indices],
                    )
                )
            ]
            available_order = available_indices[
                np.lexsort(
                    (
                        -available_indices,
                        -utility_values[available_indices],
                        -privacy_values[available_indices],
                        -swap_score[available_indices],
                    )
                )
            ]

            local_swaps = 0
            pair_count = min(len(selected_order), len(available_order))
            for pair_idx in range(pair_count):
                if applied_swaps >= max_swap_budget:
                    break
                remove_idx = selected_order[pair_idx]
                add_idx = available_order[pair_idx]
                privacy_gain = float(privacy_values[add_idx] - privacy_values[remove_idx])
                utility_gain = float(utility_values[add_idx] - utility_values[remove_idx])
                score_gain = float(swap_score[add_idx] - swap_score[remove_idx])
                if score_gain <= 1e-12:
                    continue
                if not (
                    (privacy_gain >= 0.003 and utility_gain >= -0.002)
                    or (utility_gain >= 0.003 and privacy_gain >= -0.002)
                    or (privacy_gain >= 0.0 and utility_gain >= 0.0)
                ):
                    continue

                selected_mask[remove_idx] = False
                selected_mask[add_idx] = True
                applied_swaps += 1
                local_swaps += 1
                privacy_gain_total += privacy_gain
                utility_gain_total += utility_gain
                score_gain_total += score_gain

            if local_swaps > 0:
                groups_with_opportunity += 1
            if applied_swaps >= max_swap_budget:
                break

        return selected_mask, {
            "applied": bool(applied_swaps > 0),
            "reason": None if applied_swaps > 0 else "no_positive_same_signature_swap",
            "swaps": int(applied_swaps),
            "groups": int(len(group_bounds) - 1),
            "groups_with_opportunity": int(groups_with_opportunity),
            "score_gain_total": float(score_gain_total),
            "privacy_component_gain_total": float(privacy_gain_total),
            "utility_component_gain_total": float(utility_gain_total),
            "max_swaps": int(max_swap_budget),
            "skipped_invalid_rows": int(skipped_invalid),
            "score_weights": {
                "privacy": 0.55,
                "utility": 0.45,
            },
        }

    def _reference_target_counts_from_records(
        self,
        reference_records: list[dict[str, Any]],
    ) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
        if not reference_records:
            return {}, []
        key = self._record_cache_key(reference_records)
        cached = self._record_target_counts_cache.get(key)
        if cached is not None:
            return cached
        reference_bucket_indices, reference_pair_codes = self._bucket_pair_state_for_records(reference_records)
        target_counts_1d = {
            column: self._column_counts_from_bucket_indices(column, reference_bucket_indices[column]).astype(int)
            for column in self.fidelity_columns
        }
        target_counts_2d = [
            np.bincount(codes[codes >= 0], minlength=len(edge["probs"])).astype(int)
            for edge, codes in zip(self.pair_marginal_edges, reference_pair_codes)
        ]
        if len(self._record_target_counts_cache) >= 6:
            self._record_target_counts_cache.pop(next(iter(self._record_target_counts_cache)))
        self._record_target_counts_cache[key] = (target_counts_1d, target_counts_2d)
        return target_counts_1d, target_counts_2d

    def _selection_objective_components(
        self,
        exact_records: list[dict[str, Any]],
        *,
        mode: str,
        fidelity_1d_weight: float = 0.25,
        fidelity_2d_weight: float = 0.25,
        privacy_weight: float = 0.5,
        utility_weight: float = 0.0,
    ) -> dict[str, Any]:
        num_records = len(exact_records)
        if num_records == 0:
            empty = np.zeros(0, dtype=float)
            return {
                "utility": empty,
                "fidelity_1d_component": empty,
                "fidelity_2d_component": empty,
                "privacy_component": empty,
                "utility_component": empty,
                "mode": mode,
            }

        if mode == "scalar_naive":
            fidelity_raw = np.asarray([float(record.get("fid_marginal", 0.0)) for record in exact_records], dtype=float)
            privacy_raw = np.asarray([float(record.get("privacy_score_v1", 0.0)) for record in exact_records], dtype=float)
            fidelity_component = _minmax_normalize(fidelity_raw)
            privacy_component = _minmax_normalize(privacy_raw)
            utility_component = np.zeros_like(fidelity_component, dtype=float)
            utility = (fidelity_1d_weight + fidelity_2d_weight) * fidelity_component + privacy_weight * privacy_component
            return {
                "utility": utility,
                "fidelity_1d_component": fidelity_component,
                "fidelity_2d_component": fidelity_component,
                "privacy_component": privacy_component,
                "utility_component": utility_component,
                "mode": mode,
            }

        fidelity_1d_raw = np.asarray([float(record.get("pareto_fid_1d_obj", 0.0)) for record in exact_records], dtype=float)
        fidelity_2d_raw = np.asarray([float(record.get("pareto_fid_2d_obj", 0.0)) for record in exact_records], dtype=float)
        privacy_raw = np.asarray([float(record.get("pareto_priv_obj", 0.0)) for record in exact_records], dtype=float)
        utility_raw = self._pareto_util_values(exact_records)
        fidelity_1d_component = _minmax_normalize(fidelity_1d_raw)
        fidelity_2d_component = _minmax_normalize(fidelity_2d_raw)
        privacy_component = _minmax_normalize(privacy_raw)
        utility_component = _minmax_normalize(utility_raw)
        weights = np.asarray([fidelity_1d_weight, fidelity_2d_weight, privacy_weight, utility_weight], dtype=float)
        weights = np.where(weights <= 0.0, 1e-6, weights)

        if mode == "scalar_matched":
            utility = (
                weights[0] * fidelity_1d_component
                + weights[1] * fidelity_2d_component
                + weights[2] * privacy_component
                + weights[3] * utility_component
            )
            return {
                "utility": utility,
                "fidelity_1d_component": fidelity_1d_component,
                "fidelity_2d_component": fidelity_2d_component,
                "privacy_component": privacy_component,
                "utility_component": utility_component,
                "mode": mode,
            }

        points = np.column_stack([fidelity_1d_raw, fidelity_2d_raw, privacy_raw, utility_raw])
        if mode == "chebyshev":
            normalized_points = np.column_stack(
                [fidelity_1d_component, fidelity_2d_component, privacy_component, utility_component]
            )
            reference = normalized_points.max(axis=0)
            chebyshev_scores = np.max(weights[None, :] * (reference[None, :] - normalized_points), axis=1)
            utility = 1.0 - _minmax_normalize(chebyshev_scores)
            utility = 0.85 * utility + 0.10 * privacy_component + 0.05 * utility_component
            return {
                "utility": utility,
                "fidelity_1d_component": fidelity_1d_component,
                "fidelity_2d_component": fidelity_2d_component,
                "privacy_component": privacy_component,
                "utility_component": utility_component,
                "chebyshev_scores": chebyshev_scores,
                "mode": mode,
            }

        fronts = _non_dominated_sort(points)
        front_rank, crowding = self._front_rank_and_crowding(points, fronts)

        front_component = 1.0 - _minmax_normalize(front_rank)
        crowding_component = np.zeros_like(crowding)
        finite_mask = np.isfinite(crowding)
        if np.any(finite_mask):
            crowding_component[finite_mask] = _minmax_normalize(crowding[finite_mask])
            crowding_component[~finite_mask] = 1.0
        else:
            crowding_component.fill(1.0)

        utility = (
            0.22 * privacy_component
            + 0.18 * fidelity_1d_component
            + 0.18 * fidelity_2d_component
            + 0.22 * utility_component
            + 0.25 * front_component
            + 0.05 * crowding_component
        )
        return {
            "utility": utility,
            "fidelity_1d_component": fidelity_1d_component,
            "fidelity_2d_component": fidelity_2d_component,
            "privacy_component": privacy_component,
            "utility_component": utility_component,
            "front_component": front_component,
            "crowding_component": crowding_component,
            "fronts": fronts,
            "mode": mode,
        }

    def _construct_constrained_keep_subset(
        self,
        preselected_records: list[dict[str, Any]],
        exact_records: list[dict[str, Any]],
        keep_k: int,
        *,
        mode: str,
        floor_reference: dict[str, Any] | None,
        constraint_reference_records: list[dict[str, Any]] | None,
        fidelity_1d_weight: float = 0.25,
        fidelity_2d_weight: float = 0.25,
        privacy_weight: float = 0.5,
        utility_weight: float = 0.0,
        floor_mode: str = "hard",
        soft_fidelity_floor_eps: float | None = None,
        soft_trend_floor_eps: float | None = None,
        soft_privacy_floor_eps: float = 0.005,
        soft_utility_floor_eps: float = 0.005,
        soft_min_score_delta: float = 0.0,
    ) -> tuple[list[int], dict[str, Any]]:
        if not preselected_records or not exact_records or keep_k <= 0:
            return [], {"applied": False, "mode": "empty"}
        if floor_reference is None:
            return [], {"applied": False, "mode": "disabled_no_floor_reference"}

        keep_k = min(int(keep_k), len(preselected_records))
        bucket_indices, pair_codes = self._bucket_pair_state_for_records(preselected_records)
        shape_constraint_enabled = bool(constraint_reference_records) and bool(
            floor_reference.get("enforce_reference_shape", False)
        )
        if shape_constraint_enabled:
            target_counts_1d, target_counts_2d = self._reference_target_counts_from_records(constraint_reference_records)
        else:
            target_counts_1d, target_counts_2d = {}, []
        objective = self._selection_objective_components(
            exact_records,
            mode=mode,
            fidelity_1d_weight=fidelity_1d_weight,
            fidelity_2d_weight=fidelity_2d_weight,
            privacy_weight=privacy_weight,
            utility_weight=utility_weight,
        )
        utility = np.asarray(objective["utility"], dtype=float)
        fidelity_1d_component = np.asarray(objective["fidelity_1d_component"], dtype=float)
        fidelity_2d_component = np.asarray(objective["fidelity_2d_component"], dtype=float)
        privacy_component = np.asarray(objective["privacy_component"], dtype=float)
        utility_component = np.asarray(objective.get("utility_component", np.zeros(len(exact_records), dtype=float)), dtype=float)

        candidate_ids = np.fromiter(
            (int(record.get("candidate_id", idx)) for idx, record in enumerate(preselected_records)),
            dtype=int,
            count=len(preselected_records),
        )
        reference_mask = np.zeros(len(preselected_records), dtype=bool)
        if constraint_reference_records:
            reference_ids = np.fromiter(
                (int(record.get("candidate_id", idx)) for idx, record in enumerate(constraint_reference_records)),
                dtype=int,
                count=len(constraint_reference_records),
            )
            reference_mask = np.isin(candidate_ids, reference_ids)
        selected_mask = reference_mask.astype(bool, copy=True)
        initial_reference_rows = int(selected_mask.sum())
        if initial_reference_rows > keep_k:
            selected_indices = np.flatnonzero(selected_mask)
            remove_count = initial_reference_rows - keep_k
            order = selected_indices[np.argsort(utility[selected_indices], kind="mergesort")]
            selected_mask[order[:remove_count]] = False
        elif 0 < initial_reference_rows < keep_k:
            remaining_indices = np.flatnonzero(~selected_mask)
            if remaining_indices.size > 0:
                order = remaining_indices[np.argsort(-utility[remaining_indices], kind="mergesort")]
                fill_count = min(int(keep_k - initial_reference_rows), int(order.size))
                selected_mask[order[:fill_count]] = True

        selected_counts_1d, selected_counts_2d = self._subset_count_state_from_mask(
            bucket_indices,
            pair_codes,
            selected_mask,
        )
        target_fid_1d = max(
            0.0,
            float(floor_reference.get("fidelity_1d", 0.0)) - self.final_fidelity_floor_eps,
        )
        target_fid_2d = max(
            0.0,
            float(floor_reference.get("fidelity_2d", 0.0)) - self.final_trend_floor_eps,
        )
        normalized_floor_mode = str(floor_mode or "hard").strip().lower()
        if normalized_floor_mode not in {"hard", "soft"}:
            normalized_floor_mode = "hard"
        use_soft_floor = normalized_floor_mode == "soft"
        soft_target_fid_1d = max(
            0.0,
            float(floor_reference.get("fidelity_1d", 0.0))
            - float(self.final_fidelity_floor_eps if soft_fidelity_floor_eps is None else soft_fidelity_floor_eps),
        )
        soft_target_fid_2d = max(
            0.0,
            float(floor_reference.get("fidelity_2d", 0.0))
            - float(self.final_trend_floor_eps if soft_trend_floor_eps is None else soft_trend_floor_eps),
        )
        privacy_raw = np.asarray([float(record.get("pareto_priv_obj", 0.0)) for record in exact_records], dtype=float)
        zero_support = np.zeros(len(preselected_records), dtype=float)

        def _mask_counts(mask: np.ndarray) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
            return self._subset_count_state_from_mask(
                bucket_indices=bucket_indices,
                pair_codes=pair_codes,
                selected_mask=mask,
            )

        def _score_from_components(fid_1d: float, fid_2d: float, privacy_mean: float, utility_mean: float) -> float:
            return float(0.18 * fid_1d + 0.18 * fid_2d + 0.34 * privacy_mean + 0.30 * utility_mean)

        def _mask_stats(mask: np.ndarray) -> dict[str, float]:
            if not np.any(mask):
                return {
                    "fid_1d": 0.0,
                    "fid_2d": 0.0,
                    "score": float("-inf"),
                    "objective_mean": 0.0,
                    "privacy_mean": 0.0,
                    "privacy_raw_mean": 0.0,
                    "utility_mean": 0.0,
                }
            counts_1d, counts_2d = _mask_counts(mask)
            fid_1d = self._subset_fidelity_from_counts(counts_1d)
            fid_2d = self._subset_pair_fidelity_from_counts(counts_2d)
            privacy_mean = float(privacy_component[mask].mean())
            privacy_raw_mean = float(privacy_raw[mask].mean())
            util_mean = float(utility_component[mask].mean())
            objective_mean = float(utility[mask].mean())
            score = _score_from_components(fid_1d, fid_2d, privacy_mean, util_mean)
            return {
                "fid_1d": float(fid_1d),
                "fid_2d": float(fid_2d),
                "score": float(score),
                "objective_mean": objective_mean,
                "privacy_mean": privacy_mean,
                "privacy_raw_mean": privacy_raw_mean,
                "utility_mean": util_mean,
            }

        reference_mask = selected_mask.copy()
        reference_available = bool(np.any(reference_mask))
        reference_stats = _mask_stats(reference_mask)
        reference_privacy_component_mean = float(privacy_component[reference_mask].mean()) if np.any(reference_mask) else 0.0
        reference_privacy_raw_mean = float(privacy_raw[reference_mask].mean()) if np.any(reference_mask) else 0.0

        def _soft_component_deltas(stats: dict[str, float]) -> dict[str, float]:
            return {
                "fid_1d": float(stats["fid_1d"] - reference_stats["fid_1d"]),
                "fid_2d": float(stats["fid_2d"] - reference_stats["fid_2d"]),
                "privacy": float(stats["privacy_mean"] - reference_stats["privacy_mean"]),
                "utility": float(stats["utility_mean"] - reference_stats["utility_mean"]),
                "score": float(stats["score"] - reference_stats["score"]),
            }

        def _hard_floor_satisfied(stats: dict[str, float]) -> bool:
            return bool(stats["fid_1d"] >= target_fid_1d and stats["fid_2d"] >= target_fid_2d)

        def _soft_floor_satisfied(stats: dict[str, float]) -> bool:
            deltas = _soft_component_deltas(stats)
            return bool(
                stats["fid_1d"] >= soft_target_fid_1d
                and stats["fid_2d"] >= soft_target_fid_2d
                and deltas["privacy"] >= -float(soft_privacy_floor_eps)
                and deltas["utility"] >= -float(soft_utility_floor_eps)
            )

        def _candidate_satisfied(stats: dict[str, float]) -> bool:
            if use_soft_floor:
                return _soft_floor_satisfied(stats)
            return _hard_floor_satisfied(stats)

        def _candidate_tradeoff_acceptable(stats: dict[str, float], *, allow_reference: bool = False) -> bool:
            deltas = _soft_component_deltas(stats)
            if allow_reference and abs(deltas["score"]) <= 1e-12:
                return True
            return bool(deltas["score"] > float(soft_min_score_delta))

        init_masks: list[tuple[str, np.ndarray]] = []
        if int(reference_mask.sum()) == keep_k and reference_available:
            init_masks.append(("reference_anchor", reference_mask.copy()))
        all_indices = np.arange(len(preselected_records), dtype=int)
        composite_score = (
            0.30 * privacy_component
            + 0.30 * utility_component
            + 0.20 * fidelity_1d_component
            + 0.20 * fidelity_2d_component
        )
        if all_indices.size > 0:
            order = np.argsort(-composite_score, kind="mergesort")
            init_mask = np.zeros(len(preselected_records), dtype=bool)
            init_mask[order[:keep_k]] = True
            init_masks.append(("global_4d_composite", init_mask))

        fidelity_safe_score = 0.50 * fidelity_1d_component + 0.50 * fidelity_2d_component
        band_target = min(len(preselected_records), max(keep_k, int(round(1.75 * keep_k))))
        if band_target > 0:
            band_order = np.argsort(-fidelity_safe_score, kind="mergesort")[:band_target]
            band_select_score = (
                0.36 * privacy_component[band_order]
                + 0.34 * utility_component[band_order]
                + 0.15 * fidelity_1d_component[band_order]
                + 0.15 * fidelity_2d_component[band_order]
            )
            chosen = band_order[np.argsort(-band_select_score, kind="mergesort")[:keep_k]]
            init_mask = np.zeros(len(preselected_records), dtype=bool)
            init_mask[chosen] = True
            init_masks.append(("fidelity_safe_4d_band", init_mask))

        unique_init_masks: list[tuple[str, np.ndarray]] = []
        seen_init_keys: set[tuple[int, ...]] = set()
        for init_name, init_mask in init_masks:
            if int(init_mask.sum()) != keep_k:
                continue
            init_key = tuple(np.flatnonzero(init_mask).tolist())
            if init_key in seen_init_keys:
                continue
            seen_init_keys.add(init_key)
            unique_init_masks.append((init_name, init_mask))
        init_masks = unique_init_masks

        def _repair_mask_to_floor(init_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
            selected_mask = init_mask.astype(bool, copy=True)
            selected_counts_1d, selected_counts_2d = _mask_counts(selected_mask)
            current_fid_1d = self._subset_fidelity_from_counts(selected_counts_1d)
            current_fid_2d = self._subset_pair_fidelity_from_counts(selected_counts_2d)
            current_privacy = float(privacy_component[selected_mask].mean()) if np.any(selected_mask) else 0.0
            current_utility = float(utility_component[selected_mask].mean()) if np.any(selected_mask) else 0.0
            current_quality = 0.25 * current_fid_1d + 0.25 * current_fid_2d + 0.25 * current_privacy + 0.25 * current_utility

            batch_size = max(32, min(512, int(round(0.02 * keep_k))))
            rounds_applied = 0
            max_rounds = 6

            for _ in range(max_rounds):
                if current_fid_1d >= target_fid_1d and current_fid_2d >= target_fid_2d:
                    break

                remove_1d, add_1d = self._train_prob_support_scores_1d(
                    bucket_indices,
                    selected_counts_1d,
                    keep_k,
                )
                remove_2d, add_2d = self._train_prob_support_scores_2d(
                    pair_codes,
                    selected_counts_2d,
                    keep_k,
                )

                remove_score = 0.6 * remove_1d + 0.4 * remove_2d
                add_score = 0.6 * add_1d + 0.4 * add_2d

                remove_priority = (
                    remove_score
                    - 0.10 * utility_component
                    - 0.08 * privacy_component
                    - 0.06 * fidelity_1d_component
                    - 0.06 * fidelity_2d_component
                )
                add_priority = (
                    add_score
                    + 0.12 * utility_component
                    + 0.08 * privacy_component
                    + 0.05 * fidelity_1d_component
                    + 0.05 * fidelity_2d_component
                )

                selected_idx = np.flatnonzero(selected_mask)
                available_idx = np.flatnonzero(~selected_mask)
                if selected_idx.size == 0 or available_idx.size == 0:
                    break

                remove_order = selected_idx[np.argsort(-remove_priority[selected_idx], kind="mergesort")]
                add_order = available_idx[np.argsort(-add_priority[available_idx], kind="mergesort")]
                local_batch = min(int(batch_size), int(remove_order.size), int(add_order.size))
                accepted = False

                while local_batch > 0:
                    remove_batch = remove_order[:local_batch]
                    add_batch = add_order[:local_batch]
                    beneficial = add_priority[add_batch] > (remove_priority[remove_batch] + 1e-9)
                    if not np.any(beneficial):
                        local_batch //= 2
                        continue
                    remove_batch = remove_batch[beneficial]
                    add_batch = add_batch[beneficial]
                    if remove_batch.size == 0 or add_batch.size == 0:
                        local_batch //= 2
                        continue

                    trial_counts_1d = {column: counts.copy() for column, counts in selected_counts_1d.items()}
                    trial_counts_2d = [counts.copy() for counts in selected_counts_2d]
                    self._update_subset_count_state(
                        selected_counts_1d=trial_counts_1d,
                        selected_counts_2d=trial_counts_2d,
                        bucket_indices=bucket_indices,
                        pair_codes=pair_codes,
                        remove_indices=remove_batch,
                        add_indices=add_batch,
                    )

                    trial_fid_1d = self._subset_fidelity_from_counts(trial_counts_1d)
                    trial_fid_2d = self._subset_pair_fidelity_from_counts(trial_counts_2d)
                    delta_privacy = float(
                        (privacy_component[add_batch].sum() - privacy_component[remove_batch].sum()) / max(float(keep_k), 1.0)
                    )
                    delta_utility = float(
                        (utility_component[add_batch].sum() - utility_component[remove_batch].sum()) / max(float(keep_k), 1.0)
                    )
                    trial_privacy = float(current_privacy + delta_privacy)
                    trial_utility = float(current_utility + delta_utility)
                    trial_quality = (
                        0.25 * trial_fid_1d
                        + 0.25 * trial_fid_2d
                        + 0.25 * trial_privacy
                        + 0.25 * trial_utility
                    )
                    alignment_gain = (trial_fid_1d + trial_fid_2d) - (current_fid_1d + current_fid_2d)
                    quality_delta = float(trial_quality - current_quality)
                    if (
                        alignment_gain > 1e-9
                        and (quality_delta >= -0.002 or (trial_fid_1d >= target_fid_1d and trial_fid_2d >= target_fid_2d))
                    ):
                        selected_mask[remove_batch] = False
                        selected_mask[add_batch] = True
                        selected_counts_1d = trial_counts_1d
                        selected_counts_2d = trial_counts_2d
                        current_fid_1d = trial_fid_1d
                        current_fid_2d = trial_fid_2d
                        current_privacy = trial_privacy
                        current_utility = trial_utility
                        current_quality = trial_quality
                        rounds_applied += 1
                        accepted = True
                        break
                    local_batch //= 2

                if not accepted:
                    break

            final_stats = _mask_stats(selected_mask)
            return selected_mask, {
                "applied": bool(rounds_applied > 0),
                "mode": "floor_only_repair",
                "target_fid_1d": target_fid_1d,
                "target_fid_2d": target_fid_2d,
                "current_fid_1d": float(final_stats["fid_1d"]),
                "current_fid_2d": float(final_stats["fid_2d"]),
                "current_privacy_mean": float(final_stats["privacy_mean"]),
                "current_utility_mean": float(final_stats["utility_mean"]),
                "current_quality": float(
                    0.25 * final_stats["fid_1d"]
                    + 0.25 * final_stats["fid_2d"]
                    + 0.25 * final_stats["privacy_mean"]
                    + 0.25 * final_stats["utility_mean"]
                ),
                "rounds": int(rounds_applied),
                "satisfied": bool(final_stats["fid_1d"] >= target_fid_1d and final_stats["fid_2d"] >= target_fid_2d),
            }, final_stats

        def _local_refine(init_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
            selected_mask = init_mask.astype(bool, copy=True)
            selected_counts_1d, selected_counts_2d = _mask_counts(selected_mask)
            current_fid_1d = self._subset_fidelity_from_counts(selected_counts_1d)
            current_fid_2d = self._subset_pair_fidelity_from_counts(selected_counts_2d)
            current_privacy_mean = float(privacy_component[selected_mask].mean()) if np.any(selected_mask) else 0.0
            current_utility_mean = float(utility_component[selected_mask].mean()) if np.any(selected_mask) else 0.0
            current_objective_mean = float(utility[selected_mask].mean()) if np.any(selected_mask) else 0.0
            current_score = _score_from_components(
                current_fid_1d,
                current_fid_2d,
                current_privacy_mean,
                current_utility_mean,
            )
            accepted_rounds = 0
            accepted_swaps = 0
            max_rounds = 10
            base_batch_size = max(8, min(128, int(round(0.0025 * keep_k))))

            for _ in range(max_rounds):
                selected_idx = np.flatnonzero(selected_mask)
                available_idx = np.flatnonzero(~selected_mask)
                if selected_idx.size == 0 or available_idx.size == 0:
                    break

                if shape_constraint_enabled:
                    remove_support_1d, add_support_1d = self._target_count_support_scores_1d(
                        bucket_indices,
                        selected_counts_1d,
                        target_counts_1d,
                    )
                    remove_support_2d, add_support_2d = self._target_count_support_scores_2d(
                        pair_codes,
                        selected_counts_2d,
                        target_counts_2d,
                    )
                else:
                    remove_support_1d = zero_support
                    add_support_1d = zero_support
                    remove_support_2d = zero_support
                    add_support_2d = zero_support

                remove_priority = (
                    0.32 * (1.0 - composite_score)
                    + 0.18 * (1.0 - utility_component)
                    + 0.14 * (1.0 - privacy_component)
                    + 0.18 * remove_support_1d
                    + 0.18 * remove_support_2d
                )
                add_priority = (
                    0.32 * composite_score
                    + 0.20 * utility_component
                    + 0.20 * privacy_component
                    + 0.14 * add_support_1d
                    + 0.14 * add_support_2d
                )

                remove_order = selected_idx[np.argsort(-remove_priority[selected_idx], kind="mergesort")]
                add_order = available_idx[np.argsort(-add_priority[available_idx], kind="mergesort")]
                local_batch = min(int(base_batch_size), int(remove_order.size), int(add_order.size))
                accepted = False

                while local_batch > 0:
                    remove_batch = remove_order[:local_batch]
                    add_batch = add_order[:local_batch]
                    beneficial = add_priority[add_batch] > (remove_priority[remove_batch] + 1e-12)
                    if not np.any(beneficial):
                        local_batch //= 2
                        continue
                    remove_batch = remove_batch[beneficial]
                    add_batch = add_batch[beneficial]
                    if remove_batch.size == 0 or add_batch.size == 0:
                        local_batch //= 2
                        continue

                    trial_counts_1d = {column: counts.copy() for column, counts in selected_counts_1d.items()}
                    trial_counts_2d = [counts.copy() for counts in selected_counts_2d]
                    self._update_subset_count_state(
                        trial_counts_1d,
                        trial_counts_2d,
                        bucket_indices,
                        pair_codes,
                        remove_batch,
                        add_batch,
                    )
                    trial_fid_1d = self._subset_fidelity_from_counts(trial_counts_1d)
                    trial_fid_2d = self._subset_pair_fidelity_from_counts(trial_counts_2d)
                    delta_privacy = float(
                        (privacy_component[add_batch].sum() - privacy_component[remove_batch].sum()) / max(float(keep_k), 1.0)
                    )
                    delta_utility_mean = float(
                        (utility_component[add_batch].sum() - utility_component[remove_batch].sum()) / max(float(keep_k), 1.0)
                    )
                    delta_objective_mean = float(
                        (utility[add_batch].sum() - utility[remove_batch].sum()) / max(float(keep_k), 1.0)
                    )
                    trial_privacy_mean = float(current_privacy_mean + delta_privacy)
                    trial_utility_mean = float(current_utility_mean + delta_utility_mean)
                    trial_objective_mean = float(current_objective_mean + delta_objective_mean)
                    trial_score = _score_from_components(
                        trial_fid_1d,
                        trial_fid_2d,
                        trial_privacy_mean,
                        trial_utility_mean,
                    )
                    trial_stats = {
                        "fid_1d": float(trial_fid_1d),
                        "fid_2d": float(trial_fid_2d),
                        "score": float(trial_score),
                        "objective_mean": float(trial_objective_mean),
                        "privacy_mean": float(trial_privacy_mean),
                        "privacy_raw_mean": 0.0,
                        "utility_mean": float(trial_utility_mean),
                    }
                    if (
                        (
                            _candidate_satisfied(trial_stats)
                            if use_soft_floor
                            else (trial_fid_1d >= target_fid_1d and trial_fid_2d >= target_fid_2d)
                        )
                        and trial_score > (current_score + 1e-12)
                    ):
                        selected_mask[remove_batch] = False
                        selected_mask[add_batch] = True
                        selected_counts_1d = trial_counts_1d
                        selected_counts_2d = trial_counts_2d
                        current_fid_1d = trial_fid_1d
                        current_fid_2d = trial_fid_2d
                        current_privacy_mean = trial_privacy_mean
                        current_utility_mean = trial_utility_mean
                        current_objective_mean = trial_objective_mean
                        current_score = float(trial_score)
                        accepted_rounds += 1
                        accepted_swaps += int(remove_batch.size)
                        accepted = True
                        break
                    local_batch //= 2

                if not accepted:
                    break

            return selected_mask, {
                "accepted_rounds": int(accepted_rounds),
                "accepted_swaps": int(accepted_swaps),
                "batch_size": int(base_batch_size),
                "current_fid_1d": float(current_fid_1d),
                "current_fid_2d": float(current_fid_2d),
                "score": float(current_score),
            }

        candidate_solutions: list[dict[str, Any]] = []
        for init_name, init_mask in init_masks:
            refined_mask, refine_report = _local_refine(init_mask)
            stats = _mask_stats(refined_mask)
            satisfied = _candidate_satisfied(stats)
            candidate_solutions.append(
                {
                    "init": init_name,
                    "mask": refined_mask,
                    "stats": stats,
                    "refine_report": refine_report,
                    "repair_report": {"applied": False, "mode": "not_needed" if satisfied else "not_applied"},
                    "satisfied": satisfied,
                    "hard_floor_satisfied": _hard_floor_satisfied(stats),
                    "soft_floor_satisfied": _soft_floor_satisfied(stats),
                    "tradeoff_acceptable": _candidate_tradeoff_acceptable(
                        stats,
                        allow_reference=bool(init_name == "reference_anchor"),
                    ),
                    "component_deltas": _soft_component_deltas(stats),
                }
            )
            if not satisfied and not shape_constraint_enabled:
                repaired_mask, repair_report, repaired_stats = _repair_mask_to_floor(refined_mask)
                repaired_satisfied = _candidate_satisfied(repaired_stats)
                if bool(repair_report.get("applied", False)) or repaired_satisfied:
                    candidate_solutions.append(
                        {
                            "init": f"{init_name}_floor_repair",
                            "mask": repaired_mask,
                            "stats": repaired_stats,
                            "refine_report": refine_report,
                            "repair_report": repair_report,
                            "satisfied": repaired_satisfied,
                            "hard_floor_satisfied": _hard_floor_satisfied(repaired_stats),
                            "soft_floor_satisfied": _soft_floor_satisfied(repaired_stats),
                            "tradeoff_acceptable": _candidate_tradeoff_acceptable(repaired_stats),
                            "component_deltas": _soft_component_deltas(repaired_stats),
                        }
                    )

        feasible = [
            item
            for item in candidate_solutions
            if bool(item["satisfied"]) and (not use_soft_floor or bool(item["tradeoff_acceptable"]))
        ]
        if feasible:
            best_solution = max(feasible, key=lambda item: (item["stats"]["score"], item["stats"]["privacy_mean"], item["stats"]["utility_mean"]))
        else:
            best_solution = max(candidate_solutions, key=lambda item: (item["stats"]["fid_1d"] + item["stats"]["fid_2d"], item["stats"]["score"]))

        selected_mask = np.asarray(best_solution["mask"], dtype=bool)
        pre_signature_mask = selected_mask.copy()
        selected_mask, signature_refine_report = self._refine_subset_with_signature_swaps(
            selected_mask=selected_mask,
            bucket_indices=bucket_indices,
            privacy_component=privacy_component,
            utility_component=utility_component,
        )
        selected_counts_1d, selected_counts_2d = _mask_counts(selected_mask)
        current_fid_1d = self._subset_fidelity_from_counts(selected_counts_1d)
        current_fid_2d = self._subset_pair_fidelity_from_counts(selected_counts_2d)
        current_stats = _mask_stats(selected_mask)
        if use_soft_floor and not (
            _candidate_satisfied(current_stats)
            and _candidate_tradeoff_acceptable(
                current_stats,
                allow_reference=bool(str(best_solution["init"]) == "reference_anchor"),
            )
        ):
            previous_signature_refine_report = signature_refine_report
            selected_mask = pre_signature_mask
            selected_counts_1d, selected_counts_2d = _mask_counts(selected_mask)
            current_fid_1d = self._subset_fidelity_from_counts(selected_counts_1d)
            current_fid_2d = self._subset_pair_fidelity_from_counts(selected_counts_2d)
            current_stats = _mask_stats(selected_mask)
            signature_refine_report = {
                "applied": False,
                "reason": "reverted_by_soft_floor_tradeoff_gate",
                "swaps": 0,
                "previous_report": previous_signature_refine_report,
            }

        privacy_delta = float(current_stats["privacy_mean"] - reference_stats["privacy_mean"])
        utility_delta = float(current_stats["utility_mean"] - reference_stats["utility_mean"])
        score_delta = float(current_stats["score"] - reference_stats["score"])
        benefit_gate_enabled = bool(reference_available and mode in {"pareto", "chebyshev", "scalar_matched"})
        if use_soft_floor:
            benefit_satisfied = (
                not benefit_gate_enabled
                or (
                    _candidate_satisfied(current_stats)
                    and _candidate_tradeoff_acceptable(
                        current_stats,
                        allow_reference=bool(str(best_solution["init"]) == "reference_anchor"),
                    )
                )
            )
        else:
            benefit_satisfied = (
                not benefit_gate_enabled
                or (
                    current_fid_1d >= target_fid_1d
                    and current_fid_2d >= target_fid_2d
                    and score_delta > 0.0
                    and (
                        (privacy_delta >= 0.005 and utility_delta >= -0.002)
                        or (utility_delta >= 0.003 and privacy_delta >= -0.002)
                        or (privacy_delta >= 0.003 and utility_delta >= 0.003)
                    )
                )
            )
        reverted_to_reference = False
        if not benefit_satisfied and mode != "pareto" and shape_constraint_enabled and reference_available:
            selected_mask = reference_mask.copy()
            current_stats = reference_stats
            current_fid_1d = reference_stats["fid_1d"]
            current_fid_2d = reference_stats["fid_2d"]
            privacy_delta = 0.0
            utility_delta = 0.0
            score_delta = 0.0
            signature_refine_report = {
                "applied": False,
                "reason": "reverted_to_reference",
                "swaps": 0,
            }
            reverted_to_reference = True

        final_privacy_component_mean = float(privacy_component[selected_mask].mean()) if np.any(selected_mask) else 0.0
        final_privacy_raw_mean = float(privacy_raw[selected_mask].mean()) if np.any(selected_mask) else 0.0

        final_indices = np.sort(np.flatnonzero(selected_mask).astype(int, copy=False)).tolist()
        return final_indices, {
            "applied": True,
            "mode": "constrained_subset_construction",
            "selection_mode": mode,
            "reference_name": floor_reference.get("name", "floor_reference"),
            "reference_fid_1d": float(floor_reference.get("fidelity_1d", 0.0)),
            "reference_fid_2d": float(floor_reference.get("fidelity_2d", 0.0)),
            "target_fid_1d": target_fid_1d,
            "target_fid_2d": target_fid_2d,
            "floor_mode": normalized_floor_mode,
            "soft_target_fid_1d": soft_target_fid_1d,
            "soft_target_fid_2d": soft_target_fid_2d,
            "soft_privacy_floor_eps": float(soft_privacy_floor_eps),
            "soft_utility_floor_eps": float(soft_utility_floor_eps),
            "soft_min_score_delta": float(soft_min_score_delta),
            "current_fid_1d": current_fid_1d,
            "current_fid_2d": current_fid_2d,
            "satisfied": _candidate_satisfied(current_stats),
            "hard_floor_satisfied": _hard_floor_satisfied(current_stats),
            "soft_floor_satisfied": _soft_floor_satisfied(current_stats),
            "tradeoff_acceptable": _candidate_tradeoff_acceptable(
                current_stats,
                allow_reference=bool(str(best_solution["init"]) == "reference_anchor"),
            ),
            "component_deltas": _soft_component_deltas(current_stats),
            "shape_constraint_enabled": bool(shape_constraint_enabled),
            "constraint_reference_rows": int(len(constraint_reference_records or [])),
            "reference_anchor_available": bool(reference_available),
            "initial_reference_rows": initial_reference_rows,
            "init_mode": str(best_solution["init"]),
            "accepted_rounds": int(best_solution["refine_report"].get("accepted_rounds", 0)),
            "accepted_swaps": int(best_solution["refine_report"].get("accepted_swaps", 0)),
            "batch_size": int(best_solution["refine_report"].get("batch_size", 0)),
            "signature_refine": signature_refine_report,
            "repair_report": dict(best_solution.get("repair_report", {})),
            "utility_mean": float(current_stats["objective_mean"]),
            "utility_weight": float(utility_weight),
            "reference_score": float(reference_stats["score"]),
            "final_score": float(current_stats["score"]),
            "score_delta": float(score_delta),
            "reference_utility_component_mean": float(reference_stats["utility_mean"]),
            "final_utility_component_mean": float(current_stats["utility_mean"]),
            "utility_component_delta": float(utility_delta),
            "reference_privacy_component_mean": float(reference_privacy_component_mean),
            "reference_privacy_raw_mean": float(reference_privacy_raw_mean),
            "final_privacy_component_mean": float(final_privacy_component_mean),
            "final_privacy_raw_mean": float(final_privacy_raw_mean),
            "privacy_component_delta": float(privacy_delta),
            "candidate_solutions": [
                {
                    "init": str(item["init"]),
                    "satisfied": bool(item["satisfied"]),
                    "hard_floor_satisfied": bool(item.get("hard_floor_satisfied", False)),
                    "soft_floor_satisfied": bool(item.get("soft_floor_satisfied", False)),
                    "tradeoff_acceptable": bool(item.get("tradeoff_acceptable", False)),
                    "component_deltas": dict(item.get("component_deltas", {})),
                    "fid_1d": float(item["stats"]["fid_1d"]),
                    "fid_2d": float(item["stats"]["fid_2d"]),
                    "score": float(item["stats"]["score"]),
                    "privacy_mean": float(item["stats"]["privacy_mean"]),
                    "utility_mean": float(item["stats"]["utility_mean"]),
                    "repair_applied": bool(item.get("repair_report", {}).get("applied", False)),
                    "repair_mode": item.get("repair_report", {}).get("mode"),
                    "repair_satisfied": bool(item.get("repair_report", {}).get("satisfied", item["satisfied"])),
                }
                for item in candidate_solutions
            ],
            "privacy_gain_gate": {
                "enabled": False,
                "threshold": 0.0,
                "satisfied": True,
                "reverted_to_reference": bool(reverted_to_reference),
            },
            "benefit_gate": {
                "enabled": bool(benefit_gate_enabled),
                "satisfied": bool(benefit_satisfied),
                "reverted_to_reference": bool(reverted_to_reference),
                "privacy_delta_min": 0.005,
                "utility_delta_min": 0.003,
                "allowed_counterpart_drop": -0.002,
            },
        }

    def _apply_exact_floor_repair(
        self,
        preselected_records: list[dict[str, Any]],
        exact_records: list[dict[str, Any]],
        selected_indices: list[int],
        keep_k: int,
        floor_reference: dict[str, Any] | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        if not preselected_records or not exact_records or not selected_indices:
            return selected_indices, {"applied": False, "mode": "empty"}
        if floor_reference is None:
            return selected_indices, {"applied": False, "mode": "disabled_no_reference"}

        bucket_indices, pair_codes = self._bucket_pair_state_for_records(preselected_records)

        target_fid_1d = max(
            0.0,
            float(floor_reference.get("fidelity_1d", 0.0)) - self.final_fidelity_floor_eps,
        )
        target_fid_2d = max(
            0.0,
            float(floor_reference.get("fidelity_2d", 0.0)) - self.final_trend_floor_eps,
        )

        selected_mask = np.zeros(len(preselected_records), dtype=bool)
        selected_mask[np.asarray(selected_indices, dtype=int)] = True
        selected_counts_1d = {
            column: self._column_counts_from_bucket_indices(column, bucket_indices[column][selected_mask]).astype(int)
            for column in self.fidelity_columns
        }
        selected_counts_2d = [
            np.bincount(codes[selected_mask & (codes >= 0)], minlength=len(edge["probs"])).astype(int)
            for edge, codes in zip(self.pair_marginal_edges, pair_codes)
        ]

        current_fid_1d = self._subset_fidelity_from_counts(selected_counts_1d)
        current_fid_2d = self._subset_pair_fidelity_from_counts(selected_counts_2d)
        privacy_rank = _rank_normalize(
            np.asarray([float(record.get("pareto_priv_obj", 0.0)) for record in exact_records], dtype=float)
        )
        utility_rank = _rank_normalize(
            np.asarray([float(record.get("pareto_util_proxy_obj", 0.0)) for record in exact_records], dtype=float)
        )
        if current_fid_1d >= target_fid_1d and current_fid_2d >= target_fid_2d:
            selected_mask, signature_refine_report = self._refine_subset_with_signature_swaps(
                selected_mask=selected_mask,
                bucket_indices=bucket_indices,
                privacy_component=privacy_rank,
                utility_component=utility_rank,
            )
            selected_counts_1d = {
                column: self._column_counts_from_bucket_indices(column, bucket_indices[column][selected_mask]).astype(int)
                for column in self.fidelity_columns
            }
            selected_counts_2d = [
                np.bincount(codes[selected_mask & (codes >= 0)], minlength=len(edge["probs"])).astype(int)
                for edge, codes in zip(self.pair_marginal_edges, pair_codes)
            ]
            current_fid_1d = self._subset_fidelity_from_counts(selected_counts_1d)
            current_fid_2d = self._subset_pair_fidelity_from_counts(selected_counts_2d)
            current_privacy = float(privacy_rank[selected_mask].mean()) if np.any(selected_mask) else 0.0
            current_utility = float(utility_rank[selected_mask].mean()) if np.any(selected_mask) else 0.0
            current_quality = 0.25 * current_fid_1d + 0.25 * current_fid_2d + 0.25 * current_privacy + 0.25 * current_utility
            final_indices = np.sort(np.flatnonzero(selected_mask).astype(int, copy=False)).tolist()
            return final_indices, {
                "applied": bool(signature_refine_report.get("applied", False)),
                "mode": "already_satisfied",
                "reference_name": floor_reference.get("name", "floor_reference"),
                "target_fid_1d": target_fid_1d,
                "target_fid_2d": target_fid_2d,
                "current_fid_1d": current_fid_1d,
                "current_fid_2d": current_fid_2d,
                "current_privacy_mean": float(current_privacy),
                "current_utility_mean": float(current_utility),
                "current_quality": float(current_quality),
                "signature_refine": signature_refine_report,
                "satisfied": True,
            }
        fid1_rank = _rank_normalize(
            np.asarray([float(record.get("fid_marginal_1d", record.get("fid_marginal", 0.0))) for record in exact_records], dtype=float)
        )
        fid2_rank = _rank_normalize(
            np.asarray([float(record.get("fid_marginal_2d", 0.0)) for record in exact_records], dtype=float)
        )
        current_privacy = float(privacy_rank[selected_mask].mean()) if np.any(selected_mask) else 0.0
        current_utility = float(utility_rank[selected_mask].mean()) if np.any(selected_mask) else 0.0
        current_quality = 0.25 * current_fid_1d + 0.25 * current_fid_2d + 0.25 * current_privacy + 0.25 * current_utility

        batch_size = max(32, min(512, int(round(0.02 * keep_k))))
        rounds_applied = 0
        max_rounds = 6

        for _ in range(max_rounds):
            if current_fid_1d >= target_fid_1d and current_fid_2d >= target_fid_2d:
                break

            remove_1d, add_1d = self._train_prob_support_scores_1d(
                bucket_indices,
                selected_counts_1d,
                keep_k,
            )
            remove_2d, add_2d = self._train_prob_support_scores_2d(
                pair_codes,
                selected_counts_2d,
                keep_k,
            )

            remove_score = 0.6 * remove_1d + 0.4 * remove_2d
            add_score = 0.6 * add_1d + 0.4 * add_2d

            remove_priority = (
                remove_score
                - 0.10 * utility_rank
                - 0.08 * privacy_rank
                - 0.06 * fid1_rank
                - 0.06 * fid2_rank
            )
            add_priority = (
                add_score
                + 0.12 * utility_rank
                + 0.08 * privacy_rank
                + 0.05 * fid1_rank
                + 0.05 * fid2_rank
            )

            selected_idx = np.flatnonzero(selected_mask)
            available_idx = np.flatnonzero(~selected_mask)
            if selected_idx.size == 0 or available_idx.size == 0:
                break

            remove_order = selected_idx[np.argsort(-remove_priority[selected_idx], kind="mergesort")]
            add_order = available_idx[np.argsort(-add_priority[available_idx], kind="mergesort")]
            pair_count = min(int(batch_size), int(remove_order.size), int(add_order.size))
            if pair_count <= 0:
                break

            remove_batch = remove_order[:pair_count]
            add_batch = add_order[:pair_count]
            beneficial = add_priority[add_batch] > (remove_priority[remove_batch] + 1e-9)
            if not np.any(beneficial):
                break
            remove_batch = remove_batch[beneficial]
            add_batch = add_batch[beneficial]

            trial_mask = selected_mask.copy()
            trial_mask[remove_batch] = False
            trial_mask[add_batch] = True

            trial_counts_1d = {column: counts.copy() for column, counts in selected_counts_1d.items()}
            trial_counts_2d = [counts.copy() for counts in selected_counts_2d]
            self._update_subset_count_state(
                selected_counts_1d=trial_counts_1d,
                selected_counts_2d=trial_counts_2d,
                bucket_indices=bucket_indices,
                pair_codes=pair_codes,
                remove_indices=remove_batch,
                add_indices=add_batch,
            )

            trial_fid_1d = self._subset_fidelity_from_counts(trial_counts_1d)
            trial_fid_2d = self._subset_pair_fidelity_from_counts(trial_counts_2d)
            trial_privacy = float(privacy_rank[trial_mask].mean()) if np.any(trial_mask) else 0.0
            trial_utility = float(utility_rank[trial_mask].mean()) if np.any(trial_mask) else 0.0
            trial_quality = 0.25 * trial_fid_1d + 0.25 * trial_fid_2d + 0.25 * trial_privacy + 0.25 * trial_utility
            alignment_gain = (trial_fid_1d + trial_fid_2d) - (current_fid_1d + current_fid_2d)
            quality_drop = trial_quality - current_quality
            if (
                alignment_gain > 1e-9
                and (quality_drop >= -0.002 or (trial_fid_1d >= target_fid_1d and trial_fid_2d >= target_fid_2d))
            ):
                selected_mask = trial_mask
                selected_counts_1d = trial_counts_1d
                selected_counts_2d = trial_counts_2d
                current_fid_1d = trial_fid_1d
                current_fid_2d = trial_fid_2d
                current_privacy = trial_privacy
                current_utility = trial_utility
                current_quality = trial_quality
                rounds_applied += 1

        signature_refine_report = {
            "applied": False,
            "reason": "floors_not_satisfied",
            "swaps": 0,
        }
        if current_fid_1d >= target_fid_1d and current_fid_2d >= target_fid_2d:
            selected_mask, signature_refine_report = self._refine_subset_with_signature_swaps(
                selected_mask=selected_mask,
                bucket_indices=bucket_indices,
                privacy_component=privacy_rank,
                utility_component=utility_rank,
            )
            selected_counts_1d = {
                column: self._column_counts_from_bucket_indices(column, bucket_indices[column][selected_mask]).astype(int)
                for column in self.fidelity_columns
            }
            selected_counts_2d = [
                np.bincount(codes[selected_mask & (codes >= 0)], minlength=len(edge["probs"])).astype(int)
                for edge, codes in zip(self.pair_marginal_edges, pair_codes)
            ]
            current_fid_1d = self._subset_fidelity_from_counts(selected_counts_1d)
            current_fid_2d = self._subset_pair_fidelity_from_counts(selected_counts_2d)
            current_privacy = float(privacy_rank[selected_mask].mean()) if np.any(selected_mask) else 0.0
            current_utility = float(utility_rank[selected_mask].mean()) if np.any(selected_mask) else 0.0
            current_quality = 0.25 * current_fid_1d + 0.25 * current_fid_2d + 0.25 * current_privacy + 0.25 * current_utility

        final_indices = np.sort(np.flatnonzero(selected_mask).astype(int, copy=False)).tolist()
        return final_indices, {
            "applied": bool(rounds_applied > 0 or signature_refine_report.get("applied", False)),
            "mode": "exact_1d_2d_floor_repair",
            "reference_name": floor_reference.get("name", "floor_reference"),
            "reference_fid_1d": float(floor_reference.get("fidelity_1d", 0.0)),
            "reference_fid_2d": float(floor_reference.get("fidelity_2d", 0.0)),
            "target_fid_1d": target_fid_1d,
            "target_fid_2d": target_fid_2d,
            "current_fid_1d": current_fid_1d,
            "current_fid_2d": current_fid_2d,
            "current_privacy_mean": float(current_privacy),
            "current_utility_mean": float(current_utility),
            "current_quality": float(current_quality),
            "rounds": rounds_applied,
            "signature_refine": signature_refine_report,
            "satisfied": bool(current_fid_1d >= target_fid_1d and current_fid_2d >= target_fid_2d),
        }

    def construct_fidelity_ceiling_subset(
        self,
        preselected_records: list[dict[str, Any]],
        exact_records: list[dict[str, Any]],
        keep_k: int,
        *,
        utility_scores_by_id: dict[int, float] | None = None,
        utility_weight: float = 0.04,
        refine_utility_weight: float = 0.15,
        utility_score_name: str = "utility_static_balanced",
        show_progress: bool = False,
        progress_desc: str = "fidelity ceiling",
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
        if not preselected_records or not exact_records or keep_k <= 0:
            return pd.DataFrame(columns=self.column_order), [], {"selected": 0, "mode": "empty"}

        keep_k = min(int(keep_k), len(preselected_records))
        bucket_indices, pair_codes = self._bucket_pair_state_for_records(preselected_records)

        fid1_values = np.asarray(
            [float(record.get("fid_marginal_1d", record.get("fid_marginal", 0.0))) for record in exact_records],
            dtype=float,
        )
        fid2_values = np.asarray(
            [float(record.get("fid_marginal_2d", 0.0)) for record in exact_records],
            dtype=float,
        )
        fid1_rank = _rank_normalize(fid1_values)
        fid2_rank = _rank_normalize(fid2_values)
        candidate_ids = np.fromiter(
            (int(record.get("candidate_id", idx)) for idx, record in enumerate(preselected_records)),
            dtype=int,
            count=len(preselected_records),
        )
        utility_values = np.zeros(len(preselected_records), dtype=float)
        if utility_scores_by_id:
            utility_values = np.asarray(
                [float(utility_scores_by_id.get(int(candidate_id), 0.0)) for candidate_id in candidate_ids],
                dtype=float,
            )
            utility_values = np.where(np.isfinite(utility_values), utility_values, 0.0)
        use_utility_weight = float(utility_weight) if utility_scores_by_id else 0.0
        use_utility_weight = float(np.clip(use_utility_weight, 0.0, 0.30))
        utility_score_label = str(utility_score_name or "utility_score")
        fid2_tiebreak_weight = 0.02 if use_utility_weight < 0.98 else 0.0
        quota_weight = max(0.0, 1.0 - use_utility_weight - fid2_tiebreak_weight) / 2.0

        quota_targets_1d, quota_targets_2d, selected_counts_1d, selected_counts_2d = self._build_preselect_quota_targets(
            bucket_indices=bucket_indices,
            pair_codes=pair_codes,
            budget=keep_k,
            target_mode="train_clipped_by_availability",
        )
        pair_weights: list[float] = [max(float(edge.get("mi", 0.0)), 1e-6) for edge in self.pair_marginal_edges]

        def _lookup_normalized_deficit(codes: np.ndarray, quotas: np.ndarray, selected_counts: np.ndarray) -> np.ndarray:
            deficit = np.maximum(quotas.astype(float) - selected_counts.astype(float), 0.0)
            normalized = deficit / np.clip(quotas.astype(float), 1.0, None)
            padded = np.concatenate([normalized, np.asarray([0.0], dtype=float)])
            safe_codes = codes.copy()
            safe_codes[safe_codes < 0] = len(quotas)
            return padded[safe_codes]

        batch_size = max(64, min(1024, int(round(keep_k / 24.0))))
        total_pair_weight = max(float(sum(pair_weights)), 1.0)
        selected_mask = np.zeros(len(preselected_records), dtype=bool)
        batch_id = np.full(len(preselected_records), -1, dtype=int)
        batch_score = np.zeros(len(preselected_records), dtype=float)
        remaining_target = int(keep_k)
        num_batches = int(np.ceil(keep_k / max(batch_size, 1)))

        batch_iter = _progress(
            range(num_batches),
            total=num_batches,
            desc=progress_desc,
            disable=not show_progress,
        )
        for current_batch in batch_iter:
            if remaining_target <= 0:
                break
            score_1d = np.zeros(len(preselected_records), dtype=float)
            for column in self.fidelity_columns:
                score_1d += _lookup_normalized_deficit(
                    bucket_indices[column],
                    quota_targets_1d[column],
                    selected_counts_1d[column],
                )
            score_1d /= max(float(len(self.fidelity_columns)), 1.0)

            score_2d = np.zeros(len(preselected_records), dtype=float)
            for flat_codes, quotas, counts, weight in zip(pair_codes, quota_targets_2d, selected_counts_2d, pair_weights):
                score_2d += float(weight) * _lookup_normalized_deficit(flat_codes, quotas, counts)
            score_2d /= total_pair_weight

            final_score = (
                quota_weight * score_1d
                + quota_weight * score_2d
                + fid2_tiebreak_weight * fid2_rank
                + use_utility_weight * utility_values
            )
            final_score[selected_mask] = -np.inf

            available_indices = np.flatnonzero(~selected_mask)
            if available_indices.size == 0:
                break
            take_k = min(int(remaining_target), int(batch_size), int(available_indices.size))
            if available_indices.size <= take_k:
                chosen = available_indices
            else:
                local_scores = final_score[available_indices]
                top_local = np.argpartition(-local_scores, take_k - 1)[:take_k]
                chosen = available_indices[top_local]
                chosen = chosen[
                    np.lexsort(
                        (
                            chosen,
                            -fid1_rank[chosen],
                            -fid2_rank[chosen],
                            -score_2d[chosen],
                            -score_1d[chosen],
                            -final_score[chosen],
                        )
                    )
                ]

            selected_mask[chosen] = True
            batch_id[chosen] = int(current_batch)
            batch_score[chosen] = final_score[chosen]
            for column in self.fidelity_columns:
                self._add_code_count_delta(selected_counts_1d[column], bucket_indices[column][chosen], 1)
            for pair_idx, codes in enumerate(pair_codes):
                self._add_code_count_delta(selected_counts_2d[pair_idx], codes[chosen], 1)
            remaining_target -= int(len(chosen))
            if hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(batch=current_batch, remaining=remaining_target)

        use_refine_utility_weight = (
            float(refine_utility_weight)
            if utility_scores_by_id and use_utility_weight > 0.0
            else 0.0
        )
        use_refine_utility_weight = float(np.clip(use_refine_utility_weight, 0.0, 0.40))
        if use_refine_utility_weight > 0.0:
            remaining_refine_weight = 1.0 - use_refine_utility_weight
            refine_fid1_weight = remaining_refine_weight * (0.40 / 0.85)
            refine_fid2_weight = remaining_refine_weight * (0.45 / 0.85)
            refine_utility = (
                refine_fid1_weight * fid1_rank
                + refine_fid2_weight * fid2_rank
                + use_refine_utility_weight * utility_values
            )
        else:
            refine_fid1_weight = 0.44
            refine_fid2_weight = 0.56
            refine_utility = refine_fid1_weight * fid1_rank + refine_fid2_weight * fid2_rank
        selected_mask, refine_report = self._refine_subset_to_target_counts(
            bucket_indices=bucket_indices,
            pair_codes=pair_codes,
            selected_mask=selected_mask,
            target_counts_1d=quota_targets_1d,
            target_counts_2d=quota_targets_2d,
            utility=refine_utility,
            privacy_component=np.zeros_like(fid1_rank, dtype=float),
            max_rounds=10,
            batch_scale=0.0025,
        )

        selected_indices = np.sort(np.flatnonzero(selected_mask).astype(int, copy=False))
        keep_records = [preselected_records[idx] for idx in selected_indices.tolist()]
        keep_df = pd.DataFrame([record["row"] for record in keep_records], columns=self.column_order)
        selected_counts_1d, selected_counts_2d = self._subset_count_state_from_mask(
            bucket_indices=bucket_indices,
            pair_codes=pair_codes,
            selected_mask=selected_mask,
        )
        fidelity_1d = self._subset_fidelity_from_counts(selected_counts_1d)
        fidelity_2d = self._subset_pair_fidelity_from_counts(selected_counts_2d)
        if use_utility_weight <= 0.0:
            ceiling_mode = "fidelity_ceiling_anchor_v4"
        elif utility_score_label == "utility_proxy_second_pass":
            ceiling_mode = "fidelity_ceiling_anchor_v6_dynamic_utility"
        else:
            ceiling_mode = "fidelity_ceiling_anchor_v5_static_utility"
        report = {
            "selected": int(len(keep_records)),
            "keep_k": int(keep_k),
            "mode": ceiling_mode,
            "fidelity_1d": float(fidelity_1d),
            "fidelity_2d": float(fidelity_2d),
            "batch_size": int(batch_size),
            "batches_used": int(batch_id[selected_indices].max() + 1) if selected_indices.size > 0 else 0,
            "target_source": "train_target_clipped_by_availability",
            "score_weights": {
                "quota_1d": float(quota_weight),
                "quota_2d": float(quota_weight),
                "fid2_tiebreak": float(fid2_tiebreak_weight),
                utility_score_label: float(use_utility_weight),
            },
            "refine_weights": {
                "fid1_rank": float(refine_fid1_weight),
                "fid2_rank": float(refine_fid2_weight),
                utility_score_label: float(use_refine_utility_weight),
            },
            "utility_score": {
                "name": utility_score_label,
                "available": bool(utility_scores_by_id),
                "weight": float(use_utility_weight),
                "refine_weight": float(use_refine_utility_weight),
                "mean_all": float(np.mean(utility_values)) if utility_values.size > 0 else 0.0,
                "mean_selected": float(np.mean(utility_values[selected_mask])) if np.any(selected_mask) else 0.0,
            },
            "refine_report": refine_report,
            "reference": {
                "name": "preselected_fidelity_ceiling_keep_k",
                "rows": int(len(keep_records)),
                "fidelity_1d": float(fidelity_1d),
                "fidelity_2d": float(fidelity_2d),
            },
        }
        return keep_df, keep_records, report

    def _fidelity_guard_subset(
        self,
        preselected_records: list[dict[str, Any]],
        exact_records: list[dict[str, Any]],
        keep_k: int,
        *,
        mode: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        if not preselected_records or not exact_records:
            return preselected_records, exact_records, {"applied": False, "mode": "empty"}

        if len(exact_records) <= keep_k:
            return preselected_records, exact_records, {"applied": False, "mode": "pool_not_larger_than_keep"}

        fidelity_field = "fid_marginal_1d" if mode == "naive" else "fid_marginal"
        fidelity_values = np.asarray([float(record.get(fidelity_field, 0.0)) for record in exact_records], dtype=float)
        nonnegative_indices = np.where(fidelity_values >= 0.0)[0].tolist()
        removable_gap = max(0, len(exact_records) - keep_k)
        slack_rows = max(128, int(round(0.25 * removable_gap)))
        band_target = min(len(exact_records), keep_k + slack_rows)

        if len(nonnegative_indices) >= band_target:
            selected_indices = sorted(nonnegative_indices)
            threshold = 0.0
            guard_mode = "nonnegative_band"
        else:
            ordered = np.argsort(-fidelity_values).tolist()
            selected_indices = sorted(ordered[:band_target])
            threshold = float(fidelity_values[selected_indices[-1]]) if selected_indices else float("-inf")
            guard_mode = "top_fidelity_band"

        guarded_records = [preselected_records[idx] for idx in selected_indices]
        guarded_exact = [exact_records[idx] for idx in selected_indices]
        return guarded_records, guarded_exact, {
            "applied": len(guarded_records) < len(preselected_records),
            "mode": guard_mode,
            "fidelity_field": fidelity_field,
            "eligible_rows": len(guarded_records),
            "original_rows": len(preselected_records),
            "threshold": threshold,
            "nonnegative_rows": len(nonnegative_indices),
            "band_target": band_target,
        }

    def select_keep(
        self,
        preselected_records: list[dict[str, Any]],
        surrogate_records: list[dict[str, Any]],
        exact_records: list[dict[str, Any]],
        keep_k: int,
        floor_reference: dict[str, Any] | None = None,
        constraint_reference_records: list[dict[str, Any]] | None = None,
        floor_mode: str = "hard",
        soft_fidelity_floor_eps: float | None = None,
        soft_trend_floor_eps: float | None = None,
        soft_privacy_floor_eps: float = 0.005,
        soft_utility_floor_eps: float = 0.005,
        soft_min_score_delta: float = 0.0,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
        if not preselected_records or not exact_records:
            return pd.DataFrame(columns=self.column_order), [], {"fronts": [], "selected": 0, "source": self.source}

        keep_k = min(keep_k, len(preselected_records))
        if floor_reference is not None:
            fidelity_guard = {"applied": False, "mode": "disabled_for_constrained_subset"}
            points = self._pareto_points(exact_records)
            front_summaries = [
                {
                    "front_size": None,
                    "mode": "reference_constrained_deferred",
                    "reason": "fronts_computed_inside_constrained_objective",
                    "rows": int(len(exact_records)),
                }
            ]
            selected_indices, floor_repair_report = self._construct_constrained_keep_subset(
                preselected_records=preselected_records,
                exact_records=exact_records,
                keep_k=keep_k,
                mode="pareto",
                floor_reference=floor_reference,
                constraint_reference_records=constraint_reference_records,
                floor_mode=floor_mode,
                soft_fidelity_floor_eps=soft_fidelity_floor_eps,
                soft_trend_floor_eps=soft_trend_floor_eps,
                soft_privacy_floor_eps=soft_privacy_floor_eps,
                soft_utility_floor_eps=soft_utility_floor_eps,
                soft_min_score_delta=soft_min_score_delta,
            )
        else:
            preselected_records, exact_records, fidelity_guard = self._fidelity_guard_subset(
                preselected_records=preselected_records,
                exact_records=exact_records,
                keep_k=keep_k,
                mode="pareto",
            )
            points = self._pareto_points(exact_records)
            keep_k = min(keep_k, len(preselected_records))
            selected_indices, front_summaries, fronts = self._select_indices_by_nsga(points, exact_records, keep_k)
            selected_indices, floor_repair_report = self._apply_exact_floor_repair(
                preselected_records=preselected_records,
                exact_records=exact_records,
                selected_indices=selected_indices,
                keep_k=keep_k,
                floor_reference=floor_reference,
            )
        keep_records = [preselected_records[idx] for idx in selected_indices]
        selected_exact_records = [exact_records[idx] for idx in selected_indices]
        keep_df = pd.DataFrame([record["row"] for record in keep_records], columns=self.column_order)
        return keep_df, keep_records, {
            "fronts": front_summaries,
            "selected": len(keep_records),
            "keep_k": keep_k,
            "source": self.source,
            "mode": "pareto",
            "point_dimension": int(points.shape[1]) if len(exact_records) > 0 else 0,
            "floor_reference_name": floor_reference.get("name") if floor_reference is not None else None,
            "selected_privacy_component_mean": (
                float(np.mean([float(record.get("pareto_priv_obj", 0.0)) for record in selected_exact_records]))
                if selected_exact_records
                else 0.0
            ),
            "selected_privacy_raw_mean": (
                float(np.mean([float(record.get("privacy_score_selected", 0.0)) for record in selected_exact_records]))
                if selected_exact_records
                else 0.0
            ),
            "selected_utility_mean": (
                float(np.mean([float(record.get("pareto_util_proxy_obj", 0.0)) for record in selected_exact_records]))
                if selected_exact_records
                else 0.0
            ),
            "fidelity_guard": fidelity_guard,
            "exact_floor_repair": floor_repair_report,
        }

    def reduce_archive(
        self,
        archive_records: list[dict[str, Any]],
        archive_exact_records: list[dict[str, Any]],
        budget: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        if len(archive_records) <= budget:
            return archive_records, archive_exact_records, {
                "archive_rows_before_reduction": len(archive_records),
                "archive_rows_after_reduction": len(archive_records),
                "reduction_applied": False,
                "secondary_filter": {"applied": False},
            }

        points = self._pareto_points(archive_exact_records)
        fronts = _non_dominated_sort(points)
        front_rank, crowding = self._front_rank_and_crowding(points, fronts)
        priority = self._candidate_priority(
            points,
            archive_exact_records,
            fronts,
            front_rank=front_rank,
            crowding=crowding,
        )
        selected_indices, front_summaries, fronts = self._select_indices_by_nsga(
            points,
            archive_exact_records,
            budget,
            fronts=fronts,
            front_rank=front_rank,
            crowding=crowding,
        )
        if len(selected_indices) > budget:
            selected_indices = selected_indices[:budget]
        selected_indices, secondary_report = self._secondary_rarity_reduce(
            selected_indices=selected_indices,
            exact_records=archive_exact_records,
            candidate_records=archive_records,
            points=points,
            fronts=fronts,
            budget=budget,
            priority=priority,
        )
        reduced_records = [archive_records[idx] for idx in selected_indices]
        reduced_exact = [archive_exact_records[idx] for idx in selected_indices]
        return reduced_records, reduced_exact, {
            "archive_rows_before_reduction": len(archive_records),
            "archive_rows_after_reduction": len(reduced_records),
            "reduction_applied": True,
            "point_dimension": int(points.shape[1]) if len(archive_exact_records) > 0 else 0,
            "fronts": front_summaries,
            "secondary_filter": secondary_report,
        }

    def select_keep_random(
        self,
        candidate_records: list[dict[str, Any]],
        keep_k: int,
        rng_seed: int | None = None,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
        if not candidate_records:
            return pd.DataFrame(columns=self.column_order), [], {"selected": 0, "mode": "random", "source": self.source}
        keep_k = min(keep_k, len(candidate_records))
        rng = np.random.default_rng(self.seed if rng_seed is None else rng_seed)
        chosen = rng.choice(len(candidate_records), size=keep_k, replace=False).tolist()
        keep_records = [candidate_records[idx] for idx in chosen]
        keep_df = pd.DataFrame([record["row"] for record in keep_records], columns=self.column_order)
        return keep_df, keep_records, {
            "selected": len(keep_records),
            "keep_k": keep_k,
            "mode": "random",
            "source": self.source,
        }

    def select_keep_scalarization(
        self,
        preselected_records: list[dict[str, Any]],
        exact_records: list[dict[str, Any]],
        keep_k: int,
        fidelity_1d_weight: float = 0.25,
        fidelity_2d_weight: float = 0.25,
        privacy_weight: float = 0.5,
        utility_weight: float = 0.0,
        mode: str = "matched",
        floor_reference: dict[str, Any] | None = None,
        constraint_reference_records: list[dict[str, Any]] | None = None,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
        if not preselected_records or not exact_records:
            return (
                pd.DataFrame(columns=self.column_order),
                [],
                {
                    "selected": 0,
                    "keep_k": keep_k,
                    "mode": f"scalarization_{mode}",
                    "source": self.source,
                },
            )

        if mode != "naive" and not (floor_reference is not None and constraint_reference_records):
            preselected_records, exact_records, fidelity_guard = self._fidelity_guard_subset(
                preselected_records=preselected_records,
                exact_records=exact_records,
                keep_k=keep_k,
                mode="matched",
            )
        elif mode != "naive":
            fidelity_guard = {"applied": False, "mode": "disabled_for_constrained_subset"}
        else:
            fidelity_guard = {"applied": False, "mode": "disabled_for_naive"}
        keep_k = min(keep_k, len(exact_records))
        objective_mode = "scalar_naive" if mode == "naive" else "scalar_matched"
        objective = self._selection_objective_components(
            exact_records,
            mode=objective_mode,
            fidelity_1d_weight=fidelity_1d_weight,
            fidelity_2d_weight=fidelity_2d_weight,
            privacy_weight=privacy_weight,
            utility_weight=utility_weight,
        )
        scalar_scores = np.asarray(objective["utility"], dtype=float)
        constraint_reference_used = bool(mode != "naive" and floor_reference is not None and constraint_reference_records)
        if mode != "naive" and floor_reference is not None and constraint_reference_records:
            selected_indices, floor_repair_report = self._construct_constrained_keep_subset(
                preselected_records=preselected_records,
                exact_records=exact_records,
                keep_k=keep_k,
                mode="scalar_matched",
                floor_reference=floor_reference,
                constraint_reference_records=constraint_reference_records,
                fidelity_1d_weight=fidelity_1d_weight,
                fidelity_2d_weight=fidelity_2d_weight,
                privacy_weight=privacy_weight,
                utility_weight=utility_weight,
            )
        else:
            ordered = np.argsort(-scalar_scores).tolist()
            selected_indices = ordered[:keep_k]
            selected_indices, floor_repair_report = self._apply_exact_floor_repair(
                preselected_records=preselected_records,
                exact_records=exact_records,
                selected_indices=selected_indices,
                keep_k=keep_k,
                floor_reference=floor_reference if mode != "naive" else None,
            )
        keep_records = [preselected_records[idx] for idx in selected_indices]
        selected_exact_records = [exact_records[idx] for idx in selected_indices]
        keep_df = pd.DataFrame([record["row"] for record in keep_records], columns=self.column_order)
        return keep_df, keep_records, {
            "selected": len(keep_records),
            "keep_k": keep_k,
            "mode": f"scalarization_{mode}",
            "source": self.source,
            "constraint_reference_used": bool(constraint_reference_used),
            "selection_path": (
                "constrained_subset_construction"
                if constraint_reference_used
                else (
                    "direct_preselected_with_floor_repair"
                    if mode != "naive"
                    else "direct_preselected_without_floor_repair"
                )
            ),
            "fidelity_1d_weight": fidelity_1d_weight,
            "fidelity_2d_weight": fidelity_2d_weight,
            "privacy_weight": privacy_weight,
            "utility_weight": utility_weight,
            "point_dimension": int(self._pareto_points(exact_records).shape[1]) if len(exact_records) > 0 else 0,
            "floor_reference_name": floor_reference.get("name") if floor_reference is not None else None,
            "top_scalar_score": float(scalar_scores[selected_indices[0]]) if selected_indices else 0.0,
            "selected_privacy_component_mean": (
                float(np.mean([float(record.get("pareto_priv_obj", 0.0)) for record in selected_exact_records]))
                if selected_exact_records
                else 0.0
            ),
            "selected_privacy_raw_mean": (
                float(np.mean([float(record.get("privacy_score_selected", 0.0)) for record in selected_exact_records]))
                if selected_exact_records
                else 0.0
            ),
            "selected_utility_mean": (
                float(np.mean([float(record.get("pareto_util_proxy_obj", 0.0)) for record in selected_exact_records]))
                if selected_exact_records
                else 0.0
            ),
            "fidelity_guard": fidelity_guard,
            "exact_floor_repair": floor_repair_report,
        }

    def select_keep_chebyshev(
        self,
        preselected_records: list[dict[str, Any]],
        exact_records: list[dict[str, Any]],
        keep_k: int,
        fidelity_1d_weight: float,
        fidelity_2d_weight: float,
        privacy_weight: float,
        utility_weight: float = 0.0,
        floor_reference: dict[str, Any] | None = None,
        constraint_reference_records: list[dict[str, Any]] | None = None,
        floor_mode: str = "hard",
        soft_fidelity_floor_eps: float | None = None,
        soft_trend_floor_eps: float | None = None,
        soft_privacy_floor_eps: float = 0.005,
        soft_utility_floor_eps: float = 0.005,
        soft_min_score_delta: float = 0.0,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
        if not preselected_records or not exact_records:
            return (
                pd.DataFrame(columns=self.column_order),
                [],
                {
                    "selected": 0,
                    "keep_k": keep_k,
                    "mode": "chebyshev",
                    "source": self.source,
                },
            )

        keep_k = min(keep_k, len(exact_records))
        if floor_reference is not None:
            fidelity_guard = {"applied": False, "mode": "disabled_for_constrained_subset"}
            objective = self._selection_objective_components(
                exact_records,
                mode="chebyshev",
                fidelity_1d_weight=fidelity_1d_weight,
                fidelity_2d_weight=fidelity_2d_weight,
                privacy_weight=privacy_weight,
                utility_weight=utility_weight,
            )
            chebyshev_scores = np.asarray(objective.get("chebyshev_scores", np.zeros(len(exact_records), dtype=float)), dtype=float)
            selected_indices, floor_repair_report = self._construct_constrained_keep_subset(
                preselected_records=preselected_records,
                exact_records=exact_records,
                keep_k=keep_k,
                mode="chebyshev",
                floor_reference=floor_reference,
                constraint_reference_records=constraint_reference_records,
                fidelity_1d_weight=fidelity_1d_weight,
                fidelity_2d_weight=fidelity_2d_weight,
                privacy_weight=privacy_weight,
                utility_weight=utility_weight,
                floor_mode=floor_mode,
                soft_fidelity_floor_eps=soft_fidelity_floor_eps,
                soft_trend_floor_eps=soft_trend_floor_eps,
                soft_privacy_floor_eps=soft_privacy_floor_eps,
                soft_utility_floor_eps=soft_utility_floor_eps,
                soft_min_score_delta=soft_min_score_delta,
            )
        else:
            preselected_records, exact_records, fidelity_guard = self._fidelity_guard_subset(
                preselected_records=preselected_records,
                exact_records=exact_records,
                keep_k=keep_k,
                mode="matched",
            )
            keep_k = min(keep_k, len(exact_records))
            points = self._pareto_points(exact_records)
            objective = self._selection_objective_components(
                exact_records,
                mode="chebyshev",
                fidelity_1d_weight=fidelity_1d_weight,
                fidelity_2d_weight=fidelity_2d_weight,
                privacy_weight=privacy_weight,
                utility_weight=utility_weight,
            )
            chebyshev_scores = np.asarray(
                objective.get("chebyshev_scores", np.zeros(len(exact_records), dtype=float)),
                dtype=float,
            )
            fronts = _non_dominated_sort(points)
            front_rank = self._build_front_rank_map(fronts)
            ordered = sorted(
                range(len(exact_records)),
                key=lambda idx: (
                    front_rank.get(idx, len(fronts) + 1),
                    float(chebyshev_scores[idx]),
                    float(-exact_records[idx].get("pareto_util_proxy_obj", 0.0)),
                    float(-exact_records[idx].get("pareto_fid_2d_obj", 0.0)),
                    float(-exact_records[idx].get("pareto_fid_1d_obj", 0.0)),
                ),
            )
            selected_indices = ordered[:keep_k]
            selected_indices, floor_repair_report = self._apply_exact_floor_repair(
                preselected_records=preselected_records,
                exact_records=exact_records,
                selected_indices=selected_indices,
                keep_k=keep_k,
                floor_reference=floor_reference,
            )
        keep_records = [preselected_records[idx] for idx in selected_indices]
        selected_exact_records = [exact_records[idx] for idx in selected_indices]
        keep_df = pd.DataFrame([record["row"] for record in keep_records], columns=self.column_order)
        return keep_df, keep_records, {
            "selected": len(keep_records),
            "keep_k": keep_k,
            "mode": "chebyshev",
            "source": self.source,
            "fidelity_1d_weight": fidelity_1d_weight,
            "fidelity_2d_weight": fidelity_2d_weight,
            "privacy_weight": privacy_weight,
            "utility_weight": utility_weight,
            "point_dimension": int(self._pareto_points(exact_records).shape[1]) if len(exact_records) > 0 else 0,
            "floor_reference_name": floor_reference.get("name") if floor_reference is not None else None,
            "best_chebyshev_score": float(chebyshev_scores[selected_indices[0]]) if selected_indices else 0.0,
            "selected_privacy_component_mean": (
                float(np.mean([float(record.get("pareto_priv_obj", 0.0)) for record in selected_exact_records]))
                if selected_exact_records
                else 0.0
            ),
            "selected_privacy_raw_mean": (
                float(np.mean([float(record.get("privacy_score_selected", 0.0)) for record in selected_exact_records]))
                if selected_exact_records
                else 0.0
            ),
            "selected_utility_mean": (
                float(np.mean([float(record.get("pareto_util_proxy_obj", 0.0)) for record in selected_exact_records]))
                if selected_exact_records
                else 0.0
            ),
            "fidelity_guard": fidelity_guard,
            "exact_floor_repair": floor_repair_report,
        }

    def _candidate_ids(self, records: list[dict[str, Any]]) -> list[int]:
        return [int(record.get("candidate_id", idx)) for idx, record in enumerate(records)]

    def compute_rarity_stratified_keep_rate(
        self,
        candidate_records: list[dict[str, Any]],
        keep_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidate_records:
            return []

        candidate_ids = np.asarray(self._candidate_ids(candidate_records), dtype=int)
        keep_ids = np.asarray(self._candidate_ids(keep_records), dtype=int)
        if candidate_records and "gate_stratum" in candidate_records[0]:
            strata = np.asarray([int(record["gate_stratum"]) for record in candidate_records], dtype=int)
        else:
            candidate_df = pd.DataFrame([record["row"] for record in candidate_records], columns=self.column_order)
            gate_probs = self._prob_geomean_for_df(candidate_df, columns=self.feature_columns)
            strata = self._assign_bins_from_edges(gate_probs, self.train_gate_edges)
        num_strata = len(self.train_gate_edges) - 1
        candidate_rows = np.bincount(strata, minlength=num_strata).astype(int)
        selected_mask = np.isin(candidate_ids, keep_ids, assume_unique=False)
        selected_rows = np.bincount(strata[selected_mask], minlength=num_strata).astype(int)
        return [
            {
                "stratum": stratum,
                "prob_low": float(self.train_gate_edges[stratum]),
                "prob_high": float(self.train_gate_edges[stratum + 1]),
                "candidate_rows": int(candidate_rows[stratum]),
                "selected_rows": int(selected_rows[stratum]),
                "keep_rate": float(selected_rows[stratum] / candidate_rows[stratum]) if candidate_rows[stratum] else 0.0,
            }
            for stratum in range(num_strata)
        ]

    def compute_rare_bin_inflation(self, selected_df: pd.DataFrame, rare_threshold: float = 0.05) -> list[dict[str, Any]]:
        if selected_df.empty:
            return []
        relevant_columns = [
            column
            for column in self.fidelity_columns
            if self.schema_card["columns"][column]["type"] in {"categorical", "discrete_numerical"}
        ]
        if not relevant_columns:
            return []

        bucket_indices_map = self._column_bucket_indices_for_df(selected_df, relevant_columns)
        frames: list[pd.DataFrame] = []
        for column in relevant_columns:
            train_dist = self.train_distributions[column]
            train_probs = np.asarray(train_dist["probs"], dtype=float)
            rare_mask = train_probs <= float(rare_threshold)
            if not np.any(rare_mask):
                continue

            selected_counts = self._column_counts_from_bucket_indices(column, bucket_indices_map[column])
            selected_probs = selected_counts / max(float(selected_counts.sum()), 1.0)
            rare_indices = np.flatnonzero(rare_mask)
            values = np.asarray(train_dist["values"], dtype=object)
            frames.append(
                pd.DataFrame(
                    {
                        "column": column,
                        "value": values[rare_indices],
                        "train_prob": train_probs[rare_indices],
                        "selected_prob": selected_probs[rare_indices],
                    }
                )
            )

        if not frames:
            return []

        report_df = pd.concat(frames, axis=0, ignore_index=True)
        report_df["inflation_diff"] = report_df["selected_prob"] - report_df["train_prob"]
        report_df["inflation_ratio"] = report_df["selected_prob"] / np.clip(
            report_df["train_prob"].to_numpy(dtype=float, copy=False),
            1e-12,
            None,
        )
        report_df = report_df.sort_values(
            by=["inflation_diff", "column", "value"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        return report_df.to_dict(orient="records")

    def compute_smoke_metrics(
        self,
        surrogate_records_all: list[dict[str, Any]],
        keep_records: list[dict[str, Any]],
        keep_df: pd.DataFrame,
    ) -> dict[str, Any]:
        if not surrogate_records_all:
            return {
                "pass": False,
                "reason": "no_valid_candidates",
                "source": self.source,
            }

        all_surrogates_df = pd.DataFrame.from_records(surrogate_records_all)
        all_fid = float(all_surrogates_df["s_fid_sur"].mean())
        all_priv = float(all_surrogates_df["s_priv_sur"].mean())
        if keep_df.empty:
            return {
                "pass": False,
                "reason": "no_keep_candidates",
                "source": self.source,
                "avg_bin_hit_all": all_fid,
                "avg_nn_distance_all": all_priv,
            }

        keep_surrogates_df: pd.DataFrame | None = None
        if "candidate_id" in all_surrogates_df.columns:
            keep_ids = np.asarray(self._candidate_ids(keep_records), dtype=int)
            if keep_ids.size > 0:
                matched = all_surrogates_df[all_surrogates_df["candidate_id"].isin(keep_ids)]
                if len(matched) == len(keep_ids):
                    keep_surrogates_df = matched

        if keep_surrogates_df is None:
            keep_surrogates_df = pd.DataFrame.from_records(self.compute_surrogates(keep_df))

        keep_fid = float(keep_surrogates_df["s_fid_sur"].mean())
        keep_priv = float(keep_surrogates_df["s_priv_sur"].mean())
        fid_improved = keep_fid > all_fid
        priv_improved = keep_priv > all_priv
        fid_degradation = max(0.0, (all_fid - keep_fid) / max(all_fid, 1e-12))
        priv_degradation = max(0.0, (all_priv - keep_priv) / max(all_priv, 1e-12))
        smoke_pass = (fid_improved and priv_degradation <= 0.05) or (priv_improved and fid_degradation <= 0.05)
        return {
            "source": self.source,
            "pass": smoke_pass,
            "avg_bin_hit_all": all_fid,
            "avg_bin_hit_keep": keep_fid,
            "avg_nn_distance_all": all_priv,
            "avg_nn_distance_keep": keep_priv,
            "avg_fidelity_sur_all": all_fid,
            "avg_fidelity_sur_keep": keep_fid,
            "avg_privacy_sur_all": all_priv,
            "avg_privacy_sur_keep": keep_priv,
            "fid_improved": fid_improved,
            "priv_improved": priv_improved,
            "fid_degradation_ratio": fid_degradation,
            "priv_degradation_ratio": priv_degradation,
        }
