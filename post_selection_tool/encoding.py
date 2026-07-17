from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .logging_utils import get_logger

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

def _progress(iterable: Any, *, total: int, desc: str, disable: bool) -> Any:
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, disable=disable)

def _progress_write(message: str) -> None:
    get_logger().info(message)

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



def _robust_unit_scale(values: np.ndarray, *, lower_q: float = 0.05, upper_q: float = 0.95) -> tuple[np.ndarray, dict[str, float]]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if array.size == 0:
        return array.astype(float), {"lower": 0.0, "upper": 0.0, "lower_q": float(lower_q), "upper_q": float(upper_q)}
    if finite.size == 0:
        return np.zeros_like(array, dtype=float), {"lower": 0.0, "upper": 0.0, "lower_q": float(lower_q), "upper_q": float(upper_q)}
    lower = float(np.quantile(finite, lower_q))
    upper = float(np.quantile(finite, upper_q))
    if upper - lower <= 1e-12:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
    if upper - lower <= 1e-12:
        return np.zeros_like(array, dtype=float), {"lower": lower, "upper": upper, "lower_q": float(lower_q), "upper_q": float(upper_q)}
    scaled = (array - lower) / (upper - lower)
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return np.clip(scaled, 0.0, 1.0), {"lower": lower, "upper": upper, "lower_q": float(lower_q), "upper_q": float(upper_q)}


class EncodingMixin:
    def _resolve_guided_column_scope(
        self,
        name: str,
        columns: list[str] | None,
        *,
        default_columns: list[str],
        allow_target_when_guided: bool,
        require_non_empty: bool,
    ) -> list[str]:
        if columns is None:
            return list(default_columns)
        normalized = [
            str(column).strip()
            for column in columns
            if column is not None and str(column).strip()
        ]
        resolved = list(dict.fromkeys(normalized))
        if require_non_empty and not resolved:
            raise ValueError(f"{name} must not be empty.")
        known_columns = set(self.column_order)
        unknown = [column for column in resolved if column not in known_columns]
        if unknown:
            raise ValueError(f"{name} contains unknown columns: {unknown}")
        if not allow_target_when_guided:
            feature_set = set(self.feature_columns)
            invalid = [column for column in resolved if column not in feature_set]
            if invalid:
                raise ValueError(f"{name} must contain only non-target feature columns: {invalid}")
        return resolved

    def _resolve_guided_single_feature(self, name: str, column: str | None) -> str | None:
        if column is None:
            return None
        normalized = str(column).strip()
        if not normalized:
            return None
        if normalized not in set(self.column_order):
            raise ValueError(f"{name} contains unknown column: {normalized}")
        if normalized not in set(self.feature_columns):
            raise ValueError(f"{name} must be a non-target feature column: {normalized}")
        return normalized

    def _build_numeric_impute_values(self) -> dict[str, float]:
        impute_values: dict[str, float] = {}
        for column in self.numeric_columns:
            values = pd.to_numeric(self.train_df[column], errors="coerce").to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            impute_values[column] = float(np.median(finite_values)) if finite_values.size else 0.0
        return impute_values

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

        categorical_series = self.high_cardinality_compressor.transform_series(series, column)
        categorical = pd.Categorical(categorical_series.astype(str), categories=train_dist["values"])
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
        bucket_indices = self._column_bucket_indices_for_df(candidate_df, self.fidelity_bucket_columns)
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

    def _numeric_frame_for_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df[self.privacy_numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        fill_values = pd.Series(self.privacy_numeric_impute_values, dtype=float)
        return numeric_df.fillna(fill_values).astype(float)

    def _encode_df(self, df: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.privacy_numeric_columns:
            assert self.scaler is not None
            parts.append(self.scaler.transform(self._numeric_frame_for_encoding(df)))
        if self.privacy_categorical_columns:
            assert self.ohe is not None
            search_df = self.high_cardinality_compressor.transform_df(df)
            parts.append(self.ohe.transform(search_df[self.privacy_categorical_columns].astype(str)))
        if not parts:
            return np.zeros((len(df), 0), dtype=float)
        return np.concatenate(parts, axis=1)

    def _encode_row(self, row: dict[str, Any]) -> np.ndarray:
        df = pd.DataFrame([row], columns=self.column_order)
        return self._encode_df(df)

    def _match_discrete_value(self, value: Any, legal_values: list[Any]) -> int:
        legal_array = np.asarray([float(v) for v in legal_values], dtype=float)
        return int(np.argmin(np.abs(legal_array - float(value))))

    def _assign_bins_from_edges(self, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        if len(edges) <= 2:
            return np.zeros(len(values), dtype=int)
        clipped = np.clip(values, float(edges[0]), float(edges[-1]))
        bins = np.digitize(clipped, edges[1:-1], right=False)
        return bins.astype(int)
