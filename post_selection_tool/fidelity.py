from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .encoding import (
    _digitize_value,
    _js_divergence,
    _progress,
    _robust_unit_scale,
    _safe_geometric_mean,
)


class FidelityMixin:
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
        bucket_indices = self._column_bucket_indices_for_df(d_cur_df, self.fidelity_2d_anchor_columns)
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

        bucket_indices = self._column_bucket_indices_for_df(candidate_df, self.fidelity_2d_anchor_columns)
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
                legal_values = self.high_cardinality_compressor.values_for_column(info["legal_values"], column)
                categorical = pd.Categorical(self.search_train_df[column].astype(str), categories=legal_values)
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
        if self.max_pair_marginal_edges <= 0 or len(self.fidelity_2d_anchor_columns) < 2:
            return []

        train_bucket_indices = self._column_bucket_indices_for_df(self.search_train_df, self.fidelity_2d_anchor_columns)
        candidates: list[dict[str, Any]] = []
        for left_pos, left in enumerate(self.fidelity_2d_anchor_columns):
            left_probs = np.asarray(self.train_distributions[left]["probs"], dtype=float)
            left_idx = train_bucket_indices[left]
            left_bins = int(len(left_probs))
            if left_bins <= 0:
                continue
            for right in self.fidelity_2d_anchor_columns[left_pos + 1 :]:
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
        if column in self.high_cardinality_compressor.mappings:
            value = self.high_cardinality_compressor.transform_series(pd.Series([value]), column).iloc[0]
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

        benefit_1d_norm, benefit_1d_calibration = _robust_unit_scale(marginal_fidelity_1d)
        benefit_2d_norm, benefit_2d_calibration = _robust_unit_scale(marginal_fidelity_2d)
        penalty_1d_norm, penalty_1d_calibration = _robust_unit_scale(fidelity_penalty_1d)
        penalty_2d_norm, penalty_2d_calibration = _robust_unit_scale(fidelity_penalty_2d)

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
        raw_pareto_fid_obj = 0.5 * raw_pareto_fid_1d_obj + 0.5 * raw_pareto_fid_2d_obj
        pareto_fid_1d_obj, pareto_fid_1d_calibration = _robust_unit_scale(raw_pareto_fid_1d_obj)
        pareto_fid_2d_obj, pareto_fid_2d_calibration = _robust_unit_scale(raw_pareto_fid_2d_obj)
        pareto_fid_obj = 0.5 * pareto_fid_1d_obj + 0.5 * pareto_fid_2d_obj

        if show_progress:
            for _ in _progress(range(1), total=1, desc=progress_desc, disable=False):
                pass

        privacy_score_selected = privacy_df["privacy_score_selected"].to_numpy(dtype=float, copy=False)
        pareto_priv_obj, privacy_calibration = _robust_unit_scale(privacy_score_selected)

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
                "privacy_score_selected": privacy_score_selected,
                "pareto_priv_obj_raw": privacy_score_selected,
                "pareto_priv_obj": pareto_priv_obj,
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
            "fidelity_objective_scaling": "candidate_pool_robust_quantile",
            "fidelity_components": ["1d_exact", "2d_exact", "over_frequency_penalty"],
            "gamma": self.gamma,
            "privacy_version": self.privacy_version,
            "privacy_objective_scaling": "candidate_pool_robust_quantile",
            "objective_calibration": {
                "basis": "same_preselected_candidate_pool",
                "method": "clip((x - q05) / (q95 - q05), 0, 1)",
                "fidelity_marginal_1d": benefit_1d_calibration,
                "fidelity_marginal_2d": benefit_2d_calibration,
                "fidelity_penalty_1d": penalty_1d_calibration,
                "fidelity_penalty_2d": penalty_2d_calibration,
                "pareto_fid_1d_obj_raw": pareto_fid_1d_calibration,
                "pareto_fid_2d_obj_raw": pareto_fid_2d_calibration,
                "pareto_priv_obj_raw": privacy_calibration,
            },
        }
