from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .encoding import _make_ohe, _progress, _quantile_edges


DEFAULT_D_CUR_SIZE = 200


class PrivacyMixin:
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
        if self.privacy_numeric_columns:
            self.scaler: StandardScaler | None = StandardScaler()
            self.scaler.fit(self._numeric_frame_for_encoding(self.train_df))
        else:
            self.scaler = None

        if self.privacy_categorical_columns:
            self.ohe: OneHotEncoder | None = _make_ohe()
            cat_fit = pd.concat(
                [
                    self.search_train_df[self.privacy_categorical_columns].astype(str),
                    self.search_holdout_df[self.privacy_categorical_columns].astype(str),
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
        feature_gate_probs = self._prob_geomean_for_df(self.train_df, columns=self.feature_columns)
        self.train_feature_gate_probs = feature_gate_probs
        self.train_feature_gate_edges = _quantile_edges(feature_gate_probs, n_bins=self.rarity_strata)
        feature_gate_bins = self._assign_bins_from_edges(feature_gate_probs, self.train_feature_gate_edges)
        feature_counts = np.bincount(feature_gate_bins, minlength=len(self.train_feature_gate_edges) - 1).astype(float)
        self.train_feature_gate_strata_probs = feature_counts / max(feature_counts.sum(), 1.0)

        gate_probs = self._prob_geomean_for_df(self.train_df, columns=self.privacy_columns)
        self.train_gate_probs = gate_probs
        self.train_gate_edges = _quantile_edges(gate_probs, n_bins=self.rarity_strata)
        gate_bins = self._assign_bins_from_edges(gate_probs, self.train_gate_edges)
        counts = np.bincount(gate_bins, minlength=len(self.train_gate_edges) - 1).astype(float)
        probs = counts / max(counts.sum(), 1.0)
        self.train_gate_strata_probs = probs

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
        gate_bucket_indices = self._column_bucket_indices_for_df(normalized_df, self.privacy_columns)
        gate_probs = self._prob_geomean_from_bucket_indices(gate_bucket_indices, self.privacy_columns)
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
        finite_scores = scores[np.isfinite(scores)]
        return float(np.mean(finite_scores)) if finite_scores.size else 0.0

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
        finite_scores = scores[np.isfinite(scores)]
        return float(np.mean(finite_scores)) if finite_scores.size else 0.0
