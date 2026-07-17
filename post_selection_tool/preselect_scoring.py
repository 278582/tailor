from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .encoding import _progress, _rank_normalize


class PreselectScoringMixin:
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
        fidelity_bucket_indices = self._column_bucket_indices_for_df(normalized_df, self.fidelity_bucket_columns)
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
                "holdout_gap": (
                    privacy_df["nn_distance_holdout"].to_numpy(dtype=float, copy=False)
                    - privacy_df["nn_distance_train"].to_numpy(dtype=float, copy=False)
                ),
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
        fidelity_bucket_indices = self._column_bucket_indices_for_df(candidate_df, self.fidelity_bucket_columns)
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
        fidelity_bucket_indices = self._column_bucket_indices_for_df(candidate_df, self.fidelity_bucket_columns)
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
