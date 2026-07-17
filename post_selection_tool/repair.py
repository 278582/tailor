from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .encoding import _progress, _progress_write, _rank_normalize


class RepairMixin:
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

        group_iter = _progress(
            range(len(group_bounds) - 1),
            total=len(group_bounds) - 1,
            desc="signature swaps",
            disable=not bool(getattr(self, "progress_enabled", True)),
        )
        for group_pos in group_iter:
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
            if hasattr(group_iter, "set_postfix"):
                group_iter.set_postfix(swaps=applied_swaps, budget=max_swap_budget)

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
        allow_reference_anchor: bool = True,
    ) -> tuple[list[int], dict[str, Any]]:
        if not preselected_records or not exact_records or keep_k <= 0:
            return [], {"applied": False, "mode": "empty"}
        if floor_reference is None:
            return [], {"applied": False, "mode": "disabled_no_floor_reference"}

        keep_k = min(int(keep_k), len(preselected_records))
        _progress_write(
            "constrained subset start "
            f"mode={mode} rows={len(preselected_records)} keep_k={keep_k} floor_mode={floor_mode}"
        )
        _progress_write("constrained subset: build bucket/pair state")
        bucket_indices, pair_codes = self._bucket_pair_state_for_records(preselected_records)
        _progress_write(
            "constrained subset: bucket/pair state ready "
            f"columns={len(bucket_indices)} pair_edges={len(pair_codes)}"
        )
        shape_constraint_enabled = bool(constraint_reference_records) and bool(
            floor_reference.get("enforce_reference_shape", False)
        )
        if shape_constraint_enabled:
            target_counts_1d, target_counts_2d = self._reference_target_counts_from_records(constraint_reference_records)
        else:
            target_counts_1d, target_counts_2d = {}, []
        _progress_write("constrained subset: build objective components")
        objective = self._selection_objective_components(
            exact_records,
            mode=mode,
            fidelity_1d_weight=fidelity_1d_weight,
            fidelity_2d_weight=fidelity_2d_weight,
            privacy_weight=privacy_weight,
            utility_proxy_weight=utility_weight,
        )
        objective_score = np.asarray(objective["objective_score"], dtype=float)
        fidelity_1d_component = np.asarray(objective["fidelity_1d_component"], dtype=float)
        fidelity_2d_component = np.asarray(objective["fidelity_2d_component"], dtype=float)
        privacy_component = np.asarray(objective["privacy_component"], dtype=float)
        normalized_utility_proxy = np.asarray(
            objective.get("normalized_utility_proxy", np.zeros(len(exact_records), dtype=float)),
            dtype=float,
        )
        _progress_write("constrained subset: objective components ready")

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
            order = selected_indices[np.argsort(objective_score[selected_indices], kind="mergesort")]
            selected_mask[order[:remove_count]] = False
        elif 0 < initial_reference_rows < keep_k:
            remaining_indices = np.flatnonzero(~selected_mask)
            if remaining_indices.size > 0:
                order = remaining_indices[np.argsort(-objective_score[remaining_indices], kind="mergesort")]
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
            util_mean = float(normalized_utility_proxy[mask].mean())
            objective_mean = float(objective_score[mask].mean())
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
        if bool(allow_reference_anchor) and int(reference_mask.sum()) == keep_k and reference_available:
            init_masks.append(("reference_anchor", reference_mask.copy()))
        all_indices = np.arange(len(preselected_records), dtype=int)
        composite_score = (
            0.30 * privacy_component
            + 0.30 * normalized_utility_proxy
            + 0.20 * fidelity_1d_component
            + 0.20 * fidelity_2d_component
        )
        refine_band_score = 0.55 * privacy_component + 0.45 * normalized_utility_proxy
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
                + 0.34 * normalized_utility_proxy[band_order]
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
            current_utility = float(normalized_utility_proxy[selected_mask].mean()) if np.any(selected_mask) else 0.0
            current_quality = 0.25 * current_fid_1d + 0.25 * current_fid_2d + 0.25 * current_privacy + 0.25 * current_utility

            batch_size = max(32, min(512, int(round(0.02 * keep_k))))
            rounds_applied = 0
            max_rounds = 6

            repair_iter = _progress(
                range(max_rounds),
                total=max_rounds,
                desc="floor repair",
                disable=not bool(getattr(self, "progress_enabled", True)),
            )
            for _ in repair_iter:
                if current_fid_1d >= target_fid_1d and current_fid_2d >= target_fid_2d:
                    break
                if hasattr(repair_iter, "set_postfix"):
                    repair_iter.set_postfix(fid1=f"{current_fid_1d:.4f}", fid2=f"{current_fid_2d:.4f}")

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
                    - 0.10 * normalized_utility_proxy
                    - 0.08 * privacy_component
                    - 0.06 * fidelity_1d_component
                    - 0.06 * fidelity_2d_component
                )
                add_priority = (
                    add_score
                    + 0.12 * normalized_utility_proxy
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
                        (normalized_utility_proxy[add_batch].sum() - normalized_utility_proxy[remove_batch].sum())
                        / max(float(keep_k), 1.0)
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
            current_utility_mean = float(normalized_utility_proxy[selected_mask].mean()) if np.any(selected_mask) else 0.0
            current_objective_mean = float(objective_score[selected_mask].mean()) if np.any(selected_mask) else 0.0
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

            refine_iter = _progress(
                range(max_rounds),
                total=max_rounds,
                desc="local refine",
                disable=not bool(getattr(self, "progress_enabled", True)),
            )
            for _ in refine_iter:
                selected_idx = np.flatnonzero(selected_mask)
                available_idx = np.flatnonzero(~selected_mask)
                if selected_idx.size == 0 or available_idx.size == 0:
                    break
                selected_band_size = min(
                    int(selected_idx.size),
                    max(512, int(16 * base_batch_size), int(round(0.20 * float(selected_idx.size)))),
                )
                available_band_size = min(
                    int(available_idx.size),
                    max(1024, int(32 * base_batch_size), int(round(0.20 * float(available_idx.size)))),
                )
                selected_idx = selected_idx[
                    np.argsort(refine_band_score[selected_idx], kind="mergesort")[:selected_band_size]
                ]
                available_idx = available_idx[
                    np.argsort(-refine_band_score[available_idx], kind="mergesort")[:available_band_size]
                ]
                if hasattr(refine_iter, "set_postfix"):
                    refine_iter.set_postfix(
                        fid1=f"{current_fid_1d:.4f}",
                        fid2=f"{current_fid_2d:.4f}",
                        accepted=accepted_rounds,
                        sel_band=selected_band_size,
                        add_band=available_band_size,
                    )

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
                    + 0.18 * (1.0 - normalized_utility_proxy)
                    + 0.14 * (1.0 - privacy_component)
                    + 0.18 * remove_support_1d
                    + 0.18 * remove_support_2d
                )
                add_priority = (
                    0.32 * composite_score
                    + 0.20 * normalized_utility_proxy
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
                        (normalized_utility_proxy[add_batch].sum() - normalized_utility_proxy[remove_batch].sum())
                        / max(float(keep_k), 1.0)
                    )
                    delta_objective_mean = float(
                        (objective_score[add_batch].sum() - objective_score[remove_batch].sum())
                        / max(float(keep_k), 1.0)
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
                "top_band_refine": {
                    "enabled": True,
                    "score": "0.55 * privacy_component + 0.45 * normalized_utility_proxy",
                    "selected_band": "bottom 20% with minimum max(512, 16 * batch_size)",
                    "available_band": "top 20% with minimum max(1024, 32 * batch_size)",
                },
                "current_fid_1d": float(current_fid_1d),
                "current_fid_2d": float(current_fid_2d),
                "score": float(current_score),
            }

        candidate_solutions: list[dict[str, Any]] = []
        init_iter = _progress(
            init_masks,
            total=len(init_masks),
            desc="constrained candidates",
            disable=not bool(getattr(self, "progress_enabled", True)),
        )
        for init_name, init_mask in init_iter:
            if hasattr(init_iter, "set_postfix"):
                init_iter.set_postfix(init=init_name)
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
        _progress_write(
            "constrained subset: best candidate selected "
            f"init={best_solution['init']} satisfied={best_solution['satisfied']}"
        )

        selected_mask = np.asarray(best_solution["mask"], dtype=bool)
        pre_signature_mask = selected_mask.copy()
        _progress_write("constrained subset: signature swaps start")
        selected_mask, signature_refine_report = self._refine_subset_with_signature_swaps(
            selected_mask=selected_mask,
            bucket_indices=bucket_indices,
            privacy_component=privacy_component,
            utility_component=normalized_utility_proxy,
        )
        _progress_write(
            "constrained subset: signature swaps done "
            f"swaps={signature_refine_report.get('swaps', 0)}"
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
        _progress_write(
            "constrained subset done "
            f"selected={len(final_indices)} fid1={current_fid_1d:.4f} fid2={current_fid_2d:.4f}"
        )
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
            "reference_anchor_allowed": bool(allow_reference_anchor),
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
            "objective_front_report": dict(objective.get("front_report", {})),
            "front_component_mode": objective.get("front_component_mode"),
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
            ceiling_mode = "fidelity_ceiling_anchor_v6_static_second_pass_utility"
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
