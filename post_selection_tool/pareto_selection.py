from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import drop_constant_objectives, non_dominated_sort_report
from .encoding import _minmax_normalize


PARETO_AUX_FRONT_MAX_ROWS = 16_000


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

def _expand_unique_fronts(unique_fronts: list[list[int]], inverse: np.ndarray) -> list[list[int]]:
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

def _non_dominated_sort_exact(points: np.ndarray) -> list[list[int]]:
    sort_points = drop_constant_objectives(points)
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
            return _expand_unique_fronts(unique_fronts, inverse)
    if sort_points.ndim == 2 and sort_points.shape[1] == 2:
        return _non_dominated_sort_2d(sort_points)
    if sort_points.ndim == 2 and sort_points.shape[1] == 3:
        return _non_dominated_sort_3d(sort_points)
    return _non_dominated_sort_generic(sort_points)

def _non_dominated_sort(points: np.ndarray) -> list[list[int]]:
    sort_points = drop_constant_objectives(points)
    if sort_points.ndim == 2 and sort_points.shape[1] == 0:
        return [list(range(len(sort_points)))] if len(sort_points) else []
    return _non_dominated_sort_exact(sort_points)

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



class ParetoSelectionMixin:
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
            self._last_non_dominated_sort_report = non_dominated_sort_report(points, fronts)
        else:
            self._last_non_dominated_sort_report = non_dominated_sort_report(points, fronts)
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

    def _selection_objective_components(
        self,
        exact_records: list[dict[str, Any]],
        *,
        mode: str,
        fidelity_1d_weight: float = 0.25,
        fidelity_2d_weight: float = 0.25,
        privacy_weight: float = 0.5,
        utility_proxy_weight: float = 0.0,
    ) -> dict[str, Any]:
        num_records = len(exact_records)
        if num_records == 0:
            empty = np.zeros(0, dtype=float)
            return {
                "objective_score": empty,
                "fidelity_1d_component": empty,
                "fidelity_2d_component": empty,
                "privacy_component": empty,
                "normalized_utility_proxy": empty,
                "mode": mode,
            }

        if mode == "scalar_naive":
            fidelity_raw = np.asarray([float(record.get("fid_marginal", 0.0)) for record in exact_records], dtype=float)
            privacy_raw = np.asarray([float(record.get("privacy_score_v1", 0.0)) for record in exact_records], dtype=float)
            fidelity_component = _minmax_normalize(fidelity_raw)
            privacy_component = _minmax_normalize(privacy_raw)
            normalized_utility_proxy = np.zeros_like(fidelity_component, dtype=float)
            objective_score = (fidelity_1d_weight + fidelity_2d_weight) * fidelity_component + privacy_weight * privacy_component
            return {
                "objective_score": objective_score,
                "fidelity_1d_component": fidelity_component,
                "fidelity_2d_component": fidelity_component,
                "privacy_component": privacy_component,
                "normalized_utility_proxy": normalized_utility_proxy,
                "mode": mode,
            }

        fidelity_1d_raw = np.asarray([float(record.get("pareto_fid_1d_obj", 0.0)) for record in exact_records], dtype=float)
        fidelity_2d_raw = np.asarray([float(record.get("pareto_fid_2d_obj", 0.0)) for record in exact_records], dtype=float)
        privacy_raw = np.asarray([float(record.get("pareto_priv_obj", 0.0)) for record in exact_records], dtype=float)
        raw_utility_proxy = self._pareto_util_values(exact_records)

        def _unit_component(values: np.ndarray) -> np.ndarray:
            array = np.asarray(values, dtype=float)
            array = np.where(np.isfinite(array), array, 0.0)
            return np.clip(array, 0.0, 1.0)

        fidelity_1d_component = _unit_component(fidelity_1d_raw)
        fidelity_2d_component = _unit_component(fidelity_2d_raw)
        privacy_component = _unit_component(privacy_raw)
        normalized_utility_proxy = _unit_component(raw_utility_proxy)
        weights = np.asarray(
            [fidelity_1d_weight, fidelity_2d_weight, privacy_weight, utility_proxy_weight],
            dtype=float,
        )
        weights = np.where(weights <= 0.0, 1e-6, weights)

        if mode == "scalar_matched":
            objective_score = (
                weights[0] * fidelity_1d_component
                + weights[1] * fidelity_2d_component
                + weights[2] * privacy_component
                + weights[3] * normalized_utility_proxy
            )
            return {
                "objective_score": objective_score,
                "fidelity_1d_component": fidelity_1d_component,
                "fidelity_2d_component": fidelity_2d_component,
                "privacy_component": privacy_component,
                "normalized_utility_proxy": normalized_utility_proxy,
                "mode": mode,
            }

        points = np.column_stack([fidelity_1d_raw, fidelity_2d_raw, privacy_raw, raw_utility_proxy])
        if mode == "chebyshev":
            normalized_points = np.column_stack(
                [fidelity_1d_component, fidelity_2d_component, privacy_component, normalized_utility_proxy]
            )
            reference = normalized_points.max(axis=0)
            chebyshev_scores = np.max(weights[None, :] * (reference[None, :] - normalized_points), axis=1)
            objective_score = 1.0 - _minmax_normalize(chebyshev_scores)
            objective_score = 0.85 * objective_score + 0.10 * privacy_component + 0.05 * normalized_utility_proxy
            return {
                "objective_score": objective_score,
                "fidelity_1d_component": fidelity_1d_component,
                "fidelity_2d_component": fidelity_2d_component,
                "privacy_component": privacy_component,
                "normalized_utility_proxy": normalized_utility_proxy,
                "chebyshev_scores": chebyshev_scores,
                "mode": mode,
            }

        if num_records <= PARETO_AUX_FRONT_MAX_ROWS:
            fronts = _non_dominated_sort(points)
            self._last_objective_front_report = non_dominated_sort_report(points, fronts)
            front_rank, crowding = self._front_rank_and_crowding(points, fronts)
            front_component = 1.0 - _minmax_normalize(front_rank)
            crowding_component = np.zeros_like(crowding)
            finite_mask = np.isfinite(crowding)
            if np.any(finite_mask):
                crowding_component[finite_mask] = _minmax_normalize(crowding[finite_mask])
                crowding_component[~finite_mask] = 1.0
            else:
                crowding_component.fill(1.0)
            front_component_mode = "deterministic_exact_front_rank"
        else:
            fronts = []
            front_component = np.zeros(num_records, dtype=float)
            crowding_component = np.zeros(num_records, dtype=float)
            front_component_mode = "skipped_large_constrained_auxiliary_objective"
            self._last_objective_front_report = {
                "algorithm": "not_run",
                "exact": None,
                "approximate": False,
                "rows": int(num_records),
                "reason": "large_pool_constrained_objective_uses_calibrated_components_without_random_front_approximation",
                "threshold": int(PARETO_AUX_FRONT_MAX_ROWS),
            }

        objective_score = (
            0.22 * privacy_component
            + 0.18 * fidelity_1d_component
            + 0.18 * fidelity_2d_component
            + 0.22 * normalized_utility_proxy
            + 0.25 * front_component
            + 0.05 * crowding_component
        )
        return {
            "objective_score": objective_score,
            "fidelity_1d_component": fidelity_1d_component,
            "fidelity_2d_component": fidelity_2d_component,
            "privacy_component": privacy_component,
            "normalized_utility_proxy": normalized_utility_proxy,
            "front_component": front_component,
            "crowding_component": crowding_component,
            "fronts": fronts,
            "front_component_mode": front_component_mode,
            "front_report": getattr(self, "_last_objective_front_report", {}),
            "mode": mode,
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
            utility_proxy_weight=utility_weight,
        )
        scalar_scores = np.asarray(objective["objective_score"], dtype=float)
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
        allow_reference_anchor: bool = True,
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
                allow_reference_anchor=allow_reference_anchor,
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
            "non_dominated_sort": (
                floor_repair_report.get("objective_front_report", {})
                if floor_reference is not None
                else getattr(self, "_last_non_dominated_sort_report", {})
            ),
            "front_component_mode": (
                floor_repair_report.get("front_component_mode")
                if floor_reference is not None
                else "deterministic_exact_nsga_front_rank"
            ),
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
