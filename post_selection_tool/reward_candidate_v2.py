from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .io import records_to_df


def _finite_float(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(record.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _component_array(records: list[dict[str, Any]], key: str, default: float = 0.0) -> np.ndarray:
    return np.asarray([_finite_float(record, key, default) for record in records], dtype=float)


def _unit_array(records: list[dict[str, Any]], key: str, default: float = 0.0) -> np.ndarray:
    return np.clip(_component_array(records, key, default), 0.0, 1.0)


def _shifted_geomean_reward(
    *,
    shape: float,
    trend: float,
    dcr_privacy: float,
    utility: float,
    rho: float = 0.05,
) -> float:
    scores = np.clip(np.asarray([shape, trend, dcr_privacy, utility], dtype=float), 0.0, 1.0)
    shifted = (scores + float(rho)) / (1.0 + float(rho))
    raw = float(np.prod(np.power(shifted, 0.25)))
    floor = float(rho) / (1.0 + float(rho))
    denom = max(1.0 - floor, 1e-12)
    return float(np.clip((raw - floor) / denom, 0.0, 1.0))


def _candidate_id(record: dict[str, Any], fallback: int) -> int:
    try:
        return int(record.get("candidate_id", fallback))
    except (TypeError, ValueError):
        return int(fallback)


def _selected_mask(
    preselected_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
) -> np.ndarray:
    selected_ids = {_candidate_id(record, idx) for idx, record in enumerate(selected_records)}
    return np.asarray(
        [_candidate_id(record, idx) in selected_ids for idx, record in enumerate(preselected_records)],
        dtype=bool,
    )


def _dcr_proxy_stats(mask: np.ndarray, holdout_gap: np.ndarray) -> dict[str, float]:
    if mask.size == 0 or not np.any(mask):
        return {"dcr_proxy": 0.5, "dcr_privacy_proxy": 1.0, "real_closer_rows": 0.0}
    selected_gap = holdout_gap[mask]
    finite = np.isfinite(selected_gap)
    if not np.any(finite):
        return {"dcr_proxy": 0.5, "dcr_privacy_proxy": 1.0, "real_closer_rows": 0.0}
    real_closer = selected_gap[finite] >= 0.0
    dcr_proxy = float(np.mean(real_closer))
    return {
        "dcr_proxy": dcr_proxy,
        "dcr_privacy_proxy": float(1.0 - abs(dcr_proxy - 0.5)),
        "real_closer_rows": float(np.sum(real_closer)),
    }


def _subset_proxy_stats(
    mask: np.ndarray,
    *,
    fidelity_1d: np.ndarray,
    fidelity_2d: np.ndarray,
    pareto_privacy: np.ndarray,
    utility: np.ndarray,
    holdout_gap: np.ndarray,
) -> dict[str, float]:
    if mask.size == 0 or not np.any(mask):
        return {
            "shape_proxy": 0.0,
            "trend_proxy": 0.0,
            "pareto_privacy_mean": 0.0,
            "utility_proxy": 0.0,
            "dcr_proxy": 0.5,
            "dcr_privacy_proxy": 1.0,
            "reward_proxy": 0.0,
        }
    dcr_stats = _dcr_proxy_stats(mask, holdout_gap)
    shape = float(np.mean(fidelity_1d[mask]))
    trend = float(np.mean(fidelity_2d[mask]))
    util = float(np.mean(utility[mask]))
    dcr_privacy = float(dcr_stats["dcr_privacy_proxy"])
    return {
        "shape_proxy": shape,
        "trend_proxy": trend,
        "pareto_privacy_mean": float(np.mean(pareto_privacy[mask])),
        "utility_proxy": util,
        "dcr_proxy": float(dcr_stats["dcr_proxy"]),
        "dcr_privacy_proxy": dcr_privacy,
        "reward_proxy": _shifted_geomean_reward(
            shape=shape,
            trend=trend,
            dcr_privacy=dcr_privacy,
            utility=util,
        ),
    }


def _subset_proxy_stats_from_indices(
    indices: np.ndarray,
    *,
    fidelity_1d: np.ndarray,
    fidelity_2d: np.ndarray,
    pareto_privacy: np.ndarray,
    utility: np.ndarray,
    holdout_gap: np.ndarray,
) -> dict[str, float]:
    if indices.size == 0:
        return {
            "shape_proxy": 0.0,
            "trend_proxy": 0.0,
            "pareto_privacy_mean": 0.0,
            "utility_proxy": 0.0,
            "dcr_proxy": 0.5,
            "dcr_privacy_proxy": 1.0,
            "reward_proxy": 0.0,
        }
    selected_gap = holdout_gap[indices]
    finite = np.isfinite(selected_gap)
    if np.any(finite):
        dcr_proxy = float(np.mean(selected_gap[finite] >= 0.0))
    else:
        dcr_proxy = 0.5
    dcr_privacy = float(1.0 - abs(dcr_proxy - 0.5))
    shape = float(np.mean(fidelity_1d[indices]))
    trend = float(np.mean(fidelity_2d[indices]))
    util = float(np.mean(utility[indices]))
    return {
        "shape_proxy": shape,
        "trend_proxy": trend,
        "pareto_privacy_mean": float(np.mean(pareto_privacy[indices])),
        "utility_proxy": util,
        "dcr_proxy": dcr_proxy,
        "dcr_privacy_proxy": dcr_privacy,
        "reward_proxy": _shifted_geomean_reward(
            shape=shape,
            trend=trend,
            dcr_privacy=dcr_privacy,
            utility=util,
        ),
    }


def _candidate_sizes(target_swaps: int, max_swaps: int, max_candidates: int) -> list[int]:
    if max_swaps <= 0:
        return []
    target = max(1, min(int(target_swaps), int(max_swaps)))
    raw_sizes = {
        1,
        target,
        max(1, int(round(0.5 * target))),
        max(1, int(round(0.75 * target))),
        max(1, int(round(1.25 * target))),
        max(1, int(round(1.5 * target))),
        max_swaps,
    }
    for fraction in (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12):
        raw_sizes.add(max(1, min(max_swaps, int(round(max_swaps * fraction / 0.16)))))
    sizes = sorted({min(max_swaps, size) for size in raw_sizes if size > 0})
    if len(sizes) <= max(1, int(max_candidates)):
        return sizes
    keep = np.linspace(0, len(sizes) - 1, max(1, int(max_candidates)), dtype=int)
    return sorted({sizes[int(idx)] for idx in keep})


def refine_selection_for_reward_v2(
    *,
    preselected_records: list[dict[str, Any]],
    exact_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    keep_k: int,
    column_order: list[str],
    max_swap_fraction: float = 0.16,
    max_candidate_sizes: int = 10,
    min_proxy_delta: float = 0.0,
    fidelity_floor_eps: float = 0.015,
    utility_floor_eps: float = 0.02,
    dcr_proxy_max_balance_error: float | None = None,
    allow_duplicate_adds: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Fast proxy refinement for the final post-selection candidate.

    This uses only row-level exact/proxy fields already computed during
    post-selection. It does not run density, DCR, or utility evaluation.
    """

    dcr_guard_enabled = dcr_proxy_max_balance_error is not None
    dcr_guard_limit = (
        max(0.0, float(dcr_proxy_max_balance_error))
        if dcr_proxy_max_balance_error is not None
        else None
    )
    effective_min_proxy_delta = float(min_proxy_delta)
    dcr_guarded_duplicate_mode = bool(dcr_guard_enabled and allow_duplicate_adds)
    if dcr_guarded_duplicate_mode:
        effective_min_proxy_delta = min(effective_min_proxy_delta, -0.0035)
    report_base: dict[str, Any] = {
        "enabled": True,
        "version": "reward_candidate_v2",
        "selection_signal": "holdout_gap_dcr_balance_proxy",
        "full_eval_used": False,
        "allow_duplicate_adds": bool(allow_duplicate_adds),
        "effective_min_proxy_delta": float(effective_min_proxy_delta),
        "dcr_guarded_duplicate_mode": bool(dcr_guarded_duplicate_mode),
        "dcr_proxy_guard": {
            "enabled": bool(dcr_guard_enabled),
            "max_balance_error": dcr_guard_limit,
            "target": 0.5,
        },
    }
    if not preselected_records or not exact_records or not selected_records or keep_k <= 0:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {**report_base, "applied": False, "reason": "empty_inputs"},
        )
    if len(preselected_records) != len(exact_records):
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {**report_base, "applied": False, "reason": "record_length_mismatch"},
        )

    selected_mask = _selected_mask(preselected_records, selected_records)
    if int(selected_mask.sum()) != min(int(keep_k), len(preselected_records)):
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "selected_mask_size_mismatch",
                "selected_mask_rows": int(selected_mask.sum()),
                "keep_k": int(keep_k),
            },
        )

    fidelity_1d = _unit_array(exact_records, "pareto_fid_1d_obj")
    fidelity_2d = _unit_array(exact_records, "pareto_fid_2d_obj")
    pareto_privacy = _unit_array(exact_records, "pareto_priv_obj")
    utility = _unit_array(exact_records, "pareto_util_proxy_obj")
    holdout_gap = _component_array(exact_records, "holdout_gap", default=0.0)
    if not np.any(np.isfinite(holdout_gap)):
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {**report_base, "applied": False, "reason": "missing_holdout_gap"},
        )

    base_stats = _subset_proxy_stats(
        selected_mask,
        fidelity_1d=fidelity_1d,
        fidelity_2d=fidelity_2d,
        pareto_privacy=pareto_privacy,
        utility=utility,
        holdout_gap=holdout_gap,
    )
    reduce_real_closer = bool(base_stats["dcr_proxy"] >= 0.5)
    row_is_real_closer = holdout_gap >= 0.0
    remove_idx = np.flatnonzero(selected_mask & (row_is_real_closer == reduce_real_closer))
    if bool(allow_duplicate_adds):
        add_idx = np.flatnonzero(row_is_real_closer != reduce_real_closer)
    else:
        add_idx = np.flatnonzero((~selected_mask) & (row_is_real_closer != reduce_real_closer))
    if remove_idx.size == 0 or add_idx.size == 0:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "no_directional_swap_candidates",
                "base": base_stats,
            },
        )

    composite = 0.25 * fidelity_1d + 0.25 * fidelity_2d + 0.20 * pareto_privacy + 0.30 * utility
    gap_strength = np.abs(holdout_gap)
    remove_priority = gap_strength + 0.35 * (1.0 - composite) + 0.15 * (1.0 - utility)
    add_priority = gap_strength + 0.35 * composite + 0.25 * utility + 0.10 * pareto_privacy
    remove_order = remove_idx[np.argsort(-remove_priority[remove_idx], kind="mergesort")]
    add_order = add_idx[np.argsort(-add_priority[add_idx], kind="mergesort")]

    selected_positions = np.flatnonzero(selected_mask).astype(np.int64, copy=False)
    selected_pos_by_index = {int(idx): pos for pos, idx in enumerate(selected_positions.tolist())}
    add_capacity = int(add_order.size)
    if bool(allow_duplicate_adds) and add_order.size > 0:
        add_capacity = max(
            int(add_order.size),
            max(1, int(round(float(keep_k) * max(0.0, float(max_swap_fraction))))),
        )
    max_swaps = min(
        int(remove_order.size),
        int(add_capacity),
        max(1, int(round(float(keep_k) * max(0.0, float(max_swap_fraction))))),
    )
    target_swaps = int(round(abs(float(base_stats["dcr_proxy"]) - 0.5) * float(keep_k)))
    sizes = _candidate_sizes(target_swaps=target_swaps, max_swaps=max_swaps, max_candidates=max_candidate_sizes)
    if not sizes:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {**report_base, "applied": False, "reason": "no_candidate_sizes", "base": base_stats},
        )

    candidates: list[dict[str, Any]] = []
    for size in sizes:
        if bool(allow_duplicate_adds):
            trial_indices = selected_positions.copy()
            remove_slice = remove_order[:size]
            add_slice = np.resize(add_order, int(size))
            replace_positions = np.asarray(
                [selected_pos_by_index[int(idx)] for idx in remove_slice],
                dtype=np.int64,
            )
            trial_indices[replace_positions] = add_slice
            stats = _subset_proxy_stats_from_indices(
                trial_indices,
                fidelity_1d=fidelity_1d,
                fidelity_2d=fidelity_2d,
                pareto_privacy=pareto_privacy,
                utility=utility,
                holdout_gap=holdout_gap,
            )
        else:
            trial_mask = selected_mask.copy()
            trial_mask[remove_order[:size]] = False
            trial_mask[add_order[:size]] = True
            stats = _subset_proxy_stats(
                trial_mask,
                fidelity_1d=fidelity_1d,
                fidelity_2d=fidelity_2d,
                pareto_privacy=pareto_privacy,
                utility=utility,
                holdout_gap=holdout_gap,
            )
        floor_ok = bool(
            stats["shape_proxy"] >= base_stats["shape_proxy"] - float(fidelity_floor_eps)
            and stats["trend_proxy"] >= base_stats["trend_proxy"] - float(fidelity_floor_eps)
            and stats["utility_proxy"] >= base_stats["utility_proxy"] - float(utility_floor_eps)
        )
        dcr_guard_ok = True
        if dcr_guard_limit is not None:
            dcr_guard_ok = bool(abs(float(stats["dcr_proxy"]) - 0.5) <= float(dcr_guard_limit))
        candidates.append(
            {
                "swaps": int(size),
                "stats": stats,
                "floor_ok": floor_ok,
                "dcr_guard_ok": dcr_guard_ok,
                "reward_delta": float(stats["reward_proxy"] - base_stats["reward_proxy"]),
                "dcr_privacy_delta": float(stats["dcr_privacy_proxy"] - base_stats["dcr_privacy_proxy"]),
                "utility_delta": float(stats["utility_proxy"] - base_stats["utility_proxy"]),
            }
        )

    feasible = [
        item
        for item in candidates
        if bool(item["floor_ok"])
        and bool(item["dcr_guard_ok"])
        and float(item["reward_delta"]) > float(effective_min_proxy_delta)
    ]
    if not feasible:
        return (
            records_to_df(selected_records, column_order),
            selected_records,
            {
                **report_base,
                "applied": False,
                "reason": "no_feasible_proxy_improvement",
                "base": base_stats,
                "candidates": candidates,
            },
        )

    if dcr_guarded_duplicate_mode:
        best = max(
            feasible,
            key=lambda item: (
                float(item["stats"]["dcr_privacy_proxy"]),
                float(item["stats"]["reward_proxy"]),
                float(item["stats"]["utility_proxy"]),
            ),
        )
    else:
        best = max(
            feasible,
            key=lambda item: (
                float(item["stats"]["reward_proxy"]),
                float(item["stats"]["dcr_privacy_proxy"]),
                float(item["stats"]["utility_proxy"]),
            ),
        )
    best_size = int(best["swaps"])
    if bool(allow_duplicate_adds):
        final_indices_array = selected_positions.copy()
        remove_slice = remove_order[:best_size]
        add_slice = np.resize(add_order, int(best_size))
        replace_positions = np.asarray(
            [selected_pos_by_index[int(idx)] for idx in remove_slice],
            dtype=np.int64,
        )
        final_indices_array[replace_positions] = add_slice
        final_indices = final_indices_array.astype(int, copy=False).tolist()
        added_indices_for_report = add_slice.astype(int, copy=False).tolist()
    else:
        final_mask = selected_mask.copy()
        final_mask[remove_order[:best_size]] = False
        final_mask[add_order[:best_size]] = True
        final_indices = np.flatnonzero(final_mask).astype(int, copy=False).tolist()
        added_indices_for_report = add_order[:best_size].astype(int, copy=False).tolist()
    final_records = [preselected_records[idx] for idx in final_indices]
    final_df = records_to_df(final_records, column_order)
    return (
        final_df,
        final_records,
        {
            **report_base,
            "applied": True,
            "direction": "reduce_real_closer" if reduce_real_closer else "increase_real_closer",
            "selected_swaps": best_size,
            "base": base_stats,
            "best": best,
            "candidate_sizes": sizes,
            "candidate_count": len(candidates),
            "remove_pool_rows": int(remove_order.size),
            "add_pool_rows": int(add_order.size),
            "duplicate_add_rows_allowed": bool(allow_duplicate_adds),
            "unique_added_rows": int(len(set(added_indices_for_report))),
            "removed_candidate_ids": [
                _candidate_id(preselected_records[idx], idx) for idx in remove_order[:best_size].tolist()
            ],
            "added_candidate_ids": [
                _candidate_id(preselected_records[idx], idx) for idx in added_indices_for_report
            ],
        },
    )
