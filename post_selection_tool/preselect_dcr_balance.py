from __future__ import annotations

from typing import Any

import numpy as np


def _candidate_id(record: dict[str, Any], fallback: int) -> int:
    try:
        return int(record.get("candidate_id", fallback))
    except (TypeError, ValueError):
        return int(fallback)


def _finite_float(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(record.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _holdout_gap(record: dict[str, Any]) -> float:
    if record.get("holdout_gap") is not None:
        return _finite_float(record, "holdout_gap")
    return _finite_float(record, "nn_distance_holdout") - _finite_float(record, "nn_distance_train")


def _surrogate_quality(record: dict[str, Any]) -> float:
    stage = _finite_float(record, "s_preselect_stage_b", _finite_float(record, "s_preselect_band"))
    fid1 = _finite_float(record, "s_pareto_fid_1d_sur", _finite_float(record, "s_fid_sur_1d_rank"))
    fid2 = _finite_float(record, "s_pareto_fid_2d_sur", _finite_float(record, "s_fid_sur_2d_rank"))
    support = _finite_float(record, "s_preselect_support_tiebreak")
    privacy = _finite_float(record, "s_preselect_priv_tiebreak")
    return float(0.30 * stage + 0.25 * fid1 + 0.25 * fid2 + 0.12 * support + 0.08 * privacy)


def rebalance_preselected_for_dcr_surrogate(
    *,
    pool_records: list[dict[str, Any]],
    surrogate_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    selected_surrogates: list[dict[str, Any]],
    target_fraction: float = 0.50,
    max_exchange_fraction: float = 0.30,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep more holdout-gap minority candidates before exact scoring.

    The final metric rewards DCR close to 0.5. Preselect can otherwise discard
    most rows from the minority holdout-gap direction, leaving final selection
    unable to rebalance privacy without running a full evaluation.
    """

    report_base: dict[str, Any] = {
        "enabled": True,
        "version": "preselect_dcr_balance_v3",
        "selection_signal": "surrogate_holdout_gap_direction",
        "full_eval_used": False,
    }
    if not pool_records or not surrogate_records or not selected_records:
        return (
            selected_records,
            selected_surrogates,
            {**report_base, "applied": False, "reason": "empty_inputs"},
        )
    if len(pool_records) != len(surrogate_records):
        return (
            selected_records,
            selected_surrogates,
            {**report_base, "applied": False, "reason": "pool_surrogate_length_mismatch"},
        )

    total_rows = len(pool_records)
    selected_count = len(selected_records)
    pool_ids = np.asarray([_candidate_id(record, idx) for idx, record in enumerate(pool_records)], dtype=int)
    surrogate_by_id = {
        _candidate_id(record, idx): record
        for idx, record in enumerate(surrogate_records)
    }
    selected_ids = {_candidate_id(record, idx) for idx, record in enumerate(selected_records)}
    selected_mask = np.asarray([candidate_id in selected_ids for candidate_id in pool_ids], dtype=bool)
    if int(selected_mask.sum()) != selected_count:
        return (
            selected_records,
            selected_surrogates,
            {
                **report_base,
                "applied": False,
                "reason": "selected_mask_size_mismatch",
                "selected_rows": int(selected_count),
                "matched_rows": int(selected_mask.sum()),
            },
        )

    holdout_gap = np.asarray([_holdout_gap(record) for record in surrogate_records], dtype=float)
    finite = np.isfinite(holdout_gap)
    if not np.any(finite):
        return (
            selected_records,
            selected_surrogates,
            {**report_base, "applied": False, "reason": "missing_holdout_gap"},
        )

    real_closer = holdout_gap >= 0.0
    full_real = int(np.count_nonzero(real_closer))
    full_holdout = int(total_rows - full_real)
    minority_real_closer = bool(full_real <= full_holdout)
    minority_mask = real_closer == minority_real_closer
    majority_mask = ~minority_mask
    full_minority = int(np.count_nonzero(minority_mask))
    selected_minority = int(np.count_nonzero(selected_mask & minority_mask))
    selected_majority = int(np.count_nonzero(selected_mask & majority_mask))

    target_minority = min(
        full_minority,
        int(round(float(selected_count) * float(np.clip(target_fraction, 0.0, 1.0)))),
    )
    needed = max(0, target_minority - selected_minority)
    unselected_minority = np.flatnonzero((~selected_mask) & minority_mask)
    removable_majority = np.flatnonzero(selected_mask & majority_mask)
    max_exchange = max(0, int(round(float(selected_count) * max(0.0, float(max_exchange_fraction)))))
    exchange = min(needed, int(unselected_minority.size), int(removable_majority.size), max_exchange)
    if exchange <= 0:
        return (
            selected_records,
            selected_surrogates,
            {
                **report_base,
                "applied": False,
                "reason": "already_balanced_or_no_exchange",
                "full": {
                    "rows": int(total_rows),
                    "real_closer_rows": full_real,
                    "holdout_closer_rows": full_holdout,
                    "minority_real_closer": minority_real_closer,
                    "minority_rows": full_minority,
                },
                "selected_before": {
                    "rows": int(selected_count),
                    "minority_rows": selected_minority,
                    "majority_rows": selected_majority,
                    "target_minority_rows": target_minority,
                    "needed_exchange_rows": needed,
                },
            },
        )

    quality = np.asarray([_surrogate_quality(record) for record in surrogate_records], dtype=float)
    add_order = unselected_minority[
        np.lexsort(
            (
                unselected_minority,
                -np.abs(holdout_gap[unselected_minority]),
                -quality[unselected_minority],
            )
        )
    ]
    remove_order = removable_majority[
        np.lexsort(
            (
                removable_majority,
                np.abs(holdout_gap[removable_majority]),
                quality[removable_majority],
            )
        )
    ]
    add_idx = add_order[:exchange]
    remove_idx = remove_order[:exchange]

    final_mask = selected_mask.copy()
    final_mask[remove_idx] = False
    final_mask[add_idx] = True
    final_indices = np.flatnonzero(final_mask).astype(int, copy=False).tolist()
    final_records = [pool_records[idx] for idx in final_indices]

    selected_sur_by_id = {
        _candidate_id(record, idx): record
        for idx, record in enumerate(selected_surrogates)
    }
    final_surrogates = [
        selected_sur_by_id.get(_candidate_id(pool_records[idx], idx), surrogate_by_id[_candidate_id(pool_records[idx], idx)])
        for idx in final_indices
    ]
    final_minority = int(np.count_nonzero(final_mask & minority_mask))
    return (
        final_records,
        final_surrogates,
        {
            **report_base,
            "applied": True,
            "exchange_rows": int(exchange),
            "target_fraction": float(target_fraction),
            "max_exchange_fraction": float(max_exchange_fraction),
            "full": {
                "rows": int(total_rows),
                "real_closer_rows": full_real,
                "holdout_closer_rows": full_holdout,
                "minority_real_closer": minority_real_closer,
                "minority_rows": full_minority,
            },
            "selected_before": {
                "rows": int(selected_count),
                "minority_rows": selected_minority,
                "majority_rows": selected_majority,
                "target_minority_rows": target_minority,
                "needed_exchange_rows": needed,
            },
            "selected_after": {
                "rows": int(len(final_records)),
                "minority_rows": final_minority,
                "majority_rows": int(len(final_records) - final_minority),
            },
            "removed_candidate_ids": [_candidate_id(pool_records[idx], idx) for idx in remove_idx.tolist()],
            "added_candidate_ids": [_candidate_id(pool_records[idx], idx) for idx in add_idx.tolist()],
        },
    )
