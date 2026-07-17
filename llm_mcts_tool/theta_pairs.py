from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def build_theta_pair_candidates(col_2ds: Sequence[str]) -> list[tuple[str, str]]:
    columns = sorted(dict.fromkeys(str(column).strip() for column in col_2ds if str(column).strip()))
    return [(left, right) for left, right in combinations(columns, 2)]


def _bucket_series(series: pd.Series, column: str, schema_card: Mapping[str, Any]) -> np.ndarray:
    info = schema_card.get("columns", {}).get(column, {})
    column_type = str(info.get("type", "categorical"))
    if column_type in {"numerical", "discrete_numerical"}:
        numeric = pd.to_numeric(series, errors="coerce")
        if column_type == "discrete_numerical" and info.get("legal_values"):
            legal = np.asarray([float(value) for value in info["legal_values"]], dtype=float)
            values = numeric.fillna(float(np.nanmedian(legal)) if legal.size else 0.0).to_numpy(dtype=float)
            if legal.size == 0:
                return np.zeros(len(series), dtype=int)
            return np.argmin(np.abs(values[:, None] - legal[None, :]), axis=1).astype(int, copy=False)
        non_null = numeric.dropna()
        if non_null.nunique(dropna=True) <= 1:
            return np.zeros(len(series), dtype=int)
        bins = int(min(10, max(2, round(math.sqrt(max(len(non_null), 1))))))
        try:
            codes = pd.qcut(numeric, q=bins, labels=False, duplicates="drop")
        except ValueError:
            codes = pd.cut(numeric, bins=bins, labels=False, duplicates="drop")
        return pd.Series(codes).fillna(-1).to_numpy(dtype=int)

    return pd.factorize(series.astype(str).fillna("__MISSING__"), sort=True)[0].astype(int, copy=False)


def _normalized_mutual_information(left_codes: np.ndarray, right_codes: np.ndarray) -> float:
    valid = (left_codes >= 0) & (right_codes >= 0)
    if not bool(valid.any()):
        return 0.0
    left = left_codes[valid]
    right = right_codes[valid]
    left_bins = int(left.max()) + 1
    right_bins = int(right.max()) + 1
    if left_bins <= 1 or right_bins <= 1:
        return 0.0
    flat = left * right_bins + right
    joint = np.bincount(flat, minlength=left_bins * right_bins).astype(float).reshape(left_bins, right_bins)
    joint /= max(float(joint.sum()), 1.0)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = np.clip(px * py, 1e-12, None)
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log(np.clip(joint[mask] / denom[mask], 1e-12, None))))
    hx = float(-np.sum(px[px > 0] * np.log(np.clip(px[px > 0], 1e-12, None))))
    hy = float(-np.sum(py[py > 0] * np.log(np.clip(py[py > 0], 1e-12, None))))
    return mi / max(math.sqrt(max(hx, 1e-12) * max(hy, 1e-12)), 1e-12)


def rank_theta_pairs_by_mi(
    train_df: pd.DataFrame,
    col_2ds: Sequence[str],
    schema_card: Mapping[str, Any],
    max_pairs: int,
) -> list[tuple[str, str]]:
    pairs = build_theta_pair_candidates(col_2ds)
    if max_pairs <= 0 or not pairs:
        return []
    bucket_cache = {
        column: _bucket_series(train_df[column], column, schema_card)
        for column in sorted({column for pair in pairs for column in pair})
        if column in train_df.columns
    }
    scored: list[tuple[float, str, str]] = []
    for left, right in pairs:
        if left not in bucket_cache or right not in bucket_cache:
            continue
        mi = _normalized_mutual_information(bucket_cache[left], bucket_cache[right])
        scored.append((float(mi), left, right))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(left, right) for _, left, right in scored[: int(max_pairs)]]
