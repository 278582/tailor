from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from .frame_canon import strip_string_cells
from .timing import accumulate_timed_dcr_repeat


DEFAULT_DCR_REPEATS = 10
DEFAULT_DCR_CAP = 8000
DEFAULT_DCR_SEED = 20260420
ALIGN_MODE_PAIRED = "paired_nn_coreset"
ALIGN_MODE_INDEPENDENT = "independent_thinning"
DEFAULT_DCR_ALIGN_MODE = ALIGN_MODE_PAIRED
PAIRED_NN_CANDIDATES = 16
DEFAULT_DCR_QUERY_BATCH_SIZE = 2048
DEFAULT_DCR_REFERENCE_CHUNK_SIZE = 8192


def expanded_dcr_indices(info: dict[str, Any]) -> tuple[list[int], list[int]]:
    num_col_idx = [int(idx) for idx in info.get("num_col_idx", [])]
    cat_col_idx = [int(idx) for idx in info.get("cat_col_idx", [])]
    target_col_idx = [int(idx) for idx in info.get("target_col_idx", [])]
    if str(info.get("task_type", "binclass")) == "regression":
        num_col_idx = num_col_idx + target_col_idx
    else:
        cat_col_idx = cat_col_idx + target_col_idx
    return num_col_idx, cat_col_idx


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _positional_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work.columns = list(range(len(work.columns)))
    return work


def _align_frame_to_reference(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    if list(df.columns) == list(reference.columns):
        return df
    df_names = [str(column) for column in df.columns]
    ref_names = [str(column) for column in reference.columns]
    if df_names == ref_names:
        aligned = df.copy()
        aligned.columns = list(reference.columns)
        return aligned
    if set(df_names) == set(ref_names) and len(df_names) == len(ref_names):
        rename = {str(column): column for column in reference.columns}
        aligned = df.copy()
        aligned.columns = df_names
        aligned = aligned[ref_names]
        aligned.columns = list(reference.columns)
        return aligned
    if len(df.columns) != len(reference.columns):
        raise ValueError(
            f"DCR frame column count mismatch: query={list(df.columns)} reference={list(reference.columns)}"
        )
    return df


def min_l1_distances(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    n_query = int(query.shape[0])
    if n_query == 0:
        return np.zeros(0, dtype=np.float64)
    if reference.shape[0] == 0:
        return np.full(n_query, np.inf, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=1, metric="manhattan", algorithm="brute")
    nn.fit(reference)
    out = np.empty(n_query, dtype=np.float64)
    for start in range(0, n_query, batch_size):
        stop = min(start + batch_size, n_query)
        dist, _ = nn.kneighbors(query[start:stop], n_neighbors=1, return_distance=True)
        out[start:stop] = dist.ravel()
    return out


def subsample_indices(n_rows: int, n: int, rng: np.random.Generator) -> np.ndarray:
    if n_rows <= n:
        return np.arange(n_rows, dtype=np.int64)
    return rng.choice(n_rows, size=n, replace=False).astype(np.int64, copy=False)


def _effective_repeats(*, real_rows: int, test_rows: int, query_rows: int | None, n: int, repeats: int) -> int:
    if repeats <= 1:
        return 1
    query_ok = query_rows is None or int(query_rows) <= n
    if real_rows <= n and test_rows <= n and query_ok:
        return 1
    return int(repeats)


def _paired_effective_repeats(*, test_rows: int, query_rows: int, n: int, repeats: int) -> int:
    if repeats <= 1:
        return 1
    if int(query_rows) <= n and int(test_rows) <= n:
        return 1
    return int(repeats)


def _resolve_compute_cap(cap: int | None) -> int | None:
    if cap is None:
        return None
    parsed = int(cap)
    if parsed <= 0:
        return None
    return parsed


def stratified_subsample_indices(
    n_rows: int,
    n: int,
    rng: np.random.Generator,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    if n_rows <= n:
        return np.arange(n_rows, dtype=np.int64)
    if labels is None:
        return subsample_indices(n_rows, n, rng)
    labels = np.asarray(labels)
    if labels.shape[0] != n_rows:
        return subsample_indices(n_rows, n, rng)
    groups: dict[str, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        groups.setdefault(str(label), []).append(int(idx))
    keys = list(groups)
    sizes = np.asarray([len(groups[key]) for key in keys], dtype=float)
    raw = sizes / max(float(sizes.sum()), 1.0) * float(n)
    alloc = np.minimum(np.floor(raw).astype(int), sizes.astype(int))
    leftover = int(n) - int(alloc.sum())
    order = np.argsort(-(raw - np.floor(raw)), kind="mergesort")
    cursor = 0
    while leftover > 0 and cursor < n_rows * 4:
        group_i = int(order[cursor % len(keys)])
        if alloc[group_i] < len(groups[keys[group_i]]):
            alloc[group_i] += 1
            leftover -= 1
        cursor += 1
    chosen: list[int] = []
    for key, count in zip(keys, alloc.tolist()):
        members = np.asarray(groups[key], dtype=np.int64)
        take = min(int(count), int(members.size))
        if take <= 0:
            continue
        if take >= members.size:
            chosen.extend(members.tolist())
        else:
            chosen.extend(rng.choice(members, size=take, replace=False).astype(int).tolist())
    if len(chosen) < n:
        remaining = np.setdiff1d(np.arange(n_rows, dtype=np.int64), np.asarray(chosen, dtype=np.int64), assume_unique=False)
        need = n - len(chosen)
        if remaining.size < need:
            raise RuntimeError("stratified DCR subsample could not fill n rows")
        chosen.extend(rng.choice(remaining, size=need, replace=False).astype(int).tolist())
    return np.asarray(chosen[:n], dtype=np.int64)


def min_l1_neighbors(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    k: int = 1,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    n_query = int(query.shape[0])
    n_ref = int(reference.shape[0])
    k_eff = max(1, min(int(k), max(n_ref, 1)))
    if n_query == 0:
        return np.zeros((0, k_eff), dtype=np.float64), np.zeros((0, k_eff), dtype=np.int64)
    if n_ref == 0:
        return (
            np.full((n_query, k_eff), np.inf, dtype=np.float64),
            np.full((n_query, k_eff), -1, dtype=np.int64),
        )
    nn = NearestNeighbors(n_neighbors=k_eff, metric="manhattan", algorithm="brute")
    nn.fit(reference)
    dist_out = np.empty((n_query, k_eff), dtype=np.float64)
    idx_out = np.empty((n_query, k_eff), dtype=np.int64)
    for start in range(0, n_query, batch_size):
        stop = min(start + batch_size, n_query)
        dist, idx = nn.kneighbors(query[start:stop], n_neighbors=k_eff, return_distance=True)
        dist_out[start:stop] = dist
        idx_out[start:stop] = idx
    return dist_out, idx_out


def torch_min_l1_fn(
    device: str = "cuda:0",
    query_batch_size: int = DEFAULT_DCR_QUERY_BATCH_SIZE,
    reference_chunk_size: int = DEFAULT_DCR_REFERENCE_CHUNK_SIZE,
):
    import torch

    torch_device = torch.device(device)
    ref_cache: dict[tuple[int, int, int], torch.Tensor] = {}

    def _ref_tensor(reference: np.ndarray) -> torch.Tensor:
        key = (id(reference), int(reference.shape[0]), int(reference.shape[1]) if reference.ndim > 1 else 1)
        tensor = ref_cache.get(key)
        if tensor is None:
            if len(ref_cache) >= 4:
                ref_cache.clear()
            tensor = torch.as_tensor(np.asarray(reference, dtype=np.float32), dtype=torch.float32, device=torch_device)
            ref_cache[key] = tensor
        return tensor

    def _min_l1(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
        n_query = int(query.shape[0])
        if n_query == 0:
            return np.zeros(0, dtype=np.float64)
        if reference.shape[0] == 0:
            return np.full(n_query, np.inf, dtype=np.float64)
        ref = _ref_tensor(reference)
        out = np.empty(n_query, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, n_query, int(query_batch_size)):
                stop = min(start + int(query_batch_size), n_query)
                query_t = torch.as_tensor(
                    np.asarray(query[start:stop], dtype=np.float32),
                    dtype=torch.float32,
                    device=torch_device,
                )
                best: torch.Tensor | None = None
                for r_start in range(0, int(ref.shape[0]), int(reference_chunk_size)):
                    r_end = min(r_start + int(reference_chunk_size), int(ref.shape[0]))
                    chunk_best = torch.cdist(query_t, ref[r_start:r_end], p=1).min(dim=1).values
                    best = chunk_best if best is None else torch.minimum(best, chunk_best)
                assert best is not None
                out[start:stop] = best.detach().cpu().numpy()
        return out

    return _min_l1


def torch_min_l1_neighbors_fn(
    device: str = "cuda:0",
    query_batch_size: int = DEFAULT_DCR_QUERY_BATCH_SIZE,
    reference_chunk_size: int = DEFAULT_DCR_REFERENCE_CHUNK_SIZE,
):
    import torch

    torch_device = torch.device(device)

    def _neighbors(query: np.ndarray, reference: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        n_query = int(query.shape[0])
        n_ref = int(reference.shape[0])
        k_eff = max(1, min(int(k), max(n_ref, 1)))
        if n_query == 0:
            return np.zeros((0, k_eff), dtype=np.float64), np.zeros((0, k_eff), dtype=np.int64)
        if n_ref == 0:
            return (
                np.full((n_query, k_eff), np.inf, dtype=np.float64),
                np.full((n_query, k_eff), -1, dtype=np.int64),
            )
        ref = torch.as_tensor(np.asarray(reference, dtype=np.float32), dtype=torch.float32, device=torch_device)
        dist_out = np.empty((n_query, k_eff), dtype=np.float64)
        idx_out = np.empty((n_query, k_eff), dtype=np.int64)
        with torch.no_grad():
            for start in range(0, n_query, int(query_batch_size)):
                stop = min(start + int(query_batch_size), n_query)
                query_t = torch.as_tensor(
                    np.asarray(query[start:stop], dtype=np.float32),
                    dtype=torch.float32,
                    device=torch_device,
                )
                best_dist: torch.Tensor | None = None
                best_idx: torch.Tensor | None = None
                for r_start in range(0, n_ref, int(reference_chunk_size)):
                    r_end = min(r_start + int(reference_chunk_size), n_ref)
                    dist = torch.cdist(query_t, ref[r_start:r_end], p=1)
                    k_chunk = min(k_eff, r_end - r_start)
                    chunk_dist, chunk_idx = torch.topk(dist, k=k_chunk, dim=1, largest=False)
                    chunk_idx = chunk_idx + r_start
                    if best_dist is None:
                        best_dist, best_idx = chunk_dist, chunk_idx
                        continue
                    merged_dist = torch.cat([best_dist, chunk_dist], dim=1)
                    merged_idx = torch.cat([best_idx, chunk_idx], dim=1)
                    pick = torch.topk(merged_dist, k=k_eff, dim=1, largest=False)
                    best_dist = pick.values
                    best_idx = torch.gather(merged_idx, 1, pick.indices)
                assert best_dist is not None and best_idx is not None
                dist_out[start:stop] = best_dist.detach().cpu().numpy()
                idx_out[start:stop] = best_idx.detach().cpu().numpy()
        return dist_out, idx_out

    return _neighbors


def resolve_dcr_device(device: str = "cpu") -> str:
    resolved = str(device or "cpu")
    if resolved != "auto":
        return resolved
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_dcr_kernels(
    device: str = "cpu",
    *,
    query_batch_size: int = DEFAULT_DCR_QUERY_BATCH_SIZE,
    reference_chunk_size: int = DEFAULT_DCR_REFERENCE_CHUNK_SIZE,
) -> tuple[
    Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None,
]:
    resolved = resolve_dcr_device(device)
    if not resolved.startswith("cuda"):
        return None, None
    return (
        torch_min_l1_fn(
            device=resolved,
            query_batch_size=int(query_batch_size),
            reference_chunk_size=int(reference_chunk_size),
        ),
        torch_min_l1_neighbors_fn(
            device=resolved,
            query_batch_size=int(query_batch_size),
            reference_chunk_size=int(reference_chunk_size),
        ),
    )


def _stratum_labels(syn_df: pd.DataFrame, info: dict[str, Any]) -> np.ndarray | None:
    target_idx = [int(idx) for idx in info.get("target_col_idx", [])]
    if not target_idx or syn_df.empty:
        return None
    col = int(target_idx[0])
    if col < 0 or col >= syn_df.shape[1]:
        return None
    series = syn_df.iloc[:, col]
    if str(info.get("task_type", "binclass")) == "regression":
        values = pd.to_numeric(series, errors="coerce")
        try:
            return pd.qcut(values, q=10, labels=False, duplicates="drop").to_numpy()
        except (TypeError, ValueError):
            return series.astype(str).to_numpy()
    return series.astype(str).to_numpy()


def paired_train_indices(
    syn_x: np.ndarray,
    real_x: np.ndarray,
    syn_idx: np.ndarray,
    *,
    n: int,
    rng: np.random.Generator,
    neighbor_fn: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    query = syn_x[np.asarray(syn_idx, dtype=np.int64)]
    k = min(max(int(PAIRED_NN_CANDIDATES), 1), int(real_x.shape[0]))
    if neighbor_fn is None:
        _, nn_idx = min_l1_neighbors(query, real_x, k=k)
    else:
        _, nn_idx = neighbor_fn(query, real_x, k)
    chosen: list[int] = []
    used: set[int] = set()
    for rank in range(int(nn_idx.shape[1])):
        if len(chosen) >= n:
            break
        for row in range(int(nn_idx.shape[0])):
            if len(chosen) >= n:
                break
            candidate = int(nn_idx[row, rank])
            if candidate < 0 or candidate in used:
                continue
            used.add(candidate)
            chosen.append(candidate)
    if len(chosen) < n:
        remaining = np.asarray([idx for idx in range(int(real_x.shape[0])) if idx not in used], dtype=np.int64)
        need = n - len(chosen)
        if remaining.size < need:
            raise RuntimeError("paired DCR could not fill a train coreset of size n")
        chosen.extend(rng.choice(remaining, size=need, replace=False).astype(int).tolist())
    return np.asarray(chosen[:n], dtype=np.int64)


def _dense_onehot(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def _numeric_block(frame: pd.DataFrame, num_idx: list[int], ranges: np.ndarray) -> np.ndarray:
    values = frame[num_idx].to_numpy(dtype=np.float64)
    return np.nan_to_num(values / ranges, nan=0.0, posinf=0.0, neginf=0.0)


def _categorical_block(frame: pd.DataFrame, cat_idx: list[int]) -> np.ndarray:
    return frame[cat_idx].astype(str).apply(lambda series: series.str.strip()).to_numpy()


@dataclass
class TabDiffL1Encoding:
    info: dict[str, Any]
    column_reference: pd.DataFrame
    num_idx: list[int]
    cat_idx: list[int]
    ranges: np.ndarray | None
    encoder: OneHotEncoder | None
    real_x: np.ndarray
    test_x: np.ndarray

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        work = strip_string_cells(_align_frame_to_reference(df, self.column_reference))
        pos = _positional_frame(work)
        parts: list[np.ndarray] = []
        if self.num_idx:
            assert self.ranges is not None
            parts.append(_numeric_block(pos, self.num_idx, self.ranges))
        if self.cat_idx:
            assert self.encoder is not None
            parts.append(_dense_onehot(self.encoder.transform(_categorical_block(pos, self.cat_idx))))
        if not parts:
            raise ValueError("aligned DCR encoding produced no features")
        return np.concatenate(parts, axis=1)


def fit_tabdiff_l1_encoding(
    real_df: pd.DataFrame,
    test_df: pd.DataFrame,
    info: dict[str, Any],
) -> TabDiffL1Encoding:
    real_df = strip_string_cells(real_df)
    test_df = strip_string_cells(_align_frame_to_reference(test_df, real_df))
    real = _positional_frame(real_df)
    test = _positional_frame(test_df)
    num_idx, cat_idx = expanded_dcr_indices(info)
    ranges: np.ndarray | None = None
    encoder: OneHotEncoder | None = None
    parts_real: list[np.ndarray] = []
    parts_test: list[np.ndarray] = []
    if num_idx:
        num_real = real[num_idx].to_numpy(dtype=np.float64)
        ranges = np.nanmax(num_real, axis=0) - np.nanmin(num_real, axis=0)
        ranges = np.where(np.isfinite(ranges) & (ranges > 0), ranges, 1.0)
        parts_real.append(_numeric_block(real, num_idx, ranges))
        parts_test.append(_numeric_block(test, num_idx, ranges))
    if cat_idx:
        cat_real = _categorical_block(real, cat_idx)
        cat_test = _categorical_block(test, cat_idx)
        encoder = _one_hot_encoder()
        encoder.fit(np.concatenate([cat_real, cat_test], axis=0))
        parts_real.append(_dense_onehot(encoder.transform(cat_real)))
        parts_test.append(_dense_onehot(encoder.transform(cat_test)))
    if not parts_real:
        raise ValueError("aligned DCR encoding produced no features")
    return TabDiffL1Encoding(
        info=info,
        column_reference=real_df.iloc[:0].copy(),
        num_idx=num_idx,
        cat_idx=cat_idx,
        ranges=ranges,
        encoder=encoder,
        real_x=np.concatenate(parts_real, axis=1),
        test_x=np.concatenate(parts_test, axis=1),
    )


def encode_tabdiff_l1(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    test_df: pd.DataFrame,
    info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoding = fit_tabdiff_l1_encoding(real_df, test_df, info)
    return encoding.real_x, encoding.transform(syn_df), encoding.test_x


def independent_closer_rate(
    real_x: np.ndarray,
    test_x: np.ndarray,
    syn_x: np.ndarray,
    *,
    repeats: int = DEFAULT_DCR_REPEATS,
    cap: int = DEFAULT_DCR_CAP,
    seed: int = DEFAULT_DCR_SEED,
    min_l1_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    distance_fn = min_l1_distances if min_l1_fn is None else min_l1_fn
    n = min(int(real_x.shape[0]), int(test_x.shape[0]), int(syn_x.shape[0]), int(cap))
    if n <= 0:
        raise ValueError("aligned DCR requires positive n")
    effective_repeats = _effective_repeats(
        real_rows=int(real_x.shape[0]),
        test_rows=int(test_x.shape[0]),
        query_rows=int(syn_x.shape[0]),
        n=n,
        repeats=repeats,
    )
    rng = np.random.default_rng(int(seed))
    rates: list[float] = []
    last_d_real = np.zeros(0, dtype=np.float64)
    last_d_test = np.zeros(0, dtype=np.float64)
    timed_seconds = 0.0
    timed_repeats = 0
    for repeat_index in range(effective_repeats):
        started = time.perf_counter()
        real_idx = subsample_indices(int(real_x.shape[0]), n, rng)
        test_idx = subsample_indices(int(test_x.shape[0]), n, rng)
        syn_idx = subsample_indices(int(syn_x.shape[0]), n, rng)
        last_d_real = np.asarray(distance_fn(syn_x[syn_idx], real_x[real_idx]), dtype=np.float64)
        last_d_test = np.asarray(distance_fn(syn_x[syn_idx], test_x[test_idx]), dtype=np.float64)
        rates.append(float(np.mean(last_d_real < last_d_test)) if last_d_real.size else float("nan"))
        timed_seconds, timed_repeats = accumulate_timed_dcr_repeat(
            repeat_index,
            time.perf_counter() - started,
            timed_seconds=timed_seconds,
            timed_repeats=timed_repeats,
        )
    rate_arr = np.asarray(rates, dtype=np.float64)
    return {
        "dcr": float(np.mean(rate_arr)),
        "dcr_std": float(np.std(rate_arr, ddof=1)) if rate_arr.size > 1 else 0.0,
        "dcr_n": int(n),
        "dcr_repeats": int(effective_repeats),
        "dcr_requested_repeats": int(repeats),
        "dcr_aligned": True,
        "dcr_align_mode": ALIGN_MODE_INDEPENDENT,
        "dcr_repeat_rates": [float(rate) for rate in rates],
        "dcr_real": last_d_real,
        "dcr_test": last_d_test,
        "aligned_dcr_seconds": float(timed_seconds),
        "aligned_dcr_timed_repeats": int(timed_repeats),
    }


def _maybe_index(matrix: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if int(idx.shape[0]) == int(matrix.shape[0]):
        return matrix
    return matrix[np.asarray(idx, dtype=np.int64)]


def paired_closer_rate(
    real_x: np.ndarray,
    test_x: np.ndarray,
    syn_x: np.ndarray,
    *,
    repeats: int = DEFAULT_DCR_REPEATS,
    cap: int | None = None,
    seed: int = DEFAULT_DCR_SEED,
    min_l1_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    neighbor_fn: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None,
    stratum_labels: np.ndarray | None = None,
    materialize_coreset: bool = False,
) -> dict[str, Any]:
    distance_fn = min_l1_distances if min_l1_fn is None else min_l1_fn
    compute_cap = _resolve_compute_cap(cap)
    n = min(int(syn_x.shape[0]), int(test_x.shape[0]))
    if compute_cap is not None:
        n = min(n, int(compute_cap))
    if n <= 0:
        raise ValueError("paired DCR requires positive n")
    effective_repeats = _paired_effective_repeats(
        test_rows=int(test_x.shape[0]),
        query_rows=int(syn_x.shape[0]),
        n=n,
        repeats=repeats,
    )
    rng = np.random.default_rng(int(seed))
    rates: list[float] = []
    last_d_real = np.zeros(0, dtype=np.float64)
    last_d_test = np.zeros(0, dtype=np.float64)
    last_syn_idx = np.zeros(0, dtype=np.int64)
    last_real_idx = np.zeros(0, dtype=np.int64)
    timed_seconds = 0.0
    timed_repeats = 0
    for repeat_index in range(effective_repeats):
        started = time.perf_counter()
        syn_idx = stratified_subsample_indices(int(syn_x.shape[0]), n, rng, labels=stratum_labels)
        test_idx = subsample_indices(int(test_x.shape[0]), n, rng)
        query = syn_x[np.asarray(syn_idx, dtype=np.int64)]
        test_ref = _maybe_index(test_x, test_idx)
        if materialize_coreset:
            last_real_idx = paired_train_indices(
                syn_x,
                real_x,
                syn_idx,
                n=n,
                rng=rng,
                neighbor_fn=neighbor_fn,
            )
        else:
            last_real_idx = np.zeros(0, dtype=np.int64)
        # Pairing keeps d(c, A') = d(c, A_full) when the unique 1-NN exists.
        # Eval uses the full-train 1-NN directly and skips the extra coreset sweep.
        last_d_real = np.asarray(distance_fn(query, real_x), dtype=np.float64)
        last_d_test = np.asarray(distance_fn(query, test_ref), dtype=np.float64)
        last_syn_idx = syn_idx
        rates.append(float(np.mean(last_d_real < last_d_test)) if last_d_real.size else float("nan"))
        timed_seconds, timed_repeats = accumulate_timed_dcr_repeat(
            repeat_index,
            time.perf_counter() - started,
            timed_seconds=timed_seconds,
            timed_repeats=timed_repeats,
        )
    rate_arr = np.asarray(rates, dtype=np.float64)
    return {
        "dcr": float(np.mean(rate_arr)),
        "dcr_std": float(np.std(rate_arr, ddof=1)) if rate_arr.size > 1 else 0.0,
        "dcr_n": int(n),
        "dcr_repeats": int(effective_repeats),
        "dcr_requested_repeats": int(repeats),
        "dcr_aligned": True,
        "dcr_align_mode": ALIGN_MODE_PAIRED,
        "dcr_compute_cap": int(compute_cap) if compute_cap is not None else 0,
        "dcr_repeat_rates": [float(rate) for rate in rates],
        "dcr_real": last_d_real,
        "dcr_test": last_d_test,
        "dcr_syn_idx": last_syn_idx,
        "dcr_train_idx": last_real_idx,
        "aligned_dcr_seconds": float(timed_seconds),
        "aligned_dcr_timed_repeats": int(timed_repeats),
    }


def aligned_closer_rate(
    real_x: np.ndarray,
    test_x: np.ndarray,
    syn_x: np.ndarray,
    *,
    repeats: int = DEFAULT_DCR_REPEATS,
    cap: int | None = 0,
    seed: int = DEFAULT_DCR_SEED,
    min_l1_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    neighbor_fn: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None,
    stratum_labels: np.ndarray | None = None,
    align_mode: str = DEFAULT_DCR_ALIGN_MODE,
    materialize_coreset: bool = False,
) -> dict[str, Any]:
    mode = str(align_mode or DEFAULT_DCR_ALIGN_MODE)
    if mode == ALIGN_MODE_INDEPENDENT:
        independent_cap = DEFAULT_DCR_CAP if cap is None or int(cap) <= 0 else int(cap)
        return independent_closer_rate(
            real_x,
            test_x,
            syn_x,
            repeats=repeats,
            cap=independent_cap,
            seed=seed,
            min_l1_fn=min_l1_fn,
        )
    return paired_closer_rate(
        real_x,
        test_x,
        syn_x,
        repeats=repeats,
        cap=None if cap is None else cap,
        seed=seed,
        min_l1_fn=min_l1_fn,
        neighbor_fn=neighbor_fn,
        stratum_labels=stratum_labels,
        materialize_coreset=materialize_coreset,
    )


def row_signals_from_matrices(
    pool_x: np.ndarray,
    real_x: np.ndarray,
    test_x: np.ndarray,
    *,
    repeats: int = DEFAULT_DCR_REPEATS,
    cap: int | None = 0,
    seed: int = DEFAULT_DCR_SEED,
    min_l1_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    distance_fn = min_l1_distances if min_l1_fn is None else min_l1_fn
    n_pool = int(pool_x.shape[0])
    compute_cap = _resolve_compute_cap(cap)
    if n_pool == 0:
        return {
            "aligned_closer_rate": np.zeros(0, dtype=np.float64),
            "dcr_real": np.zeros(0, dtype=np.float64),
            "dcr_test": np.zeros(0, dtype=np.float64),
            "is_real_closer": np.zeros(0, dtype=bool),
            "margin": np.zeros(0, dtype=np.float64),
            "dcr_n": min(int(real_x.shape[0]), int(test_x.shape[0])),
            "dcr_repeats": 1,
            "dcr_requested_repeats": int(repeats),
            "dcr_aligned": True,
            "dcr_align_mode": ALIGN_MODE_PAIRED,
            "dcr_cap": 0,
            "dcr_seed": int(seed),
        }
    if compute_cap is None:
        d_real = np.asarray(distance_fn(pool_x, real_x), dtype=np.float64)
        d_test = np.asarray(distance_fn(pool_x, test_x), dtype=np.float64)
        closer = (d_real < d_test).astype(np.float64)
        return {
            "aligned_closer_rate": closer,
            "dcr_real": d_real,
            "dcr_test": d_test,
            "is_real_closer": closer >= 0.5,
            "margin": d_test - d_real,
            "dcr_n": min(int(real_x.shape[0]), int(test_x.shape[0])),
            "dcr_repeats": 1,
            "dcr_requested_repeats": int(repeats),
            "dcr_aligned": True,
            "dcr_align_mode": ALIGN_MODE_PAIRED,
            "dcr_cap": 0,
            "dcr_seed": int(seed),
        }
    n = min(int(real_x.shape[0]), int(test_x.shape[0]), int(compute_cap))
    if n <= 0:
        raise ValueError("aligned DCR row signals require positive n")
    effective_repeats = _effective_repeats(
        real_rows=int(real_x.shape[0]),
        test_rows=int(test_x.shape[0]),
        query_rows=None,
        n=n,
        repeats=repeats,
    )
    rng = np.random.default_rng(int(seed))
    closer = np.zeros(n_pool, dtype=np.float64)
    d_real_acc = np.zeros(n_pool, dtype=np.float64)
    d_test_acc = np.zeros(n_pool, dtype=np.float64)
    for _ in range(effective_repeats):
        real_idx = subsample_indices(int(real_x.shape[0]), n, rng)
        test_idx = subsample_indices(int(test_x.shape[0]), n, rng)
        d_real = np.asarray(distance_fn(pool_x, real_x[real_idx]), dtype=np.float64)
        d_test = np.asarray(distance_fn(pool_x, test_x[test_idx]), dtype=np.float64)
        closer += (d_real < d_test).astype(np.float64)
        d_real_acc += d_real
        d_test_acc += d_test
    aligned_closer = closer / float(effective_repeats)
    dcr_real = d_real_acc / float(effective_repeats)
    dcr_test = d_test_acc / float(effective_repeats)
    return {
        "aligned_closer_rate": aligned_closer,
        "dcr_real": dcr_real,
        "dcr_test": dcr_test,
        "is_real_closer": aligned_closer >= 0.5,
        "margin": dcr_test - dcr_real,
        "dcr_n": int(n),
        "dcr_repeats": int(effective_repeats),
        "dcr_requested_repeats": int(repeats),
        "dcr_aligned": True,
        "dcr_align_mode": ALIGN_MODE_INDEPENDENT,
        "dcr_cap": int(compute_cap),
        "dcr_seed": int(seed),
    }


def evaluate_aligned_dcr(
    syn_df: pd.DataFrame,
    real_df: pd.DataFrame,
    test_df: pd.DataFrame,
    info: dict[str, Any],
    *,
    repeats: int = DEFAULT_DCR_REPEATS,
    cap: int | None = 0,
    seed: int = DEFAULT_DCR_SEED,
    min_l1_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    neighbor_fn: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None,
    align_mode: str = DEFAULT_DCR_ALIGN_MODE,
    encoding: TabDiffL1Encoding | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = str(align_mode or DEFAULT_DCR_ALIGN_MODE)
    if syn_df.empty:
        extras = {
            "dcr_real": np.zeros(0, dtype=np.float64),
            "dcr_test": np.zeros(0, dtype=np.float64),
            "dcr_std": 0.0,
            "dcr_n": 0,
            "dcr_repeats": 0,
            "dcr_requested_repeats": int(repeats),
            "dcr_aligned": True,
            "dcr_align_mode": mode,
            "dcr_repeat_rates": [],
            "aligned_dcr_seconds": 0.0,
            "aligned_dcr_timed_repeats": 0,
        }
        return {"dcr": float("nan")}, extras
    if encoding is None:
        encoding = fit_tabdiff_l1_encoding(real_df, test_df, info)
    syn_x = encoding.transform(syn_df)
    result = aligned_closer_rate(
        encoding.real_x,
        encoding.test_x,
        syn_x,
        repeats=repeats,
        cap=cap,
        seed=seed,
        min_l1_fn=min_l1_fn,
        neighbor_fn=neighbor_fn,
        stratum_labels=_stratum_labels(syn_df, info),
        align_mode=mode,
    )
    metrics = {"dcr": float(result["dcr"])}
    extras = {
        "dcr_real": result["dcr_real"],
        "dcr_test": result["dcr_test"],
        "dcr_std": float(result["dcr_std"]),
        "dcr_n": int(result["dcr_n"]),
        "dcr_repeats": int(result["dcr_repeats"]),
        "dcr_requested_repeats": int(result["dcr_requested_repeats"]),
        "dcr_aligned": True,
        "dcr_align_mode": str(result.get("dcr_align_mode", mode)),
        "dcr_compute_cap": int(result.get("dcr_compute_cap", 0) or 0),
        "dcr_repeat_rates": list(result["dcr_repeat_rates"]),
        "aligned_dcr_seconds": float(result.get("aligned_dcr_seconds", 0.0) or 0.0),
        "aligned_dcr_timed_repeats": int(result.get("aligned_dcr_timed_repeats", 0) or 0),
    }
    return metrics, extras
