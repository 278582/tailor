from __future__ import annotations

from typing import Any

import pandas as pd

from .aligned_dcr import (
    DEFAULT_DCR_ALIGN_MODE,
    DEFAULT_DCR_QUERY_BATCH_SIZE,
    DEFAULT_DCR_REFERENCE_CHUNK_SIZE,
    DEFAULT_DCR_REPEATS,
    DEFAULT_DCR_SEED,
    TabDiffL1Encoding,
    evaluate_aligned_dcr,
    fit_tabdiff_l1_encoding,
    resolve_dcr_kernels,
)


def bind_dcr_kernels(
    runner: Any,
    *,
    device: str,
    query_batch_size: int = DEFAULT_DCR_QUERY_BATCH_SIZE,
    reference_chunk_size: int = DEFAULT_DCR_REFERENCE_CHUNK_SIZE,
) -> None:
    min_l1_fn, neighbor_fn = resolve_dcr_kernels(
        device,
        query_batch_size=int(query_batch_size),
        reference_chunk_size=int(reference_chunk_size),
    )
    runner.dcr_min_l1_fn = min_l1_fn
    runner.dcr_neighbor_fn = neighbor_fn
    runner._dcr_encoding = None


def cached_dcr_encoding(runner: Any) -> TabDiffL1Encoding:
    encoding = getattr(runner, "_dcr_encoding", None)
    if encoding is not None:
        return encoding
    real_df = pd.read_csv(runner.real_data_path)
    test_df = pd.read_csv(runner.test_data_path)
    encoding = fit_tabdiff_l1_encoding(real_df, test_df, runner.info)
    runner._dcr_encoding = encoding
    return encoding


def evaluate_dcr(runner: Any, df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    cap = int(getattr(runner, "dcr_cap", 0) or 0)
    encoding = cached_dcr_encoding(runner)
    return evaluate_aligned_dcr(
        syn_df=df.copy(),
        real_df=pd.DataFrame(),
        test_df=pd.DataFrame(),
        info=runner.info,
        repeats=int(getattr(runner, "dcr_repeats", DEFAULT_DCR_REPEATS)),
        cap=cap,
        seed=int(getattr(runner, "dcr_seed", DEFAULT_DCR_SEED)),
        min_l1_fn=getattr(runner, "dcr_min_l1_fn", None),
        neighbor_fn=getattr(runner, "dcr_neighbor_fn", None),
        align_mode=str(getattr(runner, "dcr_align_mode", DEFAULT_DCR_ALIGN_MODE)),
        encoding=encoding,
    )
