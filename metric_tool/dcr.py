from __future__ import annotations

from pathlib import Path
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
    runner._dcr_hold_encoding = None


def cached_dcr_encoding(runner: Any) -> TabDiffL1Encoding:
    encoding = getattr(runner, "_dcr_encoding", None)
    if encoding is not None:
        return encoding
    real_df = pd.read_csv(runner.real_data_path)
    test_df = pd.read_csv(runner.test_data_path)
    encoding = fit_tabdiff_l1_encoding(real_df, test_df, runner.info)
    runner._dcr_encoding = encoding
    return encoding


def cached_hold_dcr_encoding(runner: Any) -> TabDiffL1Encoding:
    encoding = getattr(runner, "_dcr_hold_encoding", None)
    if encoding is not None:
        return encoding
    hold_path = getattr(runner, "val_data_path", None)
    if hold_path is None or not Path(hold_path).is_file():
        raise FileNotFoundError("hold DCR requested, but the metric runner has no hold CSV")
    real_df = pd.read_csv(runner.real_data_path)
    hold_df = pd.read_csv(hold_path)
    encoding = fit_tabdiff_l1_encoding(real_df, hold_df, runner.info)
    runner._dcr_hold_encoding = encoding
    return encoding


def _evaluate_dcr_with_encoding(
    runner: Any,
    df: pd.DataFrame,
    encoding: TabDiffL1Encoding,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if getattr(runner, "dcr_align_mode", None) == "query_batches_full_train_v1":
        from .query_batches import evaluate
        return evaluate(runner,df,encoding)
    cap = int(getattr(runner, "dcr_cap", 0) or 0)
    if getattr(runner, "dcr_align_mode", None) == "target_pool_mc":
        from .target_pool_mc import evaluate, sampling_plan
        if cap:
            raise ValueError("target_pool_mc forbids dcr-cap")
        if getattr(runner, "enable_hold_dcr", False):
            # Validate equality without fitting or exposing test to selection callers.
            ref_rows = len(pd.read_csv(runner.test_data_path))
            hold_rows = len(pd.read_csv(runner.val_data_path))
            if ref_rows != hold_rows:
                raise ValueError("target_pool_mc requires equal hold/test rows")
        return evaluate(encoding.transform(df.copy()), encoding.real_x, encoding.test_x,
            repeats=int(runner.dcr_repeats), seed=int(runner.dcr_seed),
            plan_path=getattr(runner, "dcr_plan_path", None),
            min_l1_fn=getattr(runner, "dcr_min_l1_fn", None))
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


def evaluate_dcr(runner: Any, df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    return _evaluate_dcr_with_encoding(runner, df, cached_dcr_encoding(runner))


def evaluate_hold_dcr(runner: Any, df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    return _evaluate_dcr_with_encoding(runner, df, cached_hold_dcr_encoding(runner))
