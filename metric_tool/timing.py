from __future__ import annotations

from typing import Any


EVAL_TIMING_SCHEMA_VERSION = 1
TIMING_BASIS = "wall_clock_perf_counter"
TIMED_DCR_REPEATS = 2


def accumulate_timed_dcr_repeat(
    repeat_index: int,
    elapsed: float,
    *,
    timed_seconds: float,
    timed_repeats: int,
    limit: int = TIMED_DCR_REPEATS,
) -> tuple[float, int]:
    if int(repeat_index) < int(limit):
        return float(timed_seconds) + float(elapsed), int(timed_repeats) + 1
    return float(timed_seconds), int(timed_repeats)


def build_eval_timing(
    *,
    sdmetrics_seconds: float,
    aligned_dcr_seconds: float,
    aligned_dcr_timed_repeats: int,
    aligned_dcr_computed_repeats: int,
    utility_exact_seconds: float,
    summary_seconds: float,
) -> dict[str, Any]:
    sdmetrics_seconds = float(sdmetrics_seconds)
    aligned_dcr_seconds = float(aligned_dcr_seconds)
    utility_exact_seconds = float(utility_exact_seconds)
    summary_seconds = float(summary_seconds)
    return {
        "schema_version": EVAL_TIMING_SCHEMA_VERSION,
        "timing_basis": TIMING_BASIS,
        "sdmetrics_seconds": sdmetrics_seconds,
        "aligned_dcr_seconds": aligned_dcr_seconds,
        "aligned_dcr_timed_repeats": int(aligned_dcr_timed_repeats),
        "aligned_dcr_computed_repeats": int(aligned_dcr_computed_repeats),
        "utility_exact_seconds": utility_exact_seconds,
        "summary_seconds": summary_seconds,
        "total_seconds": (
            sdmetrics_seconds + aligned_dcr_seconds + utility_exact_seconds + summary_seconds
        ),
    }


def add_summary_elapsed(eval_timing: dict[str, Any] | None, extra_seconds: float) -> dict[str, Any]:
    timing = dict(eval_timing or {})
    return build_eval_timing(
        sdmetrics_seconds=float(timing.get("sdmetrics_seconds") or 0.0),
        aligned_dcr_seconds=float(timing.get("aligned_dcr_seconds") or 0.0),
        aligned_dcr_timed_repeats=int(timing.get("aligned_dcr_timed_repeats") or 0),
        aligned_dcr_computed_repeats=int(timing.get("aligned_dcr_computed_repeats") or 0),
        utility_exact_seconds=float(timing.get("utility_exact_seconds") or 0.0),
        summary_seconds=float(timing.get("summary_seconds") or 0.0) + float(extra_seconds),
    )
