from __future__ import annotations

import time

from .config import progress_enabled
from .logging_utils import get_logger
from .state import SelectionState


def compute_global_exact_scores(state: SelectionState) -> SelectionState:
    if state.selector is None or state.d_cur_df is None:
        raise RuntimeError("initialize_selector_and_pool must run before compute_global_exact_scores")
    if not state.preselected_valid:
        raise RuntimeError("build_preselected_valid must run before compute_global_exact_scores")

    logger = get_logger()
    candidate_rows = len(state.preselected_valid)
    logger.info(
        "[objective_score] exact start rows=%d objectives=fidelity_1d,fidelity_2d,privacy",
        candidate_rows,
    )
    started = time.perf_counter()
    state.global_exact_records, state.global_baselines = state.selector.compute_exact_scores(
        state.d_cur_df,
        state.preselected_valid,
        show_progress=progress_enabled(state.config),
        progress_desc="global exact scoring",
    )
    elapsed = float(time.perf_counter() - started)
    objective_timing = state.timing_report.setdefault("objective_scoring", {})
    objective_timing.update(
        {
            "schema_version": 1,
            "candidate_rows": int(candidate_rows),
            "objectives": [
                "pareto_fid_1d_obj",
                "pareto_fid_2d_obj",
                "pareto_priv_obj",
                "pareto_util_proxy_obj",
            ],
            "exact_fidelity_privacy_seconds": elapsed,
            "exact_fidelity_privacy_recorded": True,
            "timing_basis": "wall_clock_perf_counter",
            "shared_by": ["scalar", "pareto"],
        }
    )
    logger.info(
        "[objective_score] exact done rows=%d elapsed=%.2fs objectives=fidelity_1d,fidelity_2d,privacy",
        len(state.global_exact_records),
        elapsed,
    )
    return state
