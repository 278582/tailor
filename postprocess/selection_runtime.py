from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from metric_tool.evaluator import build_metric_directions

from .data_io import ensure_dir, load_csv, save_csv, save_json, save_jsonl, set_seed
from .cards import build_and_save_cards
from .pareto import ParetoSelector
from .tabdiff_eval import TabDiffSelectionEvaluator
from .tabdiff_protocol import normalize_tabdiff_dataframe_columns, resolve_tabdiff_selection_context
from .tabdiff_utils import find_latest_tabdiff_sample
from .utility_proxy import (
    build_static_balanced_utility_scores,
    build_utility_proxy_scores,
    compute_utility_exact_metrics,
)
from .validator import TabularValidator

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


DIRECTION_SPECS: list[tuple[str, float, float, float, float]] = [
    ("f100_t000_p000_u000", 1.0, 0.0, 0.0, 0.0),
    ("f000_t100_p000_u000", 0.0, 1.0, 0.0, 0.0),
    ("f000_t000_p100_u000", 0.0, 0.0, 1.0, 0.0),
    ("f000_t000_p000_u100", 0.0, 0.0, 0.0, 1.0),
    ("f050_t050_p000_u000", 0.5, 0.5, 0.0, 0.0),
    ("f050_t000_p050_u000", 0.5, 0.0, 0.5, 0.0),
    ("f000_t050_p050_u000", 0.0, 0.5, 0.5, 0.0),
    ("f050_t000_p000_u050", 0.5, 0.0, 0.0, 0.5),
    ("f000_t050_p000_u050", 0.0, 0.5, 0.0, 0.5),
    ("f000_t000_p050_u050", 0.0, 0.0, 0.5, 0.5),
    ("f025_t025_p025_u025", 0.25, 0.25, 0.25, 0.25),
]

DEFAULT_D_CUR_SIZE = 200
DEFAULT_KEEP_K = 50
DEFAULT_PRESELECT_TARGET = 50
DEFAULT_TABDIFF_DATASET_NAME = "adult_tgm_w1"


class _NullProgress:
    def __init__(self, iterable: Any | None = None) -> None:
        self.iterable = iterable

    def __iter__(self) -> Any:
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_description(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def _progress(iterable: Any | None = None, **kwargs: Any) -> Any:
    if _tqdm is None:
        return _NullProgress(iterable)
    if iterable is None:
        return _tqdm(**kwargs)
    return _tqdm(iterable, **kwargs)


def _progress_write(message: str) -> None:
    if _tqdm is None:
        print(message)
    else:
        _tqdm.write(message)


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M5 selection and evaluation on TabDiff synthetic tabular data.")
    parser.add_argument("--synthetic-csv", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_TABDIFF_DATASET_NAME)
    parser.add_argument("--exp-name", type=str, default="adult_tgm_w1")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--source", type=str, default="tabdiff")
    parser.add_argument("--keep-k", type=int, default=DEFAULT_KEEP_K)
    parser.add_argument("--preselect-target", type=int, default=DEFAULT_PRESELECT_TARGET)
    parser.add_argument("--d-cur-size", type=int, default=DEFAULT_D_CUR_SIZE)
    parser.add_argument("--d-cur-source", choices=["synthetic", "train"], default="synthetic")
    parser.add_argument("--scalar-fidelity-weight", type=float, default=0.5)
    parser.add_argument("--scalar-privacy-weight", type=float, default=0.3)
    parser.add_argument("--scalar-utility-weight", type=float, default=0.2)
    parser.add_argument(
        "--selection-chunk-size",
        type=int,
        default=0,
        help="Chunk size for streaming archive. If 0, derive a bounded-memory default.",
    )
    parser.add_argument("--archive-budget-scale", type=float, default=1.2)
    parser.add_argument("--local-keep-factor", type=float, default=3.0)
    parser.add_argument("--lambda-penalty", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--privacy-version", choices=["v1", "v2", "v3"], default="v2")
    parser.add_argument(
        "--nn-device",
        type=str,
        default="auto",
        help="Nearest-neighbor backend device for M5 privacy scoring: auto, cpu, or cuda:<idx>.",
    )
    parser.add_argument(
        "--nn-query-batch-size",
        type=int,
        default=2048,
        help="Query batch size for GPU M5 nearest-neighbor scoring.",
    )
    parser.add_argument(
        "--nn-reference-chunk-size",
        type=int,
        default=8192,
        help="Reference chunk size for GPU M5 nearest-neighbor scoring.",
    )
    parser.add_argument(
        "--density-reference-size",
        type=int,
        default=5000,
        help="Max train rows used to estimate density-normalized privacy strata. Use 0 for full train.",
    )
    parser.add_argument("--jsd-epsilon", type=float, default=0.15)
    parser.add_argument("--rare-threshold", type=float, default=0.05)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--final-fidelity-floor-eps", type=float, default=0.01)
    parser.add_argument("--final-trend-floor-eps", type=float, default=0.01)
    parser.add_argument("--fidelity-ceiling-utility-weight", type=float, default=0.04)
    parser.add_argument("--fidelity-ceiling-refine-utility-weight", type=float, default=0.15)
    parser.add_argument("--fidelity-ceiling-second-pass-utility-weight", type=float, default=0.08)
    parser.add_argument("--fidelity-ceiling-second-pass-refine-utility-weight", type=float, default=0.20)
    parser.add_argument("--pareto-rerank-utility-switch-min", type=float, default=0.002)
    parser.add_argument("--pareto-rerank-privacy-switch-min", type=float, default=0.005)
    parser.add_argument("--pareto-floor-mode", choices=["hard", "soft"], default="soft")
    parser.add_argument("--pareto-soft-fidelity-floor-eps", type=float, default=0.02)
    parser.add_argument("--pareto-soft-trend-floor-eps", type=float, default=0.02)
    parser.add_argument("--pareto-soft-privacy-floor-eps", type=float, default=0.005)
    parser.add_argument("--pareto-soft-utility-floor-eps", type=float, default=0.005)
    parser.add_argument("--pareto-soft-min-score-delta", type=float, default=0.0)
    parser.add_argument("--eval-device", type=str, default="auto")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def _df_to_candidate_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    row_dicts = df.to_dict(orient="records")
    return [{"candidate_id": idx, "row": row} for idx, row in enumerate(row_dicts)]


def _records_to_df(records: list[dict[str, Any]], column_order: list[str]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=column_order)
    return pd.DataFrame([record["row"] for record in records], columns=column_order)


def _resolve_synthetic_csv(args: argparse.Namespace) -> Path:
    if args.synthetic_csv is not None:
        return args.synthetic_csv
    return find_latest_tabdiff_sample(dataset_name=args.dataset_name, exp_name=args.exp_name)


def _preselect_objective_manifest(selected_mode: str) -> tuple[str, dict[str, object]]:
    if selected_mode == "two_stage_band_quota_v2":
        return (
            "density_normalized_nn_distance_v2_band_limited_tiebreak",
            {
                "type": "two_stage_band_quota_v2",
                "stage_a": {
                    "components": [
                        "1d_train_clipped_quota_alignment",
                        "2d_train_clipped_quota_alignment",
                        "fidelity_safe_band_score",
                    ],
                    "privacy_in_primary_score": False,
                    "target_mode": "train_clipped_by_availability",
                },
                "stage_b": {
                    "components": [
                        "1d_band_empirical_quota_alignment",
                        "2d_band_empirical_quota_alignment",
                        "fidelity_safe_stage_b_score",
                        "weak_privacy_tiebreak",
                    ],
                    "target_source": "fidelity_safe_band_empirical",
                    "band_target_scale": 1.4,
                    "privacy_weight_max": 0.05,
                },
                "support_diagnostics": ["1d_train_support", "2d_graph_support", "density_normalized_nn_distance"],
            },
        )
    if selected_mode == "three_objective_preselect_v3":
        return (
            "density_normalized_nn_distance_v2_secondary",
            {
                "type": "three_objective_preselect_v3",
                "components": ["1d_empirical_quota_alignment", "2d_empirical_quota_alignment", "privacy_tiebreak"],
                "support_diagnostics": ["1d_train_support", "2d_graph_support", "density_normalized_nn_distance"],
            },
        )
    return (
        "not_applied",
        {
            "type": selected_mode,
            "components": [],
            "support_diagnostics": [],
        },
    )


def _attach_utility_proxy_fields(
    exact_records: list[dict[str, object]],
    proxy_scores: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    proxy_by_id = {int(record["candidate_id"]): record for record in proxy_scores}
    merged_records: list[dict[str, object]] = []
    matched_rows = 0
    missing_rows = 0
    for idx, record in enumerate(exact_records):
        candidate_id = int(record.get("candidate_id", idx))
        proxy = proxy_by_id.get(candidate_id)
        merged = dict(record)
        if proxy is None:
            missing_rows += 1
            merged["pareto_util_proxy_obj"] = 0.0
            merged["utility_proxy_static"] = 0.0
            merged["utility_proxy_dynamic"] = 0.0
            merged["utility_proxy_total"] = 0.0
            merged["utility_proxy_static_norm"] = 0.0
            merged["utility_proxy_dynamic_norm"] = 0.0
            merged["utility_proxy_static_raw"] = 0.0
            merged["utility_proxy_static_group_rank"] = 0.0
            merged["utility_proxy_density_weight"] = 0.0
            merged["utility_proxy_coverage_gain"] = 0.0
            merged["utility_proxy_gate_stratum"] = -1
            merged["utility_proxy_target_label"] = None
            merged["utility_anchor_member"] = False
        else:
            matched_rows += 1
            merged["pareto_util_proxy_obj"] = float(proxy.get("u_proxy", 0.0))
            merged["utility_proxy_static"] = float(proxy.get("u_static", 0.0))
            merged["utility_proxy_dynamic"] = float(proxy.get("u_dynamic", 0.0))
            merged["utility_proxy_total"] = float(proxy.get("u_proxy", 0.0))
            merged["utility_proxy_static_norm"] = float(proxy.get("u_static_norm", 0.0))
            merged["utility_proxy_dynamic_norm"] = float(proxy.get("u_dynamic_norm", 0.0))
            merged["utility_proxy_static_raw"] = float(proxy.get("u_static_raw", proxy.get("u_static", 0.0)))
            merged["utility_proxy_static_group_rank"] = float(proxy.get("u_static_group_rank", 0.0))
            merged["utility_proxy_density_weight"] = float(proxy.get("density_weight", 0.0))
            merged["utility_proxy_coverage_gain"] = float(proxy.get("coverage_gain", 0.0))
            merged["utility_proxy_gate_stratum"] = int(proxy.get("gate_stratum", -1))
            merged["utility_proxy_target_label"] = proxy.get("target_label")
            merged["utility_anchor_member"] = bool(proxy.get("is_anchor_member", False))
        merged_records.append(merged)
    return merged_records, {
        "matched_rows": matched_rows,
        "missing_rows": missing_rows,
        "proxy_rows": len(proxy_scores),
        "exact_rows": len(exact_records),
    }


def _resolve_eval_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _resolve_nn_device(nn_device_arg: str, eval_device: str) -> str:
    if nn_device_arg != "auto":
        return nn_device_arg
    if eval_device.startswith("cuda"):
        return eval_device
    return "auto"


def _subset_metrics(selector: ParetoSelector, df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "rows": 0,
            "avg_fidelity_sur": 0.0,
            "avg_privacy_sur": 0.0,
            "avg_nn_distance": 0.0,
            "fidelity": 0.0,
            "privacy": 0.0,
        }
    surrogates = selector.compute_surrogates(df.reset_index(drop=True))
    return {
        "rows": int(len(df)),
        "avg_fidelity_sur": float(np.mean([record["s_fid_sur"] for record in surrogates])) if surrogates else 0.0,
        "avg_privacy_sur": float(np.mean([record["s_priv_sur"] for record in surrogates])) if surrogates else 0.0,
        "avg_nn_distance": selector.compute_dataset_mean_nn_distance(df),
        "fidelity": selector.compute_dataset_fidelity(df),
        "privacy": selector.compute_dataset_privacy(df),
    }


def _save_selection_csvs(
    versions_dir: Path,
    raw_df: pd.DataFrame,
    random_df: pd.DataFrame,
    scalar_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    *,
    raw_tag: str = "raw",
    random_tag: str = "random",
) -> None:
    save_csv(versions_dir / f"selection_{raw_tag}.csv", raw_df)
    save_csv(versions_dir / f"selection_{random_tag}.csv", random_df)
    save_csv(versions_dir / "selection_scalar.csv", scalar_df)
    save_csv(versions_dir / "selection_pareto.csv", pareto_df)
    save_csv(versions_dir / f"{raw_tag}_keep.csv", raw_df)
    save_csv(versions_dir / f"{random_tag}_keep.csv", random_df)
    save_csv(versions_dir / "scalarization_keep.csv", scalar_df)
    save_csv(versions_dir / "pareto_keep.csv", pareto_df)


def _save_eval_extras(eval_dir: Path, selection_name: str, extras: dict[str, Any]) -> None:
    target_dir = ensure_dir(eval_dir / selection_name)
    for key, value in extras.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(target_dir / f"{key}.csv", index=False)
        elif isinstance(value, dict):
            save_json(target_dir / f"{key}.json", value)
        elif key in {"dcr_real", "dcr_test"}:
            continue
        else:
            save_json(target_dir / f"{key}.json", {"value": value})
    if "dcr_real" in extras and "dcr_test" in extras:
        dcr_df = pd.DataFrame({"dcr_real": extras["dcr_real"], "dcr_test": extras["dcr_test"]})
        dcr_df.to_csv(target_dir / "dcr.csv", index=False)


def _chunk_slices(total_size: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= total_size:
        return [(0, total_size)]
    return [(start, min(start + chunk_size, total_size)) for start in range(0, total_size, chunk_size)]


def _chunk_keep_quota(
    chunk_rows: int,
    remaining_keep: int,
    remaining_rows: int,
    is_last_chunk: bool,
) -> int:
    if remaining_keep <= 0 or chunk_rows <= 0:
        return 0
    if is_last_chunk:
        return min(remaining_keep, chunk_rows)
    lower_bound = max(0, remaining_keep - (remaining_rows - chunk_rows))
    upper_bound = min(chunk_rows, remaining_keep)
    proportional = int(round((remaining_keep * chunk_rows) / max(remaining_rows, 1)))
    return max(lower_bound, min(upper_bound, proportional))


def _candidate_ids(records: list[dict[str, Any]]) -> set[int]:
    return {int(record.get("candidate_id", idx)) for idx, record in enumerate(records)}


def _merge_reference_df(
    d_cur_df: pd.DataFrame,
    archive_records: list[dict[str, Any]],
    column_order: list[str],
) -> pd.DataFrame:
    if not archive_records:
        return d_cur_df.reset_index(drop=True)
    archive_df = _records_to_df(archive_records, column_order)
    return pd.concat([d_cur_df.reset_index(drop=True), archive_df.reset_index(drop=True)], axis=0, ignore_index=True)


def _streaming_chunk_size(total_rows: int, keep_k: int, user_chunk_size: int) -> int:
    if user_chunk_size > 0:
        return min(total_rows, user_chunk_size)
    return min(total_rows, max(1024, keep_k * 4))


def _extract_selected_exact_records(
    selected_records: list[dict[str, Any]],
    exact_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_ids = _candidate_ids(selected_records)
    return [record for record in exact_records if int(record["candidate_id"]) in selected_ids]


def _run_streaming_archive(
    selector: ParetoSelector,
    pool_records: list[dict[str, Any]],
    d_cur_df: pd.DataFrame,
    keep_k: int,
    preselect_target: int,
    chunk_size: int,
    archive_budget: int,
    local_keep_factor: float,
    show_progress: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not pool_records:
        return [], [], {"chunks": [], "archive_budget": archive_budget, "chunk_size": chunk_size}

    chunk_plan = _chunk_slices(len(pool_records), chunk_size)
    archive_records: list[dict[str, Any]] = []
    archive_exact_records: list[dict[str, Any]] = []
    chunk_reports: list[dict[str, Any]] = []
    remaining_keep = min(keep_k, len(pool_records))
    remaining_rows = len(pool_records)

    chunk_iter = _progress(
        enumerate(chunk_plan),
        total=len(chunk_plan),
        desc="streaming archive",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for chunk_id, (start, end) in chunk_iter:
        chunk_records = pool_records[start:end]
        chunk_rows = len(chunk_records)
        batch_keep = _chunk_keep_quota(
            chunk_rows=chunk_rows,
            remaining_keep=remaining_keep,
            remaining_rows=remaining_rows,
            is_last_chunk=(chunk_id == len(chunk_plan) - 1),
        )
        chunk_df = _records_to_df(chunk_records, selector.column_order)
        chunk_surrogates = selector.compute_surrogates(
            chunk_df,
            show_progress=False,
            progress_desc=f"chunk {chunk_id} surrogate",
            candidate_ids=np.fromiter(
                (int(record.get("candidate_id", idx)) for idx, record in enumerate(chunk_records)),
                dtype=int,
                count=len(chunk_records),
            ),
        )

        proportional_preselect = int(round(preselect_target * chunk_rows / max(len(pool_records), 1)))
        chunk_preselect_target = min(
            chunk_rows,
            max(batch_keep, proportional_preselect, int(round(batch_keep * local_keep_factor))),
        )
        preselected_records, preselected_surrogates = selector.dual_median_filter(
            valid_records=chunk_records,
            surrogate_records=chunk_surrogates,
            target_preselect=max(1, chunk_preselect_target),
        )

        reference_df = d_cur_df.reset_index(drop=True)
        exact_records, baselines = selector.compute_exact_scores(
            reference_df,
            preselected_records,
            show_progress=False,
            progress_desc=f"chunk {chunk_id} exact",
        )
        local_keep = min(len(preselected_records), max(batch_keep, int(round(batch_keep * local_keep_factor))))
        if local_keep > 0:
            _, chunk_keep_records, chunk_keep_report = selector.select_keep(
                preselected_records=preselected_records,
                surrogate_records=preselected_surrogates,
                exact_records=exact_records,
                keep_k=local_keep,
            )
            chunk_keep_exact = _extract_selected_exact_records(chunk_keep_records, exact_records)
        else:
            chunk_keep_records = []
            chunk_keep_exact = []
            chunk_keep_report = {"selected": 0, "fronts": []}

        archive_records.extend(chunk_keep_records)
        archive_exact_records.extend(chunk_keep_exact)
        archive_before_reduction = len(archive_records)
        if len(archive_records) > archive_budget:
            archive_records, archive_exact_records, reduction_report = selector.reduce_archive(
                archive_records=archive_records,
                archive_exact_records=archive_exact_records,
                budget=archive_budget,
            )
        else:
            reduction_report = {
                "archive_rows_before_reduction": archive_before_reduction,
                "archive_rows_after_reduction": archive_before_reduction,
                "reduction_applied": False,
                "secondary_filter": {"applied": False},
            }

        chunk_reports.append(
            {
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
                "chunk_rows": chunk_rows,
                "batch_keep": batch_keep,
                "chunk_preselect_target": chunk_preselect_target,
                "preselected_rows": len(preselected_records),
                "chunk_selected_rows": len(chunk_keep_records),
                "archive_rows_before_reduction": archive_before_reduction,
                "archive_rows_after_reduction": len(archive_records),
                "baseline_fidelity": baselines.get("baseline_fidelity", 0.0),
                "baseline_privacy": baselines.get("baseline_privacy", 0.0),
                "fronts": chunk_keep_report.get("fronts", []),
                "reduction": reduction_report,
            }
        )
        chunk_iter.set_postfix(
            chunk_id=chunk_id,
            chunk_rows=chunk_rows,
            preselected=len(preselected_records),
            archive=len(archive_records),
        )

        remaining_keep = max(0, remaining_keep - batch_keep)
        remaining_rows -= chunk_rows

    return archive_records, archive_exact_records, {
        "mode": "streaming_archive",
        "chunk_size": chunk_size,
        "archive_budget": archive_budget,
        "fixed_reference_baseline": True,
        "chunks": chunk_reports,
        "archive_rows_final": len(archive_records),
    }


def _build_direction_family(
    selector: ParetoSelector,
    records: list[dict[str, Any]],
    exact_records: list[dict[str, Any]],
    keep_k: int,
    family_type: str,
    floor_reference: dict[str, Any] | None = None,
    constraint_reference_records: list[dict[str, Any]] | None = None,
    floor_mode: str = "hard",
    soft_fidelity_floor_eps: float | None = None,
    soft_trend_floor_eps: float | None = None,
    soft_privacy_floor_eps: float = 0.005,
    soft_utility_floor_eps: float = 0.005,
    soft_min_score_delta: float = 0.0,
    show_progress: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    family_frames: dict[str, pd.DataFrame] = {}
    family_records: dict[str, list[dict[str, Any]]] = {}
    family_reports: dict[str, dict[str, Any]] = {}

    direction_iter = _progress(
        DIRECTION_SPECS,
        total=len(DIRECTION_SPECS),
        desc=f"{family_type} family",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for tag, fid1_w, fid2_w, priv_w, util_w in direction_iter:
        if family_type == "scalar_naive":
            df, keep_records, report = selector.select_keep_scalarization(
                preselected_records=records,
                exact_records=exact_records,
                keep_k=keep_k,
                fidelity_1d_weight=fid1_w,
                fidelity_2d_weight=fid2_w,
                privacy_weight=priv_w,
                utility_weight=util_w,
                mode="naive",
                floor_reference=None,
            )
        elif family_type == "scalar_matched":
            df, keep_records, report = selector.select_keep_scalarization(
                preselected_records=records,
                exact_records=exact_records,
                keep_k=keep_k,
                fidelity_1d_weight=fid1_w,
                fidelity_2d_weight=fid2_w,
                privacy_weight=priv_w,
                utility_weight=util_w,
                mode="matched",
                floor_reference=floor_reference,
            )
        else:
            df, keep_records, report = selector.select_keep_chebyshev(
                preselected_records=records,
                exact_records=exact_records,
                keep_k=keep_k,
                fidelity_1d_weight=fid1_w,
                fidelity_2d_weight=fid2_w,
                privacy_weight=priv_w,
                utility_weight=util_w,
                floor_reference=floor_reference,
                constraint_reference_records=constraint_reference_records,
                floor_mode=floor_mode,
                soft_fidelity_floor_eps=soft_fidelity_floor_eps,
                soft_trend_floor_eps=soft_trend_floor_eps,
                soft_privacy_floor_eps=soft_privacy_floor_eps,
                soft_utility_floor_eps=soft_utility_floor_eps,
                soft_min_score_delta=soft_min_score_delta,
            )
        family_frames[tag] = df
        family_records[tag] = keep_records
        family_reports[tag] = report
        direction_iter.set_postfix(direction=tag, rows=len(df))
    return family_frames, family_records, family_reports


def _rerank_pareto_finalists_on_search_holdout(
    *,
    selector: ParetoSelector,
    exact_records: list[dict[str, Any]],
    pareto_keep_df: pd.DataFrame,
    pareto_keep_records: list[dict[str, Any]],
    pareto_report: dict[str, Any],
    pareto_family_df: dict[str, pd.DataFrame],
    pareto_family_records: dict[str, list[dict[str, Any]]],
    pareto_family_reports: dict[str, dict[str, Any]],
    floor_reference: dict[str, Any] | None,
    fidelity_ceiling_df: pd.DataFrame | None = None,
    fidelity_ceiling_records: list[dict[str, Any]] | None = None,
    utility_slack: float = 0.005,
    privacy_weight: float = 0.55,
    utility_weight: float = 0.45,
    utility_switch_min: float = 0.002,
    privacy_switch_min: float = 0.005,
    floor_mode: str = "hard",
    soft_fidelity_floor_eps: float | None = None,
    soft_trend_floor_eps: float | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    reference_tag = "fidelity_ceiling_anchor"
    if not pareto_keep_records:
        return pareto_keep_df, pareto_keep_records, pareto_report, {
            "enabled": False,
            "reason": "empty_pareto_keep",
        }

    exact_by_id = {int(record.get("candidate_id", idx)): record for idx, record in enumerate(exact_records)}

    def _candidate_key(records: list[dict[str, Any]]) -> tuple[int, ...]:
        return tuple(sorted(int(record.get("candidate_id", idx)) for idx, record in enumerate(records)))

    def _utility_pref(report: dict[str, Any]) -> float | None:
        if not report.get("available", False):
            return None
        return _utility_preference_value(report.get("metric"), report.get("overall"))

    def _normalize(values: list[float | None]) -> list[float]:
        valid_values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
        if not valid_values:
            return [0.0 for _ in values]
        min_value = min(valid_values)
        max_value = max(valid_values)
        if max_value - min_value <= 1e-12:
            return [1.0 if value is not None and np.isfinite(float(value)) else 0.0 for value in values]
        normalized: list[float] = []
        for value in values:
            if value is None or not np.isfinite(float(value)):
                normalized.append(0.0)
            else:
                normalized.append((float(value) - min_value) / (max_value - min_value))
        return normalized

    unique_candidates: dict[tuple[int, ...], dict[str, Any]] = {}

    def _register_candidate(
        source_tag: str,
        records: list[dict[str, Any]],
        df: pd.DataFrame,
        source_report: dict[str, Any] | None = None,
    ) -> None:
        if not records:
            return
        key = _candidate_key(records)
        if not key:
            return
        bucket = unique_candidates.setdefault(
            key,
            {
                "key": key,
                "records": records,
                "df": df.reset_index(drop=True),
                "sources": [],
                "source_reports": {},
            },
        )
        if source_tag not in bucket["sources"]:
            bucket["sources"].append(source_tag)
        if source_report is not None:
            bucket["source_reports"][source_tag] = source_report

    _register_candidate("pareto_main", pareto_keep_records, pareto_keep_df, pareto_report)
    for tag, family_records in pareto_family_records.items():
        family_df = pareto_family_df.get(tag)
        if family_df is None:
            family_df = _records_to_df(family_records, selector.column_order)
        _register_candidate(tag, family_records, family_df, pareto_family_reports.get(tag))
    if fidelity_ceiling_records:
        ceiling_df = fidelity_ceiling_df
        if ceiling_df is None:
            ceiling_df = _records_to_df(fidelity_ceiling_records, selector.column_order)
        _register_candidate(reference_tag, fidelity_ceiling_records, ceiling_df, floor_reference)

    candidate_items = list(unique_candidates.values())
    if not candidate_items:
        return pareto_keep_df, pareto_keep_records, pareto_report, {
            "enabled": False,
            "reason": "no_finalist_candidates",
        }

    target_fid_1d = None
    target_fid_2d = None
    normalized_floor_mode = str(floor_mode or "hard").strip().lower()
    if normalized_floor_mode not in {"hard", "soft"}:
        normalized_floor_mode = "hard"
    if floor_reference is not None:
        fid_eps = (
            float(selector.final_fidelity_floor_eps)
            if normalized_floor_mode != "soft" or soft_fidelity_floor_eps is None
            else float(soft_fidelity_floor_eps)
        )
        trend_eps = (
            float(selector.final_trend_floor_eps)
            if normalized_floor_mode != "soft" or soft_trend_floor_eps is None
            else float(soft_trend_floor_eps)
        )
        target_fid_1d = max(0.0, float(floor_reference.get("fidelity_1d", 0.0)) - fid_eps)
        target_fid_2d = max(0.0, float(floor_reference.get("fidelity_2d", 0.0)) - trend_eps)

    evaluated_candidates: list[dict[str, Any]] = []
    for item in candidate_items:
        candidate_ids = list(item["key"])
        utility_report = compute_utility_exact_metrics(
            selector,
            item["df"],
            selector.holdout_df,
            search_holdout_used=True,
        )
        utility_pref = _utility_pref(utility_report)
        fidelity_1d = float(selector.compute_dataset_fidelity(item["df"]))
        fidelity_2d = float(selector.compute_dataset_pair_fidelity(item["df"]))
        privacy_mean = float(selector.compute_dataset_privacy(item["df"]))
        utility_proxy_values = [
            float(exact_by_id[candidate_id].get("pareto_util_proxy_obj", 0.0))
            for candidate_id in candidate_ids
            if candidate_id in exact_by_id
        ]
        proxy_utility_mean = float(np.mean(utility_proxy_values)) if utility_proxy_values else 0.0
        floor_satisfied = bool(
            target_fid_1d is None
            or (
                fidelity_1d >= float(target_fid_1d) - 1e-12
                and fidelity_2d >= float(target_fid_2d) - 1e-12
            )
        )
        evaluated_candidates.append(
            {
                "key": item["key"],
                "candidate_ids": candidate_ids,
                "records": item["records"],
                "df": item["df"],
                "sources": sorted(item["sources"]),
                "matches_reference": bool(reference_tag in item["sources"]),
                "reference_only": bool(all(source == reference_tag for source in item["sources"])),
                "fidelity_1d": fidelity_1d,
                "fidelity_2d": fidelity_2d,
                "privacy_mean": privacy_mean,
                "utility_proxy_mean": proxy_utility_mean,
                "utility_exact_report": utility_report,
                "utility_exact_overall": utility_report.get("overall"),
                "utility_exact_metric": utility_report.get("metric"),
                "utility_exact_available": bool(utility_report.get("available", False)),
                "utility_pref": utility_pref,
                "floor_satisfied": floor_satisfied,
            }
        )

    privacy_norm = _normalize([item["privacy_mean"] for item in evaluated_candidates])
    utility_norm = _normalize([item["utility_pref"] for item in evaluated_candidates])
    for idx, item in enumerate(evaluated_candidates):
        item["privacy_norm"] = float(privacy_norm[idx])
        item["utility_norm"] = float(utility_norm[idx])
        item["rerank_score"] = float(privacy_weight) * float(item["privacy_norm"]) + float(utility_weight) * float(item["utility_norm"])

    reference_candidate = next((item for item in evaluated_candidates if item["matches_reference"]), None)
    reference_utility_pref = reference_candidate.get("utility_pref") if reference_candidate is not None else None
    reference_utility_overall = reference_candidate.get("utility_exact_overall") if reference_candidate is not None else None
    reference_utility_metric = reference_candidate.get("utility_exact_metric") if reference_candidate is not None else None

    for item in evaluated_candidates:
        if reference_utility_pref is None or item["utility_pref"] is None:
            item["utility_guard"] = True
        else:
            item["utility_guard"] = bool(float(item["utility_pref"]) >= float(reference_utility_pref) - float(utility_slack))
        item["reference_utility_delta"] = (
            None
            if reference_utility_pref is None or item["utility_pref"] is None
            else float(item["utility_pref"]) - float(reference_utility_pref)
        )
        item["reference_privacy_delta"] = (
            0.0
            if reference_candidate is None
            else float(item["privacy_mean"]) - float(reference_candidate["privacy_mean"])
        )
        item["reference_switch_margin"] = bool(
            reference_candidate is None
            or item["matches_reference"]
            or (
                (
                    item["reference_utility_delta"] is not None
                    and float(item["reference_utility_delta"]) >= float(utility_switch_min)
                )
                or float(item["reference_privacy_delta"]) >= float(privacy_switch_min)
            )
        )

    selectable_candidates = [item for item in evaluated_candidates if not item["reference_only"]]

    def _sort_key(item: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        utility_pref = item["utility_pref"]
        return (
            float(item["rerank_score"]),
            float(item["privacy_mean"]),
            float("-inf") if utility_pref is None else float(utility_pref),
            float(item["utility_proxy_mean"]),
            float(item["fidelity_2d"]),
            float(item["fidelity_1d"]),
        )

    rerank_stage = "no_selectable_candidate"
    if selectable_candidates:
        guarded_candidates = [
            item
            for item in selectable_candidates
            if bool(item["floor_satisfied"]) and bool(item["utility_guard"]) and bool(item["reference_switch_margin"])
        ]
        feasible_candidates = [item for item in selectable_candidates if bool(item["floor_satisfied"])]
        if guarded_candidates:
            selected_candidate = max(guarded_candidates, key=_sort_key)
            rerank_stage = "floor_utility_guard_and_switch_margin"
        elif reference_candidate is not None:
            selected_candidate = reference_candidate
            rerank_stage = "reference_fallback_no_switch_margin"
        elif feasible_candidates:
            selected_candidate = max(feasible_candidates, key=_sort_key)
            rerank_stage = "floor_only"
        else:
            selected_candidate = max(selectable_candidates, key=_sort_key)
            rerank_stage = "best_effort_no_floor"
    else:
        selected_candidate = reference_candidate if reference_candidate is not None else max(evaluated_candidates, key=_sort_key)
        rerank_stage = "reference_only"

    selected_key = tuple(selected_candidate["key"])
    main_key = _candidate_key(pareto_keep_records)
    selected_exact_records = [
        exact_by_id[candidate_id] for candidate_id in selected_candidate["candidate_ids"] if candidate_id in exact_by_id
    ]

    rerank_report = {
        "enabled": True,
        "search_split": "derived_holdout",
        "candidate_count_unique": int(len(evaluated_candidates)),
        "candidate_count_selectable": int(len(selectable_candidates)),
        "selection_stage": rerank_stage,
        "score_weights": {
            "privacy": float(privacy_weight),
            "utility_exact_search": float(utility_weight),
        },
        "utility_guard": {
            "reference_name": floor_reference.get("name") if floor_reference is not None else None,
            "reference_metric": reference_utility_metric,
            "reference_overall": reference_utility_overall,
            "slack": float(utility_slack),
        },
        "reference_switch_margin": {
            "utility_min": float(utility_switch_min),
            "privacy_min": float(privacy_switch_min),
        },
        "floor_thresholds": {
            "mode": normalized_floor_mode,
            "fidelity_1d": target_fid_1d,
            "fidelity_2d": target_fid_2d,
        },
        "selected_sources": selected_candidate["sources"],
        "selected_matches_reference": bool(selected_candidate["matches_reference"]),
        "main_candidate_retained": bool(selected_key == main_key),
        "selected_summary": {
            "rows": int(len(selected_candidate["records"])),
            "fidelity_1d": float(selected_candidate["fidelity_1d"]),
            "fidelity_2d": float(selected_candidate["fidelity_2d"]),
            "privacy_mean": float(selected_candidate["privacy_mean"]),
            "utility_proxy_mean": float(selected_candidate["utility_proxy_mean"]),
            "utility_exact_metric": selected_candidate["utility_exact_metric"],
            "utility_exact_overall": selected_candidate["utility_exact_overall"],
            "rerank_score": float(selected_candidate["rerank_score"]),
            "floor_satisfied": bool(selected_candidate["floor_satisfied"]),
            "utility_guard": bool(selected_candidate["utility_guard"]),
            "reference_utility_delta": selected_candidate["reference_utility_delta"],
            "reference_privacy_delta": float(selected_candidate["reference_privacy_delta"]),
            "reference_switch_margin": bool(selected_candidate["reference_switch_margin"]),
        },
        "reference_summary": (
            None
            if reference_candidate is None
            else {
                "sources": reference_candidate["sources"],
                "rows": int(len(reference_candidate["records"])),
                "fidelity_1d": float(reference_candidate["fidelity_1d"]),
                "fidelity_2d": float(reference_candidate["fidelity_2d"]),
                "privacy_mean": float(reference_candidate["privacy_mean"]),
                "utility_exact_metric": reference_candidate["utility_exact_metric"],
                "utility_exact_overall": reference_candidate["utility_exact_overall"],
                "utility_proxy_mean": float(reference_candidate["utility_proxy_mean"]),
            }
        ),
        "candidates": [
            {
                "sources": item["sources"],
                "matches_reference": bool(item["matches_reference"]),
                "reference_only": bool(item["reference_only"]),
                "rows": int(len(item["records"])),
                "fidelity_1d": float(item["fidelity_1d"]),
                "fidelity_2d": float(item["fidelity_2d"]),
                "privacy_mean": float(item["privacy_mean"]),
                "utility_proxy_mean": float(item["utility_proxy_mean"]),
                "utility_exact_metric": item["utility_exact_metric"],
                "utility_exact_available": bool(item["utility_exact_available"]),
                "utility_exact_overall": item["utility_exact_overall"],
                "floor_satisfied": bool(item["floor_satisfied"]),
                "utility_guard": bool(item["utility_guard"]),
                "reference_utility_delta": item["reference_utility_delta"],
                "reference_privacy_delta": float(item["reference_privacy_delta"]),
                "reference_switch_margin": bool(item["reference_switch_margin"]),
                "privacy_norm": float(item["privacy_norm"]),
                "utility_norm": float(item["utility_norm"]),
                "rerank_score": float(item["rerank_score"]),
            }
            for item in sorted(evaluated_candidates, key=_sort_key, reverse=True)
        ],
    }

    updated_pareto_report = dict(pareto_report)
    updated_pareto_report["selected"] = int(len(selected_candidate["records"]))
    updated_pareto_report["selected_privacy_component_mean"] = (
        float(np.mean([float(record.get("pareto_priv_obj", 0.0)) for record in selected_exact_records]))
        if selected_exact_records
        else 0.0
    )
    updated_pareto_report["selected_privacy_raw_mean"] = (
        float(np.mean([float(record.get("privacy_score_selected", 0.0)) for record in selected_exact_records]))
        if selected_exact_records
        else 0.0
    )
    updated_pareto_report["selected_utility_mean"] = (
        float(np.mean([float(record.get("pareto_util_proxy_obj", 0.0)) for record in selected_exact_records]))
        if selected_exact_records
        else 0.0
    )
    updated_pareto_report["finalist_rerank"] = {
        "enabled": True,
        "selection_stage": rerank_stage,
        "selected_sources": selected_candidate["sources"],
        "selected_matches_reference": bool(selected_candidate["matches_reference"]),
        "main_candidate_retained": bool(selected_key == main_key),
        "candidate_count_unique": int(len(evaluated_candidates)),
        "candidate_count_selectable": int(len(selectable_candidates)),
        "search_split": "derived_holdout",
        "utility_guard_slack": float(utility_slack),
        "utility_switch_min": float(utility_switch_min),
        "privacy_switch_min": float(privacy_switch_min),
        "reference_search_utility": reference_utility_overall,
        "reference_search_metric": reference_utility_metric,
        "selected_search_utility": selected_candidate["utility_exact_overall"],
        "selected_search_metric": selected_candidate["utility_exact_metric"],
        "selected_rerank_score": float(selected_candidate["rerank_score"]),
        "selected_floor_satisfied": bool(selected_candidate["floor_satisfied"]),
        "selected_utility_guard": bool(selected_candidate["utility_guard"]),
        "selected_reference_switch_margin": bool(selected_candidate["reference_switch_margin"]),
    }
    return selected_candidate["df"], selected_candidate["records"], updated_pareto_report, rerank_report


def _selection_name(prefix: str, tag: str) -> str:
    return f"{prefix}_{tag}"


def _save_family_csvs(
    versions_dir: Path,
    prefix: str,
    family_frames: dict[str, pd.DataFrame],
) -> None:
    for tag, df in family_frames.items():
        save_csv(versions_dir / f"{prefix}_{tag}.csv", df)


def _posterior_jsd_report(selector: ParetoSelector, df: pd.DataFrame, epsilon: float) -> dict[str, Any]:
    jsd_per_column = selector.compute_column_jsd(df)
    violations = {column: value for column, value in jsd_per_column.items() if value > epsilon}
    return {
        "epsilon": epsilon,
        "jsd_per_column": jsd_per_column,
        "max_jsd": float(max(jsd_per_column.values()) if jsd_per_column else 0.0),
        "pass": len(violations) == 0,
        "violations": violations,
    }


def _compute_mode_vs_tail_tstr(selector: ParetoSelector, syn_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    if syn_df.empty or test_df.empty or selector.target_column not in syn_df.columns:
        return {"available": False, "reason": "empty_input"}

    train_target = syn_df[selector.target_column].astype(str)
    test_target = test_df[selector.target_column].astype(str)
    classes = sorted(set(train_target.tolist()) | set(test_target.tolist()))
    if len(classes) != 2:
        return {"available": False, "reason": "non_binary_target"}

    positive_label = classes[-1]
    y_train = (train_target == positive_label).astype(int)
    y_test = (test_target == positive_label).astype(int)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return {"available": False, "reason": "single_class"}

    num_features = [
        column
        for column in selector.feature_columns
        if selector.schema_card["columns"][column]["type"] in {"numerical", "discrete_numerical"}
    ]
    cat_features = [
        column for column in selector.feature_columns if selector.schema_card["columns"][column]["type"] == "categorical"
    ]

    transformers: list[tuple[str, Any, list[str]]] = []
    if num_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            )
        )
    if cat_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _make_ohe()),
                    ]
                ),
                cat_features,
            )
        )
    if not transformers:
        return {"available": False, "reason": "no_features"}

    model = Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(syn_df[selector.feature_columns], y_train)
    y_prob = model.predict_proba(test_df[selector.feature_columns])[:, 1]

    gate_probs = selector._prob_geomean_for_df(test_df.reset_index(drop=True), columns=selector.feature_columns)
    quantiles = np.quantile(gate_probs, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    quantiles = np.unique(quantiles)
    if len(quantiles) < 4:
        quantiles = np.array([gate_probs.min() - 1e-12, gate_probs.mean(), gate_probs.mean(), gate_probs.max() + 1e-12])
    bins = np.digitize(gate_probs, quantiles[1:-1], right=False)

    def _auc(mask: np.ndarray) -> float | None:
        if int(mask.sum()) < 2:
            return None
        subset_y = y_test[mask]
        if subset_y.nunique() < 2:
            return None
        return float(roc_auc_score(subset_y, y_prob[mask]))

    return {
        "available": True,
        "metric": "roc_auc",
        "overall": float(roc_auc_score(y_test, y_prob)),
        "tail": _auc(bins == 0),
        "middle": _auc(bins == 1),
        "mode": _auc(bins == 2),
        "rows": {
            "tail": int((bins == 0).sum()),
            "middle": int((bins == 1).sum()),
            "mode": int((bins == 2).sum()),
        },
    }


def _evaluate_selection(
    selection_name: str,
    df: pd.DataFrame,
    keep_records: list[dict[str, Any]],
    pool_records: list[dict[str, Any]],
    selector: ParetoSelector,
    evaluator: TabDiffSelectionEvaluator,
    eval_dir: Path,
    test_df: pd.DataFrame,
    jsd_epsilon: float,
    rare_threshold: float,
) -> dict[str, Any]:
    metrics, extras = evaluator.evaluate(df)
    rarity_report = selector.compute_rarity_stratified_keep_rate(pool_records, keep_records)
    rare_bin_report = selector.compute_rare_bin_inflation(df, rare_threshold=rare_threshold)
    mode_vs_tail_report = _compute_mode_vs_tail_tstr(selector, df, test_df)
    utility_exact_report = compute_utility_exact_metrics(selector, df, test_df)
    posterior_jsd = _posterior_jsd_report(selector, df, epsilon=jsd_epsilon)
    extras = {
        **extras,
        "rarity_stratified_keep_rate": {"rows": rarity_report},
        "rare_bin_inflation_report": {"rows": rare_bin_report},
        "mode_vs_tail_tstr_report": mode_vs_tail_report,
        "utility_exact_report": utility_exact_report,
        "posterior_jsd_filter": posterior_jsd,
    }
    _save_eval_extras(eval_dir=eval_dir, selection_name=selection_name, extras=extras)
    save_json(eval_dir / selection_name / "utility_metrics_summary.json", utility_exact_report)

    raw_dcr = float(metrics.get("dcr", 0.0)) if "dcr" in metrics else None
    dcr_balance_error = abs(raw_dcr - 0.5) if raw_dcr is not None else None
    dcr_privacy_reward = 1.0 - dcr_balance_error if dcr_balance_error is not None else None
    summary = {
        "rows": int(len(df)),
        "fidelity": selector.compute_dataset_fidelity(df),
        "shape": float(metrics.get("density/Shape", 0.0)),
        "trend": float(metrics.get("density/Trend", 0.0)),
        "density_overall": float(metrics.get("density/Overall", 0.0)),
        "privacy": selector.compute_dataset_privacy(df),
        "privacy_mean_nn_distance": selector.compute_dataset_mean_nn_distance(df),
        "dcr": raw_dcr,
        "dcr_privacy": dcr_privacy_reward,
        "raw_dcr_real_closer_rate": raw_dcr,
        "target_raw_dcr": 0.5,
        "dcr_balance_error_abs": dcr_balance_error,
        "dcr_privacy_reward": dcr_privacy_reward,
        "dcr_semantics": "raw DCR is mean(distance_to_train < distance_to_test); best near 0.5; dcr_privacy_reward is higher-better",
        "metric_directions": build_metric_directions(utility_exact_report.get("metric")),
        "posterior_jsd_max": float(posterior_jsd["max_jsd"]),
        "posterior_jsd_pass": bool(posterior_jsd["pass"]),
        "mode_region_tstr": mode_vs_tail_report.get("mode"),
        "tail_region_tstr": mode_vs_tail_report.get("tail"),
        "utility_exact_metric": utility_exact_report.get("metric"),
        "utility_exact_available": bool(utility_exact_report.get("available", False)),
        "utility_exact_overall": utility_exact_report.get("overall"),
        "utility_exact_mode": utility_exact_report.get("mode"),
        "utility_exact_middle": utility_exact_report.get("middle"),
        "utility_exact_tail": utility_exact_report.get("tail"),
        "rarity_stratified_keep_rate": rarity_report,
        "rare_bin_inflation_top": rare_bin_report[:20],
    }
    save_json(eval_dir / selection_name / "metrics_summary.json", summary)
    return summary


def _normalize_family_points(point_sets: list[np.ndarray]) -> list[np.ndarray]:
    if not point_sets:
        return []
    combined = np.vstack([points for points in point_sets if len(points) > 0])
    lower = combined.min(axis=0)
    upper = combined.max(axis=0)
    denom = np.where((upper - lower) <= 1e-12, 1.0, upper - lower)
    return [(points - lower) / denom if len(points) > 0 else points for points in point_sets]


def _non_dominated_indices(points: np.ndarray) -> list[int]:
    if len(points) == 0:
        return []
    ge = np.all(points[None, :, :] >= points[:, None, :], axis=2)
    gt = np.any(points[None, :, :] > points[:, None, :], axis=2)
    dominated = np.any(ge & gt, axis=1)
    return np.flatnonzero(~dominated).tolist()


def _hypervolume_2d(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    nd = points[_non_dominated_indices(points)]
    if len(nd) == 0:
        return 0.0
    order = np.argsort(nd[:, 0])
    nd = nd[order]
    hv = 0.0
    prev_x = 0.0
    for x, y in nd:
        hv += max(0.0, float(x) - prev_x) * max(0.0, float(y))
        prev_x = max(prev_x, float(x))
    return float(hv)


def _hypervolume_3d(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    nd = points[_non_dominated_indices(points)]
    if len(nd) == 0:
        return 0.0
    xs = np.unique(np.sort(nd[:, 0]))
    hv = 0.0
    prev_x = 0.0
    active = nd.copy()
    for x in xs:
        hv += max(0.0, float(x) - prev_x) * _hypervolume_2d(active[:, 1:])
        active = active[active[:, 0] > float(x) + 1e-12]
        prev_x = float(x)
    return float(hv)


def _hypervolume(points: np.ndarray) -> float:
    if points.ndim != 2:
        return 0.0
    if points.shape[1] == 2:
        return _hypervolume_2d(points)
    if points.shape[1] == 3:
        return _hypervolume_3d(points)
    raise ValueError(f"Unsupported hypervolume dimension: {points.shape[1]}")


def _igd(points: np.ndarray, reference_front: np.ndarray) -> float:
    if len(points) == 0 or len(reference_front) == 0:
        return float("inf")
    pairwise = np.linalg.norm(reference_front[:, None, :] - points[None, :, :], axis=2)
    return float(np.mean(np.min(pairwise, axis=1)))


def _dominates_metrics(a: dict[str, Any], b: dict[str, Any], *, objective_keys: list[str]) -> bool:
    return all(a[key] >= b[key] for key in objective_keys) and any(a[key] > b[key] for key in objective_keys)


def _utility_preference_value(metric: Any, overall: Any) -> float | None:
    if overall is None:
        return None
    value = float(overall)
    if not np.isfinite(value):
        return None
    metric_name = str(metric or "").strip().lower()
    if metric_name in {"rmse", "mae", "mse"}:
        return -value
    return value


def _compare_family_space(
    pareto_family_metrics: dict[str, dict[str, Any]],
    scalar_family_metrics: dict[str, dict[str, Any]],
    *,
    privacy_key: str,
) -> dict[str, Any]:
    direction_tags = [tag for tag, *_ in DIRECTION_SPECS]
    if any(
        pareto_family_metrics.get(tag, {}).get(privacy_key) is None
        or scalar_family_metrics.get(tag, {}).get(privacy_key) is None
        or pareto_family_metrics.get(tag, {}).get("trend") is None
        or scalar_family_metrics.get(tag, {}).get("trend") is None
        for tag in direction_tags
    ):
        return {
            "available": False,
            "privacy_key": privacy_key,
            "pareto_hv": None,
            "scalar_hv": None,
            "pareto_igd": None,
            "scalar_igd": None,
            "pointwise_dominance_count": None,
            "dominance_certificates": [],
        }

    pareto_points = np.asarray(
        [
            [
                pareto_family_metrics[tag]["fidelity"],
                pareto_family_metrics[tag]["trend"],
                pareto_family_metrics[tag][privacy_key],
            ]
            for tag in direction_tags
        ],
        dtype=float,
    )
    scalar_points = np.asarray(
        [
            [
                scalar_family_metrics[tag]["fidelity"],
                scalar_family_metrics[tag]["trend"],
                scalar_family_metrics[tag][privacy_key],
            ]
            for tag in direction_tags
        ],
        dtype=float,
    )
    normalized_pareto, normalized_scalar = _normalize_family_points([pareto_points, scalar_points])
    reference_union = np.vstack([normalized_pareto, normalized_scalar])
    reference_front = reference_union[_non_dominated_indices(reference_union)]

    certificates: list[dict[str, Any]] = []
    dominance_count = 0
    for tag in direction_tags:
        pareto_metrics = pareto_family_metrics[tag]
        scalar_metrics = scalar_family_metrics[tag]
        dominates = _dominates_metrics(
            pareto_metrics,
            scalar_metrics,
            objective_keys=["fidelity", "trend", privacy_key],
        )
        if dominates:
            dominance_count += 1
        certificates.append(
            {
                "direction": tag,
                "privacy_key": privacy_key,
                "pareto_fidelity": pareto_metrics["fidelity"],
                "pareto_trend": pareto_metrics["trend"],
                "pareto_privacy": pareto_metrics[privacy_key],
                "scalar_fidelity": scalar_metrics["fidelity"],
                "scalar_trend": scalar_metrics["trend"],
                "scalar_privacy": scalar_metrics[privacy_key],
                "dominates": dominates,
                "strictly_better_fidelity": pareto_metrics["fidelity"] > scalar_metrics["fidelity"],
                "strictly_better_trend": pareto_metrics["trend"] > scalar_metrics["trend"],
                "strictly_better_privacy": pareto_metrics[privacy_key] > scalar_metrics[privacy_key],
            }
        )

    return {
        "available": True,
        "privacy_key": privacy_key,
        "pareto_hv": _hypervolume(normalized_pareto),
        "scalar_hv": _hypervolume(normalized_scalar),
        "pareto_igd": _igd(normalized_pareto, reference_front),
        "scalar_igd": _igd(normalized_scalar, reference_front),
        "pointwise_dominance_count": dominance_count,
        "dominance_certificates": certificates,
    }


def _compare_family_utility_space(
    pareto_family_metrics: dict[str, dict[str, Any]],
    scalar_family_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    direction_tags = [tag for tag, *_ in DIRECTION_SPECS]
    utility_key = "utility_exact_overall"
    pareto_utility_pref = {
        tag: _utility_preference_value(
            pareto_family_metrics.get(tag, {}).get("utility_exact_metric"),
            pareto_family_metrics.get(tag, {}).get(utility_key),
        )
        for tag in direction_tags
    }
    scalar_utility_pref = {
        tag: _utility_preference_value(
            scalar_family_metrics.get(tag, {}).get("utility_exact_metric"),
            scalar_family_metrics.get(tag, {}).get(utility_key),
        )
        for tag in direction_tags
    }
    if any(
        pareto_utility_pref[tag] is None
        or scalar_utility_pref[tag] is None
        or pareto_family_metrics.get(tag, {}).get("fidelity") is None
        or scalar_family_metrics.get(tag, {}).get("fidelity") is None
        or pareto_family_metrics.get(tag, {}).get("trend") is None
        or scalar_family_metrics.get(tag, {}).get("trend") is None
        for tag in direction_tags
    ):
        return {
            "available": False,
            "utility_key": utility_key,
            "pareto_hv": None,
            "scalar_hv": None,
            "pareto_igd": None,
            "scalar_igd": None,
            "pointwise_dominance_count": None,
            "dominance_certificates": [],
        }

    pareto_points = np.asarray(
        [
            [
                pareto_family_metrics[tag]["fidelity"],
                pareto_family_metrics[tag]["trend"],
                pareto_utility_pref[tag],
            ]
            for tag in direction_tags
        ],
        dtype=float,
    )
    scalar_points = np.asarray(
        [
            [
                scalar_family_metrics[tag]["fidelity"],
                scalar_family_metrics[tag]["trend"],
                scalar_utility_pref[tag],
            ]
            for tag in direction_tags
        ],
        dtype=float,
    )
    normalized_pareto, normalized_scalar = _normalize_family_points([pareto_points, scalar_points])
    reference_union = np.vstack([normalized_pareto, normalized_scalar])
    reference_front = reference_union[_non_dominated_indices(reference_union)]

    certificates: list[dict[str, Any]] = []
    dominance_count = 0
    for tag in direction_tags:
        pareto_metrics = dict(pareto_family_metrics[tag])
        scalar_metrics = dict(scalar_family_metrics[tag])
        pareto_metrics["_utility_pref"] = float(pareto_utility_pref[tag])
        scalar_metrics["_utility_pref"] = float(scalar_utility_pref[tag])
        dominates = _dominates_metrics(
            pareto_metrics,
            scalar_metrics,
            objective_keys=["fidelity", "trend", "_utility_pref"],
        )
        if dominates:
            dominance_count += 1
        certificates.append(
            {
                "direction": tag,
                "utility_key": utility_key,
                "utility_preference_key": "_utility_pref",
                "pareto_fidelity": pareto_metrics["fidelity"],
                "pareto_trend": pareto_metrics["trend"],
                "pareto_utility": pareto_metrics[utility_key],
                "pareto_utility_preference": pareto_metrics["_utility_pref"],
                "scalar_fidelity": scalar_metrics["fidelity"],
                "scalar_trend": scalar_metrics["trend"],
                "scalar_utility": scalar_metrics[utility_key],
                "scalar_utility_preference": scalar_metrics["_utility_pref"],
                "dominates": dominates,
                "strictly_better_fidelity": pareto_metrics["fidelity"] > scalar_metrics["fidelity"],
                "strictly_better_trend": pareto_metrics["trend"] > scalar_metrics["trend"],
                "strictly_better_utility": pareto_metrics["_utility_pref"] > scalar_metrics["_utility_pref"],
            }
        )

    return {
        "available": True,
        "utility_key": utility_key,
        "pareto_hv": _hypervolume(normalized_pareto),
        "scalar_hv": _hypervolume(normalized_scalar),
        "pareto_igd": _igd(normalized_pareto, reference_front),
        "scalar_igd": _igd(normalized_scalar, reference_front),
        "pointwise_dominance_count": dominance_count,
        "dominance_certificates": certificates,
    }


def _compare_families(
    pareto_family_metrics: dict[str, dict[str, Any]],
    scalar_family_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dcr_space = _compare_family_space(
        pareto_family_metrics=pareto_family_metrics,
        scalar_family_metrics=scalar_family_metrics,
        privacy_key="dcr_privacy",
    )
    proxy_space = _compare_family_space(
        pareto_family_metrics=pareto_family_metrics,
        scalar_family_metrics=scalar_family_metrics,
        privacy_key="privacy",
    )
    utility_space = _compare_family_utility_space(
        pareto_family_metrics=pareto_family_metrics,
        scalar_family_metrics=scalar_family_metrics,
    )
    primary = dcr_space if dcr_space["available"] else proxy_space
    return {
        "primary_privacy_key": primary["privacy_key"],
        "pareto_hv": primary["pareto_hv"],
        "scalar_hv": primary["scalar_hv"],
        "pareto_igd": primary["pareto_igd"],
        "scalar_igd": primary["scalar_igd"],
        "pointwise_dominance_count": primary["pointwise_dominance_count"],
        "dominance_certificates": primary["dominance_certificates"],
        "dcr_space": dcr_space,
        "proxy_space": proxy_space,
        "utility_space": utility_space,
    }


def _selection_delta(base_metrics: dict[str, Any], target_metrics: dict[str, Any]) -> dict[str, Any]:
    if not base_metrics or not target_metrics:
        return {
            "available": False,
            "fidelity_drop": None,
            "trend_drop": None,
            "dcr_gain": None,
            "privacy_gain": None,
        }

    base_dcr = base_metrics.get("dcr")
    target_dcr = target_metrics.get("dcr")
    if base_metrics.get("dcr_privacy") is not None and target_metrics.get("dcr_privacy") is not None:
        dcr_gain = float(target_metrics["dcr_privacy"]) - float(base_metrics["dcr_privacy"])
    elif base_dcr is not None and target_dcr is not None:
        dcr_gain = abs(float(base_dcr) - 0.5) - abs(float(target_dcr) - 0.5)
    else:
        dcr_gain = None

    privacy_gain = None
    if base_metrics.get("privacy") is not None and target_metrics.get("privacy") is not None:
        privacy_gain = float(target_metrics["privacy"]) - float(base_metrics["privacy"])

    return {
        "available": True,
        "fidelity_drop": float(base_metrics.get("fidelity", 0.0) - target_metrics.get("fidelity", 0.0)),
        "trend_drop": float(base_metrics.get("trend", 0.0) - target_metrics.get("trend", 0.0)),
        "dcr_gain": dcr_gain,
        "privacy_gain": privacy_gain,
    }


def _subset_gate_metrics(
    selector: ParetoSelector,
    evaluator: TabDiffSelectionEvaluator,
    df: pd.DataFrame,
) -> dict[str, Any]:
    metrics, _ = evaluator.evaluate(df)
    raw_dcr = float(metrics.get("dcr", 0.0)) if "dcr" in metrics else None
    dcr_balance_error = abs(raw_dcr - 0.5) if raw_dcr is not None else None
    dcr_privacy_reward = 1.0 - dcr_balance_error if dcr_balance_error is not None else None
    return {
        "rows": int(len(df)),
        "fidelity": selector.compute_dataset_fidelity(df),
        "trend": float(metrics.get("density/Trend", 0.0)),
        "dcr": raw_dcr,
        "dcr_privacy": dcr_privacy_reward,
        "raw_dcr_real_closer_rate": raw_dcr,
        "target_raw_dcr": 0.5,
        "dcr_balance_error_abs": dcr_balance_error,
        "dcr_privacy_reward": dcr_privacy_reward,
        "metric_directions": build_metric_directions(None),
        "privacy": selector.compute_dataset_privacy(df),
    }


def _build_preselect_gate_report(
    raw_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    *,
    candidate_mode: str = "candidate_preselect",
    baseline_mode: str = "baseline_preselect",
    fidelity_max_drop: float = 0.01,
    trend_max_drop: float = 0.01,
    dcr_min_gain: float = 0.02,
    candidate_vs_baseline_max_drop: float = 0.001,
    candidate_vs_baseline_min_dcr_gain: float = 0.002,
) -> dict[str, Any]:
    candidate_delta = _selection_delta(raw_metrics, candidate_metrics)
    baseline_delta = _selection_delta(raw_metrics, baseline_metrics)
    candidate_vs_baseline = _selection_delta(baseline_metrics, candidate_metrics)

    def _passes(delta: dict[str, Any]) -> bool:
        return bool(
            delta.get("available")
            and delta.get("fidelity_drop") is not None
            and delta.get("trend_drop") is not None
            and delta.get("dcr_gain") is not None
            and float(delta["fidelity_drop"]) <= float(fidelity_max_drop)
            and float(delta["trend_drop"]) <= float(trend_max_drop)
            and float(delta["dcr_gain"]) >= float(dcr_min_gain)
        )

    candidate_pass = _passes(candidate_delta)
    baseline_pass = _passes(baseline_delta)
    candidate_beats_baseline = bool(
        candidate_vs_baseline.get("available")
        and candidate_vs_baseline.get("fidelity_drop") is not None
        and candidate_vs_baseline.get("trend_drop") is not None
        and candidate_vs_baseline.get("dcr_gain") is not None
        and float(candidate_vs_baseline["fidelity_drop"]) <= float(candidate_vs_baseline_max_drop)
        and float(candidate_vs_baseline["trend_drop"]) <= float(candidate_vs_baseline_max_drop)
        and float(candidate_vs_baseline["dcr_gain"]) >= float(candidate_vs_baseline_min_dcr_gain)
    )

    if candidate_pass and candidate_beats_baseline:
        selected_source = "candidate_preselect"
        selected_mode = candidate_mode
    else:
        selected_source = "baseline_preselect"
        selected_mode = baseline_mode

    return {
        "thresholds": {
            "fidelity_max_drop": float(fidelity_max_drop),
            "trend_max_drop": float(trend_max_drop),
            "dcr_min_gain": float(dcr_min_gain),
        },
        "raw_reference": raw_metrics,
        "candidate": {
            "mode": candidate_mode,
            "metrics": candidate_metrics,
            "delta_vs_raw": candidate_delta,
            "delta_vs_baseline": candidate_vs_baseline,
            "pass": bool(candidate_pass),
            "beats_baseline": bool(candidate_beats_baseline),
        },
        "baseline": {
            "mode": baseline_mode,
            "metrics": baseline_metrics,
            "delta_vs_raw": baseline_delta,
            "pass": bool(baseline_pass),
        },
        "selected_source": selected_source,
        "selected_mode": selected_mode,
        "selected_pass": bool(candidate_pass if selected_source == "candidate_preselect" else baseline_pass),
        "fallback_applied": bool(selected_source != "candidate_preselect"),
        "candidate_vs_baseline_thresholds": {
            "fidelity_max_drop": float(candidate_vs_baseline_max_drop),
            "trend_max_drop": float(candidate_vs_baseline_max_drop),
            "dcr_min_gain": float(candidate_vs_baseline_min_dcr_gain),
        },
    }


def _build_selection_gate_report(
    selection_metrics: dict[str, dict[str, Any]],
    family_comparison: dict[str, Any],
    *,
    fidelity_max_drop: float = 0.01,
    trend_max_drop: float = 0.01,
    dcr_min_gain: float = 0.02,
    dominance_min_count: int = 3,
) -> dict[str, Any]:
    raw_metrics = selection_metrics.get("raw_full", {})
    pareto_metrics = selection_metrics.get("pareto", {})
    scalar_metrics = selection_metrics.get("scalar", {})
    preselected_metrics = selection_metrics.get("preselected_valid", {})

    pareto_vs_raw = _selection_delta(raw_metrics, pareto_metrics)
    scalar_vs_raw = _selection_delta(raw_metrics, scalar_metrics)
    pareto_vs_preselected = _selection_delta(preselected_metrics, pareto_metrics)

    family_available = bool(family_comparison.get("pareto_hv") is not None and family_comparison.get("scalar_hv") is not None)
    family_pass = bool(
        family_available
        and float(family_comparison["pareto_hv"]) > float(family_comparison["scalar_hv"])
        and float(family_comparison["pareto_igd"]) < float(family_comparison["scalar_igd"])
        and int(family_comparison.get("pointwise_dominance_count", 0)) >= int(dominance_min_count)
    )

    single_point_pass = bool(
        pareto_vs_raw["available"]
        and pareto_vs_raw["fidelity_drop"] is not None
        and pareto_vs_raw["trend_drop"] is not None
        and pareto_vs_raw["dcr_gain"] is not None
        and float(pareto_vs_raw["fidelity_drop"]) <= float(fidelity_max_drop)
        and float(pareto_vs_raw["trend_drop"]) <= float(trend_max_drop)
        and float(pareto_vs_raw["dcr_gain"]) >= float(dcr_min_gain)
    )

    return {
        "thresholds": {
            "fidelity_max_drop": float(fidelity_max_drop),
            "trend_max_drop": float(trend_max_drop),
            "dcr_min_gain": float(dcr_min_gain),
            "dominance_min_count": int(dominance_min_count),
        },
        "family_pass": family_pass,
        "single_point_pass": single_point_pass,
        "overall_pass": bool(family_pass and single_point_pass),
        "family_gate": {
            "available": family_available,
            "pareto_hv": family_comparison.get("pareto_hv"),
            "scalar_hv": family_comparison.get("scalar_hv"),
            "pareto_igd": family_comparison.get("pareto_igd"),
            "scalar_igd": family_comparison.get("scalar_igd"),
            "pointwise_dominance_count": family_comparison.get("pointwise_dominance_count"),
        },
        "single_point_gate": {
            "raw_to_pareto": pareto_vs_raw,
            "raw_to_scalar": scalar_vs_raw,
            "preselected_to_pareto": pareto_vs_preselected,
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    progress_enabled = not args.disable_progress
    overall = _progress(total=13, desc="run_tabdiff_m5", dynamic_ncols=True, disable=not progress_enabled)
    eval_device = _resolve_eval_device(args.eval_device)
    nn_device = _resolve_nn_device(args.nn_device, eval_device)

    dataset_ctx = resolve_tabdiff_selection_context(
        dataset_name=args.dataset_name,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
    )
    synthetic_csv = _resolve_synthetic_csv(args)
    artifact_root = dataset_ctx.artifact_root if args.artifact_dir is None else args.artifact_dir
    artifact_dir = ensure_dir(artifact_root / args.exp_name)
    input_dir = ensure_dir(artifact_dir / "input")
    cards_dir = ensure_dir(artifact_dir / "cards")
    validation_dir = ensure_dir(artifact_dir / "validation")
    selection_dir = ensure_dir(artifact_dir / "selection")
    versions_dir = ensure_dir(artifact_dir / "versions")
    eval_dir = ensure_dir(artifact_dir / "eval")
    report_dir = ensure_dir(artifact_dir / "report")

    _progress_write("[1/13] load inputs")
    train_df = dataset_ctx.train_df.copy()
    holdout_df = dataset_ctx.holdout_df.copy()
    test_df = dataset_ctx.test_df.copy()
    synthetic_df = normalize_tabdiff_dataframe_columns(args.dataset_name, load_csv(synthetic_csv))
    save_csv(input_dir / "synthetic_raw.csv", synthetic_df)
    save_csv(input_dir / "eval_train.csv", train_df)
    save_csv(input_dir / "eval_holdout.csv", holdout_df)
    save_csv(input_dir / "eval_test.csv", test_df)
    save_json(input_dir / "selection_context.json", dataset_ctx.to_manifest())
    overall.update(1)

    _progress_write("[2/13] build cards")
    cards = build_and_save_cards(
        train_df=train_df,
        output_dir=cards_dir,
        seed=args.seed,
        dataset_name=args.dataset_name,
        target_column=dataset_ctx.target_column,
        categorical_columns=dataset_ctx.categorical_columns,
        numerical_columns=dataset_ctx.numerical_columns,
        discrete_numerical_columns=dataset_ctx.discrete_numerical_columns,
        privacy_sensitive_columns=dataset_ctx.privacy_sensitive_columns,
    )
    overall.update(1)

    _progress_write("[3/13] validate candidates")
    validator = TabularValidator(cards.schema_card, cards.stats_card)
    validation_bundle = validator.validate(
        _df_to_candidate_records(synthetic_df),
        show_progress=progress_enabled,
        progress_desc="validate candidates",
    )
    valid_df = validation_bundle.valid_df.reset_index(drop=True)
    save_json(validation_dir / "validator_report.json", validation_bundle.report)
    save_jsonl(validation_dir / "candidates_valid.jsonl", validation_bundle.valid_records)
    save_jsonl(validation_dir / "candidates_rejected.jsonl", validation_bundle.rejected_records)
    save_csv(versions_dir / "raw_valid.csv", valid_df)
    overall.update(1)

    _progress_write("[4/13] initialize selector and candidate pool")
    selector = ParetoSelector(
        train_df=train_df,
        holdout_df=holdout_df,
        schema_card=cards.schema_card,
        stats_card=cards.stats_card,
        seed=args.seed,
        source=args.source,
        lambda_penalty=args.lambda_penalty,
        gamma=args.gamma,
        privacy_version=args.privacy_version,
        density_reference_size=args.density_reference_size,
        nn_device=nn_device,
        nn_query_batch_size=args.nn_query_batch_size,
        nn_reference_chunk_size=args.nn_reference_chunk_size,
        final_fidelity_floor_eps=args.final_fidelity_floor_eps,
        final_trend_floor_eps=args.final_trend_floor_eps,
    )

    pool_df = valid_df.copy()
    pool_records = validation_bundle.valid_records.copy()
    if args.d_cur_source == "synthetic" and not valid_df.empty:
        max_d_cur = max(1, len(valid_df) - min(args.keep_k, len(valid_df)))
        d_cur_size = min(args.d_cur_size, max_d_cur)
        d_cur_indices = valid_df.sample(n=d_cur_size, random_state=args.seed, replace=False).index.to_list()
        d_cur_index_set = set(d_cur_indices)
        d_cur_df = valid_df.loc[d_cur_indices].reset_index(drop=True)
        pool_df = valid_df.drop(index=d_cur_indices).reset_index(drop=True)
        pool_records = [record for idx, record in enumerate(validation_bundle.valid_records) if idx not in d_cur_index_set]
    else:
        d_cur_df = selector.initialize_d_cur(size=args.d_cur_size)

    save_csv(selection_dir / "d_cur_init.csv", d_cur_df)
    save_csv(selection_dir / "candidate_pool.csv", pool_df)
    evaluator = TabDiffSelectionEvaluator(
        dataset_name=args.dataset_name,
        device=eval_device,
        metric_list=["density", "dcr"],
        real_data_path=input_dir / "eval_train.csv",
        test_data_path=input_dir / "eval_test.csv",
        val_data_path=input_dir / "eval_holdout.csv",
    )
    overall.update(1)

    _progress_write("[5/13] surrogate scoring and preselect")
    desired_keep_k = min(args.keep_k, len(pool_records))
    requested_preselect_target = min(len(pool_records), max(args.preselect_target, desired_keep_k))
    surrogate_records_all = selector.compute_surrogates(
        pool_df,
        show_progress=progress_enabled,
        progress_desc="surrogate scoring",
        candidate_ids=np.fromiter(
            (int(record.get("candidate_id", idx)) for idx, record in enumerate(pool_records)),
            dtype=int,
            count=len(pool_records),
        ),
    )
    preselect_should_run = requested_preselect_target < len(pool_records) and len(pool_records) > desired_keep_k
    preselect_gate: dict[str, Any]
    candidate_preselect_report: dict[str, Any] = {}
    baseline_preselect_report: dict[str, Any] = {}
    if preselect_should_run:
        baseline_surrogates = [dict(record) for record in surrogate_records_all]
        candidate_surrogates = [dict(record) for record in surrogate_records_all]
        baseline_valid, baseline_sur = selector.dual_median_filter_baseline(
            valid_records=pool_records,
            surrogate_records=baseline_surrogates,
            target_preselect=requested_preselect_target,
            show_progress=progress_enabled,
            progress_desc="preselect baseline",
        )
        baseline_preselect_report = dict(selector.last_preselect_report)
        baseline_candidate_ids = np.asarray(
            [int(record.get("candidate_id", idx)) for idx, record in enumerate(baseline_sur)],
            dtype=int,
        )
        preselected_valid_candidate, preselected_sur_candidate = selector.dual_median_filter(
            valid_records=pool_records,
            surrogate_records=candidate_surrogates,
            target_preselect=requested_preselect_target,
            anchor_candidate_ids=baseline_candidate_ids,
            show_progress=progress_enabled,
            progress_desc="preselect candidate",
        )
        candidate_preselect_report = dict(selector.last_preselect_report)

        raw_reference_df, _, _ = selector.select_keep_random(
            candidate_records=pool_records,
            keep_k=requested_preselect_target,
            rng_seed=args.seed,
        )
        candidate_preselected_df = _records_to_df(preselected_valid_candidate, selector.column_order)
        baseline_preselected_df = _records_to_df(baseline_valid, selector.column_order)
        preselect_gate = _build_preselect_gate_report(
            raw_metrics=_subset_gate_metrics(selector, evaluator, raw_reference_df),
            candidate_metrics=_subset_gate_metrics(selector, evaluator, candidate_preselected_df),
            baseline_metrics=_subset_gate_metrics(selector, evaluator, baseline_preselected_df),
            candidate_mode=str(candidate_preselect_report.get("mode", "candidate_preselect")),
            baseline_mode=str(baseline_preselect_report.get("mode", "baseline_preselect")),
        )
        preselect_gate["candidate"]["preselect_report"] = candidate_preselect_report
        preselect_gate["baseline"]["preselect_report"] = baseline_preselect_report

        if preselect_gate["selected_source"] == "candidate_preselect":
            preselected_valid = preselected_valid_candidate
            preselected_sur = preselected_sur_candidate
        else:
            preselected_valid = baseline_valid
            preselected_sur = baseline_sur
        preselect_status = {
            "applied": True,
            "mode": str(preselect_gate["selected_mode"]),
            "reason": None if not preselect_gate["fallback_applied"] else "preselect_gate_fallback_to_baseline",
            "rows_before": len(pool_records),
            "rows_after": len(preselected_valid),
            "selected_source": preselect_gate["selected_source"],
            "fallback_applied": bool(preselect_gate["fallback_applied"]),
            "selected_pass": bool(preselect_gate["selected_pass"]),
        }
    else:
        preselected_valid = pool_records.copy()
        preselected_sur = surrogate_records_all.copy()
        preselect_gate = {
            "skipped": True,
            "reason": (
                "target_not_reductive" if requested_preselect_target >= len(pool_records) else "pool_too_close_to_keep_k"
            ),
            "selected_source": "full_pool",
            "selected_mode": "skipped_full_pool",
            "selected_pass": None,
            "fallback_applied": False,
        }
        preselect_status = {
            "applied": False,
            "mode": "skipped_full_pool",
            "reason": (
                "target_not_reductive" if requested_preselect_target >= len(pool_records) else "pool_too_close_to_keep_k"
            ),
            "rows_before": len(pool_records),
            "rows_after": len(preselected_valid),
        }
    if len(preselected_valid) < desired_keep_k:
        preselected_valid = pool_records.copy()
        preselected_sur = surrogate_records_all.copy()
        preselect_gate = {
            **preselect_gate,
            "selected_source": "full_pool",
            "selected_mode": "fallback_full_pool",
            "selected_pass": False,
            "fallback_applied": True,
            "reason": "selected_preselect_below_keep_k",
        }
        preselect_status = {
            "applied": False,
            "mode": "fallback_full_pool",
            "reason": "dual_median_filter_below_keep_k",
            "rows_before": len(pool_records),
            "rows_after": len(preselected_valid),
            "selected_source": "full_pool",
            "fallback_applied": True,
            "selected_pass": False,
        }
    effective_preselect_target = len(preselected_valid)
    effective_keep_k = min(desired_keep_k, len(preselected_valid))
    if effective_keep_k <= 0:
        raise RuntimeError("effective_keep_k <= 0. Increase TabDiff sample size or reduce d_cur_size / keep_k.")
    preselect_privacy_objective, preselect_fidelity_objective = _preselect_objective_manifest(preselect_status["mode"])
    overall.update(1)

    _progress_write("[6/13] global exact scoring")
    global_exact_records, global_baselines = selector.compute_exact_scores(
        d_cur_df,
        preselected_valid,
        show_progress=progress_enabled,
        progress_desc="global exact scoring",
    )
    overall.update(1)

    _progress_write("[7/13] fidelity ceiling and utility proxy")
    selection_records = preselected_valid
    preselected_valid_df = _records_to_df(preselected_valid, selector.column_order)
    pre_ceiling_static_utility_bundle = build_static_balanced_utility_scores(
        selector=selector,
        preselected_records=selection_records,
        random_state=args.seed,
    )
    (
        initial_fidelity_ceiling_df,
        initial_fidelity_ceiling_records,
        initial_fidelity_ceiling_report,
    ) = selector.construct_fidelity_ceiling_subset(
        preselected_records=selection_records,
        exact_records=global_exact_records,
        keep_k=effective_keep_k,
        utility_scores_by_id=pre_ceiling_static_utility_bundle["score_by_id"],
        utility_weight=args.fidelity_ceiling_utility_weight,
        refine_utility_weight=args.fidelity_ceiling_refine_utility_weight,
        utility_score_name="utility_static_balanced",
        show_progress=progress_enabled,
        progress_desc="initial fidelity ceiling",
    )
    utility_proxy_bundle = build_utility_proxy_scores(
        selector=selector,
        preselected_records=selection_records,
        anchor_records=initial_fidelity_ceiling_records,
        random_state=args.seed,
        show_progress=progress_enabled,
    )
    global_exact_records, utility_proxy_merge_report = _attach_utility_proxy_fields(
        global_exact_records,
        utility_proxy_bundle["proxy_scores"],
    )
    second_pass_utility_scores_by_id = {
        int(record["candidate_id"]): float(record.get("u_proxy", 0.0))
        for record in utility_proxy_bundle["proxy_scores"]
    }
    second_pass_enabled = bool(
        second_pass_utility_scores_by_id and float(args.fidelity_ceiling_second_pass_utility_weight) > 0.0
    )
    if second_pass_enabled:
        fidelity_ceiling_df, fidelity_ceiling_records, fidelity_ceiling_report = selector.construct_fidelity_ceiling_subset(
            preselected_records=selection_records,
            exact_records=global_exact_records,
            keep_k=effective_keep_k,
            utility_scores_by_id=second_pass_utility_scores_by_id,
            utility_weight=args.fidelity_ceiling_second_pass_utility_weight,
            refine_utility_weight=args.fidelity_ceiling_second_pass_refine_utility_weight,
            utility_score_name="utility_proxy_second_pass",
            show_progress=progress_enabled,
            progress_desc="dynamic utility ceiling",
        )
        fidelity_ceiling_report["initial_anchor"] = initial_fidelity_ceiling_report.get("reference", {})
        fidelity_ceiling_report["second_pass"] = {
            "applied": True,
            "utility_source": "utility_proxy_scores.u_proxy",
            "dynamic_anchor": "initial_static_fidelity_ceiling",
            "initial_rows": int(len(initial_fidelity_ceiling_records)),
            "final_rows": int(len(fidelity_ceiling_records)),
            "utility_weight": float(args.fidelity_ceiling_second_pass_utility_weight),
            "refine_utility_weight": float(args.fidelity_ceiling_second_pass_refine_utility_weight),
        }
    else:
        fidelity_ceiling_df = initial_fidelity_ceiling_df
        fidelity_ceiling_records = initial_fidelity_ceiling_records
        fidelity_ceiling_report = initial_fidelity_ceiling_report
        fidelity_ceiling_report["second_pass"] = {
            "applied": False,
            "reason": "disabled_or_empty_utility_scores",
        }
    floor_reference = fidelity_ceiling_report.get(
        "reference",
        {
            "name": "preselected_fidelity_ceiling_keep_k",
            "rows": len(fidelity_ceiling_records),
            "fidelity_1d": selector.compute_dataset_fidelity(fidelity_ceiling_df),
            "fidelity_2d": selector.compute_dataset_pair_fidelity(fidelity_ceiling_df),
            "privacy_mean": selector.compute_dataset_privacy(fidelity_ceiling_df),
        },
    )
    overall.update(1)

    _progress_write("[8/13] streaming archive diagnostic")
    archive_budget = max(effective_keep_k, int(round(args.archive_budget_scale * effective_keep_k)))
    chunk_size = _streaming_chunk_size(len(preselected_valid), effective_keep_k, args.selection_chunk_size)
    archive_should_run = archive_budget < len(preselected_valid) and len(preselected_valid) > effective_keep_k
    if archive_should_run:
        archive_records, archive_exact_records, streaming_report = _run_streaming_archive(
            selector=selector,
            pool_records=preselected_valid,
            d_cur_df=d_cur_df,
            keep_k=effective_keep_k,
            preselect_target=effective_preselect_target,
            chunk_size=chunk_size,
            archive_budget=archive_budget,
            local_keep_factor=args.local_keep_factor,
            show_progress=progress_enabled,
        )
        archive_status = {
            "applied": True,
            "mode": "streaming_archive",
            "reason": None,
            "rows_before": len(preselected_valid),
            "rows_after": len(archive_records),
            "fixed_reference_baseline": True,
        }
    else:
        archive_records = preselected_valid.copy()
        archive_exact_records = global_exact_records.copy()
        streaming_report = {
            "mode": "skipped",
            "reason": "archive_budget_not_binding",
            "chunk_size": chunk_size,
            "archive_budget": archive_budget,
            "archive_rows_final": len(archive_records),
            "fixed_reference_baseline": True,
            "chunks": [],
        }
        archive_status = {
            "applied": False,
            "mode": "skipped_full_preselected_pool",
            "reason": "archive_budget_not_binding",
            "rows_before": len(preselected_valid),
            "rows_after": len(archive_records),
            "fixed_reference_baseline": True,
        }
    archive_df = _records_to_df(archive_records, selector.column_order)
    save_csv(selection_dir / "archive_pool.csv", archive_df)
    overall.update(1)

    _progress_write("[9/13] core selections on unified comparison pool")
    if archive_should_run:
        archive_rescored_exact_records, archive_rescore_baselines = selector.compute_exact_scores(
            d_cur_df,
            archive_records,
            show_progress=progress_enabled,
            progress_desc="archive exact scoring",
        )
        archive_rescored_exact_records, archive_utility_proxy_merge_report = _attach_utility_proxy_fields(
            archive_rescored_exact_records,
            utility_proxy_bundle["proxy_scores"],
        )
    else:
        archive_rescored_exact_records = global_exact_records.copy()
        archive_rescore_baselines = dict(global_baselines)
        archive_utility_proxy_merge_report = dict(utility_proxy_merge_report)

    baseline_full_records = pool_records.copy()
    selection_exact_records = global_exact_records
    raw_baseline_pool_name = "pool_records"
    selection_pool_name = "preselected_valid"

    raw_full_keep_records = baseline_full_records[:effective_keep_k]
    raw_full_selection_df = _records_to_df(raw_full_keep_records, selector.column_order)
    random_full_keep_df, random_full_keep_records, random_full_report = selector.select_keep_random(
        candidate_records=baseline_full_records,
        keep_k=effective_keep_k,
        rng_seed=args.seed,
    )
    scalar_keep_df, scalar_keep_records, scalar_report = selector.select_keep_scalarization(
        preselected_records=selection_records,
        exact_records=selection_exact_records,
        keep_k=effective_keep_k,
        fidelity_1d_weight=0.5 * args.scalar_fidelity_weight,
        fidelity_2d_weight=0.5 * args.scalar_fidelity_weight,
        privacy_weight=args.scalar_privacy_weight,
        utility_weight=args.scalar_utility_weight,
        mode="matched",
        floor_reference=floor_reference,
    )
    scalar_naive_keep_df, scalar_naive_keep_records, scalar_naive_report = selector.select_keep_scalarization(
        preselected_records=selection_records,
        exact_records=selection_exact_records,
        keep_k=effective_keep_k,
        fidelity_1d_weight=0.5 * args.scalar_fidelity_weight,
        fidelity_2d_weight=0.5 * args.scalar_fidelity_weight,
        privacy_weight=args.scalar_privacy_weight,
        utility_weight=0.0,
        mode="naive",
    )
    pareto_keep_df, pareto_keep_records, pareto_report = selector.select_keep(
        preselected_records=selection_records,
        surrogate_records=[],
        exact_records=selection_exact_records,
        keep_k=effective_keep_k,
        floor_reference=floor_reference,
        constraint_reference_records=fidelity_ceiling_records,
        floor_mode=args.pareto_floor_mode,
        soft_fidelity_floor_eps=args.pareto_soft_fidelity_floor_eps,
        soft_trend_floor_eps=args.pareto_soft_trend_floor_eps,
        soft_privacy_floor_eps=args.pareto_soft_privacy_floor_eps,
        soft_utility_floor_eps=args.pareto_soft_utility_floor_eps,
        soft_min_score_delta=args.pareto_soft_min_score_delta,
    )
    overall.update(1)

    _progress_write("[10/13] build 4D direction families")
    scalar_naive_family_df, scalar_naive_family_records, scalar_naive_family_reports = _build_direction_family(
        selector=selector,
        records=selection_records,
        exact_records=selection_exact_records,
        keep_k=effective_keep_k,
        family_type="scalar_naive",
        show_progress=progress_enabled,
    )
    scalar_matched_family_df, scalar_matched_family_records, scalar_matched_family_reports = _build_direction_family(
        selector=selector,
        records=selection_records,
        exact_records=selection_exact_records,
        keep_k=effective_keep_k,
        family_type="scalar_matched",
        floor_reference=floor_reference,
        show_progress=progress_enabled,
    )
    pareto_family_df, pareto_family_records, pareto_family_reports = _build_direction_family(
        selector=selector,
        records=selection_records,
        exact_records=selection_exact_records,
        keep_k=effective_keep_k,
        family_type="pareto",
        floor_reference=floor_reference,
        constraint_reference_records=fidelity_ceiling_records,
        floor_mode=args.pareto_floor_mode,
        soft_fidelity_floor_eps=args.pareto_soft_fidelity_floor_eps,
        soft_trend_floor_eps=args.pareto_soft_trend_floor_eps,
        soft_privacy_floor_eps=args.pareto_soft_privacy_floor_eps,
        soft_utility_floor_eps=args.pareto_soft_utility_floor_eps,
        soft_min_score_delta=args.pareto_soft_min_score_delta,
        show_progress=progress_enabled,
    )
    pareto_keep_df, pareto_keep_records, pareto_report, pareto_finalist_rerank = _rerank_pareto_finalists_on_search_holdout(
        selector=selector,
        exact_records=selection_exact_records,
        pareto_keep_df=pareto_keep_df,
        pareto_keep_records=pareto_keep_records,
        pareto_report=pareto_report,
        pareto_family_df=pareto_family_df,
        pareto_family_records=pareto_family_records,
        pareto_family_reports=pareto_family_reports,
        floor_reference=floor_reference,
        fidelity_ceiling_df=fidelity_ceiling_df,
        fidelity_ceiling_records=fidelity_ceiling_records,
        utility_switch_min=args.pareto_rerank_utility_switch_min,
        privacy_switch_min=args.pareto_rerank_privacy_switch_min,
        floor_mode=args.pareto_floor_mode,
        soft_fidelity_floor_eps=args.pareto_soft_fidelity_floor_eps,
        soft_trend_floor_eps=args.pareto_soft_trend_floor_eps,
    )
    overall.update(1)

    save_jsonl(selection_dir / "surrogate_scores.jsonl", surrogate_records_all)
    save_jsonl(selection_dir / "preselected_surrogates.jsonl", preselected_sur)
    save_jsonl(selection_dir / "utility_pre_ceiling_static_scores.jsonl", pre_ceiling_static_utility_bundle["static_scores"])
    save_jsonl(selection_dir / "utility_static_scores.jsonl", utility_proxy_bundle["static_scores"])
    save_jsonl(selection_dir / "utility_dynamic_scores.jsonl", utility_proxy_bundle["dynamic_scores"])
    save_jsonl(selection_dir / "utility_proxy_scores.jsonl", utility_proxy_bundle["proxy_scores"])
    save_jsonl(selection_dir / "exact_scores.jsonl", global_exact_records)
    save_jsonl(selection_dir / "archive_exact_scores.jsonl", archive_rescored_exact_records)
    save_json(selection_dir / "baselines.json", global_baselines)
    save_json(selection_dir / "archive_rescore_baselines.json", archive_rescore_baselines)
    save_json(selection_dir / "utility_dynamic_blocks.json", utility_proxy_bundle["dynamic_blocks"])
    save_json(
        selection_dir / "utility_proxy_manifest.json",
        {
            **utility_proxy_bundle["manifest"],
            "teacher_manifest": utility_proxy_bundle.get("teacher_manifest", {}),
            "pre_ceiling_static": pre_ceiling_static_utility_bundle["manifest"],
            "fidelity_ceiling_second_pass": fidelity_ceiling_report.get("second_pass", {}),
            "merge_report": utility_proxy_merge_report,
            "archive_merge_report": archive_utility_proxy_merge_report,
        },
    )
    save_json(selection_dir / "random_full_report.json", random_full_report)
    save_json(selection_dir / "scalarization_report.json", scalar_report)
    save_json(selection_dir / "scalarization_naive_report.json", scalar_naive_report)
    save_json(selection_dir / "pareto_report.json", pareto_report)
    save_json(selection_dir / "pareto_finalist_rerank.json", pareto_finalist_rerank)
    save_json(selection_dir / "fidelity_ceiling_initial_report.json", initial_fidelity_ceiling_report)
    save_json(selection_dir / "fidelity_ceiling_report.json", fidelity_ceiling_report)
    save_json(selection_dir / "streaming_archive_report.json", streaming_report)
    save_json(selection_dir / "exact_chunk_report.json", streaming_report)
    save_json(selection_dir / "preselect_gate.json", preselect_gate)
    save_json(selection_dir / "scalar_family_naive_reports.json", scalar_naive_family_reports)
    save_json(selection_dir / "scalar_family_matched_reports.json", scalar_matched_family_reports)
    save_json(selection_dir / "pareto_family_reports.json", pareto_family_reports)
    save_json(
        selection_dir / "selection_manifest.json",
        {
            "dataset_name": args.dataset_name,
            "synthetic_csv": str(synthetic_csv),
            "protocol": "tabdiff_canonical_train_with_derived_holdout",
            "requested_keep_k": args.keep_k,
            "requested_preselect_target": requested_preselect_target,
            "effective_preselect_target": effective_preselect_target,
            "effective_keep_k": effective_keep_k,
            "candidate_pool_rows": len(pool_records),
            "preselected_rows": len(preselected_valid),
            "raw_baseline_pool_name": raw_baseline_pool_name,
            "raw_baseline_pool_rows": len(baseline_full_records),
            "selection_pool_name": selection_pool_name,
            "selection_pool_rows": len(selection_records),
            "comparison_pool_name": selection_pool_name,
            "comparison_pool_rows": len(selection_records),
            "final_floor_reference_name": floor_reference.get("name", "preselected_fidelity_ceiling_keep_k"),
            "final_floor_reference_rows": len(fidelity_ceiling_records),
            "archive_budget": archive_budget,
            "archive_rows": len(archive_records),
            "d_cur_rows": len(d_cur_df),
            "selection_chunk_size": chunk_size,
            "lambda_penalty": args.lambda_penalty,
            "gamma": args.gamma,
            "privacy_version": args.privacy_version,
            "scalar_fidelity_weight": args.scalar_fidelity_weight,
            "scalar_fidelity_1d_weight": 0.5 * args.scalar_fidelity_weight,
            "scalar_fidelity_2d_weight": 0.5 * args.scalar_fidelity_weight,
            "scalar_privacy_weight": args.scalar_privacy_weight,
            "scalar_utility_weight": args.scalar_utility_weight,
            "three_objective_enabled": True,
            "final_fidelity_floor_eps": args.final_fidelity_floor_eps,
            "final_trend_floor_eps": args.final_trend_floor_eps,
            "fidelity_ceiling_utility_weight": args.fidelity_ceiling_utility_weight,
            "fidelity_ceiling_refine_utility_weight": args.fidelity_ceiling_refine_utility_weight,
            "fidelity_ceiling_second_pass_utility_weight": args.fidelity_ceiling_second_pass_utility_weight,
            "fidelity_ceiling_second_pass_refine_utility_weight": args.fidelity_ceiling_second_pass_refine_utility_weight,
            "pareto_rerank_utility_switch_min": args.pareto_rerank_utility_switch_min,
            "pareto_rerank_privacy_switch_min": args.pareto_rerank_privacy_switch_min,
            "pareto_floor_mode": args.pareto_floor_mode,
            "pareto_soft_fidelity_floor_eps": args.pareto_soft_fidelity_floor_eps,
            "pareto_soft_trend_floor_eps": args.pareto_soft_trend_floor_eps,
            "pareto_soft_privacy_floor_eps": args.pareto_soft_privacy_floor_eps,
            "pareto_soft_utility_floor_eps": args.pareto_soft_utility_floor_eps,
            "pareto_soft_min_score_delta": args.pareto_soft_min_score_delta,
            "preselect_privacy_objective": preselect_privacy_objective,
            "preselect_fidelity_objective": {
                **preselect_fidelity_objective,
                "pair_edges": len(selector.pair_marginal_edges),
            },
            "final_selection_floor_proxy": {
                "fidelity": "exact_1d_marginal_similarity",
                "trend": "exact_2d_pair_similarity",
            },
            "nn_device": nn_device,
            "nn_query_batch_size": args.nn_query_batch_size,
            "nn_reference_chunk_size": args.nn_reference_chunk_size,
            "density_reference_size": args.density_reference_size,
            "holdout_strategy": dataset_ctx.holdout_strategy,
            "holdout_fraction": dataset_ctx.holdout_fraction,
            "preselect_fallback": preselect_status,
            "preselect_status": preselect_status,
            "preselect_gate": preselect_gate,
            "archive_status": archive_status,
            "fidelity_ceiling_second_pass": fidelity_ceiling_report.get("second_pass", {}),
            "utility_proxy": {
                **utility_proxy_bundle["manifest"],
                "pre_ceiling_static": pre_ceiling_static_utility_bundle["manifest"],
                "merge_report": utility_proxy_merge_report,
                "archive_merge_report": archive_utility_proxy_merge_report,
            },
            "pareto_finalist_rerank": pareto_finalist_rerank,
        },
    )

    _save_selection_csvs(
        versions_dir=versions_dir,
        raw_df=raw_full_selection_df,
        random_df=random_full_keep_df,
        scalar_df=scalar_keep_df,
        pareto_df=pareto_keep_df,
        raw_tag="raw_full",
        random_tag="random_full",
    )
    save_csv(versions_dir / "selection_scalar_naive.csv", scalar_naive_keep_df)
    save_csv(versions_dir / "selection_archive_pool.csv", archive_df)
    save_csv(versions_dir / "selection_preselected_valid.csv", preselected_valid_df)
    save_csv(versions_dir / "selection_preselected_fidelity_ceiling_initial_keep_k.csv", initial_fidelity_ceiling_df)
    save_csv(versions_dir / "selection_preselected_fidelity_ceiling_keep_k.csv", fidelity_ceiling_df)
    save_csv(versions_dir / "preselected_valid_keep.csv", preselected_valid_df)
    save_csv(versions_dir / "preselected_fidelity_ceiling_keep_k.csv", fidelity_ceiling_df)
    _save_family_csvs(versions_dir, "selection_scalar_family_naive", scalar_naive_family_df)
    _save_family_csvs(versions_dir, "selection_scalar_family_matched", scalar_matched_family_df)
    _save_family_csvs(versions_dir, "selection_endpoint", pareto_family_df)

    _progress_write("[11/13] evaluate core selections")
    selection_inputs = [
        ("raw_full", raw_full_selection_df, raw_full_keep_records, baseline_full_records),
        ("random_full", random_full_keep_df, random_full_keep_records, baseline_full_records),
        ("preselected_valid", preselected_valid_df, preselected_valid, baseline_full_records),
        ("preselected_fidelity_ceiling_keep_k", fidelity_ceiling_df, fidelity_ceiling_records, selection_records),
        ("scalar", scalar_keep_df, scalar_keep_records, pool_records),
        ("scalar_naive", scalar_naive_keep_df, scalar_naive_keep_records, pool_records),
        ("pareto", pareto_keep_df, pareto_keep_records, pool_records),
    ]
    selection_metrics: dict[str, dict[str, Any]] = {}
    selection_iter = _progress(
        selection_inputs,
        total=len(selection_inputs),
        desc="evaluate selections",
        dynamic_ncols=True,
        disable=not progress_enabled,
    )
    for selection_name, selection_df, keep_records, source_records in selection_iter:
        selection_metrics[selection_name] = _evaluate_selection(
            selection_name,
            selection_df,
            keep_records,
            selection_records if selection_name in {"scalar", "scalar_naive", "pareto"} else source_records,
            selector,
            evaluator,
            eval_dir,
            test_df,
            args.jsd_epsilon,
            args.rare_threshold,
        )
        selection_iter.set_postfix(selection=selection_name, rows=len(selection_df))
    overall.update(1)

    scalar_naive_family_metrics: dict[str, dict[str, Any]] = {}
    scalar_matched_family_metrics: dict[str, dict[str, Any]] = {}
    pareto_family_metrics: dict[str, dict[str, Any]] = {}

    _progress_write("[12/13] evaluate 4D direction families")
    family_eval_iter = _progress(
        DIRECTION_SPECS,
        total=len(DIRECTION_SPECS),
        desc="evaluate families",
        dynamic_ncols=True,
        disable=not progress_enabled,
    )
    for tag, _, _, _, _ in family_eval_iter:
        scalar_naive_family_metrics[tag] = _evaluate_selection(
            _selection_name("scalar_family_naive", tag),
            scalar_naive_family_df[tag],
            scalar_naive_family_records[tag],
            selection_records,
            selector,
            evaluator,
            eval_dir,
            test_df,
            args.jsd_epsilon,
            args.rare_threshold,
        )
        scalar_matched_family_metrics[tag] = _evaluate_selection(
            _selection_name("scalar_family_matched", tag),
            scalar_matched_family_df[tag],
            scalar_matched_family_records[tag],
            selection_records,
            selector,
            evaluator,
            eval_dir,
            test_df,
            args.jsd_epsilon,
            args.rare_threshold,
        )
        pareto_family_metrics[tag] = _evaluate_selection(
            _selection_name("endpoint", tag),
            pareto_family_df[tag],
            pareto_family_records[tag],
            selection_records,
            selector,
            evaluator,
            eval_dir,
            test_df,
            args.jsd_epsilon,
            args.rare_threshold,
        )
        family_eval_iter.set_postfix(direction=tag)

    family_comparison = _compare_families(
        pareto_family_metrics=pareto_family_metrics,
        scalar_family_metrics=scalar_matched_family_metrics,
    )
    utility_family_comparison = dict(family_comparison.get("utility_space", {}))
    selection_gate = _build_selection_gate_report(
        selection_metrics=selection_metrics,
        family_comparison=family_comparison,
    )
    save_json(report_dir / "family_comparison.json", family_comparison)
    save_json(report_dir / "utility_family_comparison.json", utility_family_comparison)
    save_json(report_dir / "selection_gate.json", selection_gate)
    save_json(report_dir / "preselect_gate.json", preselect_gate)
    overall.update(1)

    _progress_write("[13/13] write summary")
    summary = {
        "source": args.source,
        "synthetic_csv": str(synthetic_csv),
        "protocol": "tabdiff_canonical_train_with_derived_holdout",
        "eval_device": eval_device,
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "test_rows": int(len(test_df)),
        "raw_rows": int(len(synthetic_df)),
        "valid_rows": int(len(valid_df)),
        "rejected_rows": int(len(validation_bundle.rejected_records)),
        "validator_reject_rate": float(validation_bundle.report.get("reject_rate", 0.0)),
        "d_cur_source": args.d_cur_source,
        "d_cur_rows": int(len(d_cur_df)),
        "candidate_pool_rows": int(len(pool_df)),
        "preselected_rows": int(len(preselected_valid)),
        "raw_baseline_pool_name": raw_baseline_pool_name,
        "raw_baseline_pool_rows": int(len(baseline_full_records)),
        "selection_pool_name": selection_pool_name,
        "selection_pool_rows": int(len(selection_records)),
        "comparison_pool_name": selection_pool_name,
        "comparison_pool_rows": int(len(selection_records)),
        "final_floor_reference_name": floor_reference.get("name", "preselected_fidelity_ceiling_keep_k"),
        "final_floor_reference_rows": int(len(fidelity_ceiling_records)),
        "archive_rows": int(len(archive_records)),
        "requested_preselect_target": int(requested_preselect_target),
        "effective_preselect_target": int(effective_preselect_target),
        "requested_keep_k": int(args.keep_k),
        "effective_keep_k": int(effective_keep_k),
        "selection_chunk_size": int(chunk_size),
        "archive_budget": int(archive_budget),
        "lambda_penalty": float(args.lambda_penalty),
        "gamma": float(args.gamma),
        "privacy_version": args.privacy_version,
        "final_fidelity_floor_eps": float(args.final_fidelity_floor_eps),
        "final_trend_floor_eps": float(args.final_trend_floor_eps),
        "preselect_privacy_objective": preselect_privacy_objective,
        "preselect_fidelity_objective": {
            **preselect_fidelity_objective,
            "pair_edges": int(len(selector.pair_marginal_edges)),
        },
        "final_selection_floor_proxy": {
            "fidelity": "exact_1d_marginal_similarity",
            "trend": "exact_2d_pair_similarity",
        },
        "nn_device": nn_device,
        "nn_query_batch_size": int(args.nn_query_batch_size),
        "nn_reference_chunk_size": int(args.nn_reference_chunk_size),
        "density_reference_size": int(args.density_reference_size),
        "holdout_fraction": float(args.holdout_fraction),
        "preselect_fallback": preselect_status,
        "preselect_status": preselect_status,
        "preselect_gate": preselect_gate,
        "archive_status": archive_status,
        "utility_proxy": {
            **utility_proxy_bundle["manifest"],
            "merge_report": utility_proxy_merge_report,
            "archive_merge_report": archive_utility_proxy_merge_report,
        },
        "pareto_finalist_rerank": pareto_finalist_rerank,
        "raw_valid": _subset_metrics(selector, valid_df),
        "selection_metrics": selection_metrics,
        "scalar_family_naive_metrics": scalar_naive_family_metrics,
        "scalar_family_matched_metrics": scalar_matched_family_metrics,
        "pareto_family_metrics": pareto_family_metrics,
        "family_comparison": family_comparison,
        "utility_family_comparison": utility_family_comparison,
        "selection_gate": selection_gate,
        "streaming_archive_report": streaming_report,
        "random_full_smoke": selector.compute_smoke_metrics(
            surrogate_records_all=surrogate_records_all,
            keep_records=random_full_keep_records,
            keep_df=random_full_keep_df,
        ),
        "preselected_valid_smoke": selector.compute_smoke_metrics(
            surrogate_records_all=surrogate_records_all,
            keep_records=preselected_valid,
            keep_df=preselected_valid_df,
        ),
        "preselected_fidelity_ceiling_smoke": selector.compute_smoke_metrics(
            surrogate_records_all=surrogate_records_all,
            keep_records=fidelity_ceiling_records,
            keep_df=fidelity_ceiling_df,
        ),
        "scalarization_smoke": selector.compute_smoke_metrics(
            surrogate_records_all=surrogate_records_all,
            keep_records=scalar_keep_records,
            keep_df=scalar_keep_df,
        ),
        "pareto_smoke": selector.compute_smoke_metrics(
            surrogate_records_all=surrogate_records_all,
            keep_records=pareto_keep_records,
            keep_df=pareto_keep_df,
        ),
    }
    save_json(report_dir / "summary.json", summary)

    summary_lines = [
        "# TabDiff -> M5 Selection Summary",
        "",
        f"- source: `{args.source}`",
        f"- synthetic_csv: `{synthetic_csv}`",
        f"- eval_device: `{eval_device}`",
        f"- raw_rows: `{len(synthetic_df)}`",
        f"- valid_rows: `{len(valid_df)}`",
        f"- rejected_rows: `{len(validation_bundle.rejected_records)}`",
        f"- d_cur_rows: `{len(d_cur_df)}`",
        f"- candidate_pool_rows: `{len(pool_df)}`",
        f"- preselected_rows: `{len(preselected_valid)}`",
        f"- final_floor_reference_rows: `{len(fidelity_ceiling_records)}`",
        f"- archive_rows: `{len(archive_records)}`",
        f"- requested_keep_k: `{args.keep_k}`",
        f"- effective_keep_k: `{effective_keep_k}`",
        f"- archive_budget: `{archive_budget}`",
        f"- selection_chunk_size: `{chunk_size}`",
        f"- lambda_penalty: `{args.lambda_penalty}`",
        f"- gamma: `{args.gamma}`",
        f"- privacy_version: `{args.privacy_version}`",
        f"- scalar_utility_weight: `{args.scalar_utility_weight}`",
        f"- nn_device: `{nn_device}`",
        f"- nn_query_batch_size: `{args.nn_query_batch_size}`",
        f"- nn_reference_chunk_size: `{args.nn_reference_chunk_size}`",
        f"- density_reference_size: `{args.density_reference_size}`",
        f"- pareto_hv: `{family_comparison['pareto_hv']}`",
        f"- scalar_hv: `{family_comparison['scalar_hv']}`",
        f"- pareto_igd: `{family_comparison['pareto_igd']}`",
        f"- scalar_igd: `{family_comparison['scalar_igd']}`",
        f"- pointwise_dominance_count: `{family_comparison['pointwise_dominance_count']}`",
        f"- utility_family_hv: `{utility_family_comparison.get('pareto_hv')}`",
    ]
    (report_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    overall.update(1)
    overall.close()
    print(f"M5 summary saved to {report_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
