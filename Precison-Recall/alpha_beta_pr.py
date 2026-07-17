#!/usr/bin/env python3
"""Evaluate alpha-Precision and beta-Recall for tabular synthetic data.

The implementation follows the sample-level 3D metric used for synthetic data
auditing. Tabular rows are embedded by real-data-standardized numeric columns
and one-hot encoded categorical columns, then alpha-Precision and beta-Recall
curves are computed in that feature space.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


DEFAULT_DATASETS = [
    "shoppers",
    "news",
    "adult",
    "default",
    "diabetes",
    "magic",
    "beijing",
]

MISSING_CATEGORY = "<NA>"


@dataclass(frozen=True)
class ColumnSplit:
    numeric: list[str]
    categorical: list[str]
    target: list[str]


@dataclass(frozen=True)
class DatasetPaths:
    real: Path
    synthetic: Path
    info: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute alpha-Precision and beta-Recall for tabular synthetic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--model", default="tabdiff")
    parser.add_argument("--exp", default="no_theta_0_1")
    parser.add_argument("--data", default="selection_random_full")
    parser.add_argument("--real-root", type=Path, default=Path("third_party/TabDiff/synthetic"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("artifacts/postprocess"))
    parser.add_argument("--info-root", type=Path, default=Path("third_party/TabDiff/data"))
    parser.add_argument("--max-rows", type=int, default=20_000, help="Use 0 for all available rows.")
    parser.add_argument("--num-points", type=int, default=101)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("Precison-Recall/results"))
    parser.add_argument(
        "--exclude-target",
        action="store_true",
        help="Exclude target_col_idx from the evaluated joint table distribution.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue evaluating remaining datasets if one dataset fails.",
    )
    return parser.parse_args()


def data_filename(name: str) -> str:
    return name if name.endswith(".csv") else f"{name}.csv"


def build_paths(args: argparse.Namespace, dataset: str) -> DatasetPaths:
    return DatasetPaths(
        real=args.real_root / dataset / "real.csv",
        synthetic=args.synthetic_root / args.model / dataset / args.exp / "versions" / data_filename(args.data),
        info=args.info_root / dataset / "info.json",
    )


def read_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def valid_indices(values: Any, width: int) -> set[int]:
    if not isinstance(values, list):
        return set()
    out: set[int] = set()
    for value in values:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < width:
            out.add(idx)
    return out


def infer_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    non_missing = series.notna().sum()
    if non_missing == 0:
        return False
    return bool(converted.notna().sum() / non_missing >= 0.95)


def split_columns(real_df: pd.DataFrame, info: dict[str, Any], include_target: bool) -> ColumnSplit:
    width = len(real_df.columns)
    columns = list(real_df.columns)

    numeric_idx = valid_indices(info.get("num_col_idx"), width)
    categorical_idx = valid_indices(info.get("cat_col_idx"), width)
    target_idx = valid_indices(info.get("target_col_idx"), width)
    task_type = str(info.get("task_type", "")).lower()

    if include_target:
        if task_type == "regression":
            numeric_idx.update(target_idx)
            categorical_idx.difference_update(target_idx)
        else:
            categorical_idx.update(target_idx)
            numeric_idx.difference_update(target_idx)
    else:
        numeric_idx.difference_update(target_idx)
        categorical_idx.difference_update(target_idx)

    selected_idx = set(range(width))
    if not include_target:
        selected_idx.difference_update(target_idx)

    assigned_idx = (numeric_idx | categorical_idx) & selected_idx
    for idx in sorted(selected_idx - assigned_idx):
        if infer_numeric(real_df.iloc[:, idx]):
            numeric_idx.add(idx)
        else:
            categorical_idx.add(idx)

    numeric_idx &= selected_idx
    categorical_idx &= selected_idx
    categorical_idx.difference_update(numeric_idx)

    return ColumnSplit(
        numeric=[columns[idx] for idx in sorted(numeric_idx)],
        categorical=[columns[idx] for idx in sorted(categorical_idx)],
        target=[columns[idx] for idx in sorted(target_idx)],
    )


def align_synthetic_columns(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    real_columns = list(real_df.columns)
    synthetic_columns = list(synthetic_df.columns)
    if synthetic_columns == real_columns:
        return synthetic_df
    if set(synthetic_columns) == set(real_columns):
        return synthetic_df[real_columns]
    real_stripped = {col.strip(): col for col in real_columns}
    synthetic_stripped = {col.strip(): col for col in synthetic_columns}
    if (
        len(real_stripped) == len(real_columns)
        and len(synthetic_stripped) == len(synthetic_columns)
        and set(real_stripped) == set(synthetic_stripped)
    ):
        rename_map = {
            synthetic_stripped[stripped]: real_stripped[stripped]
            for stripped in real_stripped
        }
        return synthetic_df.rename(columns=rename_map)[real_columns]
    missing = sorted(set(real_columns) - set(synthetic_columns))
    extra = sorted(set(synthetic_columns) - set(real_columns))
    raise ValueError(
        f"{dataset}: real/synthetic columns differ; missing={missing[:10]}, extra={extra[:10]}"
    )


def sample_same_size(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    common_n = min(len(real_df), len(synthetic_df))
    if max_rows > 0:
        common_n = min(common_n, max_rows)
    if common_n < 2:
        raise ValueError("At least two real and synthetic rows are required.")

    rng = np.random.default_rng(seed)
    real_idx = rng.choice(len(real_df), size=common_n, replace=False)
    synthetic_idx = rng.choice(len(synthetic_df), size=common_n, replace=False)
    return (
        real_df.iloc[np.sort(real_idx)].reset_index(drop=True),
        synthetic_df.iloc[np.sort(synthetic_idx)].reset_index(drop=True),
        common_n,
    )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float64)


def numeric_block(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    columns: list[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if not columns:
        return sparse.csr_matrix((len(real_df), 0)), sparse.csr_matrix((len(synthetic_df), 0))

    real_num = real_df[columns].apply(pd.to_numeric, errors="coerce")
    synthetic_num = synthetic_df[columns].apply(pd.to_numeric, errors="coerce")
    real_num = real_num.replace([np.inf, -np.inf], np.nan)
    synthetic_num = synthetic_num.replace([np.inf, -np.inf], np.nan)

    median = real_num.median(axis=0, skipna=True).fillna(0.0)
    real_num = real_num.fillna(median)
    synthetic_num = synthetic_num.fillna(median)

    mean = real_num.mean(axis=0)
    std = real_num.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)

    real_arr = ((real_num - mean) / std).to_numpy(dtype=np.float64)
    synthetic_arr = ((synthetic_num - mean) / std).to_numpy(dtype=np.float64)
    real_arr = np.nan_to_num(real_arr, nan=0.0, posinf=0.0, neginf=0.0)
    synthetic_arr = np.nan_to_num(synthetic_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return sparse.csr_matrix(real_arr), sparse.csr_matrix(synthetic_arr)


def categorical_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].copy()
    for col in columns:
        out[col] = out[col].where(out[col].notna(), MISSING_CATEGORY).astype(str).str.strip()
    return out


def category_overlap_diagnostics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    columns: list[str],
) -> dict[str, float | bool]:
    exact_scores: list[float] = []
    stripped_scores: list[float] = []
    for col in columns:
        real_values = set(real_df[col].where(real_df[col].notna(), MISSING_CATEGORY).astype(str).unique())
        synthetic_values = set(
            synthetic_df[col].where(synthetic_df[col].notna(), MISSING_CATEGORY).astype(str).unique()
        )
        real_stripped = {value.strip() for value in real_values}
        synthetic_stripped = {value.strip() for value in synthetic_values}

        exact_scores.append(len(real_values & synthetic_values) / max(1, len(real_values)))
        stripped_scores.append(len(real_stripped & synthetic_stripped) / max(1, len(real_stripped)))

    if not columns:
        exact_min = 1.0
        stripped_min = 1.0
    else:
        exact_min = float(min(exact_scores))
        stripped_min = float(min(stripped_scores))
    return {
        "categorical_exact_overlap_min": exact_min,
        "categorical_strip_overlap_min": stripped_min,
        "categorical_value_normalized": True,
    }


def categorical_block(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    columns: list[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, int]:
    if not columns:
        return (
            sparse.csr_matrix((len(real_df), 0)),
            sparse.csr_matrix((len(synthetic_df), 0)),
            0,
        )

    real_cat = categorical_frame(real_df, columns)
    synthetic_cat = categorical_frame(synthetic_df, columns)
    encoder = make_one_hot_encoder()
    encoder.fit(pd.concat([real_cat, synthetic_cat], axis=0, ignore_index=True))
    real_encoded = encoder.transform(real_cat).tocsr()
    synthetic_encoded = encoder.transform(synthetic_cat).tocsr()
    return real_encoded, synthetic_encoded, int(real_encoded.shape[1])


def build_feature_matrices(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    split: ColumnSplit,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, int]:
    real_num, synthetic_num = numeric_block(real_df, synthetic_df, split.numeric)
    real_cat, synthetic_cat, categorical_features = categorical_block(real_df, synthetic_df, split.categorical)
    real_x = sparse.hstack([real_num, real_cat], format="csr")
    synthetic_x = sparse.hstack([synthetic_num, synthetic_cat], format="csr")
    if real_x.shape[1] == 0:
        raise ValueError("No evaluable columns were found.")
    return real_x, synthetic_x, categorical_features


def matrix_mean(x: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(x.mean(axis=0)).ravel()


def distances_to_center(x: sparse.csr_matrix, center: np.ndarray) -> np.ndarray:
    row_norm_sq = np.asarray(x.multiply(x).sum(axis=1)).ravel()
    dot = np.asarray(x.dot(center)).ravel()
    center_norm_sq = float(np.dot(center, center))
    dist_sq = np.maximum(row_norm_sq - 2.0 * dot + center_norm_sq, 0.0)
    return np.sqrt(dist_sq)


def nearest_neighbors(
    train: sparse.csr_matrix,
    query: sparse.csr_matrix,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    model.fit(train)
    distances, indices = model.kneighbors(query, return_distance=True)
    return np.asarray(distances), np.asarray(indices)


def delta_to_diagonal(grid: np.ndarray, curve: np.ndarray) -> float:
    denominator = float(np.sum(grid))
    if math.isclose(denominator, 0.0):
        return float("nan")
    return float(1.0 - np.sum(np.abs(grid - curve)) / denominator)


def curve_value_at(grid: np.ndarray, curve: np.ndarray, point: float) -> float:
    idx = int(np.argmin(np.abs(grid - point)))
    return float(curve[idx])


def auc(grid: np.ndarray, curve: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(curve, grid))
    return float(np.trapz(curve, grid))


def compute_alpha_beta(
    real_x: sparse.csr_matrix,
    synthetic_x: sparse.csr_matrix,
    num_points: int,
) -> dict[str, Any]:
    if real_x.shape[0] != synthetic_x.shape[0]:
        raise ValueError("real_x and synthetic_x must have the same row count.")
    if real_x.shape[0] < 2:
        raise ValueError("At least two rows are required.")
    if num_points < 2:
        raise ValueError("--num-points must be at least 2.")

    grid = np.linspace(0.0, 1.0, num_points)

    real_center = matrix_mean(real_x)
    real_to_real_center = distances_to_center(real_x, real_center)
    synthetic_to_real_center = distances_to_center(synthetic_x, real_center)
    alpha_radii = np.quantile(real_to_real_center, grid)
    alpha_precision_curve = np.array(
        [np.mean(synthetic_to_real_center <= radius) for radius in alpha_radii],
        dtype=np.float64,
    )

    synthetic_center = matrix_mean(synthetic_x)
    synthetic_to_synthetic_center = distances_to_center(synthetic_x, synthetic_center)
    real_to_synthetic_center = distances_to_center(real_x, synthetic_center)
    beta_support_radii = np.quantile(synthetic_to_synthetic_center, grid)
    beta_support_recall_curve = np.array(
        [np.mean(real_to_synthetic_center <= radius) for radius in beta_support_radii],
        dtype=np.float64,
    )

    real_to_real_distances, _ = nearest_neighbors(real_x, real_x, n_neighbors=2)
    real_to_synthetic_distances, real_to_synthetic_indices = nearest_neighbors(
        synthetic_x,
        real_x,
        n_neighbors=1,
    )
    synthetic_to_real_distances, synthetic_to_real_indices = nearest_neighbors(
        real_x,
        synthetic_x,
        n_neighbors=1,
    )
    real_to_real_nn = real_to_real_distances[:, 1]
    real_to_synthetic_nn = real_to_synthetic_distances[:, 0]
    closest_synthetic_idx = real_to_synthetic_indices[:, 0]
    closest_synthetic = synthetic_x[closest_synthetic_idx]
    closest_synthetic_to_center = distances_to_center(closest_synthetic, synthetic_center)
    beta_radii = np.quantile(closest_synthetic_to_center, grid)
    covered_by_nearest_synthetic = real_to_synthetic_nn <= real_to_real_nn
    beta_coverage_curve = np.array(
        [
            np.mean(covered_by_nearest_synthetic & (closest_synthetic_to_center <= radius))
            for radius in beta_radii
        ],
        dtype=np.float64,
    )

    closest_real_idx = synthetic_to_real_indices[:, 0]
    synthetic_to_real_nn = synthetic_to_real_distances[:, 0]
    authenticity_fixed = float(np.mean(synthetic_to_real_nn > real_to_real_nn[closest_real_idx]))
    authenticity_legacy = float(np.mean(real_to_real_nn[closest_synthetic_idx] < real_to_synthetic_nn))
    beta_coverage_at_095 = curve_value_at(grid, beta_coverage_curve, 0.95)
    beta_support_recall_at_095 = curve_value_at(grid, beta_support_recall_curve, 0.95)
    return {
        "alpha_grid": grid.tolist(),
        "alpha_precision_curve": alpha_precision_curve.tolist(),
        "beta_grid": grid.tolist(),
        "beta_recall_curve": beta_coverage_curve.tolist(),
        "beta_coverage_curve": beta_coverage_curve.tolist(),
        "beta_support_recall_curve": beta_support_recall_curve.tolist(),
        "delta_alpha_precision": delta_to_diagonal(grid, alpha_precision_curve),
        "delta_beta_recall": delta_to_diagonal(grid, beta_coverage_curve),
        "delta_beta_coverage": delta_to_diagonal(grid, beta_coverage_curve),
        "delta_beta_support_recall": delta_to_diagonal(grid, beta_support_recall_curve),
        "alpha_precision_auc": auc(grid, alpha_precision_curve),
        "beta_recall_auc": auc(grid, beta_coverage_curve),
        "beta_coverage_auc": auc(grid, beta_coverage_curve),
        "beta_support_recall_auc": auc(grid, beta_support_recall_curve),
        "alpha_precision_at_0.95": curve_value_at(grid, alpha_precision_curve, 0.95),
        "beta_recall_at_0.95": beta_coverage_at_095,
        "beta_coverage_at_0.95": beta_coverage_at_095,
        "beta_support_recall_at_0.95": beta_support_recall_at_095,
        "authenticity": authenticity_fixed,
        "authenticity_fixed": authenticity_fixed,
        "authenticity_legacy": authenticity_legacy,
    }


def evaluate_dataset(args: argparse.Namespace, dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = build_paths(args, dataset)
    for path in [paths.real, paths.synthetic, paths.info]:
        if not path.exists():
            raise FileNotFoundError(f"{dataset}: missing required file: {path}")

    real_df = pd.read_csv(paths.real)
    synthetic_df = pd.read_csv(paths.synthetic)
    synthetic_df = align_synthetic_columns(real_df, synthetic_df, dataset)
    info = read_info(paths.info)
    split = split_columns(real_df, info, include_target=not args.exclude_target)
    real_eval, synthetic_eval, n_eval = sample_same_size(real_df, synthetic_df, args.max_rows, args.seed)
    real_x, synthetic_x, categorical_features = build_feature_matrices(real_eval, synthetic_eval, split)
    metrics = compute_alpha_beta(real_x, synthetic_x, args.num_points)
    categorical_diagnostics = category_overlap_diagnostics(real_eval, synthetic_eval, split.categorical)

    summary = {
        "dataset": dataset,
        "real_path": str(paths.real),
        "synthetic_path": str(paths.synthetic),
        "info_path": str(paths.info),
        "real_rows": int(len(real_df)),
        "synthetic_rows": int(len(synthetic_df)),
        "eval_rows": int(n_eval),
        "feature_columns": int(len(split.numeric) + len(split.categorical)),
        "numeric_columns": int(len(split.numeric)),
        "categorical_columns": int(len(split.categorical)),
        "encoded_features": int(real_x.shape[1]),
        "encoded_categorical_features": categorical_features,
        "target_columns": "|".join(split.target),
        "target_included": not args.exclude_target,
        "max_rows": int(args.max_rows),
        "num_points": int(args.num_points),
        "seed": int(args.seed),
        **categorical_diagnostics,
        "delta_alpha_precision": metrics["delta_alpha_precision"],
        "delta_beta_recall": metrics["delta_beta_recall"],
        "delta_beta_coverage": metrics["delta_beta_coverage"],
        "delta_beta_support_recall": metrics["delta_beta_support_recall"],
        "alpha_precision_auc": metrics["alpha_precision_auc"],
        "beta_recall_auc": metrics["beta_recall_auc"],
        "beta_coverage_auc": metrics["beta_coverage_auc"],
        "beta_support_recall_auc": metrics["beta_support_recall_auc"],
        "alpha_precision_at_0.95": metrics["alpha_precision_at_0.95"],
        "beta_recall_at_0.95": metrics["beta_recall_at_0.95"],
        "beta_coverage_at_0.95": metrics["beta_coverage_at_0.95"],
        "beta_support_recall_at_0.95": metrics["beta_support_recall_at_0.95"],
        "authenticity": metrics["authenticity"],
        "authenticity_fixed": metrics["authenticity_fixed"],
        "authenticity_legacy": metrics["authenticity_legacy"],
    }
    curves = {
        "dataset": dataset,
        "metadata": summary,
        "curves": {
            "alpha_grid": metrics["alpha_grid"],
            "alpha_precision_curve": metrics["alpha_precision_curve"],
            "beta_grid": metrics["beta_grid"],
            "beta_recall_curve": metrics["beta_recall_curve"],
            "beta_coverage_curve": metrics["beta_coverage_curve"],
            "beta_support_recall_curve": metrics["beta_support_recall_curve"],
        },
    }
    return summary, curves


def write_outputs(
    out_dir: Path,
    summaries: list[dict[str, Any]],
    curves: dict[str, Any],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    curves_path = out_dir / "curves.json"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    with curves_path.open("w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2, ensure_ascii=False)
    return summary_path, curves_path


def print_summary(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    columns = [
        "dataset",
        "eval_rows",
        "encoded_features",
        "delta_alpha_precision",
        "delta_beta_recall",
        "alpha_precision_at_0.95",
        "beta_recall_at_0.95",
        "beta_support_recall_at_0.95",
        "authenticity",
        "categorical_exact_overlap_min",
        "categorical_strip_overlap_min",
    ]
    table = pd.DataFrame(summaries)[columns].copy()
    for col in columns[3:]:
        table[col] = table[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "nan")
    print(table.to_string(index=False))


def main() -> int:
    args = parse_args()
    summaries: list[dict[str, Any]] = []
    curves: dict[str, Any] = {
        "config": {
            "datasets": args.datasets,
            "model": args.model,
            "exp": args.exp,
            "data": args.data,
            "real_root": str(args.real_root),
            "synthetic_root": str(args.synthetic_root),
            "info_root": str(args.info_root),
            "max_rows": args.max_rows,
            "num_points": args.num_points,
            "seed": args.seed,
            "target_included": not args.exclude_target,
        },
        "datasets": {},
    }
    errors: list[str] = []

    for dataset in tqdm(args.datasets, desc="datasets"):
        try:
            summary, dataset_curves = evaluate_dataset(args, dataset)
        except Exception as exc:  # noqa: BLE001 - CLI should report dataset-specific failures.
            message = f"{dataset}: {exc}"
            if not args.continue_on_error:
                raise
            errors.append(message)
            print(f"[ERROR] {message}", file=sys.stderr)
            continue
        summaries.append(summary)
        curves["datasets"][dataset] = dataset_curves

    curves["errors"] = errors
    summary_path, curves_path = write_outputs(args.out_dir, summaries, curves)
    print_summary(summaries)
    print(f"\nsummary_csv={summary_path}")
    print(f"curves_json={curves_path}")
    if errors:
        print(f"errors={len(errors)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
