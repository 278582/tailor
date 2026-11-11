from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_io import ensure_dir, load_csv, save_csv, save_json
from .paths import (
    ADULT_HOLDOUT_PATH,
    ADULT_TEST_PATH,
    ADULT_TRAIN_PATH,
    DEFAULT_TABDIFF_DATASET_NAME,
    TABDIFF_DIR,
)


@dataclass
class TabDiffPaths:
    dataset_name: str
    data_dir: Path
    synthetic_dir: Path
    ckpt_root: Path
    result_root: Path


def get_tabdiff_paths(dataset_name: str = DEFAULT_TABDIFF_DATASET_NAME) -> TabDiffPaths:
    return TabDiffPaths(
        dataset_name=dataset_name,
        data_dir=TABDIFF_DIR / "data" / dataset_name,
        synthetic_dir=TABDIFF_DIR / "synthetic" / dataset_name,
        ckpt_root=TABDIFF_DIR / "tabdiff" / "ckpt" / dataset_name,
        result_root=TABDIFF_DIR / "tabdiff" / "result" / dataset_name,
    )


def load_tabdiff_base_info() -> dict[str, Any]:
    info_path = TABDIFF_DIR / "data" / "Info" / "adult.json"
    with info_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def get_column_name_mapping(
    column_names: list[str],
    num_col_idx: list[int],
    cat_col_idx: list[int],
    target_col_idx: list[int],
) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    idx_mapping: dict[int, int] = {}
    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)
    for idx in range(len(column_names)):
        if idx in num_col_idx:
            idx_mapping[idx] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[idx] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[idx] = curr_target_idx
            curr_target_idx += 1
    inverse_idx_mapping = {value: key for key, value in idx_mapping.items()}
    idx_name_mapping = {idx: name for idx, name in enumerate(column_names)}
    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def _infer_integer_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_columns: list[str],
    name_to_idx: dict[str, int],
) -> tuple[list[int], list[str], list[int]]:
    complete_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    int_col_idx: list[int] = []
    int_columns: list[str] = []
    int_col_idx_wrt_num: list[int] = []
    for pos, column in enumerate(num_columns):
        values = pd.to_numeric(complete_df[column], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        if np.allclose(values, np.rint(values)):
            int_col_idx.append(name_to_idx[column])
            int_columns.append(column)
            int_col_idx_wrt_num.append(pos)
    return int_col_idx, int_columns, int_col_idx_wrt_num


def _build_column_info(
    train_df: pd.DataFrame,
    num_col_idx: list[int],
    cat_col_idx: list[int],
    target_col_idx: list[int],
    column_names: list[str],
    task_type: str,
) -> dict[int, dict[str, Any]]:
    column_info: dict[int, dict[str, Any]] = {}
    for col_idx in num_col_idx:
        column = column_names[col_idx]
        series = train_df[column].astype(float)
        column_info[col_idx] = {
            "type": "numerical",
            "max": float(series.max()),
            "min": float(series.min()),
        }
    for col_idx in cat_col_idx:
        column = column_names[col_idx]
        column_info[col_idx] = {
            "type": "categorical",
            "categorizes": sorted(train_df[column].astype(str).unique().tolist()),
        }
    for col_idx in target_col_idx:
        column = column_names[col_idx]
        if task_type == "regression":
            series = train_df[column].astype(float)
            column_info[col_idx] = {
                "type": "numerical",
                "max": float(series.max()),
                "min": float(series.min()),
            }
        else:
            column_info[col_idx] = {
                "type": "categorical",
                "categorizes": sorted(train_df[column].astype(str).unique().tolist()),
            }
    return column_info


def _build_metadata(
    num_col_idx: list[int],
    cat_col_idx: list[int],
    target_col_idx: list[int],
    task_type: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"columns": {}}
    for col_idx in num_col_idx:
        metadata["columns"][col_idx] = {
            "sdtype": "numerical",
            "computer_representation": "Float",
        }
    for col_idx in cat_col_idx:
        metadata["columns"][col_idx] = {
            "sdtype": "categorical",
        }
    for col_idx in target_col_idx:
        metadata["columns"][col_idx] = (
            {
                "sdtype": "numerical",
                "computer_representation": "Float",
            }
            if task_type == "regression"
            else {"sdtype": "categorical"}
        )
    return metadata


def _normalize_split(
    df: pd.DataFrame,
    num_columns: list[str],
    cat_columns: list[str],
    target_columns: list[str],
) -> pd.DataFrame:
    normalized = df.copy()
    for column in num_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in cat_columns + target_columns:
        normalized[column] = normalized[column].fillna("nan").astype(str).str.strip()
    return normalized


def prepare_tabdiff_adult_dataset(
    dataset_name: str = DEFAULT_TABDIFF_DATASET_NAME,
    train_path: Path = ADULT_TRAIN_PATH,
    val_path: Path = ADULT_HOLDOUT_PATH,
    test_path: Path = ADULT_TEST_PATH,
) -> dict[str, Any]:
    base_info = load_tabdiff_base_info()
    paths = get_tabdiff_paths(dataset_name)
    data_dir = ensure_dir(paths.data_dir)
    synthetic_dir = ensure_dir(paths.synthetic_dir)

    train_df = load_csv(train_path)
    val_df = load_csv(val_path)
    test_df = load_csv(test_path)

    column_names = list(base_info["column_names"])
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if list(df.columns) != column_names:
            raise ValueError(
                f"{split_name} columns do not match TabDiff Adult schema. "
                f"expected={column_names}, actual={list(df.columns)}"
            )

    num_col_idx = [int(x) for x in base_info["num_col_idx"]]
    cat_col_idx = [int(x) for x in base_info["cat_col_idx"]]
    target_col_idx = [int(x) for x in base_info["target_col_idx"]]
    num_columns = [column_names[idx] for idx in num_col_idx]
    cat_columns = [column_names[idx] for idx in cat_col_idx]
    target_columns = [column_names[idx] for idx in target_col_idx]

    train_df = _normalize_split(train_df, num_columns=num_columns, cat_columns=cat_columns, target_columns=target_columns)
    val_df = _normalize_split(val_df, num_columns=num_columns, cat_columns=cat_columns, target_columns=target_columns)
    test_df = _normalize_split(test_df, num_columns=num_columns, cat_columns=cat_columns, target_columns=target_columns)

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
        column_names=column_names,
        num_col_idx=num_col_idx,
        cat_col_idx=cat_col_idx,
        target_col_idx=target_col_idx,
    )
    name_to_idx = {name: idx for idx, name in idx_name_mapping.items()}
    int_col_idx, int_columns, int_col_idx_wrt_num = _infer_integer_metadata(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        num_columns=num_columns,
        name_to_idx=name_to_idx,
    )

    np.save(data_dir / "X_num_train.npy", train_df[num_columns].to_numpy(dtype=np.float32))
    np.save(data_dir / "X_cat_train.npy", train_df[cat_columns].to_numpy(dtype=object))
    np.save(data_dir / "y_train.npy", train_df[target_columns].to_numpy(dtype=object))

    np.save(data_dir / "X_num_val.npy", val_df[num_columns].to_numpy(dtype=np.float32))
    np.save(data_dir / "X_cat_val.npy", val_df[cat_columns].to_numpy(dtype=object))
    np.save(data_dir / "y_val.npy", val_df[target_columns].to_numpy(dtype=object))

    np.save(data_dir / "X_num_test.npy", test_df[num_columns].to_numpy(dtype=np.float32))
    np.save(data_dir / "X_cat_test.npy", test_df[cat_columns].to_numpy(dtype=object))
    np.save(data_dir / "y_test.npy", test_df[target_columns].to_numpy(dtype=object))

    save_csv(data_dir / "train.csv", train_df)
    save_csv(data_dir / "val.csv", val_df)
    save_csv(data_dir / "test.csv", test_df)
    save_csv(synthetic_dir / "real.csv", train_df)
    save_csv(synthetic_dir / "val.csv", val_df)
    save_csv(synthetic_dir / "test.csv", test_df)

    info = {
        "name": dataset_name,
        "task_type": base_info["task_type"],
        "header": None,
        "column_names": column_names,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": f"data/{dataset_name}/train.csv",
        "val_path": f"data/{dataset_name}/val.csv",
        "test_path": f"data/{dataset_name}/test.csv",
        "train_num": int(len(train_df)),
        "val_num": int(len(val_df)),
        "test_num": int(len(test_df)),
        "idx_mapping": idx_mapping,
        "inverse_idx_mapping": inverse_idx_mapping,
        "idx_name_mapping": idx_name_mapping,
        "int_col_idx": int_col_idx,
        "int_columns": int_columns,
        "int_col_idx_wrt_num": int_col_idx_wrt_num,
        "column_info": _build_column_info(
            train_df=train_df,
            num_col_idx=num_col_idx,
            cat_col_idx=cat_col_idx,
            target_col_idx=target_col_idx,
            column_names=column_names,
            task_type=base_info["task_type"],
        ),
        "metadata": _build_metadata(
            num_col_idx=num_col_idx,
            cat_col_idx=cat_col_idx,
            target_col_idx=target_col_idx,
            task_type=base_info["task_type"],
        ),
    }
    save_json(data_dir / "info.json", info)

    manifest = {
        "dataset_name": dataset_name,
        "tabdiff_root": str(TABDIFF_DIR),
        "data_dir": str(data_dir),
        "synthetic_dir": str(synthetic_dir),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "num_columns": num_columns,
        "cat_columns": cat_columns,
        "target_columns": target_columns,
        "int_columns": int_columns,
    }
    save_json(data_dir / "prepare_manifest.json", manifest)
    return manifest


def find_latest_tabdiff_sample(dataset_name: str, exp_name: str) -> Path:
    paths = get_tabdiff_paths(dataset_name)
    candidates = sorted(paths.result_root.joinpath(exp_name).glob("**/samples.csv"))
    if not candidates:
        fallback_candidates = sorted(paths.result_root.glob("**/samples.csv"))
        if not fallback_candidates:
            raise FileNotFoundError(
                f"No TabDiff sample found for dataset={dataset_name}, exp_name={exp_name} "
                f"under {paths.result_root / exp_name}"
            )
        fallback_candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return fallback_candidates[0]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def find_latest_tabdiff_checkpoint(dataset_name: str, exp_name: str) -> Path:
    paths = get_tabdiff_paths(dataset_name)
    ckpt_dir = paths.ckpt_root / exp_name
    patterns = [
        "best_ema_model_*.pt",
        "best_model_*.pt",
        "model_*.pt",
        "ema_model_*.pt",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        matches = sorted(ckpt_dir.glob(pattern))
        if matches:
            candidates.extend(matches)
            break
    if not candidates:
        raise FileNotFoundError(
            f"No TabDiff checkpoint found for dataset={dataset_name}, exp_name={exp_name} under {ckpt_dir}"
        )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]
