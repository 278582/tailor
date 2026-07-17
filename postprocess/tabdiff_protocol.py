from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data_io import load_csv
from .tabdiff_utils import get_tabdiff_paths


ROOT_DIR = Path(__file__).resolve().parents[1]


_TABDIFF_SELECTION_OVERRIDES: dict[str, dict[str, Any]] = {
    "adult": {
        "logical_name": "adult",
        "target_column": "income",
        "discrete_numerical_columns": ["education.num"],
        "privacy_sensitive_columns": ["age", "fnlwgt", "race", "sex", "native.country"],
        "artifact_root": ROOT_DIR / "artifacts" / "postprocess" / "tabdiff" / "adult",
    },
    "adult_tgm_w1": {
        "logical_name": "adult",
        "target_column": "income",
        "discrete_numerical_columns": ["education.num"],
        "privacy_sensitive_columns": ["age", "fnlwgt", "race", "sex", "native.country"],
        "artifact_root": ROOT_DIR / "artifacts" / "postprocess" / "tabdiff" / "adult_tgm_w1",
    },
    "magic": {
        "logical_name": "magic",
        "target_column": "class",
        "discrete_numerical_columns": [],
        "privacy_sensitive_columns": ["Length", "Width", "Size", "Alpha", "Dist"],
        "artifact_root": ROOT_DIR / "artifacts" / "postprocess" / "tabdiff" / "magic",
    },
    "shoppers": {
        "logical_name": "shoppers",
        "target_column": "Revenue",
        "discrete_numerical_columns": ["Administrative", "Informational", "ProductRelated"],
        "privacy_sensitive_columns": [
            "Administrative",
            "Administrative_Duration",
            "Informational",
            "Informational_Duration",
            "ProductRelated",
        ],
        "artifact_root": ROOT_DIR / "artifacts" / "postprocess" / "tabdiff" / "shoppers",
    },
}


@dataclass
class TabDiffSelectionContext:
    dataset_name: str
    logical_name: str
    info: dict[str, Any]
    task_type: str
    train_df: pd.DataFrame
    holdout_df: pd.DataFrame
    test_df: pd.DataFrame
    target_column: str
    numerical_columns: list[str]
    categorical_columns: list[str]
    discrete_numerical_columns: list[str]
    privacy_sensitive_columns: list[str]
    artifact_root: Path
    train_source_path: Path
    test_source_path: Path
    holdout_fraction: float
    holdout_strategy: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "logical_name": self.logical_name,
            "task_type": self.task_type,
            "target_column": self.target_column,
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "discrete_numerical_columns": self.discrete_numerical_columns,
            "privacy_sensitive_columns": self.privacy_sensitive_columns,
            "artifact_root": str(self.artifact_root),
            "train_source_path": str(self.train_source_path),
            "test_source_path": str(self.test_source_path),
            "holdout_fraction": self.holdout_fraction,
            "holdout_strategy": self.holdout_strategy,
            "train_rows": int(len(self.train_df)),
            "holdout_rows": int(len(self.holdout_df)),
            "test_rows": int(len(self.test_df)),
        }


def _load_info(dataset_name: str) -> dict[str, Any]:
    paths = get_tabdiff_paths(dataset_name)
    info_path = paths.data_dir / "info.json"
    with info_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _uses_lstripped_column_names(dataset_name: str) -> bool:
    return dataset_name == "news"


def _lstrip_column_names(columns: list[Any], *, dataset_name: str) -> list[str]:
    normalized = [str(column).lstrip() for column in columns]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Removing leading whitespace creates duplicate columns for dataset={dataset_name}")
    return normalized


def normalize_tabdiff_info(dataset_name: str, info: dict[str, Any]) -> dict[str, Any]:
    if not _uses_lstripped_column_names(dataset_name):
        return info
    normalized = dict(info)
    if "column_names" in normalized:
        normalized["column_names"] = _lstrip_column_names(
            list(normalized["column_names"]),
            dataset_name=dataset_name,
        )
    return normalized


def normalize_tabdiff_dataframe_columns(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if not _uses_lstripped_column_names(dataset_name):
        return df
    normalized_columns = _lstrip_column_names(list(df.columns), dataset_name=dataset_name)
    if list(df.columns) == normalized_columns:
        return df
    normalized = df.copy()
    normalized.columns = normalized_columns
    return normalized


def _dataset_override(dataset_name: str) -> dict[str, Any]:
    return _TABDIFF_SELECTION_OVERRIDES.get(dataset_name, {})


def _resolve_target_column(dataset_name: str, info: dict[str, Any], override: dict[str, Any]) -> str:
    if "target_column" in override:
        return str(override["target_column"])
    target_indices = info.get("target_col_idx", [])
    column_names = info.get("column_names", [])
    if target_indices and column_names:
        return str(column_names[int(target_indices[0])])
    raise KeyError(f"Cannot resolve target column for dataset={dataset_name}")


def _default_privacy_columns(info: dict[str, Any], target_column: str) -> list[str]:
    columns = [str(name) for name in info.get("column_names", [])]
    return [column for column in columns if column != target_column][:5]


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize_split(df: pd.DataFrame, info: dict[str, Any]) -> pd.DataFrame:
    if info.get("header", "infer") == "none":
        columns = info.get("column_names", [])
        if columns and list(df.columns) != columns:
            df = df.copy()
            df.columns = columns
    return df.reset_index(drop=True)


def resolve_tabdiff_selection_context(
    dataset_name: str,
    seed: int,
    holdout_fraction: float = 0.1,
) -> TabDiffSelectionContext:
    paths = get_tabdiff_paths(dataset_name)
    override = _dataset_override(dataset_name)
    info = normalize_tabdiff_info(dataset_name, _load_info(dataset_name))
    logical_name = str(override.get("logical_name", dataset_name))

    synthetic_train_path = paths.synthetic_dir / "real.csv"
    synthetic_val_path = paths.synthetic_dir / "val.csv"
    synthetic_test_path = paths.synthetic_dir / "test.csv"
    train_path = synthetic_train_path if synthetic_train_path.exists() else paths.data_dir / "train.csv"
    val_path = synthetic_val_path if synthetic_val_path.exists() else paths.data_dir / "val.csv"
    test_path = synthetic_test_path if synthetic_test_path.exists() else paths.data_dir / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Cannot resolve TabDiff train split for dataset={dataset_name}: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Cannot resolve TabDiff test split for dataset={dataset_name}: {test_path}")

    train_full_df = normalize_tabdiff_dataframe_columns(
        dataset_name,
        _normalize_split(load_csv(train_path), info=info),
    )
    test_df = normalize_tabdiff_dataframe_columns(
        dataset_name,
        _normalize_split(load_csv(test_path), info=info),
    )
    if val_path.exists():
        val_df = normalize_tabdiff_dataframe_columns(
            dataset_name,
            _normalize_split(load_csv(val_path), info=info),
        )
        holdout_df = val_df.reset_index(drop=True)
        holdout_strategy = "val_as_holdout_full_train"
    else:
        holdout_df = test_df.copy().reset_index(drop=True)
        holdout_strategy = "test_as_holdout_full_train"

    # Keep holdout_fraction for CLI/API compatibility; selection now uses full train and a file-backed holdout.
    train_df = train_full_df.reset_index(drop=True)

    target_column = _resolve_target_column(dataset_name=dataset_name, info=info, override=override)
    column_names = [str(name) for name in info.get("column_names", list(train_df.columns))]
    target_indices = [int(idx) for idx in info.get("target_col_idx", [])]
    target_columns = [column_names[idx] for idx in target_indices if 0 <= idx < len(column_names)]
    task_type = str(info.get("task_type", "binclass"))
    numerical_columns = [column_names[int(idx)] for idx in info.get("num_col_idx", [])]
    categorical_columns = [column_names[int(idx)] for idx in info.get("cat_col_idx", [])]
    if task_type == "regression":
        numerical_columns = _ordered_unique(numerical_columns + target_columns)
    else:
        categorical_columns = _ordered_unique(categorical_columns + target_columns)
    discrete_numerical_columns = list(override.get("discrete_numerical_columns", []))
    privacy_sensitive_columns = list(
        override.get("privacy_sensitive_columns", _default_privacy_columns(info=info, target_column=target_column))
    )
    artifact_root = Path(override.get("artifact_root", ROOT_DIR / "artifacts" / "postprocess" / "tabdiff" / logical_name))

    return TabDiffSelectionContext(
        dataset_name=dataset_name,
        logical_name=logical_name,
        info=info,
        task_type=task_type,
        train_df=train_df,
        holdout_df=holdout_df,
        test_df=test_df,
        target_column=target_column,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        discrete_numerical_columns=discrete_numerical_columns,
        privacy_sensitive_columns=privacy_sensitive_columns,
        artifact_root=artifact_root,
        train_source_path=train_path,
        test_source_path=test_path,
        holdout_fraction=float(holdout_fraction),
        holdout_strategy=holdout_strategy,
    )
