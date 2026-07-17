from __future__ import annotations

import importlib.util
import inspect
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .logging_utils import get_logger

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

if TYPE_CHECKING:  # pragma: no cover
    from .selector import ParetoSelector


ROOT_DIR = Path(__file__).resolve().parents[1]
_TABDIFF_MLE_MODULE: Any | None = None
UTILITY_EXACT_EVALUATORS = ("tabdiff_mle", "torch_lightweight_mlp")
TORCH_LIGHTWEIGHT_MLP_DEFAULT_EPOCHS = 6
TORCH_LIGHTWEIGHT_MLP_DEFAULT_BATCH_SIZE = 2048
TORCH_LIGHTWEIGHT_MLP_DEFAULT_HIDDEN_DIM = 64
TORCH_LIGHTWEIGHT_MLP_DEFAULT_IMPORTANCE_SAMPLE_SIZE = 2000
REGRESSION_UTILITY_TARGET_TRANSFORM = "log_clip_1_20000"
REGRESSION_UTILITY_TARGET_CLIP_MIN = 1.0
REGRESSION_UTILITY_TARGET_CLIP_MAX = 20000.0


def _progress(iterable: Any | None = None, **kwargs: Any) -> Any:
    if _tqdm is None:
        return iterable if iterable is not None else _NullProgress()
    if iterable is None:
        return _tqdm(**kwargs)
    return _tqdm(iterable, **kwargs)


def _progress_write(message: str) -> None:
    get_logger().info(message)


class _NullProgress:
    def update(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _minmax_normalize(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.zeros(0, dtype=float)
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return np.zeros_like(array, dtype=float)
    valid = array[finite_mask]
    min_value = float(valid.min())
    max_value = float(valid.max())
    if max_value <= min_value + 1e-12:
        output = np.zeros_like(array, dtype=float)
        output[~finite_mask] = 0.0
        return output
    output = (array - min_value) / (max_value - min_value)
    output[~finite_mask] = 0.0
    return output.astype(float, copy=False)


def _regression_target_eval_scale(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.log(
        np.clip(
            np.asarray(values, dtype=float),
            REGRESSION_UTILITY_TARGET_CLIP_MIN,
            REGRESSION_UTILITY_TARGET_CLIP_MAX,
        )
    )


def _grouped_rank_normalize(values: Sequence[float] | np.ndarray, group_labels: Sequence[Any]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.zeros(0, dtype=float)
    if len(group_labels) != array.size:
        raise ValueError("group_labels length mismatch in _grouped_rank_normalize")

    finite_mask = np.isfinite(array)
    safe_values = np.where(finite_mask, array, np.nan)
    value_series = pd.Series(safe_values)
    group_series = pd.Series(group_labels)
    ranks = value_series.groupby(group_series, sort=False).rank(method="average", na_option="bottom")
    counts = value_series.groupby(group_series, sort=False).transform("count")
    denom = np.maximum(counts.to_numpy(dtype=float) - 1.0, 1.0)
    ranked = (ranks.to_numpy(dtype=float) - 1.0) / denom
    singleton_mask = counts.to_numpy(dtype=float) <= 1.0
    ranked[singleton_mask & finite_mask] = 1.0
    ranked[~finite_mask] = 0.0
    return np.clip(ranked, 0.0, 1.0).astype(float, copy=False)


def _gate_strata_for_df(selector: "ParetoSelector", df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.zeros(0, dtype=int)
    gate_probs = selector._prob_geomean_for_df(df.reset_index(drop=True), columns=selector.feature_columns)
    gate_edges = getattr(selector, "train_feature_gate_edges", selector.train_gate_edges)
    return selector._assign_bins_from_edges(gate_probs, gate_edges).astype(int, copy=False)


def _target_group_labels(
    target_series: pd.Series,
    reference_target: pd.Series,
    *,
    task_type: str,
) -> np.ndarray:
    if task_type == "classification":
        return target_series.astype(str).to_numpy(dtype=object)

    reference_values = pd.to_numeric(reference_target, errors="coerce").dropna().to_numpy(dtype=float)
    values = pd.to_numeric(target_series, errors="coerce").to_numpy(dtype=float)
    if reference_values.size == 0:
        return np.full(len(target_series), "reg_bin_0", dtype=object)

    bin_count = int(min(10, max(2, round(math.sqrt(reference_values.size)))))
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.unique(np.quantile(reference_values, quantiles))
    if edges.size <= 2:
        return np.full(len(target_series), "reg_bin_0", dtype=object)

    safe_values = np.where(np.isfinite(values), values, float(np.nanmedian(reference_values)))
    clipped = np.clip(safe_values, float(edges[0]), float(edges[-1]))
    bins = np.digitize(clipped, edges[1:-1], right=False).astype(int, copy=False)
    return pd.Series(bins, dtype="int64").astype(str).radd("reg_bin_").to_numpy(dtype=object)


def _balance_bucket_labels(selector: "ParetoSelector", df: pd.DataFrame, balance_column: str) -> np.ndarray:
    if df.empty:
        return np.zeros(0, dtype=object)
    if balance_column not in df.columns:
        raise ValueError(f"balance_column={balance_column!r} is not present in dataframe")

    info = selector.schema_card["columns"][balance_column]
    column_type = str(info.get("type", "categorical"))
    series = df[balance_column]
    if column_type == "categorical":
        filled = series.astype(object).where(series.notna(), "__MISSING__")
        labels = selector.high_cardinality_compressor.transform_series(pd.Series(filled), balance_column)
        return labels.astype(str).fillna("__MISSING__").to_numpy(dtype=object)

    values = pd.to_numeric(series, errors="coerce")
    fill_value = float(selector.numeric_impute_values.get(balance_column, 0.0))
    safe_values = values.fillna(fill_value).to_numpy(dtype=float)
    if column_type == "discrete_numerical":
        train_dist = selector.train_distributions[balance_column]
        legal_values = np.asarray(train_dist.get("values", []), dtype=float)
        if legal_values.size == 0:
            return np.full(len(series), "value_0", dtype=object)
        distances = np.abs(safe_values[:, None] - legal_values[None, :])
        matched = np.argmin(distances, axis=1).astype(int, copy=False)
        return pd.Series(matched).astype(str).radd("value_").to_numpy(dtype=object)

    train_dist = selector.train_distributions[balance_column]
    edges = np.asarray(train_dist.get("edges", [0.0, 1.0]), dtype=float)
    if edges.size <= 2:
        return np.full(len(series), "bin_0", dtype=object)
    clipped = np.clip(safe_values, float(edges[0]), float(edges[-1]))
    bins = np.digitize(clipped, edges[1:-1], right=False).astype(int, copy=False)
    return pd.Series(bins).astype(str).radd("bin_").to_numpy(dtype=object)


def _train_balance_target_mass(
    selector: "ParetoSelector",
    task_type: str,
    *,
    balance_column: str | None = None,
) -> pd.DataFrame:
    cache = getattr(selector, "_utility_balance_target_mass_cache", None)
    if cache is None:
        cache = {}
        setattr(selector, "_utility_balance_target_mass_cache", cache)
    cache_key = f"{task_type}|{balance_column or '__gate_stratum__'}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    train_target_labels = _target_group_labels(
        selector.train_df[selector.target_column],
        selector.train_df[selector.target_column],
        task_type=task_type,
    )
    if balance_column is None:
        train_gate_strata = _gate_strata_for_df(selector, selector.train_df)
        target_groups = pd.DataFrame(
            {
                "target_label": train_target_labels.astype(str),
                "gate_stratum": train_gate_strata.astype(int, copy=False),
            }
        )
        group_cols = ["target_label", "gate_stratum"]
    else:
        train_balance_buckets = _balance_bucket_labels(selector, selector.train_df, balance_column)
        target_groups = pd.DataFrame(
            {
                "target_label": train_target_labels.astype(str),
                "balance_bucket": train_balance_buckets.astype(str),
            }
        )
        group_cols = ["target_label", "balance_bucket"]
    target_mass = target_groups.value_counts(group_cols, normalize=True).rename("target_mass").reset_index()
    cache[cache_key] = target_mass
    return target_mass


def _balanced_static_components(
    selector: "ParetoSelector",
    candidate_df: pd.DataFrame,
    raw_static: Sequence[float] | np.ndarray,
    *,
    task_type: str,
    density_clip: tuple[float, float] = (0.5, 2.0),
    balance_column: str | None = None,
) -> dict[str, Any]:
    raw_array = np.asarray(raw_static, dtype=float)
    if candidate_df.empty or raw_array.size == 0:
        return {
            "target_labels": np.zeros(0, dtype=object),
            "gate_strata": np.zeros(0, dtype=int),
            "balance_buckets": np.zeros(0, dtype=object),
            "static_rank": np.zeros(0, dtype=float),
            "density_weight": np.zeros(0, dtype=float),
            "coverage_gain": np.zeros(0, dtype=float),
            "u_static_balanced": np.zeros(0, dtype=float),
        }

    gate_strata = _gate_strata_for_df(selector, candidate_df)
    target_labels = _target_group_labels(
        candidate_df[selector.target_column],
        selector.train_df[selector.target_column],
        task_type=task_type,
    )
    if balance_column is None:
        balance_buckets = pd.Series(gate_strata).astype(str).radd("gate_").to_numpy(dtype=object)
        pool_groups = pd.DataFrame(
            {
                "target_label": target_labels.astype(str),
                "gate_stratum": gate_strata.astype(int, copy=False),
            }
        )
        group_cols = ["target_label", "gate_stratum"]
    else:
        balance_buckets = _balance_bucket_labels(selector, candidate_df, balance_column)
        pool_groups = pd.DataFrame(
            {
                "target_label": target_labels.astype(str),
                "balance_bucket": balance_buckets.astype(str),
            }
        )
        group_cols = ["target_label", "balance_bucket"]
    group_codes = pd.factorize(pd.MultiIndex.from_frame(pool_groups), sort=False)[0]
    static_rank = _grouped_rank_normalize(raw_array, group_codes)

    pool_mass = pool_groups.value_counts(group_cols, normalize=True).rename("pool_mass").reset_index()
    target_mass = _train_balance_target_mass(selector, task_type, balance_column=balance_column)
    masses = pool_groups.merge(pool_mass, on=group_cols, how="left").merge(target_mass, on=group_cols, how="left")
    masses["pool_mass"] = masses["pool_mass"].fillna(0.0)
    masses["target_mass"] = masses["target_mass"].fillna(0.0)

    target_mass_values = masses["target_mass"].to_numpy(dtype=float)
    pool_mass_values = masses["pool_mass"].to_numpy(dtype=float)
    ratio = target_mass_values / np.clip(pool_mass_values, 1e-12, None)
    density_weight = np.clip(ratio, float(density_clip[0]), float(density_clip[1]))
    relative_deficit = (target_mass_values - pool_mass_values) / np.clip(target_mass_values, 1e-12, None)
    coverage_gain = np.clip(relative_deficit, 0.0, 1.0)
    u_static_balanced = static_rank * density_weight

    return {
        "target_labels": target_labels,
        "gate_strata": gate_strata,
        "balance_buckets": balance_buckets,
        "static_rank": static_rank,
        "density_weight": density_weight.astype(float, copy=False),
        "coverage_gain": coverage_gain.astype(float, copy=False),
        "u_static_balanced": u_static_balanced.astype(float, copy=False),
    }


def _records_to_df(records: list[dict[str, Any]], column_order: list[str]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=column_order)
    return pd.DataFrame([record["row"] for record in records], columns=column_order)


def _candidate_ids(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(record.get("candidate_id", idx)) for idx, record in enumerate(records)], dtype=int)


def _resolve_task_type(
    target_series: pd.Series,
    task_type: str | None = None,
) -> str:
    if task_type is not None:
        normalized = str(task_type).strip().lower()
        if normalized not in {"classification", "regression"}:
            raise ValueError(f"Unsupported task_type={task_type!r}")
        return normalized

    non_null = target_series.dropna()
    if non_null.empty:
        return "classification"
    if not pd.api.types.is_numeric_dtype(non_null):
        return "classification"
    unique_count = int(non_null.nunique(dropna=True))
    if unique_count <= max(20, int(round(0.01 * len(non_null)))):
        return "classification"
    return "regression"


def _build_feature_preprocessor(selector: "ParetoSelector") -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    numeric_features = [
        column
        for column in selector.feature_columns
        if selector.schema_card["columns"][column]["type"] in {"numerical", "discrete_numerical"}
    ]
    categorical_features = [
        column
        for column in selector.feature_columns
        if selector.schema_card["columns"][column]["type"] == "categorical"
    ]
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _make_ohe()),
                    ]
                ),
                categorical_features,
            )
        )
    if not transformers:
        raise ValueError("No feature columns available for utility proxy.")
    return ColumnTransformer(transformers=transformers)


def _build_static_estimator(
    *,
    task_type: str,
    backend: str,
    random_state: int,
) -> tuple[Any, str]:
    normalized = str(backend).strip().lower()
    if normalized == "auto":
        normalized = "logistic" if task_type == "classification" else "ridge"

    if task_type == "classification":
        if normalized == "logistic":
            return LogisticRegression(max_iter=1000), "logistic"
        if normalized in {"random_forest", "rf"}:
            return RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            ), "random_forest"
        raise ValueError(f"Unsupported classification backend={backend!r}")

    if normalized == "ridge":
        return Ridge(alpha=1.0, random_state=random_state), "ridge"
    if normalized in {"random_forest", "rf"}:
        return RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ), "random_forest"
    raise ValueError(f"Unsupported regression backend={backend!r}")


def _fit_lightweight_utility_model(
    selector: "ParetoSelector",
    train_df: pd.DataFrame,
    *,
    task_type: str,
    random_state: int,
) -> Pipeline:
    preprocess = _build_feature_preprocessor(selector)
    if task_type == "classification":
        estimator: Any = LogisticRegression(max_iter=1000)
    else:
        estimator = Ridge(alpha=1.0, random_state=random_state)
    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )
    model.fit(train_df[selector.feature_columns], train_df[selector.target_column])
    return model


def _score_subset_utility(
    selector: "ParetoSelector",
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    *,
    task_type: str,
    random_state: int,
) -> dict[str, Any]:
    if train_df.empty or holdout_df.empty:
        return {"available": False, "reason": "empty_input", "utility": 0.0, "metric": None}

    y_train = train_df[selector.target_column]
    y_holdout = holdout_df[selector.target_column]

    if task_type == "classification":
        if y_train.nunique(dropna=True) < 2 or y_holdout.nunique(dropna=True) < 2:
            return {"available": False, "reason": "single_class", "utility": 0.0, "metric": None}
        model = _fit_lightweight_utility_model(
            selector,
            train_df,
            task_type=task_type,
            random_state=random_state,
        )
        holdout_features = holdout_df[selector.feature_columns]
        holdout_labels = y_holdout
        classes_union = sorted(set(y_train.astype(str).tolist()) | set(y_holdout.astype(str).tolist()))
        if len(classes_union) == 2:
            positive_label = classes_union[-1]
            if positive_label in model.named_steps["model"].classes_:
                pos_idx = list(model.named_steps["model"].classes_).index(positive_label)
                probs = model.predict_proba(holdout_features)[:, pos_idx]
                y_binary = (holdout_labels.astype(str) == positive_label).astype(int)
                if int(np.unique(y_binary).size) == 2:
                    return {
                        "available": True,
                        "reason": None,
                        "utility": float(roc_auc_score(y_binary, probs)),
                        "metric": "roc_auc",
                    }
        preds = model.predict(holdout_features)
        return {
            "available": True,
            "reason": None,
            "utility": float(balanced_accuracy_score(holdout_labels.astype(str), pd.Series(preds).astype(str))),
            "metric": "balanced_accuracy",
        }

    train_target = pd.to_numeric(y_train, errors="coerce")
    holdout_target = pd.to_numeric(y_holdout, errors="coerce")
    valid_train = train_target.notna()
    valid_holdout = holdout_target.notna()
    if not bool(valid_train.all()) or not bool(valid_holdout.all()):
        return {"available": False, "reason": "non_numeric_regression_target", "utility": 0.0, "metric": None}
    model = _fit_lightweight_utility_model(
        selector,
        train_df,
        task_type=task_type,
        random_state=random_state,
    )
    preds = model.predict(holdout_df[selector.feature_columns])
    rmse = math.sqrt(mean_squared_error(holdout_target.to_numpy(dtype=float), np.asarray(preds, dtype=float)))
    scale = float(np.std(holdout_target.to_numpy(dtype=float), ddof=0))
    return {
        "available": True,
        "reason": None,
        "utility": float(1.0 - rmse / (scale + 1e-12)),
        "metric": "one_minus_rmse_over_std",
        "rmse": float(rmse),
        "target_std": scale,
    }


def _resolve_tabdiff_dataset_name(selector: "ParetoSelector") -> str | None:
    for source in (selector.schema_card, selector.stats_card):
        if not isinstance(source, dict):
            continue
        for key in ("dataset", "dataset_name", "logical_name"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return None


def _normalize_tabdiff_task_type(
    selector: "ParetoSelector",
    *,
    task_type: str | None = None,
) -> str:
    resolved = _resolve_task_type(selector.train_df[selector.target_column], task_type=task_type)
    if resolved == "regression":
        return "regression"
    class_count = int(selector.train_df[selector.target_column].astype(str).nunique(dropna=True))
    return "binclass" if class_count <= 2 else "multiclass"


def _infer_tabdiff_info(
    selector: "ParetoSelector",
    *,
    task_type: str | None = None,
) -> dict[str, Any]:
    column_names = list(selector.column_order)
    target_idx = column_names.index(selector.target_column)
    num_col_idx: list[int] = []
    cat_col_idx: list[int] = []
    for idx, column in enumerate(column_names):
        if idx == target_idx:
            continue
        column_type = selector.schema_card["columns"][column]["type"]
        if column_type in {"numerical", "discrete_numerical"}:
            num_col_idx.append(idx)
        else:
            cat_col_idx.append(idx)
    return {
        "task_type": _normalize_tabdiff_task_type(selector, task_type=task_type),
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": [target_idx],
        "column_names": column_names,
        "header": "infer",
    }


def _load_tabdiff_info(
    selector: "ParetoSelector",
    *,
    task_type: str | None = None,
) -> tuple[dict[str, Any], str | None, str | None, str]:
    dataset_name = _resolve_tabdiff_dataset_name(selector)
    if dataset_name is not None:
        info_path = ROOT_DIR / "third_party" / "TabDiff" / "data" / dataset_name / "info.json"
        if info_path.exists():
            with info_path.open("r", encoding="utf-8") as fp:
                return json.load(fp), dataset_name, str(info_path), "tabdiff_data_info"
    return _infer_tabdiff_info(selector, task_type=task_type), dataset_name, None, "inferred_from_schema_card"


def _normalize_tabdiff_split(
    df: pd.DataFrame,
    column_names: Sequence[str],
) -> pd.DataFrame:
    use_columns = list(column_names)
    normalized = df.reset_index(drop=True).copy()
    if len(normalized.columns) == len(use_columns):
        missing = [column for column in use_columns if column not in normalized.columns]
        if missing:
            normalized.columns = use_columns
    return normalized.loc[:, use_columns].reset_index(drop=True)


def _coerce_tabdiff_mle_frame(df: pd.DataFrame, info: dict[str, Any]) -> pd.DataFrame:
    column_names = list(info.get("column_names") or df.columns)
    normalized = _normalize_tabdiff_split(df, column_names).copy()
    task_type = str(info.get("task_type", "binclass"))
    num_indices = [int(idx) for idx in info.get("num_col_idx", [])]
    cat_indices = [int(idx) for idx in info.get("cat_col_idx", [])]
    target_indices = [int(idx) for idx in info.get("target_col_idx", [])]
    if task_type == "regression":
        num_indices += target_indices
    else:
        cat_indices += target_indices

    for idx in num_indices:
        if 0 <= idx < len(column_names):
            column = column_names[idx]
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for idx in cat_indices:
        if 0 <= idx < len(column_names):
            column = column_names[idx]
            normalized[column] = normalized[column].astype(str)
    return normalized


def _load_tabdiff_val_split(
    dataset_name: str | None,
    column_names: Sequence[str],
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    if not dataset_name:
        return None, None, None
    candidate_paths = [
        ROOT_DIR / "third_party" / "TabDiff" / "synthetic" / dataset_name / "val.csv",
        ROOT_DIR / "third_party" / "TabDiff" / "data" / dataset_name / "val.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            df = pd.read_csv(path)
            return _normalize_tabdiff_split(df, column_names), "tabdiff_val_csv", str(path)
    return None, "tabdiff_internal_split", None


def _load_tabdiff_mle_module() -> Any:
    global _TABDIFF_MLE_MODULE
    if _TABDIFF_MLE_MODULE is not None:
        return _TABDIFF_MLE_MODULE

    module_path = ROOT_DIR / "third_party" / "TabDiff" / "eval" / "mle" / "mle.py"
    spec = importlib.util.spec_from_file_location("tabdiff_mle_adapter", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load TabDiff MLE module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.tqdm = lambda iterable, *args, **kwargs: iterable
    _patch_tabdiff_mle_sklearn_compat(module)
    _TABDIFF_MLE_MODULE = module
    return module


def _patch_tabdiff_mle_sklearn_compat(mle_module: Any) -> None:
    mse = getattr(mle_module, "mean_squared_error", None)
    if mse is None:
        return
    try:
        parameters = inspect.signature(mse).parameters
    except (TypeError, ValueError):
        return
    if "squared" in parameters:
        return

    def _mean_squared_error_compat(y_true: Any, y_pred: Any, *args: Any, squared: bool = True, **kwargs: Any) -> float:
        value = float(mse(y_true, y_pred, *args, **kwargs))
        return value if squared else math.sqrt(value)

    mle_module.mean_squared_error = _mean_squared_error_compat


def _configure_tabdiff_tree_method(
    mle_module: Any,
    *,
    tree_method: str,
) -> None:
    for model_specs in getattr(mle_module, "_MODELS", {}).values():
        for model_spec in model_specs:
            kwargs = model_spec.get("kwargs")
            if isinstance(kwargs, dict) and "tree_method" in kwargs:
                kwargs["tree_method"] = [str(tree_method)]


def _scores_to_named_map(scores: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in scores:
        row = dict(item)
        name = str(row.pop("name"))
        output[name] = row
    return output


def _build_tabdiff_overall_scores(
    *,
    tabdiff_task_type: str,
    evaluator_output: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    if tabdiff_task_type == "regression":
        best_r2_scores, best_rmse_scores = evaluator_output
        return {
            "best_r2_scores": _scores_to_named_map(best_r2_scores),
            "best_rmse_scores": _scores_to_named_map(best_rmse_scores),
        }

    best_f1_scores, best_weighted_scores, best_auroc_scores, best_acc_scores, best_avg_scores = evaluator_output
    return {
        "best_f1_scores": _scores_to_named_map(best_f1_scores),
        "best_weighted_scores": _scores_to_named_map(best_weighted_scores),
        "best_auroc_scores": _scores_to_named_map(best_auroc_scores),
        "best_acc_scores": _scores_to_named_map(best_acc_scores),
        "best_avg_scores": _scores_to_named_map(best_avg_scores),
    }


def _extract_tabdiff_primary_metric(
    *,
    tabdiff_task_type: str,
    overall_scores: dict[str, dict[str, dict[str, Any]]],
) -> tuple[str, float, str, str]:
    if tabdiff_task_type == "regression":
        score_group = "best_rmse_scores"
        preferred_model = "XGBRegressor"
        metric = "RMSE"
    else:
        score_group = "best_auroc_scores"
        preferred_model = "XGBClassifier"
        metric = "roc_auc"

    score_block = overall_scores.get(score_group, {})
    model_name = preferred_model if preferred_model in score_block else next(iter(score_block), None)
    if model_name is None:
        raise KeyError(f"Cannot resolve primary TabDiff MLE score from {score_group}")
    metric_block = score_block.get(model_name, {})
    if metric not in metric_block:
        raise KeyError(f"Cannot resolve metric={metric} for model={model_name}")
    return metric, float(metric_block[metric]), score_group, model_name


def _normalize_utility_exact_evaluator(value: Any) -> str:
    normalized = str(value or "tabdiff_mle").strip().lower().replace("-", "_")
    aliases = {
        "xgboost": "tabdiff_mle",
        "tabdiff": "tabdiff_mle",
        "tabdiff_xgboost": "tabdiff_mle",
        "torch": "torch_lightweight_mlp",
        "torch_mlp": "torch_lightweight_mlp",
        "lightweight_torch_mlp": "torch_lightweight_mlp",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in UTILITY_EXACT_EVALUATORS:
        raise ValueError(
            f"Unsupported utility_exact_evaluator={value!r}; expected one of {', '.join(UTILITY_EXACT_EVALUATORS)}"
        )
    return normalized


def _select_torch_device(selector: "ParetoSelector") -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torch_unavailable: {exc}") from exc

    requested = str(getattr(selector, "nn_device", "cpu") or "cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"nn_device={requested} requested but torch.cuda.is_available() is False")
        return torch.device(requested)
    return torch.device("cpu")


class _TorchUtilityMLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        import torch

        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(int(hidden_dim), max(2, int(hidden_dim) // 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(max(2, int(hidden_dim) // 2), int(output_dim)),
        )

    def to(self, device: Any) -> "_TorchUtilityMLP":
        self.model.to(device)
        return self

    def parameters(self) -> Any:
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def __call__(self, x: Any) -> Any:
        return self.model(x)


def _torch_predict(
    model: _TorchUtilityMLP,
    x: np.ndarray,
    *,
    device: Any,
    task_type: str,
    class_count: int,
    batch_size: int,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> np.ndarray:
    import torch

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            xb = torch.as_tensor(x[start : start + int(batch_size)], dtype=torch.float32, device=device)
            logits = model(xb)
            if task_type == "classification":
                if class_count == 2:
                    pred = torch.sigmoid(logits.reshape(-1))
                else:
                    pred = torch.softmax(logits, dim=1)
            else:
                pred = logits.reshape(-1)
                if target_mean is not None and target_std is not None:
                    pred = pred * float(target_std) + float(target_mean)
            outputs.append(pred.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _torch_score_predictions(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    task_type: str,
    class_count: int,
) -> tuple[str, float, str]:
    if task_type == "classification":
        if class_count == 2 and np.unique(y_true).size == 2:
            return "roc_auc", float(roc_auc_score(y_true, pred.reshape(-1))), "best_auroc_scores"
        labels = np.asarray(pred).argmax(axis=1) if np.asarray(pred).ndim > 1 else (np.asarray(pred) >= 0.5).astype(int)
        return "accuracy", float(balanced_accuracy_score(y_true, labels)), "best_acc_scores"
    rmse = math.sqrt(mean_squared_error(np.asarray(y_true, dtype=float), np.asarray(pred, dtype=float)))
    return "RMSE", float(rmse), "best_rmse_scores"


def _torch_permutation_importance(
    *,
    selector: "ParetoSelector",
    model: _TorchUtilityMLP,
    preprocess: ColumnTransformer,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    task_type: str,
    class_count: int,
    metric: str,
    baseline: float,
    device: Any,
    batch_size: int,
    random_state: int,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> list[dict[str, Any]]:
    feature_columns = list(selector.feature_columns)
    if not feature_columns:
        return []
    sample_size = int(
        getattr(
            selector,
            "utility_exact_torch_importance_sample_size",
            TORCH_LIGHTWEIGHT_MLP_DEFAULT_IMPORTANCE_SAMPLE_SIZE,
        )
    )
    use_df = test_df[feature_columns].reset_index(drop=True)
    use_y = np.asarray(y_test)
    if sample_size > 0 and len(use_df) > sample_size:
        rng = np.random.default_rng(int(random_state) + 913)
        indices = rng.choice(len(use_df), size=sample_size, replace=False)
        use_df = use_df.iloc[indices].reset_index(drop=True)
        use_y = use_y[indices]
        x_base = preprocess.transform(use_df)
        base_pred = _torch_predict(
            model,
            np.asarray(x_base, dtype=np.float32),
            device=device,
            task_type=task_type,
            class_count=class_count,
            batch_size=batch_size,
            target_mean=target_mean,
            target_std=target_std,
        )
        base_metric, base_score, _ = _torch_score_predictions(use_y, base_pred, task_type=task_type, class_count=class_count)
        if base_metric == metric:
            baseline = base_score

    rng = np.random.default_rng(int(random_state) + 991)
    drops: dict[str, float] = {}
    for feature in feature_columns:
        permuted = use_df.copy()
        values = permuted[feature].to_numpy(copy=True)
        rng.shuffle(values)
        permuted[feature] = values
        x_perm = preprocess.transform(permuted)
        pred = _torch_predict(
            model,
            np.asarray(x_perm, dtype=np.float32),
            device=device,
            task_type=task_type,
            class_count=class_count,
            batch_size=batch_size,
            target_mean=target_mean,
            target_std=target_std,
        )
        perm_metric, perm_score, _ = _torch_score_predictions(use_y, pred, task_type=task_type, class_count=class_count)
        if perm_metric != metric:
            drops[feature] = 0.0
        elif str(metric).lower() == "rmse":
            drops[feature] = max(0.0, float(perm_score) - float(baseline))
        else:
            drops[feature] = max(0.0, float(baseline) - float(perm_score))
    total = float(sum(drops.values()))
    if total <= 0.0:
        return [
            {"feature": feature, "importance": 0.0, "rank": rank}
            for rank, feature in enumerate(feature_columns[:10], start=1)
        ]
    rows = [
        {"feature": feature, "importance": float(value / total)}
        for feature, value in drops.items()
    ]
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return [
        {"feature": item["feature"], "importance": float(item["importance"]), "rank": rank}
        for rank, item in enumerate(rows[:10], start=1)
    ]


def _evaluate_torch_lightweight_mlp_utility(
    selector: "ParetoSelector",
    syn_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    task_type: str | None = None,
    random_state: int,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    use_task_type = _resolve_task_type(selector.train_df[selector.target_column], task_type=task_type)
    preprocess = _build_feature_preprocessor(selector)
    train_features = syn_df[selector.feature_columns].reset_index(drop=True)
    test_features = test_df[selector.feature_columns].reset_index(drop=True)
    x_train = np.asarray(preprocess.fit_transform(train_features), dtype=np.float32)
    x_test = np.asarray(preprocess.transform(test_features), dtype=np.float32)
    if x_train.size == 0 or x_test.size == 0:
        raise ValueError("empty_torch_utility_features")

    device = _select_torch_device(selector)
    epochs = max(1, int(getattr(selector, "utility_exact_torch_epochs", TORCH_LIGHTWEIGHT_MLP_DEFAULT_EPOCHS)))
    batch_size = max(1, int(getattr(selector, "utility_exact_torch_batch_size", TORCH_LIGHTWEIGHT_MLP_DEFAULT_BATCH_SIZE)))
    hidden_dim = max(4, int(getattr(selector, "utility_exact_torch_hidden_dim", TORCH_LIGHTWEIGHT_MLP_DEFAULT_HIDDEN_DIM)))
    torch.manual_seed(int(random_state))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(random_state))

    target_mean: float | None = None
    target_std: float | None = None
    regression_target_raw_std: float | None = None
    regression_target_eval_std: float | None = None
    if use_task_type == "classification":
        train_labels = syn_df[selector.target_column].astype(str)
        test_labels = test_df[selector.target_column].astype(str)
        labels = sorted(set(train_labels.dropna().tolist()) | set(test_labels.dropna().tolist()))
        label_map = {label: idx for idx, label in enumerate(labels)}
        y_train = train_labels.map(label_map).to_numpy(dtype=np.int64)
        y_test = test_labels.map(label_map).to_numpy(dtype=np.int64)
        class_count = len(labels)
        if class_count < 2 or np.unique(y_train).size < 2:
            raise ValueError("single_class_torch_utility")
        output_dim = 1 if class_count == 2 else class_count
        y_tensor = (
            torch.as_tensor(y_train.astype(np.float32).reshape(-1, 1), dtype=torch.float32)
            if class_count == 2
            else torch.as_tensor(y_train, dtype=torch.long)
        )
        criterion: Any = torch.nn.BCEWithLogitsLoss() if class_count == 2 else torch.nn.CrossEntropyLoss()
        primary_model = "TorchMLPClassifier"
    else:
        target_train = pd.to_numeric(syn_df[selector.target_column], errors="coerce")
        target_test = pd.to_numeric(test_df[selector.target_column], errors="coerce")
        if target_train.isna().any() or target_test.isna().any():
            raise ValueError("non_numeric_regression_target")
        y_train_raw = target_train.to_numpy(dtype=float)
        y_test_raw = target_test.to_numpy(dtype=float)
        y_train_eval = _regression_target_eval_scale(y_train_raw)
        y_test = _regression_target_eval_scale(y_test_raw)
        regression_target_raw_std = float(np.std(y_train_raw, ddof=0))
        regression_target_eval_std = float(np.std(y_train_eval, ddof=0))
        target_mean = float(np.mean(y_train_eval))
        target_std = regression_target_eval_std or 1.0
        y_train = ((y_train_eval - target_mean) / target_std).astype(np.float32)
        class_count = 1
        output_dim = 1
        y_tensor = torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        criterion = torch.nn.MSELoss()
        primary_model = "TorchMLPRegressor"

    model = _TorchUtilityMLP(input_dim=x_train.shape[1], hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    dataset = TensorDataset(torch.as_tensor(x_train, dtype=torch.float32), y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_history.append({"epoch": int(epoch), "loss": float(np.mean(losses)) if losses else None})

    pred = _torch_predict(
        model,
        x_test,
        device=device,
        task_type=use_task_type,
        class_count=class_count,
        batch_size=batch_size,
        target_mean=target_mean,
        target_std=target_std,
    )
    metric, overall_value, primary_group = _torch_score_predictions(
        np.asarray(y_test),
        pred,
        task_type=use_task_type,
        class_count=class_count,
    )
    importance_sample_size = int(
        getattr(
            selector,
            "utility_exact_torch_importance_sample_size",
            TORCH_LIGHTWEIGHT_MLP_DEFAULT_IMPORTANCE_SAMPLE_SIZE,
        )
    )
    importance_rows = int(
        importance_sample_size
        if importance_sample_size > 0 and len(test_df) > importance_sample_size
        else len(test_df)
    )
    feature_importance = _torch_permutation_importance(
        selector=selector,
        model=model,
        preprocess=preprocess,
        test_df=test_df.reset_index(drop=True),
        y_test=np.asarray(y_test),
        task_type=use_task_type,
        class_count=class_count,
        metric=metric,
        baseline=overall_value,
        device=device,
        batch_size=batch_size,
        random_state=random_state,
        target_mean=target_mean,
        target_std=target_std,
    )
    return {
        "available": True,
        "reason": None,
        "metric": metric,
        "overall": float(overall_value),
        "task_type": use_task_type,
        "tabdiff_task_type": "regression" if use_task_type == "regression" else ("binclass" if class_count == 2 else "multiclass"),
        "dataset_name": _resolve_tabdiff_dataset_name(selector),
        "info_source": "selector_schema_card",
        "info_path": None,
        "val_source": None,
        "val_path": None,
        "search_holdout_used": False,
        "runtime_device": str(device),
        "runtime_model_device": str(next(model.parameters()).device),
        "runtime_note": "torch_lightweight_mlp exact utility evaluator",
        "runtime_torch": {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "primary_score_group": primary_group,
        "primary_model": primary_model,
        "overall_scores": {primary_group: {primary_model: {metric: float(overall_value)}}},
        "feature_importance": feature_importance,
        "feature_importance_method": "permutation_importance_on_holdout",
        "regression_target_transform": REGRESSION_UTILITY_TARGET_TRANSFORM if use_task_type == "regression" else None,
        "regression_target_clip_min": REGRESSION_UTILITY_TARGET_CLIP_MIN if use_task_type == "regression" else None,
        "regression_target_clip_max": REGRESSION_UTILITY_TARGET_CLIP_MAX if use_task_type == "regression" else None,
        "regression_target_raw_std": regression_target_raw_std,
        "regression_target_eval_std": regression_target_eval_std,
        "train_history": train_history,
        "torch_train_rows": int(len(syn_df)),
        "torch_test_rows": int(len(test_df)),
        "torch_importance_rows": int(importance_rows),
        "torch_importance_sample_size": int(importance_sample_size),
        "torch_epochs": int(epochs),
        "torch_batch_size": int(batch_size),
    }


def _evaluate_tabdiff_exact_utility(
    selector: "ParetoSelector",
    syn_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    task_type: str | None = None,
    random_state: int,
) -> dict[str, Any]:
    info, dataset_name, info_path, info_source = _load_tabdiff_info(selector, task_type=task_type)
    column_names = list(info.get("column_names") or selector.column_order)
    syn_norm = _coerce_tabdiff_mle_frame(syn_df, info)
    test_norm = _coerce_tabdiff_mle_frame(test_df, info)
    val_df, val_source, val_path = _load_tabdiff_val_split(dataset_name, column_names)
    if val_df is not None:
        val_df = _coerce_tabdiff_mle_frame(val_df, info)

    mle_module = _load_tabdiff_mle_module()
    tree_method = "gpu_hist" if str(getattr(selector, "nn_device", "cpu")).startswith("cuda") else "hist"
    _configure_tabdiff_tree_method(mle_module, tree_method=tree_method)
    evaluator = mle_module.get_evaluator(str(info["task_type"]))

    rng_state = np.random.get_state()
    np.random.seed(int(random_state))
    try:
        evaluator_output = evaluator(
            syn_norm.to_numpy(),
            test_norm.to_numpy(),
            dict(info),
            val=None if val_df is None else val_df.to_numpy(),
        )
    except Exception as exc:
        if tree_method == "gpu_hist":
            _configure_tabdiff_tree_method(mle_module, tree_method="hist")
            evaluator_output = evaluator(
                syn_norm.to_numpy(),
                test_norm.to_numpy(),
                dict(info),
                val=None if val_df is None else val_df.to_numpy(),
            )
            tree_method = "hist"
            runtime_note = f"gpu_hist_failed_fallback_to_hist: {exc}"
        else:
            raise
    else:
        runtime_note = None
    finally:
        np.random.set_state(rng_state)

    overall_scores = _build_tabdiff_overall_scores(
        tabdiff_task_type=str(info["task_type"]),
        evaluator_output=evaluator_output,
    )
    metric, overall_value, primary_group, primary_model = _extract_tabdiff_primary_metric(
        tabdiff_task_type=str(info["task_type"]),
        overall_scores=overall_scores,
    )
    return {
        "available": True,
        "reason": None,
        "metric": metric,
        "overall": overall_value,
        "task_type": _resolve_task_type(selector.train_df[selector.target_column], task_type=task_type),
        "tabdiff_task_type": str(info["task_type"]),
        "dataset_name": dataset_name,
        "info_source": info_source,
        "info_path": info_path,
        "val_source": val_source,
        "val_path": val_path,
        "search_holdout_used": False,
        "runtime_tree_method": tree_method,
        "runtime_note": runtime_note,
        "primary_score_group": primary_group,
        "primary_model": primary_model,
        "overall_scores": overall_scores,
    }


def compute_utility_exact_metrics(
    selector: "ParetoSelector",
    syn_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    task_type: str | None = None,
    random_state: int | None = None,
    search_holdout_used: bool = False,
    evaluator: str | None = None,
) -> dict[str, Any]:
    if syn_df.empty or test_df.empty or selector.target_column not in syn_df.columns:
        return {"available": False, "reason": "empty_input"}

    use_random_state = selector.seed if random_state is None else int(random_state)
    evaluator_name = _normalize_utility_exact_evaluator(
        evaluator if evaluator is not None else getattr(selector, "utility_exact_evaluator", "tabdiff_mle")
    )
    try:
        if evaluator_name == "tabdiff_mle":
            overall = _evaluate_tabdiff_exact_utility(
                selector,
                syn_df.reset_index(drop=True),
                test_df.reset_index(drop=True),
                task_type=task_type,
                random_state=use_random_state,
            )
            protocol = "tabdiff_mle"
            region_reason = "disabled_for_tabdiff_mle_cost"
        else:
            overall = _evaluate_torch_lightweight_mlp_utility(
                selector,
                syn_df.reset_index(drop=True),
                test_df.reset_index(drop=True),
                task_type=task_type,
                random_state=use_random_state,
            )
            protocol = "torch_lightweight_mlp"
            region_reason = "disabled_for_torch_lightweight_mlp_cost"
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{evaluator_name}_failed",
            "error": str(exc),
            "protocol": evaluator_name,
        }

    gate_probs = selector._prob_geomean_for_df(test_df.reset_index(drop=True), columns=selector.feature_columns)
    if gate_probs.size == 0:
        return {
            "available": True,
            "reason": None,
            "protocol": protocol,
            "task_type": overall.get("task_type"),
            "tabdiff_task_type": overall.get("tabdiff_task_type"),
            "metric": overall.get("metric"),
            "overall": float(overall.get("overall", 0.0)),
            "tail": None,
            "middle": None,
            "mode": None,
            "rows": {"tail": 0, "middle": 0, "mode": 0},
            "region_metrics_available": False,
            "region_metrics_reason": region_reason,
            "dataset_name": overall.get("dataset_name"),
            "info_source": overall.get("info_source"),
            "info_path": overall.get("info_path"),
            "val_source": overall.get("val_source"),
            "val_path": overall.get("val_path"),
            "search_holdout_used": bool(search_holdout_used or overall.get("search_holdout_used", False)),
            "runtime_tree_method": overall.get("runtime_tree_method"),
            "runtime_device": overall.get("runtime_device"),
            "runtime_model_device": overall.get("runtime_model_device"),
            "runtime_note": overall.get("runtime_note"),
            "runtime_torch": overall.get("runtime_torch"),
            "primary_score_group": overall.get("primary_score_group"),
            "primary_model": overall.get("primary_model"),
            "overall_scores": overall.get("overall_scores"),
            "feature_importance": overall.get("feature_importance"),
            "feature_importance_method": overall.get("feature_importance_method"),
            "regression_target_transform": overall.get("regression_target_transform"),
            "regression_target_clip_min": overall.get("regression_target_clip_min"),
            "regression_target_clip_max": overall.get("regression_target_clip_max"),
            "regression_target_raw_std": overall.get("regression_target_raw_std"),
            "regression_target_eval_std": overall.get("regression_target_eval_std"),
            "train_history": overall.get("train_history"),
            "torch_train_rows": overall.get("torch_train_rows"),
            "torch_test_rows": overall.get("torch_test_rows"),
            "torch_importance_rows": overall.get("torch_importance_rows"),
            "torch_importance_sample_size": overall.get("torch_importance_sample_size"),
            "torch_epochs": overall.get("torch_epochs"),
            "torch_batch_size": overall.get("torch_batch_size"),
        }

    quantiles = np.quantile(gate_probs, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    quantiles = np.unique(quantiles)
    if len(quantiles) < 4:
        quantiles = np.array([gate_probs.min() - 1e-12, gate_probs.mean(), gate_probs.mean(), gate_probs.max() + 1e-12])
    bins = np.digitize(gate_probs, quantiles[1:-1], right=False)

    return {
        "available": True,
        "reason": None,
        "protocol": protocol,
        "task_type": overall.get("task_type"),
        "tabdiff_task_type": overall.get("tabdiff_task_type"),
        "metric": overall.get("metric"),
        "overall": float(overall.get("overall", 0.0)),
        "tail": None,
        "middle": None,
        "mode": None,
        "rows": {
            "tail": int((bins == 0).sum()),
            "middle": int((bins == 1).sum()),
            "mode": int((bins == 2).sum()),
        },
        "region_metrics_available": False,
        "region_metrics_reason": region_reason,
        "dataset_name": overall.get("dataset_name"),
        "info_source": overall.get("info_source"),
        "info_path": overall.get("info_path"),
        "val_source": overall.get("val_source"),
        "val_path": overall.get("val_path"),
        "search_holdout_used": bool(search_holdout_used or overall.get("search_holdout_used", False)),
        "runtime_tree_method": overall.get("runtime_tree_method"),
        "runtime_device": overall.get("runtime_device"),
        "runtime_model_device": overall.get("runtime_model_device"),
        "runtime_note": overall.get("runtime_note"),
        "runtime_torch": overall.get("runtime_torch"),
        "primary_score_group": overall.get("primary_score_group"),
        "primary_model": overall.get("primary_model"),
        "overall_scores": overall.get("overall_scores"),
        "feature_importance": overall.get("feature_importance"),
        "feature_importance_method": overall.get("feature_importance_method"),
        "regression_target_transform": overall.get("regression_target_transform"),
        "regression_target_clip_min": overall.get("regression_target_clip_min"),
        "regression_target_clip_max": overall.get("regression_target_clip_max"),
        "regression_target_raw_std": overall.get("regression_target_raw_std"),
        "regression_target_eval_std": overall.get("regression_target_eval_std"),
        "train_history": overall.get("train_history"),
        "torch_train_rows": overall.get("torch_train_rows"),
        "torch_test_rows": overall.get("torch_test_rows"),
        "torch_importance_rows": overall.get("torch_importance_rows"),
        "torch_importance_sample_size": overall.get("torch_importance_sample_size"),
        "torch_epochs": overall.get("torch_epochs"),
        "torch_batch_size": overall.get("torch_batch_size"),
    }


def fit_static_teacher(
    selector: "ParetoSelector",
    train_df: pd.DataFrame | None = None,
    *,
    task_type: str | None = None,
    backend: str = "auto",
    random_state: int | None = None,
) -> dict[str, Any]:
    use_train_df = selector.train_df if train_df is None else train_df.reset_index(drop=True)
    use_task_type = _resolve_task_type(use_train_df[selector.target_column], task_type=task_type)
    use_random_state = selector.seed if random_state is None else int(random_state)
    preprocess = _build_feature_preprocessor(selector)
    estimator, backend_used = _build_static_estimator(
        task_type=use_task_type,
        backend=backend,
        random_state=use_random_state,
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )
    model.fit(use_train_df[selector.feature_columns], use_train_df[selector.target_column])

    manifest: dict[str, Any] = {
        "task_type": use_task_type,
        "backend_requested": str(backend),
        "backend_used": backend_used,
        "target_column": selector.target_column,
        "feature_columns": list(selector.feature_columns),
        "train_rows": int(len(use_train_df)),
    }
    if use_task_type == "classification":
        classes = [str(value) for value in model.named_steps["model"].classes_]
        manifest["classes"] = classes
    else:
        target_values = pd.to_numeric(use_train_df[selector.target_column], errors="coerce").to_numpy(dtype=float)
        manifest["target_std"] = float(np.std(target_values, ddof=0))

    return {
        "model": model,
        "manifest": manifest,
    }


def score_static_utility(
    selector: "ParetoSelector",
    candidate_df: pd.DataFrame,
    teacher_bundle: dict[str, Any],
    *,
    candidate_ids: Sequence[int] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if candidate_df.empty:
        return []

    manifest = dict(teacher_bundle.get("manifest", {}))
    model = teacher_bundle["model"]
    use_candidate_ids = (
        np.arange(len(candidate_df), dtype=int)
        if candidate_ids is None
        else np.asarray(candidate_ids, dtype=int)
    )
    if len(use_candidate_ids) != len(candidate_df):
        raise ValueError("candidate_ids length mismatch in score_static_utility")

    features = candidate_df[selector.feature_columns]
    task_type = str(manifest.get("task_type", "classification"))
    records: list[dict[str, Any]] = []

    if task_type == "classification":
        probs = np.asarray(model.predict_proba(features), dtype=float)
        class_labels = np.asarray([str(value) for value in model.named_steps["model"].classes_], dtype=object)
        targets = candidate_df[selector.target_column].astype(str).to_numpy(dtype=object)
        if probs.shape[1] <= 1:
            entropy_norm = np.zeros(len(candidate_df), dtype=float)
            p_true = np.ones(len(candidate_df), dtype=float)
        else:
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            entropy = -np.sum(probs * log_probs, axis=1)
            entropy_norm = entropy / max(math.log(probs.shape[1]), 1e-12)
            class_index = pd.Series(np.arange(len(class_labels), dtype=int), index=class_labels)
            target_indices = pd.Series(targets).map(class_index).fillna(-1).to_numpy(dtype=int)
            row_indices = np.arange(len(candidate_df), dtype=int)
            p_true = np.where(target_indices >= 0, probs[row_indices, np.maximum(target_indices, 0)], 0.0)
        utility = p_true * (0.5 + 0.5 * entropy_norm)
        for idx, candidate_id in enumerate(use_candidate_ids.tolist()):
            records.append(
                {
                    "candidate_id": int(candidate_id),
                    "u_static": float(utility[idx]),
                    "u_static_raw": float(utility[idx]),
                    "u_static_p_true": float(p_true[idx]),
                    "u_static_entropy_norm": float(entropy_norm[idx]),
                    "teacher_backend": str(manifest.get("backend_used", "unknown")),
                    "task_type": task_type,
                }
            )
        return records

    preds = np.asarray(model.predict(features), dtype=float)
    targets = pd.to_numeric(candidate_df[selector.target_column], errors="coerce").to_numpy(dtype=float)
    target_std = float(manifest.get("target_std", np.std(targets, ddof=0)))
    residual = np.abs(targets - preds)
    utility = np.exp(-residual / (target_std + 1e-12))
    for idx, candidate_id in enumerate(use_candidate_ids.tolist()):
        records.append(
            {
                "candidate_id": int(candidate_id),
                "u_static": float(utility[idx]),
                "u_static_raw": float(utility[idx]),
                "u_static_abs_residual": float(residual[idx]),
                "teacher_backend": str(manifest.get("backend_used", "unknown")),
                "task_type": task_type,
            }
        )
    return records


def build_static_balanced_utility_scores(
    selector: "ParetoSelector",
    preselected_records: list[dict[str, Any]],
    *,
    task_type: str | None = None,
    static_backend: str = "auto",
    random_state: int | None = None,
    density_clip: tuple[float, float] = (0.5, 2.0),
    show_progress: bool = False,
) -> dict[str, Any]:
    if not preselected_records:
        empty_manifest = {
            "available": False,
            "reason": "empty_preselected_records",
            "final_test_used": False,
        }
        return {
            "static_scores": [],
            "score_by_id": {},
            "manifest": empty_manifest,
            "teacher_manifest": {},
        }

    use_task_type = _resolve_task_type(selector.train_df[selector.target_column], task_type=task_type)
    use_random_state = selector.seed if random_state is None else int(random_state)
    balance_column = getattr(selector, "utility_balance_column", None)
    balance_group = ["target_label", f"bucket({balance_column})"] if balance_column else ["target_label", "gate_stratum"]
    preselected_df = _records_to_df(preselected_records, selector.column_order)
    preselected_ids = _candidate_ids(preselected_records)
    static_progress = _progress(
        total=4,
        desc="utility proxy static",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    try:
        if show_progress:
            _progress_write(
                f"utility proxy static: fit static teacher train_rows={len(selector.train_df)} "
                f"preselected_rows={len(preselected_records)}"
            )
        teacher_bundle = fit_static_teacher(
            selector,
            selector.train_df,
            task_type=use_task_type,
            backend=static_backend,
            random_state=use_random_state,
        )
        static_progress.update(1)
        if hasattr(static_progress, "set_postfix"):
            static_progress.set_postfix(step="fit_static_teacher")

        if show_progress:
            _progress_write(f"utility proxy static: score rows={len(preselected_df)}")
        static_scores = score_static_utility(
            selector,
            preselected_df,
            teacher_bundle,
            candidate_ids=preselected_ids,
        )
        static_progress.update(1)
        if hasattr(static_progress, "set_postfix"):
            static_progress.set_postfix(step="score_static")

        static_df = pd.DataFrame(static_scores)
        raw_static = (
            static_df["u_static"].to_numpy(dtype=float)
            if "u_static" in static_df.columns
            else np.zeros(len(preselected_records), dtype=float)
        )
        if show_progress:
            _progress_write(f"utility proxy static: balance rows={len(preselected_df)}")
        balanced_components = _balanced_static_components(
            selector,
            preselected_df,
            raw_static,
            task_type=use_task_type,
            density_clip=density_clip,
            balance_column=balance_column,
        )
        static_progress.update(1)
        if hasattr(static_progress, "set_postfix"):
            static_progress.set_postfix(step="balance_static")

        if show_progress:
            _progress_write(f"utility proxy static: build output rows={len(static_scores)}")
        output_scores: list[dict[str, Any]] = []
        score_by_id: dict[int, float] = {}
        for idx, base_row in enumerate(
            _progress(
                static_scores,
                total=len(static_scores),
                desc="utility proxy static rows",
                dynamic_ncols=True,
                disable=not show_progress,
            )
        ):
            candidate_id = int(base_row["candidate_id"])
            u_static_balanced = float(balanced_components["u_static_balanced"][idx])
            score_by_id[candidate_id] = u_static_balanced
            output_scores.append(
                {
                    "candidate_id": candidate_id,
                    "u_static": float(base_row.get("u_static", 0.0)),
                    "u_static_raw": float(base_row.get("u_static_raw", base_row.get("u_static", 0.0))),
                    "target_label": str(balanced_components["target_labels"][idx]),
                    "gate_stratum": int(balanced_components["gate_strata"][idx]),
                    "balance_bucket": str(balanced_components["balance_buckets"][idx]),
                    "u_static_group_rank": float(balanced_components["static_rank"][idx]),
                    "density_weight": float(balanced_components["density_weight"][idx]),
                    "coverage_gain": float(balanced_components["coverage_gain"][idx]),
                    "u_static_balanced": u_static_balanced,
                    "task_type": use_task_type,
                }
            )
        static_progress.update(1)
        if hasattr(static_progress, "set_postfix"):
            static_progress.set_postfix(step="build_output")
    finally:
        static_progress.close()

    manifest = {
        "available": True,
        "reason": None,
        "task_type": use_task_type,
        "teacher_train_rows": int(len(selector.train_df)),
        "preselected_rows": int(len(preselected_records)),
        "final_test_used": False,
        "static_backend": teacher_bundle.get("manifest", {}).get("backend_used"),
        "static_backend_requested": teacher_bundle.get("manifest", {}).get("backend_requested"),
        "static_formula": "p_true * (0.5 + 0.5 * entropy_norm)",
        "static_normalization": "rank_normalize_within_" + "_and_".join(balance_group),
        "utility_balance_column": balance_column,
        "balance_group": balance_group,
        "density_weight": {
            "formula": "target_group_mass / pool_group_mass",
            "clip_min": float(density_clip[0]),
            "clip_max": float(density_clip[1]),
            "target_distribution": "train_df",
            "pool_distribution": "preselected_records",
            "group": balance_group,
        },
        "coverage_gain": {
            "formula": "max((target_group_mass - pool_group_mass) / target_group_mass, 0)",
            "clip_min": 0.0,
            "clip_max": 1.0,
            "group": balance_group,
        },
    }
    return {
        "static_scores": output_scores,
        "score_by_id": score_by_id,
        "manifest": manifest,
        "teacher_manifest": teacher_bundle.get("manifest", {}),
    }


def parse_utility_source_prior(value: str | None) -> dict[str, float]:
    if value is None or not str(value).strip():
        return {}
    weights: dict[str, float] = {}
    for raw_item in str(value).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid utility source prior item {item!r}; expected source:weight")
        source_id, raw_weight = item.split(":", 1)
        source_id = source_id.strip().lower()
        if not source_id:
            raise ValueError(f"Invalid utility source prior item {item!r}; source is empty")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"Invalid utility source prior weight for {source_id!r}: {raw_weight!r}") from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Invalid utility source prior weight for {source_id!r}: {weight!r}")
        weights[source_id] = weight
    return weights


def apply_utility_source_prior_to_proxy_scores(
    proxy_scores: list[dict[str, object]],
    *,
    source_by_id: dict[int, str],
    prior: str | None,
    default_weight: float = 1.0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    weights = parse_utility_source_prior(prior)
    if not weights:
        return proxy_scores, {"enabled": False, "reason": "no_utility_source_prior"}
    if not math.isfinite(float(default_weight)) or float(default_weight) < 0.0:
        raise ValueError(f"Invalid utility source prior default weight: {default_weight!r}")

    source_counts: dict[str, int] = {}
    matched_rows = 0
    missing_source_rows = 0
    adjusted_scores: list[dict[str, object]] = []
    for row_idx, record in enumerate(proxy_scores):
        adjusted = dict(record)
        try:
            candidate_id = int(adjusted.get("candidate_id", row_idx))
        except (TypeError, ValueError):
            candidate_id = row_idx
        source_id = source_by_id.get(candidate_id)
        if source_id:
            source_id = str(source_id).strip().lower()
            matched_rows += 1
        else:
            source_id = None
            missing_source_rows += 1
        source_key = source_id if source_id else "__missing__"
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        weight = float(weights.get(source_key, default_weight))
        for field in ("u_static_balanced", "u_static_norm", "u_proxy"):
            before = float(adjusted.get(field, 0.0))
            adjusted[f"{field}_before_source_prior"] = before
            adjusted[field] = float(np.clip(before * weight, 0.0, 1.0))
        adjusted["source_id"] = source_id
        adjusted["utility_source_prior_weight"] = weight
        adjusted_scores.append(adjusted)

    return adjusted_scores, {
        "enabled": True,
        "formula": "clip(proxy_score * source_weight, 0, 1)",
        "weights": weights,
        "default_weight": float(default_weight),
        "matched_rows": int(matched_rows),
        "missing_source_rows": int(missing_source_rows),
        "source_counts": source_counts,
    }


def attach_utility_proxy_fields(
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
            merged["utility_proxy_total"] = 0.0
            merged["utility_proxy_static_norm"] = 0.0
            merged["utility_proxy_static_raw"] = 0.0
            merged["utility_proxy_static_group_rank"] = 0.0
            merged["utility_proxy_density_weight"] = 0.0
            merged["utility_proxy_coverage_gain"] = 0.0
            merged["utility_proxy_gate_stratum"] = -1
            merged["utility_proxy_balance_bucket"] = None
            merged["utility_proxy_target_label"] = None
            merged["utility_anchor_member"] = False
        else:
            matched_rows += 1
            merged["pareto_util_proxy_obj"] = float(proxy.get("u_proxy", 0.0))
            merged["utility_proxy_static"] = float(proxy.get("u_static", 0.0))
            merged["utility_proxy_total"] = float(proxy.get("u_proxy", 0.0))
            merged["utility_proxy_static_norm"] = float(proxy.get("u_static_norm", 0.0))
            merged["utility_proxy_static_raw"] = float(proxy.get("u_static_raw", proxy.get("u_static", 0.0)))
            merged["utility_proxy_static_group_rank"] = float(proxy.get("u_static_group_rank", 0.0))
            merged["utility_proxy_density_weight"] = float(proxy.get("density_weight", 0.0))
            merged["utility_proxy_coverage_gain"] = float(proxy.get("coverage_gain", 0.0))
            merged["utility_proxy_gate_stratum"] = int(proxy.get("gate_stratum", -1))
            merged["utility_proxy_balance_bucket"] = proxy.get("balance_bucket")
            merged["utility_proxy_target_label"] = proxy.get("target_label")
            merged["utility_anchor_member"] = bool(proxy.get("is_anchor_member", False))
            merged["source_id"] = proxy.get("source_id")
            merged["utility_source_prior_weight"] = float(proxy.get("utility_source_prior_weight", 1.0))
        merged_records.append(merged)
    return merged_records, {
        "matched_rows": matched_rows,
        "missing_rows": missing_rows,
        "proxy_rows": len(proxy_scores),
        "exact_rows": len(exact_records),
    }
