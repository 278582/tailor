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

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

if TYPE_CHECKING:  # pragma: no cover
    from .pareto import ParetoSelector


ROOT_DIR = Path(__file__).resolve().parents[1]
_TABDIFF_MLE_MODULE: Any | None = None


def _progress(iterable: Any | None = None, **kwargs: Any) -> Any:
    if _tqdm is None:
        return iterable if iterable is not None else _NullProgress()
    if iterable is None:
        return _tqdm(**kwargs)
    return _tqdm(iterable, **kwargs)


def _progress_write(message: str) -> None:
    if _tqdm is None:
        print(message)
    else:
        _tqdm.write(message)


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
    return selector._assign_bins_from_edges(gate_probs, selector.train_gate_edges).astype(int, copy=False)


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


def _train_balance_target_mass(selector: "ParetoSelector", task_type: str) -> pd.DataFrame:
    cache = getattr(selector, "_utility_balance_target_mass_cache", None)
    if cache is None:
        cache = {}
        setattr(selector, "_utility_balance_target_mass_cache", cache)
    cache_key = str(task_type)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    train_gate_strata = _gate_strata_for_df(selector, selector.train_df)
    train_target_labels = _target_group_labels(
        selector.train_df[selector.target_column],
        selector.train_df[selector.target_column],
        task_type=task_type,
    )
    target_groups = pd.DataFrame(
        {
            "target_label": train_target_labels.astype(str),
            "gate_stratum": train_gate_strata.astype(int, copy=False),
        }
    )
    target_mass = (
        target_groups.value_counts(["target_label", "gate_stratum"], normalize=True)
        .rename("target_mass")
        .reset_index()
    )
    cache[cache_key] = target_mass
    return target_mass


def _balanced_static_components(
    selector: "ParetoSelector",
    candidate_df: pd.DataFrame,
    raw_static: Sequence[float] | np.ndarray,
    *,
    task_type: str,
    density_clip: tuple[float, float] = (0.5, 2.0),
) -> dict[str, Any]:
    raw_array = np.asarray(raw_static, dtype=float)
    if candidate_df.empty or raw_array.size == 0:
        return {
            "target_labels": np.zeros(0, dtype=object),
            "gate_strata": np.zeros(0, dtype=int),
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
    pool_groups = pd.DataFrame(
        {
            "target_label": target_labels.astype(str),
            "gate_stratum": gate_strata.astype(int, copy=False),
        }
    )
    group_codes = pd.factorize(pd.MultiIndex.from_frame(pool_groups), sort=False)[0]
    static_rank = _grouped_rank_normalize(raw_array, group_codes)

    group_cols = ["target_label", "gate_stratum"]
    pool_mass = pool_groups.value_counts(group_cols, normalize=True).rename("pool_mass").reset_index()
    target_mass = _train_balance_target_mass(selector, task_type)
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
) -> dict[str, Any]:
    if syn_df.empty or test_df.empty or selector.target_column not in syn_df.columns:
        return {"available": False, "reason": "empty_input"}

    use_random_state = selector.seed if random_state is None else int(random_state)
    try:
        overall = _evaluate_tabdiff_exact_utility(
            selector,
            syn_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            task_type=task_type,
            random_state=use_random_state,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": "tabdiff_mle_failed",
            "error": str(exc),
        }

    gate_probs = selector._prob_geomean_for_df(test_df.reset_index(drop=True), columns=selector.feature_columns)
    if gate_probs.size == 0:
        return {
            "available": True,
            "reason": None,
            "protocol": "tabdiff_mle",
            "task_type": overall.get("task_type"),
            "tabdiff_task_type": overall.get("tabdiff_task_type"),
            "metric": overall.get("metric"),
            "overall": float(overall.get("overall", 0.0)),
            "tail": None,
            "middle": None,
            "mode": None,
            "rows": {"tail": 0, "middle": 0, "mode": 0},
            "region_metrics_available": False,
            "region_metrics_reason": "disabled_for_tabdiff_mle_cost",
            "dataset_name": overall.get("dataset_name"),
            "info_source": overall.get("info_source"),
            "info_path": overall.get("info_path"),
            "val_source": overall.get("val_source"),
            "val_path": overall.get("val_path"),
            "search_holdout_used": bool(search_holdout_used or overall.get("search_holdout_used", False)),
            "runtime_tree_method": overall.get("runtime_tree_method"),
            "runtime_note": overall.get("runtime_note"),
            "primary_score_group": overall.get("primary_score_group"),
            "primary_model": overall.get("primary_model"),
            "overall_scores": overall.get("overall_scores"),
        }

    quantiles = np.quantile(gate_probs, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    quantiles = np.unique(quantiles)
    if len(quantiles) < 4:
        quantiles = np.array([gate_probs.min() - 1e-12, gate_probs.mean(), gate_probs.mean(), gate_probs.max() + 1e-12])
    bins = np.digitize(gate_probs, quantiles[1:-1], right=False)

    return {
        "available": True,
        "reason": None,
        "protocol": "tabdiff_mle",
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
        "region_metrics_reason": "disabled_for_tabdiff_mle_cost",
        "dataset_name": overall.get("dataset_name"),
        "info_source": overall.get("info_source"),
        "info_path": overall.get("info_path"),
        "val_source": overall.get("val_source"),
        "val_path": overall.get("val_path"),
        "search_holdout_used": bool(search_holdout_used or overall.get("search_holdout_used", False)),
        "runtime_tree_method": overall.get("runtime_tree_method"),
        "runtime_note": overall.get("runtime_note"),
        "primary_score_group": overall.get("primary_score_group"),
        "primary_model": overall.get("primary_model"),
        "overall_scores": overall.get("overall_scores"),
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
    preselected_df = _records_to_df(preselected_records, selector.column_order)
    preselected_ids = _candidate_ids(preselected_records)

    teacher_bundle = fit_static_teacher(
        selector,
        selector.train_df,
        task_type=use_task_type,
        backend=static_backend,
        random_state=use_random_state,
    )
    static_scores = score_static_utility(
        selector,
        preselected_df,
        teacher_bundle,
        candidate_ids=preselected_ids,
    )
    static_df = pd.DataFrame(static_scores)
    raw_static = (
        static_df["u_static"].to_numpy(dtype=float)
        if "u_static" in static_df.columns
        else np.zeros(len(preselected_records), dtype=float)
    )
    balanced_components = _balanced_static_components(
        selector,
        preselected_df,
        raw_static,
        task_type=use_task_type,
        density_clip=density_clip,
    )

    output_scores: list[dict[str, Any]] = []
    score_by_id: dict[int, float] = {}
    for idx, base_row in enumerate(static_scores):
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
                "u_static_group_rank": float(balanced_components["static_rank"][idx]),
                "density_weight": float(balanced_components["density_weight"][idx]),
                "coverage_gain": float(balanced_components["coverage_gain"][idx]),
                "u_static_balanced": u_static_balanced,
                "task_type": use_task_type,
            }
        )

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
        "static_normalization": "rank_normalize_within_target_label_and_gate_stratum",
        "density_weight": {
            "formula": "target_group_mass / pool_group_mass",
            "clip_min": float(density_clip[0]),
            "clip_max": float(density_clip[1]),
            "target_distribution": "train_df",
            "pool_distribution": "preselected_records",
            "group": ["target_label", "gate_stratum"],
        },
        "coverage_gain": {
            "formula": "max((target_group_mass - pool_group_mass) / target_group_mass, 0)",
            "clip_min": 0.0,
            "clip_max": 1.0,
            "group": ["target_label", "gate_stratum"],
        },
    }
    return {
        "static_scores": output_scores,
        "score_by_id": score_by_id,
        "manifest": manifest,
        "teacher_manifest": teacher_bundle.get("manifest", {}),
    }


def estimate_dynamic_utility_blocks(
    selector: "ParetoSelector",
    anchor_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    holdout_df: pd.DataFrame | None = None,
    task_type: str | None = None,
    num_rounds: int = 8,
    block_size: int | None = None,
    random_state: int | None = None,
    show_progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    use_holdout_df = selector.holdout_df if holdout_df is None else holdout_df.reset_index(drop=True)
    if not anchor_records:
        return [], {"available": False, "reason": "empty_anchor_records"}
    use_task_type = _resolve_task_type(selector.train_df[selector.target_column], task_type=task_type)
    anchor_df = _records_to_df(anchor_records, selector.column_order)
    reference_score = _score_subset_utility(
        selector,
        anchor_df,
        use_holdout_df,
        task_type=use_task_type,
        random_state=selector.seed if random_state is None else int(random_state),
    )
    if not candidate_records:
        return [], {
            "available": True,
            "reason": "empty_dynamic_candidate_set",
            "reference_rows": int(len(anchor_records)),
            "dynamic_candidate_rows": 0,
            "reference_utility": float(reference_score.get("utility", 0.0)),
            "reference_metric": reference_score.get("metric"),
            "rounds": int(max(num_rounds, 0)),
            "block_size": int(block_size or 0),
            "round_summaries": [],
        }

    candidate_df = _records_to_df(candidate_records, selector.column_order)
    candidate_ids = _candidate_ids(candidate_records)
    use_random_state = selector.seed if random_state is None else int(random_state)
    keep_k = max(len(anchor_records), 1)
    use_block_size = (
        max(128, min(512, keep_k // 32))
        if block_size is None
        else max(1, int(block_size))
    )
    use_block_size = max(1, int(use_block_size))
    use_num_rounds = max(1, int(num_rounds))
    rng = np.random.default_rng(use_random_state)

    reference_utility = float(reference_score.get("utility", 0.0))
    block_results: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    blocks_per_round = int(math.ceil(len(candidate_records) / max(float(use_block_size), 1.0)))
    total_blocks = int(use_num_rounds * blocks_per_round)
    block_progress = _progress(
        total=total_blocks,
        desc="dynamic utility proxy",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    try:
        for round_id in range(use_num_rounds):
            shuffled = rng.permutation(len(candidate_records))
            round_deltas: list[float] = []
            block_counter = 0
            for start in range(0, len(shuffled), use_block_size):
                block_indices = np.asarray(shuffled[start : start + use_block_size], dtype=int)
                if block_indices.size == 0:
                    continue
                block_df = candidate_df.iloc[block_indices].reset_index(drop=True)
                training_df = pd.concat([anchor_df, block_df], axis=0, ignore_index=True)
                utility_score = _score_subset_utility(
                    selector,
                    training_df,
                    use_holdout_df,
                    task_type=use_task_type,
                    random_state=use_random_state + round_id + block_counter + 1,
                )
                utility_value = float(utility_score.get("utility", 0.0))
                delta_utility = float(utility_value - reference_utility)
                round_deltas.append(delta_utility)
                block_results.append(
                    {
                        "round_id": int(round_id),
                        "block_id": int(block_counter),
                        "candidate_ids": candidate_ids[block_indices].astype(int, copy=False),
                        "block_rows": int(block_indices.size),
                        "utility_value": utility_value,
                        "delta_u": delta_utility,
                        "metric": utility_score.get("metric"),
                        "available": bool(utility_score.get("available", False)),
                    }
                )
                block_counter += 1
                block_progress.update(1)
                if hasattr(block_progress, "set_postfix"):
                    block_progress.set_postfix(
                        round=f"{round_id + 1}/{use_num_rounds}",
                        block=block_counter,
                        delta=f"{delta_utility:.4g}",
                    )
            round_summaries.append(
                {
                    "round_id": int(round_id),
                    "num_blocks": int(block_counter),
                    "delta_u_mean": float(np.mean(round_deltas)) if round_deltas else 0.0,
                    "delta_u_std": float(np.std(round_deltas, ddof=0)) if round_deltas else 0.0,
                }
            )
    finally:
        block_progress.close()

    report = {
        "available": True,
        "reason": None,
        "task_type": use_task_type,
        "reference_rows": int(len(anchor_records)),
        "dynamic_candidate_rows": int(len(candidate_records)),
        "reference_utility": reference_utility,
        "reference_metric": reference_score.get("metric"),
        "rounds": int(use_num_rounds),
        "block_size": int(use_block_size),
        "round_summaries": round_summaries,
    }
    return block_results, report


def aggregate_dynamic_row_scores(
    candidate_ids: Sequence[int] | np.ndarray,
    block_results: list[dict[str, Any]],
    *,
    lambda_var: float = 0.5,
    anchor_candidate_ids: Sequence[int] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    use_candidate_ids = np.asarray(candidate_ids, dtype=int)
    anchor_ids = (
        np.zeros(0, dtype=int)
        if anchor_candidate_ids is None
        else np.asarray(anchor_candidate_ids, dtype=int)
    )
    id_to_pos = {int(candidate_id): pos for pos, candidate_id in enumerate(use_candidate_ids.tolist())}
    delta_sum = np.zeros(use_candidate_ids.size, dtype=float)
    delta_sum_sq = np.zeros(use_candidate_ids.size, dtype=float)
    delta_count = np.zeros(use_candidate_ids.size, dtype=int)
    for block in block_results:
        delta_value = float(block.get("delta_u", 0.0))
        block_ids = np.asarray(block.get("candidate_ids", []), dtype=int)
        if block_ids.size == 0:
            continue
        positions = np.fromiter(
            (id_to_pos[int(candidate_id)] for candidate_id in block_ids if int(candidate_id) in id_to_pos),
            dtype=int,
        )
        if positions.size == 0:
            continue
        np.add.at(delta_sum, positions, delta_value)
        np.add.at(delta_sum_sq, positions, delta_value * delta_value)
        np.add.at(delta_count, positions, 1)

    records: list[dict[str, Any]] = []
    anchor_id_set = set(anchor_ids.tolist())
    mean_values = np.divide(
        delta_sum,
        np.maximum(delta_count, 1),
        out=np.zeros_like(delta_sum, dtype=float),
        where=delta_count > 0,
    )
    variance = np.divide(
        delta_sum_sq,
        np.maximum(delta_count, 1),
        out=np.zeros_like(delta_sum_sq, dtype=float),
        where=delta_count > 0,
    ) - mean_values * mean_values
    std_values = np.sqrt(np.maximum(variance, 0.0))
    dynamic_values = mean_values - float(lambda_var) * std_values

    for pos, candidate_id in enumerate(use_candidate_ids.tolist()):
        if int(candidate_id) in anchor_id_set:
            mean_value = 0.0
            std_value = 0.0
            dynamic_value = 0.0
            num_blocks = int(delta_count[pos])
        elif int(delta_count[pos]) == 0:
            mean_value = 0.0
            std_value = 0.0
            dynamic_value = 0.0
            num_blocks = 0
        else:
            mean_value = float(mean_values[pos])
            std_value = float(std_values[pos])
            dynamic_value = float(dynamic_values[pos])
            num_blocks = int(delta_count[pos])
        records.append(
            {
                "candidate_id": int(candidate_id),
                "u_dynamic": float(dynamic_value),
                "u_dynamic_mean_delta": float(mean_value),
                "u_dynamic_std_delta": float(std_value),
                "u_dynamic_num_blocks": int(num_blocks),
            }
        )
    return records


def build_utility_proxy_scores(
    selector: "ParetoSelector",
    preselected_records: list[dict[str, Any]],
    anchor_records: list[dict[str, Any]],
    *,
    task_type: str | None = None,
    static_backend: str = "auto",
    num_rounds: int = 8,
    block_size: int | None = None,
    lambda_var: float = 0.5,
    random_state: int | None = None,
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
            "dynamic_scores": [],
            "proxy_scores": [],
            "dynamic_blocks": {"available": False, "reason": "empty_preselected_records"},
            "manifest": empty_manifest,
        }

    use_task_type = _resolve_task_type(selector.train_df[selector.target_column], task_type=task_type)
    use_random_state = selector.seed if random_state is None else int(random_state)
    preselected_df = _records_to_df(preselected_records, selector.column_order)
    preselected_ids = _candidate_ids(preselected_records)
    anchor_ids = _candidate_ids(anchor_records) if anchor_records else np.zeros(0, dtype=int)
    anchor_id_set = set(anchor_ids.tolist())
    dynamic_records = [
        record for record in preselected_records if int(record.get("candidate_id", -1)) not in anchor_id_set
    ]
    proxy_progress = _progress(
        total=5,
        desc="utility proxy",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    try:
        if show_progress:
            _progress_write(
                f"utility proxy: fit static teacher train_rows={len(selector.train_df)} "
                f"preselected_rows={len(preselected_records)}"
            )
        teacher_bundle = fit_static_teacher(
            selector,
            selector.train_df,
            task_type=use_task_type,
            backend=static_backend,
            random_state=use_random_state,
        )
        proxy_progress.update(1)
        if hasattr(proxy_progress, "set_postfix"):
            proxy_progress.set_postfix(step="fit_static_teacher")

        if show_progress:
            _progress_write(f"utility proxy: score static utility rows={len(preselected_df)}")
        static_scores = score_static_utility(
            selector,
            preselected_df,
            teacher_bundle,
            candidate_ids=preselected_ids,
        )
        proxy_progress.update(1)
        if hasattr(proxy_progress, "set_postfix"):
            proxy_progress.set_postfix(step="score_static")

        if show_progress:
            _progress_write(
                f"utility proxy: estimate dynamic blocks anchor_rows={len(anchor_records)} "
                f"dynamic_rows={len(dynamic_records)} rounds={num_rounds}"
            )
        block_results, dynamic_report = estimate_dynamic_utility_blocks(
            selector,
            anchor_records=anchor_records,
            candidate_records=dynamic_records,
            holdout_df=selector.holdout_df,
            task_type=use_task_type,
            num_rounds=num_rounds,
            block_size=block_size,
            random_state=use_random_state,
            show_progress=show_progress,
        )
        proxy_progress.update(1)
        if hasattr(proxy_progress, "set_postfix"):
            proxy_progress.set_postfix(step="dynamic_blocks")

        if show_progress:
            _progress_write(f"utility proxy: aggregate dynamic scores blocks={len(block_results)}")
        dynamic_scores = aggregate_dynamic_row_scores(
            preselected_ids,
            block_results,
            lambda_var=lambda_var,
            anchor_candidate_ids=anchor_ids,
        )
        proxy_progress.update(1)
        if hasattr(proxy_progress, "set_postfix"):
            proxy_progress.set_postfix(step="aggregate")

        static_df = pd.DataFrame(static_scores)
        dynamic_df = pd.DataFrame(dynamic_scores)
        merged = pd.DataFrame({"candidate_id": preselected_ids})
        if not static_df.empty:
            merged = merged.merge(static_df, on="candidate_id", how="left")
        if not dynamic_df.empty:
            merged = merged.merge(dynamic_df, on="candidate_id", how="left")
        merged["u_static"] = merged.get("u_static", pd.Series(np.zeros(len(merged), dtype=float))).fillna(0.0)
        merged["u_dynamic"] = merged.get("u_dynamic", pd.Series(np.zeros(len(merged), dtype=float))).fillna(0.0)
        merged["u_dynamic_mean_delta"] = merged.get(
            "u_dynamic_mean_delta", pd.Series(np.zeros(len(merged), dtype=float))
        ).fillna(0.0)
        merged["u_dynamic_std_delta"] = merged.get(
            "u_dynamic_std_delta", pd.Series(np.zeros(len(merged), dtype=float))
        ).fillna(0.0)
        merged["u_dynamic_num_blocks"] = merged.get(
            "u_dynamic_num_blocks", pd.Series(np.zeros(len(merged), dtype=int))
        ).fillna(0).astype(int)
        merged["is_anchor_member"] = merged["candidate_id"].isin(anchor_id_set)

        if show_progress:
            _progress_write(f"utility proxy: merge and normalize rows={len(merged)}")
        balanced_components = _balanced_static_components(
            selector,
            preselected_df,
            merged["u_static"].to_numpy(dtype=float),
            task_type=use_task_type,
            density_clip=(0.5, 2.0),
        )
        merged["utility_target_label"] = balanced_components["target_labels"]
        merged["utility_gate_stratum"] = balanced_components["gate_strata"]
        merged["u_static_group_rank"] = balanced_components["static_rank"]
        merged["utility_density_weight"] = balanced_components["density_weight"]
        merged["utility_coverage_gain"] = balanced_components["coverage_gain"]
        merged["u_static_balanced"] = balanced_components["u_static_balanced"]
        merged["u_static_norm"] = merged["u_static_balanced"]
        merged["u_dynamic_norm"] = _minmax_normalize(merged["u_dynamic"].to_numpy(dtype=float))
        merged.loc[merged["is_anchor_member"], "u_dynamic_norm"] = 0.0
        merged["u_proxy"] = (
            0.60 * merged["u_static_balanced"]
            + 0.30 * merged["u_dynamic_norm"]
            + 0.10 * merged["utility_coverage_gain"]
        )
        proxy_progress.update(1)
        if hasattr(proxy_progress, "set_postfix"):
            proxy_progress.set_postfix(step="merge_normalize")
    finally:
        proxy_progress.close()

    proxy_scores = []
    for row in merged.to_dict(orient="records"):
        proxy_scores.append(
            {
                "candidate_id": int(row["candidate_id"]),
                "u_static": float(row["u_static"]),
                "u_static_raw": float(row.get("u_static_raw", row["u_static"])),
                "u_dynamic": float(row["u_dynamic"]),
                "target_label": str(row["utility_target_label"]),
                "gate_stratum": int(row["utility_gate_stratum"]),
                "u_static_group_rank": float(row["u_static_group_rank"]),
                "density_weight": float(row["utility_density_weight"]),
                "coverage_gain": float(row["utility_coverage_gain"]),
                "u_static_balanced": float(row["u_static_balanced"]),
                "u_static_norm": float(row["u_static_norm"]),
                "u_dynamic_norm": float(row["u_dynamic_norm"]),
                "u_proxy": float(row["u_proxy"]),
                "u_dynamic_mean_delta": float(row["u_dynamic_mean_delta"]),
                "u_dynamic_std_delta": float(row["u_dynamic_std_delta"]),
                "u_dynamic_num_blocks": int(row["u_dynamic_num_blocks"]),
                "is_anchor_member": bool(row["is_anchor_member"]),
                "task_type": use_task_type,
            }
        )

    manifest = {
        "available": True,
        "reason": None,
        "task_type": use_task_type,
        "teacher_train_rows": int(len(selector.train_df)),
        "search_holdout_rows": int(len(selector.holdout_df)),
        "final_test_used": False,
        "preselected_rows": int(len(preselected_records)),
        "anchor_rows": int(len(anchor_records)),
        "dynamic_candidate_rows": int(len(dynamic_records)),
        "static_backend": teacher_bundle.get("manifest", {}).get("backend_used"),
        "static_backend_requested": teacher_bundle.get("manifest", {}).get("backend_requested"),
        "num_rounds": int(num_rounds),
        "block_size": int(dynamic_report.get("block_size", block_size or 0)),
        "lambda_var": float(lambda_var),
        "reference_name": "preselected_fidelity_ceiling_keep_k",
        "static_formula": "p_true * (0.5 + 0.5 * entropy_norm)",
        "static_normalization": "rank_normalize_within_target_label_and_gate_stratum",
        "density_weight": {
            "formula": "target_group_mass / pool_group_mass",
            "clip_min": 0.5,
            "clip_max": 2.0,
            "target_distribution": "train_df",
            "pool_distribution": "preselected_records",
            "group": ["target_label", "gate_stratum"],
        },
        "coverage_gain": {
            "formula": "max((target_group_mass - pool_group_mass) / target_group_mass, 0)",
            "clip_min": 0.0,
            "clip_max": 1.0,
            "group": ["target_label", "gate_stratum"],
        },
        "proxy_formula": "0.60 * u_static_balanced + 0.30 * u_dynamic_norm + 0.10 * coverage_gain",
        "proxy_weights": {
            "u_static_balanced": 0.60,
            "u_dynamic_norm": 0.30,
            "coverage_gain": 0.10,
        },
    }

    return {
        "static_scores": static_scores,
        "dynamic_scores": dynamic_scores,
        "proxy_scores": proxy_scores,
        "dynamic_blocks": dynamic_report,
        "manifest": manifest,
        "teacher_manifest": teacher_bundle.get("manifest", {}),
    }
