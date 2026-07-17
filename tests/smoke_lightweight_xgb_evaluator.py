from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


FEATURE_COLUMNS = ["age", "education.num", "hours.per.week", "workclass", "marital.status"]
NUMERIC_COLUMNS = ["age", "education.num", "hours.per.week"]
CATEGORICAL_COLUMNS = ["workclass", "marital.status"]
TARGET_COLUMN = "income"

REQUIRED_UTILITY_EXACT_FIELDS = [
    "available",
    "protocol",
    "task_type",
    "tabdiff_task_type",
    "metric",
    "overall",
    "tail",
    "middle",
    "mode",
    "rows",
    "region_metrics_available",
    "region_metrics_reason",
    "primary_score_group",
    "primary_model",
    "runtime_tree_method",
    "runtime_device",
    "overall_scores",
]


def _make_adult_like_frame(n_rows: int, *, seed: int, include_unknown_workclass: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    workclasses = np.array(["Private", "Self-emp-not-inc", "Federal-gov"], dtype=object)
    marital_statuses = np.array(["Never-married", "Married-civ-spouse", "Divorced"], dtype=object)

    age = rng.integers(18, 70, size=n_rows)
    education = rng.integers(1, 17, size=n_rows)
    hours = rng.integers(20, 65, size=n_rows)
    workclass = rng.choice(workclasses, size=n_rows, p=[0.72, 0.18, 0.10])
    marital = rng.choice(marital_statuses, size=n_rows, p=[0.36, 0.46, 0.18])

    if include_unknown_workclass and n_rows >= 4:
        workclass[:4] = "Local-gov"

    signal = (
        0.035 * (age - 35)
        + 0.22 * (education - 9)
        + 0.035 * (hours - 40)
        + np.where(marital == "Married-civ-spouse", 1.0, -0.2)
        + np.where(workclass == "Self-emp-not-inc", 0.25, 0.0)
        + rng.normal(0.0, 0.8, size=n_rows)
    )
    income = np.where(signal > 0.75, ">50K", "<=50K")
    return pd.DataFrame(
        {
            "age": age,
            "education.num": education,
            "hours.per.week": hours,
            "workclass": workclass,
            "marital.status": marital,
            "income": income,
        }
    )


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _xgb_classifier_kwargs(*, seed: int, device: str, class_count: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "n_estimators": 24,
        "max_depth": 3,
        "learning_rate": 0.12,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic" if class_count == 2 else "multi:softprob",
        "eval_metric": "logloss" if class_count == 2 else "mlogloss",
        "random_state": int(seed),
        "verbosity": 0,
        "n_jobs": 1,
    }
    if class_count > 2:
        kwargs["num_class"] = int(class_count)
    if device == "cuda":
        kwargs["tree_method"] = "hist"
        kwargs["device"] = "cuda"
    else:
        kwargs["tree_method"] = "hist"
    return kwargs


def _encoded_feature_to_original(encoded_feature: str, categorical_columns: list[str]) -> str:
    if encoded_feature.startswith("num__"):
        return encoded_feature.removeprefix("num__")
    if encoded_feature.startswith("cat__"):
        raw = encoded_feature.removeprefix("cat__")
        for column in categorical_columns:
            if raw.startswith(f"{column}_"):
                return column
    return encoded_feature.split("_", 1)[0]


def _aggregate_importance(
    feature_names: np.ndarray,
    importances: np.ndarray,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> list[dict[str, Any]]:
    grouped = {feature: 0.0 for feature in feature_columns}
    for encoded_feature, importance in zip(feature_names, importances):
        original = _encoded_feature_to_original(str(encoded_feature), categorical_columns)
        grouped[original] = grouped.get(original, 0.0) + float(importance)
    total = float(sum(grouped.values()))
    if total <= 0.0:
        raise RuntimeError("XGBoost returned empty or zero feature importances.")
    rows = [
        {"feature": feature, "importance": float(value / total)}
        for feature, value in grouped.items()
    ]
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return [
        {"feature": item["feature"], "importance": round(float(item["importance"]), 6), "rank": rank}
        for rank, item in enumerate(rows, start=1)
    ]


def _infer_target_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested target column {requested!r} is not present in CSV columns.")
        return requested
    for candidate in ("income", "target", "label", "class", "y"):
        if candidate in df.columns:
            return candidate
    return str(df.columns[-1])


def _infer_feature_types(df: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in feature_columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        non_null = int(series.notna().sum())
        numeric_ratio = float(coerced.notna().sum() / max(non_null, 1))
        if numeric_ratio >= 0.98:
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns


def _prepare_frame_for_preprocess(
    df: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    output = df.copy()
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    for column in categorical_columns:
        output[column] = output[column].astype("object").where(output[column].notna(), "__missing__").astype(str)
    return output


def _find_json_values(payload: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, dict):
        for item_key, item_value in payload.items():
            if item_key == key:
                values.append(item_value)
            values.extend(_find_json_values(item_value, key))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_find_json_values(item, key))
    return values


def _booster_runtime_info(model: Any) -> dict[str, Any]:
    booster = model.get_booster()
    config_text = booster.save_config()
    try:
        config = json.loads(config_text)
    except Exception:
        return {"raw_config": config_text[:2000], "device_values": []}
    device_values = [str(value) for value in _find_json_values(config, "device")]
    tree_methods = [str(value) for value in _find_json_values(config, "tree_method")]
    updaters = [str(value) for value in _find_json_values(config, "updater")]
    return {
        "device_values": sorted(set(device_values)),
        "tree_method_values": sorted(set(tree_methods)),
        "updater_values": sorted(set(updaters)),
    }


def _metric_from_predictions(model: Any, x_test: np.ndarray, y_test: np.ndarray, class_count: int) -> tuple[str, float]:
    if class_count == 2:
        proba = model.predict_proba(x_test)[:, 1]
        return "roc_auc", float(roc_auc_score(y_test, proba))
    pred = model.predict(x_test)
    return "accuracy", float(accuracy_score(y_test, pred))


def _build_report(
    *,
    device: str,
    metric_name: str,
    score: float,
    feature_importance: list[dict[str, Any]],
    row_count: int,
    train_rows: int,
    test_rows: int,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    classes: list[str],
    runtime_info: dict[str, Any],
    input_csv: str | None,
) -> dict[str, Any]:
    primary_score_group = "best_auroc_scores" if metric_name == "roc_auc" else "best_acc_scores"
    report = {
        "available": True,
        "protocol": "gpu_lightweight_xgb" if device == "cuda" else "lightweight_xgb_smoke",
        "task_type": "classification",
        "tabdiff_task_type": "binclass" if len(classes) == 2 else "multiclass",
        "metric": metric_name,
        "overall": score,
        "tail": None,
        "middle": None,
        "mode": None,
        "rows": {"tail": 0, "middle": 0, "mode": 0},
        "region_metrics_available": False,
        "region_metrics_reason": "disabled_for_lightweight_xgb_smoke",
        "primary_score_group": primary_score_group,
        "primary_model": "XGBClassifier",
        "runtime_tree_method": "hist",
        "runtime_device": device,
        "runtime_note": "smoke test for one CSV/environment; not a general CUDA hardware benchmark",
        "runtime_xgboost": runtime_info,
        "overall_scores": {
            primary_score_group: {
                "XGBClassifier": {metric_name: score},
            }
        },
        "feature_importance": feature_importance,
        "input_csv": input_csv,
        "row_count": int(row_count),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "classes": classes,
    }
    missing = [field for field in REQUIRED_UTILITY_EXACT_FIELDS if field not in report]
    if missing:
        raise RuntimeError(f"Schema is incomplete; missing fields: {missing}")
    if metric_name == "roc_auc" and not (0.0 <= score <= 1.0):
        raise RuntimeError(f"roc_auc is out of range: {score}")
    if metric_name == "accuracy" and not (0.0 <= score <= 1.0):
        raise RuntimeError(f"accuracy is out of range: {score}")
    if not feature_importance:
        raise RuntimeError("feature_importance is empty.")
    if device == "cuda":
        device_values = [str(value).lower() for value in runtime_info.get("device_values", [])]
        if not any("cuda" in value for value in device_values):
            raise RuntimeError(
                "Requested device=cuda, but the fitted XGBoost booster config does not report a CUDA device. "
                f"device_values={runtime_info.get('device_values')!r}"
            )
    return report


def run_smoke(*, device: str, seed: int) -> dict[str, Any]:
    from xgboost import XGBClassifier

    syn_df = _make_adult_like_frame(256, seed=seed, include_unknown_workclass=False)
    test_df = _make_adult_like_frame(96, seed=seed + 1, include_unknown_workclass=True)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLUMNS),
            ("cat", _one_hot_encoder(), CATEGORICAL_COLUMNS),
        ]
    )
    x_train = preprocess.fit_transform(syn_df[FEATURE_COLUMNS])
    x_test = preprocess.transform(test_df[FEATURE_COLUMNS])
    y_train = (syn_df[TARGET_COLUMN].astype(str) == ">50K").astype(int).to_numpy()
    y_test = (test_df[TARGET_COLUMN].astype(str) == ">50K").astype(int).to_numpy()
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        raise RuntimeError("Synthetic smoke data does not contain both target classes.")

    model = XGBClassifier(**_xgb_classifier_kwargs(seed=seed, device=device, class_count=2))
    model.fit(np.asarray(x_train, dtype=float), y_train)
    x_test_np = np.asarray(x_test, dtype=float)
    metric_name, score = _metric_from_predictions(model, x_test_np, y_test, 2)

    feature_names = preprocess.get_feature_names_out()
    importances = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
    feature_importance = _aggregate_importance(
        feature_names,
        importances,
        feature_columns=FEATURE_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
    )
    report = _build_report(
        device=device,
        metric_name=metric_name,
        score=score,
        feature_importance=feature_importance,
        row_count=int(len(syn_df) + len(test_df)),
        train_rows=int(len(syn_df)),
        test_rows=int(len(test_df)),
        feature_columns=FEATURE_COLUMNS,
        numeric_columns=NUMERIC_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        target_column=TARGET_COLUMN,
        classes=["<=50K", ">50K"],
        runtime_info=_booster_runtime_info(model),
        input_csv=None,
    )
    return {
        "smoke_status": "passed",
        "gpu_acceleration_used": device == "cuda",
        "note": "Synthetic smoke data only.",
        "report": report,
    }


def run_csv_smoke(*, csv_path: str, target_column: str | None, device: str, seed: int, test_size: float) -> dict[str, Any]:
    from xgboost import XGBClassifier

    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"CSV is empty: {csv_path}")
    target = _infer_target_column(df, target_column)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if len(df) < 4:
        raise RuntimeError(f"CSV has too few usable rows after dropping missing target values: {len(df)}")

    feature_columns = [str(column) for column in df.columns if column != target]
    if not feature_columns:
        raise RuntimeError("No feature columns are available.")
    class_count = int(df[target].astype(str).nunique(dropna=True))
    if class_count < 2:
        raise RuntimeError(f"Target column {target!r} has fewer than two classes.")
    if class_count > 20:
        raise RuntimeError(
            f"Target column {target!r} has {class_count} classes; this smoke script only handles classification."
        )

    numeric_columns, categorical_columns = _infer_feature_types(df, feature_columns)
    prepared = _prepare_frame_for_preprocess(
        df[[*feature_columns, target]],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(prepared[target].astype(str))
    min_class_count = int(pd.Series(y).value_counts().min())
    stratify = y if min_class_count >= 2 else None
    train_df, test_df, y_train, y_test = train_test_split(
        prepared[feature_columns],
        y,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=stratify,
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_columns),
            ("cat", _one_hot_encoder(), categorical_columns),
        ],
        remainder="drop",
    )
    x_train = preprocess.fit_transform(train_df)
    x_test = preprocess.transform(test_df)
    model = XGBClassifier(**_xgb_classifier_kwargs(seed=seed, device=device, class_count=class_count))
    model.fit(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=int))

    x_test_np = np.asarray(x_test, dtype=float)
    metric_name, score = _metric_from_predictions(model, x_test_np, np.asarray(y_test, dtype=int), class_count)
    feature_importance = _aggregate_importance(
        preprocess.get_feature_names_out(),
        np.asarray(getattr(model, "feature_importances_", []), dtype=float),
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
    )
    report = _build_report(
        device=device,
        metric_name=metric_name,
        score=score,
        feature_importance=feature_importance,
        row_count=int(len(df)),
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        target_column=target,
        classes=[str(item) for item in label_encoder.classes_],
        runtime_info=_booster_runtime_info(model),
        input_csv=csv_path,
    )
    return {
        "smoke_status": "passed",
        "gpu_acceleration_used": device == "cuda",
        "note": "CSV smoke validates this XGBoost run on this environment only.",
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a schema-compatible lightweight XGBoost utility evaluator.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--csv", default=None, help="Optional full CSV to process instead of synthetic smoke data.")
    parser.add_argument("--target-column", default=None, help="Target column name. Defaults to income/target/label/class/y/last column.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction used when --csv is provided.")
    args = parser.parse_args()

    try:
        if args.csv:
            result = run_csv_smoke(
                csv_path=args.csv,
                target_column=args.target_column,
                device=args.device,
                seed=args.seed,
                test_size=args.test_size,
            )
        else:
            result = run_smoke(device=args.device, seed=args.seed)
    except Exception as exc:
        failure = {
            "smoke_status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "gpu_acceleration_used": False,
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
