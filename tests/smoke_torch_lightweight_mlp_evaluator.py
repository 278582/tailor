from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
    "runtime_device",
    "overall_scores",
]


class TorchMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, class_count: int) -> None:
        super().__init__()
        output_dim = 1 if class_count == 2 else class_count
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _infer_target_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested target column {requested!r} is not present.")
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


def _prepare_frame(
    df: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    prepared = df.copy()
    for column in numeric_columns:
        values = pd.to_numeric(prepared[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        prepared[column] = values.fillna(median)
    for column in categorical_columns:
        prepared[column] = prepared[column].astype("object").where(prepared[column].notna(), "__missing__").astype(str)
    return prepared


def _encoded_feature_to_original(encoded_feature: str, categorical_columns: list[str]) -> str:
    if encoded_feature.startswith("num__"):
        return encoded_feature.removeprefix("num__")
    if encoded_feature.startswith("cat__"):
        raw = encoded_feature.removeprefix("cat__")
        for column in categorical_columns:
            if raw.startswith(f"{column}_"):
                return column
    return encoded_feature.split("_", 1)[0]


def _aggregate_encoded_importance(
    feature_names: np.ndarray,
    importances: np.ndarray,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    grouped = {feature: 0.0 for feature in feature_columns}
    for encoded_feature, importance in zip(feature_names, importances):
        original = _encoded_feature_to_original(str(encoded_feature), categorical_columns)
        grouped[original] = grouped.get(original, 0.0) + float(max(0.0, importance))
    total = float(sum(grouped.values()))
    if total <= 0.0:
        total = 1.0
    rows = [
        {"feature": feature, "importance": float(value / total)}
        for feature, value in grouped.items()
    ]
    rows.sort(key=lambda item: item["importance"], reverse=True)
    if limit is not None:
        rows = rows[: int(limit)]
    return [
        {"feature": item["feature"], "importance": round(float(item["importance"]), 6), "rank": rank}
        for rank, item in enumerate(rows, start=1)
    ]


def _resolve_torch_device(requested: str) -> torch.device:
    normalized = str(requested or "cuda:0").strip().lower()
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
        return torch.device(normalized)
    return torch.device("cpu")


def _predict_scores(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    class_count: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            batch = torch.as_tensor(x[start : start + int(batch_size)], dtype=torch.float32, device=device)
            logits = model(batch)
            if class_count == 2:
                scores = torch.sigmoid(logits.reshape(-1))
            else:
                scores = torch.softmax(logits, dim=1)
            outputs.append(scores.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _score_predictions(y_true: np.ndarray, scores: np.ndarray, *, class_count: int) -> tuple[str, float]:
    if class_count == 2:
        return "roc_auc", float(roc_auc_score(y_true, scores.reshape(-1)))
    pred = np.asarray(scores).argmax(axis=1)
    return "accuracy", float(accuracy_score(y_true, pred))


def _train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    device: torch.device,
    class_count: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[TorchMLPClassifier, list[dict[str, Any]]]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = TorchMLPClassifier(input_dim=x_train.shape[1], hidden_dim=int(hidden_dim), class_count=class_count).to(device)
    x_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    if class_count == 2:
        y_tensor = torch.as_tensor(y_train.astype(np.float32).reshape(-1, 1), dtype=torch.float32)
        criterion: nn.Module = nn.BCEWithLogitsLoss()
    else:
        y_tensor = torch.as_tensor(y_train.astype(np.int64), dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else None})
    return model, history


def _permutation_importance(
    *,
    model: nn.Module,
    preprocess: ColumnTransformer,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    feature_columns: list[str],
    class_count: int,
    baseline_metric: str,
    baseline_score: float,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed) + 991)
    drops: dict[str, float] = {}
    for feature in feature_columns:
        permuted = test_df.copy()
        values = permuted[feature].to_numpy(copy=True)
        rng.shuffle(values)
        permuted[feature] = values
        x_perm = preprocess.transform(permuted)
        scores = _predict_scores(
            model,
            np.asarray(x_perm, dtype=np.float32),
            device=device,
            class_count=class_count,
            batch_size=batch_size,
        )
        metric, score = _score_predictions(y_test, scores, class_count=class_count)
        if metric != baseline_metric:
            raise RuntimeError(f"Permutation metric changed from {baseline_metric} to {metric}.")
        drops[feature] = max(0.0, float(baseline_score) - float(score))
    total = float(sum(drops.values()))
    if total <= 0.0:
        return [
            {"feature": feature, "importance": 0.0, "rank": rank}
            for rank, feature in enumerate(feature_columns, start=1)
        ]
    rows = [
        {"feature": feature, "importance": float(value / total)}
        for feature, value in drops.items()
    ]
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return [
        {"feature": item["feature"], "importance": round(float(item["importance"]), 6), "rank": rank}
        for rank, item in enumerate(rows, start=1)
    ]


def _schema_card_from_columns(
    *,
    dataset_name: str,
    target_column: str,
    column_order: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for column in column_order:
        if column == target_column:
            col_type = "categorical"
        elif column in numeric_columns:
            col_type = "numerical"
        elif column in categorical_columns:
            col_type = "categorical"
        else:
            col_type = "unknown"
        columns[column] = {"type": col_type, "is_target": column == target_column}
    return {
        "dataset": dataset_name,
        "target_column": target_column,
        "column_order": column_order,
        "columns": columns,
    }


def _target_summary(series: pd.Series) -> dict[str, Any]:
    counts = series.astype(str).value_counts(dropna=False)
    total = int(counts.sum())
    minority = str(counts.idxmin()) if not counts.empty else None
    positive = ">50K" if ">50K" in counts.index else (str(counts.index[-1]) if not counts.empty else None)
    return {
        "target": series.name,
        "minority": minority,
        "positive_class": positive,
        "task_type": "binclass" if len(counts) == 2 else "multiclass",
        "utility_notes": [
            f"class balance: {', '.join(f'{label}={int(count)}' for label, count in counts.items())}",
            f"positive class for ROC-AUC smoke: {positive}",
        ],
        "class_distribution": [
            {"label": str(label), "count": int(count), "fraction": round(float(count / max(total, 1)), 6)}
            for label, count in counts.items()
        ],
    }


def _build_utility_report(
    *,
    metric_name: str,
    score: float,
    feature_importance: list[dict[str, Any]],
    csv_path: Path,
    row_count: int,
    train_rows: int,
    test_rows: int,
    target_column: str,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    classes: list[str],
    device: torch.device,
    model_device: str,
    history: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    primary_score_group = "best_auroc_scores" if metric_name == "roc_auc" else "best_acc_scores"
    report = {
        "available": True,
        "reason": None,
        "protocol": "torch_lightweight_mlp",
        "task_type": "classification",
        "tabdiff_task_type": "binclass" if len(classes) == 2 else "multiclass",
        "metric": metric_name,
        "overall": float(score),
        "tail": None,
        "middle": None,
        "mode": None,
        "rows": {"tail": 0, "middle": 0, "mode": 0},
        "region_metrics_available": False,
        "region_metrics_reason": "disabled_for_torch_lightweight_mlp_smoke",
        "primary_score_group": primary_score_group,
        "primary_model": "TorchMLPClassifier",
        "runtime_device": str(device),
        "runtime_model_device": model_device,
        "runtime_note": "GPU is used only if runtime_model_device starts with cuda.",
        "runtime_torch": {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "overall_scores": {
            primary_score_group: {
                "TorchMLPClassifier": {metric_name: float(score)},
            }
        },
        "feature_importance": feature_importance,
        "feature_importance_method": "permutation_importance_on_holdout",
        "input_csv": str(csv_path),
        "row_count": int(row_count),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "classes": classes,
        "train_history": history,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
    }
    missing = [field for field in REQUIRED_UTILITY_EXACT_FIELDS if field not in report]
    if missing:
        raise RuntimeError(f"utility report missing required fields: {missing}")
    if device.type == "cuda" and not str(model_device).startswith("cuda"):
        raise RuntimeError(f"Requested CUDA but model parameters are on {model_device}.")
    return report


def _metrics_summary_like(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": report.get("row_count"),
        "shape": None,
        "trend": None,
        "dcr_privacy_reward": None,
        "metric_directions": {
            "shape": "higher_better",
            "trend": "higher_better",
            "utility_exact_metric": report.get("metric"),
            "utility_exact_raw_direction": "higher_better" if str(report.get("metric")).lower() != "rmse" else "lower_better",
            "utility_exact_overall": "normalized_higher_better",
        },
        "utility_exact_metric": report.get("metric"),
        "utility_exact_available": bool(report.get("available")),
        "utility_exact_overall": report.get("overall"),
        "utility_exact_mode": report.get("mode"),
        "utility_exact_middle": report.get("middle"),
        "utility_exact_tail": report.get("tail"),
        "utility_exact_task_type": report.get("task_type"),
        "utility_exact_tabdiff_task_type": report.get("tabdiff_task_type"),
        "utility_exact_primary_score_group": report.get("primary_score_group"),
        "utility_exact_primary_model": report.get("primary_model"),
        "audit_metrics": {
            "utility_exact": report.get("overall"),
            "utility_exact_available": bool(report.get("available")),
        },
    }


def _render_prompt_restore_outputs(
    *,
    output_dir: Path,
    dataset_name: str,
    schema_card: dict[str, Any],
    target_summary: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    from llm_mcts_tool import v2_pipeline as v2

    config = v2.V2MCTSConfig(dataset_name=dataset_name, prompt_pack_dir=Path("prompt_pack"))
    dataset_context = v2._load_dataset_prompt_context(config, schema_card)
    dataset_context = dict(dataset_context)
    existing_target_summary = (
        dataset_context.get("target_summary", {}) if isinstance(dataset_context.get("target_summary"), dict) else {}
    )
    dataset_context["target_summary"] = {**existing_target_summary, **target_summary}

    utility_reference = {
        "dataset": dataset_name,
        "train_rows": report.get("train_rows"),
        "test_rows": report.get("test_rows"),
        "utility_feature_importance": {
            "backend": "torch_lightweight_mlp",
            "metric": report.get("metric"),
            "test_score": report.get("overall"),
            "top_features": report.get("feature_importance", [])[:8],
            "feature_importance_method": report.get("feature_importance_method"),
        },
        "target_summary": dataset_context["target_summary"],
        "semantic_summary": (
            f"torch_lightweight_mlp utility {report.get('metric')}={round(float(report.get('overall')), 4)}; "
            f"top utility features: "
            f"{', '.join(item['feature'] for item in report.get('feature_importance', [])[:4])}"
        ),
    }
    feedback = {
        "diagnostics": {
            "utility_xgb_feature_importance": report.get("feature_importance", [])[:8],
            "utility_feature_importance": report.get("feature_importance", [])[:8],
        },
        "utility_summary": {
            "utility_exact": report.get("overall"),
            "utility_exact_available": report.get("available"),
        },
        "privacy_summary": {},
    }
    compact_feedback = v2._compact_feedback_for_prompt(feedback)
    real_utility_prompt_payload = {
        "dataset_brief": v2._dataset_brief_for_prompt(schema_card, dataset_context),
        "real_utility_profile": v2._real_utility_for_prompt(utility_reference),
    }
    init_theta_payload = {
        "n_theta": 1,
        "dataset_brief": v2._dataset_brief_for_prompt(schema_card, dataset_context),
        "dcr_balance_semantics": v2.DCR_PROMPT_SEMANTICS,
        "real_utility_reference": v2._real_utility_for_prompt(utility_reference),
        "s_context": None,
        "seed_theta_examples": [],
    }
    top_features = [item["feature"] for item in report.get("feature_importance", [])[:3]]
    theta_batch = [
        {
            "node_id": "smoke_torch_mlp_node",
            "theta": {
                "col_1ds": [report["target_column"], *top_features[:2]],
                "col_2ds": [[report["target_column"], top_features[0]]] if top_features else [],
                "col_ps": [report["target_column"], *top_features[:2]],
                "col_u": top_features[0] if top_features else None,
            },
            "metrics_4d_reward": {"utility_exact": round(float(report.get("overall")), 4)},
            "exact_reward": None,
            "exact_reward_available": False,
            "search_reward": None,
            "search_reward_available": False,
            "reward_type": "utility_evaluator_smoke_only",
            "diagnostics": compact_feedback,
        }
    ]
    diagnosis_payload = {
        "real_utility_reference": v2._real_utility_for_prompt(utility_reference),
        "dcr_balance_semantics": v2.DCR_PROMPT_SEMANTICS,
        "s_context": None,
        "theta_batch": theta_batch,
        "best_so_far": None,
    }
    rendered = {
        "real_utility_profile_summary_prompt.md": v2._render_prompt(
            config,
            "v2_real_utility_profile_summary_prompt.j2",
            real_utility_prompt_payload,
        ),
        "init_node_prompt.md": v2._render_prompt(config, "v2_init_node_prompt.j2", init_theta_payload),
        "init_node_diagnosis_prompt.md": v2._render_prompt(
            config,
            "v2_init_node_diagnosis_prompt.j2",
            diagnosis_payload,
        ),
    }
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in rendered.items():
        (prompt_dir / filename).write_text(text, encoding="utf-8")

    payloads = {
        "real_utility_prompt_payload": real_utility_prompt_payload,
        "init_theta_payload": init_theta_payload,
        "diagnosis_payload": diagnosis_payload,
        "compact_feedback": compact_feedback,
    }
    _write_json(output_dir / "prompt_payloads.json", payloads)
    return {
        "prompt_dir": str(prompt_dir),
        "rendered_prompts": {name: {"chars": len(text), "lines": text.count("\n") + 1} for name, text in rendered.items()},
        "restored_prompt_fields": {
            "real_utility_reference": True,
            "utility_feature_importance": bool(report.get("feature_importance")),
            "utility_top": bool(compact_feedback.get("utility_top")),
            "utility_summary": bool(compact_feedback.get("utility_summary")),
            "init_theta_prompt": True,
            "init_node_diagnosis_prompt": True,
        },
        "known_missing_from_this_smoke": [
            "shape/trend density diagnostics are not computed",
            "DCR quantiles and DCR privacy reward are not computed",
            "source contribution is not computed",
            "real MCTS ThetaNode/SNode state is not reconstructed",
            "LLM completion is not called; prompts are rendered locally",
        ],
    }


def run_csv_smoke(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_torch_device(args.device)
    df = pd.read_csv(csv_path)
    target_column = _infer_target_column(df, args.target_column)
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    feature_columns = [str(column) for column in df.columns if column != target_column]
    if not feature_columns:
        raise RuntimeError("No feature columns are available.")
    class_count = int(df[target_column].astype(str).nunique(dropna=True))
    if class_count < 2:
        raise RuntimeError("Target has fewer than two classes.")
    if class_count > 20:
        raise RuntimeError(f"Only classification smoke is implemented; target has {class_count} classes.")

    numeric_columns, categorical_columns = _infer_feature_types(df, feature_columns)
    prepared = _prepare_frame(
        df[[*feature_columns, target_column]],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(prepared[target_column].astype(str))
    stratify = y if int(pd.Series(y).value_counts().min()) >= 2 else None
    train_df, test_df, y_train, y_test = train_test_split(
        prepared[feature_columns],
        y,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=stratify,
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", _one_hot_encoder(), categorical_columns),
        ],
        remainder="drop",
    )
    x_train = np.asarray(preprocess.fit_transform(train_df), dtype=np.float32)
    x_test = np.asarray(preprocess.transform(test_df), dtype=np.float32)
    model, history = _train_mlp(
        x_train,
        np.asarray(y_train, dtype=np.int64),
        device=device,
        class_count=class_count,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    scores = _predict_scores(
        model,
        x_test,
        device=device,
        class_count=class_count,
        batch_size=int(args.batch_size),
    )
    metric_name, score = _score_predictions(np.asarray(y_test, dtype=np.int64), scores, class_count=class_count)
    feature_importance = _permutation_importance(
        model=model,
        preprocess=preprocess,
        test_df=test_df.reset_index(drop=True),
        y_test=np.asarray(y_test, dtype=np.int64),
        feature_columns=feature_columns,
        class_count=class_count,
        baseline_metric=metric_name,
        baseline_score=score,
        device=device,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    model_device = str(next(model.parameters()).device)
    report = _build_utility_report(
        metric_name=metric_name,
        score=score,
        feature_importance=feature_importance,
        csv_path=csv_path,
        row_count=int(len(df)),
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        target_column=target_column,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        classes=[str(item) for item in label_encoder.classes_],
        device=device,
        model_device=model_device,
        history=history,
        elapsed_seconds=time.perf_counter() - started,
    )
    metrics_summary = _metrics_summary_like(report)
    schema_card = _schema_card_from_columns(
        dataset_name=args.dataset_name,
        target_column=target_column,
        column_order=[*feature_columns, target_column],
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    prompt_restore = _render_prompt_restore_outputs(
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        schema_card=schema_card,
        target_summary=_target_summary(df[target_column]),
        report=report,
    )
    restoration_report = {
        "status": "passed",
        "gpu_acceleration_used": bool(device.type == "cuda" and model_device.startswith("cuda")),
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "utility_exact_schema_restored": all(field in report for field in REQUIRED_UTILITY_EXACT_FIELDS),
        "metrics_summary_utility_fields_restored": all(
            key in metrics_summary
            for key in (
                "utility_exact_metric",
                "utility_exact_available",
                "utility_exact_overall",
                "utility_exact_task_type",
                "utility_exact_primary_model",
            )
        ),
        "prompt_restore": prompt_restore,
        "restoration_level": {
            "utility_exact_report": "full_schema",
            "utility_feature_importance": "restored_as_torch_permutation_importance",
            "real_utility_reference_prompt": "rendered",
            "init_theta_prompt": "rendered",
            "init_node_diagnosis_prompt": "rendered",
            "full_mcts_rollout_context": "not_restored_by_this_smoke",
        },
    }
    _write_json(output_dir / "utility_exact_report.json", report)
    _write_json(output_dir / "metrics_summary_like.json", metrics_summary)
    _write_json(output_dir / "restore_report.json", restoration_report)
    return restoration_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test torch_lightweight_mlp evaluator and prompt restoration.")
    parser.add_argument("--csv", default="/mnt/lustre/liuzhiwei/cpx/cpj/TGM/selection_pareto.csv")
    parser.add_argument("--target-column", default="income")
    parser.add_argument("--dataset-name", default="adult")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dir", default="artifacts/smoke_torch_mlp_selection_pareto")
    args = parser.parse_args()

    try:
        result = run_csv_smoke(args)
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(_json_safe(failure), ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
