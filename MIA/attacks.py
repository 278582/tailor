from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .preprocess import TabularEncoder, row_key_frame, select_attack_columns


@dataclass
class AttackData:
    member: pd.DataFrame
    nonmember: pd.DataFrame
    synthetic: pd.DataFrame
    reference: pd.DataFrame
    columns: list[str]

    @property
    def target_df(self) -> pd.DataFrame:
        return pd.concat([self.member[self.columns], self.nonmember[self.columns]], ignore_index=True)

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([1] * len(self.member) + [0] * len(self.nonmember), dtype=int)


@dataclass
class AttackOutput:
    name: str
    scores: np.ndarray
    details: dict[str, Any]


def make_attack_data(
    *,
    member: pd.DataFrame,
    nonmember: pd.DataFrame,
    synthetic: pd.DataFrame,
    reference: pd.DataFrame,
    requested_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> AttackData:
    columns = select_attack_columns(
        [member, nonmember, synthetic, reference],
        requested_columns=requested_columns,
        exclude_columns=exclude_columns,
    )
    return AttackData(
        member=member.reset_index(drop=True),
        nonmember=nonmember.reset_index(drop=True),
        synthetic=synthetic.reset_index(drop=True),
        reference=reference.reset_index(drop=True),
        columns=columns,
    )


def exact_match_attack(data: AttackData) -> AttackOutput:
    synthetic_keys = set(row_key_frame(data.synthetic, data.columns).tolist())
    target_keys = row_key_frame(data.target_df, data.columns)
    scores = target_keys.map(lambda key: 1.0 if key in synthetic_keys else 0.0).to_numpy(dtype=float)
    return AttackOutput(
        name="exact_match",
        scores=scores,
        details={
            "threat_model": "release_only",
            "matched_member_rows": int(np.sum(scores[: len(data.member)] == 1.0)),
            "matched_nonmember_rows": int(np.sum(scores[len(data.member) :] == 1.0)),
        },
    )


def nearest_neighbor_attack(data: AttackData) -> AttackOutput:
    if data.synthetic.empty:
        return AttackOutput("nearest_neighbor", np.zeros(len(data.labels), dtype=float), {"skipped": "empty_synthetic"})
    encoder = TabularEncoder.fit(
        [data.synthetic[data.columns], data.reference[data.columns], data.target_df[data.columns]],
        data.columns,
    )
    synthetic_matrix = encoder.transform(data.synthetic[data.columns])
    target_matrix = encoder.transform(data.target_df[data.columns])
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(synthetic_matrix)
    distances = nn.kneighbors(target_matrix, return_distance=True)[0][:, 0].astype(float)
    return AttackOutput(
        name="nearest_neighbor",
        scores=-distances,
        details={
            "threat_model": "release_only",
            "score_semantics": "higher means closer to the released synthetic table",
            "mean_member_distance": float(np.mean(distances[: len(data.member)])) if len(data.member) else None,
            "mean_nonmember_distance": float(np.mean(distances[len(data.member) :])) if len(data.nonmember) else None,
        },
    )


def density_ratio_attack(data: AttackData, *, k: int = 5, eps: float = 1e-8) -> AttackOutput:
    if data.synthetic.empty or data.reference.empty:
        return AttackOutput(
            "density_ratio",
            np.zeros(len(data.labels), dtype=float),
            {"skipped": "empty_synthetic_or_reference"},
        )
    encoder = TabularEncoder.fit(
        [data.synthetic[data.columns], data.reference[data.columns], data.target_df[data.columns]],
        data.columns,
    )
    target_matrix = encoder.transform(data.target_df[data.columns])
    synthetic_matrix = encoder.transform(data.synthetic[data.columns])
    reference_matrix = encoder.transform(data.reference[data.columns])
    k_syn = max(1, min(int(k), synthetic_matrix.shape[0]))
    k_ref = max(1, min(int(k), reference_matrix.shape[0]))
    syn_dist = _kth_distance(target_matrix, synthetic_matrix, k_syn)
    ref_dist = _kth_distance(target_matrix, reference_matrix, k_ref)
    scores = np.log((ref_dist + eps) / (syn_dist + eps))
    return AttackOutput(
        name="density_ratio",
        scores=scores.astype(float),
        details={
            "threat_model": "release_only_calibrated",
            "method": "DOMIAS-inspired kNN density ratio",
            "k_synthetic": k_syn,
            "k_reference": k_ref,
            "score_semantics": "higher means denser under synthetic than under reference/control",
        },
    )


def _kth_distance(query, reference, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(reference)
    distances = nn.kneighbors(query, return_distance=True)[0]
    return distances[:, k - 1].astype(float)


def attribute_error_attack(
    data: AttackData,
    *,
    max_columns: int = 20,
    random_state: int = 20260420,
) -> AttackOutput:
    if data.synthetic.empty or len(data.columns) < 2:
        return AttackOutput(
            "attribute_error",
            np.zeros(len(data.labels), dtype=float),
            {"skipped": "requires_non_empty_synthetic_and_at_least_two_columns"},
        )

    columns = _choose_attribute_columns(data.synthetic, data.columns, max_columns=max_columns)
    target_df = data.target_df[data.columns].reset_index(drop=True)
    errors: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for target_column in columns:
        feature_columns = [column for column in data.columns if column != target_column]
        report, error = _fit_attribute_predictor(
            synthetic=data.synthetic[data.columns],
            target=target_df,
            feature_columns=feature_columns,
            target_column=target_column,
            random_state=random_state,
        )
        reports.append(report)
        if error is not None:
            errors.append(error)

    if not errors:
        return AttackOutput(
            "attribute_error",
            np.zeros(len(data.labels), dtype=float),
            {"skipped": "no_attribute_predictor_could_be_fit", "column_reports": reports},
        )

    error_matrix = np.vstack(errors)
    mean_error = np.nanmean(error_matrix, axis=0)
    return AttackOutput(
        name="attribute_error",
        scores=-mean_error.astype(float),
        details={
            "threat_model": "release_only",
            "method": "MIA-EPT-inspired per-column prediction error profile",
            "score_semantics": "higher means lower reconstruction/prediction error from synthetic-trained models",
            "evaluated_columns": [report["column"] for report in reports if report.get("used")],
            "column_reports": reports,
        },
    )


def _choose_attribute_columns(synthetic: pd.DataFrame, columns: list[str], *, max_columns: int) -> list[str]:
    if len(columns) <= max_columns:
        return list(columns)
    nunique = synthetic[columns].nunique(dropna=True).sort_values(ascending=False)
    return [str(column) for column in nunique.index[:max_columns]]


def _fit_attribute_predictor(
    *,
    synthetic: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    random_state: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    y_train = synthetic[target_column]
    if y_train.nunique(dropna=True) < 2:
        return {"column": target_column, "used": False, "reason": "constant_target"}, None

    encoder = TabularEncoder.fit([synthetic[feature_columns], target[feature_columns]], feature_columns)
    x_train = encoder.transform(synthetic[feature_columns])
    x_target = encoder.transform(target[feature_columns])

    if pd.api.types.is_numeric_dtype(y_train):
        y_numeric = pd.to_numeric(y_train, errors="coerce")
        valid = y_numeric.notna().to_numpy()
        if int(np.sum(valid)) < 5:
            return {"column": target_column, "used": False, "reason": "too_few_numeric_values"}, None
        model = ExtraTreesRegressor(
            n_estimators=64,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        )
        model.fit(_slice_matrix(x_train, valid), y_numeric.to_numpy(dtype=float)[valid])
        pred = model.predict(x_target)
        true = pd.to_numeric(target[target_column], errors="coerce").to_numpy(dtype=float)
        scale = float(np.nanstd(y_numeric.to_numpy(dtype=float)[valid]))
        scale = max(scale, 1e-8)
        error = np.abs(true - pred) / scale
        error[~np.isfinite(error)] = np.nanmax(error[np.isfinite(error)]) if np.any(np.isfinite(error)) else 1.0
        return {"column": target_column, "used": True, "type": "numeric", "scale": scale}, error.astype(float)

    y_text = y_train.fillna("__MISSING__").astype(str)
    class_counts = y_text.value_counts()
    if len(class_counts) < 2 or int(class_counts.sum()) < 5:
        return {"column": target_column, "used": False, "reason": "too_few_categorical_values"}, None
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_text)
    model = ExtraTreesClassifier(
        n_estimators=64,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(x_train, y_encoded)
    proba = model.predict_proba(x_target)
    class_to_index = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    true = target[target_column].fillna("__MISSING__").astype(str).to_numpy()
    error = np.empty(len(target), dtype=float)
    default_error = -np.log(1e-8)
    for idx, value in enumerate(true):
        class_idx = class_to_index.get(value)
        if class_idx is None:
            error[idx] = default_error
        else:
            error[idx] = -np.log(max(float(proba[idx, class_idx]), 1e-8))
    return {"column": target_column, "used": True, "type": "categorical"}, error.astype(float)


def _slice_matrix(matrix, mask: np.ndarray):
    if sparse.issparse(matrix):
        return matrix[mask]
    return np.asarray(matrix)[mask]


def run_release_attacks(
    data: AttackData,
    *,
    density_k: int = 5,
    max_attribute_columns: int = 20,
    random_state: int = 20260420,
) -> list[AttackOutput]:
    return [
        exact_match_attack(data),
        nearest_neighbor_attack(data),
        density_ratio_attack(data, k=density_k),
        attribute_error_attack(data, max_columns=max_attribute_columns, random_state=random_state),
    ]


def attack_score_frame(outputs: list[AttackOutput], labels: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"label": labels.astype(int)})
    for output in outputs:
        frame[output.name] = output.scores.astype(float)
    return frame


def supervised_profile_attack(
    score_frame: pd.DataFrame,
    *,
    random_state: int = 20260420,
) -> AttackOutput | None:
    feature_columns = [column for column in score_frame.columns if column != "label"]
    if not feature_columns:
        return None
    x = score_frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = score_frame["label"].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    if np.any(class_counts < 4):
        return None
    n_splits = int(min(5, class_counts.min()))
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    return AttackOutput(
        name="supervised_error_profile",
        scores=scores.astype(float),
        details={
            "threat_model": "upper_bound_not_release_only",
            "method": "cross-validated classifier over release-only attack scores",
            "feature_columns": feature_columns,
            "cv_splits": n_splits,
        },
    )


def shadow_attack(
    *,
    target_score_frame: pd.DataFrame,
    shadow_score_frames: list[pd.DataFrame],
    random_state: int = 20260420,
) -> AttackOutput | None:
    target_feature_columns = [column for column in target_score_frame.columns if column != "label"]
    if not target_feature_columns or not shadow_score_frames:
        return None
    train_frame = pd.concat(shadow_score_frames, ignore_index=True)
    if "label" not in train_frame.columns:
        return None
    feature_columns = [column for column in target_feature_columns if column in train_frame.columns]
    if not feature_columns:
        return None
    y = train_frame["label"].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    if np.any(class_counts < 2):
        return None
    x_train = train_frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_target = target_score_frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model = RandomForestClassifier(
        n_estimators=128,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(x_train, y)
    scores = model.predict_proba(x_target)[:, 1]
    return AttackOutput(
        name="shadow_attack",
        scores=scores.astype(float),
        details={
            "threat_model": "shadow_attack",
            "method": "classifier trained on supplied shadow run attack-score profiles",
            "shadow_rows": int(len(train_frame)),
            "shadow_runs": int(len(shadow_score_frames)),
            "feature_columns": feature_columns,
        },
    )
