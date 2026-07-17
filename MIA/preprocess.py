from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MISSING_TOKEN = "__MISSING__"


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def common_columns(frames: Iterable[pd.DataFrame]) -> list[str]:
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        return []
    ordered = list(first.columns)
    common = set(ordered)
    for frame in iterator:
        common &= set(frame.columns)
    return [column for column in ordered if column in common]


def select_attack_columns(
    frames: Iterable[pd.DataFrame],
    *,
    requested_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> list[str]:
    frames = list(frames)
    available = common_columns(frames)
    exclude = set(exclude_columns or [])
    if requested_columns:
        missing = [column for column in requested_columns if column not in available]
        if missing:
            raise ValueError(f"Requested attack columns are missing from at least one input: {missing}")
        selected = list(requested_columns)
    else:
        selected = available
    selected = [column for column in selected if column not in exclude]
    if not selected:
        raise ValueError("No attack columns remain after alignment and exclusions.")
    return selected


def split_column_types(df: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


@dataclass
class TabularEncoder:
    columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    categorical_imputer: SimpleImputer | None = None
    ohe: OneHotEncoder | None = None

    @classmethod
    def fit(cls, frames: list[pd.DataFrame], columns: list[str]) -> "TabularEncoder":
        fit_df = pd.concat([frame[columns] for frame in frames if not frame.empty], ignore_index=True)
        numeric_columns, categorical_columns = split_column_types(fit_df, columns)
        encoder = cls(
            columns=list(columns),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )
        if numeric_columns:
            encoder.numeric_imputer = SimpleImputer(strategy="median")
            numeric = encoder.numeric_imputer.fit_transform(fit_df[numeric_columns])
            encoder.scaler = StandardScaler()
            encoder.scaler.fit(numeric)
        if categorical_columns:
            encoder.categorical_imputer = SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)
            categorical = encoder.categorical_imputer.fit_transform(fit_df[categorical_columns].astype(object))
            encoder.ohe = _make_ohe()
            encoder.ohe.fit(categorical.astype(str))
        return encoder

    def transform(self, df: pd.DataFrame):
        pieces = []
        if self.numeric_columns:
            assert self.numeric_imputer is not None
            assert self.scaler is not None
            numeric = self.numeric_imputer.transform(df[self.numeric_columns])
            pieces.append(sparse.csr_matrix(self.scaler.transform(numeric)))
        if self.categorical_columns:
            assert self.categorical_imputer is not None
            assert self.ohe is not None
            categorical = self.categorical_imputer.transform(df[self.categorical_columns].astype(object))
            pieces.append(self.ohe.transform(categorical.astype(str)))
        if not pieces:
            return sparse.csr_matrix((len(df), 0), dtype=float)
        if len(pieces) == 1:
            return pieces[0].tocsr()
        return sparse.hstack(pieces, format="csr")

    def dense_transform(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self.transform(df)
        if sparse.issparse(matrix):
            return matrix.toarray()
        return np.asarray(matrix, dtype=float)


def row_key_frame(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    normalized = df[columns].copy()
    for column in columns:
        normalized[column] = normalized[column].map(_normalize_value_for_key)
    return normalized.apply(lambda row: tuple(row.tolist()), axis=1)


def _normalize_value_for_key(value) -> str:
    if pd.isna(value):
        return MISSING_TOKEN
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)

