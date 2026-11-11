from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_io import save_json, save_jsonl
from .types import CardsBundle

DEFAULT_NUM_PROTOTYPES = 32


def _build_schema_card(
    train_df: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    discrete_numerical_columns: list[str],
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
) -> dict[str, Any]:
    columns: dict[str, dict[str, Any]] = {}
    discrete_set = set(discrete_numerical_columns)
    categorical_set = set(categorical_columns or [])
    numerical_set = set(numerical_columns or [])
    for column in train_df.columns:
        is_target = column == target_column
        inferred_numeric = pd.api.types.is_numeric_dtype(train_df[column]) and not is_target
        if column in discrete_set:
            column_type = "discrete_numerical"
        elif column in categorical_set:
            column_type = "categorical"
        elif column in numerical_set:
            column_type = "numerical"
        else:
            column_type = "numerical" if inferred_numeric else "categorical"
        entry: dict[str, Any] = {
            "type": column_type,
            "is_target": is_target,
            "missing_allowed": bool(train_df[column].isna().any()),
        }
        if column_type == "discrete_numerical":
            legal_values = sorted({int(v) for v in train_df[column].dropna().tolist()})
            entry["legal_values"] = legal_values
        elif column_type == "categorical":
            categories = sorted({str(v) for v in train_df[column].dropna().tolist()})
            entry["legal_values"] = categories
        columns[column] = entry
    return {
        "dataset": dataset_name,
        "column_order": list(train_df.columns),
        "target_column": target_column,
        "columns": columns,
    }


def _quantile_bins(series: pd.Series, n_bins: int = 8) -> list[float]:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(series.to_numpy(dtype=float), quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        value = float(series.iloc[0])
        edges = np.array([value - 0.5, value + 0.5], dtype=float)
    return [float(x) for x in edges]


def _build_stats_card(
    train_df: pd.DataFrame,
    schema_card: dict[str, Any],
    dataset_name: str,
    privacy_sensitive_columns: list[str],
) -> dict[str, Any]:
    numeric_stats: dict[str, dict[str, float]] = {}
    numeric_bins: dict[str, list[float]] = {}
    discrete_numeric_freqs: dict[str, list[dict[str, Any]]] = {}
    categorical_freqs: dict[str, list[dict[str, Any]]] = {}
    categorical_top_values: dict[str, list[dict[str, Any]]] = {}

    for column in schema_card["column_order"]:
        info = schema_card["columns"][column]
        if info["type"] == "numerical":
            series = train_df[column].astype(float)
            numeric_stats[column] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "p05": float(series.quantile(0.05)),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
                "p95": float(series.quantile(0.95)),
            }
            numeric_bins[column] = _quantile_bins(series)
        elif info["type"] == "discrete_numerical":
            counts = Counter(int(v) for v in train_df[column].dropna().tolist())
            total = sum(counts.values()) or 1
            discrete_numeric_freqs[column] = [
                {"value": value, "count": count, "freq": count / total}
                for value, count in sorted(counts.items())
            ]
        else:
            counts = Counter(str(v) for v in train_df[column].dropna().tolist())
            total = sum(counts.values()) or 1
            freq_list = [
                {"value": key, "count": count, "freq": count / total}
                for key, count in counts.most_common()
            ]
            categorical_freqs[column] = freq_list
            categorical_top_values[column] = freq_list[:10]

    return {
        "dataset": dataset_name,
        "privacy_sensitive_columns": privacy_sensitive_columns,
        "numeric_stats": numeric_stats,
        "numeric_bins": numeric_bins,
        "discrete_numeric_freqs": discrete_numeric_freqs,
        "categorical_freqs": categorical_freqs,
        "categorical_top_values": categorical_top_values,
    }


def _build_prototype_entries(
    train_df: pd.DataFrame,
    seed: int,
    target_column: str,
    num_prototypes: int = DEFAULT_NUM_PROTOTYPES,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    target_series = train_df[target_column]
    if target_series.nunique(dropna=False) > max(8, num_prototypes // 2):
        proto_df = train_df.sample(n=min(num_prototypes, len(train_df)), random_state=seed, replace=False).reset_index(
            drop=True
        )
    else:
        groups: list[pd.DataFrame] = []
        target_counts = target_series.value_counts().to_dict()
        total = sum(target_counts.values())
        for label, count in target_counts.items():
            share = count / total
            num_take = max(1, int(round(num_prototypes * share)))
            sample = train_df[train_df[target_column] == label].sample(
                n=min(num_take, count),
                random_state=seed,
                replace=False,
            )
            groups.append(sample)
        proto_df = pd.concat(groups, ignore_index=True)
        if len(proto_df) > num_prototypes:
            proto_df = proto_df.sample(n=num_prototypes, random_state=seed, replace=False)
        proto_df = proto_df.reset_index(drop=True)

    entries: list[dict[str, Any]] = []
    for idx, row in proto_df.iterrows():
        entries.append(
            {
                "prototype_id": idx,
                "target_value": row[target_column],
                "row": row.to_dict(),
            }
        )
    rng.shuffle(entries)
    return entries


def _build_residual_card() -> dict[str, Any]:
    return {
        "tail": [],
        "rare_category": [],
        "rare_subgroup": [],
        "strong_dependency": [],
        "status": "placeholder_only_for_postprocess",
    }


def build_and_save_cards(
    train_df: pd.DataFrame,
    output_dir: Path,
    seed: int,
    dataset_name: str = "tabular",
    target_column: str | None = None,
    discrete_numerical_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
    privacy_sensitive_columns: list[str] | None = None,
    num_prototypes: int = DEFAULT_NUM_PROTOTYPES,
) -> CardsBundle:
    if target_column is None:
        target_column = str(train_df.columns[-1])
    discrete_numerical_columns = [] if discrete_numerical_columns is None else list(discrete_numerical_columns)
    privacy_sensitive_columns = [] if privacy_sensitive_columns is None else list(privacy_sensitive_columns)
    schema_card = _build_schema_card(
        train_df,
        dataset_name=dataset_name,
        target_column=target_column,
        discrete_numerical_columns=discrete_numerical_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )
    stats_card = _build_stats_card(
        train_df,
        schema_card,
        dataset_name=dataset_name,
        privacy_sensitive_columns=privacy_sensitive_columns,
    )
    prototype_entries = _build_prototype_entries(
        train_df,
        seed=seed,
        target_column=target_column,
        num_prototypes=num_prototypes,
    )
    residual_card = _build_residual_card()

    save_json(output_dir / "schema_card.json", schema_card)
    save_json(output_dir / "stats_card.json", stats_card)
    save_jsonl(output_dir / "prototype_card.jsonl", prototype_entries)
    save_json(output_dir / "residual_card.json", residual_card)

    return CardsBundle(
        schema_card=schema_card,
        stats_card=stats_card,
        prototype_entries=prototype_entries,
        residual_card=residual_card,
    )
