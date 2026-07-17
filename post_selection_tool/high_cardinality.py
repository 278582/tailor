from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class HighCardinalityColumnMapping:
    column: str
    top_values: list[str]
    value_to_cluster: dict[str, str]
    cluster_tokens: list[str]
    unknown_token: str
    frequency_mass_by_cluster: dict[str, float]


class HighCardinalityCompressor:
    def __init__(
        self,
        *,
        enabled: bool,
        threshold: int = 256,
        top_k: int = 64,
        tail_cluster_count: int = 16,
    ) -> None:
        self.enabled = bool(enabled)
        self.threshold = int(threshold)
        self.top_k = int(top_k)
        self.tail_cluster_count = int(tail_cluster_count)
        self.mappings: dict[str, HighCardinalityColumnMapping] = {}

    @property
    def active_columns(self) -> list[str]:
        return sorted(self.mappings)

    def fit(
        self,
        train_df: pd.DataFrame,
        *,
        categorical_columns: list[str],
        target_column: str,
    ) -> "HighCardinalityCompressor":
        if not self.enabled:
            return self

        for column in categorical_columns:
            values = train_df[column].astype(str)
            counts = values.value_counts(dropna=False)
            if int(counts.size) < self.threshold:
                continue

            top_values = [str(value) for value in counts.head(self.top_k).index.tolist()]
            top_set = set(top_values)
            tail_values = [str(value) for value in counts.index.tolist() if str(value) not in top_set]
            if not tail_values:
                continue

            total_tail_count = int(counts.loc[tail_values].sum())
            if total_tail_count <= 0:
                continue
            cluster_count = min(self.tail_cluster_count, max(1, len(tail_values)))
            cluster_tokens = [self._cluster_token(column, idx) for idx in range(cluster_count)]
            value_to_cluster: dict[str, str] = {}
            cluster_masses = {token: 0 for token in cluster_tokens}
            cumulative = 0
            for value in sorted(tail_values, key=lambda item: (-int(counts.loc[item]), item)):
                count = int(counts.loc[value])
                midpoint = cumulative + 0.5 * count
                cluster_idx = min(int(midpoint * cluster_count / max(total_tail_count, 1)), cluster_count - 1)
                token = cluster_tokens[cluster_idx]
                value_to_cluster[value] = token
                cluster_masses[token] += count
                cumulative += count

            frequency_mass_by_cluster = {
                token: float(count / max(total_tail_count, 1))
                for token, count in cluster_masses.items()
            }
            unknown_idx = int(np.argmax(np.asarray([cluster_masses[token] for token in cluster_tokens], dtype=float)))
            self.mappings[column] = HighCardinalityColumnMapping(
                column=column,
                top_values=top_values,
                value_to_cluster=value_to_cluster,
                cluster_tokens=cluster_tokens,
                unknown_token=cluster_tokens[unknown_idx],
                frequency_mass_by_cluster=frequency_mass_by_cluster,
            )
        return self

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.mappings or df.empty:
            return df.copy()
        transformed = df.copy()
        for column in self.mappings:
            if column in transformed.columns:
                transformed[column] = self.transform_series(transformed[column], column)
        return transformed

    def transform_series(self, series: pd.Series, column: str) -> pd.Series:
        mapping = self.mappings.get(column)
        if mapping is None:
            return series
        top_set = set(mapping.top_values)

        def _map(value: Any) -> str:
            text = str(value)
            if text in top_set or text in mapping.cluster_tokens:
                return text
            return mapping.value_to_cluster.get(text, mapping.unknown_token)

        return series.map(_map).astype(str)

    def values_for_column(self, original_values: list[Any], column: str) -> list[str]:
        mapping = self.mappings.get(column)
        if mapping is None:
            return [str(value) for value in original_values]
        return mapping.top_values + mapping.cluster_tokens

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "strategy": "unsupervised_frequency_tail_clustering",
            "supervised": False,
            "threshold": int(self.threshold),
            "top_k": int(self.top_k),
            "tail_cluster_count": int(self.tail_cluster_count),
            "columns": {
                column: {
                    "top_values": mapping.top_values,
                    "cluster_tokens": mapping.cluster_tokens,
                    "unknown_token": mapping.unknown_token,
                    "frequency_mass_by_cluster": mapping.frequency_mass_by_cluster,
                    "tail_values": len(mapping.value_to_cluster),
                }
                for column, mapping in self.mappings.items()
            },
        }

    def _cluster_token(self, column: str, idx: int) -> str:
        return f"__HC_{column}_TAIL_{idx:02d}__"
