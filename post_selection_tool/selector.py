from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .encoding import EncodingMixin
from .fidelity import FidelityMixin
from .high_cardinality import HighCardinalityCompressor
from .pareto_selection import ParetoSelectionMixin
from .preselect_scoring import PreselectScoringMixin
from .privacy import PrivacyMixin
from .repair import RepairMixin


DEFAULT_D_CUR_SIZE = 200


class ParetoSelector(
    EncodingMixin,
    FidelityMixin,
    PrivacyMixin,
    PreselectScoringMixin,
    RepairMixin,
    ParetoSelectionMixin,
):
    def __init__(
        self,
        train_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        schema_card: dict[str, Any],
        stats_card: dict[str, Any],
        seed: int,
        source: str = "llm",
        lambda_penalty: float = 1.0,
        gamma: float = 0.5,
        privacy_version: str = "v2",
        density_k: int = 10,
        density_reference_size: int = 5000,
        nn_device: str = "auto",
        nn_query_batch_size: int = 2048,
        nn_reference_chunk_size: int = 8192,
        rarity_strata: int = 5,
        max_pair_marginal_edges: int = 32,
        fidelity_1d_columns: list[str] | None = None,
        fidelity_2d_anchor_columns: list[str] | None = None,
        privacy_columns: list[str] | None = None,
        utility_balance_column: str | None = None,
        allow_target_in_fidelity_columns: bool = False,
        allow_target_in_privacy_columns: bool = False,
        privacy_encoding_column_mode: str = "privacy_columns",
        max_theta_pairs: int | None = None,
        final_fidelity_floor_eps: float = 0.01,
        final_trend_floor_eps: float = 0.01,
        high_cardinality_enabled: bool | None = None,
        high_cardinality_threshold: int = 256,
        high_cardinality_top_k: int = 64,
        high_cardinality_tail_clusters: int = 16,
    ) -> None:
        self.train_df = train_df.reset_index(drop=True)
        self.holdout_df = holdout_df.reset_index(drop=True)
        self.schema_card = schema_card
        self.stats_card = stats_card
        self.seed = seed
        self.source = source
        self.lambda_penalty = float(lambda_penalty)
        self.gamma = float(gamma)
        self.privacy_version = privacy_version
        self.density_k = max(1, int(density_k))
        self.density_reference_size = max(0, int(density_reference_size))
        self.nn_device_arg = nn_device
        self.nn_query_batch_size = max(1, int(nn_query_batch_size))
        self.nn_reference_chunk_size = max(1, int(nn_reference_chunk_size))
        self.rarity_strata = max(3, int(rarity_strata))
        if max_theta_pairs is not None:
            max_pair_marginal_edges = int(max_theta_pairs)
        self.max_pair_marginal_edges = max(0, int(max_pair_marginal_edges))
        self.final_fidelity_floor_eps = max(0.0, float(final_fidelity_floor_eps))
        self.final_trend_floor_eps = max(0.0, float(final_trend_floor_eps))
        self.progress_enabled = True
        self.nn_backend, self.nn_device = self._resolve_nn_backend(nn_device)

        self.column_order = schema_card["column_order"]
        self.target_column = schema_card["target_column"]
        self.feature_columns = [c for c in self.column_order if not schema_card["columns"][c]["is_target"]]
        self.fidelity_1d_columns = self._resolve_guided_column_scope(
            "fidelity_1d_columns",
            fidelity_1d_columns,
            default_columns=list(self.column_order),
            allow_target_when_guided=bool(allow_target_in_fidelity_columns),
            require_non_empty=True,
        )
        self.fidelity_columns = self.fidelity_1d_columns
        self.fidelity_2d_anchor_columns = self._resolve_guided_column_scope(
            "fidelity_2d_anchor_columns",
            fidelity_2d_anchor_columns,
            default_columns=list(self.column_order),
            allow_target_when_guided=bool(allow_target_in_fidelity_columns),
            require_non_empty=False,
        )
        self.fidelity_bucket_columns = list(
            dict.fromkeys([*self.fidelity_1d_columns, *self.fidelity_2d_anchor_columns])
        )
        if privacy_columns is None:
            self.privacy_columns = list(self.feature_columns)
            privacy_encoding_columns = list(self.column_order)
        else:
            self.privacy_columns = self._resolve_guided_column_scope(
                "privacy_columns",
                privacy_columns,
                default_columns=list(self.feature_columns),
                allow_target_when_guided=bool(allow_target_in_privacy_columns),
                require_non_empty=True,
            )
            if str(privacy_encoding_column_mode) == "column_order":
                privacy_encoding_columns = list(self.column_order)
            else:
                privacy_encoding_columns = list(self.privacy_columns)
        self.privacy_encoding_columns = privacy_encoding_columns
        self.utility_balance_column = self._resolve_guided_single_feature(
            "utility_balance_column",
            utility_balance_column,
        )
        self.guided_mode = any(
            value is not None
            for value in (
                fidelity_1d_columns,
                fidelity_2d_anchor_columns,
                privacy_columns,
                utility_balance_column,
                max_theta_pairs,
            )
        )
        self.numeric_columns = [
            c
            for c in self.column_order
            if schema_card["columns"][c]["type"] in {"numerical", "discrete_numerical"}
        ]
        self.categorical_columns = [c for c in self.column_order if schema_card["columns"][c]["type"] == "categorical"]
        self.numeric_impute_values = self._build_numeric_impute_values()
        self.privacy_numeric_columns = [
            c
            for c in self.privacy_encoding_columns
            if schema_card["columns"][c]["type"] in {"numerical", "discrete_numerical"}
        ]
        self.privacy_categorical_columns = [
            c for c in self.privacy_encoding_columns if schema_card["columns"][c]["type"] == "categorical"
        ]
        self.privacy_numeric_impute_values = {
            column: self.numeric_impute_values[column] for column in self.privacy_numeric_columns
        }
        use_high_cardinality = (
            str(schema_card.get("dataset", "")).lower() == "diabetes"
            if high_cardinality_enabled is None
            else bool(high_cardinality_enabled)
        )
        self.high_cardinality_compressor = HighCardinalityCompressor(
            enabled=use_high_cardinality,
            threshold=high_cardinality_threshold,
            top_k=high_cardinality_top_k,
            tail_cluster_count=high_cardinality_tail_clusters,
        ).fit(
            self.train_df,
            categorical_columns=self.categorical_columns,
            target_column=self.target_column,
        )
        self.search_train_df = self.high_cardinality_compressor.transform_df(self.train_df)
        self.search_holdout_df = self.high_cardinality_compressor.transform_df(self.holdout_df)

        self.train_distributions = self._build_train_distributions()
        self.pair_marginal_edges = self._build_pair_marginals()
        self.num_fidelity_columns = max(len(self.fidelity_columns), 1)
        self.pair_weights = np.asarray(
            [max(float(edge.get("mi", 0.0)), 1e-6) for edge in self.pair_marginal_edges],
            dtype=float,
        )
        self.total_pair_weight = max(float(self.pair_weights.sum()), 1.0)
        self._fit_privacy_encoder()
        self._fit_density_reference()
        self._fit_gate_rarity_reference()
        self.last_preselect_report: dict[str, Any] = {}
        self._record_bucket_pair_cache: dict[tuple[int, int], tuple[dict[str, np.ndarray], list[np.ndarray]]] = {}
        self._record_target_counts_cache: dict[tuple[int, int], tuple[dict[str, np.ndarray], list[np.ndarray]]] = {}
