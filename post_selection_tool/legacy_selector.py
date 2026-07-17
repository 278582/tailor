from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .encoding import _js_divergence
from .pareto_selection import _non_dominated_sort


class LegacySelectorMixin:
    """Compatibility methods kept out of the main ParetoSelector surface."""

    def compute_column_jsd(self, df: pd.DataFrame) -> dict[str, float]:
        if df.empty:
            return {column: 0.0 for column in self.fidelity_columns}
        bucket_indices_map = self._column_bucket_indices_for_df(df, self.fidelity_columns)
        report: dict[str, float] = {}
        for column in self.fidelity_columns:
            train_dist = self.train_distributions[column]
            counts = self._column_counts_from_bucket_indices(column, bucket_indices_map[column])
            report[column] = _js_divergence(counts, train_dist["probs"])
        return report

    def reduce_archive(
        self,
        archive_records: list[dict[str, Any]],
        archive_exact_records: list[dict[str, Any]],
        budget: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        if len(archive_records) <= budget:
            return archive_records, archive_exact_records, {
                "archive_rows_before_reduction": len(archive_records),
                "archive_rows_after_reduction": len(archive_records),
                "reduction_applied": False,
                "secondary_filter": {"applied": False},
            }

        points = self._pareto_points(archive_exact_records)
        fronts = _non_dominated_sort(points)
        front_rank, crowding = self._front_rank_and_crowding(points, fronts)
        priority = self._candidate_priority(
            points,
            archive_exact_records,
            fronts,
            front_rank=front_rank,
            crowding=crowding,
        )
        selected_indices, front_summaries, fronts = self._select_indices_by_nsga(
            points,
            archive_exact_records,
            budget,
            fronts=fronts,
            front_rank=front_rank,
            crowding=crowding,
        )
        if len(selected_indices) > budget:
            selected_indices = selected_indices[:budget]
        selected_indices, secondary_report = self._secondary_rarity_reduce(
            selected_indices=selected_indices,
            exact_records=archive_exact_records,
            candidate_records=archive_records,
            points=points,
            fronts=fronts,
            budget=budget,
            priority=priority,
        )
        reduced_records = [archive_records[idx] for idx in selected_indices]
        reduced_exact = [archive_exact_records[idx] for idx in selected_indices]
        return reduced_records, reduced_exact, {
            "archive_rows_before_reduction": len(archive_records),
            "archive_rows_after_reduction": len(reduced_records),
            "reduction_applied": True,
            "point_dimension": int(points.shape[1]) if len(archive_exact_records) > 0 else 0,
            "fronts": front_summaries,
            "secondary_filter": secondary_report,
        }

    def select_keep_chebyshev(
        self,
        preselected_records: list[dict[str, Any]],
        exact_records: list[dict[str, Any]],
        keep_k: int,
        fidelity_1d_weight: float,
        fidelity_2d_weight: float,
        privacy_weight: float,
        utility_weight: float = 0.0,
        floor_reference: dict[str, Any] | None = None,
        constraint_reference_records: list[dict[str, Any]] | None = None,
        floor_mode: str = "hard",
        soft_fidelity_floor_eps: float | None = None,
        soft_trend_floor_eps: float | None = None,
        soft_privacy_floor_eps: float = 0.005,
        soft_utility_floor_eps: float = 0.005,
        soft_min_score_delta: float = 0.0,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
        if not preselected_records or not exact_records:
            return (
                pd.DataFrame(columns=self.column_order),
                [],
                {
                    "selected": 0,
                    "keep_k": keep_k,
                    "mode": "chebyshev",
                    "source": self.source,
                },
            )

        keep_k = min(keep_k, len(exact_records))
        if floor_reference is not None:
            fidelity_guard = {"applied": False, "mode": "disabled_for_constrained_subset"}
            objective = self._selection_objective_components(
                exact_records,
                mode="chebyshev",
                fidelity_1d_weight=fidelity_1d_weight,
                fidelity_2d_weight=fidelity_2d_weight,
                privacy_weight=privacy_weight,
                utility_proxy_weight=utility_weight,
            )
            chebyshev_scores = np.asarray(objective.get("chebyshev_scores", np.zeros(len(exact_records), dtype=float)), dtype=float)
            selected_indices, floor_repair_report = self._construct_constrained_keep_subset(
                preselected_records=preselected_records,
                exact_records=exact_records,
                keep_k=keep_k,
                mode="chebyshev",
                floor_reference=floor_reference,
                constraint_reference_records=constraint_reference_records,
                fidelity_1d_weight=fidelity_1d_weight,
                fidelity_2d_weight=fidelity_2d_weight,
                privacy_weight=privacy_weight,
                utility_weight=utility_weight,
                floor_mode=floor_mode,
                soft_fidelity_floor_eps=soft_fidelity_floor_eps,
                soft_trend_floor_eps=soft_trend_floor_eps,
                soft_privacy_floor_eps=soft_privacy_floor_eps,
                soft_utility_floor_eps=soft_utility_floor_eps,
                soft_min_score_delta=soft_min_score_delta,
            )
        else:
            preselected_records, exact_records, fidelity_guard = self._fidelity_guard_subset(
                preselected_records=preselected_records,
                exact_records=exact_records,
                keep_k=keep_k,
                mode="matched",
            )
            keep_k = min(keep_k, len(exact_records))
            points = self._pareto_points(exact_records)
            objective = self._selection_objective_components(
                exact_records,
                mode="chebyshev",
                fidelity_1d_weight=fidelity_1d_weight,
                fidelity_2d_weight=fidelity_2d_weight,
                privacy_weight=privacy_weight,
                utility_proxy_weight=utility_weight,
            )
            chebyshev_scores = np.asarray(
                objective.get("chebyshev_scores", np.zeros(len(exact_records), dtype=float)),
                dtype=float,
            )
            fronts = _non_dominated_sort(points)
            front_rank = self._build_front_rank_map(fronts)
            ordered = sorted(
                range(len(exact_records)),
                key=lambda idx: (
                    front_rank.get(idx, len(fronts) + 1),
                    float(chebyshev_scores[idx]),
                    float(-exact_records[idx].get("pareto_util_proxy_obj", 0.0)),
                    float(-exact_records[idx].get("pareto_fid_2d_obj", 0.0)),
                    float(-exact_records[idx].get("pareto_fid_1d_obj", 0.0)),
                ),
            )
            selected_indices = ordered[:keep_k]
            selected_indices, floor_repair_report = self._apply_exact_floor_repair(
                preselected_records=preselected_records,
                exact_records=exact_records,
                selected_indices=selected_indices,
                keep_k=keep_k,
                floor_reference=floor_reference,
            )
        keep_records = [preselected_records[idx] for idx in selected_indices]
        selected_exact_records = [exact_records[idx] for idx in selected_indices]
        keep_df = pd.DataFrame([record["row"] for record in keep_records], columns=self.column_order)
        return keep_df, keep_records, {
            "selected": len(keep_records),
            "keep_k": keep_k,
            "mode": "chebyshev",
            "source": self.source,
            "fidelity_1d_weight": fidelity_1d_weight,
            "fidelity_2d_weight": fidelity_2d_weight,
            "privacy_weight": privacy_weight,
            "utility_weight": utility_weight,
            "point_dimension": int(self._pareto_points(exact_records).shape[1]) if len(exact_records) > 0 else 0,
            "floor_reference_name": floor_reference.get("name") if floor_reference is not None else None,
            "best_chebyshev_score": float(chebyshev_scores[selected_indices[0]]) if selected_indices else 0.0,
            "selected_privacy_component_mean": (
                float(np.mean([float(record.get("pareto_priv_obj", 0.0)) for record in selected_exact_records]))
                if selected_exact_records
                else 0.0
            ),
            "selected_privacy_raw_mean": (
                float(np.mean([float(record.get("privacy_score_selected", 0.0)) for record in selected_exact_records]))
                if selected_exact_records
                else 0.0
            ),
            "selected_utility_mean": (
                float(np.mean([float(record.get("pareto_util_proxy_obj", 0.0)) for record in selected_exact_records]))
                if selected_exact_records
                else 0.0
            ),
            "fidelity_guard": fidelity_guard,
            "exact_floor_repair": floor_repair_report,
        }

    def _candidate_ids(self, records: list[dict[str, Any]]) -> list[int]:
        return [int(record.get("candidate_id", idx)) for idx, record in enumerate(records)]

    def compute_rarity_stratified_keep_rate(
        self,
        candidate_records: list[dict[str, Any]],
        keep_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidate_records:
            return []

        candidate_ids = np.asarray(self._candidate_ids(candidate_records), dtype=int)
        keep_ids = np.asarray(self._candidate_ids(keep_records), dtype=int)
        if candidate_records and "gate_stratum" in candidate_records[0]:
            strata = np.asarray([int(record["gate_stratum"]) for record in candidate_records], dtype=int)
        else:
            candidate_df = pd.DataFrame([record["row"] for record in candidate_records], columns=self.column_order)
            gate_probs = self._prob_geomean_for_df(candidate_df, columns=self.feature_columns)
            strata = self._assign_bins_from_edges(gate_probs, self.train_gate_edges)
        num_strata = len(self.train_gate_edges) - 1
        candidate_rows = np.bincount(strata, minlength=num_strata).astype(int)
        selected_mask = np.isin(candidate_ids, keep_ids, assume_unique=False)
        selected_rows = np.bincount(strata[selected_mask], minlength=num_strata).astype(int)
        return [
            {
                "stratum": stratum,
                "prob_low": float(self.train_gate_edges[stratum]),
                "prob_high": float(self.train_gate_edges[stratum + 1]),
                "candidate_rows": int(candidate_rows[stratum]),
                "selected_rows": int(selected_rows[stratum]),
                "keep_rate": float(selected_rows[stratum] / candidate_rows[stratum]) if candidate_rows[stratum] else 0.0,
            }
            for stratum in range(num_strata)
        ]

    def compute_rare_bin_inflation(self, selected_df: pd.DataFrame, rare_threshold: float = 0.05) -> list[dict[str, Any]]:
        if selected_df.empty:
            return []
        relevant_columns = [
            column
            for column in self.fidelity_columns
            if self.schema_card["columns"][column]["type"] in {"categorical", "discrete_numerical"}
        ]
        if not relevant_columns:
            return []

        bucket_indices_map = self._column_bucket_indices_for_df(selected_df, relevant_columns)
        frames: list[pd.DataFrame] = []
        for column in relevant_columns:
            train_dist = self.train_distributions[column]
            train_probs = np.asarray(train_dist["probs"], dtype=float)
            rare_mask = train_probs <= float(rare_threshold)
            if not np.any(rare_mask):
                continue

            selected_counts = self._column_counts_from_bucket_indices(column, bucket_indices_map[column])
            selected_probs = selected_counts / max(float(selected_counts.sum()), 1.0)
            rare_indices = np.flatnonzero(rare_mask)
            values = np.asarray(train_dist["values"], dtype=object)
            frames.append(
                pd.DataFrame(
                    {
                        "column": column,
                        "value": values[rare_indices],
                        "train_prob": train_probs[rare_indices],
                        "selected_prob": selected_probs[rare_indices],
                    }
                )
            )

        if not frames:
            return []

        report_df = pd.concat(frames, axis=0, ignore_index=True)
        report_df["inflation_diff"] = report_df["selected_prob"] - report_df["train_prob"]
        report_df["inflation_ratio"] = report_df["selected_prob"] / np.clip(
            report_df["train_prob"].to_numpy(dtype=float, copy=False),
            1e-12,
            None,
        )
        report_df = report_df.sort_values(
            by=["inflation_diff", "column", "value"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        return report_df.to_dict(orient="records")

    def compute_smoke_metrics(
        self,
        surrogate_records_all: list[dict[str, Any]],
        keep_records: list[dict[str, Any]],
        keep_df: pd.DataFrame,
    ) -> dict[str, Any]:
        if not surrogate_records_all:
            return {
                "pass": False,
                "reason": "no_valid_candidates",
                "source": self.source,
            }

        all_surrogates_df = pd.DataFrame.from_records(surrogate_records_all)
        all_fid = float(all_surrogates_df["s_fid_sur"].mean())
        all_priv = float(all_surrogates_df["s_priv_sur"].mean())
        if keep_df.empty:
            return {
                "pass": False,
                "reason": "no_keep_candidates",
                "source": self.source,
                "avg_bin_hit_all": all_fid,
                "avg_nn_distance_all": all_priv,
            }

        keep_surrogates_df: pd.DataFrame | None = None
        if "candidate_id" in all_surrogates_df.columns:
            keep_ids = np.asarray(self._candidate_ids(keep_records), dtype=int)
            if keep_ids.size > 0:
                matched = all_surrogates_df[all_surrogates_df["candidate_id"].isin(keep_ids)]
                if len(matched) == len(keep_ids):
                    keep_surrogates_df = matched

        if keep_surrogates_df is None:
            keep_surrogates_df = pd.DataFrame.from_records(self.compute_surrogates(keep_df))

        keep_fid = float(keep_surrogates_df["s_fid_sur"].mean())
        keep_priv = float(keep_surrogates_df["s_priv_sur"].mean())
        fid_improved = keep_fid > all_fid
        priv_improved = keep_priv > all_priv
        fid_degradation = max(0.0, (all_fid - keep_fid) / max(all_fid, 1e-12))
        priv_degradation = max(0.0, (all_priv - keep_priv) / max(all_priv, 1e-12))
        smoke_pass = (fid_improved and priv_degradation <= 0.05) or (priv_improved and fid_degradation <= 0.05)
        return {
            "source": self.source,
            "pass": smoke_pass,
            "avg_bin_hit_all": all_fid,
            "avg_bin_hit_keep": keep_fid,
            "avg_nn_distance_all": all_priv,
            "avg_nn_distance_keep": keep_priv,
            "avg_fidelity_sur_all": all_fid,
            "avg_fidelity_sur_keep": keep_fid,
            "avg_privacy_sur_all": all_priv,
            "avg_privacy_sur_keep": keep_priv,
            "fid_improved": fid_improved,
            "priv_improved": priv_improved,
            "fid_degradation_ratio": fid_degradation,
            "priv_degradation_ratio": priv_degradation,
        }
