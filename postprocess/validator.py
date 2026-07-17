from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .types import ValidationBundle

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def _progress(iterable: Any, *, total: int, desc: str, disable: bool) -> Any:
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, disable=disable)


class TabularValidator:
    def __init__(self, schema_card: dict[str, Any], stats_card: dict[str, Any]) -> None:
        self.schema_card = schema_card
        self.stats_card = stats_card
        self.column_order = schema_card["column_order"]
        self.column_set = set(self.column_order)
        self.column_infos = schema_card["columns"]
        self.discrete_legal_arrays: dict[str, np.ndarray] = {}
        self.discrete_legal_values: dict[str, list[Any]] = {}
        self.categorical_legal_sets: dict[str, set[Any]] = {}
        self.categorical_lowered_maps: dict[str, dict[str, Any]] = {}
        self.categorical_numeric_maps: dict[str, dict[str, Any]] = {}
        for column in self.column_order:
            info = self.column_infos[column]
            if info["type"] == "discrete_numerical":
                legal_values = list(info.get("legal_values", []))
                self.discrete_legal_values[column] = legal_values
                self.discrete_legal_arrays[column] = np.asarray([float(value) for value in legal_values], dtype=float)
            elif info["type"] == "categorical":
                raw_legal_values = list(info.get("legal_values", []))
                legal_values = set(raw_legal_values)
                self.categorical_legal_sets[column] = legal_values
                self.categorical_lowered_maps[column] = {str(candidate).lower(): candidate for candidate in legal_values}
                numeric_map: dict[str, Any] = {}
                for candidate in raw_legal_values:
                    token = self._numeric_category_token(candidate)
                    if token is not None:
                        numeric_map.setdefault(token, candidate)
                self.categorical_numeric_maps[column] = numeric_map

    def _repair_discrete_numeric(
        self,
        value: Any,
        column: str,
    ) -> int | float:
        legal_values = self.discrete_legal_values[column]
        legal_array = self.discrete_legal_arrays[column]
        value_float = float(value)
        nearest_idx = int(np.argmin(np.abs(legal_array - value_float)))
        repaired = legal_values[nearest_idx]
        if float(repaired).is_integer():
            return int(repaired)
        return float(repaired)

    def _is_missing_category(self, value: Any) -> bool:
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except Exception:
            pass
        if isinstance(value, str) and value.strip().lower() in {"", "na", "nan", "none", "null"}:
            return True
        return False

    def _numeric_category_token(self, value: Any) -> str | None:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                numeric = float(text)
            except Exception:
                return None
        else:
            try:
                numeric = float(value)
            except Exception:
                return None
        if not np.isfinite(numeric):
            return None
        rounded = round(numeric)
        if np.isclose(numeric, rounded, rtol=0.0, atol=1e-12):
            return str(int(rounded))
        return None

    def _fallback_category(self, column: str) -> str:
        top_values = self.stats_card.get("categorical_top_values", {}).get(column, [])
        if top_values:
            return str(top_values[0]["value"])
        legal_values = self.schema_card["columns"][column].get("legal_values", [])
        if legal_values:
            return str(legal_values[0])
        return "NA"

    def _validate_one(
        self,
        record: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
        row = record["row"]
        row_keys = list(row.keys())
        if len(row_keys) != len(set(row_keys)):
            return None, {"candidate_id": record["candidate_id"], "reason": "duplicate_columns", "row": row}, []
        missing = [col for col in self.column_order if col not in row]
        unknown = [col for col in row.keys() if col not in self.column_set]
        if missing:
            return None, {"candidate_id": record["candidate_id"], "reason": f"missing_columns:{missing}", "row": row}, []
        if unknown:
            return None, {"candidate_id": record["candidate_id"], "reason": f"unknown_columns:{unknown}", "row": row}, []

        repaired: dict[str, Any] = {}
        repair_actions: list[str] = []
        for column in self.column_order:
            info = self.column_infos[column]
            value = row[column]
            if isinstance(value, str):
                value = value.strip()
            if info["type"] == "numerical":
                try:
                    value = float(value)
                except Exception:
                    return (
                        None,
                        {"candidate_id": record["candidate_id"], "reason": f"invalid_numeric:{column}", "row": row},
                        [],
                    )
                stats = self.stats_card["numeric_stats"][column]
                clipped = min(max(value, stats["min"]), stats["max"])
                if clipped != value:
                    repair_actions.append(f"numeric_clip:{column}")
                value = clipped
                repaired[column] = value
            elif info["type"] == "discrete_numerical":
                try:
                    value = float(value)
                except Exception:
                    return (
                        None,
                        {
                            "candidate_id": record["candidate_id"],
                            "reason": f"invalid_discrete_numeric:{column}",
                            "row": row,
                        },
                        [],
                    )
                repaired_value = self._repair_discrete_numeric(value=value, column=column)
                if float(repaired_value) != float(value):
                    repair_actions.append(f"discrete_snap:{column}")
                repaired[column] = repaired_value
            else:
                legal_values = self.categorical_legal_sets[column]
                if self._is_missing_category(value):
                    repaired[column] = self._fallback_category(column)
                    repair_actions.append(f"missing_fill:{column}")
                    continue
                normalized = str(value).strip()
                if normalized not in legal_values:
                    lowered_map = self.categorical_lowered_maps[column]
                    if normalized.lower() in lowered_map:
                        normalized = lowered_map[normalized.lower()]
                        repair_actions.append(f"category_case_normalize:{column}")
                    else:
                        numeric_map = self.categorical_numeric_maps[column]
                        numeric_token = self._numeric_category_token(value)
                        if numeric_token is not None and numeric_token in numeric_map:
                            normalized = numeric_map[numeric_token]
                            repair_actions.append(f"category_numeric_normalize:{column}")
                        else:
                            return (
                                None,
                                {
                                    "candidate_id": record["candidate_id"],
                                    "reason": f"invalid_category:{column}",
                                    "row": row,
                                },
                                [],
                            )
                repaired[column] = normalized

        valid_record = dict(record)
        valid_record["row"] = repaired
        if repair_actions:
            valid_record["repair_actions"] = repair_actions
        return valid_record, None, repair_actions

    def validate(
        self,
        candidate_records: list[dict[str, Any]],
        show_progress: bool = False,
        progress_desc: str = "validate candidates",
    ) -> ValidationBundle:
        valid_records: list[dict[str, Any]] = []
        rejected_records: list[dict[str, Any]] = []
        repaired_records = 0
        repair_action_hist: dict[str, int] = {}
        record_iter = _progress(
            candidate_records,
            total=len(candidate_records),
            desc=progress_desc,
            disable=not show_progress,
        )
        for record in record_iter:
            valid, rejected, repair_actions = self._validate_one(record)
            if valid is not None:
                valid_records.append(valid)
                if repair_actions:
                    repaired_records += 1
                    for action in repair_actions:
                        repair_action_hist[action] = repair_action_hist.get(action, 0) + 1
            if rejected is not None:
                rejected_records.append(rejected)

        valid_df = pd.DataFrame([record["row"] for record in valid_records], columns=self.column_order)
        report = {
            "total_candidates": len(candidate_records),
            "num_valid": len(valid_records),
            "num_rejected": len(rejected_records),
            "reject_rate": (len(rejected_records) / len(candidate_records)) if candidate_records else 0.0,
            "num_repaired": repaired_records,
            "repair_rate": (repaired_records / len(candidate_records)) if candidate_records else 0.0,
            "repair_action_histogram": repair_action_hist,
        }
        return ValidationBundle(
            valid_df=valid_df,
            valid_records=valid_records,
            rejected_records=rejected_records,
            report=report,
        )
