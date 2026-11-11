from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ModelSpec:
    key: str
    display_name: str
    path: Path


@dataclass
class CardsBundle:
    schema_card: dict[str, Any]
    stats_card: dict[str, Any]
    prototype_entries: list[dict[str, Any]]
    residual_card: dict[str, Any]


@dataclass
class GenerationBundle:
    raw_generations: list[dict[str, Any]]
    candidate_records: list[dict[str, Any]]


@dataclass
class ValidationBundle:
    valid_df: pd.DataFrame
    valid_records: list[dict[str, Any]]
    rejected_records: list[dict[str, Any]]
    report: dict[str, Any]


@dataclass
class ParetoBundle:
    d_cur_df: pd.DataFrame
    surrogate_records: list[dict[str, Any]]
    exact_records: list[dict[str, Any]]
    keep_df: pd.DataFrame
    keep_records: list[dict[str, Any]]
    nsga_report: dict[str, Any]
    smoke_metrics: dict[str, Any]
