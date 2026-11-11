from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_string(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["?", ""], keep_default_na=True)
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].map(normalize_string)
    return df


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=True)
        fp.write("\n")


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=True))
            fp.write("\n")


def save_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.to_dict(orient="records")
