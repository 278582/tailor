from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SELECTION_PREFIX = "selection_"


@dataclass(frozen=True)
class RunInputs:
    run_dir: Path
    train_csv: Path
    control_csv: Path
    reference_csv: Path
    versions_dir: Path
    context: dict[str, Any]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["?", ""], keep_default_na=True)
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    return df


def save_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def resolve_run_inputs(run_dir: Path, *, reference_split: str = "test") -> RunInputs:
    run_dir = Path(run_dir)
    input_dir = run_dir / "input"
    versions_dir = run_dir / "versions"
    train_csv = input_dir / "eval_train.csv"
    control_csv = input_dir / "eval_holdout.csv"
    if reference_split == "holdout":
        reference_csv = control_csv
    elif reference_split == "test":
        reference_csv = input_dir / "eval_test.csv"
    else:
        raise ValueError(f"Unsupported reference_split={reference_split!r}; expected holdout or test")

    missing = [path for path in (train_csv, control_csv, reference_csv) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required MIA input split(s): {missing_text}")
    if not versions_dir.exists():
        raise FileNotFoundError(f"Missing versions directory: {versions_dir}")

    context_path = input_dir / "selection_context.json"
    context = load_json(context_path) if context_path.exists() else {}
    return RunInputs(
        run_dir=run_dir,
        train_csv=train_csv,
        control_csv=control_csv,
        reference_csv=reference_csv,
        versions_dir=versions_dir,
        context=context,
    )


def selection_name_from_path(path: Path) -> str:
    stem = Path(path).stem
    if stem.startswith(SELECTION_PREFIX):
        return stem[len(SELECTION_PREFIX) :]
    return stem


def list_selection_csvs(versions_dir: Path) -> list[Path]:
    paths = sorted(Path(versions_dir).glob("selection_*.csv"))
    return [path for path in paths if path.is_file()]


def find_selection_csv(versions_dir: Path, selection_name: str) -> Path | None:
    direct = Path(versions_dir) / f"{SELECTION_PREFIX}{selection_name}.csv"
    if direct.exists():
        return direct
    for path in list_selection_csvs(versions_dir):
        if selection_name_from_path(path) == selection_name:
            return path
    return None


def infer_target_column(context: dict[str, Any], columns: list[str]) -> str | None:
    target = context.get("target_column")
    if isinstance(target, str) and target in columns:
        return target
    return None

