from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SELECTION_PREFIX = "selection_"
_HOME_PREFIX = "/home/liuzhiwei"
_LUSTRE_PREFIX = "/mnt/lustre/liuzhiwei"


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


def _alt_mount_paths(path: Path) -> list[Path]:
    text = str(path)
    alts = [path]
    for src, dst in ((_HOME_PREFIX, _LUSTRE_PREFIX), (_LUSTRE_PREFIX, _HOME_PREFIX)):
        if text == src or text.startswith(src + "/"):
            alts.append(Path(dst + text[len(src) :]))
    return alts


def existing_file(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    candidates = list(_alt_mount_paths(path))
    try:
        if path.is_symlink():
            target = Path(os.readlink(str(path)))
            if not target.is_absolute():
                target = path.parent / target
            candidates.extend(_alt_mount_paths(target))
    except OSError:
        pass
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _synthetic_split_path(context: dict[str, Any], filename: str) -> Path | None:
    dataset = context.get("dataset_name") or context.get("logical_name")
    if not isinstance(dataset, str) or not dataset:
        return None
    return Path.cwd() / "synthetic" / dataset / filename


def _resolve_split(*candidates: Path | str | None) -> Path | None:
    for candidate in candidates:
        found = existing_file(candidate)
        if found is not None:
            return found
    return None


def resolve_run_inputs(run_dir: Path, *, reference_split: str = "test") -> RunInputs:
    run_dir = Path(run_dir)
    input_dir = run_dir / "input"
    versions_dir = run_dir / "versions"
    context_path = input_dir / "selection_context.json"
    context = load_json(context_path) if context_path.exists() else {}

    train_csv = _resolve_split(
        input_dir / "eval_train.csv",
        context.get("train_source_path"),
        _synthetic_split_path(context, "train.csv"),
    )
    control_csv = _resolve_split(
        input_dir / "eval_holdout.csv",
        context.get("holdout_source_path"),
        _synthetic_split_path(context, "hold.csv"),
    )
    if reference_split == "holdout":
        reference_csv = control_csv
    elif reference_split == "test":
        reference_csv = _resolve_split(
            input_dir / "eval_test.csv",
            context.get("test_source_path"),
            _synthetic_split_path(context, "test.csv"),
        )
    else:
        raise ValueError(f"Unsupported reference_split={reference_split!r}; expected holdout or test")

    missing_labels = [
        name
        for name, path in (
            ("eval_train.csv", train_csv),
            ("eval_holdout.csv", control_csv),
            ("eval_test.csv" if reference_split == "test" else "eval_holdout.csv", reference_csv),
        )
        if path is None
    ]
    if missing_labels:
        missing_text = ", ".join(str(input_dir / name) for name in missing_labels)
        raise FileNotFoundError(f"Missing required MIA input split(s): {missing_text}")
    if not versions_dir.exists():
        raise FileNotFoundError(f"Missing versions directory: {versions_dir}")

    assert train_csv is not None
    assert control_csv is not None
    assert reference_csv is not None
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

