from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def link_or_copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if src.exists() and dst.exists() and os.path.samefile(src, dst):
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def remove_known_mirror(path: Path, filenames: tuple[str, ...]) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if not path.exists() or not path.is_dir():
        return
    for filename in filenames:
        child = path / filename
        if child.is_symlink() or child.is_file():
            child.unlink()
    try:
        path.rmdir()
    except OSError:
        pass


def save_shared_csv(path: Path, df: pd.DataFrame, shared_path: Path | None = None) -> None:
    if shared_path is None:
        save_csv(path, df)
        return
    if not shared_path.exists():
        save_csv(shared_path, df)
    link_or_copy_file(shared_path, path)


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_shared_json(path: Path, payload: dict[str, Any] | list[Any], shared_path: Path | None = None) -> None:
    if shared_path is None:
        save_json(path, payload)
        return
    if not shared_path.exists():
        save_json(shared_path, payload)
    link_or_copy_file(shared_path, path)


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def df_to_candidate_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{"candidate_id": int(idx), "row": row.to_dict()} for idx, row in df.reset_index(drop=True).iterrows()]


def records_to_df(records: list[dict[str, Any]], column_order: list[str]) -> pd.DataFrame:
    return pd.DataFrame([record["row"] for record in records], columns=column_order)
