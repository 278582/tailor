from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


CORE_SELECTION_FILES = {
    "preselected_valid": "selection_preselected_valid.csv",
    "preselected_fidelity_ceiling_keep_k": "selection_preselected_fidelity_ceiling_keep_k.csv",
    "random_full": "selection_random_full.csv",
    "scalar": "selection_scalar.csv",
    "pareto": "selection_pareto.csv",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_core_selection_frames(versions_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, filename in CORE_SELECTION_FILES.items():
        path = versions_dir / filename
        if path.exists():
            frames[name] = pd.read_csv(path)
    return frames


def save_eval_extras(eval_dir: Path, selection_name: str, extras: dict[str, Any]) -> None:
    target_dir = ensure_dir(eval_dir / selection_name)
    for key, value in extras.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(target_dir / f"{key}.csv", index=False)
        elif isinstance(value, dict):
            save_json(target_dir / f"{key}.json", value)
        elif key in {"dcr_real", "dcr_test"}:
            continue
        else:
            save_json(target_dir / f"{key}.json", {"value": value})
    if "dcr_real" in extras and "dcr_test" in extras:
        pd.DataFrame({"dcr_real": extras["dcr_real"], "dcr_test": extras["dcr_test"]}).to_csv(
            target_dir / "dcr.csv",
            index=False,
        )
