from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from postprocess.tabdiff_protocol import normalize_tabdiff_dataframe_columns

from .io import ensure_dir, save_csv, save_json, save_jsonl


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload if isinstance(payload, dict) else {}


def _source_path_from_manifest(
    *,
    manifest: dict[str, Any],
    dataset_name: str,
    source_id: str,
    sample_root: Path,
    sample_id: int = 0,
) -> Path:
    if sample_id < 0:
        raise ValueError(f"sample_id must be non-negative, got {sample_id}")
    if sample_id != 0:
        preferred = Path(sample_root) / source_id / dataset_name / f"sample_{sample_id}.csv"
        if preferred.exists():
            return preferred
        raise FileNotFoundError(f"Cannot find synthetic sample for source={source_id}: {preferred}")

    source_validation = manifest.get("source_validation")
    if isinstance(source_validation, dict):
        source_report = source_validation.get(source_id)
        if isinstance(source_report, dict):
            source_path = source_report.get("source_path")
            if source_path:
                path = Path(str(source_path))
                if path.exists():
                    return path
    preferred = Path(sample_root) / source_id / dataset_name / "sample_0.csv"
    if preferred.exists():
        return preferred
    legacy = Path(sample_root) / source_id / dataset_name / "samples_0.csv"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Cannot find synthetic sample for source={source_id}: {preferred}")


def _pool_units_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for item in manifest.get("pool_units", []) or []:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip().lower()
        if not source_id:
            continue
        try:
            multiplier = int(round(float(item.get("multiplier", 0))))
        except (TypeError, ValueError):
            continue
        if multiplier <= 0:
            continue
        units.append({"source_id": source_id, "multiplier": multiplier})
    return units


def build_theta_synthetic_pool_from_manifest(
    *,
    manifest_path: Path,
    dataset_name: str,
    train_rows: int,
    seed: int,
    output_dir: Path,
    sample_root: Path,
    sample_id: int = 0,
) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = Path(manifest_path)
    manifest = _load_json(manifest_path)
    pool_units = _pool_units_from_manifest(manifest)
    if not pool_units:
        raise ValueError(f"No valid pool_units found in {manifest_path}")

    output_dir = ensure_dir(Path(output_dir))
    frames: list[pd.DataFrame] = []
    row_map: list[dict[str, Any]] = []
    source_paths: dict[str, str] = {}
    pool_row_id = 0

    if len(pool_units) == 1:
        source_id = pool_units[0]["source_id"]
        source_path = _source_path_from_manifest(
            manifest=manifest,
            dataset_name=dataset_name,
            source_id=source_id,
            sample_root=sample_root,
            sample_id=sample_id,
        )
        source_paths[source_id] = str(source_path)
        source_df = normalize_tabdiff_dataframe_columns(dataset_name, pd.read_csv(source_path))
        synthetic_csv = output_dir / "theta_s_pool.csv"
        row_map_path = output_dir / "theta_s_pool_rows.jsonl"
        save_csv(synthetic_csv, source_df.reset_index(drop=True))
        save_jsonl(
            row_map_path,
            [
                {
                    "pool_row_id": int(idx),
                    "source_id": source_id,
                    "source_row_id": int(source_idx),
                    "draw_index": 0,
                    "draw_local_index": int(idx),
                    "sample_seed": None,
                    "with_replacement": False,
                    "sampling_mode": "single_full_source_no_random_sampling",
                }
                for idx, source_idx in enumerate(source_df.index.tolist())
            ],
        )
        report = {
            "mode": "llm_mcts_theta_s_pool_rebuild",
            "s_id": manifest.get("s_id"),
            "manifest_path": str(manifest_path),
            "rows": int(len(source_df)),
            "target_rows": int(len(source_df)),
            "pool_units": pool_units,
            "source_counts": {source_id: int(len(source_df))},
            "source_paths": source_paths,
            "sample_id": int(sample_id),
            "sampling_mode": "single_full_source_no_random_sampling",
        }
        save_json(output_dir / "theta_s_pool_manifest.json", report)
        return synthetic_csv, row_map_path, report

    for draw_index, unit in enumerate(pool_units):
        source_id = str(unit["source_id"])
        multiplier = int(unit["multiplier"])
        rows_to_draw = max(1, int(multiplier) * int(train_rows))
        source_path = _source_path_from_manifest(
            manifest=manifest,
            dataset_name=dataset_name,
            source_id=source_id,
            sample_root=sample_root,
            sample_id=sample_id,
        )
        source_paths[source_id] = str(source_path)
        source_df = normalize_tabdiff_dataframe_columns(dataset_name, pd.read_csv(source_path))
        replace = rows_to_draw > len(source_df)
        sample_seed = int(seed + 997 * (draw_index + 1) + sum(ord(ch) for ch in source_id))
        sampled = source_df.sample(n=rows_to_draw, replace=replace, random_state=sample_seed)
        source_indices = list(sampled.index)
        sampled = sampled.reset_index(drop=True)
        frames.append(sampled)
        for local_idx, source_row_id in enumerate(source_indices):
            row_map.append(
                {
                    "pool_row_id": int(pool_row_id),
                    "source_id": source_id,
                    "source_row_id": int(source_row_id),
                    "draw_index": int(draw_index),
                    "draw_local_index": int(local_idx),
                    "sample_seed": int(sample_seed),
                    "with_replacement": bool(replace),
                }
            )
            pool_row_id += 1

    pool_df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    target_multiplier = sum(int(unit["multiplier"]) for unit in pool_units)
    target_rows = max(1, int(target_multiplier) * int(train_rows))
    if len(pool_df) > target_rows:
        pool_df = pool_df.iloc[:target_rows].reset_index(drop=True)
        row_map = row_map[:target_rows]

    synthetic_csv = output_dir / "theta_s_pool.csv"
    row_map_path = output_dir / "theta_s_pool_rows.jsonl"
    save_csv(synthetic_csv, pool_df)
    save_jsonl(row_map_path, row_map)
    report = {
        "mode": "llm_mcts_theta_s_pool_rebuild",
        "s_id": manifest.get("s_id"),
        "manifest_path": str(manifest_path),
        "rows": int(len(pool_df)),
        "target_rows": int(target_rows),
        "pool_units": pool_units,
        "source_counts": {
            source_id: int(sum(1 for row in row_map if row["source_id"] == source_id))
            for source_id in sorted({row["source_id"] for row in row_map})
        },
        "source_paths": source_paths,
        "sample_id": int(sample_id),
        "sampling_mode": "mixed_source_resample_from_s_pool_units",
    }
    save_json(output_dir / "theta_s_pool_manifest.json", report)
    return synthetic_csv, row_map_path, report
