from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
PRESELECT_MULTIPLIER = 1.4

DATASET_SEARCH_FILL_FIELDS = (
    "keep_k",
    "preselect_target",
    "d_cur_size",
    "density_reference_size",
    "rollout_direct_dcr_repair_enabled",
    "rollout_direct_dcr_target_margin",
    "rollout_direct_dcr_max_swap_fraction",
    "rollout_direct_dcr_candidate_neighbors",
)


@dataclass(frozen=True)
class V2DatasetSearchDefaults:
    """Search-time defaults after real.csv was split into train/hold (90/10).

    keep_k matches synthetic/{dataset}/train.csv. Original real size is n_train + n_hold,
    except diabetes which was not re-split and still uses the full 61059-row train set.
    """

    keep_k: int
    preselect_target: int
    d_cur_size: int
    density_reference_size: int
    rollout_direct_dcr_repair_enabled: bool
    rollout_direct_dcr_target_margin: float
    rollout_direct_dcr_max_swap_fraction: float
    rollout_direct_dcr_candidate_neighbors: int
    n_train: int
    n_hold: int
    n_real_original: int
    note: str = ""


def preselect_target_for_keep_k(keep_k: int) -> int:
    return int(round(PRESELECT_MULTIPLIER * int(keep_k)))


def _preselect(keep_k: int) -> int:
    return preselect_target_for_keep_k(keep_k)


def _row(
    *,
    n_train: int,
    n_hold: int,
    n_real_original: int,
    density_reference_size: int,
    max_swap_fraction: float,
    d_cur_size: int = 2000,
    repair_enabled: bool = True,
    target_margin: float = 0.03,
    candidate_neighbors: int = 64,
    note: str = "",
) -> V2DatasetSearchDefaults:
    keep_k = int(n_train)
    return V2DatasetSearchDefaults(
        keep_k=keep_k,
        preselect_target=_preselect(keep_k),
        d_cur_size=int(d_cur_size),
        density_reference_size=int(density_reference_size),
        rollout_direct_dcr_repair_enabled=bool(repair_enabled),
        rollout_direct_dcr_target_margin=float(target_margin),
        rollout_direct_dcr_max_swap_fraction=float(max_swap_fraction),
        rollout_direct_dcr_candidate_neighbors=int(candidate_neighbors),
        n_train=int(n_train),
        n_hold=int(n_hold),
        n_real_original=int(n_real_original),
        note=note,
    )


V2_DATASET_SEARCH_DEFAULTS: dict[str, V2DatasetSearchDefaults] = {
    "adult": _row(
        n_train=29274,
        n_hold=3287,
        n_real_original=32561,
        density_reference_size=5000,
        max_swap_fraction=0.30,
    ),
    "beijing": _row(
        n_train=33825,
        n_hold=3756,
        n_real_original=37581,
        density_reference_size=8000,
        max_swap_fraction=0.35,
    ),
    "default": _row(
        n_train=24289,
        n_hold=2711,
        n_real_original=27000,
        density_reference_size=5000,
        max_swap_fraction=0.35,
    ),
    "diabetes": _row(
        n_train=61059,
        n_hold=20353,
        n_real_original=61059,
        density_reference_size=10000,
        max_swap_fraction=0.35,
        note="diabetes was not re-split 90/10; keep_k stays the full original train/real size.",
    ),
    "magic": _row(
        n_train=15404,
        n_hold=1713,
        n_real_original=17117,
        density_reference_size=5000,
        max_swap_fraction=0.35,
    ),
    "news": _row(
        n_train=32113,
        n_hold=3566,
        n_real_original=35679,
        density_reference_size=8000,
        max_swap_fraction=0.35,
    ),
    "shoppers": _row(
        n_train=9973,
        n_hold=1124,
        n_real_original=11097,
        density_reference_size=5000,
        max_swap_fraction=0.35,
    ),
    "us_census_data_1990": _row(
        n_train=899991,
        n_hold=100009,
        n_real_original=1000000,
        density_reference_size=10000,
        max_swap_fraction=0.30,
        target_margin=0.07,
        candidate_neighbors=256,
        note=(
            "census keep-k matches the 90/10 train split; density uses the diabetes "
            "maximum, while DCR margin/neighbors match the census post-selection job."
        ),
    ),
}

GENERIC_SEARCH_DEFAULTS = {
    "d_cur_size": 2000,
    "density_reference_size": 5000,
    "rollout_direct_dcr_repair_enabled": True,
    "rollout_direct_dcr_target_margin": 0.03,
    "rollout_direct_dcr_max_swap_fraction": 0.35,
    "rollout_direct_dcr_candidate_neighbors": 64,
}


def _train_rows_from_disk(dataset_name: str) -> int | None:
    report_path = ROOT_DIR / "synthetic" / dataset_name / "split_report.json"
    if report_path.exists():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        n_train = payload.get("n_train")
        if n_train is not None:
            return int(n_train)
    train_csv = ROOT_DIR / "synthetic" / dataset_name / "train.csv"
    if not train_csv.exists():
        return None
    with train_csv.open("r", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def search_defaults_for_dataset(dataset_name: str) -> dict[str, Any]:
    key = str(dataset_name or "").strip().lower()
    row = V2_DATASET_SEARCH_DEFAULTS.get(key)
    if row is not None:
        return {
            "keep_k": row.keep_k,
            "preselect_target": row.preselect_target,
            "d_cur_size": row.d_cur_size,
            "density_reference_size": row.density_reference_size,
            "rollout_direct_dcr_repair_enabled": row.rollout_direct_dcr_repair_enabled,
            "rollout_direct_dcr_target_margin": row.rollout_direct_dcr_target_margin,
            "rollout_direct_dcr_max_swap_fraction": row.rollout_direct_dcr_max_swap_fraction,
            "rollout_direct_dcr_candidate_neighbors": row.rollout_direct_dcr_candidate_neighbors,
        }
    n_train = _train_rows_from_disk(key)
    defaults = dict(GENERIC_SEARCH_DEFAULTS)
    if n_train:
        defaults["keep_k"] = int(n_train)
        defaults["preselect_target"] = _preselect(n_train)
    return defaults


def fill_none_search_fields(values: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    defaults = search_defaults_for_dataset(dataset_name)
    filled = dict(values)
    preselect_was_none = filled.get("preselect_target") is None
    for key in DATASET_SEARCH_FILL_FIELDS:
        if filled.get(key) is None and key in defaults:
            filled[key] = defaults[key]
    if preselect_was_none and filled.get("keep_k") is not None:
        filled["preselect_target"] = preselect_target_for_keep_k(int(filled["keep_k"]))
    return filled
