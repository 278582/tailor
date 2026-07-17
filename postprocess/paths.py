from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = ROOT_DIR / "third_party"
TABDIFF_DIR = THIRD_PARTY_DIR / "TabDiff"

DEFAULT_TABDIFF_DATASET_NAME = "adult_tgm_w1"

ADULT_DATA_DIR = ROOT_DIR / "data" / "adult"
ADULT_TRAIN_PATH = ADULT_DATA_DIR / "train.csv"
ADULT_HOLDOUT_PATH = ADULT_DATA_DIR / "holdout.csv"
ADULT_TEST_PATH = ADULT_DATA_DIR / "test.csv"

