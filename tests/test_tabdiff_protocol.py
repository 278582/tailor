from __future__ import annotations

import pandas as pd

from postprocess.data_io import load_csv
from postprocess.tabdiff_protocol import (
    _dataset_override,
    _normalize_split,
    normalize_tabdiff_dataframe_columns,
    resolve_tabdiff_selection_context,
)
from postprocess.tabdiff_utils import get_tabdiff_paths


def _load_protocol_split(dataset_name: str, filename: str, info: dict) -> pd.DataFrame:
    paths = get_tabdiff_paths(dataset_name)
    return normalize_tabdiff_dataframe_columns(
        dataset_name,
        _normalize_split(load_csv(paths.data_dir / filename), info=info),
    )


def test_shoppers_override_matches_prompt_context_roles() -> None:
    override = _dataset_override("shoppers")

    assert override["logical_name"] == "shoppers"
    assert override["target_column"] == "Revenue"
    assert override["discrete_numerical_columns"] == [
        "Administrative",
        "Informational",
        "ProductRelated",
    ]
    assert override["privacy_sensitive_columns"] == [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
    ]


def test_existing_dataset_overrides_are_unchanged() -> None:
    assert _dataset_override("adult")["target_column"] == "income"
    assert _dataset_override("adult_tgm_w1")["target_column"] == "income"
    assert _dataset_override("magic")["target_column"] == "class"
    assert _dataset_override("beijing") == {}
    assert _dataset_override("default") == {}


def test_default_selection_context_resolves_tabdiff_protocol_splits() -> None:
    context = resolve_tabdiff_selection_context("default", seed=20260420)

    assert context.dataset_name == "default"
    assert context.logical_name == "default"
    assert context.target_column == "default payment next month"
    assert context.task_type == "binclass"
    assert len(context.train_df) == 27000
    assert len(context.test_df) == 3000
    assert len(context.holdout_df) == 3000
    assert context.holdout_strategy == "test_as_holdout_full_train"
    assert "PAY_0" in context.categorical_columns
    assert "LIMIT_BAL" in context.numerical_columns


def test_selection_context_falls_back_to_test_as_holdout_without_val() -> None:
    context = resolve_tabdiff_selection_context("adult", seed=20260420)
    paths = get_tabdiff_paths("adult")
    assert not (paths.data_dir / "val.csv").exists()
    train_df = _load_protocol_split("adult", "train.csv", context.info)
    test_df = _load_protocol_split("adult", "test.csv", context.info)

    pd.testing.assert_frame_equal(context.train_df, train_df)
    pd.testing.assert_frame_equal(context.holdout_df, test_df)
    pd.testing.assert_frame_equal(context.test_df, test_df)
    assert context.holdout_strategy == "test_as_holdout_full_train"


def test_selection_context_uses_val_as_holdout_when_available() -> None:
    context = resolve_tabdiff_selection_context("diabetes", seed=20260420)
    paths = get_tabdiff_paths("diabetes")
    assert (paths.data_dir / "val.csv").exists()
    train_df = _load_protocol_split("diabetes", "train.csv", context.info)
    val_df = _load_protocol_split("diabetes", "val.csv", context.info)
    test_df = _load_protocol_split("diabetes", "test.csv", context.info)

    pd.testing.assert_frame_equal(context.train_df, train_df)
    pd.testing.assert_frame_equal(context.holdout_df, val_df)
    pd.testing.assert_frame_equal(context.test_df, test_df)
    assert context.holdout_strategy == "val_as_holdout_full_train"
