from __future__ import annotations

import pandas as pd

from post_selection_tool.utility_proxy import (
    apply_utility_source_prior_to_proxy_scores,
    parse_utility_source_prior,
)
from post_selection_tool.validation import _attach_candidate_source_metadata


def test_parse_utility_source_prior() -> None:
    assert parse_utility_source_prior("tabdiff:1.0,tabsyn:0.7") == {
        "tabdiff": 1.0,
        "tabsyn": 0.7,
    }


def test_apply_utility_source_prior_adjusts_proxy_scores() -> None:
    proxy_scores = [
        {"candidate_id": 0, "u_static_balanced": 0.8, "u_static_norm": 0.8, "u_proxy": 0.8},
        {"candidate_id": 1, "u_static_balanced": 0.8, "u_static_norm": 0.8, "u_proxy": 0.8},
        {"candidate_id": 2, "u_static_balanced": 0.8, "u_static_norm": 0.8, "u_proxy": 0.8},
    ]

    adjusted, report = apply_utility_source_prior_to_proxy_scores(
        proxy_scores,
        source_by_id={0: "tabdiff", 1: "tabsyn"},
        prior="tabdiff:1.0,tabsyn:0.5",
        default_weight=0.25,
    )

    assert report["enabled"] is True
    assert report["matched_rows"] == 2
    assert report["missing_source_rows"] == 1
    assert adjusted[0]["u_proxy"] == 0.8
    assert adjusted[1]["u_proxy"] == 0.4
    assert adjusted[2]["u_proxy"] == 0.2
    assert adjusted[1]["source_id"] == "tabsyn"
    assert adjusted[1]["u_proxy_before_source_prior"] == 0.8


def test_candidate_source_metadata_stays_out_of_row_dataframe() -> None:
    records = [
        {"candidate_id": 0, "row": {"x": 1, "y": "a"}},
        {"candidate_id": 1, "row": {"x": 2, "y": "b"}},
    ]

    updated = _attach_candidate_source_metadata(records, {0: "tabdiff", 1: "tabsyn"})
    df = pd.DataFrame([record["row"] for record in updated])

    assert updated[0]["_source_id"] == "tabdiff"
    assert updated[1]["_source_id"] == "tabsyn"
    assert list(df.columns) == ["x", "y"]
