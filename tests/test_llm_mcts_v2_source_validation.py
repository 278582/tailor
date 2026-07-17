from __future__ import annotations

import json

from pathlib import Path

import pandas as pd

from llm_mcts_tool.v2_pipeline import V2MCTSConfig, _load_valid_source_frame, resolve_sources


def test_load_valid_source_frame_filters_invalid_categories(tmp_path) -> None:
    source_path = tmp_path / "source.csv"
    pd.DataFrame(
        [
            {"age": 30, "relationship": "Husband", "income": "<=50K"},
            {"age": 150, "relationship": "Wife", "income": ">50K"},
            {"age": 40, "relationship": "Local-gov", "income": "<=50K"},
        ]
    ).to_csv(source_path, index=False)
    schema_card = {
        "column_order": ["age", "relationship", "income"],
        "columns": {
            "age": {"type": "numerical", "is_target": False},
            "relationship": {"type": "categorical", "is_target": False, "legal_values": ["Husband", "Wife"]},
            "income": {"type": "categorical", "is_target": True, "legal_values": ["<=50K", ">50K"]},
        },
    }
    stats_card = {
        "numeric_stats": {"age": {"min": 0, "max": 100}},
        "categorical_top_values": {
            "relationship": [{"value": "Husband"}],
            "income": [{"value": "<=50K"}],
        },
    }

    valid_df, report = _load_valid_source_frame(
        dataset_name="adult",
        source_id="great",
        path=source_path,
        column_order=["age", "relationship", "income"],
        schema_card=schema_card,
        stats_card=stats_card,
        validation_dir=tmp_path / "validation",
    )

    assert list(valid_df.index) == [0, 1]
    assert valid_df.to_dict(orient="records") == [
        {"age": 30.0, "relationship": "Husband", "income": "<=50K"},
        {"age": 100.0, "relationship": "Wife", "income": ">50K"},
    ]
    assert report["num_rejected"] == 1
    assert report["reject_reason_histogram"] == {"invalid_category:relationship": 1}
    assert report["repair_action_histogram"] == {"numeric_clip:age": 1}

    saved = json.loads((tmp_path / "validation" / "source_validation_summary.json").read_text(encoding="utf-8"))
    assert saved["rows_before"] == 3
    assert saved["rows_after"] == 2


def test_default_v2_sources_resolve_from_sample_root() -> None:
    sources = resolve_sources(
        V2MCTSConfig(
            dataset_name="default",
            mode="mixed",
            source_names=("tabsyn", "smote", "great"),
            sample_root=Path("third_party/sample"),
        )
    )

    assert set(sources) == {"tabsyn", "smote", "great"}
    assert all(info.path.exists() for info in sources.values())
    assert all(info.rows == 108000 for info in sources.values())
    assert "default payment next month" in sources["tabsyn"].columns
