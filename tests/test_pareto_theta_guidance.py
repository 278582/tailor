from __future__ import annotations

import json
from pathlib import Path

from post_selection_tool.theta_guidance import (
    load_best_rollout_theta,
    load_final_theta,
    override_report_col_ps_with_all_features,
    resolve_mcts_dir,
)
from post_selection_tool.theta_pool import build_theta_synthetic_pool_from_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_final_theta_star(tmp_path: Path) -> None:
    mcts_dir = tmp_path / "mcts"
    theta = {
        "col_1ds": ["a", "b"],
        "col_2ds": ["a", "c"],
        "col_ps": ["a", "b", "c"],
        "col_u": "a",
    }
    _write_json(
        mcts_dir / "final" / "theta_star.json",
        {"theta_id": "t_final", "Q_self": 0.7, "theta": theta},
    )

    guidance = load_final_theta(mcts_dir)

    assert guidance.theta == theta
    assert guidance.theta_id == "t_final"
    assert guidance.reward == 0.7
    assert guidance.source_kind == "final"


def test_load_final_theta_star_from_mcts_v2_exposes_s_pool_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "adult" / "run_a"
    mcts_dir = run_dir / "mcts_v2"
    theta = {
        "col_1ds": ["a"],
        "col_2ds": ["a"],
        "col_ps": ["a", "b"],
        "col_u": "b",
    }
    _write_json(
        mcts_dir / "final" / "theta_star.json",
        {"theta_id": "t_final", "Q_self": 0.7, "s_id": "s_000001", "theta": theta},
    )
    _write_json(
        mcts_dir / "s_nodes" / "s_000001" / "synthetic_pool_manifest.json",
        {"s_id": "s_000001", "pool_units": [{"source_id": "tabdiff", "multiplier": 1}]},
    )

    resolved = resolve_mcts_dir(
        dataset_name="adult",
        theta_artifact_root=tmp_path / "artifacts",
        theta_run_name="run_a",
        theta_mcts_dir=None,
    )
    guidance = load_final_theta(resolved)

    assert resolved == mcts_dir
    assert guidance.s_id == "s_000001"
    assert guidance.synthetic_pool_manifest == mcts_dir / "s_nodes" / "s_000001" / "synthetic_pool_manifest.json"


def test_best_rollout_theta_uses_max_q_self(tmp_path: Path) -> None:
    mcts_dir = tmp_path / "mcts"
    low_theta = {
        "col_1ds": ["a"],
        "col_2ds": ["a"],
        "col_ps": ["a", "b"],
        "col_u": "a",
    }
    high_theta = {
        "col_1ds": ["b"],
        "col_2ds": ["b"],
        "col_ps": ["a", "b"],
        "col_u": "b",
    }
    high_rollout = mcts_dir / "rollouts" / "high"
    _write_json(high_rollout / "theta.json", {"theta_id": "t_high", "theta": high_theta})
    archive_path = mcts_dir / "archive" / "all_rollouts.jsonl"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(
        "\n".join(
            [
                json.dumps({"theta_id": "t_low", "Q_self": 0.1, "theta": low_theta}),
                json.dumps(
                    {
                        "theta_id": "t_high",
                        "Q_self": 0.9,
                        "theta": high_theta,
                        "rollout_dir": str(high_rollout),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    guidance = load_best_rollout_theta(mcts_dir)

    assert guidance.theta == high_theta
    assert guidance.theta_id == "t_high"
    assert guidance.reward == 0.9
    assert guidance.source_path == high_rollout / "theta.json"


def test_override_col_ps_uses_all_non_target_features() -> None:
    schema_card = {
        "column_order": ["a", "b", "target"],
        "columns": {
            "a": {"is_target": False},
            "b": {"is_target": False},
            "target": {"is_target": True},
        },
    }
    report = {
        "enabled": True,
        "theta": {
            "col_1ds": ["a"],
            "col_2ds": ["a"],
            "col_ps": ["a"],
            "col_u": "a",
        },
    }

    privacy_columns, updated_report = override_report_col_ps_with_all_features(report, schema_card)

    assert privacy_columns == ["a", "b"]
    assert updated_report["theta"]["col_ps"] == ["a", "b"]
    assert updated_report["col_ps_override"]["applied"] is True
    assert updated_report["col_ps_override"]["original_count"] == 1
    assert updated_report["col_ps_override"]["replacement_count"] == 2


def test_theta_pool_rebuild_samples_sources_from_s_manifest(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample"
    tabdiff_dir = sample_root / "tabdiff" / "adult"
    tabsyn_dir = sample_root / "tabsyn" / "adult"
    tabdiff_dir.mkdir(parents=True)
    tabsyn_dir.mkdir(parents=True)
    (tabdiff_dir / "sample_0.csv").write_text("x,y\n1,10\n2,20\n3,30\n", encoding="utf-8")
    (tabsyn_dir / "sample_0.csv").write_text("x,y\n4,40\n5,50\n6,60\n", encoding="utf-8")
    manifest_path = tmp_path / "s_nodes" / "s_000002" / "synthetic_pool_manifest.json"
    _write_json(
        manifest_path,
        {
            "s_id": "s_000002",
            "pool_units": [
                {"source_id": "tabdiff", "multiplier": 1},
                {"source_id": "tabsyn", "multiplier": 2},
            ],
        },
    )

    synthetic_csv, row_map_path, report = build_theta_synthetic_pool_from_manifest(
        manifest_path=manifest_path,
        dataset_name="adult",
        train_rows=2,
        seed=11,
        output_dir=tmp_path / "rebuilt",
        sample_root=sample_root,
    )

    assert synthetic_csv.exists()
    assert row_map_path.exists()
    assert report["sampling_mode"] == "mixed_source_resample_from_s_pool_units"
    assert report["source_counts"] == {"tabdiff": 2, "tabsyn": 4}
