from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from MIA.attacks import make_attack_data, run_release_attacks
from MIA.audit import AuditConfig, audit_run
from MIA.attacks import shadow_attack
from MIA.metrics import summarize_binary_scores


def test_exact_and_nearest_neighbor_detect_copy_leakage() -> None:
    member = pd.DataFrame({"age": [20, 30, 40, 50], "city": ["a", "b", "c", "d"]})
    nonmember = pd.DataFrame({"age": [21, 31, 41, 51], "city": ["x", "y", "z", "w"]})
    synthetic = member.copy()
    reference = nonmember.copy()
    data = make_attack_data(member=member, nonmember=nonmember, synthetic=synthetic, reference=reference)

    outputs = run_release_attacks(data, max_attribute_columns=2)
    by_name = {output.name: output for output in outputs}
    exact_report = summarize_binary_scores("exact_match", data.labels, by_name["exact_match"].scores)
    nn_report = summarize_binary_scores("nearest_neighbor", data.labels, by_name["nearest_neighbor"].scores)

    assert exact_report.auroc == 1.0
    assert exact_report.threshold is not None
    assert np.isfinite(exact_report.threshold)
    assert nn_report.auroc is not None
    assert nn_report.auroc >= 0.75


def test_metric_report_handles_constant_scores() -> None:
    report = summarize_binary_scores("constant", [1, 1, 0, 0], [0.0, 0.0, 0.0, 0.0])

    assert report.auroc is None
    assert report.attack_advantage is None


def test_shadow_attack_uses_common_score_columns() -> None:
    target = pd.DataFrame(
        {
            "label": [1, 1, 0, 0],
            "nearest_neighbor": [0.9, 0.8, 0.2, 0.1],
            "supervised_error_profile": [0.7, 0.6, 0.4, 0.3],
        }
    )
    shadow = pd.DataFrame({"label": [1, 1, 0, 0], "nearest_neighbor": [0.95, 0.85, 0.15, 0.05]})

    output = shadow_attack(target_score_frame=target, shadow_score_frames=[shadow])

    assert output is not None
    assert output.name == "shadow_attack"
    assert output.details["feature_columns"] == ["nearest_neighbor"]
    assert len(output.scores) == 4


def test_cli_audit_artifact_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    input_dir = run_dir / "input"
    versions_dir = run_dir / "versions"
    input_dir.mkdir(parents=True)
    versions_dir.mkdir(parents=True)

    member = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "cat": ["a", "a", "b", "b"], "target": [0, 0, 1, 1]})
    holdout = pd.DataFrame({"x": [10.0, 11.0, 12.0, 13.0], "cat": ["c", "c", "d", "d"], "target": [0, 1, 0, 1]})
    test = pd.DataFrame({"x": [20.0, 21.0, 22.0, 23.0], "cat": ["e", "e", "f", "f"], "target": [0, 1, 0, 1]})
    synthetic = member.copy()
    member.to_csv(input_dir / "eval_train.csv", index=False)
    holdout.to_csv(input_dir / "eval_holdout.csv", index=False)
    test.to_csv(input_dir / "eval_test.csv", index=False)
    synthetic.to_csv(versions_dir / "selection_pareto.csv", index=False)
    (input_dir / "selection_context.json").write_text(json.dumps({"target_column": "target"}), encoding="utf-8")

    summary = audit_run(
        run_dir=run_dir,
        out_dir=tmp_path / "mia_out",
        config=AuditConfig(exclude_target=True, max_attribute_columns=2),
        all_selections=True,
    )

    assert summary["selection_count"] == 1
    assert (tmp_path / "mia_out" / "summary.json").exists()
    assert (tmp_path / "mia_out" / "metrics.csv").exists()
    assert (tmp_path / "mia_out" / "pareto" / "scores.csv").exists()
