from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metric_tool.reward import compute_metric_reward


class _Selector:
    target_column = "target"


def _base_summary() -> dict[str, object]:
    return {
        "shape": 1.0,
        "trend": 1.0,
        "dcr": 0.5,
        "utility_exact_available": True,
        "utility_exact_primary_score_group": "best_rmse_scores",
        "utility_exact_primary_model": "XGBRegressor",
    }


def test_regression_utility_score_uses_log1p_target_scale() -> None:
    rmse = 3.0
    test_df = pd.DataFrame({"target": [1.0, 10.0, 100.0, 1000.0]})
    summary = {
        **_base_summary(),
        "utility_exact_task_type": "regression",
        "utility_exact_tabdiff_task_type": "regression",
        "utility_exact_metric": "RMSE",
        "utility_exact_overall": rmse,
    }

    report = compute_metric_reward(summary=summary, selector=_Selector(), test_df=test_df)
    utility = report["components"]["utility"]

    transformed_target = np.log1p(np.maximum(test_df["target"].to_numpy(dtype=float), 0.0))
    expected_std = float(np.std(transformed_target, ddof=0))
    expected_relative_rmse = float(rmse / expected_std)
    expected_improvement = float(1.0 - expected_relative_rmse)
    expected_score = float(1.0 / (1.0 + np.exp(-4.0 * expected_improvement)))
    old_linear_clip_score = float(np.clip(expected_improvement, 0.0, 1.0))

    assert utility["available"] is True
    assert utility["target_transform"] == "log1p_nonnegative"
    assert utility["target_std"] == pytest.approx(expected_std)
    assert utility["target_raw_std"] == pytest.approx(float(np.std(test_df["target"], ddof=0)))
    assert utility["relative_rmse"] == pytest.approx(expected_relative_rmse)
    assert utility["improvement_vs_mean_baseline"] == pytest.approx(expected_improvement)
    assert utility["score"] == pytest.approx(expected_score)
    assert utility["score"] > old_linear_clip_score
    assert utility["normalization"] == "sigmoid(4 * (1 - rmse / std(log1p(max(y_test, 0)))))"


def test_classification_utility_score_stays_roc_auc_normalized() -> None:
    summary = {
        **_base_summary(),
        "utility_exact_primary_score_group": "best_auroc_scores",
        "utility_exact_primary_model": "XGBClassifier",
        "utility_exact_task_type": "classification",
        "utility_exact_tabdiff_task_type": "binclass",
        "utility_exact_metric": "roc_auc",
        "utility_exact_overall": 0.75,
    }

    report = compute_metric_reward(
        summary=summary,
        selector=_Selector(),
        test_df=pd.DataFrame({"target": [0, 1]}),
    )
    utility = report["components"]["utility"]

    assert utility["available"] is True
    assert utility["score"] == pytest.approx(0.5)
    assert utility["normalization"] == "clip((roc_auc - 0.5) / 0.5, 0, 1)"
