from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import llm_mcts_tool.v2_pipeline as v2
from post_selection_tool.utility_proxy import compute_utility_exact_metrics
from llm_mcts_tool.v2_pipeline import (
    V2MCTSConfig,
    build_real_utility_reference,
    _real_utility_for_prompt,
    _render_prompt,
    _utility_importance_for_selected_evaluator,
)


pytest.importorskip("torch")


class _Selector:
    def __init__(self) -> None:
        self.seed = 123
        self.nn_device = "cpu"
        self.utility_exact_evaluator = "torch_lightweight_mlp"
        self.utility_exact_torch_epochs = 2
        self.utility_exact_torch_batch_size = 32
        self.utility_exact_torch_importance_sample_size = 32
        self.target_column = "income"
        self.feature_columns = ["age", "education.num", "workclass"]
        self.column_order = [*self.feature_columns, self.target_column]
        self.schema_card = {
            "dataset": "adult",
            "target_column": self.target_column,
            "column_order": self.column_order,
            "columns": {
                "age": {"type": "numerical", "is_target": False},
                "education.num": {"type": "discrete_numerical", "is_target": False},
                "workclass": {"type": "categorical", "is_target": False},
                "income": {"type": "categorical", "is_target": True},
            },
        }
        self.stats_card = {}
        self.train_df = pd.DataFrame()

    def _prob_geomean_for_df(self, df: pd.DataFrame, columns: list[str]) -> np.ndarray:
        return np.linspace(0.05, 0.95, len(df), dtype=float)


def _make_frame(n_rows: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, size=n_rows)
    education = rng.integers(1, 17, size=n_rows)
    workclass = rng.choice(["Private", "Self-emp-not-inc", "Local-gov"], size=n_rows)
    score = 0.04 * (age - 35) + 0.25 * (education - 9) + rng.normal(0, 0.8, size=n_rows)
    income = np.where(score > 0.5, ">50K", "<=50K")
    return pd.DataFrame(
        {
            "age": age.astype(float),
            "education.num": education,
            "workclass": workclass,
            "income": income,
        }
    )


def test_torch_lightweight_mlp_exact_utility_schema() -> None:
    selector = _Selector()
    syn_df = _make_frame(96, seed=1)
    test_df = _make_frame(48, seed=2)
    selector.train_df = syn_df.copy()

    report = compute_utility_exact_metrics(selector, syn_df, test_df)

    assert report["available"] is True
    assert report["protocol"] == "torch_lightweight_mlp"
    assert report["task_type"] == "classification"
    assert report["tabdiff_task_type"] == "binclass"
    assert report["metric"] == "roc_auc"
    assert 0.0 <= float(report["overall"]) <= 1.0
    assert report["primary_score_group"] == "best_auroc_scores"
    assert report["primary_model"] == "TorchMLPClassifier"
    assert report["runtime_model_device"] == "cpu"
    assert isinstance(report["overall_scores"]["best_auroc_scores"]["TorchMLPClassifier"]["roc_auc"], float)
    assert report["rows"]["tail"] + report["rows"]["middle"] + report["rows"]["mode"] == len(test_df)
    assert report["torch_train_rows"] == len(syn_df)
    assert report["torch_test_rows"] == len(test_df)
    assert report["torch_importance_sample_size"] == selector.utility_exact_torch_importance_sample_size
    assert report["torch_importance_rows"] == selector.utility_exact_torch_importance_sample_size
    assert report["feature_importance"]
    assert {item["feature"] for item in report["feature_importance"]}.issubset(set(selector.feature_columns))


def test_v2_torch_diagnostics_prefers_exact_report_feature_importance() -> None:
    report = {
        "protocol": "torch_lightweight_mlp",
        "feature_importance": [
            {"feature": "marital.status", "importance": 0.7, "rank": 1},
            {"feature": "age", "importance": 0.3, "rank": 2},
        ],
    }

    rows = _utility_importance_for_selected_evaluator(
        config=V2MCTSConfig(utility_exact_evaluator="torch_lightweight_mlp"),
        train_like_df=pd.DataFrame({"age": [1], "income": ["<=50K"]}),
        test_df=pd.DataFrame({"age": [1], "income": ["<=50K"]}),
        schema_card={},
        dataset_context={},
        seed=1,
        sample_size=1,
        utility_report=report,
    )

    assert rows == [
        {"feature": "marital.status", "importance": 0.7, "rank": 1},
        {"feature": "age", "importance": 0.3, "rank": 2},
    ]


def test_real_utility_profile_uses_configured_torch_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _DummyParetoSelector:
        def __init__(
            self,
            *,
            train_df: pd.DataFrame,
            holdout_df: pd.DataFrame,
            schema_card: dict[str, object],
            stats_card: dict[str, object],
            seed: int,
            source: str,
            privacy_version: str,
            density_reference_size: int,
            nn_device: str,
            high_cardinality_enabled: bool,
        ) -> None:
            calls["selector_source"] = source
            calls["selector_nn_device"] = nn_device
            calls["density_reference_size"] = density_reference_size
            self.train_df = train_df
            self.holdout_df = holdout_df
            self.schema_card = schema_card
            self.stats_card = stats_card
            self.seed = seed
            self.target_column = str(schema_card["target_column"])
            self.column_order = list(schema_card["column_order"])
            self.feature_columns = [column for column in self.column_order if column != self.target_column]

    def _fake_compute_utility_exact_metrics(
        selector: object,
        syn_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        evaluator: str | None = None,
        random_state: int | None = None,
        **_: object,
    ) -> dict[str, object]:
        calls["evaluator"] = evaluator
        calls["random_state"] = random_state
        calls["syn_rows"] = len(syn_df)
        calls["test_rows"] = len(test_df)
        calls["torch_epochs"] = getattr(selector, "utility_exact_torch_epochs")
        calls["importance_sample_size"] = getattr(selector, "utility_exact_torch_importance_sample_size")
        return {
            "available": True,
            "protocol": "torch_lightweight_mlp",
            "metric": "roc_auc",
            "overall": 0.6789,
            "runtime_model_device": "cpu",
            "feature_importance_method": "permutation_importance_on_holdout",
            "feature_importance": [
                {"feature": "marital.status", "importance": 0.6, "rank": 1},
                {"feature": "age", "importance": 0.4, "rank": 2},
            ],
        }

    monkeypatch.setattr(v2, "ParetoSelector", _DummyParetoSelector)
    monkeypatch.setattr(v2, "compute_utility_exact_metrics", _fake_compute_utility_exact_metrics)

    schema_card = {
        "dataset": "adult",
        "target_column": "income",
        "column_order": ["age", "marital.status", "income"],
        "columns": {
            "age": {"type": "numerical", "is_target": False},
            "marital.status": {"type": "categorical", "is_target": False},
            "income": {"type": "categorical", "is_target": True},
        },
    }
    train_df = pd.DataFrame(
        {
            "age": [25, 46, 51, 39],
            "marital.status": ["Never-married", "Married-civ-spouse", "Married-civ-spouse", "Divorced"],
            "income": ["<=50K", ">50K", ">50K", "<=50K"],
        }
    )
    test_df = pd.DataFrame(
        {
            "age": [31, 58, 44],
            "marital.status": ["Never-married", "Married-civ-spouse", "Divorced"],
            "income": ["<=50K", ">50K", "<=50K"],
        }
    )
    config = V2MCTSConfig(
        dataset_name="adult",
        utility_exact_evaluator="torch_lightweight_mlp",
        utility_exact_torch_epochs=9,
        utility_diag_sample_size=3,
        eval_device="cpu",
        nn_device="cpu",
    )

    reference = build_real_utility_reference(
        train_df=train_df,
        test_df=test_df,
        schema_card=schema_card,
        stats_card={},
        dataset_context={"target_summary": {"task_type": "classification"}},
        config=config,
    )

    utility = reference["utility_feature_importance"]
    assert utility["backend"] == "torch_lightweight_mlp"
    assert utility["metric"] == "roc_auc"
    assert utility["test_score"] == 0.6789
    assert utility["feature_importance_method"] == "permutation_importance_on_holdout"
    assert utility["top_features"] == [
        {"feature": "marital.status", "importance": 0.6, "rank": 1},
        {"feature": "age", "importance": 0.4, "rank": 2},
    ]
    assert calls["evaluator"] == "torch_lightweight_mlp"
    assert calls["torch_epochs"] == 9
    assert calls["importance_sample_size"] == 0
    assert calls["syn_rows"] == 4
    assert calls["test_rows"] == 3
    assert utility["utility_full_eval"] is True
    assert utility["train_rows_used"] == 4
    assert utility["test_rows_used"] == 3
    assert utility["feature_importance_test_rows_used"] == 3
    assert utility["utility_exact_overall"] == 0.6789

    prompt = _render_prompt(
        config,
        "v2_real_utility_profile_summary_prompt.j2",
        {
            "dataset_brief": {"dataset": "adult", "target": "income"},
            "real_utility_profile": _real_utility_for_prompt(reference),
        },
    )
    assert '"backend": "torch_lightweight_mlp"' in prompt
    assert "configured utility evaluator feature ranking" in prompt
    assert "XGBoost feature ranking" not in prompt
