from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from llm_mcts_tool.rollout import GuidedRolloutConfig, _build_core_config, _save_rollout_artifacts
from llm_mcts_tool.strategy import StrategyTheta
from post_selection_tool.config import CoreSelectionConfig
from post_selection_tool.context import build_artifact_paths
from post_selection_tool.io import save_json, save_shared_csv, save_shared_json
from post_selection_tool.state import ArtifactPaths, SelectionState
from post_selection_tool.validation import CARD_FILENAMES, _build_or_share_cards, build_cards_and_validate, initialize_selector_and_pool


def test_save_shared_csv_uses_one_shared_file_for_multiple_targets(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    shared = tmp_path / "shared" / "versions" / "raw_valid.csv"
    first = tmp_path / "rollouts" / "a" / "versions" / "raw_valid.csv"
    second = tmp_path / "rollouts" / "b" / "versions" / "raw_valid.csv"

    save_shared_csv(first, df, shared)
    save_shared_csv(second, df, shared)

    assert first.read_text() == second.read_text() == shared.read_text()
    if os.stat(first).st_dev == os.stat(shared).st_dev:
        assert os.path.samefile(first, shared)
        assert os.path.samefile(second, shared)


def test_save_shared_json_uses_one_shared_file_for_multiple_targets(tmp_path: Path) -> None:
    payload = {"dataset": "adult", "seed": 7}
    shared = tmp_path / "shared" / "input" / "selection_context.json"
    first = tmp_path / "rollouts" / "a" / "input" / "selection_context.json"
    second = tmp_path / "rollouts" / "b" / "input" / "selection_context.json"

    save_shared_json(first, payload, shared)
    save_shared_json(second, payload, shared)

    assert first.read_text() == second.read_text() == shared.read_text()
    if os.stat(first).st_dev == os.stat(shared).st_dev:
        assert os.path.samefile(first, shared)
        assert os.path.samefile(second, shared)


def test_guided_rollout_core_config_points_to_mcts_shared_dir(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "mcts" / "rollouts" / "theta_1"
    config = GuidedRolloutConfig(
        theta_id="theta_1",
        dataset_name="adult",
        exp_name="test",
        artifact_dir=tmp_path,
        synthetic_csv=tmp_path / "samples.csv",
        seed=1,
        keep_k=10,
        preselect_target=20,
        d_cur_size=5,
        max_theta_pairs=4,
        rollout_dir=rollout_dir,
    )
    theta = StrategyTheta(
        col_1ds=("age",),
        col_2ds=("age", "education"),
        col_ps=("age",),
        col_u="age",
    )

    core_config = _build_core_config(config, theta)

    assert core_config.artifact_dir == rollout_dir.parent
    assert core_config.shared_artifact_dir == tmp_path / "mcts" / "shared"


def test_shared_artifact_paths_read_common_input_and_cards(tmp_path: Path) -> None:
    config = CoreSelectionConfig(
        dataset_name="adult",
        exp_name="theta_a",
        artifact_dir=tmp_path / "mcts" / "rollouts",
        shared_artifact_dir=tmp_path / "mcts" / "shared",
    )

    paths = build_artifact_paths(config, config.artifact_dir)

    assert paths.artifact_dir == tmp_path / "mcts" / "rollouts" / "theta_a"
    assert paths.input_dir == tmp_path / "mcts" / "shared" / "input"
    assert paths.cards_dir == tmp_path / "mcts" / "shared" / "cards"


def test_shared_artifact_paths_remove_old_rollout_input_mirror(tmp_path: Path) -> None:
    rollout_input = tmp_path / "mcts" / "rollouts" / "theta_a" / "input"
    rollout_input.mkdir(parents=True)
    (rollout_input / "eval_train.csv").write_text("old\n", encoding="utf-8")
    config = CoreSelectionConfig(
        dataset_name="adult",
        exp_name="theta_a",
        artifact_dir=tmp_path / "mcts" / "rollouts",
        shared_artifact_dir=tmp_path / "mcts" / "shared",
    )

    paths = build_artifact_paths(config, config.artifact_dir)

    assert paths.input_dir == tmp_path / "mcts" / "shared" / "input"
    assert not rollout_input.exists()


def test_shared_cards_are_reused_by_rollouts(tmp_path: Path) -> None:
    shared_dir = tmp_path / "mcts" / "shared"
    train_df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "workclass": ["a", "b", "a", "c"],
            "income": ["<=50K", ">50K", "<=50K", ">50K"],
        }
    )
    config = SimpleNamespace(
        seed=7,
        dataset_name="adult",
        shared_artifact_dir=shared_dir,
    )
    dataset_ctx = SimpleNamespace(
        target_column="income",
        categorical_columns=["workclass", "income"],
        numerical_columns=["age"],
        discrete_numerical_columns=[],
        privacy_sensitive_columns=["age"],
    )

    def make_state(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            config=config,
            dataset_ctx=dataset_ctx,
            train_df=train_df,
            paths=SimpleNamespace(cards_dir=tmp_path / "mcts" / "rollouts" / name / "cards"),
        )

    first_cards = _build_or_share_cards(make_state("theta_a"))
    second_cards = _build_or_share_cards(make_state("theta_b"))

    assert first_cards.schema_card == second_cards.schema_card
    assert not (tmp_path / "mcts" / "rollouts" / "theta_a" / "cards").exists()
    assert not (tmp_path / "mcts" / "rollouts" / "theta_b" / "cards").exists()
    for filename in CARD_FILENAMES:
        shared_file = shared_dir / "cards" / filename
        assert shared_file.exists()


def test_shared_cards_remove_old_rollout_cards_mirror(tmp_path: Path) -> None:
    shared_dir = tmp_path / "mcts" / "shared"
    rollout_cards = tmp_path / "mcts" / "rollouts" / "theta_a" / "cards"
    rollout_cards.mkdir(parents=True)
    save_json(rollout_cards / "schema_card.json", {"old": True})
    train_df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "workclass": ["a", "b", "a", "c"],
            "income": ["<=50K", ">50K", "<=50K", ">50K"],
        }
    )
    config = SimpleNamespace(
        seed=7,
        dataset_name="adult",
        shared_artifact_dir=shared_dir,
    )
    dataset_ctx = SimpleNamespace(
        target_column="income",
        categorical_columns=["workclass", "income"],
        numerical_columns=["age"],
        discrete_numerical_columns=[],
        privacy_sensitive_columns=["age"],
    )
    state = SimpleNamespace(
        config=config,
        dataset_ctx=dataset_ctx,
        train_df=train_df,
        paths=SimpleNamespace(
            artifact_dir=tmp_path / "mcts" / "rollouts" / "theta_a",
            cards_dir=shared_dir / "cards",
        ),
    )

    _build_or_share_cards(state)

    assert not rollout_cards.exists()
    assert (shared_dir / "cards" / "schema_card.json").exists()


def test_valid_rows_are_deduplicated_before_shared_selection_split(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "mcts" / "rollouts" / "theta_a"
    shared_dir = tmp_path / "mcts" / "shared"
    train_df = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0],
            "workclass": ["Private", "Self", "Private", "Self"],
            "income": ["<=50K", ">50K", "<=50K", ">50K"],
        }
    )
    synthetic_df = pd.DataFrame(
        {
            "age": [20.0, 20.0, 30.0, 40.0],
            "workclass": ["Private", "private", "Self", "Private"],
            "income": ["<=50K", "<=50K", ">50K", "<=50K"],
        }
    )
    config = CoreSelectionConfig(
        dataset_name="adult",
        exp_name="theta_a",
        artifact_dir=tmp_path / "mcts" / "rollouts",
        shared_artifact_dir=shared_dir,
        seed=7,
        keep_k=1,
        d_cur_size=1,
        d_cur_source="synthetic",
        eval_device="cpu",
        nn_device="auto",
        save_validation_records=True,
    )
    paths = ArtifactPaths(
        artifact_dir=artifact_dir,
        input_dir=shared_dir / "input",
        cards_dir=shared_dir / "cards",
        validation_dir=artifact_dir / "validation",
        selection_dir=artifact_dir / "selection",
        versions_dir=artifact_dir / "versions",
        report_dir=artifact_dir / "report",
    )
    dataset_ctx = SimpleNamespace(
        target_column="income",
        categorical_columns=["workclass", "income"],
        numerical_columns=["age"],
        discrete_numerical_columns=[],
        privacy_sensitive_columns=["age"],
    )
    state = SelectionState(
        config=config,
        paths=paths,
        dataset_ctx=dataset_ctx,
        synthetic_csv=tmp_path / "synthetic.csv",
        train_df=train_df,
        holdout_df=train_df.copy(),
        test_df=train_df.copy(),
        synthetic_df=synthetic_df,
    )

    state = build_cards_and_validate(state, show_progress=False)
    state = initialize_selector_and_pool(state)

    raw_valid = pd.read_csv(shared_dir / "versions" / "raw_valid.csv")
    candidate_pool = pd.read_csv(shared_dir / "selection" / "candidate_pool.csv")
    d_cur = pd.read_csv(shared_dir / "selection" / "d_cur_init.csv")
    report = state.validation_report

    assert report["num_valid_before_dedup"] == 4
    assert report["num_valid"] == 3
    assert report["duplicate_valid_filter"]["duplicate_rows_removed"] == 1
    assert raw_valid.duplicated().sum() == 0
    assert candidate_pool.duplicated().sum() == 0
    assert d_cur.duplicated().sum() == 0
    assert len(candidate_pool) + len(d_cur) == len(raw_valid) == 3


def test_rollout_internal_large_records_are_skipped_by_default(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "mcts" / "rollouts" / "theta_1"
    config = GuidedRolloutConfig(
        theta_id="theta_1",
        dataset_name="adult",
        exp_name="test",
        artifact_dir=tmp_path,
        synthetic_csv=tmp_path / "samples.csv",
        seed=1,
        keep_k=2,
        preselect_target=2,
        d_cur_size=1,
        max_theta_pairs=4,
        rollout_dir=rollout_dir,
    )
    theta = StrategyTheta(
        col_1ds=("age",),
        col_2ds=("age", "education"),
        col_ps=("age",),
        col_u="age",
    )
    state = SimpleNamespace(
        global_exact_records=[{"candidate_id": 1}],
        utility_proxy_bundle={"proxy_scores": [{"candidate_id": 1}], "manifest": {}, "pre_ceiling_static": {}},
        utility_proxy_merge_report={},
        preselect_gate={},
        preselect_status={},
        fidelity_ceiling_report={},
        global_baselines={},
        effective_keep_k=2,
        selector=SimpleNamespace(
            last_preselect_report={},
            fidelity_1d_columns=("age",),
            fidelity_2d_anchor_columns=("age", "education"),
            privacy_columns=("age",),
            utility_balance_column="age",
            max_pair_marginal_edges=4,
            pair_marginal_edges=[],
        ),
    )

    _save_rollout_artifacts(
        config=config,
        theta=theta,
        pareto_df=pd.DataFrame({"age": [1, 2], "education": [3, 4]}),
        pareto_records=[{"candidate_id": 1}],
        pareto_report={},
        state=state,
        search_objectives={},
    )

    internal_dir = rollout_dir / "internal"
    assert not (internal_dir / "pareto_records.jsonl").exists()
    assert not (internal_dir / "exact_scores.jsonl").exists()
    assert not (internal_dir / "utility_proxy_scores.jsonl").exists()
    assert (internal_dir / "record_artifacts.json").exists()
