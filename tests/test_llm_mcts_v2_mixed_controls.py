from __future__ import annotations

from pathlib import Path

from llm_mcts_tool import v2_pipeline
from llm_mcts_tool.v2_cli import config_from_args, parse_args
from llm_mcts_tool.v2_pipeline import (
    SNode,
    SourceInfo,
    V2MCTSConfig,
    _pool_source_count,
    _should_stop_for_hard_no_improve,
    _validate_pool_units,
    select_s_pools,
)


def _sources() -> dict[str, SourceInfo]:
    return {
        source_id: SourceInfo(source_id=source_id, path=Path(f"/tmp/{source_id}.csv"), rows=100, columns=["x"])
        for source_id in ("tabdiff", "tabsyn", "smote", "great")
    }


def test_v2_cli_uses_descriptive_pool_and_theta_names() -> None:
    args = parse_args(
        [
            "--provider",
            "mock",
            "--initial-s-pool-count",
            "7",
            "--theta-proposals-per-event",
            "9",
            "--new-s-pool-stagnation-events",
            "5",
            "--early-stop-stagnation-events",
            "12",
            "--utility-exact-torch-epochs",
            "10",
            "--refine-s-pool-count",
            "3",
            "--rollout-reward-candidate-v2",
            "--rollout-reward-candidate-v2-max-swap-fraction",
            "0.2",
            "--rollout-reward-candidate-v2-max-candidate-sizes",
            "12",
        ]
    )

    config = config_from_args(args)

    assert config.initial_s_pool_count == 7
    assert config.theta_proposals_per_event == 9
    assert config.new_s_pool_stagnation_events == 5
    assert config.early_stop_stagnation_events == 12
    assert config.utility_exact_torch_epochs == 10
    assert config.refine_s_pool_count == 3
    assert config.rollout_reward_candidate_v2_enabled is True
    assert config.rollout_reward_candidate_v2_max_swap_fraction == 0.2
    assert config.rollout_reward_candidate_v2_max_candidate_sizes == 12
    assert "m1" not in config.__dict__
    assert "m2" not in config.__dict__


def test_v2_cli_keeps_legacy_m1_m2_aliases() -> None:
    args = parse_args(
        [
            "--provider",
            "mock",
            "--m1",
            "3",
            "--m2",
            "5",
            "--mixed-stagnation-events",
            "4",
        ]
    )

    config = config_from_args(args)

    assert config.initial_s_pool_count == 3
    assert config.theta_proposals_per_event == 5
    assert config.new_s_pool_stagnation_events == 4


def test_integer_pool_validation_preserves_valid_llm_multipliers() -> None:
    units = _validate_pool_units(
        [{"source_id": "tabdiff", "multiplier": 1}, {"source_id": "tabsyn", "multiplier": 3}],
        sources=_sources(),
        pool_multiplier=4,
        require_integer=True,
        min_sources=2,
    )

    assert units == [{"source_id": "tabdiff", "multiplier": 1}, {"source_id": "tabsyn", "multiplier": 3}]


def test_initial_s_pool_selection_covers_three_and_four_source_candidates(monkeypatch, tmp_path: Path) -> None:
    def fake_call_llm_json(**_: object) -> dict:
        return {
            "syn_pools": [
                {"pool_units": [{"source_id": "tabdiff", "multiplier": 2}, {"source_id": "great", "multiplier": 2}]},
                {"pool_units": [{"source_id": "tabsyn", "multiplier": 2}, {"source_id": "great", "multiplier": 2}]},
                {"pool_units": [{"source_id": "tabsyn", "multiplier": 3}, {"source_id": "great", "multiplier": 1}]},
                {"pool_units": [{"source_id": "tabdiff", "multiplier": 3}, {"source_id": "smote", "multiplier": 1}]},
            ]
        }

    monkeypatch.setattr(v2_pipeline, "_call_llm_json", fake_call_llm_json)

    pools = select_s_pools(
        config=V2MCTSConfig(mode="mixed", pool_multiplier=4, initial_s_pool_count=4),
        sources=_sources(),
        client=object(),
        n=4,
        phase="init",
        source_profiles={},
        real_utility_reference={},
        s_nodes={},
        theta_nodes={},
        trace_dir=tmp_path,
    )

    assert len(pools) == 4
    assert {2, 3, 4}.issubset({_pool_source_count(item["pool_units"]) for item in pools})


def test_refine_s_pool_selection_prefers_missing_source_count(monkeypatch, tmp_path: Path) -> None:
    def fake_call_llm_json(**_: object) -> dict:
        return {
            "syn_pool": {
                "pool_units": [{"source_id": "tabdiff", "multiplier": 2}, {"source_id": "great", "multiplier": 2}]
            }
        }

    monkeypatch.setattr(v2_pipeline, "_call_llm_json", fake_call_llm_json)
    existing = {
        "s_000000": SNode(
            s_id="s_000000",
            pool_units=[{"source_id": "tabdiff", "multiplier": 2}, {"source_id": "great", "multiplier": 2}],
            synthetic_csv=tmp_path / "synthetic.csv",
            synthetic_row_map=tmp_path / "rows.jsonl",
        )
    }

    pools = select_s_pools(
        config=V2MCTSConfig(mode="mixed", pool_multiplier=4),
        sources=_sources(),
        client=object(),
        n=1,
        phase="refine",
        source_profiles={},
        real_utility_reference={},
        s_nodes=existing,
        theta_nodes={},
        trace_dir=tmp_path,
    )

    assert len(pools) == 1
    assert _pool_source_count(pools[0]["pool_units"]) == 3


def test_refine_s_pool_selection_keeps_valid_llm_multisource_when_counts_seen(monkeypatch, tmp_path: Path) -> None:
    llm_units = [
        {"source_id": "tabdiff", "multiplier": 1},
        {"source_id": "tabsyn", "multiplier": 2},
        {"source_id": "smote", "multiplier": 1},
    ]

    def fake_call_llm_json(**_: object) -> dict:
        return {"syn_pool": {"pool_units": llm_units, "family": "balanced", "llm_score": 0.8}}

    monkeypatch.setattr(v2_pipeline, "_call_llm_json", fake_call_llm_json)
    existing = {
        "s_000000": SNode(
            s_id="s_000000",
            pool_units=[{"source_id": "tabdiff", "multiplier": 2}, {"source_id": "great", "multiplier": 2}],
            synthetic_csv=tmp_path / "synthetic0.csv",
            synthetic_row_map=tmp_path / "rows0.jsonl",
        ),
        "s_000001": SNode(
            s_id="s_000001",
            pool_units=[
                {"source_id": "tabsyn", "multiplier": 2},
                {"source_id": "smote", "multiplier": 1},
                {"source_id": "great", "multiplier": 1},
            ],
            synthetic_csv=tmp_path / "synthetic1.csv",
            synthetic_row_map=tmp_path / "rows1.jsonl",
        ),
        "s_000002": SNode(
            s_id="s_000002",
            pool_units=[
                {"source_id": "tabdiff", "multiplier": 1},
                {"source_id": "tabsyn", "multiplier": 1},
                {"source_id": "smote", "multiplier": 1},
                {"source_id": "great", "multiplier": 1},
            ],
            synthetic_csv=tmp_path / "synthetic2.csv",
            synthetic_row_map=tmp_path / "rows2.jsonl",
        ),
    }

    pools = select_s_pools(
        config=V2MCTSConfig(mode="mixed", pool_multiplier=4),
        sources=_sources(),
        client=object(),
        n=1,
        phase="refine",
        source_profiles={},
        real_utility_reference={},
        s_nodes=existing,
        theta_nodes={},
        trace_dir=tmp_path,
    )

    assert len(pools) == 1
    assert pools[0]["pool_units"] == llm_units


def test_refine_s_pool_selection_can_return_multiple_nonduplicate_pools(monkeypatch, tmp_path: Path) -> None:
    def fake_call_llm_json(**_: object) -> dict:
        return {
            "syn_pools": [
                {
                    "pool_units": [
                        {"source_id": "tabdiff", "multiplier": 1},
                        {"source_id": "tabsyn", "multiplier": 2},
                        {"source_id": "smote", "multiplier": 1},
                    ],
                },
                {
                    "pool_units": [
                        {"source_id": "tabdiff", "multiplier": 1},
                        {"source_id": "tabsyn", "multiplier": 1},
                        {"source_id": "smote", "multiplier": 1},
                        {"source_id": "great", "multiplier": 1},
                    ],
                },
            ]
        }

    monkeypatch.setattr(v2_pipeline, "_call_llm_json", fake_call_llm_json)

    pools = select_s_pools(
        config=V2MCTSConfig(mode="mixed", pool_multiplier=4),
        sources=_sources(),
        client=object(),
        n=3,
        phase="refine",
        source_profiles={},
        real_utility_reference={},
        s_nodes={
            "s_000000": SNode(
                s_id="s_000000",
                pool_units=[{"source_id": "tabdiff", "multiplier": 2}, {"source_id": "great", "multiplier": 2}],
                synthetic_csv=tmp_path / "synthetic0.csv",
                synthetic_row_map=tmp_path / "rows0.jsonl",
            )
        },
        theta_nodes={},
        trace_dir=tmp_path,
    )

    keys = {v2_pipeline._canonical_s_key(item["pool_units"]) for item in pools}
    assert len(pools) == 3
    assert len(keys) == 3
    counts = [_pool_source_count(item["pool_units"]) for item in pools]
    assert 2 in counts
    assert 3 in counts
    assert 4 in counts


def test_early_stop_stagnation_threshold_is_explicit_and_disableable() -> None:
    assert _should_stop_for_hard_no_improve(7, 6) is True
    assert _should_stop_for_hard_no_improve(6, 6) is False
    assert _should_stop_for_hard_no_improve(100, -1) is False
