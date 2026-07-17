from __future__ import annotations

import numpy as np
import pandas as pd

from tools.constrained_dcr_repair import (
    _adaptive_sizes,
    _annotate_candidate,
    _build_pairs,
    _estimate_dcr_after_swaps,
    _fast_candidate_sizes,
    _recommend_candidates,
)


def test_build_pairs_moves_dcr_toward_half_from_both_sides() -> None:
    dcr_df = pd.DataFrame(
        {
            "dcr_is_real_closer": [True, False, False, True],
            "dcr_margin": [0.4, -0.4, -0.3, 0.3],
        }
    )
    features = np.asarray([[0.0], [1.0], [0.1], [1.1]], dtype=float)
    base_to_raw_indices = np.asarray([0, 1], dtype=np.int64)
    utility_scores = np.zeros(4, dtype=float)

    high_dcr_pairs = _build_pairs(
        base_to_raw_indices=base_to_raw_indices,
        dcr_df=dcr_df,
        features=features,
        candidate_neighbors=1,
        margin_weight=0.0,
        utility_scores=utility_scores,
        utility_weight=0.0,
        target_dcr=0.8,
    )
    low_dcr_pairs = _build_pairs(
        base_to_raw_indices=base_to_raw_indices,
        dcr_df=dcr_df,
        features=features,
        candidate_neighbors=1,
        margin_weight=0.0,
        utility_scores=utility_scores,
        utility_weight=0.0,
        target_dcr=0.2,
    )

    assert high_dcr_pairs[0][0] == 0
    assert high_dcr_pairs[0][1] == 2
    assert low_dcr_pairs[0][0] == 1
    assert low_dcr_pairs[0][1] == 3


def test_build_pairs_uses_utility_gain_in_pair_ranking() -> None:
    dcr_df = pd.DataFrame(
        {
            "dcr_is_real_closer": [True, False, False],
            "dcr_margin": [0.5, -0.2, -0.2],
        }
    )
    features = np.asarray([[0.0], [0.1], [0.2]], dtype=float)
    utility_scores = np.asarray([0.0, 0.0, 1.0], dtype=float)

    pairs = _build_pairs(
        base_to_raw_indices=np.asarray([0], dtype=np.int64),
        dcr_df=dcr_df,
        features=features,
        candidate_neighbors=2,
        margin_weight=0.0,
        utility_scores=utility_scores,
        utility_weight=1.0,
        target_dcr=0.9,
    )

    assert pairs[0][0] == 0
    assert pairs[0][1] == 2
    assert pairs[0][4] == 1.0


def test_adaptive_sizes_expand_with_dcr_gap() -> None:
    small_gap = _adaptive_sizes(
        keep_k=10_000,
        pair_count=5_000,
        target_dcr=0.55,
        min_fraction=0.001,
        max_fraction=0.2,
        max_candidates=8,
    )
    large_gap = _adaptive_sizes(
        keep_k=10_000,
        pair_count=5_000,
        target_dcr=0.9,
        min_fraction=0.001,
        max_fraction=0.2,
        max_candidates=8,
    )

    assert small_gap
    assert large_gap
    assert max(large_gap) > max(small_gap)
    assert len(large_gap) <= 8


def test_dcr_estimate_moves_by_swap_fraction_toward_half() -> None:
    assert _estimate_dcr_after_swaps(base_dcr=0.8, keep_k=100, swaps=30) == 0.5
    assert _estimate_dcr_after_swaps(base_dcr=0.2, keep_k=100, swaps=30) == 0.5


def test_fast_candidate_sizes_returns_only_top_proxy_candidates() -> None:
    pairs = [
        (idx, idx + 100, 0.01 + idx * 0.001, 0.0, 0.1)
        for idx in range(40)
    ]

    selected, ranked = _fast_candidate_sizes(
        sizes=[5, 10, 20, 30, 40],
        pairs=pairs,
        keep_k=100,
        base_dcr=0.9,
        recommend_count=2,
    )

    assert len(selected) == 2
    assert selected[0] == 40
    assert ranked[0]["estimated_dcr"] == 0.5
    assert ranked[0]["estimated_dcr_privacy"] == 1.0


def test_candidate_recommendation_uses_objective_score_not_references() -> None:
    pareto_v9 = {"reward": 0.9, "shape": 0.98, "trend": 0.97, "dcr_privacy": 0.6}
    random_full = {"reward": 0.88, "shape": 0.982, "trend": 0.972, "dcr_privacy": 0.58}
    candidates = {
        "higher_objective": _annotate_candidate(
            item={"shape": 0.978, "trend": 0.973, "dcr": 0.55, "dcr_privacy": 0.95},
            pareto_v9_reference=pareto_v9,
            random_full_reference=random_full,
        ),
        "reference_closer": _annotate_candidate(
            item={"shape": 0.981, "trend": 0.971, "dcr": 0.65, "dcr_privacy": 0.85},
            pareto_v9_reference=pareto_v9,
            random_full_reference=random_full,
        ),
    }

    recommended = _recommend_candidates(candidates, recommend_count=1)

    assert recommended[0]["name"] == "higher_objective"
    assert "shape_soft_ok" not in candidates["higher_objective"]
    assert candidates["higher_objective"]["shape_delta_vs_pareto_v9"] < 0.0
