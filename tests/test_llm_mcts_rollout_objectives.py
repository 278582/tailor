from __future__ import annotations

import pandas as pd

from llm_mcts_tool.rollout import _compute_search_objectives


class _Selector:
    def compute_dataset_fidelity(self, df: pd.DataFrame) -> float:
        return 0.91

    def compute_dataset_pair_fidelity(self, df: pd.DataFrame) -> float:
        return 0.82

    def compute_dataset_privacy(self, df: pd.DataFrame) -> float:
        return 1.7


def test_search_objectives_use_selected_report_values() -> None:
    objectives = _compute_search_objectives(
        selector=_Selector(),
        pareto_df=pd.DataFrame({"age": [1, 2]}),
        pareto_records=[{"row": {"age": 1}}, {"row": {"age": 2}}],
        pareto_report={
            "selected_privacy_component_mean": 0.31,
            "selected_utility_mean": 0.64,
        },
    )

    assert objectives == {
        "F_1D_theta": 0.91,
        "F_2D_theta": 0.82,
        "P_theta": 0.31,
        "P_theta_raw": 1.7,
        "U_proxy_theta": 0.64,
    }


def test_search_objectives_fall_back_to_record_means() -> None:
    objectives = _compute_search_objectives(
        selector=_Selector(),
        pareto_df=pd.DataFrame({"age": [1, 2]}),
        pareto_records=[
            {"pareto_priv_obj": 0.2, "pareto_util_proxy_obj": 0.4},
            {"pareto_priv_obj": 0.6, "pareto_util_proxy_obj": 0.8},
        ],
        pareto_report={},
    )

    assert objectives["P_theta"] == 0.4
    assert objectives["U_proxy_theta"] == 0.6000000000000001
