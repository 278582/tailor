from __future__ import annotations

from pathlib import Path

from llm_mcts_tool.config import MCTSGuideConfig
from llm_mcts_tool.proposal import ProposalContext, ProposalProvider, _proposal_from_payload
from llm_mcts_tool.strategy import StrategyProposal, StrategyTheta, ThetaAction


def _schema_card() -> dict:
    return {
        "column_order": ["age", "workclass", "education", "income"],
        "columns": {
            "age": {"is_target": False},
            "workclass": {"is_target": False},
            "education": {"is_target": False},
            "income": {"is_target": True},
        },
    }


def _config() -> MCTSGuideConfig:
    return MCTSGuideConfig(
        dataset_name="adult",
        exp_name="test",
        synthetic_csv=None,
        artifact_dir=None,
        seed=11,
        keep_k=10,
        preselect_target=20,
        d_cur_size=10,
        n_init=4,
        n_expand=4,
        mcts_budget=1,
        ucb_c=1.0,
        p_random_replace=0.0,
        max_theta_pairs=4,
        disable_progress=True,
    )


def _context(tmp_path: Path) -> ProposalContext:
    return ProposalContext(
        schema_card=_schema_card(),
        dataset_context={},
        config=_config(),
        mcts_dir=tmp_path,
        archive_summaries=[],
    )


def test_refine_payload_theta_is_repaired_to_action_result(tmp_path: Path) -> None:
    parent = StrategyTheta(
        col_1ds=("age",),
        col_2ds=("age", "workclass"),
        col_ps=("age",),
        col_u="age",
    )
    payload = {
        "actions": [{"type": "add_col_2d", "new": "education"}],
        "theta": {
            "col_1ds": ["age"],
            "col_2ds": ["age", "workclass"],
            "col_ps": ["age"],
            "col_u": "age",
        },
        "prior_score": 0.8,
        "reason": "mismatch fixture",
    }

    proposal = _proposal_from_payload(payload, context=_context(tmp_path), parent_theta=parent)

    assert proposal is not None
    assert proposal.theta.col_2ds == ("age", "education", "workclass")
    assert proposal.action_validation["actions_match_theta"] is True
    assert proposal.action_validation["theta_source"] == "actions_repair"
    assert proposal.action_validation["theta_changed_after_repair"] is True


def test_refine_no_op_action_is_rejected(tmp_path: Path) -> None:
    parent = StrategyTheta(
        col_1ds=("age",),
        col_2ds=("age", "workclass"),
        col_ps=("age",),
        col_u="age",
    )
    payload = {
        "actions": [{"type": "add_col_1d", "new": "age"}],
        "theta": {
            "col_1ds": ["age"],
            "col_2ds": ["age", "workclass"],
            "col_ps": ["age"],
            "col_u": "age",
        },
        "prior_score": 0.8,
        "reason": "no-op fixture",
    }

    assert _proposal_from_payload(payload, context=_context(tmp_path), parent_theta=parent) is None


class _BadRandom:
    def random(self) -> float:
        return 0.0

    def choice(self, values: list[str]) -> str:
        return "age"


def test_random_replace_falls_back_when_replacement_action_is_invalid(tmp_path: Path) -> None:
    parent = StrategyTheta(
        col_1ds=("age", "workclass"),
        col_2ds=("age", "workclass"),
        col_ps=("age",),
        col_u="age",
    )
    proposal = StrategyProposal(
        theta=StrategyTheta(
            col_1ds=("age", "education"),
            col_2ds=("age", "workclass"),
            col_ps=("age",),
            col_u="age",
        ),
        actions=[ThetaAction(type="replace_col_1d", old="workclass", new="education")],
        prior_score=0.82,
        reason="valid original",
    )

    replaced = ProposalProvider().maybe_random_replace(
        proposal,
        parent_theta=parent,
        context=_context(tmp_path),
        rng=_BadRandom(),
        p_random_replace=1.0,
    )

    assert replaced is proposal
