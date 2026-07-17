from __future__ import annotations

import json
import sys
from pathlib import Path

from post_selection_tool.cli import config_from_args, parse_args


def _theta_path(tmp_path: Path) -> Path:
    path = tmp_path / "theta.json"
    path.write_text(
        json.dumps(
            {
                "theta": {
                    "col_1ds": ["age", "income"],
                    "col_2ds": ["age"],
                    "col_ps": ["age", "income"],
                    "col_u": "age",
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _parse_with(monkeypatch, argv: list[str]):
    monkeypatch.setattr(sys, "argv", ["post_selection_tool.cli", *argv])
    return parse_args()


def test_theta_guided_post_selection_auto_enables_reward_candidate_v2(monkeypatch, tmp_path: Path) -> None:
    args = _parse_with(monkeypatch, ["--theta-json", str(_theta_path(tmp_path))])

    config = config_from_args(args)

    assert config.reward_candidate_v2_enabled is True
    assert config.theta_guidance_report["reward_candidate_v2_enabled"] is True
    assert config.theta_guidance_report["reward_candidate_v2_auto_enabled"] is True


def test_theta_guided_post_selection_disable_reward_candidate_v2_wins(monkeypatch, tmp_path: Path) -> None:
    args = _parse_with(
        monkeypatch,
        ["--theta-json", str(_theta_path(tmp_path)), "--disable-reward-candidate-v2"],
    )

    config = config_from_args(args)

    assert config.reward_candidate_v2_enabled is False
    assert config.theta_guidance_report["reward_candidate_v2_enabled"] is False


def test_disable_reward_candidate_v2_pre_repair_only_disables_pre_repair(monkeypatch, tmp_path: Path) -> None:
    args = _parse_with(
        monkeypatch,
        ["--theta-json", str(_theta_path(tmp_path)), "--disable-reward-candidate-v2-pre-repair"],
    )

    config = config_from_args(args)

    assert config.reward_candidate_v2_enabled is True
    assert config.reward_candidate_v2_pre_repair_enabled is False


def test_theta_default_utility_balance_cli_flag(monkeypatch, tmp_path: Path) -> None:
    args = _parse_with(
        monkeypatch,
        ["--theta-json", str(_theta_path(tmp_path)), "--theta-default-utility-balance"],
    )

    config = config_from_args(args)

    assert config.theta_default_utility_balance is True


def test_disable_fidelity_ceiling_cli_flag(monkeypatch) -> None:
    args = _parse_with(monkeypatch, ["--disable-fidelity-ceiling"])

    config = config_from_args(args)

    assert config.fidelity_ceiling_enabled is False
