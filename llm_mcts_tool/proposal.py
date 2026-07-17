from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import MCTSGuideConfig
from .llm_client import LLMClient
from .prompt_context import build_init_prompt_context, build_refine_prompt_context
from .prompt_templates import init_prompt, refine_prompt
from .strategy import (
    StrategyProposal,
    StrategyTheta,
    ThetaAction,
    apply_actions,
    canonical_key,
    normalize_theta,
    repair_theta,
    validate_theta,
    validate_theta_actions,
)
from .tree import MCTSNode, theta_to_dict


@dataclass
class ProposalContext:
    schema_card: dict[str, Any]
    dataset_context: dict[str, Any]
    config: MCTSGuideConfig
    mcts_dir: Path
    archive_summaries: list[dict[str, Any]]
    init_prompt_context: dict[str, Any] | None = None
    pending_refine_prompt_context: dict[str, Any] | None = None

    @property
    def feature_columns(self) -> list[str]:
        columns = self.schema_card.get("columns", {})
        return [
            column
            for column in self.schema_card.get("column_order", [])
            if not bool(columns.get(column, {}).get("is_target", False))
        ]


class ProposalProvider:
    def initial(self, context: ProposalContext) -> list[StrategyProposal]:
        raise NotImplementedError

    def expand(self, node: MCTSNode, context: ProposalContext) -> list[StrategyProposal]:
        raise NotImplementedError

    def score_prior(self, theta: StrategyTheta, context: ProposalContext) -> float:
        return 0.5

    def maybe_random_replace(
        self,
        proposal: StrategyProposal,
        *,
        parent_theta: StrategyTheta | None,
        context: ProposalContext,
        rng: random.Random,
        p_random_replace: float,
    ) -> StrategyProposal:
        if parent_theta is None or not proposal.actions:
            return proposal
        if rng.random() >= float(p_random_replace):
            return proposal
        features = context.feature_columns
        if not features:
            return proposal
        new_actions: list[ThetaAction] = []
        for action in proposal.actions:
            action_type = action.type.strip().lower()
            if action_type in {
                "add_col_1d",
                "replace_col_1d",
                "add_col_2d",
                "replace_col_2d",
                "add_col_p",
                "replace_col_p",
                "replace_col_u",
            }:
                replacement = rng.choice(features)
                new_actions.append(ThetaAction(type=action.type, old=action.old, new=replacement, column=replacement))
            else:
                new_actions.append(action)
        action_validation = validate_theta_actions(parent_theta, new_actions, context.schema_card)
        if not action_validation.get("ok", False):
            return proposal
        theta = apply_actions(parent_theta, new_actions, context.schema_card)
        return StrategyProposal(
            theta=theta,
            actions=new_actions,
            prior_score=self.score_prior(theta, context),
            reason=f"random_replace_applied: {proposal.reason}",
            action_validation=action_validation,
        )


def _clip_prior(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.5
    return max(0.0, min(1.0, parsed))


def _proposal_from_payload(
    payload: dict[str, Any],
    *,
    context: ProposalContext,
    parent_theta: StrategyTheta | None = None,
) -> StrategyProposal | None:
    actions = [
        ThetaAction(
            type=str(item.get("type", "")),
            column=item.get("column"),
            old=item.get("old"),
            new=item.get("new"),
        )
        for item in payload.get("actions", [])
        if isinstance(item, dict)
    ]
    action_validation: dict[str, Any] = {}
    if parent_theta is not None:
        if not actions:
            return None
        action_validation = validate_theta_actions(parent_theta, actions, context.schema_card)
        if not action_validation.get("ok", False):
            return None
    theta_payload = payload.get("theta")
    theta_source = "payload" if isinstance(theta_payload, dict) else "actions"
    raw_theta_key = canonical_key(normalize_theta(theta_payload, context.schema_card)) if isinstance(theta_payload, dict) else None
    if isinstance(theta_payload, dict):
        theta = repair_theta(theta_payload, context.schema_card, random.Random(context.config.seed))
    elif parent_theta is not None and actions:
        theta = apply_actions(parent_theta, actions, context.schema_card)
    else:
        return None
    theta_changed_after_repair = bool(raw_theta_key and raw_theta_key != canonical_key(theta))
    report = validate_theta(theta, context.schema_card)
    if not report.ok:
        before_validation_repair_key = canonical_key(theta)
        theta = repair_theta(theta, context.schema_card, random.Random(context.config.seed + 17))
        theta_changed_after_repair = theta_changed_after_repair or before_validation_repair_key != canonical_key(theta)
        if not validate_theta(theta, context.schema_card).ok:
            return None
    action_validation = dict(action_validation)
    action_validation["theta_source"] = theta_source
    action_validation["theta_changed_after_repair"] = theta_changed_after_repair
    if parent_theta is not None and actions:
        actions_theta = apply_actions(parent_theta, actions, context.schema_card)
        actions_match_theta = canonical_key(actions_theta) == canonical_key(theta)
        if not actions_match_theta:
            before_action_repair_key = canonical_key(theta)
            theta = actions_theta
            theta_changed_after_repair = theta_changed_after_repair or before_action_repair_key != canonical_key(theta)
            warnings = list(action_validation.get("warnings", []))
            warnings.append("payload theta did not match actions; using action-derived theta")
            action_validation["warnings"] = warnings
            action_validation["theta_source"] = "actions_repair"
            action_validation["theta_changed_after_repair"] = theta_changed_after_repair
            action_validation["actions_match_theta"] = True
            if not validate_theta(theta, context.schema_card).ok:
                return None
        else:
            action_validation["actions_match_theta"] = True
    return StrategyProposal(
        theta=theta,
        actions=actions,
        prior_score=_clip_prior(payload.get("prior_score", 0.5)),
        reason=str(payload.get("reason", "")),
        action_validation=action_validation,
    )


class MockProposalProvider(ProposalProvider):
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def _sample_theta(self, context: ProposalContext) -> StrategyTheta:
        features = context.feature_columns
        if not features:
            raise ValueError("schema has no feature columns for theta proposal")
        n_features = len(features)
        n1 = min(max(2, min(3, n_features)), n_features)
        n2 = min(max(2, min(3, n_features)), n_features)
        np_ = min(max(1, min(3, n_features)), n_features)
        return StrategyTheta(
            col_1ds=tuple(sorted(self.rng.sample(features, k=n1))),
            col_2ds=tuple(sorted(self.rng.sample(features, k=n2))),
            col_ps=tuple(sorted(self.rng.sample(features, k=np_))),
            col_u=self.rng.choice(features),
        )

    def initial(self, context: ProposalContext) -> list[StrategyProposal]:
        proposals: list[StrategyProposal] = []
        for idx in range(int(context.config.n_init)):
            theta = self._sample_theta(context)
            proposals.append(
                StrategyProposal(
                    theta=theta,
                    actions=[],
                    prior_score=self.score_prior(theta, context),
                    reason=f"mock_initial_{idx}",
                )
            )
        return proposals

    def expand(self, node: MCTSNode, context: ProposalContext) -> list[StrategyProposal]:
        if node.theta is None:
            return self.initial(context)[: int(context.config.n_expand)]
        features = context.feature_columns
        actions_types = [
            "add_col_1d",
            "replace_col_1d",
            "add_col_2d",
            "replace_col_2d",
            "add_col_p",
            "replace_col_p",
            "replace_col_u",
        ]
        proposals: list[StrategyProposal] = []
        for idx in range(int(context.config.n_expand)):
            action_type = self.rng.choice(actions_types)
            old = None
            if action_type.endswith("1d") and node.theta.col_1ds:
                old = self.rng.choice(list(node.theta.col_1ds))
            elif action_type.endswith("2d") and node.theta.col_2ds:
                old = self.rng.choice(list(node.theta.col_2ds))
            elif action_type.endswith("_p") and node.theta.col_ps:
                old = self.rng.choice(list(node.theta.col_ps))
            elif action_type == "replace_col_u":
                old = node.theta.col_u
            action = ThetaAction(type=action_type, old=old, new=self.rng.choice(features))
            theta = apply_actions(node.theta, [action], context.schema_card)
            proposals.append(
                StrategyProposal(
                    theta=theta,
                    actions=[action],
                    prior_score=self.score_prior(theta, context),
                    reason=f"mock_expand_{node.node_id}_{idx}",
                )
            )
        return proposals

    def score_prior(self, theta: StrategyTheta, context: ProposalContext) -> float:
        feature_count = max(len(context.feature_columns), 1)
        coverage = len(set(theta.col_1ds) | set(theta.col_2ds) | set(theta.col_ps)) / float(feature_count)
        balance_bonus = 0.1 if theta.col_u in set(theta.col_1ds) | set(theta.col_2ds) | set(theta.col_ps) else 0.0
        return _clip_prior(0.35 + 0.5 * coverage + balance_bonus)


class JsonlProposalProvider(ProposalProvider):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def initial(self, context: ProposalContext) -> list[StrategyProposal]:
        proposals = [_proposal_from_payload(row, context=context) for row in self.rows]
        return [proposal for proposal in proposals if proposal is not None][: int(context.config.n_init)]

    def expand(self, node: MCTSNode, context: ProposalContext) -> list[StrategyProposal]:
        proposals = [_proposal_from_payload(row, context=context, parent_theta=node.theta) for row in self.rows]
        return [proposal for proposal in proposals if proposal is not None][: int(context.config.n_expand)]


class LLMProposalProvider(ProposalProvider):
    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self._call_index = 0

    def _trace_call(
        self,
        *,
        context: ProposalContext,
        schema_name: str,
        prompt: str,
    ) -> dict[str, Any]:
        trace_dir = context.mcts_dir / "llm_calls"
        trace_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self._call_index:06d}_{schema_name}"
        self._call_index += 1
        (trace_dir / f"{stem}.prompt.txt").write_text(prompt, encoding="utf-8")

        def save_last_call() -> None:
            call = getattr(self.client, "last_call", None) or {}
            request = call.get("request")
            if request is not None:
                (trace_dir / f"{stem}.request.json").write_text(
                    json.dumps(request, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            raw_response_text = call.get("raw_response_text")
            if isinstance(raw_response_text, str):
                (trace_dir / f"{stem}.raw_response.txt").write_text(raw_response_text, encoding="utf-8")
            message_content = call.get("message_content")
            if isinstance(message_content, str):
                (trace_dir / f"{stem}.message_content.txt").write_text(message_content, encoding="utf-8")

        try:
            payload = self.client.complete_json(prompt, schema_name)
        except Exception as exc:
            save_last_call()
            (trace_dir / f"{stem}.error.json").write_text(
                json.dumps({"schema_name": schema_name, "error": str(exc)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise
        save_last_call()
        (trace_dir / f"{stem}.parsed.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload

    def _parse_proposals(
        self,
        payload: dict[str, Any],
        *,
        context: ProposalContext,
        parent_theta: StrategyTheta | None = None,
    ) -> list[StrategyProposal]:
        raw = payload.get("proposals", [])
        if not isinstance(raw, list):
            return []
        proposals = [_proposal_from_payload(item, context=context, parent_theta=parent_theta) for item in raw if isinstance(item, dict)]
        return [proposal for proposal in proposals if proposal is not None]

    def initial(self, context: ProposalContext) -> list[StrategyProposal]:
        prompt_context = context.init_prompt_context or build_init_prompt_context(
            schema_card=context.schema_card,
            dataset_context=context.dataset_context,
        )
        prompt = init_prompt(prompt_context, context.config.n_init)
        payload = self._trace_call(context=context, schema_name="init_proposals", prompt=prompt)
        return self._parse_proposals(payload, context=context)[: int(context.config.n_init)]

    def expand(self, node: MCTSNode, context: ProposalContext) -> list[StrategyProposal]:
        prompt_context = context.pending_refine_prompt_context or build_refine_prompt_context(
            node=node,
            parent=None,
            siblings=[],
            archive=context.archive_summaries,
            schema_card=context.schema_card,
            dataset_context=context.dataset_context,
            include_dataset_priors=context.config.refine_prompt_use_dataset_priors,
        )
        prompt = refine_prompt(prompt_context, context.config.n_expand)
        payload = self._trace_call(context=context, schema_name=f"refine_{node.node_id}", prompt=prompt)
        return self._parse_proposals(payload, context=context, parent_theta=node.theta)[: int(context.config.n_expand)]

    def score_prior(self, theta: StrategyTheta, context: ProposalContext) -> float:
        return 0.5
