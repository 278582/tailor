from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from postprocess.cards import build_and_save_cards
from postprocess.tabdiff_protocol import resolve_tabdiff_selection_context
from post_selection_tool.io import ensure_dir, save_json

from llm_mcts_tool.strategy import StrategyProposal, canonical_key, theta_id
from llm_mcts_tool.v2_pipeline import (
    DATASET_SEARCH_FILL_FIELDS,
    SNode,
    SUPPORTED_V2_DATASETS,
    ThetaNode,
    V2MCTSConfig,
    V2RunResult,
    _attach_result,
    _best_node,
    _budget_snapshot_payload,
    _canonical_s_key,
    _compact_action_dict,
    _compact_action_validation,
    _json_safe,
    _mcts_dir,
    _node_selection_score,
    _refresh_s_best_reward,
    _refresh_theta_best_reward_path,
    _rollout_node,
    _schema_feature_columns,
    _should_stop_for_hard_no_improve,
    _write_run_state,
    adapt_census_v2_source_mode,
    build_s_pool,
    resolve_sources,
    select_s_pools,
    theta_to_dict,
)
from llm_mcts_tool.run_observability import initialize_run_observability, maybe_record_budget_snapshot

from .config import (
    PROPOSAL_SOURCE,
    SEARCH_BACKEND,
    GreedyColumnConfig,
    build_v2_config,
    greedy_run_metadata,
)
from .random_edits import greedy_start_theta, propose_random_neighbors, greedy_set_bounds


def choose_commit_child(current: ThetaNode, children: list[ThetaNode]) -> ThetaNode | None:
    """Return the best successful child if it strictly beats the committed Theta."""
    successful = [node for node in children if node.status == "success"]
    if not successful or current.status != "success":
        return None
    best = max(
        successful,
        key=lambda node: (
            _node_selection_score(node),
            float(node.audit_metrics.get("utility_exact") or 0.0),
            node.node_id,
        ),
    )
    if _node_selection_score(best) > _node_selection_score(current) + 1e-12:
        return best
    return None


def _neutral_dataset_context(config: V2MCTSConfig, schema_card: dict[str, Any]) -> dict[str, Any]:
    """Diagnostics context without prompt-pack column priorities."""
    return {
        "dataset": config.dataset_name,
        "target_column": schema_card.get("target_column"),
        "columns": {"feature": _schema_feature_columns(schema_card)},
    }


def _evaluate_proposal(
    *,
    proposal: StrategyProposal,
    config: V2MCTSConfig,
    schema_card: dict[str, Any],
    dataset_context: dict[str, Any],
    test_df: pd.DataFrame,
    s_node: SNode,
    parent: ThetaNode | None,
    theta_nodes: dict[str, ThetaNode],
    visited: set[str],
    rollouts_dir: Path,
    event_trace: list[dict[str, Any]],
    checkpoint: Callable[[], None],
    rollout_fn: Callable[..., Any] = _rollout_node,
) -> ThetaNode | None:
    key = json.dumps(
        {"s": _canonical_s_key(s_node.pool_units), "theta": canonical_key(proposal.theta)},
        sort_keys=True,
    )
    if key in visited:
        event_trace.append(
            {
                "event": "duplicate_strategy",
                "s_id": s_node.s_id,
                "theta_id": theta_id(proposal.theta),
                "generation_mode": SEARCH_BACKEND,
            }
        )
        checkpoint()
        return None
    visited.add(key)
    node_id = f"n_{len(theta_nodes):06d}"
    theta_node = ThetaNode(
        node_id=node_id,
        s_id=s_node.s_id,
        theta=proposal.theta,
        theta_id=theta_id(proposal.theta),
        parent_node_id=None if parent is None else parent.node_id,
        actions=[
            compact
            for compact in (_compact_action_dict(action) for action in proposal.actions)
            if compact
        ],
        proposal_action_validation=_compact_action_validation(proposal.action_validation),
        llm_score=0.0,
        reason=proposal.reason,
    )
    theta_nodes[node_id] = theta_node
    s_node.theta_node_ids.append(node_id)
    checkpoint()
    try:
        result = rollout_fn(config=config, s_node=s_node, theta_node=theta_node, rollouts_dir=rollouts_dir)
    except Exception as exc:
        theta_node.status = "failed"
        theta_node.error = str(exc)
        event_trace.append(
            {
                "event": "rollout_failed",
                "node_id": node_id,
                "s_id": s_node.s_id,
                "error": str(exc),
                "generation_mode": SEARCH_BACKEND,
            }
        )
    else:
        _attach_result(
            theta_node,
            result,
            config=config,
            schema_card=schema_card,
            dataset_context=dataset_context,
            test_df=test_df,
        )
        theta_node.visits = 1
        _refresh_theta_best_reward_path(theta_nodes, node_id)
        _refresh_s_best_reward(s_node, theta_nodes)
        event_trace.append(
            {
                "event": "rollout_success",
                "node_id": node_id,
                "s_id": s_node.s_id,
                "theta_id": theta_node.theta_id,
                "reward": float(theta_node.reward),
                "selection_score": float(_node_selection_score(theta_node)),
                "generation_mode": SEARCH_BACKEND,
                "proposal_source": PROPOSAL_SOURCE,
                "proposal_reason": theta_node.reason,
            }
        )
    checkpoint()
    return theta_node


def run_greedy_column_search(
    config: GreedyColumnConfig,
    *,
    rollout_fn: Callable[..., Any] = _rollout_node,
) -> V2RunResult:
    v2 = build_v2_config(config)
    extended = config.reference_split_dir is not None or config.evaluation_dcr_align_mode != "paired_nn_coreset" or config.evaluation_enable_hold_dcr or config.evaluation_seed is not None or config.evaluation_density_reference_size is not None or config.evaluation_dcr_repeats != 10 or config.evaluation_dcr_seed is not None
    if extended and rollout_fn is _rollout_node:
        from .rollout_adapter import rollout_node
        rollout_fn = rollout_node
    if config.evaluation_dcr_align_mode == "query_batches_full_train_v1":
        if not config.evaluation_enable_hold_dcr or not 1 <= config.evaluation_dcr_repeats <= 4:
            raise ValueError("query-batch DCR requires hold DCR and 1..4 repeats")
    missing = [key for key in DATASET_SEARCH_FILL_FIELDS if getattr(v2, key) is None]
    if missing:
        raise ValueError(f"dataset search defaults missing after materialize: {', '.join(missing)}")
    if v2.dataset_name not in SUPPORTED_V2_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_V2_DATASETS))
        raise ValueError(f"greedy runner supports dataset_name in {{{supported}}}")
    if v2.mode != "single":
        raise ValueError("greedy column search is single-source; do not pass mixed S-pools")

    mcts_dir = _mcts_dir(v2)
    context_dir = ensure_dir(mcts_dir / "context")
    s_dir = ensure_dir(mcts_dir / "s_nodes")
    rollouts_dir = ensure_dir(mcts_dir / "rollouts")
    ensure_dir(mcts_dir / "tree")
    ensure_dir(mcts_dir / "archive")
    initialize_run_observability(mcts_dir)

    dataset_ctx = resolve_tabdiff_selection_context(
        dataset_name=v2.dataset_name,
        seed=v2.seed,
        holdout_fraction=v2.holdout_fraction,
        **({"reference_split_dir": config.reference_split_dir} if config.reference_split_dir is not None else {}),
    )
    cards = build_and_save_cards(
        train_df=dataset_ctx.train_df.copy(),
        output_dir=context_dir / "cards",
        seed=v2.seed,
        dataset_name=v2.dataset_name,
        target_column=dataset_ctx.target_column,
        categorical_columns=dataset_ctx.categorical_columns,
        numerical_columns=dataset_ctx.numerical_columns,
        discrete_numerical_columns=dataset_ctx.discrete_numerical_columns,
        privacy_sensitive_columns=dataset_ctx.privacy_sensitive_columns,
    )
    schema_card = cards.schema_card
    stats_card = cards.stats_card
    column_caps = {
        name: greedy_set_bounds(universe, focus_size=config.greedy_focus_size,
                                max_size=config.greedy_max_size,
                                max_fraction=config.greedy_max_fraction)[1]
        for name, universe in {
            "col_1ds": list(schema_card["column_order"]),
            "col_2ds": list(schema_card["column_order"]),
            "col_ps": _schema_feature_columns(schema_card),
        }.items()
    }
    save_json(context_dir / "column_caps.json", column_caps)
    dataset_prompt_context = _neutral_dataset_context(v2, schema_card)
    sources = resolve_sources(v2)
    v2 = adapt_census_v2_source_mode(v2, sources)
    save_json(context_dir / "source_registry.json", _json_safe({key: info.__dict__ for key, info in sources.items()}))
    save_json(context_dir / "dataset_prompt_context.json", dataset_prompt_context)
    save_json(
        context_dir / "run_config.json",
        {
            **_json_safe(v2.__dict__),
            **greedy_run_metadata(config),
            "effective_column_caps": column_caps,
        },
    )

    rng = random.Random(v2.seed)
    s_nodes: dict[str, SNode] = {}
    theta_nodes: dict[str, ThetaNode] = {}
    visited: set[str] = set()
    event_trace: list[dict[str, Any]] = []

    def checkpoint_run_state(final_status: str = "running") -> None:
        _write_run_state(
            mcts_dir=mcts_dir,
            s_nodes=s_nodes,
            theta_nodes=theta_nodes,
            event_trace=event_trace,
            final_node=_best_node(theta_nodes),
            final_status=final_status,
            include_source_context=False,
        )

    checkpoint_run_state("initialized")

    def register_s(pool_payload: dict[str, Any]) -> SNode:
        s_id = f"s_{len(s_nodes):06d}"
        synthetic_csv, row_map, report = build_s_pool(
            s_id=s_id,
            pool_units=pool_payload["pool_units"],
            config=v2,
            sources=sources,
            train_rows=len(dataset_ctx.train_df),
            output_dir=s_dir / s_id,
            column_order=list(schema_card["column_order"]),
            schema_card=schema_card,
            stats_card=stats_card,
        )
        s_node = SNode(
            s_id=s_id,
            pool_units=pool_payload["pool_units"],
            synthetic_csv=synthetic_csv,
            synthetic_row_map=row_map,
            llm_score=0.0,
            reason="greedy_column_fixed_single_source",
        )
        s_nodes[s_id] = s_node
        event_trace.append(
            {
                "event": "create_s",
                "s_id": s_id,
                "pool_units": s_node.pool_units,
                "manifest": report,
                "generation_mode": SEARCH_BACKEND,
                "reason": s_node.reason,
            }
        )
        return s_node

    initial_s = select_s_pools(
        config=v2,
        sources=sources,
        client=None,
        n=1,
        phase="init",
        source_profiles={},
        real_utility_reference={},
        s_nodes=s_nodes,
        theta_nodes=theta_nodes,
        trace_dir=mcts_dir / "llm_calls",
    )
    if not initial_s:
        raise RuntimeError("greedy column search could not build a single-source S pool")
    s_node = register_s(initial_s[0])

    def evaluate(proposal: StrategyProposal, parent: ThetaNode | None) -> ThetaNode | None:
        return _evaluate_proposal(
            proposal=proposal,
            config=v2,
            schema_card=schema_card,
            dataset_context=dataset_prompt_context,
            test_df=dataset_ctx.test_df.copy(),
            s_node=s_node,
            parent=parent,
            theta_nodes=theta_nodes,
            visited=visited,
            rollouts_dir=rollouts_dir,
            event_trace=event_trace,
            checkpoint=lambda: checkpoint_run_state("running"),
            rollout_fn=rollout_fn,
        )

    start_theta = greedy_start_theta(
        schema_card,
        focus_size=int(config.greedy_focus_size),
        max_size=int(config.greedy_max_size),
        max_fraction=config.greedy_max_fraction,
    )
    current = evaluate(
        StrategyProposal(
            theta=start_theta,
            actions=[],
            prior_score=0.0,
            reason="greedy_schema_prefix",
            action_validation={
                "proposal_source": "greedy_schema_prefix",
                "llm": False,
                "mcts": False,
                "greedy_focus_size": int(config.greedy_focus_size),
                "greedy_max_size": int(config.greedy_max_size),
            },
        ),
        parent=None,
    )
    if current is None or current.status != "success":
        checkpoint_run_state("failed_no_successful_rollouts")
        return V2RunResult(
            mcts_dir=mcts_dir,
            final_node=None,
            final_status="failed_no_successful_rollouts",
            baseline_reward=None,
        )

    event_trace.append(
        {
            "event": "evaluate_greedy_start_theta",
            "node_id": current.node_id,
            "theta": theta_to_dict(current.theta),
            "selection_score": float(_node_selection_score(current)),
            "generation_mode": SEARCH_BACKEND,
        }
    )
    baseline_reward = float(_node_selection_score(current))
    no_improve = 0
    committed_steps = 0
    early_stop_stagnation_threshold = int(v2.early_stop_stagnation_events)

    for event_idx in range(int(v2.mcts_budget)):
        before = _node_selection_score(current)
        exclude_keys = {
            canonical_key(node.theta)
            for node in theta_nodes.values()
            if node.s_id == s_node.s_id
        }
        proposals = propose_random_neighbors(
            current.theta,
            schema_card=schema_card,
            n=int(v2.theta_proposals_per_event),
            rng=rng,
            exclude_keys=exclude_keys,
            focus_size=int(config.greedy_focus_size),
            max_size=int(config.greedy_max_size),
            max_fraction=config.greedy_max_fraction,
        )
        created: list[ThetaNode] = []
        for proposal in proposals:
            node = evaluate(proposal, parent=current)
            if node is not None:
                created.append(node)
        parent_node_id = current.node_id
        commit = choose_commit_child(current, created)
        improved = commit is not None
        if commit is not None:
            current = commit
            committed_steps += 1
            no_improve = 0
        else:
            no_improve += 1
        event_trace.append(
            {
                "event": "greedy_column_step",
                "event_idx": int(event_idx),
                "s_id": s_node.s_id,
                "parent_node_id": parent_node_id,
                "committed_node_id": None if commit is None else commit.node_id,
                "created_nodes": [node.node_id for node in created],
                "generation_mode": SEARCH_BACKEND,
                "proposal_source": PROPOSAL_SOURCE,
                "theta_uct_path": [],
                "s_ucb_selection": None,
                "best_selection_score_before": float(before),
                "best_selection_score_after": float(_node_selection_score(current)),
                "improved": bool(improved),
                "committed": bool(improved),
                "no_improve_expand_events": int(no_improve),
                "hard_no_improve_expand_events": int(no_improve),
                "early_stop_stagnation_threshold": int(early_stop_stagnation_threshold),
            }
        )
        maybe_record_budget_snapshot(
            mcts_dir=mcts_dir,
            event_idx=int(event_idx),
            snapshot_payload=_budget_snapshot_payload(
                s_nodes=s_nodes,
                theta_nodes=theta_nodes,
                best_node=_best_node(theta_nodes),
                include_source_context=False,
            ),
        )
        checkpoint_run_state("running")
        if _should_stop_for_hard_no_improve(no_improve, early_stop_stagnation_threshold):
            event_trace.append(
                {
                    "event": "stop_hard_no_improve",
                    "event_idx": int(event_idx),
                    "hard_no_improve_expand_events": int(no_improve),
                    "early_stop_stagnation_threshold": int(early_stop_stagnation_threshold),
                    "reason": "greedy current theta did not improve for more than early_stop_stagnation_events steps",
                    "generation_mode": SEARCH_BACKEND,
                }
            )
            break

    final_node = _best_node(theta_nodes)
    if final_node is None:
        final_status = "failed_no_successful_rollouts"
    elif final_node.guard_pass and final_node.reward_available:
        final_status = "guard_pass"
    else:
        final_status = "guard_failed_best_effort"
    save_json(
        mcts_dir / "greedy_summary.json",
        {
            **greedy_run_metadata(config),
            "effective_column_caps": column_caps,
            "final_status": final_status,
            "final_node_id": None if final_node is None else final_node.node_id,
            "final_theta_id": None if final_node is None else final_node.theta_id,
            "baseline_reward": baseline_reward,
            "final_selection_score": None if final_node is None else float(_node_selection_score(final_node)),
            "committed_steps": int(committed_steps),
            "evaluated_thetas": int(len(theta_nodes)),
            "llm_calls": 0,
        },
    )
    checkpoint_run_state(final_status)
    return V2RunResult(
        mcts_dir=mcts_dir,
        final_node=final_node,
        final_status=final_status,
        baseline_reward=baseline_reward,
    )
