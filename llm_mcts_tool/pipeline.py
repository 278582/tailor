from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from post_selection_tool.io import ensure_dir, save_json
from postprocess.cards import build_and_save_cards
from postprocess.tabdiff_protocol import resolve_tabdiff_selection_context

from .archive import MCTSArchive, select_final_node, write_final_outputs
from .config import MCTSGuideConfig
from .prompt_context import build_init_prompt_context, build_refine_prompt_context, load_dataset_prompt_context
from .prompt_templates import init_prompt, refine_prompt
from .proposal import ProposalContext, ProposalProvider
from .rollout import GuidedRolloutConfig, GuidedRolloutResult, run_guided_pareto_rollout
from .strategy import StrategyProposal, canonical_key, theta_id
from .tree import MCTSNode, MCTSTree
from .ucb_mean_max import selection, update_Q_upward


@dataclass
class MCTSRunResult:
    mcts_dir: Path
    tree: MCTSTree
    final_node: MCTSNode | None
    final_status: str
    archive: MCTSArchive


def _mcts_dir(config: MCTSGuideConfig) -> Path:
    artifact_root = Path("artifacts/llm_mcts") if config.artifact_dir is None else Path(config.artifact_dir)
    return ensure_dir(artifact_root / config.exp_name / "mcts")


def _build_context(config: MCTSGuideConfig, archive: MCTSArchive) -> ProposalContext:
    dataset_ctx = resolve_tabdiff_selection_context(
        dataset_name=config.dataset_name,
        seed=config.seed,
        holdout_fraction=config.holdout_fraction,
    )
    cards = build_and_save_cards(
        train_df=dataset_ctx.train_df.copy(),
        output_dir=archive.context_dir,
        seed=config.seed,
        dataset_name=config.dataset_name,
        target_column=dataset_ctx.target_column,
        categorical_columns=dataset_ctx.categorical_columns,
        numerical_columns=dataset_ctx.numerical_columns,
        discrete_numerical_columns=dataset_ctx.discrete_numerical_columns,
        privacy_sensitive_columns=dataset_ctx.privacy_sensitive_columns,
    )
    feature_columns = [
        column
        for column in cards.schema_card["column_order"]
        if not bool(cards.schema_card["columns"][column].get("is_target", False))
    ]
    save_json(
        archive.context_dir / "schema_summary.json",
        {
            "dataset_name": config.dataset_name,
            "target_column": cards.schema_card["target_column"],
            "feature_columns": feature_columns,
            "column_types": {
                column: info.get("type") for column, info in cards.schema_card.get("columns", {}).items()
            },
        },
    )
    save_json(
        archive.context_dir / "baseline_diagnostics.json",
        {
            "status": "not_run_in_batch_c_dry_context",
            "note": "Batch C mock dry-run does not run formal baseline experiments.",
        },
    )
    dataset_prompt_context = load_dataset_prompt_context(
        dataset_name=config.dataset_name,
        schema_card=cards.schema_card,
        prompt_pack_dir=config.prompt_pack_dir,
    )
    save_json(archive.context_dir / "dataset_prompt_context.json", dataset_prompt_context)
    init_context = build_init_prompt_context(
        schema_card=cards.schema_card,
        dataset_context=dataset_prompt_context,
        baseline_diagnostics={
            "status": "not_run_in_phase9_freeze_check",
            "note": "Implementation freeze check only; not a formal experiment.",
        },
    )
    init_context["_prompt_pack_dir"] = str(config.prompt_pack_dir)
    (archive.context_dir / "init_prompt.txt").write_text(
        init_prompt(init_context, config.n_init),
        encoding="utf-8",
    )
    return ProposalContext(
        schema_card=cards.schema_card,
        dataset_context=dataset_prompt_context,
        config=config,
        mcts_dir=archive.mcts_dir,
        archive_summaries=[],
        init_prompt_context=init_context,
    )


def _action_dicts(proposal: StrategyProposal) -> list[dict[str, Any]]:
    return [
        {"type": action.type, "column": action.column, "old": action.old, "new": action.new}
        for action in proposal.actions
    ]


def _new_child_node(
    *,
    tree: MCTSTree,
    parent: MCTSNode,
    proposal: StrategyProposal,
) -> MCTSNode:
    tid = theta_id(proposal.theta)
    return MCTSNode(
        node_id=tree.next_node_id(),
        theta_id=tid,
        theta=proposal.theta,
        parent_id=parent.node_id,
        children_ids=[],
        depth=int(parent.depth) + 1,
        Q_self=0.0,
        Q=0.0,
        N=0,
        p=max(0.0, min(1.0, float(proposal.prior_score))),
        max_child_Q=0.0,
        is_leaf=True,
        rollout_id=tid,
        reward_available=False,
        guard_pass=False,
        actions=_action_dicts(proposal),
        action_validation=dict(proposal.action_validation),
        reason=proposal.reason,
    )


def _rollout_config(config: MCTSGuideConfig, archive: MCTSArchive, node: MCTSNode) -> GuidedRolloutConfig:
    if node.theta_id is None:
        raise ValueError("Cannot build rollout config for node without theta_id")
    return GuidedRolloutConfig(
        theta_id=node.theta_id,
        dataset_name=config.dataset_name,
        exp_name=config.exp_name,
        artifact_dir=archive.rollouts_dir,
        synthetic_csv=config.synthetic_csv,
        seed=config.seed,
        keep_k=config.keep_k,
        preselect_target=config.preselect_target,
        d_cur_size=config.d_cur_size,
        max_theta_pairs=config.max_theta_pairs,
        rollout_dir=archive.rollouts_dir / node.theta_id,
        d_cur_source=config.d_cur_source,
        holdout_fraction=config.holdout_fraction,
        source=config.source,
        eval_device=config.eval_device,
        nn_device=config.nn_device,
        utility_exact_evaluator=config.utility_exact_evaluator,
        density_reference_size=config.density_reference_size,
        save_validation_records=config.save_validation_records,
        save_internal_records=config.save_rollout_internal_records,
        disable_progress=config.disable_progress,
    )


def _attach_rollout_result(node: MCTSNode, result: GuidedRolloutResult) -> None:
    node.Q_self = float(result.reward)
    node.Q = float(result.reward)
    node.reward_available = bool(result.reward_available)
    node.guard_pass = bool(result.guard.get("pass", False))
    node.rollout_status = "success"
    node.rollout_dir = str(result.artifact_dir)
    node.search_objectives = dict(result.search_objectives)
    node.audit_metrics = dict(result.audit_metrics)
    node.guard = dict(result.guard)
    node.feedback = dict(result.feedback)


def _archive_summary(node: MCTSNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "theta_id": node.theta_id,
        "theta": node.to_dict().get("theta"),
        "Q_self": float(node.Q_self),
        "reward_available": bool(node.reward_available),
        "guard_pass": bool(node.guard_pass),
        "search_objectives": dict(node.search_objectives),
        "audit_metrics": dict(node.audit_metrics),
        "action_validation": dict(node.action_validation),
    }


def _expand_with_proposals(
    *,
    tree: MCTSTree,
    parent: MCTSNode,
    proposals: list[StrategyProposal],
    config: MCTSGuideConfig,
    context: ProposalContext,
    provider: ProposalProvider,
    archive: MCTSArchive,
    visited_keys: set[str],
    rng: random.Random,
) -> list[MCTSNode]:
    children: list[MCTSNode] = []
    for raw_proposal in proposals:
        proposal = provider.maybe_random_replace(
            raw_proposal,
            parent_theta=parent.theta,
            context=context,
            rng=rng,
            p_random_replace=config.p_random_replace,
        )
        key = canonical_key(proposal.theta)
        if key in visited_keys:
            archive.save_duplicate_theta(
                {
                    "parent_id": parent.node_id,
                    "theta_id": theta_id(proposal.theta),
                    "canonical_key": key,
                    "reason": proposal.reason,
                    "action_validation": dict(proposal.action_validation),
                }
            )
            continue
        visited_keys.add(key)
        child = _new_child_node(tree=tree, parent=parent, proposal=proposal)
        tree.add_child(parent.node_id, child)
        try:
            result = run_guided_pareto_rollout(_rollout_config(config, archive, child), proposal.theta)
        except Exception as exc:
            child.rollout_status = "failed"
            child.error = str(exc)
            child.rollout_dir = str(archive.rollouts_dir / (child.theta_id or child.node_id))
            archive.save_failed_rollout(child, str(exc))
        else:
            _attach_rollout_result(child, result)
            archive.save_rollout_summary(result, child)
            context.archive_summaries.insert(0, _archive_summary(child))
            context.archive_summaries = context.archive_summaries[:32]
        children.append(child)
    return children


def run_mcts_with_provider(config: MCTSGuideConfig, provider: ProposalProvider) -> MCTSRunResult:
    archive = MCTSArchive(_mcts_dir(config))
    context = _build_context(config, archive)
    rng = random.Random(config.seed)
    tree = MCTSTree()
    root = MCTSNode(
        node_id="root",
        theta_id=None,
        theta=None,
        parent_id=None,
        children_ids=[],
        depth=0,
        Q_self=0.0,
        Q=0.0,
        N=0,
        p=1.0,
        max_child_Q=0.0,
        is_leaf=True,
        rollout_id=None,
        reward_available=False,
        guard_pass=False,
        rollout_status="root",
    )
    tree.add_node(root)
    visited_keys: set[str] = set()
    ucb_trace: list[dict[str, Any]] = []

    initial = provider.initial(context)
    _expand_with_proposals(
        tree=tree,
        parent=root,
        proposals=initial,
        config=config,
        context=context,
        provider=provider,
        archive=archive,
        visited_keys=visited_keys,
        rng=rng,
    )
    ucb_trace.append({"event": "initial_backup", "updates": update_Q_upward(tree, root.node_id)})

    for iteration in range(int(config.mcts_budget)):
        leaf, trace = selection(tree, root.node_id, config.ucb_c)
        ucb_trace.append({"event": "selection", "iteration": iteration, "leaf_id": leaf.node_id, "trace": trace})
        parent = tree.get(leaf.parent_id) if leaf.parent_id is not None else None
        siblings = [
            tree.get(child_id)
            for child_id in (parent.children_ids if parent is not None else [])
            if child_id != leaf.node_id
        ]
        refine_context = build_refine_prompt_context(
            node=leaf,
            parent=parent,
            siblings=siblings,
            archive=context.archive_summaries,
            feedback=leaf.feedback,
            schema_card=context.schema_card,
            dataset_context=context.dataset_context,
            existing_theta_keys=sorted(visited_keys),
            include_dataset_priors=config.refine_prompt_use_dataset_priors,
        )
        refine_context["_prompt_pack_dir"] = str(config.prompt_pack_dir)
        (archive.prompts_dir / f"refine_{leaf.node_id}.txt").write_text(
            refine_prompt(refine_context, config.n_expand),
            encoding="utf-8",
        )
        context.pending_refine_prompt_context = refine_context
        proposals = provider.expand(leaf, context)
        context.pending_refine_prompt_context = None
        children = _expand_with_proposals(
            tree=tree,
            parent=leaf,
            proposals=proposals,
            config=config,
            context=context,
            provider=provider,
            archive=archive,
            visited_keys=visited_keys,
            rng=rng,
        )
        ucb_trace.append(
            {
                "event": "backup",
                "iteration": iteration,
                "leaf_id": leaf.node_id,
                "children": [child.node_id for child in children],
                "updates": update_Q_upward(tree, leaf.node_id),
            }
        )

    archive.write_tree_snapshot(tree, ucb_trace)
    archive.write_archive_snapshot(tree)
    successful_nodes = tree.successful_nodes()
    if successful_nodes:
        final_node, final_status = select_final_node(successful_nodes)
        write_final_outputs(final_node, archive, final_status=final_status)
    else:
        final_node = None
        final_status = "failed_no_successful_rollouts"
        save_json(
            archive.final_dir / "final_selection_manifest.json",
            {
                "final_status": final_status,
                "theta_star": None,
                "final_pareto": None,
                "failed_rollouts": str(archive.archive_dir / "failed_rollouts.jsonl"),
            },
        )
    save_json(
        archive.mcts_dir / "run_summary.json",
        {
            "mcts_dir": str(archive.mcts_dir),
            "final_node_id": None if final_node is None else final_node.node_id,
            "final_theta_id": None if final_node is None else final_node.theta_id,
            "final_status": final_status,
            "successful_rollouts": len(successful_nodes),
            "failed_rollouts": len([node for node in tree.nodes.values() if node.rollout_status == "failed"]),
            "total_nodes": len(tree.nodes),
        },
    )
    return MCTSRunResult(
        mcts_dir=archive.mcts_dir,
        tree=tree,
        final_node=final_node,
        final_status=final_status,
        archive=archive,
    )
