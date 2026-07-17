from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"archive file not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"archive file is empty: {path}")
    return records


def _stdev_or_zero(values: list[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 2:
        return 0.0
    return float(statistics.stdev(clean))


def _base_score(node: dict[str, Any]) -> float:
    return float(node.get("best_reward") or 0.0) + 0.05 * float(node.get("llm_score") or 0.0)


def _uct_explore_scale(base_scores: list[float], ucb_c: float) -> float:
    return max(float(ucb_c) * _stdev_or_zero(base_scores), 0.01)


def _uct_score(base_score: float, explore_scale: float, total_visits: int, visits: int) -> float:
    return float(base_score) + float(explore_scale) * math.sqrt(
        math.log(float(max(0, int(total_visits))) + 1.0) / (float(max(0, int(visits))) + 1.0)
    )


def _node_depth(node: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> int:
    depth = 0
    current = node
    seen: set[str] = set()
    while current.get("parent_node_id") is not None:
        parent_id = str(current["parent_node_id"])
        if parent_id in seen or parent_id not in by_id:
            break
        seen.add(parent_id)
        current = by_id[parent_id]
        depth += 1
    return depth


def simulate_theta_uct_archive(
    archive_path: Path,
    *,
    rounds: int,
    ucb_c: float,
    sweep_sources: bool,
) -> dict[str, Any]:
    nodes = _load_jsonl(archive_path)
    by_id = {str(node["node_id"]): node for node in nodes}
    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for node in nodes:
        groups.setdefault((str(node["s_id"]), node.get("parent_node_id")), []).append(node)
    for group in groups.values():
        group.sort(key=lambda item: str(item["node_id"]))

    s_ids = sorted({str(node["s_id"]) for node in nodes})
    root_base_by_s = {
        s_id: max(_base_score(node) for node in groups[(s_id, None)])
        for s_id in s_ids
        if (s_id, None) in groups
    }
    if not root_base_by_s:
        raise ValueError("archive has no root theta groups")

    theta_visits = {str(node["node_id"]): 0 for node in nodes}
    selected_by_group: dict[tuple[str, str | None], set[str]] = {key: set() for key in groups}
    selected_roots: set[str] = set()
    selected_sources: set[str] = set()
    non_greedy_examples: list[dict[str, Any]] = []

    theta_greedy = 0
    theta_explore = 0
    max_selected_depth = 0

    traversal_sources = sorted(root_base_by_s) if sweep_sources else [next(iter(root_base_by_s))]

    for s_id in traversal_sources:
        selected_sources.add(s_id)
        for total_visits in range(1, int(rounds) + 1):
            parent_id: str | None = None
            depth = 0
            while True:
                all_siblings = groups.get((s_id, parent_id), [])
                candidates = [node for node in all_siblings if node.get("status") == "success"]
                if not candidates:
                    break
                base_scores = [_base_score(node) for node in all_siblings] or [_base_score(node) for node in candidates]
                explore_scale = _uct_explore_scale(base_scores, ucb_c)
                selected = max(
                    candidates,
                    key=lambda node: (
                        _uct_score(_base_score(node), explore_scale, total_visits, theta_visits[str(node["node_id"])]),
                        str(node["node_id"]),
                    ),
                )
                greedy = max(candidates, key=lambda node: (_base_score(node), str(node["node_id"])))
                selected_id = str(selected["node_id"])
                greedy_id = str(greedy["node_id"])
                if selected_id == greedy_id:
                    theta_greedy += 1
                else:
                    theta_explore += 1
                    if len(non_greedy_examples) < 12:
                        non_greedy_examples.append(
                            {
                                "round": int(total_visits),
                                "depth": int(depth),
                                "s_id": s_id,
                                "parent_node_id": parent_id,
                                "selected_node_id": selected_id,
                                "greedy_node_id": greedy_id,
                                "selected_base_score": round(_base_score(selected), 6),
                                "greedy_base_score": round(_base_score(greedy), 6),
                            }
                        )
                theta_visits[selected_id] += 1
                selected_by_group.setdefault((s_id, parent_id), set()).add(selected_id)
                if parent_id is None:
                    selected_roots.add(selected_id)
                max_selected_depth = max(max_selected_depth, depth)
                if (s_id, selected_id) not in groups:
                    break
                parent_id = selected_id
                depth += 1

    archive_max_depth = max(_node_depth(node, by_id) for node in nodes)
    backtrack_groups = sum(1 for selected in selected_by_group.values() if len(selected) > 1)
    theta_ratio = float(theta_greedy / max(theta_explore, 1))
    return {
        "archive": str(archive_path),
        "rounds_per_source": int(rounds),
        "simulated_source_count": int(len(traversal_sources)),
        "total_simulated_rounds": int(rounds) * int(len(traversal_sources)),
        "ucb_c": float(ucb_c),
        "node_count": int(len(nodes)),
        "s_count": int(len(root_base_by_s)),
        "archive_max_depth": int(archive_max_depth),
        "max_selected_depth": int(max_selected_depth),
        "theta_greedy_decisions": int(theta_greedy),
        "theta_explore_decisions": int(theta_explore),
        "theta_greedy_to_explore_ratio": round(theta_ratio, 6),
        "selected_source_count": int(len(selected_sources)),
        "selected_root_count": int(len(selected_roots)),
        "backtrack_group_count": int(backtrack_groups),
        "non_greedy_examples": non_greedy_examples,
    }


def check_summary(summary: dict[str, Any], args: argparse.Namespace, *, require_multi_source: bool) -> list[str]:
    failures: list[str] = []
    if require_multi_source and int(summary["s_count"]) < 2:
        failures.append(f"expected mixed archive with >=2 S pools, got {summary['s_count']}")
    if not require_multi_source and int(summary["s_count"]) != 1:
        failures.append(f"expected single archive with exactly 1 S pool, got {summary['s_count']}")
    if int(summary["theta_explore_decisions"]) < int(args.min_explore):
        failures.append(
            f"theta exploration too small: {summary['theta_explore_decisions']} < {int(args.min_explore)}"
        )
    if int(summary["theta_greedy_decisions"]) <= int(summary["theta_explore_decisions"]):
        failures.append(
            "theta greedy decisions should be greater than theta exploration decisions: "
            f"{summary['theta_greedy_decisions']} <= {summary['theta_explore_decisions']}"
        )
    if float(summary["theta_greedy_to_explore_ratio"]) > float(args.max_ratio):
        failures.append(
            f"theta greedy/explore ratio too high: {summary['theta_greedy_to_explore_ratio']} > {args.max_ratio}"
        )
    if int(summary["max_selected_depth"]) < int(args.min_selected_depth):
        failures.append(
            f"selected depth too shallow: {summary['max_selected_depth']} < {int(args.min_selected_depth)}"
        )
    if int(summary["backtrack_group_count"]) < int(args.min_backtrack_groups):
        failures.append(
            f"backtrack groups too small: {summary['backtrack_group_count']} < {int(args.min_backtrack_groups)}"
        )
    if int(summary["selected_root_count"]) < int(args.min_selected_roots):
        failures.append(
            f"selected root count too small: {summary['selected_root_count']} < {int(args.min_selected_roots)}"
        )
    if require_multi_source and int(summary["selected_source_count"]) < int(args.min_selected_sources):
        failures.append(
            f"selected source count too small: {summary['selected_source_count']} < {int(args.min_selected_sources)}"
        )
    return failures


def run_smoke(args: argparse.Namespace, *, require_multi_source: bool) -> int:
    summary = simulate_theta_uct_archive(
        Path(args.archive),
        rounds=int(args.rounds),
        ucb_c=float(args.ucb_c),
        sweep_sources=bool(require_multi_source),
    )
    failures = check_summary(summary, args, require_multi_source=require_multi_source)
    payload = {"ok": not failures, "failures": failures, "summary": summary}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not failures else 1
