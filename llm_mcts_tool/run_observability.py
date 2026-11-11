from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from post_selection_tool.io import ensure_dir, save_json

LLM_USAGE_TYPES = (
    "real_utility_profile_summary",
    "source_profile_summary",
    "init_select_syn",
    "refine_select_syn",
    "init_node",
    "refine_node",
    "init_node_diagnosis",
    "refine_node_diagnosis",
)

BUDGET_SNAPSHOTS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
BUDGET_UNIT = "expansion_event"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def llm_usage_type(schema_name: str) -> str:
    name = str(schema_name or "")
    if "real_utility_profile_summary" in name:
        return "real_utility_profile_summary"
    if "source_profile_summary" in name:
        return "source_profile_summary"
    if "init_select_syn" in name:
        return "init_select_syn"
    if "refine_select_syn" in name:
        return "refine_select_syn"
    if "init_node_diagnosis" in name:
        return "init_node_diagnosis"
    if "refine_node_diagnosis" in name:
        return "refine_node_diagnosis"
    if "init_node" in name:
        return "init_node"
    if "refine_node" in name:
        return "refine_node"
    return "other"


def llm_latency_dir(mcts_dir: Path) -> Path:
    return ensure_dir(Path(mcts_dir) / "llm_latency")


def budget_snapshot_dir(mcts_dir: Path) -> Path:
    return ensure_dir(Path(mcts_dir) / "budget_snapshots")


def budget_snapshot_filename(budget: int) -> str:
    return f"budget_{int(budget):03d}.json"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _empty_usage_stats() -> dict[str, Any]:
    return {
        "n": 0,
        "n_error": 0,
        "n_mock": 0,
        "total_sec": 0.0,
        "avg_sec": None,
        "min_sec": None,
        "max_sec": None,
        "last_sec": None,
        "last_status": None,
        "last_recorded_at": None,
    }


def _empty_latency_running() -> dict[str, Any]:
    return {
        "updated_at": utc_now_iso(),
        "avg_includes": "success",
        "usage_types": {name: _empty_usage_stats() for name in LLM_USAGE_TYPES},
    }


def _empty_budget_running() -> dict[str, Any]:
    return {
        "updated_at": utc_now_iso(),
        "budget_unit": BUDGET_UNIT,
        "targets": list(BUDGET_SNAPSHOTS),
        "reached": {},
        "pending": list(BUDGET_SNAPSHOTS),
    }


def initialize_run_observability(mcts_dir: Path) -> None:
    latency_dir = llm_latency_dir(mcts_dir)
    budget_dir = budget_snapshot_dir(mcts_dir)
    save_json(latency_dir / "running.json", _empty_latency_running())
    save_json(budget_dir / "running.json", _empty_budget_running())
    (latency_dir / "history.jsonl").write_text("", encoding="utf-8")
    (budget_dir / "snapshots.jsonl").write_text("", encoding="utf-8")


def _load_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(fallback)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(fallback)
    return payload if isinstance(payload, dict) else dict(fallback)


def record_llm_call_latency(
    *,
    mcts_dir: Path,
    schema_name: str,
    elapsed_sec: float,
    status: str,
    call_dir: Path | str | None = None,
) -> dict[str, Any]:
    usage_type = llm_usage_type(schema_name)
    elapsed = max(0.0, float(elapsed_sec))
    recorded_at = utc_now_iso()
    latency_dir = llm_latency_dir(mcts_dir)
    running_path = latency_dir / "running.json"
    running = _load_json(running_path, _empty_latency_running())
    usage_types = running.setdefault("usage_types", {})
    if not isinstance(usage_types, dict):
        usage_types = {}
        running["usage_types"] = usage_types
    stats = usage_types.get(usage_type)
    if not isinstance(stats, dict):
        stats = _empty_usage_stats()
        usage_types[usage_type] = stats
    for key, value in _empty_usage_stats().items():
        stats.setdefault(key, value)

    status_name = str(status or "unknown")
    if status_name == "success":
        stats["n"] = int(stats.get("n") or 0) + 1
        stats["total_sec"] = float(stats.get("total_sec") or 0.0) + elapsed
        stats["avg_sec"] = float(stats["total_sec"]) / float(stats["n"])
        stats["min_sec"] = elapsed if stats.get("min_sec") is None else min(float(stats["min_sec"]), elapsed)
        stats["max_sec"] = elapsed if stats.get("max_sec") is None else max(float(stats["max_sec"]), elapsed)
    elif status_name == "mock":
        stats["n_mock"] = int(stats.get("n_mock") or 0) + 1
    else:
        stats["n_error"] = int(stats.get("n_error") or 0) + 1
    stats["last_sec"] = elapsed
    stats["last_status"] = status_name
    stats["last_recorded_at"] = recorded_at
    usage_types[usage_type] = stats
    running["updated_at"] = recorded_at
    running["avg_includes"] = "success"
    save_json(running_path, running)

    history_record = {
        "recorded_at": recorded_at,
        "usage_type": usage_type,
        "schema_name": schema_name,
        "status": status_name,
        "elapsed_sec": elapsed,
        "call_dir": None if call_dir is None else str(call_dir),
        "avg_sec_after": stats.get("avg_sec"),
        "n_after": stats.get("n"),
    }
    _append_jsonl(latency_dir / "history.jsonl", history_record)
    return history_record


def maybe_record_budget_snapshot(
    *,
    mcts_dir: Path,
    event_idx: int,
    snapshot_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    budget = int(event_idx) + 1
    if budget not in BUDGET_SNAPSHOTS:
        return None
    budget_dir = budget_snapshot_dir(mcts_dir)
    snapshot_path = budget_dir / budget_snapshot_filename(budget)
    if snapshot_path.exists():
        try:
            existing = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        return existing if isinstance(existing, dict) else None

    recorded_at = utc_now_iso()
    snapshot = {
        "recorded_at": recorded_at,
        "budget": int(budget),
        "event_idx": int(event_idx),
        "budget_unit": BUDGET_UNIT,
        "status": "reached",
        **(snapshot_payload or {}),
    }
    snapshot["budget"] = int(budget)
    snapshot["event_idx"] = int(event_idx)
    snapshot["budget_unit"] = BUDGET_UNIT
    snapshot["status"] = "reached"
    snapshot["recorded_at"] = recorded_at
    save_json(snapshot_path, snapshot)
    _append_jsonl(budget_dir / "snapshots.jsonl", snapshot)

    running_path = budget_dir / "running.json"
    running = _load_json(running_path, _empty_budget_running())
    reached = running.setdefault("reached", {})
    if not isinstance(reached, dict):
        reached = {}
        running["reached"] = reached
    best = snapshot.get("best_strategy") if isinstance(snapshot.get("best_strategy"), dict) else {}
    reached[str(budget)] = {
        "budget": int(budget),
        "event_idx": int(event_idx),
        "recorded_at": recorded_at,
        "snapshot_file": str(snapshot_path),
        "best_node_id": best.get("node_id"),
        "best_theta_id": best.get("theta_id"),
        "best_selection_score": best.get("selection_score"),
    }
    running["pending"] = [item for item in BUDGET_SNAPSHOTS if str(item) not in reached]
    running["updated_at"] = recorded_at
    running["budget_unit"] = BUDGET_UNIT
    running["targets"] = list(BUDGET_SNAPSHOTS)
    save_json(running_path, running)
    return snapshot
