from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _prompt_number(value: Any, digits: int = 4) -> Any:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return value
    return round(parsed, digits)


def _rows_from_utility_report(report: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(report.get("feature_importance", []) or [], start=1):
        if not isinstance(item, dict) or item.get("feature") is None:
            continue
        rows.append(
            {
                "feature": str(item["feature"]),
                "importance": _prompt_number(item.get("importance")),
                "rank": int(item.get("rank", rank)),
            }
        )
    rows.sort(key=lambda row: (int(row.get("rank", 999)), -float(row.get("importance") or 0.0)))
    return rows[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke bridge utility_metrics_summary feature_importance into prompt utility_top.")
    parser.add_argument(
        "--utility-report",
        type=Path,
        default=Path(
            "artifacts/llm_mcts_v2/adult/smoke_v2_torch_mlp_min/mcts_v2/"
            "rollouts/s_000000_a73da2963698/eval/selection_pareto/utility_metrics_summary.json"
        ),
    )
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    report = json.loads(Path(args.utility_report).read_text(encoding="utf-8"))
    utility_top = _rows_from_utility_report(report, limit=int(args.limit))
    payload = {
        "source": str(args.utility_report),
        "protocol": report.get("protocol"),
        "primary_model": report.get("primary_model"),
        "runtime_model_device": report.get("runtime_model_device"),
        "utility_top": utility_top,
    }
    ok = (
        report.get("protocol") == "torch_lightweight_mlp"
        and bool(utility_top)
        and all("feature" in item and "importance" in item for item in utility_top)
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
