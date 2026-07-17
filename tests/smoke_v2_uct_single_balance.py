from __future__ import annotations

import argparse
from pathlib import Path

from smoke_v2_uct_balance_common import run_smoke


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline smoke: single-source theta UCT should mostly greedily dig while still revisiting siblings."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "artifacts/llm_mcts_v2/adult/"
            "adult_v2_single_tabdiff_full_llm_v4_torch_mlp_e10_full_diag/"
            "mcts_v2/archive/all_theta_nodes.jsonl"
        ),
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--ucb-c", type=float, default=2.0)
    parser.add_argument("--min-explore", type=int, default=20)
    parser.add_argument("--max-ratio", type=float, default=4.0)
    parser.add_argument("--min-selected-depth", type=int, default=10)
    parser.add_argument("--min-backtrack-groups", type=int, default=8)
    parser.add_argument("--min-selected-roots", type=int, default=2)
    parser.add_argument("--min-selected-sources", type=int, default=1)
    return run_smoke(parser.parse_args(), require_multi_source=False)


if __name__ == "__main__":
    raise SystemExit(main())
