from __future__ import annotations

import argparse
from pathlib import Path

from smoke_v2_uct_balance_common import run_smoke


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline smoke: mixed-source theta UCT should greedily dig while revisiting siblings across S pools."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "artifacts/llm_mcts_v2/adult/"
            "adult_v2_mixed_full_llm_v4_torch_mlp_e10_refine_s3_full_diag/"
            "mcts_v2/archive/all_theta_nodes.jsonl"
        ),
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--ucb-c", type=float, default=2.0)
    parser.add_argument("--min-explore", type=int, default=20)
    parser.add_argument("--max-ratio", type=float, default=3.0)
    parser.add_argument("--min-selected-depth", type=int, default=4)
    parser.add_argument("--min-backtrack-groups", type=int, default=8)
    parser.add_argument("--min-selected-roots", type=int, default=8)
    parser.add_argument("--min-selected-sources", type=int, default=8)
    return run_smoke(parser.parse_args(), require_multi_source=True)


if __name__ == "__main__":
    raise SystemExit(main())
