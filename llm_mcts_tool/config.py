from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass
class MCTSGuideConfig:
    dataset_name: str
    exp_name: str
    synthetic_csv: Path | None
    artifact_dir: Path | None
    seed: int
    keep_k: int
    preselect_target: int
    d_cur_size: int
    n_init: int
    n_expand: int
    mcts_budget: int
    ucb_c: float
    p_random_replace: float
    max_theta_pairs: int
    disable_progress: bool
    d_cur_source: str = "synthetic"
    holdout_fraction: float = 0.1
    source: str = "tabdiff"
    eval_device: str = "auto"
    nn_device: str = "auto"
    utility_exact_evaluator: str = "tabdiff_mle"
    density_reference_size: int = 5000
    save_validation_records: bool = False
    save_rollout_internal_records: bool = False
    provider: str = "mock"
    provider_jsonl: Path | None = None
    llm_model: str = "qwen3.6-plus"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key_env: str = "DASHSCOPE_API_KEY"
    llm_timeout: int = 300
    llm_max_retries: int = 2
    llm_retry_backoff: float = 3.0
    prompt_pack_dir: Path = Path("prompt_pack")
    refine_prompt_use_dataset_priors: bool = True
