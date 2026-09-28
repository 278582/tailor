from __future__ import annotations

from typing import Any

CENSUS_DATASET_NAME = "us_census_data_1990"
CENSUS_DCR_CAP = 8_000
CENSUS_DCR_REPEATS = 2
CENSUS_PRIVACY_NN_CAP = 200_000
CENSUS_LARGE_SCORE_DUMP_MAX_ROWS = 50_000
CENSUS_PRESELECT_BATCH_SIZE = 65_536
CENSUS_PRESELECT_LARGE_TARGET = 50_000
CENSUS_UTILITY_EXACT_EVALUATOR = "torch_lightweight_mlp"
CENSUS_UTILITY_TORCH_BATCH_SIZE = 65_536
CENSUS_UTILITY_TORCH_EPOCHS = 6
DEFAULT_DCR_REPEATS = 10
DEFAULT_UTILITY_EXACT_EVALUATOR = "tabdiff_mle"


def is_census_dataset(name: str | None) -> bool:
    return str(name or "").strip().lower() == CENSUS_DATASET_NAME


def apply_census_runtime_profile(config: Any) -> Any:
    """Apply census-only runtime caps. Other datasets are left unchanged.

    keep_k / preselect_target / d_cur_size stay as CLI values so smoke jobs are
    not rewritten to |train|. Formal 7-dataset formula is still passed on CLI.
    """
    if not is_census_dataset(getattr(config, "dataset_name", None)):
        return config

    applied: dict[str, Any] = {
        "dataset_name": CENSUS_DATASET_NAME,
        "dcr_cap_source": "cli",
        "dcr_repeats_source": "cli",
        "utility_exact_evaluator_source": "cli",
        "save_validation_records": bool(getattr(config, "save_validation_records", True)),
        "skip_large_score_dumps": True,
        "privacy_nn_cap": CENSUS_PRIVACY_NN_CAP,
    }
    if int(getattr(config, "dcr_cap", 0) or 0) <= 0:
        config.dcr_cap = CENSUS_DCR_CAP
        applied["dcr_cap_source"] = "census_large_dataset_default"
    if int(getattr(config, "dcr_repeats", DEFAULT_DCR_REPEATS) or DEFAULT_DCR_REPEATS) == DEFAULT_DCR_REPEATS:
        config.dcr_repeats = CENSUS_DCR_REPEATS
        applied["dcr_repeats_source"] = "census_large_dataset_default"
    current_evaluator = str(
        getattr(config, "utility_exact_evaluator", DEFAULT_UTILITY_EXACT_EVALUATOR) or DEFAULT_UTILITY_EXACT_EVALUATOR
    )
    if current_evaluator == DEFAULT_UTILITY_EXACT_EVALUATOR:
        config.utility_exact_evaluator = CENSUS_UTILITY_EXACT_EVALUATOR
        applied["utility_exact_evaluator_source"] = "census_large_dataset_default"
    if int(getattr(config, "utility_exact_torch_batch_size", 2048) or 2048) <= 2048:
        config.utility_exact_torch_batch_size = CENSUS_UTILITY_TORCH_BATCH_SIZE
        applied["utility_exact_torch_batch_size_source"] = "census_large_dataset_default"
    if int(getattr(config, "utility_exact_torch_epochs", CENSUS_UTILITY_TORCH_EPOCHS) or 0) <= 0:
        config.utility_exact_torch_epochs = CENSUS_UTILITY_TORCH_EPOCHS
    if hasattr(config, "save_validation_records") and bool(config.save_validation_records):
        config.save_validation_records = False
        applied["save_validation_records"] = False
        applied["save_validation_records_reason"] = "census_skip_large_jsonl"
    applied["dcr_cap"] = int(config.dcr_cap)
    applied["dcr_repeats"] = int(config.dcr_repeats)
    applied["utility_exact_evaluator"] = str(config.utility_exact_evaluator)
    applied["utility_exact_torch_batch_size"] = int(getattr(config, "utility_exact_torch_batch_size", CENSUS_UTILITY_TORCH_BATCH_SIZE))
    applied["preselect_batch_size"] = CENSUS_PRESELECT_BATCH_SIZE
    applied["preselect_large_target"] = CENSUS_PRESELECT_LARGE_TARGET
    config.census_runtime_profile = applied
    return config


def bind_selector_utility_runtime(selector: Any, config: Any) -> None:
    selector.utility_exact_evaluator = str(
        getattr(config, "utility_exact_evaluator", DEFAULT_UTILITY_EXACT_EVALUATOR) or DEFAULT_UTILITY_EXACT_EVALUATOR
    )
    selector.utility_exact_torch_epochs = int(
        getattr(config, "utility_exact_torch_epochs", CENSUS_UTILITY_TORCH_EPOCHS) or CENSUS_UTILITY_TORCH_EPOCHS
    )
    selector.utility_exact_torch_batch_size = int(
        getattr(config, "utility_exact_torch_batch_size", 2048) or 2048
    )
    selector.utility_exact_torch_importance_sample_size = int(
        getattr(config, "utility_exact_torch_importance_sample_size", 2000)
    )
    eval_device = getattr(config, "eval_device", None)
    if eval_device is not None:
        selector.eval_device = eval_device


def resolve_preselect_batch_size(target_preselect: int, dataset_name: str | None) -> int:
    target_preselect = max(1, int(target_preselect))
    batch_size = max(128, min(1536, int(round(target_preselect / 24.0))))
    if not is_census_dataset(dataset_name) or target_preselect < CENSUS_PRESELECT_LARGE_TARGET:
        return batch_size
    return min(target_preselect, max(batch_size, CENSUS_PRESELECT_BATCH_SIZE))


def should_skip_large_score_dumps(config: Any, row_count: int) -> bool:
    if not is_census_dataset(getattr(config, "dataset_name", None)):
        return False
    return int(row_count) > int(CENSUS_LARGE_SCORE_DUMP_MAX_ROWS)
