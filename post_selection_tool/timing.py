from __future__ import annotations

from typing import Any

from .logging_utils import get_logger


PIPELINE_STAGE_NAMES = ("preselected", "score", "nsga_ii", "dcr_repair")
PIPELINE_STAGE_EXCLUDES = [
    "validation",
    "fidelity_ceiling_subset_construction",
    "random_selection",
    "scalar_selection",
    "reward_candidate_v2",
    "io",
]


def ensure_pipeline_stages(timing_report: dict[str, Any] | None) -> dict[str, Any]:
    if timing_report is None:
        raise TypeError("timing_report is required")
    stages = timing_report.setdefault("pipeline_stages", {})
    stages.setdefault("schema_version", 1)
    stages.setdefault("timing_basis", "wall_clock_perf_counter")
    stages.setdefault("excludes", list(PIPELINE_STAGE_EXCLUDES))
    stages.setdefault("total_components", list(PIPELINE_STAGE_NAMES))
    return stages


def refresh_pipeline_stage_total(timing_report: dict[str, Any]) -> dict[str, Any]:
    stages = ensure_pipeline_stages(timing_report)
    total = 0.0
    recorded_components: list[str] = []
    for name in PIPELINE_STAGE_NAMES:
        recorded = bool(stages.get(f"{name}_recorded", False))
        value = stages.get(f"{name}_seconds")
        if not recorded or value is None:
            continue
        total += float(value)
        recorded_components.append(name)
    stages["total_seconds"] = float(total)
    stages["total_components"] = list(PIPELINE_STAGE_NAMES)
    stages["total_recorded_components"] = recorded_components
    stages["complete"] = recorded_components == list(PIPELINE_STAGE_NAMES)
    return stages


def record_pipeline_stage(
    timing_report: dict[str, Any],
    name: str,
    seconds: float,
    **metadata: Any,
) -> dict[str, Any]:
    if name not in PIPELINE_STAGE_NAMES:
        raise ValueError(f"unknown pipeline stage: {name}")
    stages = ensure_pipeline_stages(timing_report)
    stages[f"{name}_seconds"] = float(seconds)
    stages[f"{name}_recorded"] = True
    for key, value in metadata.items():
        stages[key] = value
    return refresh_pipeline_stage_total(timing_report)


def nsga_selection_path(pareto_report: dict[str, Any] | None, *, has_floor_reference: bool) -> str:
    if has_floor_reference:
        return "constrained_subset_construction"
    mode = str((pareto_report or {}).get("front_component_mode") or "").strip()
    if mode:
        return mode
    return "deterministic_exact_nsga_front_rank"


def log_pipeline_stages(timing_report: dict[str, Any] | None) -> None:
    stages = (timing_report or {}).get("pipeline_stages") or {}

    def _seconds(name: str) -> float:
        value = stages.get(f"{name}_seconds")
        return float(value) if value is not None else 0.0

    get_logger().info(
        "[pipeline_stages] preselected=%.2fs score=%.2fs nsga_ii=%.2fs dcr_repair=%.2fs "
        "total=%.2fs complete=%s",
        _seconds("preselected"),
        _seconds("score"),
        _seconds("nsga_ii"),
        _seconds("dcr_repair"),
        float(stages.get("total_seconds") or 0.0),
        bool(stages.get("complete", False)),
    )
