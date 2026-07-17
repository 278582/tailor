from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .attacks import (
    AttackData,
    attack_score_frame,
    make_attack_data,
    run_release_attacks,
    shadow_attack,
    supervised_profile_attack,
)
from .io import (
    ensure_dir,
    find_selection_csv,
    infer_target_column,
    list_selection_csvs,
    load_csv,
    resolve_run_inputs,
    save_csv,
    save_json,
    selection_name_from_path,
)
from .metrics import best_attack, summarize_binary_scores


@dataclass
class AuditConfig:
    seed: int = 20260420
    density_k: int = 5
    max_attribute_columns: int = 20
    max_member_rows: int = 0
    max_nonmember_rows: int = 0
    max_synthetic_rows: int = 0
    max_reference_rows: int = 0
    reference_split: str = "test"
    exclude_target: bool = False
    columns: list[str] | None = None
    include_supervised_profile: bool = True
    shadow_run_dirs: list[Path] = field(default_factory=list)


def audit_run(
    *,
    run_dir: Path,
    out_dir: Path,
    config: AuditConfig,
    all_selections: bool = False,
    selection_csv: Path | None = None,
    selection_name: str | None = None,
) -> dict[str, Any]:
    run_inputs = resolve_run_inputs(Path(run_dir), reference_split=config.reference_split)
    selection_paths: list[Path]
    if selection_csv is not None:
        selection_paths = [Path(selection_csv)]
    elif all_selections:
        selection_paths = list_selection_csvs(run_inputs.versions_dir)
    elif selection_name:
        resolved = find_selection_csv(run_inputs.versions_dir, selection_name)
        if resolved is None:
            raise FileNotFoundError(f"Could not find selection {selection_name!r} in {run_inputs.versions_dir}")
        selection_paths = [resolved]
    else:
        resolved = find_selection_csv(run_inputs.versions_dir, "pareto")
        if resolved is None:
            raise ValueError("Specify --all-selections, --selection-csv, or --selection-name.")
        selection_paths = [resolved]

    if not selection_paths:
        raise FileNotFoundError(f"No selection CSVs found in {run_inputs.versions_dir}")

    ensure_dir(Path(out_dir))
    selection_reports: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    for path in selection_paths:
        name = selection_name or selection_name_from_path(path)
        report = audit_selection_from_paths(
            run_dir=Path(run_dir),
            selection_csv=path,
            selection_name=name,
            out_dir=Path(out_dir) / name,
            config=config,
        )
        selection_reports[name] = report
        for metric in report["metrics"]:
            metric_rows.append({"selection": name, **metric})

    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty:
        save_csv(Path(out_dir) / "metrics.csv", metrics_df)
    summary = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "selection_count": len(selection_reports),
        "selections": selection_reports,
        "config": _config_to_dict(config),
    }
    save_json(Path(out_dir) / "summary.json", summary)
    return summary


def audit_selection_from_paths(
    *,
    run_dir: Path,
    selection_csv: Path,
    selection_name: str,
    out_dir: Path,
    config: AuditConfig,
) -> dict[str, Any]:
    run_inputs = resolve_run_inputs(Path(run_dir), reference_split=config.reference_split)
    member = load_csv(run_inputs.train_csv)
    nonmember = load_csv(run_inputs.control_csv)
    reference = load_csv(run_inputs.reference_csv)
    synthetic = load_csv(Path(selection_csv))
    member = _sample_df(member, max_rows=config.max_member_rows, seed=config.seed)
    nonmember = _sample_df(nonmember, max_rows=config.max_nonmember_rows, seed=config.seed + 1)
    reference = _sample_df(reference, max_rows=config.max_reference_rows, seed=config.seed + 2)
    synthetic = _sample_df(synthetic, max_rows=config.max_synthetic_rows, seed=config.seed + 3)
    target_column = infer_target_column(run_inputs.context, list(member.columns))
    exclude_columns = [target_column] if config.exclude_target and target_column else []
    data = make_attack_data(
        member=member,
        nonmember=nonmember,
        synthetic=synthetic,
        reference=reference,
        requested_columns=config.columns,
        exclude_columns=exclude_columns,
    )
    return audit_selection(
        data=data,
        out_dir=out_dir,
        selection_name=selection_name,
        selection_csv=Path(selection_csv),
        config=config,
        run_context=run_inputs.context,
    )


def audit_selection(
    *,
    data: AttackData,
    out_dir: Path,
    selection_name: str,
    selection_csv: Path | None,
    config: AuditConfig,
    run_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_dir(Path(out_dir))
    outputs = run_release_attacks(
        data,
        density_k=config.density_k,
        max_attribute_columns=config.max_attribute_columns,
        random_state=config.seed,
    )
    labels = data.labels
    score_frame = attack_score_frame(outputs, labels)

    if config.include_supervised_profile:
        supervised = supervised_profile_attack(score_frame, random_state=config.seed)
        if supervised is not None:
            outputs.append(supervised)
            score_frame[supervised.name] = supervised.scores

    shadow_frames = _build_shadow_score_frames(
        config=config,
        selection_name=selection_name,
        columns=data.columns,
        exclude_columns=[],
    )
    shadow_output = shadow_attack(
        target_score_frame=score_frame,
        shadow_score_frames=shadow_frames,
        random_state=config.seed,
    )
    if shadow_output is not None:
        outputs.append(shadow_output)
        score_frame[shadow_output.name] = shadow_output.scores

    metrics = [summarize_binary_scores(output.name, labels, output.scores).to_dict() for output in outputs]
    details = {output.name: output.details for output in outputs}
    score_frame.insert(0, "record_role", ["member"] * len(data.member) + ["nonmember"] * len(data.nonmember))
    save_csv(Path(out_dir) / "scores.csv", score_frame)
    save_json(Path(out_dir) / "attack_details.json", details)
    report = {
        "selection_name": selection_name,
        "selection_csv": str(selection_csv) if selection_csv is not None else None,
        "rows": {
            "member": int(len(data.member)),
            "nonmember": int(len(data.nonmember)),
            "synthetic": int(len(data.synthetic)),
            "reference": int(len(data.reference)),
        },
        "columns": list(data.columns),
        "metrics": metrics,
        "best_attack": best_attack(metrics),
        "attack_details": details,
        "config": _config_to_dict(config),
        "run_context": run_context or {},
    }
    save_json(Path(out_dir) / "summary.json", report)
    return report


def _build_shadow_score_frames(
    *,
    config: AuditConfig,
    selection_name: str,
    columns: list[str],
    exclude_columns: list[str],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for shadow_dir in config.shadow_run_dirs:
        try:
            run_inputs = resolve_run_inputs(Path(shadow_dir), reference_split=config.reference_split)
            selection_csv = find_selection_csv(run_inputs.versions_dir, selection_name)
            if selection_csv is None:
                continue
            data = make_attack_data(
                member=_sample_df(load_csv(run_inputs.train_csv), max_rows=config.max_member_rows, seed=config.seed),
                nonmember=_sample_df(
                    load_csv(run_inputs.control_csv),
                    max_rows=config.max_nonmember_rows,
                    seed=config.seed + 1,
                ),
                synthetic=_sample_df(
                    load_csv(selection_csv),
                    max_rows=config.max_synthetic_rows,
                    seed=config.seed + 3,
                ),
                reference=_sample_df(
                    load_csv(run_inputs.reference_csv),
                    max_rows=config.max_reference_rows,
                    seed=config.seed + 2,
                ),
                requested_columns=columns,
                exclude_columns=exclude_columns,
            )
            outputs = run_release_attacks(
                data,
                density_k=config.density_k,
                max_attribute_columns=config.max_attribute_columns,
                random_state=config.seed,
            )
            frames.append(attack_score_frame(outputs, data.labels))
        except Exception:
            continue
    return frames


def _config_to_dict(config: AuditConfig) -> dict[str, Any]:
    return {
        "seed": int(config.seed),
        "density_k": int(config.density_k),
        "max_attribute_columns": int(config.max_attribute_columns),
        "max_member_rows": int(config.max_member_rows),
        "max_nonmember_rows": int(config.max_nonmember_rows),
        "max_synthetic_rows": int(config.max_synthetic_rows),
        "max_reference_rows": int(config.max_reference_rows),
        "reference_split": config.reference_split,
        "exclude_target": bool(config.exclude_target),
        "columns": list(config.columns) if config.columns is not None else None,
        "include_supervised_profile": bool(config.include_supervised_profile),
        "shadow_run_dirs": [str(path) for path in config.shadow_run_dirs],
    }


def _sample_df(df: pd.DataFrame, *, max_rows: int, seed: int) -> pd.DataFrame:
    max_rows = int(max_rows or 0)
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed, replace=False).reset_index(drop=True)
