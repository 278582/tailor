from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


THETA_FIELDS = ("col_1ds", "col_2ds", "col_ps", "col_u")


@dataclass(frozen=True)
class ThetaGuidance:
    theta: dict[str, Any]
    source_kind: str
    source_path: Path
    mcts_dir: Path | None = None
    theta_id: str | None = None
    reward: float | None = None
    reward_key: str | None = None
    node_id: str | None = None
    s_id: str | None = None
    rollout_dir: Path | None = None
    synthetic_csv: Path | None = None
    synthetic_pool_manifest: Path | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "source_kind": self.source_kind,
            "source_path": str(self.source_path),
            "mcts_dir": None if self.mcts_dir is None else str(self.mcts_dir),
            "theta_id": self.theta_id,
            "node_id": self.node_id,
            "rollout_dir": None if self.rollout_dir is None else str(self.rollout_dir),
            "reward": self.reward,
            "reward_key": self.reward_key,
            "s_id": self.s_id,
            "theta": self.theta,
            "synthetic_csv": None if self.synthetic_csv is None else str(self.synthetic_csv),
            "synthetic_pool_manifest": (
                None if self.synthetic_pool_manifest is None else str(self.synthetic_pool_manifest)
            ),
        }


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record in {path}:{line_no}: {exc}") from exc
            if isinstance(payload, dict):
                yield payload


def _as_column_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = [value]
    else:
        try:
            raw_values = list(value)
        except TypeError as exc:
            raise ValueError(f"theta.{field_name} must be a list of columns") from exc
    output: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        column = str(raw).strip()
        if not column or column in seen:
            continue
        seen.add(column)
        output.append(column)
    return output


def normalize_theta_payload(payload: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    theta_obj = payload.get("theta", payload)
    if not isinstance(theta_obj, dict):
        raise ValueError(f"Cannot find theta object in {source_path}")
    missing = [field for field in THETA_FIELDS if field not in theta_obj]
    if missing:
        raise ValueError(f"Theta in {source_path} is missing fields: {missing}")
    col_u = str(theta_obj.get("col_u", "") or "").strip()
    if not col_u:
        raise ValueError(f"Theta in {source_path} has empty col_u")
    return {
        "col_1ds": _as_column_list(theta_obj.get("col_1ds"), "col_1ds"),
        "col_2ds": _as_column_list(theta_obj.get("col_2ds"), "col_2ds"),
        "col_ps": _as_column_list(theta_obj.get("col_ps"), "col_ps"),
        "col_u": col_u,
    }


def feature_columns_from_schema(schema_card: dict[str, Any]) -> list[str]:
    return [
        str(column)
        for column in schema_card.get("column_order", [])
        if not bool(schema_card.get("columns", {}).get(column, {}).get("is_target", False))
    ]


def override_report_col_ps_with_all_features(
    report: dict[str, Any] | None,
    schema_card: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    feature_columns = feature_columns_from_schema(schema_card)
    updated_report = dict(report or {"enabled": False})
    theta = updated_report.get("theta")
    applied = bool(updated_report.get("enabled", False) and isinstance(theta, dict))
    original_col_ps = _as_column_list(theta.get("col_ps"), "col_ps") if isinstance(theta, dict) else []
    if applied:
        updated_theta = dict(theta)
        updated_theta["col_ps"] = list(feature_columns)
        updated_report["theta"] = updated_theta
    updated_report["col_ps_override"] = {
        "enabled": True,
        "applied": applied,
        "mode": "all_non_target_feature_columns",
        "reason": None if applied else "theta_guidance_disabled_or_missing_theta",
        "original_count": len(original_col_ps),
        "replacement_count": len(feature_columns),
        "replacement_columns": list(feature_columns),
    }
    return feature_columns, updated_report


def mark_report_default_fidelity_columns(
    report: dict[str, Any] | None,
    schema_card: dict[str, Any],
) -> dict[str, Any]:
    updated_report = dict(report or {"enabled": False})
    theta = updated_report.get("theta")
    applied = bool(updated_report.get("enabled", False) and isinstance(theta, dict))
    original_col_1ds = _as_column_list(theta.get("col_1ds"), "col_1ds") if isinstance(theta, dict) else []
    original_col_2ds = _as_column_list(theta.get("col_2ds"), "col_2ds") if isinstance(theta, dict) else []
    updated_report["fidelity_columns_override"] = {
        "enabled": True,
        "applied": applied,
        "mode": "selector_default_column_order",
        "reason": None if applied else "theta_guidance_disabled_or_missing_theta",
        "original_col_1ds_count": len(original_col_1ds),
        "original_col_2ds_count": len(original_col_2ds),
        "replacement_count": len(schema_card.get("column_order", [])),
        "replacement_columns": list(schema_card.get("column_order", [])),
    }
    return updated_report


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _reward_from_payload(payload: dict[str, Any]) -> tuple[float | None, str | None]:
    for key in ("reward", "Q_self", "metric_reward"):
        value = _finite_float(payload.get(key))
        if value is not None:
            return value, key
    audit_metrics = payload.get("audit_metrics")
    if isinstance(audit_metrics, dict):
        value = _finite_float(audit_metrics.get("metric_reward"))
        if value is not None:
            return value, "audit_metrics.metric_reward"
    return None, None


def _path_from_record(value: Any, mcts_dir: Path | None = None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    if path.is_absolute() or mcts_dir is None:
        return path
    return path


def _infer_s_id_from_rollout_dir(rollout_dir: Path | None) -> str | None:
    if rollout_dir is None:
        return None
    match = re.match(r"^(s_\d+)(?:_|$)", Path(rollout_dir).name)
    if match:
        return match.group(1)
    return None


def _infer_mcts_dir_from_path(path: Path) -> Path | None:
    path = Path(path)
    for parent in [path.parent, *path.parents]:
        if parent.name in {"mcts", "mcts_v2"}:
            return parent
    return None


def _mcts_child_candidates(run_dir: Path) -> list[Path]:
    return [run_dir / "mcts_v2", run_dir / "mcts"]


def resolve_theta_synthetic_pool(guidance: ThetaGuidance | None) -> tuple[Path | None, Path | None]:
    if guidance is None:
        return None, None
    s_id = guidance.s_id or _infer_s_id_from_rollout_dir(guidance.rollout_dir)
    mcts_dir = guidance.mcts_dir
    if mcts_dir is None:
        mcts_dir = _infer_mcts_dir_from_path(guidance.source_path)
    if mcts_dir is None or s_id is None:
        return None, None
    s_dir = Path(mcts_dir) / "s_nodes" / s_id
    synthetic_csv = s_dir / "synthetic_pool.csv"
    manifest_path = s_dir / "synthetic_pool_manifest.json"
    if synthetic_csv.exists():
        return synthetic_csv, manifest_path if manifest_path.exists() else None
    return None, manifest_path if manifest_path.exists() else None


def load_theta_json(path: Path, *, source_kind: str = "path") -> ThetaGuidance:
    path = Path(path)
    payload = _load_json(path)
    reward, reward_key = _reward_from_payload(payload)
    rollout_dir = _path_from_record(payload.get("rollout_dir"))
    if rollout_dir is None and path.name == "theta.json" and path.parent.parent.name == "rollouts":
        rollout_dir = path.parent
    s_id = None if payload.get("s_id") is None else str(payload.get("s_id"))
    s_id = s_id or _infer_s_id_from_rollout_dir(rollout_dir)
    guidance = ThetaGuidance(
        theta=normalize_theta_payload(payload, source_path=path),
        source_kind=source_kind,
        source_path=path,
        theta_id=None if payload.get("theta_id") is None else str(payload.get("theta_id")),
        reward=reward,
        reward_key=reward_key,
        node_id=None if payload.get("node_id") is None else str(payload.get("node_id")),
        s_id=s_id,
        rollout_dir=rollout_dir,
        mcts_dir=_infer_mcts_dir_from_path(path),
    )
    synthetic_csv, synthetic_pool_manifest = resolve_theta_synthetic_pool(guidance)
    return ThetaGuidance(
        **{
            **guidance.__dict__,
            "synthetic_csv": synthetic_csv,
            "synthetic_pool_manifest": synthetic_pool_manifest,
        }
    )


def resolve_mcts_dir(
    *,
    dataset_name: str,
    theta_artifact_root: Path,
    theta_run_name: str,
    theta_mcts_dir: Path | None,
) -> Path:
    if theta_mcts_dir is not None:
        mcts_dir = Path(theta_mcts_dir)
    else:
        dataset_names = [dataset_name]
        if "_tgm_" in dataset_name:
            dataset_names.append(dataset_name.split("_tgm_", 1)[0])
        mcts_dir = Path(theta_artifact_root) / dataset_name / theta_run_name / "mcts_v2"
        for candidate_name in dict.fromkeys(dataset_names):
            run_dir = Path(theta_artifact_root) / candidate_name / theta_run_name
            for candidate in _mcts_child_candidates(run_dir):
                if candidate.exists():
                    mcts_dir = candidate
                    break
            if mcts_dir.exists():
                break
    if not mcts_dir.exists():
        raise FileNotFoundError(f"Cannot find theta MCTS directory: {mcts_dir}")
    return mcts_dir


def load_final_theta(mcts_dir: Path) -> ThetaGuidance:
    mcts_dir = Path(mcts_dir)
    path = mcts_dir / "final" / "theta_star.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find final theta_star.json: {path}")
    guidance = load_theta_json(path, source_kind="final")
    return ThetaGuidance(
        theta=guidance.theta,
        source_kind=guidance.source_kind,
        source_path=guidance.source_path,
        mcts_dir=mcts_dir,
        theta_id=guidance.theta_id,
        reward=guidance.reward,
        reward_key=guidance.reward_key,
        node_id=guidance.node_id,
        s_id=guidance.s_id,
        rollout_dir=guidance.rollout_dir,
        synthetic_csv=guidance.synthetic_csv,
        synthetic_pool_manifest=guidance.synthetic_pool_manifest,
    )


def _guidance_from_archive_record(record: dict[str, Any], *, archive_path: Path, mcts_dir: Path) -> ThetaGuidance | None:
    if not isinstance(record.get("theta"), dict):
        return None
    reward, reward_key = _reward_from_payload(record)
    if reward is None:
        return None
    rollout_dir = _path_from_record(record.get("rollout_dir"), mcts_dir=mcts_dir)
    source_path = archive_path
    if rollout_dir is not None:
        theta_path = rollout_dir / "theta.json"
        if theta_path.exists():
            source_path = theta_path
    s_id = None if record.get("s_id") is None else str(record.get("s_id"))
    s_id = s_id or _infer_s_id_from_rollout_dir(rollout_dir)
    guidance = ThetaGuidance(
        theta=normalize_theta_payload(record, source_path=archive_path),
        source_kind="best-rollout",
        source_path=source_path,
        mcts_dir=mcts_dir,
        theta_id=None if record.get("theta_id") is None else str(record.get("theta_id")),
        reward=reward,
        reward_key=reward_key,
        node_id=None if record.get("node_id") is None else str(record.get("node_id")),
        s_id=s_id,
        rollout_dir=rollout_dir,
    )
    synthetic_csv, synthetic_pool_manifest = resolve_theta_synthetic_pool(guidance)
    return ThetaGuidance(
        **{
            **guidance.__dict__,
            "synthetic_csv": synthetic_csv,
            "synthetic_pool_manifest": synthetic_pool_manifest,
        }
    )


def _guidance_from_rollout_dir(rollout_dir: Path, *, mcts_dir: Path) -> ThetaGuidance | None:
    theta_path = rollout_dir / "theta.json"
    if not theta_path.exists():
        return None
    reward_payload: dict[str, Any] = {}
    reward_path = rollout_dir / "reward.json"
    if reward_path.exists():
        reward_payload = _load_json(reward_path)
    reward, reward_key = _reward_from_payload(reward_payload)
    if reward is None:
        metrics_path = rollout_dir / "metrics_summary.json"
        if metrics_path.exists():
            reward, reward_key = _reward_from_payload(_load_json(metrics_path))
    if reward is None:
        return None
    guidance = load_theta_json(theta_path, source_kind="best-rollout")
    s_id = guidance.s_id or _infer_s_id_from_rollout_dir(rollout_dir)
    synthetic_guidance = ThetaGuidance(
        theta=guidance.theta,
        source_kind=guidance.source_kind,
        source_path=guidance.source_path,
        mcts_dir=mcts_dir,
        theta_id=guidance.theta_id,
        reward=reward,
        reward_key=reward_key,
        rollout_dir=rollout_dir,
        s_id=s_id,
    )
    synthetic_csv, synthetic_pool_manifest = resolve_theta_synthetic_pool(synthetic_guidance)
    return ThetaGuidance(
        theta=guidance.theta,
        source_kind=guidance.source_kind,
        source_path=guidance.source_path,
        mcts_dir=mcts_dir,
        theta_id=guidance.theta_id,
        reward=reward,
        reward_key=reward_key,
        s_id=s_id,
        rollout_dir=rollout_dir,
        synthetic_csv=synthetic_csv,
        synthetic_pool_manifest=synthetic_pool_manifest,
    )


def load_best_rollout_theta(mcts_dir: Path) -> ThetaGuidance:
    mcts_dir = Path(mcts_dir)
    candidates: list[ThetaGuidance] = []
    archive_path = mcts_dir / "archive" / "all_rollouts.jsonl"
    if archive_path.exists():
        for record in _iter_jsonl(archive_path):
            guidance = _guidance_from_archive_record(record, archive_path=archive_path, mcts_dir=mcts_dir)
            if guidance is not None:
                candidates.append(guidance)

    rollouts_dir = mcts_dir / "rollouts"
    if rollouts_dir.exists():
        for rollout_dir in sorted(path for path in rollouts_dir.iterdir() if path.is_dir()):
            guidance = _guidance_from_rollout_dir(rollout_dir, mcts_dir=mcts_dir)
            if guidance is not None:
                candidates.append(guidance)

    if not candidates:
        raise FileNotFoundError(f"No rollout theta with finite Q_self/reward found under {mcts_dir}")
    return max(
        candidates,
        key=lambda guidance: (
            float(guidance.reward if guidance.reward is not None else float("-inf")),
            1 if guidance.reward_key == "reward" else 0,
            str(guidance.theta_id or ""),
        ),
    )


def _dataset_name_candidates(dataset_name: str) -> list[str]:
    names = [dataset_name]
    if "_tgm_" in dataset_name:
        names.append(dataset_name.split("_tgm_", 1)[0])
    return list(dict.fromkeys(names))


def _iter_artifact_mcts_dirs(
    *,
    dataset_name: str,
    theta_artifact_root: Path,
    theta_run_name: str,
) -> list[Path]:
    mcts_dirs: list[Path] = []
    for candidate_name in _dataset_name_candidates(dataset_name):
        dataset_dir = Path(theta_artifact_root) / candidate_name
        if not dataset_dir.exists():
            continue
        if theta_run_name in {"auto", "all", "*"}:
            run_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
        else:
            run_dirs = [dataset_dir / theta_run_name]
        for run_dir in run_dirs:
            for mcts_dir in _mcts_child_candidates(run_dir):
                if mcts_dir.exists():
                    mcts_dirs.append(mcts_dir)
    return list(dict.fromkeys(mcts_dirs))


def load_best_artifact_theta(
    *,
    dataset_name: str,
    theta_artifact_root: Path,
    theta_run_name: str,
) -> ThetaGuidance:
    candidates: list[ThetaGuidance] = []
    mcts_dirs = _iter_artifact_mcts_dirs(
        dataset_name=dataset_name,
        theta_artifact_root=theta_artifact_root,
        theta_run_name=theta_run_name,
    )
    for mcts_dir in mcts_dirs:
        final_path = mcts_dir / "final" / "theta_star.json"
        if final_path.exists():
            try:
                candidates.append(load_final_theta(mcts_dir))
            except (FileNotFoundError, ValueError):
                pass
        try:
            candidates.append(load_best_rollout_theta(mcts_dir))
        except (FileNotFoundError, ValueError):
            pass

    if not candidates:
        raise FileNotFoundError(
            f"No theta with finite reward found under {Path(theta_artifact_root) / dataset_name}"
        )
    best = max(
        candidates,
        key=lambda guidance: (
            float(guidance.reward if guidance.reward is not None else float("-inf")),
            1 if guidance.reward_key == "reward" else 0,
            str(guidance.theta_id or ""),
            str(guidance.source_path),
        ),
    )
    return ThetaGuidance(
        theta=best.theta,
        source_kind="best-artifact",
        source_path=best.source_path,
        mcts_dir=best.mcts_dir,
        theta_id=best.theta_id,
        reward=best.reward,
        reward_key=best.reward_key,
        node_id=best.node_id,
        s_id=best.s_id,
        rollout_dir=best.rollout_dir,
        synthetic_csv=best.synthetic_csv,
        synthetic_pool_manifest=best.synthetic_pool_manifest,
    )


def resolve_theta_guidance(
    *,
    theta_json: Path | None,
    theta_source: str,
    dataset_name: str,
    theta_artifact_root: Path,
    theta_run_name: str,
    theta_mcts_dir: Path | None,
) -> ThetaGuidance | None:
    if theta_json is not None:
        return load_theta_json(theta_json, source_kind="path")
    if theta_source == "none":
        return None
    if theta_source == "best-artifact":
        return load_best_artifact_theta(
            dataset_name=dataset_name,
            theta_artifact_root=theta_artifact_root,
            theta_run_name=theta_run_name,
        )

    mcts_dir = resolve_mcts_dir(
        dataset_name=dataset_name,
        theta_artifact_root=theta_artifact_root,
        theta_run_name=theta_run_name,
        theta_mcts_dir=theta_mcts_dir,
    )
    if theta_source == "final":
        return load_final_theta(mcts_dir)
    if theta_source == "best-rollout":
        return load_best_rollout_theta(mcts_dir)
    if theta_source == "auto":
        final_path = mcts_dir / "final" / "theta_star.json"
        if final_path.exists():
            return load_final_theta(mcts_dir)
        return load_best_rollout_theta(mcts_dir)
    raise ValueError(f"Unsupported theta source: {theta_source}")
