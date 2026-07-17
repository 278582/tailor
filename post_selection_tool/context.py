from __future__ import annotations

from pathlib import Path

import torch

from postprocess.data_io import set_seed
from postprocess.tabdiff_protocol import normalize_tabdiff_dataframe_columns, resolve_tabdiff_selection_context
from postprocess.tabdiff_utils import find_latest_tabdiff_sample

from .config import CoreSelectionConfig
from .io import ensure_dir, load_csv, load_json, load_jsonl, remove_known_mirror, save_shared_csv, save_shared_json
from .state import ArtifactPaths, SelectionState
from .theta_pool import build_theta_synthetic_pool_from_manifest


INPUT_FILENAMES = (
    "synthetic_raw.csv",
    "eval_train.csv",
    "eval_holdout.csv",
    "eval_test.csv",
    "selection_context.json",
)


def resolve_eval_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_nn_device(nn_device_arg: str, eval_device: str) -> str:
    if nn_device_arg != "auto":
        return nn_device_arg
    if eval_device.startswith("cuda"):
        return eval_device
    return "auto"


def resolve_synthetic_csv(
    config: CoreSelectionConfig,
    *,
    train_rows: int | None = None,
    output_dir: Path | None = None,
) -> Path:
    if config.synthetic_csv is not None:
        return Path(config.synthetic_csv)
    if config.theta_s_pool_manifest is not None:
        if train_rows is None or output_dir is None:
            raise ValueError("train_rows and output_dir are required to rebuild a theta-guided S pool")
        synthetic_csv, _, _ = build_theta_synthetic_pool_from_manifest(
            manifest_path=Path(config.theta_s_pool_manifest),
            dataset_name=config.dataset_name,
            train_rows=int(train_rows),
            seed=config.seed,
            output_dir=Path(output_dir),
            sample_root=Path(config.theta_s_pool_sample_root),
            sample_id=int(getattr(config, "theta_s_pool_sample_id", 0)),
        )
        return synthetic_csv
    return find_latest_tabdiff_sample(dataset_name=config.dataset_name, exp_name=config.exp_name)


def build_artifact_paths(config: CoreSelectionConfig, artifact_root: Path) -> ArtifactPaths:
    artifact_dir = ensure_dir(artifact_root / config.exp_name)
    shared_root = getattr(config, "shared_artifact_dir", None)
    shared_root = ensure_dir(Path(shared_root)) if shared_root is not None else None
    if shared_root is not None:
        remove_known_mirror(artifact_dir / "input", INPUT_FILENAMES)
    return ArtifactPaths(
        artifact_dir=artifact_dir,
        input_dir=ensure_dir((shared_root / "input") if shared_root is not None else (artifact_dir / "input")),
        cards_dir=(shared_root / "cards") if shared_root is not None else (artifact_dir / "cards"),
        validation_dir=ensure_dir(artifact_dir / "validation"),
        selection_dir=ensure_dir(artifact_dir / "selection"),
        versions_dir=ensure_dir(artifact_dir / "versions"),
        report_dir=ensure_dir(artifact_dir / "report"),
    )


def _load_theta_pool_source_metadata(theta_pool_dir: Path) -> tuple[dict[int, str], dict[str, Any]]:
    row_map_path = Path(theta_pool_dir) / "theta_s_pool_rows.jsonl"
    manifest_path = Path(theta_pool_dir) / "theta_s_pool_manifest.json"
    if not row_map_path.exists():
        return {}, {}

    source_by_id: dict[int, str] = {}
    for record in load_jsonl(row_map_path):
        try:
            pool_row_id = int(record["pool_row_id"])
        except (KeyError, TypeError, ValueError):
            continue
        source_id = str(record.get("source_id", "")).strip().lower()
        if source_id:
            source_by_id[pool_row_id] = source_id

    report = dict(load_json(manifest_path)) if manifest_path.exists() else {}
    report["row_map_path"] = str(row_map_path)
    report["source_metadata_rows"] = len(source_by_id)
    return source_by_id, report


def prepare_context(config: CoreSelectionConfig) -> SelectionState:
    set_seed(config.seed)
    dataset_ctx = resolve_tabdiff_selection_context(
        dataset_name=config.dataset_name,
        seed=config.seed,
        holdout_fraction=config.holdout_fraction,
    )
    artifact_root = dataset_ctx.artifact_root if config.artifact_dir is None else Path(config.artifact_dir)
    paths = build_artifact_paths(config, artifact_root)
    shared_root = ensure_dir(Path(config.shared_artifact_dir)) if config.shared_artifact_dir is not None else None

    def shared_input(name: str) -> Path | None:
        return None if shared_root is None else shared_root / "input" / name

    train_df = dataset_ctx.train_df.copy()
    holdout_df = dataset_ctx.holdout_df.copy()
    test_df = dataset_ctx.test_df.copy()
    synthetic_csv = resolve_synthetic_csv(
        config,
        train_rows=len(train_df),
        output_dir=paths.input_dir / "theta_s_pool",
    )
    synthetic_df = normalize_tabdiff_dataframe_columns(config.dataset_name, load_csv(synthetic_csv))
    candidate_source_by_id, theta_s_pool_report = _load_theta_pool_source_metadata(paths.input_dir / "theta_s_pool")

    save_shared_csv(paths.input_dir / "synthetic_raw.csv", synthetic_df, shared_input("synthetic_raw.csv"))
    save_shared_csv(paths.input_dir / "eval_train.csv", train_df, shared_input("eval_train.csv"))
    save_shared_csv(paths.input_dir / "eval_holdout.csv", holdout_df, shared_input("eval_holdout.csv"))
    save_shared_csv(paths.input_dir / "eval_test.csv", test_df, shared_input("eval_test.csv"))
    save_shared_json(paths.input_dir / "selection_context.json", dataset_ctx.to_manifest(), shared_input("selection_context.json"))

    return SelectionState(
        config=config,
        paths=paths,
        dataset_ctx=dataset_ctx,
        synthetic_csv=synthetic_csv,
        train_df=train_df,
        holdout_df=holdout_df,
        test_df=test_df,
        synthetic_df=synthetic_df,
        candidate_source_by_id=candidate_source_by_id,
        theta_s_pool_report=theta_s_pool_report,
    )
