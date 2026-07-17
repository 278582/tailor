from __future__ import annotations

from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from postprocess.paths import TABDIFF_DIR
from postprocess.tabdiff_utils import get_tabdiff_paths


def _ensure_tabdiff_import_path(tabdiff_root: Path) -> None:
    root_str = str(tabdiff_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _load_tabmetrics_class(tabdiff_root: Path):
    _ensure_tabdiff_import_path(tabdiff_root)
    from tabdiff.metrics import TabMetrics  # type: ignore

    return TabMetrics


def load_tabdiff_info(dataset_name: str) -> dict[str, Any]:
    paths = get_tabdiff_paths(dataset_name)
    with (paths.data_dir / "info.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _tabdiff_ordered_indices(info: dict[str, Any]) -> tuple[list[int], list[int]]:
    num_col_idx = [int(idx) for idx in info.get("num_col_idx", [])]
    cat_col_idx = [int(idx) for idx in info.get("cat_col_idx", [])]
    target_col_idx = [int(idx) for idx in info.get("target_col_idx", [])]
    if str(info.get("task_type", "binclass")) == "regression":
        num_col_idx = num_col_idx + target_col_idx
    else:
        cat_col_idx = cat_col_idx + target_col_idx
    return num_col_idx, cat_col_idx


def _metadata_with_int_columns(info: dict[str, Any]) -> dict[str, Any]:
    metadata = deepcopy(info["metadata"])
    metadata["columns"] = {int(key): value for key, value in metadata["columns"].items()}
    return metadata


def _reorder_for_sdmetrics(
    real_data: pd.DataFrame,
    syn_data: pd.DataFrame,
    info: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    num_col_idx, cat_col_idx = _tabdiff_ordered_indices(info)
    new_real_data = pd.concat([real_data[num_col_idx], real_data[cat_col_idx]], axis=1)
    new_syn_data = pd.concat([syn_data[num_col_idx], syn_data[cat_col_idx]], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))
    new_syn_data.columns = range(len(new_syn_data.columns))

    metadata = _metadata_with_int_columns(info)
    columns = metadata["columns"]
    metadata["columns"] = {}
    for out_idx in range(len(new_real_data.columns)):
        if out_idx < len(num_col_idx):
            metadata["columns"][out_idx] = columns[num_col_idx[out_idx]]
        else:
            cat_offset = out_idx - len(num_col_idx)
            metadata["columns"][out_idx] = columns[cat_col_idx[cat_offset]]

    for out_idx in range(len(num_col_idx), len(new_real_data.columns)):
        new_real_data[out_idx] = new_real_data[out_idx].astype(str)
        new_syn_data[out_idx] = new_syn_data[out_idx].astype(str)
    return new_real_data, new_syn_data, metadata


class TabDiffMetricRunner:
    def __init__(
        self,
        dataset_name: str,
        device: str = "cpu",
        metric_list: list[str] | None = None,
        real_data_path: Path | None = None,
        test_data_path: Path | None = None,
        val_data_path: Path | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.paths = get_tabdiff_paths(dataset_name)
        self.info = load_tabdiff_info(dataset_name)
        metric_list = metric_list or ["density", "dcr"]
        TabMetrics = _load_tabmetrics_class(TABDIFF_DIR)
        real_path = self.paths.synthetic_dir / "real.csv" if real_data_path is None else Path(real_data_path)
        test_path = self.paths.synthetic_dir / "test.csv" if test_data_path is None else Path(test_data_path)
        val_path = self.paths.synthetic_dir / "val.csv" if val_data_path is None else Path(val_data_path)
        self.metric_list = list(metric_list)
        self.real_data_path = real_path
        self.test_data_path = test_path
        self.val_data_path = val_path if val_path.exists() else None
        self.metrics = TabMetrics(
            real_data_path=str(real_path),
            test_data_path=str(test_path),
            val_data_path=str(self.val_data_path) if self.val_data_path is not None else None,
            info=self.info,
            device=device,
            metric_list=metric_list,
        )

    def evaluate_density(self, syn_data: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
        from sdmetrics.reports.single_table import DiagnosticReport, QualityReport

        real_data = pd.read_csv(self.real_data_path)
        real_data.columns = range(len(real_data.columns))
        syn_data = syn_data.copy()
        syn_data.columns = range(len(syn_data.columns))
        new_real_data, new_syn_data, metadata = _reorder_for_sdmetrics(real_data, syn_data, self.info)

        qual_report = QualityReport()
        qual_report.generate(new_real_data, new_syn_data, metadata)
        diag_report = DiagnosticReport()
        diag_report.generate(new_real_data, new_syn_data, metadata)

        quality = qual_report.get_properties()
        shape = quality["Score"][0]
        trend = quality["Score"][1]
        overall = (shape + trend) / 2
        shape_details = qual_report.get_details(property_name="Column Shapes")
        trend_details = qual_report.get_details(property_name="Column Pair Trends")
        return {
            "density/Shape": shape,
            "density/Trend": trend,
            "density/Overall": overall,
        }, {
            "shapes": shape_details,
            "trends": trend_details,
        }

    def evaluate(self, df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
        metrics: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        for metric in self.metric_list:
            if metric == "density":
                out_metrics, out_extras = self.evaluate_density(df.copy())
            else:
                func = getattr(self.metrics, f"evaluate_{metric}")
                out_metrics, out_extras = func(df.copy())
            metrics.update(out_metrics)
            extras.update(out_extras)
        return metrics, extras
