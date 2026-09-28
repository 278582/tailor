from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .census_profile import CENSUS_PRESELECT_LARGE_TARGET, is_census_dataset
from .logging_utils import get_logger


def resolve_census_preselect_device(selector: Any) -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    nn_device = getattr(selector, "nn_device", None)
    if isinstance(nn_device, torch.device) and nn_device.type == "cuda":
        return nn_device
    nn_device_arg = str(getattr(selector, "nn_device_arg", "auto"))
    if nn_device_arg.startswith("cuda"):
        return torch.device(nn_device_arg)
    return torch.device("cuda:0")


def should_use_census_gpu_preselect(selector: Any, row_count: int) -> bool:
    schema = getattr(selector, "schema_card", {}) or {}
    dataset_name = str(schema.get("dataset", ""))
    if not is_census_dataset(dataset_name):
        return False
    if int(row_count) < int(CENSUS_PRESELECT_LARGE_TARGET):
        return False
    return resolve_census_preselect_device(selector) is not None


def _pad_tables(tables: list[np.ndarray], max_bins: int) -> np.ndarray:
    padded = np.zeros((len(tables), max(max_bins, 1)), dtype=np.float32)
    for idx, table in enumerate(tables):
        values = np.asarray(table, dtype=np.float32)
        padded[idx, : values.size] = values
    return padded


def _stack_codes(code_arrays: list[np.ndarray], n_rows: int) -> np.ndarray:
    stacked = np.empty((len(code_arrays), n_rows), dtype=np.int32)
    for idx, codes in enumerate(code_arrays):
        stacked[idx] = np.asarray(codes, dtype=np.int32)
    return stacked


@dataclass
class CensusGpuQuotaEngine:
    device: torch.device
    n_rows: int
    n_cols: int
    n_pairs: int
    codes_1d: torch.Tensor
    bins_1d: torch.Tensor
    pair_codes: torch.Tensor
    pair_bins: torch.Tensor
    pair_weights: torch.Tensor
    pair_weight_scale: float
    selected: torch.Tensor
    add_1d: torch.Tensor
    add_2d: torch.Tensor
    final_score: torch.Tensor
    base_score: torch.Tensor
    support_tiebreak: torch.Tensor
    privacy_tiebreak: torch.Tensor
    w_quota_1d: float
    w_quota_2d: float
    w_static: float
    w_support: float
    w_priv: float

    @classmethod
    def build(
        cls,
        *,
        selector: Any,
        bucket_indices: dict[str, np.ndarray],
        pair_codes: list[np.ndarray],
        selected_counts_1d: dict[str, np.ndarray],
        quota_targets_1d: dict[str, np.ndarray],
        selected_counts_2d: list[np.ndarray],
        quota_targets_2d: list[np.ndarray],
        base_score: np.ndarray,
        support_tiebreak: np.ndarray,
        privacy_tiebreak: np.ndarray,
        w_quota_1d: float,
        w_quota_2d: float,
        w_static: float,
        w_support: float,
        w_priv: float,
    ) -> "CensusGpuQuotaEngine":
        device = resolve_census_preselect_device(selector)
        if device is None:
            raise RuntimeError("Census GPU preselect requested without CUDA")
        columns = list(selector.fidelity_columns)
        n_rows = int(len(base_score))
        codes_1d = _stack_codes([bucket_indices[column] for column in columns], n_rows) if columns else np.zeros((0, n_rows), dtype=np.int32)
        bins_1d = np.asarray([len(quota_targets_1d[column]) for column in columns], dtype=np.int32) if columns else np.zeros(0, dtype=np.int32)
        pair_code_arrays = [np.asarray(codes, dtype=np.int32) for codes in pair_codes]
        pair_stacked = _stack_codes(pair_code_arrays, n_rows) if pair_code_arrays else np.zeros((0, n_rows), dtype=np.int32)
        pair_bins = np.asarray([len(target) for target in quota_targets_2d], dtype=np.int32) if pair_code_arrays else np.zeros(0, dtype=np.int32)
        pair_weights = np.asarray(getattr(selector, "pair_weights", np.ones(len(pair_code_arrays), dtype=float)), dtype=np.float32)
        if pair_weights.size < len(pair_code_arrays):
            pair_weights = np.ones(len(pair_code_arrays), dtype=np.float32)
        engine = cls(
            device=device,
            n_rows=n_rows,
            n_cols=len(columns),
            n_pairs=len(pair_code_arrays),
            codes_1d=torch.as_tensor(codes_1d, device=device),
            bins_1d=torch.as_tensor(bins_1d, device=device),
            pair_codes=torch.as_tensor(pair_stacked, device=device),
            pair_bins=torch.as_tensor(pair_bins, device=device),
            pair_weights=torch.as_tensor(pair_weights[: len(pair_code_arrays)], dtype=torch.float32, device=device),
            pair_weight_scale=float(max(getattr(selector, "total_pair_weight", 1.0), 1e-12)),
            selected=torch.zeros(n_rows, dtype=torch.bool, device=device),
            add_1d=torch.zeros(n_rows, dtype=torch.float32, device=device),
            add_2d=torch.zeros(n_rows, dtype=torch.float32, device=device),
            final_score=torch.zeros(n_rows, dtype=torch.float32, device=device),
            base_score=torch.as_tensor(np.asarray(base_score, dtype=np.float32), device=device),
            support_tiebreak=torch.as_tensor(np.asarray(support_tiebreak, dtype=np.float32), device=device),
            privacy_tiebreak=torch.as_tensor(np.asarray(privacy_tiebreak, dtype=np.float32), device=device),
            w_quota_1d=float(w_quota_1d),
            w_quota_2d=float(w_quota_2d),
            w_static=float(w_static),
            w_support=float(w_support),
            w_priv=float(w_priv),
        )
        get_logger().info(
            "[preselect] census GPU quota engine device=%s rows=%d cols=%d pairs=%d mem=%.1fGiB",
            device,
            n_rows,
            engine.n_cols,
            engine.n_pairs,
            torch.cuda.max_memory_allocated(device) / (1024**3),
        )
        return engine

    def mark_selected(self, chosen: np.ndarray) -> None:
        if chosen.size == 0:
            return
        self.selected[torch.as_tensor(np.asarray(chosen, dtype=np.int64), device=self.device)] = True

    def _gather_add_support(
        self,
        codes: torch.Tensor,
        bins: torch.Tensor,
        tables: list[np.ndarray],
        weights: torch.Tensor | None,
        out: torch.Tensor,
    ) -> None:
        if codes.numel() == 0 or not tables:
            out.zero_()
            return
        max_bins = max(int(table.size) for table in tables)
        padded = torch.as_tensor(_pad_tables(tables, max_bins), device=self.device)
        idx = codes.clamp(min=0, max=max(max_bins - 1, 0))
        valid = (codes >= 0) & (codes < bins.unsqueeze(1))
        gathered = torch.gather(padded, 1, idx.long())
        gathered = gathered.masked_fill(~valid, 0.0)
        if weights is None:
            scale = max(float(gathered.shape[0]), 1.0)
            out.copy_(gathered.sum(dim=0) / scale)
            return
        weighted = gathered * weights.unsqueeze(1)
        out.copy_(weighted.sum(dim=0) / max(float(self.pair_weight_scale), 1e-12))

    def choose_batch(
        self,
        *,
        selected_counts_1d: dict[str, np.ndarray],
        quota_targets_1d: dict[str, np.ndarray],
        selected_counts_2d: list[np.ndarray],
        quota_targets_2d: list[np.ndarray],
        columns: list[str],
        take_k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        deficit_1d = [
            np.maximum(np.asarray(quota_targets_1d[column], dtype=float) - np.asarray(selected_counts_1d[column], dtype=float), 0.0)
            / np.clip(np.asarray(quota_targets_1d[column], dtype=float), 1.0, None)
            for column in columns
        ]
        deficit_2d = [
            np.maximum(np.asarray(target, dtype=float) - np.asarray(current, dtype=float), 0.0)
            / np.clip(np.asarray(target, dtype=float), 1.0, None)
            for target, current in zip(quota_targets_2d, selected_counts_2d)
        ]
        self._gather_add_support(self.codes_1d, self.bins_1d, deficit_1d, None, self.add_1d)
        self._gather_add_support(self.pair_codes, self.pair_bins, deficit_2d, self.pair_weights, self.add_2d)
        self.final_score.copy_(
            self.w_quota_1d * self.add_1d
            + self.w_quota_2d * self.add_2d
            + self.w_static * self.base_score
            + self.w_support * self.support_tiebreak
            + self.w_priv * self.privacy_tiebreak
        )
        self.final_score.masked_fill_(self.selected, float("-inf"))
        remaining = int(self.n_rows - int(self.selected.sum().item()))
        if remaining <= 0:
            empty = np.zeros(0, dtype=int)
            return empty, empty.astype(float), empty.astype(float), empty.astype(float)
        take_k = max(1, min(int(take_k), remaining))
        values, chosen = torch.topk(self.final_score, k=take_k, largest=True, sorted=False)
        chosen_np = chosen.detach().cpu().numpy().astype(int, copy=False)
        add_1d = self.add_1d[chosen].detach().cpu().numpy().astype(float, copy=False)
        add_2d = self.add_2d[chosen].detach().cpu().numpy().astype(float, copy=False)
        final_chosen = values.detach().cpu().numpy().astype(float, copy=False)
        return chosen_np, add_1d, add_2d, final_chosen
