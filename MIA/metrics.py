from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_recall_curve, roc_auc_score
from sklearn.metrics import roc_curve


@dataclass(frozen=True)
class MetricReport:
    attack: str
    n_member: int
    n_nonmember: int
    auroc: float | None
    auprc: float | None
    attack_advantage: float | None
    balanced_accuracy: float | None
    threshold: float | None
    tpr_at_fpr_1pct: float | None
    tpr_at_fpr_5pct: float | None
    tpr_at_fpr_10pct: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack": self.attack,
            "n_member": self.n_member,
            "n_nonmember": self.n_nonmember,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "attack_advantage": self.attack_advantage,
            "balanced_accuracy": self.balanced_accuracy,
            "threshold": self.threshold,
            "tpr_at_fpr_1pct": self.tpr_at_fpr_1pct,
            "tpr_at_fpr_5pct": self.tpr_at_fpr_5pct,
            "tpr_at_fpr_10pct": self.tpr_at_fpr_10pct,
        }


def summarize_binary_scores(attack: str, labels, scores) -> MetricReport:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    n_member = int(np.sum(labels == 1))
    n_nonmember = int(np.sum(labels == 0))
    if n_member == 0 or n_nonmember == 0 or len(np.unique(scores)) < 2:
        return MetricReport(
            attack=attack,
            n_member=n_member,
            n_nonmember=n_nonmember,
            auroc=None,
            auprc=None,
            attack_advantage=None,
            balanced_accuracy=None,
            threshold=None,
            tpr_at_fpr_1pct=None,
            tpr_at_fpr_5pct=None,
            tpr_at_fpr_10pct=None,
        )

    fpr, tpr, thresholds = roc_curve(labels, scores)
    advantages = tpr - fpr
    candidate_indices = np.flatnonzero(np.isfinite(thresholds))
    if candidate_indices.size:
        best_idx = int(candidate_indices[np.nanargmax(advantages[candidate_indices])])
    else:
        best_idx = int(np.nanargmax(advantages))
    threshold = float(thresholds[best_idx])
    predictions = (scores >= threshold).astype(int)

    return MetricReport(
        attack=attack,
        n_member=n_member,
        n_nonmember=n_nonmember,
        auroc=float(roc_auc_score(labels, scores)),
        auprc=float(average_precision_score(labels, scores)),
        attack_advantage=float(advantages[best_idx]),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        threshold=threshold,
        tpr_at_fpr_1pct=_tpr_at_fpr(fpr, tpr, 0.01),
        tpr_at_fpr_5pct=_tpr_at_fpr(fpr, tpr, 0.05),
        tpr_at_fpr_10pct=_tpr_at_fpr(fpr, tpr, 0.10),
    )


def _tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, limit: float) -> float:
    mask = fpr <= limit
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def best_attack(metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    available = [metric for metric in metrics if metric.get("auroc") is not None]
    if not available:
        return None
    return max(available, key=lambda item: (float(item["auroc"]), float(item.get("attack_advantage") or 0.0)))


def precision_recall_points(labels, scores) -> dict[str, list[float]]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    if np.sum(labels[valid] == 1) == 0 or np.sum(labels[valid] == 0) == 0:
        return {"precision": [], "recall": [], "thresholds": []}
    precision, recall, thresholds = precision_recall_curve(labels[valid], scores[valid])
    return {
        "precision": [float(value) for value in precision],
        "recall": [float(value) for value in recall],
        "thresholds": [float(value) for value in thresholds],
    }
