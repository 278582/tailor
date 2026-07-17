from __future__ import annotations

from typing import Any

import numpy as np


def drop_constant_objectives(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or len(points) == 0:
        return points
    variable_mask = np.any(points != points[0:1, :], axis=0)
    return points[:, variable_mask]


def non_dominated_sort_report(points: np.ndarray, fronts: list[list[int]]) -> dict[str, Any]:
    sort_points = drop_constant_objectives(points)
    unique_rows = int(len(np.unique(sort_points, axis=0))) if sort_points.ndim == 2 and len(sort_points) else 0
    return {
        "algorithm": "deterministic_exact_non_dominated_sort",
        "exact": True,
        "approximate": False,
        "rows": int(len(points)),
        "objectives": int(points.shape[1]) if points.ndim == 2 else 0,
        "active_objectives": int(sort_points.shape[1]) if sort_points.ndim == 2 else 0,
        "constant_objectives_dropped": int(
            (points.shape[1] - sort_points.shape[1])
            if points.ndim == 2 and sort_points.ndim == 2
            else 0
        ),
        "unique_points": unique_rows,
        "front_count": int(len(fronts)),
        "first_front_size": int(len(fronts[0])) if fronts else 0,
    }
