from __future__ import annotations

from typing import Any

import pandas as pd

from .tabdiff_density import TabDiffMetricRunner


def evaluate_dcr(runner: TabDiffMetricRunner, df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    func = getattr(runner.metrics, "evaluate_dcr")
    return func(df.copy())
