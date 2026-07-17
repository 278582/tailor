from __future__ import annotations

from typing import Any

import pandas as pd

from post_selection_tool.utility_proxy import compute_utility_exact_metrics


def evaluate_mle(selector: Any, df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    return compute_utility_exact_metrics(selector, df, test_df)
