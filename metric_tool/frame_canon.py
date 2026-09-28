from __future__ import annotations

import pandas as pd


def strip_string_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace on string cells without changing NA.

    Adult-style UCI dumps store categoricals as ``' Male'``. Selection/validation
    often persist the stripped form ``'Male'``. End metrics must compare both
    sides after the same canonicalize, or TV/Contingency scores collapse to 0.
    """
    work = df.copy()
    for column in work.columns:
        series = work[column]
        if pd.api.types.is_object_dtype(series) or str(series.dtype).startswith("string"):
            work[column] = series.map(lambda value: value.strip() if isinstance(value, str) else value)
    return work
