"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def filter_series_outliers(
    df: pd.DataFrame,
    value_col: str,
    reference_col: Optional[str] = None,
    *,
    window: float = 7.0,
    sigma: float = 3.0,
) -> pd.DataFrame:
    """Filter outliers using a Hampel-style median/MAD window and neighbor replacement.

    The window is defined on a reference axis:
    - If reference_col is datetime-like, window is interpreted in seconds.
    - If reference_col is numeric (e.g., cumulated_distance), window is in the same units.
    - If reference_col is None, the window is in sample counts (index order).

    Outliers are replaced by the mean of immediate neighbors (by reference order).
    Falls back to the window median when neighbors are missing.
    """
    if value_col not in df.columns:
        return df.copy()

    if reference_col is not None and reference_col not in df.columns:
        return df.copy()

    if window <= 0:
        return df.copy()

    result = df.copy()

    values = pd.to_numeric(result[value_col], errors="coerce")
    if reference_col is None:
        reference = pd.Series(np.arange(len(result)), index=result.index)
        window_span = float(window)
    else:
        reference = result[reference_col]
        if is_datetime64_any_dtype(reference) or reference.dtype == "object":
            reference = pd.to_datetime(reference, errors="coerce", utc=True)
            if isinstance(window, (pd.Timedelta, str)):
                window_span = pd.to_timedelta(window).value
            else:
                window_span = float(window) * 1e9
            reference = reference.view("int64")
        else:
            reference = pd.to_numeric(reference, errors="coerce")
            window_span = float(window)

    valid_mask = values.notna() & reference.notna()
    if valid_mask.sum() < 3:
        return result

    values_valid = values[valid_mask].to_numpy()
    refs_valid = reference[valid_mask].to_numpy()
    orig_positions = np.flatnonzero(valid_mask.to_numpy())

    order = np.argsort(refs_valid)
    values_sorted = values_valid[order]
    refs_sorted = refs_valid[order]
    positions_sorted = orig_positions[order]

    half_window = window_span / 2.0
    replacements = values_sorted.copy()

    for idx in range(len(values_sorted)):
        ref_val = refs_sorted[idx]
        left = np.searchsorted(refs_sorted, ref_val - half_window, side="left")
        right = np.searchsorted(refs_sorted, ref_val + half_window, side="right")
        window_vals = values_sorted[left:right]

        if len(window_vals) == 0:
            continue

        median_val = float(np.nanmedian(window_vals))
        mad = float(np.nanmedian(np.abs(window_vals - median_val)))
        scale = 1.4826 * mad

        current_val = values_sorted[idx]
        if scale > 0:
            is_outlier = abs(current_val - median_val) > sigma * scale
        else:
            is_outlier = abs(current_val - median_val) > 0

        if not is_outlier:
            continue

        prev_val = values_sorted[idx - 1] if idx - 1 >= 0 else np.nan
        next_val = values_sorted[idx + 1] if idx + 1 < len(values_sorted) else np.nan
        neighbors = [val for val in (prev_val, next_val) if not pd.isna(val)]
        if neighbors:
            replacements[idx] = float(np.mean(neighbors))
        else:
            replacements[idx] = median_val

    filtered = values.copy()
    filtered.iloc[positions_sorted] = replacements
    result[value_col] = filtered
    return result
