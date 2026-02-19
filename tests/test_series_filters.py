"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import pandas as pd

from utils.series_filters import filter_series_outliers


def test_filter_series_outliers_time_reference() -> None:
    timestamps = pd.date_range("2025-01-01", periods=9, freq="1s")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "elevationM": [100, 101, 102, 500, 103, 104, 105, 106, 107],
        }
    )

    filtered = filter_series_outliers(
        df, value_col="elevationM", reference_col="timestamp", window=5.0, sigma=3.0
    )

    assert filtered.loc[3, "elevationM"] != 500
    assert abs(filtered.loc[3, "elevationM"] - 102.5) < 0.01


def test_filter_series_outliers_distance_reference() -> None:
    df = pd.DataFrame(
        {
            "cumulated_distance": [0.0, 0.1, 0.2, 0.3, 0.4],
            "elevationM": [10, 11, 100, 12, 13],
        }
    )

    filtered = filter_series_outliers(
        df, value_col="elevationM", reference_col="cumulated_distance", window=0.3, sigma=3.0
    )

    assert filtered.loc[2, "elevationM"] != 100
    assert abs(filtered.loc[2, "elevationM"] - 11.5) < 0.01
