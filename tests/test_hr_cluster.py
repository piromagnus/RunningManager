"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from graph.hr_cluster import _fit_regression_with_outliers


def test_fit_regression_outliers_with_non_contiguous_index(monkeypatch) -> None:
    class DummyRANSAC:
        def __init__(self, *args, **kwargs) -> None:
            self.inlier_mask_ = None

        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            mask = np.ones(len(y), dtype=bool)
            mask[-1] = False
            self.inlier_mask_ = mask

    monkeypatch.setattr("graph.hr_cluster.RANSACRegressor", DummyRANSAC)

    grouped = pd.DataFrame(
        {
            "speed_mean": [10.0, 11.0, 12.0, 13.0],
            "hr_mean": [140.0, 145.0, 150.0, 190.0],
        },
        index=[0, 1, 2, 4],
    )

    updated, regression_line_df = _fit_regression_with_outliers(grouped, x_domain=[8.0, 15.0])

    assert "is_outlier" in updated.columns
    assert bool(updated.iloc[-1]["is_outlier"]) is True
    assert regression_line_df is not None
    assert not regression_line_df.empty
