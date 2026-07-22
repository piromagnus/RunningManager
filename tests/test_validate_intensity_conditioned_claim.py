"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Tests for intensity-conditioned claim validation helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import validate_intensity_conditioned_claim as validate


def test_paired_mape_bootstrap_positive_when_b_better() -> None:
    actual = np.array([3600.0, 4000.0, 5000.0, 4200.0])
    pred_a = actual * 1.20
    pred_b = actual * 1.05
    out = validate._paired_mape_bootstrap(actual, pred_a, pred_b, n_boot=500, seed=1)
    assert out["deltaMapePct"] > 0
    assert out["ciLow"] > 0
    assert out["relativeGainVsA"] > 0


def test_select_history_hrr_rules() -> None:
    history = pd.DataFrame(
        {
            "hrReserveRatio": [0.70, 0.75, 0.80, 0.60],
            "actualTimeSec": [3600, 7200, 5400, 1800],
        }
    )
    fixed = validate._select_history_hrr("fixed_hrr_reference", history, hrr_reference=0.88)
    assert fixed == 0.88
    median = validate._select_history_hrr("median_prior_hard_hrr", history, hrr_reference=0.88)
    assert 0.60 <= median <= 0.80
    nearest = validate._select_history_hrr(
        "nearest_prior_by_duration",
        history,
        hrr_reference=0.88,
        target_duration_h=1.9,
    )
    assert abs(nearest - 0.75) < 1e-9


def test_route_for_prescribed_prediction_strips_leakage() -> None:
    segments = pd.DataFrame(
        {
            "segmentIndex": [0, 1],
            "startKm": [0.0, 1.0],
            "meanHrReserve": [0.8, 0.7],
            "actualTimeSec": [400.0, 500.0],
            "cumTrimpBefore": [0.0, 1.0],
            "progress": [0.25, 0.75],
        }
    )
    route = validate._route_for_prescribed_prediction(segments)
    assert route["meanHrReserve"].isna().all()
    assert route["actualTimeSec"].isna().all()
    assert route["cumTrimpBefore"].isna().all()
