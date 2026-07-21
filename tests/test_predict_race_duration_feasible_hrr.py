"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Tests for duration-feasible constant-HRR selection in prospective prediction.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_predict_module():
    spec = importlib.util.spec_from_file_location(
        "predict_race_constant_hrr",
        REPO_ROOT / "scripts" / "predict_race_constant_hrr.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_select_duration_feasible_picks_sustainable_not_reference() -> None:
    """Long routes should not select HRR_ref when the envelope forbids it."""
    mod = _load_predict_module()
    # Flat 40 km → ~2.5–3.5 h at high effort; envelope caps high HRR hard.
    n = 40
    segments = pd.DataFrame(
        {
            "segmentIndex": np.arange(n),
            "startKm": np.arange(n, dtype=float),
            "endKm": np.arange(1, n + 1, dtype=float),
            "distanceKm": np.ones(n),
            "elevGainM": np.zeros(n),
            "elevLossM": np.zeros(n),
            "meanAltitudeM": np.zeros(n),
            "avgGrade": np.zeros(n),
            "gapFactorIntegrated": np.ones(n),
            "gapFactorAvgGrade": np.ones(n),
            "terrainFamily": ["flat"] * n,
            "progress": (np.arange(n) + 0.5) / n,
            "activityId": ["route"] * n,
            "actualTimeSec": [np.nan] * n,
            "meanHrReserve": [np.nan] * n,
        }
    )
    physiology = {
        "vma_flat_kmh": 18.0,
        "hrr_reference": 0.88,
        "hrr_min_factor": 0.30,
        "hrr_max_factor": 1.00,
        "decay_lambda": 0.20,
        "min_fatigue_factor": 0.60,
        "gap_steep_threshold": 0.15,
        "gap_soft_start": 0.04,
        "gap_climb_scale": 1.0,
        "gap_descent_scale": 1.0,
    }
    fit = {"alpha": 0.95, "fatigueCoef": 0.4, "fatigueModel": "exponential"}
    # Synthetic envelope: HRR 0.88 only sustainable for ~20 min; 0.75 for 5 h.
    power_law = {
        "coefficient": 0.82,
        "exponent": -0.05,
        "hrrMin": 0.30,
        "hrrMax": 0.98,
        "minWindowSec": 300.0,
        "maxWindowSec": 24 * 3600.0,
    }
    best, sweep = mod.select_duration_feasible_constant_hrr(
        segments,
        fit=fit,
        physiology=physiology,
        power_law_params=power_law,
    )
    assert not sweep.empty
    assert bool(best["feasible"])
    assert float(best["hrr"]) < float(physiology["hrr_reference"])
    assert float(best["predictedTimeSec"]) <= float(best["maxSustainableSec"])
    # Reference HRR row should be infeasible on this envelope.
    ref_row = sweep.loc[sweep["hrr"].round(2) == 0.88]
    if not ref_row.empty:
        assert not bool(ref_row.iloc[0]["feasible"])
