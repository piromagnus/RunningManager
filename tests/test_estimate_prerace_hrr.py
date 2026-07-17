"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts import estimate_prerace_hrr as prerace


def test_local_time_sensitivity_uses_central_sweep_difference() -> None:
    sweep = pd.DataFrame(
        {
            "hrr": [0.70, 0.71, 0.72],
            "predictedTimeSec": [260.0, 250.0, 240.0],
        }
    )

    sensitivity = prerace._local_time_sensitivity_sec_per_hrr(sweep, 0.71)

    assert sensitivity == pytest.approx(-1000.0)


def test_route_uncertainty_combines_model_and_power_law_components() -> None:
    sweep = pd.DataFrame(
        {
            "hrr": [0.70, 0.71, 0.72],
            "predictedTimeSec": [260.0, 250.0, 240.0],
        }
    )
    best = pd.Series({"hrr": 0.71, "predictedTimeSec": 250.0})
    params = {
        "coefficient": 0.80,
        "exponent": -0.10,
        "maeHrr": 0.02,
        "minWindowSec": 300.0,
        "maxWindowSec": 86_400.0,
        "hrrMin": 0.30,
        "hrrMax": 0.98,
    }

    metrics = prerace._route_uncertainty_metrics(
        best,
        sweep,
        params,
        validation_metrics={"raceMaeSec": 30.0, "raceMapePct": 5.0, "raceR2": 0.9},
        route_model_mae_sec=None,
    )

    assert metrics["powerLawTimeUncertaintySec"] == pytest.approx(20.0)
    assert metrics["timeUncertaintySec"] == pytest.approx(math.sqrt(30.0**2 + 20.0**2))
    assert metrics["routeModelMapePct"] == pytest.approx(5.0)
