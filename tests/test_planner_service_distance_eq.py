"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import math

import pytest

from persistence.csv_storage import CsvStorage
from persistence.repositories import SettingsRepo, ThresholdsRepo
from services.planner_service import PlannerService


def _bootstrap_storage(tmp_path):
    storage = CsvStorage(base_dir=tmp_path)
    settings_repo = SettingsRepo(storage)
    settings_repo.update(
        "coach-1",
        {
            "units": "metric",
            "distanceEqFactor": 0.01,
            "stravaSyncDays": 14,
        },
    )
    thresholds_repo = ThresholdsRepo(storage)
    thresholds_repo.create(
        {
            "thresholdId": "thr-1",
            "athleteId": "athlete-1",
            "name": "Fundamental",
            "hrMin": "",
            "hrMax": "",
            "paceFlatKmhMin": 10.0,
            "paceFlatKmhMax": 10.0,
            "ascentRateMPerHMin": "",
            "ascentRateMPerHMax": "",
        }
    )
    return storage


def test_distance_eq_and_derivations(tmp_path):
    storage = _bootstrap_storage(tmp_path)
    planner = PlannerService(storage)

    eq = planner.compute_distance_eq_km(10.0, 500)
    assert math.isclose(eq, 15.0, rel_tol=1e-6)

    derived_distance = planner.derive_from_distance("athlete-1", 12.0, 400)
    assert math.isclose(derived_distance["distanceKm"], 12.0)
    assert math.isclose(derived_distance["distanceEqKm"], 16.0)
    assert derived_distance["durationSec"] == pytest.approx((16 / 10.0) * 3600, rel=1e-6)

    derived_duration = planner.derive_from_duration("athlete-1", 7200, 300)
    assert math.isclose(derived_duration["distanceEqKm"], 20.0)
    assert math.isclose(derived_duration["distanceKm"], 17.0)
    assert derived_duration["durationSec"] == 7200

    high_ascent = planner.derive_from_duration("athlete-1", 1800, 5000)
    assert high_ascent["distanceKm"] >= 0.0


def test_compute_session_distance_eq(tmp_path):
    storage = _bootstrap_storage(tmp_path)
    planner = PlannerService(storage)

    session = {
        "type": "FUNDAMENTAL_ENDURANCE",
        "plannedDistanceKm": 8.0,
        "plannedDurationSec": 0,
        "plannedAscentM": 200,
    }
    eq = planner.compute_session_distance_eq("athlete-1", session)
    assert math.isclose(eq, 10.0)

    interval_session = {
        "type": "INTERVAL_SIMPLE",
        "plannedAscentM": 150,
        "stepsJson": '{"warmupSec":600,"repeats":[{"workSec":120,"recoverSec":60,"targetType":"pace","targetLabel":"Fundamental"}],"cooldownSec":300}',
    }
    interval_eq = planner.compute_session_distance_eq("athlete-1", interval_session)
    assert interval_eq is not None
    assert interval_eq > 0.0
