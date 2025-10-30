"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import json
from pathlib import Path

from persistence.csv_storage import CsvStorage
from persistence.repositories import (
    ActivitiesMetricsRepo,
    ActivitiesRepo,
    AthletesRepo,
    DailyMetricsRepo,
    PlannedMetricsRepo,
    PlannedSessionsRepo,
    SettingsRepo,
    WeeklyMetricsRepo,
)
from services.metrics_service import MetricsComputationService


def _bootstrap(tmp_path):
    storage = CsvStorage(base_dir=tmp_path)
    athletes = AthletesRepo(storage)
    athletes.create(
        {
            "athleteId": "ath-1",
            "coachId": "coach-1",
            "name": "Test",
            "thresholdsProfileId": "",
            "units": "metric",
            "hrRest": 55.0,
            "hrMax": 205.0,
        }
    )
    settings = SettingsRepo(storage)
    settings.update(
        "coach-1",
        {
            "coachId": "coach-1",
            "units": "metric",
            "distanceEqFactor": 0.01,
            "stravaSyncDays": 14,
        },
    )
    raw_dir = tmp_path / "raw" / "strava"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return storage


def test_recompute_metrics(tmp_path):
    storage = _bootstrap(tmp_path)
    planned_repo = PlannedSessionsRepo(storage)
    activities_repo = ActivitiesRepo(storage)
    metrics_service = MetricsComputationService(storage)

    planned_repo.create(
        {
            "athleteId": "ath-1",
            "date": "2025-04-10",
            "type": "FUNDAMENTAL_ENDURANCE",
            "plannedDistanceKm": 10.0,
            "plannedDurationSec": "",
            "plannedAscentM": 400,
            "targetType": "pace",
            "targetLabel": "Fundamental",
            "notes": "",
            "stepEndMode": "",
            "stepsJson": "",
        }
    )
    planned_repo.create(
        {
            "athleteId": "ath-1",
            "date": "2025-04-11",
            "type": "FUNDAMENTAL_ENDURANCE",
            "plannedDistanceKm": "",
            "plannedDurationSec": 4500,
            "plannedAscentM": 200,
            "targetType": "pace",
            "targetLabel": "Fundamental",
            "notes": "",
            "stepEndMode": "",
            "stepsJson": "",
        }
    )

    raw_path = Path("raw/strava/act-1.json")
    full_raw_path = storage.base_dir / raw_path
    full_raw_path.parent.mkdir(parents=True, exist_ok=True)
    full_raw_path.write_text(json.dumps({"sport_type": "Run"}), encoding="utf-8")

    activities_repo.create(
        {
            "activityId": "act-1",
            "athleteId": "ath-1",
            "source": "manual",
            "startTime": "2025-04-10T08:00:00Z",
            "distanceKm": 9.5,
            "elapsedSec": 4100,
            "movingSec": 3900,
            "ascentM": 380,
            "avgHr": 150.0,
            "maxHr": 170.0,
            "hasTimeseries": False,
            "polyline": "",
            "rawJsonPath": str(raw_path),
        }
    )
    activities_repo.create(
        {
            "activityId": "act-2",
            "athleteId": "ath-1",
            "source": "manual",
            "startTime": "2025-04-11T09:30:00Z",
            "distanceKm": 12.0,
            "elapsedSec": 5000,
            "movingSec": 4800,
            "ascentM": 420,
            "avgHr": 148.0,
            "maxHr": 168.0,
            "hasTimeseries": False,
            "polyline": "",
            "rawJsonPath": str(raw_path),
        }
    )

    metrics_service.recompute_all()

    activities_metrics = ActivitiesMetricsRepo(storage).list()
    assert not activities_metrics.empty
    assert len(activities_metrics) == len(ActivitiesRepo(storage).list())
    assert {"trimp", "distanceEqKm"}.issubset(set(activities_metrics.columns))
    assert activities_metrics["trimp"].ge(0).all()
    assert activities_metrics["distanceEqKm"].ge(activities_metrics["distanceKm"]).all()

    planned_metrics = PlannedMetricsRepo(storage).list()
    planned_sessions = PlannedSessionsRepo(storage).list()
    assert len(planned_metrics) == len(planned_sessions)
    assert {"distanceEqKm", "timeSec"}.issubset(set(planned_metrics.columns))

    weekly_df = WeeklyMetricsRepo(storage).list()
    assert not weekly_df.empty
    week = weekly_df.iloc[0]
    expected_columns = [
        "athleteId",
        "isoYear",
        "isoWeek",
        "weekStartDate",
        "weekEndDate",
        "plannedTimeSec",
        "actualTimeSec",
        "plannedDistanceKm",
        "plannedDistanceEqKm",
        "actualDistanceKm",
        "actualDistanceEqKm",
        "plannedTrimp",
        "actualTrimp",
        "intenseTimeSec",
        "easyTimeSec",
        "numPlannedSessions",
        "numActualSessions",
        "adherencePct",
    ]
    assert list(weekly_df.columns) == expected_columns
    assert int(week["numPlannedSessions"]) == 2
    assert int(week["numActualSessions"]) == 2
    assert float(week["actualDistanceKm"]) > 0.0
    assert float(week["actualTimeSec"]) > 0.0
    assert float(week["actualDistanceEqKm"]) >= float(week["actualDistanceKm"])

    daily_df = DailyMetricsRepo(storage).list()
    assert not daily_df.empty
    first_day = daily_df.iloc[0]
    assert float(first_day["distanceKm"]) > 0.0
    assert float(first_day["distanceEqKm"]) >= float(first_day["distanceKm"])
    assert float(first_day["acuteDistanceKm"]) >= float(first_day["distanceKm"])
