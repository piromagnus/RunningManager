"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import json
from pathlib import Path

import pandas as pd
import pytest

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


def test_ski_distance_eq_applied(tmp_path):
    storage = _bootstrap(tmp_path)
    settings = SettingsRepo(storage)
    settings.update(
        "coach-1",
        {
            "coachId": "coach-1",
            "skiEqDistance": 0.8,
            "skiEqAscent": 0.02,
            "skiEqDescent": 0.0,
        },
    )
    activities_repo = ActivitiesRepo(storage)
    activities_repo.create(
        {
            "activityId": "ski-1",
            "athleteId": "ath-1",
            "source": "manual",
            "sportType": "BackcountrySki",
            "startTime": "2025-02-12T09:00:00Z",
            "distanceKm": 12.0,
            "elapsedSec": 4200,
            "movingSec": 4000,
            "ascentM": 500.0,
            "avgHr": 145.0,
            "maxHr": 170.0,
            "hasTimeseries": False,
            "polyline": "",
            "rawJsonPath": "",
        }
    )

    MetricsComputationService(storage).recompute_athlete("ath-1")

    metrics = ActivitiesMetricsRepo(storage).list()
    row = metrics[metrics["activityId"].astype(str) == "ski-1"].iloc[0]
    expected = 12.0 * 0.8 + 500.0 * 0.02
    assert row["category"] == "BACKCOUNTRY_SKI"
    assert pytest.approx(float(row["distanceEqKm"]), rel=1e-6) == expected


def test_recompute_for_activities_refreshes_only_impacted_tail(tmp_path):
    storage = _bootstrap(tmp_path)
    activities_repo = ActivitiesRepo(storage)
    metrics_service = MetricsComputationService(storage)

    # Seed baseline activities in early January and compute initial aggregates.
    activities_repo.create(
        {
            "activityId": "act-old-1",
            "athleteId": "ath-1",
            "source": "manual",
            "sportType": "Run",
            "startTime": "2025-01-01T08:00:00Z",
            "distanceKm": 10.0,
            "elapsedSec": 3600,
            "movingSec": 3500,
            "ascentM": 120.0,
            "avgHr": 145.0,
            "maxHr": 165.0,
            "hasTimeseries": False,
            "polyline": "",
            "rawJsonPath": "",
        }
    )
    activities_repo.create(
        {
            "activityId": "act-old-2",
            "athleteId": "ath-1",
            "source": "manual",
            "sportType": "Run",
            "startTime": "2025-01-02T08:00:00Z",
            "distanceKm": 8.0,
            "elapsedSec": 3200,
            "movingSec": 3100,
            "ascentM": 90.0,
            "avgHr": 140.0,
            "maxHr": 160.0,
            "hasTimeseries": False,
            "polyline": "",
            "rawJsonPath": "",
        }
    )
    metrics_service.recompute_all(athlete_id="ath-1")

    # Simulate a pre-existing historical value that should not be touched when
    # recomputing a much later activity.
    daily_repo = DailyMetricsRepo(storage)
    daily_df = daily_repo.list(athleteId="ath-1")
    assert not daily_df.empty
    jan1_mask = daily_df["date"].astype(str) == "2025-01-01"
    assert jan1_mask.any()
    daily_df.loc[jan1_mask, "acuteDistanceKm"] = 999.0
    storage.write_csv(daily_repo.file_name, daily_df[daily_repo.headers])

    # Add a new activity much later and recompute incrementally.
    activities_repo.create(
        {
            "activityId": "act-new",
            "athleteId": "ath-1",
            "source": "manual",
            "sportType": "Run",
            "startTime": "2025-01-15T08:00:00Z",
            "distanceKm": 12.0,
            "elapsedSec": 4200,
            "movingSec": 4050,
            "ascentM": 160.0,
            "avgHr": 150.0,
            "maxHr": 170.0,
            "hasTimeseries": False,
            "polyline": "",
            "rawJsonPath": "",
        }
    )
    metrics_service.recompute_for_activities(["act-new"])

    refreshed_daily = daily_repo.list(athleteId="ath-1")
    jan1_row = refreshed_daily[refreshed_daily["date"].astype(str) == "2025-01-01"].iloc[0]
    assert float(jan1_row["acuteDistanceKm"]) == pytest.approx(999.0)

    jan15_row = refreshed_daily[refreshed_daily["date"].astype(str) == "2025-01-15"]
    assert not jan15_row.empty
    assert pd.to_numeric(jan15_row["distanceKm"], errors="coerce").fillna(0.0).iloc[0] > 0.0
