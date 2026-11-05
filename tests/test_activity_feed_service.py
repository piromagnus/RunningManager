"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import datetime as dt

import pandas as pd
import pytest

from persistence.csv_storage import CsvStorage
from persistence.repositories import (
    ActivitiesMetricsRepo,
    ActivitiesRepo,
    LinksRepo,
    PlannedSessionsRepo,
)
from services.activity_feed_service import ActivityFeedService, filter_activity_feed


@pytest.fixture()
def storage(tmp_path):
    return CsvStorage(tmp_path)


@pytest.fixture()
def feed_service(storage):
    return ActivityFeedService(storage)


def _create_activity(repo: ActivitiesRepo, **overrides) -> str:
    payload = {
        "activityId": overrides.get("activityId"),
        "athleteId": overrides.get("athleteId", "ath1"),
        "source": overrides.get("source", "strava"),
        "sportType": overrides.get("sportType", "Run"),
        "name": overrides.get("name", "Morning Run"),
        "startTime": overrides.get("startTime", "2025-01-05T07:00:00Z"),
        "distanceKm": overrides.get("distanceKm", 10.0),
        "elapsedSec": overrides.get("elapsedSec", 4000),
        "movingSec": overrides.get("movingSec", 3600),
        "ascentM": overrides.get("ascentM", 250),
        "avgHr": overrides.get("avgHr", 140),
        "maxHr": overrides.get("maxHr", 160),
        "hasTimeseries": overrides.get("hasTimeseries", True),
        "polyline": overrides.get("polyline", ""),
        "rawJsonPath": overrides.get("rawJsonPath", ""),
    }
    return repo.create(payload)


def _create_metrics(repo: ActivitiesMetricsRepo, **overrides) -> str:
    payload = {
        "activityId": overrides.get("activityId"),
        "athleteId": overrides.get("athleteId", "ath1"),
        "startDate": overrides.get("startDate", "2025-01-05"),
        "sportType": overrides.get("sportType", "Run"),
        "category": overrides.get("category", "EASY"),
        "source": overrides.get("source", "strava"),
        "distanceKm": overrides.get("distanceKm", 10.0),
        "timeSec": overrides.get("timeSec", 3600),
        "ascentM": overrides.get("ascentM", 250),
        "distanceEqKm": overrides.get("distanceEqKm", 11.0),
        "trimp": overrides.get("trimp", 120.0),
        "avgHr": overrides.get("avgHr", 140),
    }
    return repo.create(payload)


def test_activity_feed_sorted_and_enriched(feed_service, storage):
    activities_repo = ActivitiesRepo(storage)
    metrics_repo = ActivitiesMetricsRepo(storage)
    links_repo = LinksRepo(storage)

    # Older activity without link
    _create_activity(
        activities_repo,
        activityId="act-1",
        startTime="2025-01-02T08:00:00Z",
        distanceKm=8.5,
        movingSec=3200,
        ascentM=180,
        avgHr=135,
        name="Easy Jog",
    )
    _create_metrics(
        metrics_repo,
        activityId="act-1",
        distanceEqKm=8.8,
        trimp=90.5,
    )

    # More recent activity with link and metrics
    _create_activity(
        activities_repo,
        activityId="act-2",
        startTime="2025-01-05T07:00:00Z",
        distanceKm=12.0,
        movingSec=3700,
        ascentM=310,
        avgHr=145,
        name="Long Run",
    )
    _create_metrics(
        metrics_repo,
        activityId="act-2",
        distanceEqKm=12.6,
        trimp=140.2,
    )
    links_repo.create(
        {
            "linkId": "link-1",
            "plannedSessionId": "ps-1",
            "activityId": "act-2",
            "matchScore": 0.87,
            "rpe(1-10)": "",
            "comments": "",
        }
    )

    feed = feed_service.get_feed("ath1", limit=5)
    assert [item.activity_id for item in feed] == ["act-2", "act-1"]

    recent = feed[0]
    assert recent.athlete_id == "ath1"
    assert recent.sport_type == "Run"
    assert recent.linked is True
    assert pytest.approx(recent.match_score, rel=1e-6) == 0.87
    assert pytest.approx(recent.distance_km, rel=1e-6) == 12.0
    assert pytest.approx(recent.distance_eq_km, rel=1e-6) == 12.6
    assert pytest.approx(recent.trimp, rel=1e-6) == 140.2

    older = feed[1]
    assert older.athlete_id == "ath1"
    assert older.sport_type == "Run"
    assert older.linked is False
    assert older.match_score is None
    assert pytest.approx(older.distance_eq_km, rel=1e-6) == 8.8
    assert pytest.approx(older.trimp, rel=1e-6) == 90.5


def test_activity_feed_sorts_mixed_naive_and_aware_timestamps(feed_service, storage):
    activities_repo = ActivitiesRepo(storage)
    _create_activity(
        activities_repo,
        activityId="garmin-local",
        startTime="2025-01-05 08:00:00",
    )
    _create_activity(
        activities_repo,
        activityId="strava-local",
        startTime="2025-01-05T09:00:00+02:00",
    )

    feed = feed_service.get_feed("ath1")

    assert [item.activity_id for item in feed] == ["strava-local", "garmin-local"]


def test_filter_activity_feed_by_search_sport_and_metric_ranges(feed_service, storage):
    activities_repo = ActivitiesRepo(storage)
    _create_activity(
        activities_repo,
        activityId="trail-1",
        name="Forest Trail",
        sportType="TRAIL_RUN",
        distanceKm=18.0,
        movingSec=7200,
        ascentM=900,
        avgHr=150,
    )
    _create_activity(
        activities_repo,
        activityId="road-1",
        name="City Run",
        sportType="RUN",
        distanceKm=8.0,
        movingSec=2400,
        ascentM=40,
        avgHr=130,
    )
    _create_metrics(
        ActivitiesMetricsRepo(storage),
        activityId="trail-1",
        trimp=160.0,
        distanceEqKm=19.2,
    )

    feed = feed_service.get_feed("ath1", limit=0)
    filtered = filter_activity_feed(
        feed,
        query="forest",
        sport_types={"TRAIL_RUN"},
        metric_ranges={
            "distance_km": (10.0, 20.0),
            "duration_sec": (3600.0, 8000.0),
            "ascent_m": (500.0, 1000.0),
            "avg_hr": (140.0, 160.0),
            "trimp": (100.0, 200.0),
            "distance_eq_km": (18.0, 20.0),
        },
    )

    assert [item.activity_id for item in filtered] == ["trail-1"]


def test_filter_activity_feed_keeps_missing_metrics_when_not_filtered(feed_service, storage):
    activities_repo = ActivitiesRepo(storage)
    _create_activity(activities_repo, activityId="act-1")

    feed = feed_service.get_feed("ath1", limit=0)

    assert filter_activity_feed(feed) == feed


def test_activity_duration_uses_elapsed_time_when_moving_time_is_missing(feed_service, storage):
    activities_repo = ActivitiesRepo(storage)
    _create_activity(
        activities_repo,
        activityId="elapsed-only",
        movingSec=None,
        elapsedSec=4200,
    )

    item = feed_service.get_feed("ath1", limit=0)[0]

    assert item.duration_sec == 4200


def test_available_sport_types(feed_service, storage):
    activities_repo = ActivitiesRepo(storage)
    _create_activity(activities_repo, activityId="act-1", sportType="run")
    _create_activity(activities_repo, activityId="act-2", sportType="trail_run")
    _create_activity(activities_repo, activityId="act-3", sportType="HIKE")
    types = feed_service.available_sport_types("ath1")
    assert types == ["HIKE", "RUN", "TRAIL_RUN"]


def test_unlinked_planned_sessions_sorted_recent_past_to_future(feed_service, storage):
    planned_repo = PlannedSessionsRepo(storage)
    links_repo = LinksRepo(storage)

    planned_repo.create(
        {
            "plannedSessionId": "ps-old",
            "athleteId": "ath1",
            "date": "2024-12-01",
            "type": "FUNDAMENTAL_ENDURANCE",
            "plannedDistanceKm": 20.0,
            "plannedDurationSec": 7200,
            "plannedAscentM": 500,
            "targetType": "",
            "targetLabel": "",
            "notes": "",
            "stepEndMode": "auto",
            "stepsJson": "",
        }
    )
    planned_repo.create(
        {
            "plannedSessionId": "ps-past",
            "athleteId": "ath1",
            "date": "2025-01-08",
            "type": "INTERVAL_SIMPLE",
            "plannedDistanceKm": 12.0,
            "plannedDurationSec": 3600,
            "plannedAscentM": 200,
            "targetType": "pace",
            "targetLabel": "Threshold",
            "notes": "",
            "stepEndMode": "auto",
            "stepsJson": "",
        }
    )
    planned_repo.create(
        {
            "plannedSessionId": "ps-future",
            "athleteId": "ath1",
            "date": "2025-01-12",
            "type": "LONG_RUN",
            "plannedDistanceKm": 28.0,
            "plannedDurationSec": 9300,
            "plannedAscentM": 800,
            "targetType": "pace",
            "targetLabel": "Endurance",
            "notes": "",
            "stepEndMode": "auto",
            "stepsJson": "",
        }
    )
    planned_repo.create(
        {
            "plannedSessionId": "ps-linked",
            "athleteId": "ath1",
            "date": "2025-01-10",
            "type": "FUNDAMENTAL_ENDURANCE",
            "plannedDistanceKm": 15.0,
            "plannedDurationSec": 5400,
            "plannedAscentM": 300,
            "targetType": "",
            "targetLabel": "",
            "notes": "",
            "stepEndMode": "auto",
            "stepsJson": "",
        }
    )
    links_repo.create(
        {
            "linkId": "link-existing",
            "plannedSessionId": "ps-linked",
            "activityId": "act-999",
            "matchScore": 0.9,
            "rpe(1-10)": "",
            "comments": "",
        }
    )

    reference_date = dt.date(2025, 1, 10)
    sessions = feed_service.get_unlinked_planned_sessions(
        "ath1", reference_date=reference_date, max_items=5
    )

    # Old session filtered out; order is recent past then upcoming
    assert [session.planned_session_id for session in sessions] == ["ps-past", "ps-future"]
    assert sessions[0].date == pd.Timestamp("2025-01-08").date()
    assert sessions[1].date == pd.Timestamp("2025-01-12").date()
