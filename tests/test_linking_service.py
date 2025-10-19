import pandas as pd
import pytest

from persistence.csv_storage import CsvStorage
from persistence.repositories import ActivitiesRepo, AthletesRepo, PlannedSessionsRepo, SettingsRepo
from services.linking_service import LinkingService
from services.metrics_service import MetricsComputationService


def _create_activity(repo: ActivitiesRepo, **overrides) -> str:
    payload = {
        "activityId": overrides.get("activityId"),
        "athleteId": overrides.get("athleteId", "ath1"),
        "source": overrides.get("source", "strava"),
        "startTime": overrides.get("startTime", "2025-01-05T07:00:00Z"),
        "distanceKm": overrides.get("distanceKm", 12.0),
        "elapsedSec": overrides.get("elapsedSec", 4000),
        "movingSec": overrides.get("movingSec", 3600),
        "ascentM": overrides.get("ascentM", 250),
        "avgHr": overrides.get("avgHr", 140),
        "maxHr": overrides.get("maxHr", 160),
        "hasTimeseries": overrides.get("hasTimeseries", True),
        "polyline": "",
        "rawJsonPath": "raw/strava/xyz.json",
    }
    return repo.create(payload)


def _create_planned(repo: PlannedSessionsRepo, **overrides) -> str:
    payload = {
        "plannedSessionId": overrides.get("plannedSessionId"),
        "athleteId": overrides.get("athleteId", "ath1"),
        "date": overrides.get("date", "2025-01-05"),
        "type": overrides.get("type", "FUNDAMENTAL_ENDURANCE"),
        "plannedDistanceKm": overrides.get("plannedDistanceKm", 12.0),
        "plannedDurationSec": overrides.get("plannedDurationSec", 3600),
        "plannedAscentM": overrides.get("plannedAscentM", 200),
        "targetType": "",
        "targetLabel": "",
        "notes": "",
        "stepEndMode": "auto",
        "stepsJson": "",
    }
    return repo.create(payload)


def test_linking_service_suggests_and_links(tmp_path):
    storage = CsvStorage(tmp_path)
    link_service = LinkingService(storage)
    activities_repo = link_service.activities
    planned_repo = link_service.sessions

    activity_id = _create_activity(
        activities_repo,
        activityId="act-1",
        startTime="2025-01-05T07:00:00Z",
        distanceKm=10.0,
        movingSec=3600,
    )
    _create_activity(
        activities_repo,
        activityId="act-2",
        startTime="2025-01-07T07:00:00Z",
        distanceKm=5.0,
        movingSec=1800,
    )
    planned_id = _create_planned(
        planned_repo,
        plannedSessionId="ps-1",
        date="2025-01-05",
        plannedDistanceKm=10.0,
        plannedDurationSec=3600,
    )
    _create_planned(
        planned_repo,
        plannedSessionId="ps-2",
        date="2025-02-01",
        plannedDistanceKm=20.0,
        plannedDurationSec=7200,
    )

    unlinked = link_service.unlinked_activities("ath1")
    assert unlinked.shape[0] == 2

    suggestions = link_service.suggest_for_activity("ath1", unlinked.iloc[0])
    assert suggestions
    assert suggestions[0]["plannedSessionId"] == planned_id

    link_id = link_service.create_link(
        "ath1",
        activity_id,
        planned_id,
        rpe=7,
        comments="Bon ressenti",
    )
    linked = link_service.linked_activities("ath1")
    assert linked.shape[0] == 1
    row = linked.iloc[0]
    assert row["activityId"] == activity_id
    assert row["plannedSessionId"] == planned_id
    assert row["rpe"] == 7
    assert pd.notna(row.get("matchScore"))

    link_service.update_link(link_id, rpe=5, comments="Fatigue")
    updated = link_service.linked_activities("ath1")
    assert updated.iloc[0]["rpe"] == 5
    assert updated.iloc[0]["comments"] == "Fatigue"

    link_service.delete_link(link_id)
    assert link_service.linked_activities("ath1").empty


def test_linking_service_enriches_with_metrics(tmp_path):
    storage = CsvStorage(tmp_path)
    link_service = LinkingService(storage)
    activities_repo = link_service.activities
    planned_repo = link_service.sessions

    athletes_repo = AthletesRepo(storage)
    athletes_repo.create(
        {
            "athleteId": "ath1",
            "coachId": "coach-1",
            "name": "Test Athlete",
            "thresholdsProfileId": "",
            "units": "metric",
            "hrRest": 50,
            "hrMax": 190,
        }
    )
    settings_repo = SettingsRepo(storage)
    settings_repo.update(
        "coach-1",
        {
            "coachId": "coach-1",
            "distanceEqFactor": 0.01,
            "units": "metric",
        },
    )

    planned_id = _create_planned(
        planned_repo,
        plannedSessionId="ps-10",
        athleteId="ath1",
        plannedDistanceKm=12.0,
        plannedDurationSec=4200,
        plannedAscentM=300,
        targetType="pace",
        targetLabel="Fundamental",
    )
    activity_id = _create_activity(
        activities_repo,
        activityId="act-10",
        athleteId="ath1",
        distanceKm=12.2,
        movingSec=4100,
        ascentM=320,
        avgHr=145,
    )

    metrics_service = MetricsComputationService(storage)
    metrics_service.recompute_all()

    unlinked = link_service.unlinked_activities("ath1")
    assert {"activityDistanceEqKm", "activityTimeSec", "activityTrimp"}.issubset(unlinked.columns)
    activity_row = unlinked[unlinked["activityId"].astype(str) == str(activity_id)].iloc[0]
    assert float(activity_row["activityDistanceEqKm"]) >= float(activity_row["distanceKm"])
    assert pd.notna(activity_row["activityTrimp"])
    assert float(activity_row["activityTrimp"]) >= 0.0

    link_service.create_link("ath1", activity_id, planned_id, rpe=6, comments="")
    linked = link_service.linked_activities("ath1")
    assert {"plannedMetricDistanceEqKm", "plannedMetricTimeSec", "plannedMetricTrimp"}.issubset(linked.columns)
    linked_row = linked.iloc[0]
    assert pytest.approx(float(linked_row["activityTrimp"]), rel=1e-6) == float(activity_row["activityTrimp"])
    assert pd.notna(linked_row["plannedMetricTrimp"])
    assert float(linked_row["plannedMetricDistanceEqKm"]) >= 0.0
