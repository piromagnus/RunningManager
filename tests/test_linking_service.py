import pandas as pd

from persistence.csv_storage import CsvStorage
from persistence.repositories import ActivitiesRepo, PlannedSessionsRepo
from services.linking_service import LinkingService


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
