from pathlib import Path
from persistence.csv_storage import CsvStorage
from persistence.repositories import ActivitiesRepo, PlannedSessionsRepo
from utils.ids import new_id


def test_activities_repo_crud(tmp_path):
    storage = CsvStorage(base_dir=tmp_path)
    repo = ActivitiesRepo(storage)
    aid = new_id()
    repo.create(
        {
            "activityId": aid,
            "athleteId": "ath1",
            "source": "strava",
            "startTime": "2025-10-06T10:00:00Z",
            "distanceKm": 1.0,
            "elapsedSec": 10,
            "movingSec": 10,
            "ascentM": 1,
            "avgHr": 100,
            "maxHr": 120,
            "hasTimeseries": False,
            "polyline": "",
            "rawJsonPath": "raw/strava/act.json",
        }
    )
    assert repo.get(aid)["activityId"] == aid
    repo.update(aid, {"avgHr": 101})
    assert repo.get(aid)["avgHr"] == 101
    repo.delete(aid)
    assert repo.get(aid) is None


def test_planned_sessions_repo_create(tmp_path):
    storage = CsvStorage(base_dir=tmp_path)
    repo = PlannedSessionsRepo(storage)
    sid = repo.create(
        {
            "athleteId": "ath1",
            "date": "2025-10-06",
            "type": "FUNDAMENTAL_ENDURANCE",
            "plannedDistanceKm": 10.0,
            "plannedDurationSec": 3600,
            "plannedAscentM": 300,
            "targetType": "hr",
            "targetLabel": "Fundamental",
            "notes": "",
            "stepEndMode": "auto",
        }
    )
    got = repo.get(sid)
    assert got is not None
    assert got["plannedSessionId"] == sid
