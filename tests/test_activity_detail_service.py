import json
from pathlib import Path

import pandas as pd
import pytest

from persistence.csv_storage import CsvStorage
from persistence.repositories import (
    ActivitiesMetricsRepo,
    ActivitiesRepo,
    AthletesRepo,
    LinksRepo,
    PlannedMetricsRepo,
    PlannedSessionsRepo,
)
from services.activity_detail_service import ActivityDetailService
from services.timeseries_service import TimeseriesService
from utils.config import Config


@pytest.fixture()
def cfg(tmp_path) -> Config:
    data_dir = tmp_path
    timeseries_dir = data_dir / "timeseries"
    raw_dir = data_dir / "raw" / "strava"
    laps_dir = data_dir / "laps"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    laps_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        strava_client_id=None,
        strava_client_secret=None,
        strava_redirect_uri=None,
        data_dir=data_dir,
        encryption_key=None,
        timeseries_dir=timeseries_dir,
        raw_strava_dir=raw_dir,
        laps_dir=laps_dir,
        mapbox_token=None,
    )


@pytest.fixture()
def storage(cfg: Config) -> CsvStorage:
    return CsvStorage(cfg.data_dir)


@pytest.fixture()
def detail_service(storage: CsvStorage, cfg: Config) -> ActivityDetailService:
    ts_service = TimeseriesService(cfg)
    return ActivityDetailService(storage, cfg, ts_service)


def _create_activity(
    repo: ActivitiesRepo, *, raw_path: Path, **overrides
) -> str:
    payload = {
        "activityId": overrides.get("activityId"),
        "athleteId": overrides.get("athleteId", "ath1"),
        "source": overrides.get("source", "strava"),
        "sportType": overrides.get("sportType", "Run"),
        "name": overrides.get("name", "Morning Run"),
        "startTime": overrides.get("startTime", "2025-01-05T07:00:00Z"),
        "distanceKm": overrides.get("distanceKm", 12.0),
        "elapsedSec": overrides.get("elapsedSec", 4000),
        "movingSec": overrides.get("movingSec", 3600),
        "ascentM": overrides.get("ascentM", 250),
        "avgHr": overrides.get("avgHr", 140),
        "maxHr": overrides.get("maxHr", 160),
        "hasTimeseries": overrides.get("hasTimeseries", True),
        "polyline": overrides.get("polyline", ""),
        "rawJsonPath": overrides.get("rawJsonPath", str(raw_path)),
    }
    return repo.create(payload)


def test_activity_detail_summary_and_comparison(detail_service, storage, cfg):
    activities_repo = ActivitiesRepo(storage)
    metrics_repo = ActivitiesMetricsRepo(storage)
    planned_repo = PlannedSessionsRepo(storage)
    planned_metrics_repo = PlannedMetricsRepo(storage)
    links_repo = LinksRepo(storage)
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

    raw_path = cfg.raw_strava_dir / "act-1.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(
            {
                "name": "Sunday Long Run",
                "description": "With climbs and sun",
                "type": "Run",
            }
        ),
        encoding="utf-8",
    )
    _create_activity(
        activities_repo,
        raw_path=raw_path,
        activityId="act-1",
        startTime="2025-01-05T07:00:00Z",
        distanceKm=12.3,
        movingSec=3750,
        elapsedSec=3900,
        ascentM=320,
        avgHr=144,
        polyline="_p~iF~ps|U_ulLnnqC_mqNvxq`@",
    )
    metrics_repo.create(
        {
            "activityId": "act-1",
            "athleteId": "ath1",
            "startDate": "2025-01-05",
            "sportType": "Run",
            "category": "EASY",
            "source": "strava",
            "distanceKm": 12.3,
            "timeSec": 3750,
            "ascentM": 320,
            "distanceEqKm": 13.0,
            "trimp": 135.0,
            "avgHr": 144,
        }
    )
    planned_repo.create(
        {
            "plannedSessionId": "ps-1",
            "athleteId": "ath1",
            "date": "2025-01-05",
            "type": "LONG_RUN",
            "plannedDistanceKm": 12.0,
            "plannedDurationSec": 3600,
            "plannedAscentM": 300,
            "targetType": "",
            "targetLabel": "",
            "notes": "",
            "stepEndMode": "auto",
            "stepsJson": "",
        }
    )
    planned_metrics_repo.create(
        {
            "plannedSessionId": "ps-1",
            "athleteId": "ath1",
            "date": "2025-01-05",
            "type": "LONG_RUN",
            "timeSec": 3600,
            "distanceKm": 12.0,
            "distanceEqKm": 12.5,
            "trimp": 120.0,
        }
    )
    links_repo.create(
        {
            "linkId": "link-1",
            "plannedSessionId": "ps-1",
            "activityId": "act-1",
            "matchScore": 0.92,
            "rpe(1-10)": "",
            "comments": "",
        }
    )

    detail = detail_service.get_detail("ath1", "act-1")

    assert detail.title == "Sunday Long Run"
    assert detail.description == "With climbs and sun"
    assert detail.linked is True
    assert detail.athlete_id == "ath1"
    assert detail.summary.distance_km == pytest.approx(12.3)
    assert detail.summary.moving_sec == 3750
    assert detail.summary.trimp == pytest.approx(135.0)
    assert detail.summary.distance_eq_km == pytest.approx(13.0)

    assert detail.comparison is not None
    assert detail.comparison.distance.actual == pytest.approx(12.3)
    assert detail.comparison.distance.planned == pytest.approx(12.0)
    assert detail.comparison.distance.delta == pytest.approx(0.3)
    assert detail.comparison.trimp.delta == pytest.approx(15.0)
    assert detail.comparison.ascent.delta == pytest.approx(20)


def test_activity_detail_map_path_prefers_polyline(detail_service, storage, cfg):
    activities_repo = ActivitiesRepo(storage)
    metrics_repo = ActivitiesMetricsRepo(storage)
    raw_path = cfg.raw_strava_dir / "act-poly.json"
    raw_path.write_text(json.dumps({"name": "Polyline"}), encoding="utf-8")

    _create_activity(
        activities_repo,
        raw_path=raw_path,
        activityId="act-poly",
        polyline="_p~iF~ps|U_ulLnnqC_mqNvxq`@",
    )
    metrics_repo.create(
        {
            "activityId": "act-poly",
            "athleteId": "ath1",
            "startDate": "2025-01-03",
            "sportType": "Run",
            "category": "EASY",
            "source": "strava",
            "distanceKm": 10.0,
            "timeSec": 3200,
            "ascentM": 200,
            "distanceEqKm": 10.5,
            "trimp": 90.0,
            "avgHr": 140,
        }
    )

    detail = detail_service.get_detail("ath1", "act-poly")
    assert detail.map_path is not None
    assert len(detail.map_path) >= 2
    # First point of the canonical polyline example
    first = detail.map_path[0]
    assert first.lon == pytest.approx(-120.2, rel=1e-3)
    assert first.lat == pytest.approx(38.5, rel=1e-3)


def test_activity_detail_map_path_from_timeseries_when_no_polyline(detail_service, storage, cfg):
    activities_repo = ActivitiesRepo(storage)
    metrics_repo = ActivitiesMetricsRepo(storage)

    raw_path = cfg.raw_strava_dir / "act-ts.json"
    raw_path.write_text(json.dumps({"name": "Timeseries"}), encoding="utf-8")

    activity_id = _create_activity(
        activities_repo,
        raw_path=raw_path,
        activityId="act-ts",
        polyline="",
    )
    metrics_repo.create(
        {
            "activityId": "act-ts",
            "athleteId": "ath1",
            "startDate": "2025-01-04",
            "sportType": "Run",
            "category": "EASY",
            "source": "strava",
            "distanceKm": 9.0,
            "timeSec": 3000,
            "ascentM": 150,
            "distanceEqKm": 9.2,
            "trimp": 80.0,
            "avgHr": 138,
        }
    )

    ts_path = cfg.timeseries_dir / f"{activity_id}.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2025-01-04T07:00:00Z",
                "2025-01-04T07:05:00Z",
                "2025-01-04T07:10:00Z",
            ],
            "lat": [45.0, 45.001, 45.002],
            "lon": [6.0, 6.002, 6.004],
        }
    ).to_csv(ts_path, index=False)

    detail = detail_service.get_detail("ath1", activity_id)
    assert detail.map_path is not None
    assert len(detail.map_path) == 3
    assert pytest.approx(detail.map_path[0].lat, rel=1e-6) == 45.0
    assert pytest.approx(detail.map_path[-1].lon, rel=1e-6) == 6.004


def test_activity_detail_no_geo_returns_none(detail_service, storage, cfg):
    activities_repo = ActivitiesRepo(storage)
    metrics_repo = ActivitiesMetricsRepo(storage)
    raw_path = cfg.raw_strava_dir / "act-nomap.json"
    raw_path.write_text(json.dumps({"name": "No map"}), encoding="utf-8")

    activity_id = _create_activity(
        activities_repo,
        raw_path=raw_path,
        activityId="act-nomap",
        polyline="",
        hasTimeseries=False,
    )
    metrics_repo.create(
        {
            "activityId": activity_id,
            "athleteId": "ath1",
            "startDate": "2025-01-06",
            "sportType": "Run",
            "category": "EASY",
            "source": "strava",
            "distanceKm": 6.0,
            "timeSec": 2200,
            "ascentM": 80,
            "distanceEqKm": 6.1,
            "trimp": 60.0,
            "avgHr": 130,
        }
    )

    detail = detail_service.get_detail("ath1", activity_id)
    assert detail.map_path is None
    assert detail.map_notice == "Aucune donnée de trace disponible."
