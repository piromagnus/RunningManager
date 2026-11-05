"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from persistence.csv_storage import CsvStorage
from persistence.repositories import ActivitiesRepo
from services import garmin_import_api
from services.garmin_import_api import GarminImportService
from utils.config import Config


class FakeGarmin:
    def __init__(self, activities: list[dict[str, Any]]) -> None:
        self.activities = activities
        self.login_path: str | None = None
        self.activity_requests: list[str] = []
        self.started_at = dt.datetime.now(dt.timezone.utc)

    def login(self, token_store_file: str) -> None:
        self.login_path = token_store_file

    def get_activities(self, start: int = 0, limit: int = 20, activitytype: str | None = None):
        return self.activities[start : start + limit]

    def get_activity(self, activity_id: str) -> dict[str, Any]:
        self.activity_requests.append(activity_id)
        return {
            "activityId": int(activity_id),
            "activityName": "Trail test",
            "activityTypeDTO": {"typeKey": "trail_running"},
            "startTimeLocal": self.started_at.isoformat(),
            "startTimeGMT": self.started_at.isoformat(),
            "distance": 12000,
            "duration": 5400,
            "movingDuration": 5100,
            "elevationGain": 450,
            "averageHR": 145,
            "maxHR": 178,
        }

    def get_activity_details(self, activity_id: str) -> dict[str, Any]:
        start_timestamp = int(self.started_at.timestamp() * 1000)
        return {
            "metricDescriptors": [
                {"key": "directTimestamp"},
                {"key": "directDistance"},
                {"key": "directSpeed"},
                {"key": "directHeartRate"},
                {"key": "directLatitude"},
                {"key": "directLongitude"},
                {"key": "directElevation"},
                {"key": "directRunCadence"},
            ],
            "activityDetailMetrics": [
                {
                    "metrics": [
                        start_timestamp,
                        0,
                        2.5,
                        120,
                        45.1,
                        6.1,
                        250,
                        80,
                    ]
                },
                {
                    "metrics": [
                        start_timestamp + 1000,
                        2.5,
                        3.0,
                        130,
                        -45.2,
                        -6.2,
                        251,
                        82,
                    ]
                },
            ],
        }


@pytest.fixture
def config(tmp_path: Path) -> Config:
    timeseries_dir = tmp_path / "timeseries"
    metrics_ts_dir = tmp_path / "metrics_ts"
    speed_profile_dir = tmp_path / "speed_profil"
    for path in (timeseries_dir, metrics_ts_dir, speed_profile_dir):
        path.mkdir(parents=True)
    return Config(
        strava_client_id=None,
        strava_client_secret=None,
        strava_redirect_uri=None,
        data_dir=tmp_path,
        encryption_key=None,
        timeseries_dir=timeseries_dir,
        raw_strava_dir=tmp_path / "raw" / "strava",
        laps_dir=tmp_path / "laps",
        mapbox_token=None,
        metrics_ts_dir=metrics_ts_dir,
        speed_profile_dir=speed_profile_dir,
        n_cluster=5,
        garmin_email="garmin@example.com",
        garmin_password="test-password",
        garmin_token_store=tmp_path / "garmin_tokens",
    )


def test_sync_imports_normalized_activity_and_timeseries(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    activity = FakeGarmin([])
    activity.activities.append(
        {
            "activityId": 123,
            "startTimeGMT": activity.started_at.isoformat(),
            "startTimeLocal": activity.started_at.isoformat(),
        }
    )
    recomputed: list[list[str]] = []

    class FakeMetricsService:
        def __init__(self, storage: CsvStorage, config: Config) -> None:
            pass

        def recompute_for_activities(self, activity_ids: list[str]) -> None:
            recomputed.append(activity_ids)

    monkeypatch.setattr(
        garmin_import_api,
        "MetricsComputationService",
        FakeMetricsService,
    )
    service = GarminImportService(
        storage=CsvStorage(base_dir=tmp_path),
        config=config,
        client_factory=lambda **kwargs: activity,
    )

    imported = service.sync_last_n_days("athlete-1", 14)

    assert imported == ["123"]
    assert activity.login_path == str(config.garmin_token_store)
    assert recomputed == [["123"]]
    saved_activity = ActivitiesRepo(CsvStorage(base_dir=tmp_path)).get("123")
    assert saved_activity is not None
    assert saved_activity["sportType"] == "TRAIL_RUN"
    assert saved_activity["source"] == "garmin"
    assert float(saved_activity["distanceKm"]) == 12
    assert saved_activity["hasTimeseries"]

    timeseries = pd.read_csv(config.timeseries_dir / "123.csv")
    assert list(timeseries.columns) == [
        "timestamp",
        "hr",
        "paceKmh",
        "elevationM",
        "cadence",
        "lat",
        "lon",
    ]
    assert timeseries["paceKmh"].tolist() == [9.0, 10.8]
    assert timeseries.loc[1, "lat"] == -45.2
    assert timeseries.loc[1, "lon"] == -6.2
    raw_path = config.data_dir / str(saved_activity["rawJsonPath"])
    raw_activity = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw_activity["garminActivityDetails"]["metricDescriptors"][0]["key"] == "directTimestamp"


def test_sync_skips_existing_and_out_of_range_activities(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    client = FakeGarmin(
        [
            {"activityId": 123, "startTimeGMT": now.isoformat()},
            {
                "activityId": 124,
                "startTimeGMT": (now - dt.timedelta(days=30)).isoformat(),
            },
        ]
    )
    storage = CsvStorage(base_dir=tmp_path)
    ActivitiesRepo(storage).create({"activityId": "123", "athleteId": "athlete-1"})

    class FakeMetricsService:
        def __init__(self, storage: CsvStorage, config: Config) -> None:
            pass

        def recompute_for_activities(self, activity_ids: list[str]) -> None:
            raise AssertionError("metrics should not be recomputed for an empty import")

    monkeypatch.setattr(garmin_import_api, "MetricsComputationService", FakeMetricsService)
    service = GarminImportService(
        storage=storage,
        config=config,
        client_factory=lambda **kwargs: client,
    )

    assert service.sync_last_n_days("athlete-1", 14) == []
    assert client.activity_requests == []


def test_sync_requires_garmin_credentials(tmp_path: Path, config: Config) -> None:
    service = GarminImportService(
        storage=CsvStorage(base_dir=tmp_path),
        config=replace(config, garmin_email=None, garmin_password=None),
    )

    with pytest.raises(RuntimeError, match="GARMIN_EMAIL"):
        service.sync_last_n_days("athlete-1", 14)
