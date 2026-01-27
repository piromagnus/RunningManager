"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from pathlib import Path

import pandas as pd
import pytest

from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo
from services.lap_metrics_service import LapMetricsService
from utils.config import Config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    timeseries_dir = tmp_path / "timeseries"
    raw_dir = tmp_path / "raw" / "strava"
    laps_dir = tmp_path / "laps"
    metrics_ts_dir = tmp_path / "metrics_ts"
    speed_profile_dir = tmp_path / "speed_profil"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    laps_dir.mkdir(parents=True, exist_ok=True)
    metrics_ts_dir.mkdir(parents=True, exist_ok=True)
    speed_profile_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        strava_client_id=None,
        strava_client_secret=None,
        strava_redirect_uri=None,
        data_dir=tmp_path,
        encryption_key=None,
        timeseries_dir=timeseries_dir,
        raw_strava_dir=raw_dir,
        laps_dir=laps_dir,
        mapbox_token=None,
        metrics_ts_dir=metrics_ts_dir,
        speed_profile_dir=speed_profile_dir,
        n_cluster=5,
    )


@pytest.fixture
def storage(config: Config) -> CsvStorage:
    return CsvStorage(base_dir=config.data_dir)


def test_compute_and_store_generates_lap_metrics(storage: CsvStorage, config: Config) -> None:
    athletes = AthletesRepo(storage)
    athletes.create(
        {
            "athleteId": "athlete-1",
            "coachId": "coach-1",
            "name": "Test",
            "hrRest": 50,
            "hrMax": 185,
        }
    )
    service = LapMetricsService(storage=storage, config=config)
    detail = {
        "id": "activity-1",
        "laps": [
            {
                "lap_index": 1,
                "split": 1,
                "name": "Run",
                "elapsed_time": 300,
                "moving_time": 290,
                "distance": 1000.0,
                "total_elevation_gain": 20.0,
                "average_heartrate": 155.0,
                "max_heartrate": 170.0,
                "average_speed": 3.6,  # m/s
            },
            {
                "lap_index": 2,
                "split": 2,
                "name": "Recovery",
                "elapsed_time": 180,
                "moving_time": 175,
                "distance": 400.0,
                "total_elevation_gain": 5.0,
                "average_heartrate": 100.0,
                "max_heartrate": 120.0,
                "average_speed": 1.4,
            },
        ],
    }

    path = service.compute_and_store("athlete-1", detail)
    assert path == config.laps_dir / "activity-1.csv"
    assert path.exists()

    loaded = service.load("activity-1")
    assert loaded is not None
    assert not loaded.empty
    assert list(loaded["lapIndex"]) == [1, 2]
    assert list(loaded["label"]) == ["Run", "Recovery"]

    assert pytest.approx(loaded.iloc[0]["distanceEqKm"], rel=1e-3) == 1.2
    assert loaded.iloc[0]["trimp"] > loaded.iloc[1]["trimp"]
    assert pytest.approx(loaded.iloc[0]["timeSec"], rel=1e-3) == 290.0
    assert pytest.approx(loaded.iloc[0]["avgSpeedKmh"], rel=1e-3) == 12.96


def test_compute_and_store_handles_missing_laps(storage: CsvStorage, config: Config) -> None:
    service = LapMetricsService(storage=storage, config=config)
    detail = {"id": "activity-empty"}
    path = service.compute_and_store("athlete-x", detail)
    assert path == config.laps_dir / "activity-empty.csv"
    df = pd.read_csv(path)
    assert df.empty
