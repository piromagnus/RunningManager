"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import json
from pathlib import Path

import pandas as pd

from persistence.csv_storage import CsvStorage
from persistence.repositories import ActivitiesRepo
from services.metrics_service import MetricsComputationService


def test_activity_sport_type_inferred_from_raw(tmp_path: Path) -> None:
    storage = CsvStorage(base_dir=tmp_path)

    # Prepare raw JSON with sport_type set
    raw_dir = tmp_path / "raw" / "strava"
    raw_dir.mkdir(parents=True, exist_ok=True)
    act_id = "15822816751"
    raw_path = raw_dir / f"{act_id}.json"
    raw_payload = {
        "id": int(act_id),
        "sport_type": "TrailRun",
        "type": "Run",
        "start_date_local": "2025-09-15T19:46:59",
        "distance": 9463.5,
        "elapsed_time": 4744,
        "moving_time": 4617,
        "total_elevation_gain": 414.0,
        "average_heartrate": 132.6,
        "max_heartrate": 156.0,
    }
    raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")

    # Create an activities.csv row without sportType (should fallback to raw JSON)
    activities = ActivitiesRepo(storage)
    activities.create(
        {
            "activityId": act_id,
            "athleteId": "ath-1",
            "source": "strava",
            # Do not set sportType here to exercise fallback
            "startTime": "2025-09-15T19:46:59",
            "distanceKm": 9.4635,
            "elapsedSec": 4744,
            "movingSec": 4617,
            "ascentM": 414.0,
            "avgHr": 132.6,
            "maxHr": 156.0,
            "hasTimeseries": True,
            "polyline": "",
            "rawJsonPath": f"raw/strava/{act_id}.json",
        }
    )

    # Compute metrics
    MetricsComputationService(storage).recompute_athlete("ath-1")

    # Validate metrics row
    metrics_path = tmp_path / "activities_metrics.csv"
    assert metrics_path.exists()
    df = pd.read_csv(metrics_path)
    row = df[df["activityId"].astype(str) == act_id].iloc[0]
    assert row["sportType"] == "TrailRun"
    assert row["category"] == "TRAIL_RUN"


def test_activity_sport_type_backcountry_ski_category(tmp_path: Path) -> None:
    storage = CsvStorage(base_dir=tmp_path)
    activities = ActivitiesRepo(storage)
    activities.create(
        {
            "activityId": "ski-1",
            "athleteId": "ath-1",
            "source": "manual",
            "sportType": "BackcountrySki",
            "startTime": "2025-01-10T08:00:00Z",
            "distanceKm": 15.0,
            "elapsedSec": 5400,
            "movingSec": 5200,
            "ascentM": 300.0,
            "avgHr": 140.0,
            "maxHr": 165.0,
            "hasTimeseries": False,
            "polyline": "",
            "rawJsonPath": "",
        }
    )

    MetricsComputationService(storage).recompute_athlete("ath-1")

    df = pd.read_csv(tmp_path / "activities_metrics.csv")
    row = df[df["activityId"].astype(str) == "ski-1"].iloc[0]
    assert row["category"] == "BACKCOUNTRY_SKI"


def test_activity_sport_type_virtual_ride_category(tmp_path: Path) -> None:
    storage = CsvStorage(base_dir=tmp_path)
    activities = ActivitiesRepo(storage)
    activities.create(
        {
            "activityId": "ride-1",
            "athleteId": "ath-1",
            "source": "manual",
            "sportType": "VirtualRide",
            "startTime": "2025-01-11T08:00:00Z",
            "distanceKm": 25.0,
            "elapsedSec": 3600,
            "movingSec": 3500,
            "ascentM": 200.0,
            "avgHr": 130.0,
            "maxHr": 160.0,
            "hasTimeseries": False,
            "polyline": "",
            "rawJsonPath": "",
        }
    )

    MetricsComputationService(storage).recompute_athlete("ath-1")

    df = pd.read_csv(tmp_path / "activities_metrics.csv")
    row = df[df["activityId"].astype(str) == "ride-1"].iloc[0]
    assert row["category"] == "RIDE"
