"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Tests for pacer service.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from persistence.csv_storage import CsvStorage
from services.pacer_service import PacerService


@pytest.fixture
def storage(tmp_path: Path) -> CsvStorage:
    """Create a temporary CSV storage."""
    return CsvStorage(base_dir=tmp_path)


@pytest.fixture
def pacer_service(storage: CsvStorage) -> PacerService:
    """Create a pacer service instance."""
    return PacerService(storage)


def test_classify_grade_boundaries(pacer_service: PacerService):
    """Test grade classification boundaries."""
    assert pacer_service.classify_grade(0.15) == "steep_up"  # >= 0.10
    assert pacer_service.classify_grade(0.10) == "steep_up"
    assert pacer_service.classify_grade(0.05) == "run_up"  # 0.02 <= grade < 0.10
    assert pacer_service.classify_grade(0.02) == "run_up"
    assert pacer_service.classify_grade(0.01) == "flat"  # -0.02 < grade < 0.02
    assert pacer_service.classify_grade(0.0) == "flat"
    assert pacer_service.classify_grade(-0.01) == "flat"
    assert pacer_service.classify_grade(-0.02) == "down"  # -0.25 < grade <= -0.02
    assert pacer_service.classify_grade(-0.10) == "down"
    assert pacer_service.classify_grade(-0.25) == "steep_down"  # <= -0.25 (inclusive)
    assert pacer_service.classify_grade(-0.30) == "steep_down"  # <= -0.25


def test_classify_grade_flat_with_elevation_delta(pacer_service: PacerService):
    """Test flat classification with elevation delta per km."""
    # Should be flat if elevation delta < 10m/1km even with slight grade
    assert pacer_service.classify_grade(0.03, cumulated_elevation_delta_per_km=5.0) == "flat"
    assert pacer_service.classify_grade(-0.03, cumulated_elevation_delta_per_km=8.0) == "flat"
    # Should not be flat if elevation delta >= 10m/1km
    assert pacer_service.classify_grade(0.03, cumulated_elevation_delta_per_km=15.0) == "run_up"


def test_compute_segment_time(pacer_service: PacerService):
    """Test segment time computation."""
    # speedEqKmh takes precedence
    time = pacer_service.compute_segment_time(distance_eq_km=5.0, distance_km=4.0, speed_eq_kmh=10.0, speed_kmh=12.0)
    assert time == int(round(3600 * 5.0 / 10.0))  # 1800 seconds
    
    # speedKmh used if speedEqKmh is 0
    time = pacer_service.compute_segment_time(distance_eq_km=5.0, distance_km=4.0, speed_eq_kmh=0.0, speed_kmh=12.0)
    assert time == int(round(3600 * 4.0 / 12.0))  # 1200 seconds
    
    # Returns 0 if both speeds are 0
    time = pacer_service.compute_segment_time(distance_eq_km=5.0, distance_km=4.0, speed_eq_kmh=0.0, speed_kmh=0.0)
    assert time == 0


def test_aggregate_summary(pacer_service: PacerService):
    """Test aggregate summary computation."""
    segments_df = pd.DataFrame([
        {
            "segmentId": 0,
            "distanceKm": 3.0,
            "distanceEqKm": 3.5,
            "elevGainM": 50.0,
            "elevLossM": 10.0,
            "timeSec": 1200,
        },
        {
            "segmentId": 1,
            "distanceKm": 2.0,
            "distanceEqKm": 2.2,
            "elevGainM": 30.0,
            "elevLossM": 5.0,
            "timeSec": 800,
        },
    ])
    
    summary = pacer_service.aggregate_summary(segments_df)
    assert summary["distanceKm"] == 5.0
    assert summary["distanceEqKm"] == 5.7
    assert summary["elevGainM"] == 80.0
    assert summary["elevLossM"] == 15.0
    assert summary["timeSec"] == 2000


def test_aggregate_summary_empty(pacer_service: PacerService):
    """Test aggregate summary with empty DataFrame."""
    segments_df = pd.DataFrame()
    summary = pacer_service.aggregate_summary(segments_df)
    assert summary["distanceKm"] == 0.0
    assert summary["distanceEqKm"] == 0.0
    assert summary["elevGainM"] == 0.0
    assert summary["elevLossM"] == 0.0
    assert summary["timeSec"] == 0


def test_preprocess_timeseries_for_pacing(pacer_service: PacerService):
    """Test preprocessing timeseries for pacing."""
    # Create synthetic timeseries
    timeseries_df = pd.DataFrame({
        "lat": [45.0 + i * 0.001 for i in range(200)],
        "lon": [5.0 + i * 0.001 for i in range(200)],
        "elevationM": [100 + i * 0.1 for i in range(200)],
    })
    
    result = pacer_service.preprocess_timeseries_for_pacing(timeseries_df)
    assert not result.empty
    assert "cumulated_distance" in result.columns
    assert "elevationM_ma_5" in result.columns
    assert "grade_ma_10" in result.columns


def test_preprocess_timeseries_empty(pacer_service: PacerService):
    """Test preprocessing with empty DataFrame."""
    timeseries_df = pd.DataFrame()
    result = pacer_service.preprocess_timeseries_for_pacing(timeseries_df)
    assert result.empty


def test_save_and_load_race(pacer_service: PacerService):
    """Test saving and loading race data."""
    race_name = "Test Race"
    aid_stations_km = [3.0, 8.5, 15.2]
    segments_df = pd.DataFrame([
        {
            "segmentId": 0,
            "type": "flat",
            "startKm": 0.0,
            "endKm": 3.0,
            "distanceKm": 3.0,
            "elevGainM": 50.0,
            "elevLossM": 0.0,
            "avgGrade": 0.016,
            "isAidSplit": True,
            "distanceEqKm": 3.5,
            "speedEqKmh": 10.0,
            "speedKmh": 0.0,
            "timeSec": 1260,
        },
    ])
    
    # Save race
    race_id = pacer_service.save_race(race_name, aid_stations_km, segments_df)
    assert race_id is not None
    
    # Load race
    loaded = pacer_service.load_race(race_id)
    assert loaded is not None
    loaded_name, loaded_aid, loaded_segments = loaded
    assert loaded_name == race_name
    assert loaded_aid == aid_stations_km
    assert len(loaded_segments) == len(segments_df)
    assert loaded_segments.iloc[0]["segmentId"] == 0


def test_list_races(pacer_service: PacerService):
    """Test listing races."""
    # Initially empty
    races = pacer_service.list_races()
    assert races.empty
    
    # Save a race
    segments_df = pd.DataFrame([{
        "segmentId": 0,
        "type": "flat",
        "startKm": 0.0,
        "endKm": 3.0,
        "distanceKm": 3.0,
        "elevGainM": 50.0,
        "elevLossM": 0.0,
        "avgGrade": 0.016,
        "isAidSplit": True,
        "distanceEqKm": 3.5,
        "speedEqKmh": 10.0,
        "speedKmh": 0.0,
        "timeSec": 1260,
    }])
    
    race_id1 = pacer_service.save_race("Race 1", [3.0], segments_df)
    race_id2 = pacer_service.save_race("Race 2", [5.0], segments_df)
    
    # List races
    races = pacer_service.list_races()
    assert len(races) == 2
    assert "raceId" in races.columns
    assert "name" in races.columns
    assert "createdAt" in races.columns
    assert race_id1 in races["raceId"].values
    assert race_id2 in races["raceId"].values

