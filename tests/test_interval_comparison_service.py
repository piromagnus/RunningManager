"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from pathlib import Path

import pandas as pd
import pytest

from persistence.csv_storage import CsvStorage
from services.interval_comparison_service import IntervalComparisonService
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
        hr_zone_count=5,
        hr_zone_window_days=90,
        hr_zone_fit_activity_types=("RUN", "TRAIL_RUN"),
    )


@pytest.fixture
def storage(config: Config) -> CsvStorage:
    return CsvStorage(base_dir=config.data_dir)


@pytest.fixture
def service(storage: CsvStorage, config: Config) -> IntervalComparisonService:
    return IntervalComparisonService(storage=storage, config=config)


def test_flatten_planned_segments_expands_loops_and_between(service: IntervalComparisonService) -> None:
    steps = {
        "preBlocks": [{"kind": "run", "sec": 600}],
        "loops": [
            {
                "repeats": 2,
                "actions": [
                    {"kind": "run", "sec": 30, "targetType": "hr", "targetLabel": "Threshold 30"},
                    {"kind": "recovery", "sec": 30},
                ],
            },
            {
                "repeats": 1,
                "actions": [
                    {"kind": "run", "sec": 40, "targetType": "hr", "targetLabel": "Threshold 30"},
                    {"kind": "recovery", "sec": 20},
                ],
            },
        ],
        "betweenBlock": {"kind": "recovery", "sec": 120},
        "postBlocks": [{"kind": "recovery", "sec": 300}],
    }
    flattened = service.flatten_planned_segments(steps)
    assert [segment.sec for segment in flattened] == [600, 30, 30, 30, 30, 120, 40, 20, 300]
    assert [segment.kind for segment in flattened] == [
        "run",
        "run",
        "recovery",
        "run",
        "recovery",
        "recovery",
        "run",
        "recovery",
        "recovery",
    ]
    assert flattened[1].section_label.startswith("Boucle 1 R1/2")
    assert flattened[5].section_label == "Entre blocs 1"


def test_flatten_post_blocks_are_used_between_loops_when_between_missing(
    service: IntervalComparisonService,
) -> None:
    steps = {
        "preBlocks": [{"kind": "run", "sec": 600}],
        "loops": [
            {
                "repeats": 1,
                "actions": [
                    {"kind": "run", "sec": 30},
                    {"kind": "recovery", "sec": 30},
                ],
            },
            {
                "repeats": 1,
                "actions": [
                    {"kind": "run", "sec": 20},
                    {"kind": "recovery", "sec": 20},
                ],
            },
        ],
        "postBlocks": [{"kind": "recovery", "sec": 90}],
    }
    flattened = service.flatten_planned_segments(steps)
    assert [segment.sec for segment in flattened] == [600, 30, 30, 90, 20, 20]
    labels = [segment.section_label for segment in flattened]
    assert any(label.startswith("Entre blocs 1.") for label in labels)
    assert not any(label.startswith("Apres") for label in labels)


def test_compute_lap_descent_from_timeseries(service: IntervalComparisonService, config: Config) -> None:
    activity_id = "act-descent"
    pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T10:00:00Z",
                "2026-01-01T10:00:01Z",
                "2026-01-01T10:00:02Z",
                "2026-01-01T10:00:03Z",
                "2026-01-01T10:00:04Z",
                "2026-01-01T10:00:05Z",
                "2026-01-01T10:00:06Z",
                "2026-01-01T10:00:07Z",
                "2026-01-01T10:00:08Z",
                "2026-01-01T10:00:09Z",
                "2026-01-01T10:00:10Z",
            ],
            "elevationM": [100, 102, 101, 103, 100, 99, 98, 100, 97, 97, 96],
        }
    ).to_csv(config.timeseries_dir / f"{activity_id}.csv", index=False)

    laps_df = pd.DataFrame(
        {
            "lapIndex": [1, 2],
            "startTime": ["2026-01-01T10:00:00Z", "2026-01-01T10:00:05Z"],
            "timeSec": [5, 5],
        }
    )
    descents = service.compute_lap_descent(activity_id, laps_df)
    assert descents == [5.0, 5.0]


def test_compare_moves_unmatched_laps_to_end_and_fuses(service: IntervalComparisonService, config: Config) -> None:
    activity_id = "act-interval"
    pd.DataFrame(
        {
            "lapIndex": [1, 2, 3, 4, 5, 6, 7],
            "label": ["Run", "Run", "Recovery", "Run", "Recovery", "Recovery", "Recovery"],
            "startTime": [
                "2026-01-01T10:00:00Z",
                "2026-01-01T10:03:20Z",
                "2026-01-01T10:03:52Z",
                "2026-01-01T10:04:20Z",
                "2026-01-01T10:04:51Z",
                "2026-01-01T10:05:20Z",
                "2026-01-01T10:07:00Z",
            ],
            "timeSec": [200, 32, 28, 31, 29, 100, 80],
            "distanceKm": [0.45, 0.16, 0.07, 0.15, 0.07, 0.20, 0.14],
            "avgSpeedKmh": [8.1, 18.0, 9.0, 17.4, 8.7, 7.2, 6.6],
            "distanceEqKm": [0.45, 0.16, 0.07, 0.15, 0.07, 0.20, 0.14],
            "avgHr": [132, 172, 142, 170, 140, 128, 124],
            "ascentM": [2, 0, 0, 0, 0, 0, 0],
        }
    ).to_csv(config.laps_dir / f"{activity_id}.csv", index=False)

    planned_session = {
        "stepsJson": {
            "preBlocks": [],
            "loops": [
                {
                    "repeats": 2,
                    "actions": [
                        {"kind": "run", "sec": 30},
                        {"kind": "recovery", "sec": 30},
                    ],
                }
            ],
            "postBlocks": [],
        }
    }

    matches = service.compare(activity_id, planned_session)
    assert len(matches) == 5

    assert [entry.match_status for entry in matches[:4]] == ["matched", "matched", "matched", "matched"]
    assert [entry.planned.index for entry in matches[:4] if entry.planned is not None] == [1, 2, 3, 4]

    assert matches[4].match_status == "actual_only"
    assert [lap.lap_index for lap in matches[4].laps] == [1, 6, 7]


def test_alternating_30_30_interval_sequential_assignment(
    service: IntervalComparisonService, config: Config
) -> None:
    activity_id = "act-alt-3030"
    pd.DataFrame(
        {
            "lapIndex": list(range(1, 14)),
            # Real sessions can keep all laps as "Run" because HR remains high.
            "label": ["Run"] * 13,
            "startTime": [
                "2026-01-01T10:00:00Z",
                "2026-01-01T10:05:40Z",
                "2026-01-01T10:10:40Z",
                "2026-01-01T10:11:10Z",
                "2026-01-01T10:11:40Z",
                "2026-01-01T10:12:10Z",
                "2026-01-01T10:12:40Z",
                "2026-01-01T10:14:40Z",
                "2026-01-01T10:15:10Z",
                "2026-01-01T10:15:40Z",
                "2026-01-01T10:16:10Z",
                "2026-01-01T10:16:40Z",
                "2026-01-01T10:24:40Z",
            ],
            "timeSec": [340, 300, 30, 30, 30, 30, 120, 30, 30, 30, 30, 480, 130],
            "distanceKm": [1.0, 0.9, 0.16, 0.08, 0.16, 0.08, 0.24, 0.16, 0.08, 0.16, 0.08, 1.0, 0.3],
            "avgSpeedKmh": [10.6, 10.8, 19.2, 9.6, 19.2, 9.6, 7.2, 19.2, 9.6, 19.2, 9.6, 7.5, 8.3],
            "distanceEqKm": [1.0, 0.9, 0.16, 0.08, 0.16, 0.08, 0.24, 0.16, 0.08, 0.16, 0.08, 1.0, 0.3],
            "avgHr": [140, 146, 174, 176, 178, 180, 165, 176, 178, 180, 182, 150, 148],
            "ascentM": [0] * 13,
        }
    ).to_csv(config.laps_dir / f"{activity_id}.csv", index=False)

    planned_session = {
        "stepsJson": {
            "preBlocks": [{"kind": "run", "sec": 600}],
            "loops": [
                {
                    "repeats": 2,
                    "actions": [
                        {"kind": "run", "sec": 30},
                        {"kind": "recovery", "sec": 30},
                    ],
                },
                {
                    "repeats": 2,
                    "actions": [
                        {"kind": "run", "sec": 30},
                        {"kind": "recovery", "sec": 30},
                    ],
                },
            ],
            "betweenBlock": {"kind": "recovery", "sec": 180},
            "postBlocks": [{"kind": "recovery", "sec": 300}],
        }
    }

    matches = service.compare(activity_id, planned_session)
    matched_by_planned_index = {
        entry.planned.index: [lap.lap_index for lap in entry.laps]
        for entry in matches
        if entry.match_status == "matched" and entry.planned is not None
    }

    # Loop 1 run/recovery alternation must be preserved by order.
    assert matched_by_planned_index[2] == [3]
    assert matched_by_planned_index[3] == [4]
    assert matched_by_planned_index[4] == [5]
    assert matched_by_planned_index[5] == [6]

    # Inter-loop recovery must still be aligned.
    assert matched_by_planned_index[6] == [7]

    # Loop 2 run/recovery alternation must be preserved as well.
    assert matched_by_planned_index[7] == [8]
    assert matched_by_planned_index[8] == [9]
    assert matched_by_planned_index[9] == [10]
    assert matched_by_planned_index[10] == [11]

    # Cooldown absorbs trailing extra laps once session segments are done.
    assert matched_by_planned_index[11] == [12, 13]
    actual_only = [entry for entry in matches if entry.match_status == "actual_only"]
    assert len(actual_only) == 0
