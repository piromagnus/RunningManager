"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import datetime as dt

import pandas as pd
import pytest

from persistence.csv_storage import CsvStorage
from services.analytics_service import AnalyticsService
from utils.metrics_formulas import compute_trimp_hr_reserve_from_profile


def test_compute_trimp_uses_hr_reserve(tmp_path):
    service = AnalyticsService(CsvStorage(tmp_path))
    hr_profile = (60.0, 190.0)
    expected = compute_trimp_hr_reserve_from_profile(150.0, 3600.0, hr_profile)
    assert service.compute_trimp(150.0, 3600.0, hr_profile) == expected
    # Missing or invalid inputs should yield zero
    assert service.compute_trimp(0.0, 3600.0, hr_profile) == 0.0
    assert service.compute_trimp(150.0, 0.0, hr_profile) == 0.0
    assert service.compute_trimp(150.0, 3600.0, None) == 0.0


def test_build_planned_vs_actual_segments_handles_above_and_below(tmp_path):
    service = AnalyticsService(CsvStorage(tmp_path))
    df = pd.DataFrame(
        [
            {
                "athleteId": "ath1",
                "isoYear": 2025,
                "isoWeek": 39,
                "weekLabel": "2025-W39",
                "plannedValue": 10.0,
                "actualValue": 12.0,
            },
            {
                "athleteId": "ath1",
                "isoYear": 2025,
                "isoWeek": 40,
                "weekLabel": "2025-W40",
                "plannedValue": 12.0,
                "actualValue": 8.0,
            },
        ]
    )
    segments = service.build_planned_vs_actual_segments(
        df,
        planned_column="plannedValue",
        actual_column="actualValue",
        metric_key="distance",
    )
    # Week 39: planned 10, actual 12 -> base 10, extra 2
    week39 = segments[segments["weekLabel"] == "2025-W39"]
    assert len(week39) == 2
    realised = week39[week39["segment"] == "Réalisé"].iloc[0]
    above = week39[week39["segment"] == "Au-dessus du plan"].iloc[0]
    assert realised["value"] == 10.0
    assert above["value"] == 2.0
    assert realised["maxValue"] == 12.0

    # Week 40: planned 12, actual 8 -> base 8, shortfall 4
    week40 = segments[segments["weekLabel"] == "2025-W40"]
    assert len(week40) == 2
    realised40 = week40[week40["segment"] == "Réalisé"].iloc[0]
    below40 = week40[week40["segment"] == "Plan manquant"].iloc[0]
    assert realised40["value"] == 8.0
    assert below40["value"] == 4.0
    assert realised40["maxValue"] == 12.0


def test_activity_category_breakdown_sums_and_pct(tmp_path):
    storage = CsvStorage(tmp_path)
    metrics_path = storage.base_dir / "activities_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "activityId": "run-1",
                "athleteId": "ath1",
                "startDate": "2025-05-01",
                "category": "RUN",
                "distanceKm": 10.0,
                "distanceEqKm": 10.0,
                "timeSec": 3600.0,
                "trimp": 50.0,
            },
            {
                "activityId": "trail-1",
                "athleteId": "ath1",
                "startDate": "2025-05-02",
                "category": "TRAIL_RUN",
                "distanceKm": 5.0,
                "distanceEqKm": 6.0,
                "timeSec": 1800.0,
                "trimp": 30.0,
            },
            {
                "activityId": "other-ath",
                "athleteId": "ath2",
                "startDate": "2025-05-03",
                "category": "RUN",
                "distanceKm": 99.0,
                "distanceEqKm": 99.0,
                "timeSec": 9999.0,
                "trimp": 999.0,
            },
        ]
    ).to_csv(metrics_path, index=False)

    service = AnalyticsService(storage)
    breakdown = service.activity_category_breakdown(
        athlete_id="ath1",
        metric_label="Distance",
        selected_types=["RUN", "TRAIL_RUN"],
        start_date=dt.date(2025, 5, 1),
        end_date=dt.date(2025, 5, 31),
    )
    assert set(breakdown["category"]) == {"RUN", "TRAIL_RUN"}
    assert breakdown["value"].sum() == 15.0
    assert breakdown["pct"].sum() == pytest.approx(100.0)
    run_row = breakdown[breakdown["category"] == "RUN"].iloc[0]
    assert run_row["activity_count"] == 1
    assert run_row["pct"] == 10.0 / 15.0 * 100.0


def test_activity_category_weekly_breakdown_groups_by_week(tmp_path):
    storage = CsvStorage(tmp_path)
    metrics_path = storage.base_dir / "activities_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "activityId": "run-1",
                "athleteId": "ath1",
                "startDate": "2025-05-05",
                "category": "RUN",
                "distanceKm": 10.0,
                "distanceEqKm": 10.0,
                "timeSec": 3600.0,
                "trimp": 50.0,
            },
            {
                "activityId": "trail-1",
                "athleteId": "ath1",
                "startDate": "2025-05-12",
                "category": "TRAIL_RUN",
                "distanceKm": 5.0,
                "distanceEqKm": 6.0,
                "timeSec": 1800.0,
                "trimp": 30.0,
            },
        ]
    ).to_csv(metrics_path, index=False)

    service = AnalyticsService(storage)
    weekly = service.activity_category_weekly_breakdown(
        athlete_id="ath1",
        metric_label="Distance",
        selected_types=["RUN", "TRAIL_RUN"],
        start_date=dt.date(2025, 5, 1),
        end_date=dt.date(2025, 5, 31),
    )
    assert len(weekly) == 2
    assert set(weekly["category"]) == {"RUN", "TRAIL_RUN"}
    assert weekly["value"].sum() == 15.0
