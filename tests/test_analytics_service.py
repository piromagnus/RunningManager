"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import pandas as pd

from persistence.csv_storage import CsvStorage
from services.analytics_service import AnalyticsService


def test_compute_trimp_converts_seconds_to_hours(tmp_path):
    service = AnalyticsService(CsvStorage(tmp_path))
    assert service.compute_trimp(3600, 120) == 120.0
    # Zero or negative inputs should yield zero
    assert service.compute_trimp(0, 120) == 0.0
    assert service.compute_trimp(3600, 0) == 0.0


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
