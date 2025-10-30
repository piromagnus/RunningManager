"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Timeseries visualization charts for activities.

Charts for HR, pace, elevation, and other timeseries data.
"""

from __future__ import annotations

from typing import List

import altair as alt
import pandas as pd

from services.timeseries_service import TimeseriesService


def render_timeseries_charts(ts_service: TimeseriesService, activity_id: str) -> List[alt.Chart]:
    """Render timeseries charts for an activity (HR, pace, elevation).

    Args:
        ts_service: TimeseriesService instance
        activity_id: Activity ID to load timeseries for

    Returns:
        List[alt.Chart]: List of Altair charts to render
    """
    df = ts_service.load(activity_id)
    if df is None or df.empty:
        return []

    df = df.copy()
    if "timestamp" not in df.columns:
        return []

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return []

    df = df.sort_values("timestamp")
    start = df["timestamp"].iloc[0]
    df["minutes"] = (df["timestamp"] - start).dt.total_seconds() / 60.0

    charts: List[alt.Chart] = []
    if "hr" in df.columns and df["hr"].notna().any():
        charts.append(
            alt.Chart(df)
            .mark_line(color="#ef4444")
            .encode(x=alt.X("minutes:Q", title="Temps (min)"), y=alt.Y("hr:Q", title="FC (bpm)"))
            .properties(height=180)
        )
    if "paceKmh" in df.columns and df["paceKmh"].notna().any():
        charts.append(
            alt.Chart(df)
            .mark_line(color="#3b82f6")
            .encode(
                x=alt.X("minutes:Q", title="Temps (min)"),
                y=alt.Y("paceKmh:Q", title="Vitesse (km/h)"),
            )
            .properties(height=180)
        )
    if "elevationM" in df.columns and df["elevationM"].notna().any():
        charts.append(
            alt.Chart(df)
            .mark_area(color="#10b981", opacity=0.4)
            .encode(
                x=alt.X("minutes:Q", title="Temps (min)"),
                y=alt.Y("elevationM:Q", title="Altitude (m)"),
            )
            .properties(height=160)
        )

    return charts

