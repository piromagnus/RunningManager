"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Timeseries visualization charts for activities.

Charts for HR, pace, elevation, and other timeseries data.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from services.speed_profile_service import SpeedProfileService
from services.timeseries_service import TimeseriesService

SMOOTHING_WINDOW_SECONDS = 10


def _smooth_series(series: pd.Series, window_size: int = SMOOTHING_WINDOW_SECONDS) -> pd.Series:
    return series.rolling(window=window_size, min_periods=1, center=True).mean()


def _prepare_timeseries_df(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("timestamp")
    start = df["timestamp"].iloc[0]
    df["minutes"] = (df["timestamp"] - start).dt.total_seconds() / 60.0
    return df


def _build_speed_eq_chart(
    df: pd.DataFrame,
    speed_profile_service: SpeedProfileService | None,
    window_size: int = SMOOTHING_WINDOW_SECONDS,
) -> alt.Chart | None:
    if speed_profile_service is None:
        speed_profile_service = SpeedProfileService(config=None)
    try:
        speed_eq_df = speed_profile_service.preprocess_timeseries(df)
    except Exception:
        return None
    if speed_eq_df is None or speed_eq_df.empty or "speed_km_h" not in speed_eq_df.columns:
        return None
    speed_eq_df = speed_profile_service.compute_speed_eq_column(speed_eq_df)
    if "speed_eq_km_h" not in speed_eq_df.columns:
        return None
    speed_eq_df = _prepare_timeseries_df(speed_eq_df)
    if speed_eq_df.empty:
        return None
    speed_eq_series = pd.to_numeric(speed_eq_df["speed_eq_km_h"], errors="coerce")
    speed_eq_df["speed_eq_smooth"] = _smooth_series(speed_eq_series, window_size)
    if not speed_eq_df["speed_eq_smooth"].notna().any():
        return None
    return (
        alt.Chart(speed_eq_df)
        .mark_line(color="#f97316")
        .encode(
            x=alt.X("minutes:Q", title="Temps (min)"),
            y=alt.Y("speed_eq_smooth:Q", title="Vitesse équivalente (km/h)"),
        )
        .properties(height=180)
    )


def render_timeseries_charts(
    ts_service: TimeseriesService,
    activity_id: str,
    speed_profile_service: SpeedProfileService | None = None,
) -> dict[str, alt.Chart]:
    """Render timeseries charts for an activity (HR, speed, elevation, speed-eq).

    Args:
        ts_service: TimeseriesService instance
        activity_id: Activity ID to load timeseries for

    Returns:
        dict[str, alt.Chart]: Mapping of chart keys to Altair charts
    """
    df = ts_service.load(activity_id)
    if df is None or df.empty:
        return {}

    df = _prepare_timeseries_df(df)
    if df.empty:
        return {}

    charts: dict[str, alt.Chart] = {}
    if "hr" in df.columns and df["hr"].notna().any():
        hr_series = pd.to_numeric(df["hr"], errors="coerce")
        df["hr_smooth"] = _smooth_series(hr_series)
        if df["hr_smooth"].notna().any():
            hr_max = float(df["hr_smooth"].max())
            hr_axis_max = max(60.0, hr_max + 10.0)
            charts["hr"] = (
                alt.Chart(df)
                .mark_line(color="#ef4444")
                .encode(
                    x=alt.X("minutes:Q", title="Temps (min)"),
                    y=alt.Y(
                        "hr_smooth:Q",
                        title="FC (bpm)",
                        scale=alt.Scale(domain=[60.0, hr_axis_max]),
                    ),
                )
                .properties(height=180)
            )
    if "paceKmh" in df.columns and df["paceKmh"].notna().any():
        speed_series = pd.to_numeric(df["paceKmh"], errors="coerce")
        df["speed_smooth"] = _smooth_series(speed_series)
        if df["speed_smooth"].notna().any():
            charts["speed"] = (
                alt.Chart(df)
                .mark_line(color="#3b82f6")
                .encode(
                    x=alt.X("minutes:Q", title="Temps (min)"),
                    y=alt.Y("speed_smooth:Q", title="Vitesse (km/h)"),
                )
                .properties(height=180)
            )
    if "elevationM" in df.columns and df["elevationM"].notna().any():
        elevation_series = pd.to_numeric(df["elevationM"], errors="coerce")
        df["elevation_smooth"] = _smooth_series(elevation_series)
        if df["elevation_smooth"].notna().any():
            elev_min = float(df["elevation_smooth"].min())
            elev_max = float(df["elevation_smooth"].max())
            elev_lower = max(0.0, elev_min - 50.0)
            elev_upper = elev_max + 50.0
            df["elevation_base"] = elev_lower
            charts["elevation"] = (
                alt.Chart(df)
                .mark_area(color="#10b981", opacity=0.4, line={"color": "#10b981"})
                .encode(
                    x=alt.X("minutes:Q", title="Temps (min)"),
                    y=alt.Y(
                        "elevation_smooth:Q",
                        title="Altitude (m)",
                        scale=alt.Scale(
                            domainMin=elev_lower,
                            domainMax=elev_upper,
                            nice=False,
                            zero=False,
                        ),
                    ),
                    y2=alt.Y2("elevation_base:Q"),
                )
                .properties(height=180)
            )

    speed_eq_chart = _build_speed_eq_chart(df, speed_profile_service)
    if speed_eq_chart is not None:
        charts["speed_eq"] = speed_eq_chart

    return charts

