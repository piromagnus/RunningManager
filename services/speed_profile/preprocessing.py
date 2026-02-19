"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import pandas as pd

from utils.series_filters import filter_series_outliers
from utils import timeseries_preprocessing as ts_pre


def moving_average(df: pd.DataFrame, window_size: int, col: str) -> pd.DataFrame:
    """Compute the moving average of a column over a specified window size."""
    return ts_pre.moving_average(df, window_size, col)


def distance(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    """Compute the distance between 2 consecutive rows based on latitude and longitude."""
    return ts_pre.distance(df, lat_col=lat_col, lon_col=lon_col)


def cumulated_distance(df: pd.DataFrame, distance_col: str = "distance") -> pd.DataFrame:
    """Compute the cumulated distance based on the distance column."""
    return ts_pre.cumulated_distance(df, distance_col=distance_col)


def time_from_timestamp(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Convert the timestamp column to a datetime object."""
    return ts_pre.time_from_timestamp(df, timestamp_col=timestamp_col)


def duration(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Compute the duration of the dataframe based on the timestamp column."""
    return ts_pre.duration(df, timestamp_col=timestamp_col)


def activity_duration_seconds(df: pd.DataFrame) -> float:
    """Return activity duration in seconds, falling back to sample count."""
    return ts_pre.activity_duration_seconds(df)


def speed(
    df: pd.DataFrame, distance_col: str = "distance", time_col: str = "duration_seconds"
) -> pd.DataFrame:
    """Compute the speed of the dataframe based on the distance and time columns."""
    return ts_pre.speed(df, distance_col=distance_col, time_col=time_col)


def elevation(df: pd.DataFrame, elevation_col: str = "elevationM_ma_5") -> pd.DataFrame:
    """Compute the elevation difference of the dataframe based on the elevation column."""
    return ts_pre.elevation(df, elevation_col=elevation_col)


def grade(
    df: pd.DataFrame, distance_col: str = "distance", elevation_col: str = "elevation_difference"
) -> pd.DataFrame:
    """Compute the grade of the dataframe based on the distance and elevation columns."""
    return ts_pre.grade(df, distance_col=distance_col, elevation_col=elevation_col)


def preprocess_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess activity timeseries (time-referenced pipeline)."""
    df = df.copy()

    if "lat" not in df.columns or "lon" not in df.columns:
        return pd.DataFrame()

    if df["lat"].isna().all() or df["lon"].isna().all():
        return pd.DataFrame()

    reference_col = "timestamp" if "timestamp" in df.columns else None
    df = filter_series_outliers(
        df,
        value_col="elevationM",
        reference_col=reference_col,
        window=7.0,
        sigma=3.0,
    )

    df = moving_average(df, window_size=5, col="lat")
    df = moving_average(df, window_size=5, col="lon")
    df = moving_average(df, window_size=5, col="elevationM")
    df = distance(df, lat_col="lat_ma_5", lon_col="lon_ma_5")

    df = df[df["distance"] > 1e-5].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    df["distance"] = df["distance"].interpolate(method="linear")

    df = elevation(df, elevation_col="elevationM_ma_5")

    df = time_from_timestamp(df)
    df = duration(df)

    df = cumulated_distance(df)

    df = moving_average(df, window_size=10, col="distance")

    df = speed(df, distance_col="distance", time_col="duration_seconds")
    df = df[df["speed_km_h"] < 40].reset_index(drop=True)

    df = grade(df, distance_col="distance_ma_10", elevation_col="elevation_difference")

    return df
