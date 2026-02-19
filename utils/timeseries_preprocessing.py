"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from haversine import haversine


def moving_average(df: pd.DataFrame, window_size: int, col: str) -> pd.DataFrame:
    """Compute a centered moving average for a column."""
    df = df.copy()
    df[f"{col}_ma_{window_size}"] = (
        df[col].rolling(window=window_size, min_periods=1, center=True).mean()
    )
    return df


def distance(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    """Compute the distance between consecutive points in km."""
    df = df.copy()
    df["distance"] = [
        haversine((lat1, lon1), (lat2, lon2))
        for lat1, lon1, lat2, lon2 in zip(
            df[lat_col].shift(), df[lon_col].shift(), df[lat_col], df[lon_col]
        )
    ]
    return df


def cumulated_distance(df: pd.DataFrame, distance_col: str = "distance") -> pd.DataFrame:
    """Compute cumulated distance from a per-row distance column."""
    df = df.copy()
    df["cumulated_distance"] = df[distance_col].cumsum()
    return df


def time_from_timestamp(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Convert timestamp column to datetime and add a time-only column."""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["time"] = df[timestamp_col].dt.time
    return df


def duration(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Compute per-sample and cumulated duration from timestamp."""
    df = df.copy()
    df["duration"] = df[timestamp_col].diff().fillna(pd.Timedelta(seconds=0))
    df["cumulated_duration"] = df["duration"].cumsum()
    df["duration_seconds"] = df["duration"].dt.total_seconds()
    df["cumulated_duration_seconds"] = df["duration_seconds"].cumsum()
    return df


def activity_duration_seconds(df: pd.DataFrame) -> float:
    """Return activity duration in seconds, falling back to sample count."""
    if df.empty:
        return 0.0

    if "cumulated_duration_seconds" in df.columns:
        durations = pd.to_numeric(df["cumulated_duration_seconds"], errors="coerce").dropna()
        if not durations.empty:
            duration = float(durations.iloc[-1])
            if duration > 0:
                return duration

    if "duration_seconds" in df.columns:
        durations = pd.to_numeric(df["duration_seconds"], errors="coerce").dropna()
        if not durations.empty:
            duration = float(durations.sum())
            if duration > 0:
                return duration

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        if timestamps.notna().any():
            duration = float((timestamps.max() - timestamps.min()).total_seconds())
            if duration > 0:
                return duration

    return float(len(df))


def speed(
    df: pd.DataFrame, distance_col: str = "distance", time_col: str = "duration_seconds"
) -> pd.DataFrame:
    """Compute speed in m/s and km/h from distance and duration columns."""
    df = df.copy()
    mean_time = df.loc[df[time_col] > 0, time_col].mean()
    if pd.isna(mean_time) or mean_time == 0:
        mean_time = 1.0
    df[time_col] = df[time_col].replace(0, mean_time)
    df["speed_m_s"] = 1000 * df[distance_col] / df[time_col]
    df["speed_km_h"] = 3.6 * df["speed_m_s"]
    return df


def elevation(df: pd.DataFrame, elevation_col: str = "elevationM_ma_5") -> pd.DataFrame:
    """Compute elevation deltas and cumulative gain/loss."""
    df = df.copy()
    df["elevation_difference"] = df[elevation_col].diff().fillna(0)
    df["elevation_cumulated"] = df["elevation_difference"].cumsum()
    df["elevation_gain"] = df["elevation_difference"].apply(lambda x: x if x > 0 else 0).cumsum()
    df["elevation_loss"] = df["elevation_difference"].apply(lambda x: -x if x < 0 else 0).cumsum()
    return df


def grade(
    df: pd.DataFrame, distance_col: str = "distance", elevation_col: str = "elevation_difference"
) -> pd.DataFrame:
    """Compute grade from distance and elevation difference."""
    df = df.copy()
    df["grade"] = df[elevation_col] / (df[distance_col] * 1000)
    df["grade"] = df["grade"].replace([np.inf, -np.inf], 0)
    df["grade"] = df["grade"].fillna(0)
    return df
