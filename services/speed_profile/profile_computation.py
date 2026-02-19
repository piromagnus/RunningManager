"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from services.speed_profile import hr_speed_analysis, minetti, preprocessing


def compute_speed_eq_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add speed_eq_km_h column using Minetti energy cost model."""
    df = df.copy()

    energy_cost_walking_flat = 2.5
    energy_cost_running_flat = 3.6
    walking_threshold = 6.0

    if "grade_ma_10" not in df.columns:
        if "grade" in df.columns:
            df = preprocessing.moving_average(df, window_size=10, col="grade")
        else:
            df["grade_ma_10"] = 0.0

    def calculate_speed_eq(row: pd.Series) -> float:
        speed = row.get("speed_km_h", np.nan)
        grade = row.get("grade_ma_10", 0.0)

        if pd.isna(speed) or speed <= 0:
            return np.nan

        if pd.isna(grade):
            grade = 0.0

        if speed <= walking_threshold:
            cost_at_grade = minetti.minetti_energy_cost_walking(grade)
            cost_flat = energy_cost_walking_flat
        else:
            cost_at_grade = minetti.minetti_energy_cost_running(grade)
            cost_flat = energy_cost_running_flat

        if cost_flat <= 0:
            return speed

        return max(0.0, speed * (cost_at_grade / cost_flat))

    df["speed_eq_km_h"] = df.apply(calculate_speed_eq, axis=1)
    return df


def compute_max_speed_profiles(df: pd.DataFrame, window_sizes: List[int]) -> pd.DataFrame:
    """Compute maximum rolling average speed profiles for raw and equivalent speed."""
    df = df.copy()

    if "speed_km_h_ma_10" not in df.columns:
        df = preprocessing.moving_average(df, window_size=10, col="speed_km_h")
        df = df.rename(columns={"speed_km_h_ma_10": "speed_km_h_smooth"})
    else:
        df = df.rename(columns={"speed_km_h_ma_10": "speed_km_h_smooth"})

    if "grade_ma_10" not in df.columns:
        if "grade" in df.columns:
            df = preprocessing.moving_average(df, window_size=10, col="grade")
        else:
            df["grade_ma_10"] = 0.0

    df = compute_speed_eq_column(df)

    results = []
    activity_duration_sec = preprocessing.activity_duration_seconds(df)
    for window_sec in window_sizes:
        if window_sec > activity_duration_sec:
            max_speed = np.nan
            max_speed_eq = np.nan
        else:
            rolling_speed = df["speed_km_h_smooth"].rolling(
                window=window_sec, min_periods=1, center=True
            ).mean()
            max_speed = rolling_speed.max()

            rolling_speed_eq = df["speed_eq_km_h"].rolling(
                window=window_sec, min_periods=1, center=True
            ).mean()
            max_speed_eq = rolling_speed_eq.max()

        results.append(
            {
                "windowSec": window_sec,
                "maxSpeedKmh": None if pd.isna(max_speed) else float(max_speed),
                "maxSpeedEqKmh": None if pd.isna(max_speed_eq) else float(max_speed_eq),
            }
        )

    return pd.DataFrame(results)


def compute_speed_profile_cloud(df: pd.DataFrame, window_sizes: List[int]) -> pd.DataFrame:
    """Compute per-window max speeds with associated HR values."""
    if df.empty or "speed_km_h" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    if "hr" not in df.columns:
        df["hr"] = np.nan

    df["speed_km_h"] = pd.to_numeric(df["speed_km_h"], errors="coerce")
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce")
    if df["speed_km_h"].isna().all():
        return pd.DataFrame()

    if "speed_km_h_ma_10" not in df.columns:
        df = preprocessing.moving_average(df, window_size=10, col="speed_km_h")
        df = df.rename(columns={"speed_km_h_ma_10": "speed_km_h_smooth"})
    else:
        df = df.rename(columns={"speed_km_h_ma_10": "speed_km_h_smooth"})

    if "hr_ma_10" not in df.columns:
        df = preprocessing.moving_average(df, window_size=10, col="hr")
        df = df.rename(columns={"hr_ma_10": "hr_smooth"})
    else:
        df = df.rename(columns={"hr_ma_10": "hr_smooth"})

    if "grade_ma_10" not in df.columns:
        if "grade" in df.columns:
            df = preprocessing.moving_average(df, window_size=10, col="grade")
        else:
            df["grade_ma_10"] = 0.0

    df = compute_speed_eq_column(df)

    max_speeds, hr_at_max_speeds = hr_speed_analysis.compute_profile(
        df, "speed_km_h_smooth", "hr_smooth", window_sizes, 0
    )
    max_speed_eqs, hr_at_max_speed_eqs = hr_speed_analysis.compute_profile(
        df, "speed_eq_km_h", "hr_smooth", window_sizes, 0
    )

    rows = []
    for window_sec in window_sizes:
        max_speed = max_speeds.get(window_sec)
        max_speed_eq = max_speed_eqs.get(window_sec)
        hr_at_max_speed = hr_at_max_speeds.get(window_sec)
        hr_at_max_speed_eq = hr_at_max_speed_eqs.get(window_sec)

        rows.append(
            {
                "windowSec": window_sec,
                "maxSpeedKmh": float(max_speed) if pd.notna(max_speed) else None,
                "maxSpeedEqKmh": float(max_speed_eq) if pd.notna(max_speed_eq) else None,
                "hrAtMaxSpeed": float(hr_at_max_speed) if pd.notna(hr_at_max_speed) else None,
                "hrAtMaxSpeedEq": float(hr_at_max_speed_eq)
                if pd.notna(hr_at_max_speed_eq)
                else None,
            }
        )

    return pd.DataFrame(rows)
