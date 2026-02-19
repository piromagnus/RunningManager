"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from services.speed_profile import preprocessing, profile_computation

if TYPE_CHECKING:
    from services.speed_profile_service import SpeedProfileResult, SpeedProfileService


def save_metrics_ts(
    service: "SpeedProfileService", activity_id: str, result: "SpeedProfileResult"
) -> Path:
    """Save processed metrics to metrics_ts folder."""
    path = service.metrics_ts_dir / f"{activity_id}.csv"

    min_len = len(result.hr_smooth)
    if result.speed_smooth is not None:
        min_len = min(min_len, len(result.speed_smooth))
    if result.hr_shifted is not None:
        min_len = min(min_len, len(result.hr_shifted))

    data = {
        "hr_smooth": result.hr_smooth.values[:min_len],
        "speed_smooth": result.speed_smooth.values[:min_len],
        "hr_shifted": result.hr_shifted.values[:min_len],
    }

    if result.clusters is not None and len(result.clusters) >= min_len:
        data["cluster"] = result.clusters[:min_len]

    df_save = pd.DataFrame(data)
    df_save.to_csv(path, index=False)
    return path


def compute_and_save_elevation_metrics(
    service: "SpeedProfileService", activity_id: str
) -> Optional[Path]:
    """Compute and save elevation-related metrics to metrics_ts for caching."""
    timeseries_path = service.config.timeseries_dir / f"{activity_id}.csv"
    if not timeseries_path.exists():
        return None

    try:
        df = pd.read_csv(timeseries_path)
    except Exception:
        return None

    if df.empty:
        return None

    if "lat" not in df.columns or "lon" not in df.columns:
        return None

    if df["lat"].isna().all() or df["lon"].isna().all():
        return None

    df_processed = preprocessing.preprocess_timeseries(df)
    if df_processed.empty:
        return None

    df_processed = preprocessing.moving_average(df_processed, window_size=10, col="grade")
    df_processed = profile_computation.compute_speed_eq_column(df_processed)

    if "speed_eq_km_h" in df_processed.columns:
        df_processed = preprocessing.moving_average(df_processed, window_size=10, col="speed_eq_km_h")
        df_processed.rename(columns={"speed_eq_km_h_ma_10": "speedeq_smooth"}, inplace=True)

    columns_to_save = [
        "cumulated_distance",
        "cumulated_duration_seconds",
        "elevationM_ma_5",
        "grade_ma_10",
        "speed_km_h",
        "speedeq_smooth",
    ]

    if "hr" in df_processed.columns:
        columns_to_save.append("hr")

    available_cols = [col for col in columns_to_save if col in df_processed.columns]
    if not available_cols:
        return None

    df_save = df_processed[available_cols].copy()

    existing_df = load_metrics_ts(service, activity_id)
    if existing_df is not None:
        hr_cols = ["hr_smooth", "speed_smooth", "hr_shifted", "cluster"]
        existing_hr_cols = [col for col in hr_cols if col in existing_df.columns]
        if existing_hr_cols:
            for col in existing_hr_cols:
                if len(existing_df) <= len(df_save):
                    padded_values = list(existing_df[col].values) + [np.nan] * (
                        len(df_save) - len(existing_df)
                    )
                    df_save[col] = padded_values
                else:
                    df_save[col] = existing_df[col].values[: len(df_save)]

    path = service.metrics_ts_dir / f"{activity_id}.csv"
    df_save.to_csv(path, index=False)
    return path


def compute_all_metrics_ts(
    service: "SpeedProfileService", activity_id: str
) -> Optional[Path]:
    """Compute and save all metrics_ts data: HR analysis + elevation metrics."""
    result = service.process_timeseries(activity_id, strategy="cluster")
    if result is not None:
        save_metrics_ts(service, activity_id, result)

    path = compute_and_save_elevation_metrics(service, activity_id)

    if path is not None:
        return path

    if result is not None:
        return service.metrics_ts_dir / f"{activity_id}.csv"

    return None


def load_elevation_metrics(service: "SpeedProfileService", activity_id: str) -> Optional[pd.DataFrame]:
    """Load elevation metrics from metrics_ts if available."""
    cached_df = load_metrics_ts(service, activity_id)
    if cached_df is None:
        return None

    required_cols = ["cumulated_distance", "elevationM_ma_5", "grade_ma_10"]
    if not all(col in cached_df.columns for col in required_cols):
        return None

    return cached_df


def get_or_compute_elevation_metrics(
    service: "SpeedProfileService", activity_id: str
) -> Optional[pd.DataFrame]:
    """Get elevation metrics from cache or compute and save them."""
    cached_df = load_elevation_metrics(service, activity_id)
    if cached_df is not None:
        return cached_df

    path = compute_and_save_elevation_metrics(service, activity_id)
    if path is None:
        return None

    return load_elevation_metrics(service, activity_id)


def load_metrics_ts(service: "SpeedProfileService", activity_id: str) -> Optional[pd.DataFrame]:
    """Load processed metrics from metrics_ts folder."""
    path = service.metrics_ts_dir / f"{activity_id}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def save_speed_profile(
    service: "SpeedProfileService", activity_id: str, profile_df: pd.DataFrame
) -> Path:
    """Save computed speed profile data to CSV."""
    path = service.speed_profile_dir / f"{activity_id}.csv"
    profile_df.to_csv(path, index=False)
    return path


def load_speed_profile(service: "SpeedProfileService", activity_id: str) -> Optional[pd.DataFrame]:
    """Load speed profile data from CSV."""
    path = service.speed_profile_dir / f"{activity_id}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def compute_and_store_speed_profile(
    service: "SpeedProfileService", activity_id: str, window_sizes: Optional[list[int]] = None
) -> Optional[Path]:
    """Compute and store speed profiles for an activity (idempotent)."""
    if window_sizes is None:
        window_sizes = service.profile_window_sizes

    path = service.speed_profile_dir / f"{activity_id}.csv"
    if path.exists():
        return path

    timeseries_path = service.config.timeseries_dir / f"{activity_id}.csv"
    if not timeseries_path.exists():
        return None

    try:
        df = pd.read_csv(timeseries_path)
    except Exception:
        return None

    if df.empty:
        return None

    df_processed = preprocessing.preprocess_timeseries(df)
    if df_processed.empty:
        return None

    profile_df = profile_computation.compute_max_speed_profiles(df_processed, window_sizes)
    if profile_df.empty:
        return None

    return save_speed_profile(service, activity_id, profile_df)
