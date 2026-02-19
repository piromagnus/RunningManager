"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Elevation and grade preprocessing utilities.

Separates preprocessing logic from rendering logic for elevation profiles.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from streamlit.logger import get_logger

from services.speed_profile_service import SpeedProfileService

logger = get_logger(__name__)


def preprocess_for_elevation_profile(
    timeseries_df: pd.DataFrame,
    speed_profile_service: SpeedProfileService | None = None,
    activity_id: str | None = None,
) -> pd.DataFrame | None:
    """Preprocess timeseries data for elevation profile visualization.

    First checks if cached elevation metrics are available in metrics_ts.
    If not, computes them from the raw timeseries and optionally saves them.

    Args:
        timeseries_df: Raw timeseries DataFrame with lat, lon, elevationM columns
        speed_profile_service: Optional SpeedProfileService instance (creates one if None)
        activity_id: Optional activity ID to enable caching (load/save from metrics_ts)

    Returns:
        Preprocessed DataFrame with grade metrics, or None if preprocessing fails
    """
    if speed_profile_service is None:
        speed_profile_service = SpeedProfileService(config=None)

    # Try to load cached elevation metrics if activity_id is provided
    if activity_id is not None and speed_profile_service.config is not None:
        cached_df = speed_profile_service.load_elevation_metrics(activity_id)
        if cached_df is not None:
            logger.debug(f"Loaded cached elevation metrics for activity {activity_id}")
            return cached_df

    # Check if we have GPS data for preprocessing
    if "lat" not in timeseries_df.columns or "lon" not in timeseries_df.columns:
        logger.debug("Missing GPS data (lat/lon columns) for elevation profile")
        return None

    if timeseries_df["lat"].isna().all() or timeseries_df["lon"].isna().all():
        logger.debug("Insufficient GPS data (all NaN) for elevation profile")
        return None

    try:
        metrics_df = speed_profile_service.preprocess_timeseries(timeseries_df)
    except Exception as e:
        logger.warning(f"Failed to preprocess timeseries for elevation profile: {e}", exc_info=True)
        return None

    if metrics_df.empty:
        logger.debug("Preprocessing returned empty DataFrame")
        return None

    # Apply moving average to grade for smoothing
    try:
        metrics_df = speed_profile_service.moving_average(metrics_df, window_size=10, col="grade")
    except Exception as e:
        logger.warning(f"Failed to apply moving average to grade: {e}", exc_info=True)
        return None

    # Compute and add speedeq_smooth
    try:
        metrics_df = speed_profile_service.compute_speed_eq_column(metrics_df)
        if "speed_eq_km_h" in metrics_df.columns:
            metrics_df = speed_profile_service.moving_average(metrics_df, window_size=10, col="speed_eq_km_h")
            metrics_df.rename(columns={"speed_eq_km_h_ma_10": "speedeq_smooth"}, inplace=True)
    except Exception as e:
        logger.warning(f"Failed to compute speed_eq: {e}", exc_info=True)
        # Continue without speedeq_smooth - it's optional

    # Note: We don't save to cache here to avoid overwriting HR columns (hr_smooth, hr_shifted, cluster).
    # Use SpeedProfileService.compute_all_metrics_ts() to compute and save all metrics together.

    return metrics_df

