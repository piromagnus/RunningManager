"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Elevation and grade preprocessing utilities.

Separates preprocessing logic from rendering logic for elevation profiles.
"""

from __future__ import annotations

import pandas as pd
from streamlit.logger import get_logger

from services.speed_profile_service import SpeedProfileService

logger = get_logger(__name__)


def preprocess_for_elevation_profile(
    timeseries_df: pd.DataFrame, speed_profile_service: SpeedProfileService | None = None
) -> pd.DataFrame | None:
    """Preprocess timeseries data for elevation profile visualization.

    Args:
        timeseries_df: Raw timeseries DataFrame with lat, lon, elevationM columns
        speed_profile_service: Optional SpeedProfileService instance (creates one if None)

    Returns:
        Preprocessed DataFrame with grade metrics, or None if preprocessing fails
    """
    if speed_profile_service is None:
        speed_profile_service = SpeedProfileService(config=None)

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

    return metrics_df

