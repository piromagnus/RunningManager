"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import pandas as pd
from streamlit.logger import get_logger

from utils.series_filters import filter_series_outliers
from utils import timeseries_preprocessing as ts_pre

logger = get_logger(__name__)


class PacerPreprocessor:
    """Preprocess GPX timeseries for race pacing segmentation."""

    def preprocess_timeseries_for_pacing(self, timeseries_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess timeseries for pacing (distance-referenced, time-invariant).

        This pipeline is for route preparation. It uses distance as the reference unit
        (cumulated_distance) and does NOT rely on timestamps. It computes distance,
        elevation smoothing, and grade, but intentionally skips duration/speed.

        Args:
            timeseries_df: DataFrame with lat, lon, elevationM (timestamp optional)

        Returns:
            Preprocessed DataFrame with cumulated_distance, elevationM_ma_5, grade_ma_10
        """
        if timeseries_df.empty:
            return pd.DataFrame()

        if "lat" not in timeseries_df.columns or "lon" not in timeseries_df.columns:
            logger.warning("Missing lat/lon columns for preprocessing")
            return pd.DataFrame()

        if timeseries_df["lat"].isna().all() or timeseries_df["lon"].isna().all():
            logger.warning("Insufficient GPS data")
            return pd.DataFrame()

        df = timeseries_df.copy()

        # Compute distance on ORIGINAL coordinates (for accurate total distance)
        # Don't smooth lat/lon before distance calculation as it reduces total distance
        df = ts_pre.distance(df, lat_col="lat", lon_col="lon")

        # Set first row distance to 0 (starting point)
        df.loc[0, "distance"] = 0.0

        # Filter out very small distances (GPS noise), but keep first point
        if len(df) > 1:
            mask = (df["distance"] > 1e-5) | (df.index == 0)
            df = df[mask].reset_index(drop=True)

        if df.empty:
            return pd.DataFrame()

        # Interpolate any remaining NaN distances
        df["distance"] = df["distance"].interpolate(method="linear").fillna(0.0)

        # Compute cumulated distance (using original accurate distance)
        df = ts_pre.cumulated_distance(df)

        # Remove extreme elevation spikes using distance-based reference (km).
        df = filter_series_outliers(
            df,
            value_col="elevationM",
            reference_col="cumulated_distance",
            window=0.5,
            sigma=3.0,
        )

        # Apply moving average ONLY to elevation (for noise reduction)
        df = ts_pre.moving_average(df, window_size=5, col="elevationM")

        # Compute elevation differences using smoothed elevation
        df = ts_pre.elevation(df, elevation_col="elevationM_ma_5")

        # Apply moving average to distance (for grade calculation smoothing)
        df = ts_pre.moving_average(df, window_size=10, col="distance")

        # Compute grade using smoothed distance
        df = ts_pre.grade(
            df, distance_col="distance_ma_10", elevation_col="elevation_difference"
        )

        # Apply moving average to grade
        df = ts_pre.moving_average(df, window_size=10, col="grade")

        return df
