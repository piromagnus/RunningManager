"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Speed profile analysis service for HR vs Speed correlation and clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from services.speed_profile import (
    hr_speed_analysis,
    minetti,
    persistence,
    preprocessing,
    profile_computation,
)
from utils.config import Config
from utils.constants import PROFILE_WINDOW_SIZES


@dataclass
class SpeedProfileResult:
    """Result of speed profile analysis."""

    best_offset: int
    best_correlation: float
    hr_smooth: pd.Series
    speed_smooth: pd.Series
    hr_shifted: pd.Series
    clusters: Optional[np.ndarray] = None
    cluster_centers: Optional[np.ndarray] = None
    filtered_cluster_centers: Optional[np.ndarray] = None
    filtered_cluster_ids: Optional[np.ndarray] = None
    cluster_slope: Optional[float] = None
    cluster_intercept: Optional[float] = None
    cluster_r_squared: Optional[float] = None
    cluster_std_err: Optional[float] = None
    profile_slope: Optional[float] = None
    profile_intercept: Optional[float] = None
    profile_r_squared: Optional[float] = None
    profile_std_err: Optional[float] = None
    window_sizes: Optional[List[int]] = None
    max_avg_speeds: Optional[Dict[int, float]] = None
    max_avg_hrs: Optional[Dict[int, float]] = None


class SpeedProfileService:
    """Service for analyzing speed profiles and HR vs Speed relationships."""

    def __init__(self, config: Optional[Config] = None):
        self.profile_window_sizes = PROFILE_WINDOW_SIZES

        if config is not None:
            self.config = config
            self.metrics_ts_dir = config.metrics_ts_dir
            self.metrics_ts_dir.mkdir(parents=True, exist_ok=True)
            self.speed_profile_dir = config.speed_profile_dir
            self.speed_profile_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Preprocessing Functions
    # ------------------------------------------------------------------

    @staticmethod
    def moving_average(df: pd.DataFrame, window_size: int, col: str) -> pd.DataFrame:
        """Compute the moving average of a column over a specified window size."""
        return preprocessing.moving_average(df, window_size, col)

    @staticmethod
    def distance(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
        """Compute the distance between 2 consecutive rows based on latitude and longitude."""
        return preprocessing.distance(df, lat_col=lat_col, lon_col=lon_col)

    @staticmethod
    def cumulated_distance(df: pd.DataFrame, distance_col: str = "distance") -> pd.DataFrame:
        """Compute the cumulated distance based on the distance column."""
        return preprocessing.cumulated_distance(df, distance_col=distance_col)

    @staticmethod
    def time_from_timestamp(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """Convert the timestamp column to a datetime object."""
        return preprocessing.time_from_timestamp(df, timestamp_col=timestamp_col)

    @staticmethod
    def duration(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """Compute the duration of the dataframe based on the timestamp column."""
        return preprocessing.duration(df, timestamp_col=timestamp_col)

    @staticmethod
    def activity_duration_seconds(df: pd.DataFrame) -> float:
        """Return activity duration in seconds, falling back to sample count."""
        return preprocessing.activity_duration_seconds(df)

    @staticmethod
    def speed(
        df: pd.DataFrame,
        distance_col: str = "distance",
        time_col: str = "duration_seconds",
    ) -> pd.DataFrame:
        """Compute the speed of the dataframe based on the distance and time columns."""
        return preprocessing.speed(df, distance_col=distance_col, time_col=time_col)

    @staticmethod
    def elevation(df: pd.DataFrame, elevation_col: str = "elevationM_ma_5") -> pd.DataFrame:
        """Compute the elevation difference of the dataframe based on the elevation column."""
        return preprocessing.elevation(df, elevation_col=elevation_col)

    @staticmethod
    def grade(
        df: pd.DataFrame,
        distance_col: str = "distance",
        elevation_col: str = "elevation_difference",
    ) -> pd.DataFrame:
        """Compute the grade of the dataframe based on the distance and elevation columns."""
        return preprocessing.grade(df, distance_col=distance_col, elevation_col=elevation_col)

    def preprocess_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess activity timeseries (time-referenced pipeline).

        This pipeline is for real activities: it uses timestamps as the reference unit
        to compute durations and speeds. It requires GPS data and valid timestamps to
        derive time-based metrics (duration, speed) along with grade.

        If GPS data is missing or insufficient, returns an empty DataFrame.
        """
        return preprocessing.preprocess_timeseries(df)

    # ------------------------------------------------------------------
    # Shift Computation
    # ------------------------------------------------------------------

    def compute_hr_speed_shift(
        self,
        df: pd.DataFrame,
        hr_col: str = "hr",
        speed_col: str = "speed_km_h",
        min_hr: int = 120,
    ) -> Tuple[int, float]:
        """Compute optimal HR shift for maximum correlation with speed."""
        return hr_speed_analysis.compute_hr_speed_shift(
            df,
            hr_col=hr_col,
            speed_col=speed_col,
            min_hr=min_hr,
        )

    # ------------------------------------------------------------------
    # Cluster-Based Analysis (Strategy 1)
    # ------------------------------------------------------------------

    def cluster_based_analysis(
        self,
        df: pd.DataFrame,
        best_offset: int,
        n_clusters: int = 7,
        r_threshold: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """Perform cluster-based HR vs Speed analysis.
        
        Expects df to already be filtered (hr > 120) and smoothed.
        """
        return hr_speed_analysis.cluster_based_analysis(
            df, best_offset, n_clusters=n_clusters, r_threshold=r_threshold
        )

    # ------------------------------------------------------------------
    # Profile-Based Analysis (Strategy 2)
    # ------------------------------------------------------------------

    def compute_profile(
        self,
        df: pd.DataFrame,
        col: str,
        additional_col: Optional[str],
        window_sizes: List[int],
        best_offset: int,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Compute max average speeds/HRs for different window sizes."""
        return hr_speed_analysis.compute_profile(df, col, additional_col, window_sizes, best_offset)

    def profile_based_analysis(
        self, df: pd.DataFrame, best_offset: int, window_sizes: Optional[List[int]] = None
    ) -> Tuple[float, float, float, float]:
        """Perform profile-based HR vs Speed analysis."""
        return hr_speed_analysis.profile_based_analysis(df, best_offset, window_sizes)

    # ------------------------------------------------------------------
    # Full Processing
    # ------------------------------------------------------------------

    def process_timeseries(
        self, activity_id: str, strategy: str = "cluster", n_clusters: Optional[int] = None
    ) -> Optional[SpeedProfileResult]:
        """
        Processes a complete timeseries data file for a given activity and returns analysis results according
        to the chosen strategy ("cluster" or "profile").

        This function performs a sequence of operations to transform timeseries data (such as
        HR and speed traces from running or cycling activities) into summarized analytical
        results, ready for further analysis or visualization. The function is robust,
        applying fallback strategies if certain fields are missing, and covers both
        clustering-based
        and profile-based approaches. The steps are as follows:

        1. Load the Timeseries Data:
            - Try reading the timeseries CSV corresponding to the provided `activity_id` from the configured directory.
            - If the file does not exist, or loading fails, or is missing HR data, return None.

        2. Preprocessing:
            - Attempt to preprocess the timeseries using GPS-derived speeds
              (via `preprocess_timeseries`). This typically
              refines or creates the `speed_km_h` column from raw GPS data.
            - If GPS-based speed cannot be computed, fall back to using the `paceKmh` column directly (if available).
                - Convert `paceKmh` to `speed_km_h`, remove rows with missing or
                  nonpositive speeds or HR, recompute time
                  and duration columns, and filter for realistic speed ranges.

        3. Validation:
            - After preprocessing, check that the dataframe is non-empty and contains both
              HR and speed columns.
              If not, return None.

        4. Heart Rate / Speed Shift Computation:
            - Compute the temporal shift (best_offset) between heart rate (HR) and speed
              signals to maximize their correlation.

        5. Data Smoothing and Filtering:
            - Filter data to HR values above 120 bpm (to focus on active segments).
            - Smooth the HR and speed signals using a moving average (window=10).
            - Rename the smoothed columns for clarity and shift HR by the calculated offset for alignment.

        6. Analysis Strategy:
            - If `strategy` is "cluster":
                - Call `cluster_based_analysis` to segment data into HR/speed clusters,
                  fit a regression line per cluster,
                  and summarize each segment.
                - Return a `SpeedProfileResult` object comprising shift, correlation,
                  cluster labels, smooth HR/speed,
                  and cluster-wise regression results.
            - If `strategy` is "profile":
                - Call `profile_based_analysis` to compute the relationship between HR and
                  speed profiles across
                  multiple rolling window sizes.
                - Additionally, calculate maximum average speed and HR per window size.
                - Return a `SpeedProfileResult` object summarizing profile-based regression,
                  as well as the computed
                  profiles for further review or plotting.

        Args:
            activity_id (str): Identifier for the activity whose timeseries should be
                processed. This references a specific CSV file.
            strategy (str, optional): Which analysis method to use:
                - "cluster": HR/Speed clustering analysis.
                - "profile": Profile-based (rolling average) analysis.
                Default is "cluster".
            n_clusters (Optional[int], optional): Number of clusters for cluster-based analysis. If not provided, uses
                the value from the configuration.

        Returns:
            Optional[SpeedProfileResult]:
                - Returns a populated SpeedProfileResult containing the computed analysis,
                  or None if the process failed at any stage.

        Detailed Notes:
            - Both analysis strategies return timeseries-aligned HR/speed data and regression outputs.
            - This function handles missing GPS or speed data gracefully, defaulting to
              fallback columns when appropriate.
            - Returns are always consistent in typing; None is only returned on hard failures (missing data/columns).
        """
        if n_clusters is None:
            n_clusters = self.config.n_cluster
        path = self.config.timeseries_dir / f"{activity_id}.csv"
        if not path.exists():
            return None

        try:
            df = pd.read_csv(path)
        except Exception:
            return None

        if df.empty or "hr" not in df.columns:
            return None

        # Preprocess - try GPS-based preprocessing first
        df_processed = self.preprocess_timeseries(df)

        # If preprocessing failed (no GPS data), fall back to using paceKmh directly
        if df_processed.empty or "speed_km_h" not in df_processed.columns:
            if "paceKmh" not in df.columns:
                return None
            # Use paceKmh directly - minimal preprocessing
            df_processed = df.copy()
            df_processed["speed_km_h"] = pd.to_numeric(df_processed["paceKmh"], errors="coerce")
            df_processed = df_processed.dropna(subset=["speed_km_h", "hr"])
            df_processed = df_processed[df_processed["speed_km_h"] > 0]
            df_processed = self.time_from_timestamp(df_processed)
            df_processed = self.duration(df_processed)
            # Filter unrealistic speeds
            df_processed = df_processed[df_processed["speed_km_h"] < 40].reset_index(drop=True)

        if df_processed.empty or "hr" not in df_processed.columns or "speed_km_h" not in df_processed.columns:
            return None

        # Compute shift
        best_offset, best_correlation = self.compute_hr_speed_shift(df_processed)

        # Prepare smoothed data
        df_filtered = df_processed[df_processed["hr"] > 120].copy()
        if df_filtered.empty:
            return None

        # Reset index to ensure sequential indices
        df_filtered = df_filtered.reset_index(drop=True)

        df_filtered = self.moving_average(df_filtered, window_size=10, col="hr")
        df_filtered = self.moving_average(df_filtered, window_size=10, col="speed_km_h")
        df_filtered.rename(
            columns={"hr_ma_10": "hr_smooth", "speed_km_h_ma_10": "speed_smooth"}, inplace=True
        )
        df_filtered["hr_shifted"] = df_filtered["hr_smooth"].shift(best_offset)

        # Apply strategy
        if strategy == "cluster":
            clusters, cluster_centers, cluster_ids, slope, intercept, r_squared, std_err = (
                self.cluster_based_analysis(df_filtered, best_offset, n_clusters)
            )

            return SpeedProfileResult(
                best_offset=best_offset,
                best_correlation=best_correlation,
                hr_smooth=df_filtered["hr_smooth"],
                speed_smooth=df_filtered["speed_smooth"],
                hr_shifted=df_filtered["hr_shifted"],
                clusters=clusters,
                filtered_cluster_centers=cluster_centers,
                filtered_cluster_ids=cluster_ids,
                cluster_slope=slope,
                cluster_intercept=intercept,
                cluster_r_squared=r_squared,
                cluster_std_err=std_err,
            )

        elif strategy == "profile":
            slope, intercept, r_squared, std_err = self.profile_based_analysis(df_processed, best_offset)

            window_sizes = [
                15, 20, 30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020,
                1080, 1140, 1200, 1320, 1440, 1560, 1680, 1920, 2160, 2400, 2640, 2880, 3120, 3360, 3600,
            ]

            df_filtered = df_processed[df_processed["hr"] > 120].copy()
            df_filtered = self.moving_average(df_filtered, window_size=10, col="hr")
            df_filtered = self.moving_average(df_filtered, window_size=10, col="speed_km_h")
            df_filtered.rename(
                columns={"hr_ma_10": "hr_smooth", "speed_km_h_ma_10": "speed_smooth"}, inplace=True
            )
            df_filtered["hr_shifted"] = df_filtered["hr_smooth"].shift(best_offset)

            max_avg_speeds, _ = self.compute_profile(df_filtered, "speed_smooth", None, window_sizes, best_offset)
            max_avg_hrs, _ = self.compute_profile(df_filtered, "hr_shifted", None, window_sizes, best_offset)

            return SpeedProfileResult(
                best_offset=best_offset,
                best_correlation=best_correlation,
                hr_smooth=df_filtered["hr_smooth"],
                speed_smooth=df_filtered["speed_smooth"],
                hr_shifted=df_filtered["hr_shifted"],
                profile_slope=slope,
                profile_intercept=intercept,
                profile_r_squared=r_squared,
                profile_std_err=std_err,
                window_sizes=window_sizes,
                max_avg_speeds=max_avg_speeds,
                max_avg_hrs=max_avg_hrs,
            )

        return None

    def save_metrics_ts(self, activity_id: str, result: SpeedProfileResult) -> Path:
        """Save processed metrics to metrics_ts folder."""
        return persistence.save_metrics_ts(self, activity_id, result)

    def compute_and_save_elevation_metrics(self, activity_id: str) -> Optional[Path]:
        """Compute and save elevation-related metrics to metrics_ts for caching.
        
        Computes speedeq_smooth, grade_ma_10, elevationM_ma_5, cumulated_distance,
        and other metrics needed for elevation profile visualization.
        Also preserves existing HR-related columns (hr_smooth, hr_shifted, cluster) if present.
        
        Parameters:
        activity_id (str): Activity identifier
        
        Returns:
        Optional[Path]: Path to saved metrics_ts CSV, or None if computation failed
        """
        return persistence.compute_and_save_elevation_metrics(self, activity_id)

    def compute_all_metrics_ts(self, activity_id: str) -> Optional[Path]:
        """Compute and save all metrics_ts data: HR analysis + elevation metrics.
        
        This is the main entry point for computing complete metrics_ts files.
        It computes both HR-related metrics (hr_smooth, hr_shifted, cluster) and
        elevation metrics (speedeq_smooth, grade_ma_10, elevationM_ma_5, etc.)
        
        Parameters:
        activity_id (str): Activity identifier
        
        Returns:
        Optional[Path]: Path to saved metrics_ts CSV, or None if computation failed
        """
        return persistence.compute_all_metrics_ts(self, activity_id)

    def load_elevation_metrics(self, activity_id: str) -> Optional[pd.DataFrame]:
        """Load elevation metrics from metrics_ts if available.
        
        Checks if the cached metrics_ts file contains the required elevation columns.
        
        Parameters:
        activity_id (str): Activity identifier
        
        Returns:
        Optional[pd.DataFrame]: Cached elevation metrics DataFrame or None
        """
        return persistence.load_elevation_metrics(self, activity_id)

    def get_or_compute_elevation_metrics(self, activity_id: str) -> Optional[pd.DataFrame]:
        """Get elevation metrics from cache or compute and save them.
        
        Parameters:
        activity_id (str): Activity identifier
        
        Returns:
        Optional[pd.DataFrame]: Elevation metrics DataFrame or None
        """
        return persistence.get_or_compute_elevation_metrics(self, activity_id)

    def load_metrics_ts(self, activity_id: str) -> Optional[pd.DataFrame]:
        """Load processed metrics from metrics_ts folder."""
        return persistence.load_metrics_ts(self, activity_id)

    # ------------------------------------------------------------------
    # Speed Equivalent (Minetti Energy Cost Model)
    # ------------------------------------------------------------------

    @staticmethod
    def minetti_energy_cost_walking(grade: float) -> float:
        """
        Calculate energy cost of walking at a given grade using Minetti et al. (2002).
        
        Energy cost formula: Cw(i) = 280.5i^5 - 58.7i^4 - 76.8i^3 + 51.9i^2 + 19.6i + 2.5
        
        Parameters:
        grade (float): Grade as decimal (e.g., 0.1 for 10% grade)
        
        Returns:
        float: Energy cost per unit distance
        """
        return minetti.minetti_energy_cost_walking(grade)

    @staticmethod
    def minetti_energy_cost_running(grade: float) -> float:
        """
        Calculate energy cost of running at a given grade using Minetti et al. (2002).
        
        Energy cost formula: Cr(i) = 155.4i^5 - 30.4i^4 - 43.3i^3 + 46.3i^2 + 19.5i + 3.6
        
        Parameters:
        grade (float): Grade as decimal (e.g., 0.1 for 10% grade)
        
        Returns:
        float: Energy cost per unit distance
        """
        return minetti.minetti_energy_cost_running(grade)

    @staticmethod
    def compute_speed_eq_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add speed_eq_km_h column using Minetti energy cost model.
        
        Equivalent speed represents the speed on flat ground that would require
        the same energy expenditure as the actual speed on the given grade.
        
        Calculation: speed_eq = speed * (C(grade) / C(0))
        - For walking (speed <= 6 km/h): C(0) = 2.5
        - For running (speed > 6 km/h): C(0) = 3.6
        
        Parameters:
        df (pd.DataFrame): DataFrame with speed_km_h and grade columns
        
        Returns:
        pd.DataFrame: DataFrame with added speed_eq_km_h column
        """
        return profile_computation.compute_speed_eq_column(df)

    def compute_max_speed_profiles(
        self, df: pd.DataFrame, window_sizes: List[int]
    ) -> pd.DataFrame:
        """
        Compute maximum rolling average speed profiles for both raw speed and equivalent speed.
        
        For each window size, computes centered rolling mean and records the maximum value
        for both speed_km_h and speed_eq_km_h.
        
        Parameters:
        df (pd.DataFrame): Preprocessed timeseries DataFrame with speed_km_h
        window_sizes (List[int]): List of window sizes in seconds
        
        Returns:
        pd.DataFrame: DataFrame with columns windowSec, maxSpeedKmh, maxSpeedEqKmh
        """
        return profile_computation.compute_max_speed_profiles(df, window_sizes)

    def compute_speed_profile_cloud(
        self, df: pd.DataFrame, window_sizes: List[int]
    ) -> pd.DataFrame:
        """Compute per-window max speeds with associated HR values."""
        return profile_computation.compute_speed_profile_cloud(df, window_sizes)

    def compute_speed_profile_cloud(
        self, df: pd.DataFrame, window_sizes: List[int]
    ) -> pd.DataFrame:
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
            df = self.moving_average(df, window_size=10, col="speed_km_h")
            df = df.rename(columns={"speed_km_h_ma_10": "speed_km_h_smooth"})
        else:
            df = df.rename(columns={"speed_km_h_ma_10": "speed_km_h_smooth"})

        if "hr_ma_10" not in df.columns:
            df = self.moving_average(df, window_size=10, col="hr")
            df = df.rename(columns={"hr_ma_10": "hr_smooth"})
        else:
            df = df.rename(columns={"hr_ma_10": "hr_smooth"})

        if "grade_ma_10" not in df.columns:
            if "grade" in df.columns:
                df = self.moving_average(df, window_size=10, col="grade")
            else:
                df["grade_ma_10"] = 0.0

        df = self.compute_speed_eq_column(df)

        max_speeds, hr_at_max_speeds = self.compute_profile(
            df, "speed_km_h_smooth", "hr_smooth", window_sizes, 0
        )
        max_speed_eqs, hr_at_max_speed_eqs = self.compute_profile(
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
                    "hrAtMaxSpeed": float(hr_at_max_speed)
                    if pd.notna(hr_at_max_speed)
                    else None,
                    "hrAtMaxSpeedEq": float(hr_at_max_speed_eq)
                    if pd.notna(hr_at_max_speed_eq)
                    else None,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Speed Profile Persistence
    # ------------------------------------------------------------------

    def save_speed_profile(self, activity_id: str, profile_df: pd.DataFrame) -> Path:
        """
        Save speed profile DataFrame to CSV file.
        
        Parameters:
        activity_id (str): Activity identifier
        profile_df (pd.DataFrame): Profile DataFrame with windowSec, maxSpeedKmh, maxSpeedEqKmh
        
        Returns:
        Path: Path to saved file
        """
        return persistence.save_speed_profile(self, activity_id, profile_df)

    def load_speed_profile(self, activity_id: str) -> Optional[pd.DataFrame]:
        """
        Load speed profile from CSV file.
        
        Parameters:
        activity_id (str): Activity identifier
        
        Returns:
        Optional[pd.DataFrame]: Profile DataFrame or None if not found
        """
        return persistence.load_speed_profile(self, activity_id)

    def compute_and_store_speed_profile(
        self, activity_id: str, window_sizes: Optional[List[int]] = None
    ) -> Optional[Path]:
        """
        Compute and store speed profile for an activity.
        
        Loads timeseries, preprocesses it, computes max speed profiles for all window sizes,
        and persists the result. If profile already exists, returns path without recomputing
        (idempotent).
        
        Parameters:
        activity_id (str): Activity identifier
        window_sizes (Optional[List[int]]): Window sizes in seconds. Uses PROFILE_WINDOW_SIZES if None.
        
        Returns:
        Optional[Path]: Path to saved profile CSV, or None if computation failed
        """
        return persistence.compute_and_store_speed_profile(self, activity_id, window_sizes)

