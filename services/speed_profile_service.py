"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Speed profile analysis service for HR vs Speed correlation and clustering.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from haversine import haversine
from sklearn.cluster import KMeans

import config
from utils.config import Config


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
        df = df.copy()
        df[f"{col}_ma_{window_size}"] = (
            df[col].rolling(window=window_size, min_periods=1, center=True).mean()
        )
        return df

    @staticmethod
    def distance(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
        """Compute the distance between 2 consecutive rows based on latitude and longitude."""
        df = df.copy()
        df["distance"] = [
            haversine((lat1, lon1), (lat2, lon2))
            for lat1, lon1, lat2, lon2 in zip(
                df[lat_col].shift(), df[lon_col].shift(), df[lat_col], df[lon_col]
            )
        ]
        return df

    @staticmethod
    def cumulated_distance(df: pd.DataFrame, distance_col: str = "distance") -> pd.DataFrame:
        """Compute the cumulated distance based on the distance column."""
        df = df.copy()
        df["cumulated_distance"] = df[distance_col].cumsum()
        return df

    @staticmethod
    def time_from_timestamp(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """Convert the timestamp column to a datetime object."""
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df["time"] = df[timestamp_col].dt.time
        return df

    @staticmethod
    def duration(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """Compute the duration of the dataframe based on the timestamp column."""
        df = df.copy()
        df["duration"] = df[timestamp_col].diff().fillna(pd.Timedelta(seconds=0))
        df["cumulated_duration"] = df["duration"].cumsum()
        df["duration_seconds"] = df["duration"].dt.total_seconds()
        df["cumulated_duration_seconds"] = df["duration_seconds"].cumsum()
        return df

    @staticmethod
    def speed(df: pd.DataFrame, distance_col: str = "distance", time_col: str = "duration_seconds") -> pd.DataFrame:
        """Compute the speed of the dataframe based on the distance and time columns."""
        df = df.copy()
        mean_time = df.loc[df[time_col] > 0, time_col].mean()
        if pd.isna(mean_time) or mean_time == 0:
            mean_time = 1.0
        df[time_col] = df[time_col].replace(0, mean_time)
        df["speed_m_s"] = 1000 * df[distance_col] / df[time_col]
        df["speed_km_h"] = 3.6 * df["speed_m_s"]
        return df

    @staticmethod
    def elevation(df: pd.DataFrame, elevation_col: str = "elevationM_ma_5") -> pd.DataFrame:
        """Compute the elevation difference of the dataframe based on the elevation column."""
        df = df.copy()
        df["elevation_difference"] = df[elevation_col].diff().fillna(0)
        df["elevation_cumulated"] = df["elevation_difference"].cumsum()
        df["elevation_gain"] = df["elevation_difference"].apply(lambda x: x if x > 0 else 0).cumsum()
        df["elevation_loss"] = df["elevation_difference"].apply(lambda x: -x if x < 0 else 0).cumsum()
        return df

    @staticmethod
    def grade(df: pd.DataFrame, distance_col: str = "distance", elevation_col: str = "elevation_difference") -> pd.DataFrame:
        """Compute the grade of the dataframe based on the distance and elevation columns."""
        df = df.copy()
        df["grade"] = df[elevation_col] / (df[distance_col] * 1000)
        df["grade"] = df["grade"].replace([np.inf, -np.inf], 0)
        df["grade"] = df["grade"].fillna(0)
        return df

    def preprocess_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess timeseries data following the notebook workflow.
        
        Requires lat/lon columns for GPS-based distance computation.
        If GPS data is missing or insufficient, returns empty DataFrame.
        """
        df = df.copy()

        # Check if we have GPS data
        if "lat" not in df.columns or "lon" not in df.columns:
            return pd.DataFrame()
        
        # Check if GPS data is present and valid
        if df["lat"].isna().all() or df["lon"].isna().all():
            return pd.DataFrame()

        # Apply moving average to lat/lon
        df = self.moving_average(df, window_size=5, col="lat")
        df = self.moving_average(df, window_size=5, col="lon")
        df = self.moving_average(df, window_size=5, col="elevationM")
        # Compute distance
        df = self.distance(df, lat_col="lat_ma_5", lon_col="lon_ma_5")

        # Filter out very small distances
        df = df[df["distance"] > 1e-5].reset_index(drop=True)
        
        if df.empty:
            return pd.DataFrame()

        # Interpolate distance
        df["distance"] = df["distance"].interpolate(method="linear")

        # Compute elevation
        df = self.elevation(df, elevation_col="elevationM_ma_5")

        # Convert timestamp and compute duration
        df = self.time_from_timestamp(df)
        df = self.duration(df)

        # Compute cumulated distance
        df = self.cumulated_distance(df)

        # Apply moving average to distance
        df = self.moving_average(df, window_size=10, col="distance")

        # Compute speed
        df = self.speed(df, distance_col="distance", time_col="duration_seconds")

        # Filter out unrealistic speeds
        df = df[df["speed_km_h"] < 40].reset_index(drop=True)

        # Compute grade
        df = self.grade(df, distance_col="distance_ma_10", elevation_col="elevation_difference")

        return df

    # ------------------------------------------------------------------
    # Shift Computation
    # ------------------------------------------------------------------

    def compute_hr_speed_shift(
        self, df: pd.DataFrame, hr_col: str = "hr", speed_col: str = "speed_km_h", min_hr: int = 120
    ) -> Tuple[int, float]:
        """Compute optimal HR shift for maximum correlation with speed."""
        df = df.copy()

        # Filter by minimum HR
        df = df[df[hr_col] > min_hr].copy()

        if df.empty:
            return 0, 0.0

        # Apply smoothing
        df = self.moving_average(df, window_size=10, col=hr_col)
        df = self.moving_average(df, window_size=10, col=speed_col)

        # Rename columns
        df.rename(columns={"hr_ma_10": "hr_smooth", f"{speed_col}_ma_10": "speed_smooth"}, inplace=True)

        hr_smooth_col = "hr_smooth"
        speed_smooth_col = "speed_smooth"

        # Find best offset
        best_correlation = 0.0
        best_offset = 0

        for offset in range(-60, 61):
            shifted_hr = df[hr_smooth_col].shift(offset)
            valid_data = pd.concat([df[speed_smooth_col], shifted_hr], axis=1).dropna()
            if len(valid_data) > 0:
                correlation = valid_data.corr().iloc[0, 1]
                if not pd.isna(correlation) and abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_offset = offset

        return best_offset, best_correlation

    # ------------------------------------------------------------------
    # Cluster-Based Analysis (Strategy 1)
    # ------------------------------------------------------------------

    def cluster_based_analysis(
        self, df: pd.DataFrame, best_offset: int, n_clusters: int = 7, r_threshold: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """Perform cluster-based HR vs Speed analysis.
        
        Expects df to already be filtered (hr > 120) and smoothed.
        """
        df = df.copy()

        if df.empty or "hr_shifted" not in df.columns or "speed_smooth" not in df.columns:
            return np.array([]), np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0

        # Reset index to ensure sequential indices (0, 1, 2, ...)
        df = df.reset_index(drop=True)

        # Prepare data for clustering
        X = df[["hr_shifted", "speed_smooth"]].dropna()
        if len(X) < n_clusters:
            return np.array([]), np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X.values)

        # Map clusters back to full dataframe (NaN where data was missing)
        # X.index now contains sequential indices matching df's reset index
        clusters_full = np.full(len(df), -1, dtype=int)
        valid_indices = X.index.values
        # Ensure indices are within bounds
        valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < len(df))]
        if len(valid_indices) == len(clusters):
            clusters_full[valid_indices] = clusters
        else:
            # Fallback: if length mismatch, use positional mapping
            clusters_full[:len(clusters)] = clusters[:len(clusters_full)]

        cluster_centers = kmeans.cluster_centers_

        # Linear regression on cluster centers
        if len(cluster_centers) < 2:
            return (
                clusters_full,
                cluster_centers,
                np.arange(len(cluster_centers)),
                0.0,
                0.0,
                0.0,
                0.0,
            )

        slope, intercept, r_value, p_value, std_err = stats.linregress(cluster_centers[:, 0], cluster_centers[:, 1])

        # Filter clusters if R² is below threshold
        if abs(r_value) < r_threshold and len(cluster_centers) > 1:
            n_centers = len(cluster_centers)
            remove_count = max(1, int(n_centers * 0.2))

            best_r_squared = -1
            best_filtered_centers = None
            best_keep_mask = None

            for clusters_to_remove in itertools.combinations(range(n_centers), remove_count):
                keep_mask = np.ones(n_centers, dtype=bool)
                keep_mask[list(clusters_to_remove)] = False

                current_centers = cluster_centers[keep_mask]

                if len(current_centers) > 1:
                    s, i, r, _, se = stats.linregress(current_centers[:, 0], current_centers[:, 1])
                    current_r_squared = r**2

                    if current_r_squared > best_r_squared:
                        best_r_squared = current_r_squared
                        best_filtered_centers = current_centers
                        best_keep_mask = keep_mask

            if best_filtered_centers is not None:
                filtered_cluster_centers = best_filtered_centers
                filtered_cluster_ids = np.arange(n_clusters)[best_keep_mask]
                # Recompute regression on filtered centers
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    filtered_cluster_centers[:, 0], filtered_cluster_centers[:, 1]
                )
            else:
                filtered_cluster_centers = cluster_centers
                filtered_cluster_ids = np.arange(n_clusters)
        else:
            filtered_cluster_centers = cluster_centers
            filtered_cluster_ids = np.arange(n_clusters)

        r_squared = r_value**2

        return (
            clusters_full,
            filtered_cluster_centers,
            filtered_cluster_ids,
            slope,
            intercept,
            r_squared,
            std_err,
        )

    # ------------------------------------------------------------------
    # Profile-Based Analysis (Strategy 2)
    # ------------------------------------------------------------------

    def compute_profile(
        self, df: pd.DataFrame, col: str, additional_col: Optional[str], window_sizes: List[int], best_offset: int
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Compute max average speeds/HRs for different window sizes."""
        max_avg_values = {}
        max_avg_additional_values = {}

        for window_size in window_sizes:
            rolling_avg = df[col].rolling(window=window_size, min_periods=1, center=True).mean()
            max_avg = rolling_avg.max()
            max_avg_idx = rolling_avg.idxmax()

            max_avg_values[window_size] = max_avg

            if additional_col is not None:
                if pd.notna(max_avg_idx):
                    half_window = window_size // 2
                    start_idx = max(0, int(max_avg_idx) - half_window)
                    end_idx = min(len(df), int(max_avg_idx) + half_window + 1)
                    additional_avg = df.loc[start_idx:end_idx, additional_col].mean()
                else:
                    additional_avg = None
                max_avg_additional_values[window_size] = additional_avg

        return max_avg_values, max_avg_additional_values

    def profile_based_analysis(
        self, df: pd.DataFrame, best_offset: int, window_sizes: Optional[List[int]] = None
    ) -> Tuple[float, float, float, float]:
        """Perform profile-based HR vs Speed analysis."""
        if window_sizes is None:
            window_sizes = [
                15, 20, 30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020,
                1080, 1140, 1200, 1320, 1440, 1560, 1680, 1920, 2160, 2400, 2640, 2880, 3120, 3360, 3600,
            ]

        df = df.copy()

        # Filter and smooth
        df = df[df["hr"] > 120].copy()
        if df.empty:
            return 0.0, 0.0, 0.0, 0.0

        df = self.moving_average(df, window_size=10, col="hr")
        df = self.moving_average(df, window_size=10, col="speed_km_h")
        df.rename(columns={"hr_ma_10": "hr_smooth", "speed_km_h_ma_10": "speed_smooth"}, inplace=True)

        # Apply shift
        df["hr_shifted"] = df["hr_smooth"].shift(best_offset)

        # Compute profiles
        max_avg_speeds, max_avg_additional_values = self.compute_profile(
            df, "speed_smooth", "hr_shifted", window_sizes, best_offset
        )
        max_avg_hrs, _ = self.compute_profile(df, "hr_shifted", None, window_sizes, best_offset)

        # Linear regression on profile data
        x_values = list(max_avg_hrs.values())
        y_values = list(max_avg_speeds.values())

        if len(x_values) < 2:
            return 0.0, 0.0, 0.0, 0.0

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        r_squared = r_value**2

        return slope, intercept, r_squared, std_err

    # ------------------------------------------------------------------
    # Full Processing
    # ------------------------------------------------------------------

    def process_timeseries(
        self, activity_id: str, strategy: str = "cluster", n_clusters: Optional[int] = None
    ) -> Optional[SpeedProfileResult]:
        """Process a full timeseries file and return analysis results."""
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
        path = self.metrics_ts_dir / f"{activity_id}.csv"

        # Prepare data for saving - ensure all arrays have same length
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

    def load_metrics_ts(self, activity_id: str) -> Optional[pd.DataFrame]:
        """Load processed metrics from metrics_ts folder."""
        path = self.metrics_ts_dir / f"{activity_id}.csv"
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

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
        # Clamp grade to valid range [-0.5, 0.5]
        if grade >= 0.5:
            grade = 0.5
        elif grade <= -0.5:
            grade = -0.5
        
        return (
            280.5 * grade**5 -
            58.7 * grade**4 -
            76.8 * grade**3 +
            51.9 * grade**2 +
            19.6 * grade +
            2.5
        )

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
        # Clamp grade to valid range [-0.5, 0.5]
        if grade >= 0.5:
            grade = 0.5
        elif grade <= -0.5:
            grade = -0.5
        
        return (
            155.4 * grade**5 -
            30.4 * grade**4 -
            43.3 * grade**3 +
            46.3 * grade**2 +
            19.5 * grade +
            3.6
        )

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
        df = df.copy()
        
        # Constants for energy cost on flat ground
        ENERGY_COST_WALKING_FLAT = 2.5
        ENERGY_COST_RUNNING_FLAT = 3.6
        WALKING_THRESHOLD = 6.0  # km/h
        
        # Ensure grade_ma_10 exists (should be computed by preprocess_timeseries)
        if "grade_ma_10" not in df.columns:
            if "grade" in df.columns:
                df = SpeedProfileService.moving_average(df, window_size=10, col="grade")
            else:
                # No grade data, treat as flat (grade = 0)
                df["grade_ma_10"] = 0.0
        
        def calculate_speed_eq(row: pd.Series) -> float:
            speed = row.get("speed_km_h", np.nan)
            grade = row.get("grade_ma_10", 0.0)
            
            if pd.isna(speed) or speed <= 0:
                return np.nan
            
            # Handle NaN grade as flat (0.0)
            if pd.isna(grade):
                grade = 0.0
            
            # Classify as walking or running
            if speed <= WALKING_THRESHOLD:
                cost_at_grade = SpeedProfileService.minetti_energy_cost_walking(grade)
                cost_flat = ENERGY_COST_WALKING_FLAT
            else:
                cost_at_grade = SpeedProfileService.minetti_energy_cost_running(grade)
                cost_flat = ENERGY_COST_RUNNING_FLAT
            
            # Calculate equivalent speed: speed_eq = speed * (C(grade) / C(0))
            if cost_flat <= 0:
                return speed  # Fallback to original speed
            
            speed_eq = max(0.0, speed * (cost_at_grade / cost_flat))
            return speed_eq
        
        df["speed_eq_km_h"] = df.apply(calculate_speed_eq, axis=1)
        return df

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
        df = df.copy()
        
        # Ensure speed_km_h is smoothed
        if "speed_km_h_ma_10" not in df.columns:
            df = self.moving_average(df, window_size=10, col="speed_km_h")
            df = df.rename(columns={"speed_km_h_ma_10": "speed_km_h_smooth"})
        else:
            df = df.rename(columns={"speed_km_h_ma_10": "speed_km_h_smooth"})
        
        # Ensure grade_ma_10 exists for speed_eq computation
        if "grade_ma_10" not in df.columns:
            if "grade" in df.columns:
                df = self.moving_average(df, window_size=10, col="grade")
            else:
                df["grade_ma_10"] = 0.0
        
        # Compute speed_eq_km_h
        df = self.compute_speed_eq_column(df)
        
        # Compute max profiles for each window size
        results = []
        for window_sec in window_sizes:
            # Rolling mean for raw speed (centered)
            rolling_speed = df["speed_km_h_smooth"].rolling(
                window=window_sec, min_periods=1, center=True
            ).mean()
            max_speed = rolling_speed.max()
            
            # Rolling mean for equivalent speed (centered)
            rolling_speed_eq = df["speed_eq_km_h"].rolling(
                window=window_sec, min_periods=1, center=True
            ).mean()
            max_speed_eq = rolling_speed_eq.max()
            
            results.append({
                "windowSec": window_sec,
                "maxSpeedKmh": float(max_speed) if not pd.isna(max_speed) else 0.0,
                "maxSpeedEqKmh": float(max_speed_eq) if not pd.isna(max_speed_eq) else 0.0,
            })
        
        return pd.DataFrame(results)

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
        path = self.speed_profile_dir / f"{activity_id}.csv"
        profile_df.to_csv(path, index=False)
        return path

    def load_speed_profile(self, activity_id: str) -> Optional[pd.DataFrame]:
        """
        Load speed profile from CSV file.
        
        Parameters:
        activity_id (str): Activity identifier
        
        Returns:
        Optional[pd.DataFrame]: Profile DataFrame or None if not found
        """
        path = self.speed_profile_dir / f"{activity_id}.csv"
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

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
        if window_sizes is None:
            window_sizes = config.PROFILE_WINDOW_SIZES
        
        # Check if profile already exists
        existing_profile = self.load_speed_profile(activity_id)
        if existing_profile is not None:
            return self.speed_profile_dir / f"{activity_id}.csv"
        
        # Load timeseries
        timeseries_path = self.config.timeseries_dir / f"{activity_id}.csv"
        if not timeseries_path.exists():
            return None
        
        try:
            df = pd.read_csv(timeseries_path)
        except Exception:
            return None
        
        if df.empty:
            return None
        
        # Preprocess timeseries
        df_processed = self.preprocess_timeseries(df)
        
        # If preprocessing failed (no GPS data), fall back to using paceKmh directly
        if df_processed.empty or "speed_km_h" not in df_processed.columns:
            if "paceKmh" not in df.columns:
                return None
            # Use paceKmh directly - minimal preprocessing
            df_processed = df.copy()
            df_processed["speed_km_h"] = pd.to_numeric(df_processed["paceKmh"], errors="coerce")
            df_processed = df_processed.dropna(subset=["speed_km_h"])
            df_processed = df_processed[df_processed["speed_km_h"] > 0]
            df_processed = self.time_from_timestamp(df_processed)
            df_processed = self.duration(df_processed)
            # Filter unrealistic speeds
            df_processed = df_processed[df_processed["speed_km_h"] < 40].reset_index(drop=True)
            # Add grade column (0.0 if missing)
            if "grade" not in df_processed.columns:
                df_processed["grade"] = 0.0
        
        if df_processed.empty or "speed_km_h" not in df_processed.columns:
            return None
        
        # Compute max speed profiles
        profile_df = self.compute_max_speed_profiles(df_processed, window_sizes)
        
        # Save profile
        return self.save_speed_profile(activity_id, profile_df)

