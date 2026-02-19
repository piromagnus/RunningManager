"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.cluster import KMeans

from services.speed_profile import preprocessing


def compute_hr_speed_shift(
    df: pd.DataFrame,
    hr_col: str = "hr",
    speed_col: str = "speed_km_h",
    min_hr: int = 120,
) -> Tuple[int, float]:
    """Compute optimal HR shift for maximum correlation with speed."""
    df = df.copy()

    df = df[df[hr_col] > min_hr].copy()
    if df.empty:
        return 0, 0.0

    df = preprocessing.moving_average(df, window_size=10, col=hr_col)
    df = preprocessing.moving_average(df, window_size=10, col=speed_col)
    df.rename(columns={"hr_ma_10": "hr_smooth", f"{speed_col}_ma_10": "speed_smooth"}, inplace=True)

    hr_smooth_col = "hr_smooth"
    speed_smooth_col = "speed_smooth"

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


def cluster_based_analysis(
    df: pd.DataFrame,
    best_offset: int,
    n_clusters: int = 7,
    r_threshold: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """Perform cluster-based HR vs Speed analysis."""
    df = df.copy()

    if df.empty or "hr_shifted" not in df.columns or "speed_smooth" not in df.columns:
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0

    df = df.reset_index(drop=True)

    X = df[["hr_shifted", "speed_smooth"]].dropna()
    if len(X) < n_clusters:
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X.values)

    clusters_full = np.full(len(df), -1, dtype=int)
    valid_indices = X.index.values
    valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < len(df))]
    if len(valid_indices) == len(clusters):
        clusters_full[valid_indices] = clusters
    else:
        clusters_full[: len(clusters)] = clusters[: len(clusters_full)]

    cluster_centers = kmeans.cluster_centers_

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

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        cluster_centers[:, 0], cluster_centers[:, 1]
    )

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


def compute_profile(
    df: pd.DataFrame,
    col: str,
    additional_col: Optional[str],
    window_sizes: List[int],
    best_offset: int,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute max average speeds/HRs for different window sizes."""
    max_avg_values: Dict[int, float] = {}
    max_avg_additional_values: Dict[int, float] = {}
    activity_duration_sec = preprocessing.activity_duration_seconds(df)

    for window_size in window_sizes:
        if window_size > activity_duration_sec:
            max_avg_values[window_size] = np.nan
            if additional_col is not None:
                max_avg_additional_values[window_size] = np.nan
            continue

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
    df: pd.DataFrame, best_offset: int, window_sizes: Optional[List[int]] = None
) -> Tuple[float, float, float, float]:
    """Perform profile-based HR vs Speed analysis."""
    if window_sizes is None:
        window_sizes = [
            15,
            20,
            30,
            60,
            120,
            180,
            240,
            300,
            360,
            420,
            480,
            540,
            600,
            660,
            720,
            780,
            840,
            900,
            960,
            1020,
            1080,
            1140,
            1200,
            1320,
            1440,
            1560,
            1680,
            1920,
            2160,
            2400,
            2640,
            2880,
            3120,
            3360,
            3600,
        ]

    df = df.copy()
    df = df[df["hr"] > 120].copy()
    if df.empty:
        return 0.0, 0.0, 0.0, 0.0

    df = preprocessing.moving_average(df, window_size=10, col="hr")
    df = preprocessing.moving_average(df, window_size=10, col="speed_km_h")
    df.rename(columns={"hr_ma_10": "hr_smooth", "speed_km_h_ma_10": "speed_smooth"}, inplace=True)

    df["hr_shifted"] = df["hr_smooth"].shift(best_offset)

    max_avg_speeds, max_avg_additional_values = compute_profile(
        df, "speed_smooth", "hr_shifted", window_sizes, best_offset
    )
    max_avg_hrs, _ = compute_profile(df, "hr_shifted", None, window_sizes, best_offset)

    x_values = list(max_avg_hrs.values())
    y_values = list(max_avg_speeds.values())
    valid_pairs = [(x, y) for x, y in zip(x_values, y_values) if pd.notna(x) and pd.notna(y)]

    if len(valid_pairs) < 2:
        return 0.0, 0.0, 0.0, 0.0

    x_values, y_values = zip(*valid_pairs)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
    r_squared = r_value**2

    return slope, intercept, r_squared, std_err
