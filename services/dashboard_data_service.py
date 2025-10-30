"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Data loading service for Dashboard page.

Handles loading and preprocessing of dashboard-specific data.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.cluster import KMeans

from persistence.csv_storage import CsvStorage
from services.speed_profile_service import SpeedProfileService
from utils.config import Config


def load_daily_metrics(storage: CsvStorage, athlete_id: str) -> pd.DataFrame:
    """Load and preprocess daily metrics for an athlete.

    Args:
        storage: CsvStorage instance
        athlete_id: Athlete ID to filter by

    Returns:
        Preprocessed DataFrame with daily metrics
    """
    df = storage.read_csv("daily_metrics.csv")
    if df.empty:
        return df
    df = df[df.get("athleteId") == athlete_id]
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    numeric_columns = [
        "distanceKm",
        "distanceEqKm",
        "timeSec",
        "trimp",
        "ascentM",
        "acuteDistanceKm",
        "chronicDistanceKm",
        "acuteDistanceEqKm",
        "chronicDistanceEqKm",
        "acuteTimeSec",
        "chronicTimeSec",
        "acuteTrimp",
        "chronicTrimp",
        "acuteAscentM",
        "chronicAscentM",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df = df.sort_values("date")
    return df.reset_index(drop=True)


def load_hr_speed_data(
    storage: CsvStorage,
    speed_profile_service: SpeedProfileService,
    config: Config,
    athlete_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    categories: List[str],
    min_cluster_percent: float = 5.0,
) -> Tuple[pd.DataFrame, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Load and preprocess HR vs Speed data for all activities in date range.

    Returns cluster centers with their standard deviations instead of all data points.

    Args:
        storage: CsvStorage instance
        speed_profile_service: SpeedProfileService instance
        config: Config instance
        athlete_id: Athlete ID to filter by
        start_date: Start date for filtering activities
        end_date: End date for filtering activities
        categories: List of activity categories to include
        min_cluster_percent: Minimum percentage of total points a cluster must contain
            to be included (default: 5.0%)

    Returns:
        Tuple of (centers_df, slope, intercept, r_squared, std_err)
    """
    from graph.hr_speed import compute_weighted_regression

    # Get activities in date range
    acts_df = storage.read_csv("activities_metrics.csv")
    if acts_df.empty:
        return pd.DataFrame(), None, None, None, None

    acts_df = acts_df[acts_df.get("athleteId") == athlete_id].copy()
    acts_df["date"] = pd.to_datetime(acts_df.get("startDate"), errors="coerce").dt.normalize()
    mask = (acts_df["date"] >= pd.Timestamp(start_date)) & (
        acts_df["date"] <= pd.Timestamp(end_date)
    )
    acts_df = acts_df[mask]

    # Apply category filter
    if categories:
        acts_df["category"] = (
            acts_df.get("category", pd.Series(dtype=str)).astype(str).str.upper()
        )
        acts_df = acts_df[acts_df["category"].isin([c.upper() for c in categories])]

    # Get activities info to check for timeseries
    activities_info = storage.read_csv("activities.csv")
    if not activities_info.empty:
        activities_info = activities_info[activities_info.get("athleteId") == athlete_id].copy()
        activities_info["activityId"] = activities_info["activityId"].astype(str)
        acts_df["activityId"] = acts_df["activityId"].astype(str)
        acts_df = acts_df.merge(
            activities_info[["activityId", "hasTimeseries"]],
            on="activityId",
            how="left",
        )
        # Filter to activities with timeseries
        acts_df = acts_df[acts_df.get("hasTimeseries", pd.Series(dtype=bool)) == True]  # noqa: E712
    else:
        # If no activities info, check if timeseries file exists
        acts_df = acts_df.copy()
        acts_df["activityId"] = acts_df["activityId"].astype(str)
        acts_df["hasTimeseries"] = acts_df["activityId"].apply(
            lambda aid: (config.timeseries_dir / f"{aid}.csv").exists()
        )
        acts_df = acts_df[acts_df["hasTimeseries"] == True]  # noqa: E712

    if acts_df.empty:
        return pd.DataFrame(), None, None, None, None

    # Collect all HR and Speed data
    all_data = []

    for _, row in acts_df.iterrows():
        activity_id = str(row.get("activityId", ""))
        if not activity_id:
            continue

        # Try to load precomputed metrics_ts first
        metrics_ts = speed_profile_service.load_metrics_ts(activity_id)
        if metrics_ts is None or metrics_ts.empty:
            # Process timeseries if not precomputed
            result = speed_profile_service.process_timeseries(activity_id, strategy="cluster")
            if result is None:
                continue
            # Save for future use
            speed_profile_service.save_metrics_ts(activity_id, result)
            # Extract data from result
            if result.hr_shifted is not None and result.speed_smooth is not None:
                df_act = pd.DataFrame(
                    {
                        "hr": result.hr_shifted.values,
                        "speed": result.speed_smooth.values,
                        "activityId": activity_id,
                    }
                )
                if result.clusters is not None and len(result.clusters) == len(df_act):
                    df_act["cluster"] = result.clusters
                all_data.append(df_act)
        else:
            # Use precomputed data
            if "hr_shifted" in metrics_ts.columns and "speed_smooth" in metrics_ts.columns:
                df_act = pd.DataFrame(
                    {
                        "hr": metrics_ts["hr_shifted"].values,
                        "speed": metrics_ts["speed_smooth"].values,
                        "activityId": activity_id,
                    }
                )
                if "cluster" in metrics_ts.columns and len(metrics_ts["cluster"]) == len(df_act):
                    df_act["cluster"] = metrics_ts["cluster"].values
                all_data.append(df_act)

    if not all_data:
        return pd.DataFrame(), None, None, None, None

    # Compute cluster centers for each activity separately
    n_clusters = config.n_cluster
    cluster_centers_list = []

    for df_act in all_data:
        activity_id = df_act["activityId"].iloc[0]
        df_act = df_act.dropna(subset=["hr", "speed"])

        if df_act.empty:
            continue

        total_points = len(df_act)
        min_points_per_cluster = max(1, int(total_points * min_cluster_percent / 100.0))
        min_allowed_points = max(
            min_points_per_cluster, 20
        )  # At least 20 points or percentage threshold

        # Prepare data for clustering (use individual activity's data)
        X = df_act[["hr", "speed"]].values

        if len(X) < n_clusters:
            continue

        # Perform KMeans clustering for this activity
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        df_act["cluster"] = clusters

        # Compute cluster centers and standard deviations for this activity
        # Filter out clusters with less than min_cluster_percent of total points or less than 20 points
        for cluster_id in range(n_clusters):
            cluster_data = df_act[df_act["cluster"] == cluster_id]
            cluster_count = len(cluster_data)

            # Skip clusters that are too small (outliers)
            if cluster_count < min_allowed_points:
                continue

            if cluster_count > 0:
                hr_mean = cluster_data["hr"].mean()
                speed_mean = cluster_data["speed"].mean()

                # Skip clusters with mean speed less than 6 km/h
                if speed_mean < 6:
                    continue

                hr_std = cluster_data["hr"].std()
                speed_std = cluster_data["speed"].std()

                cluster_centers_list.append(
                    {
                        "hr": hr_mean,
                        "speed": speed_mean,
                        "hr_std": hr_std if not pd.isna(hr_std) else 0.0,
                        "speed_std": speed_std if not pd.isna(speed_std) else 0.0,
                        "cluster": cluster_id,
                        "activityId": activity_id,
                        "count": cluster_count,
                    }
                )

    if not cluster_centers_list:
        return pd.DataFrame(), None, None, None, None

    centers_df = pd.DataFrame(cluster_centers_list)

    # Load activity names and dates from activities.csv
    activities_info = storage.read_csv("activities.csv")
    if not activities_info.empty:
        activities_info = activities_info[activities_info.get("athleteId") == athlete_id].copy()
        activities_info["activityId"] = activities_info["activityId"].astype(str)
        centers_df["activityId"] = centers_df["activityId"].astype(str)

        # Merge name and startTime
        merge_cols = ["activityId"]
        if "name" in activities_info.columns:
            merge_cols.append("name")
        if "startTime" in activities_info.columns:
            merge_cols.append("startTime")

        centers_df = centers_df.merge(activities_info[merge_cols], on="activityId", how="left")

        # Format date for display
        if "startTime" in centers_df.columns:
            centers_df["activity_date"] = pd.to_datetime(
                centers_df["startTime"], errors="coerce"
            )
            centers_df["activity_date_str"] = centers_df["activity_date"].dt.strftime("%Y-%m-%d")
        else:
            centers_df["activity_date_str"] = ""

        # Ensure name exists
        if "name" not in centers_df.columns:
            centers_df["name"] = ""
        centers_df["name"] = centers_df["name"].fillna("").astype(str)

        # Ensure date_str exists
        if "activity_date_str" not in centers_df.columns:
            centers_df["activity_date_str"] = ""
        centers_df["activity_date_str"] = centers_df["activity_date_str"].fillna("").astype(str)
    else:
        # If no activities info, create empty columns
        centers_df["name"] = ""
        centers_df["activity_date_str"] = ""

    # Compute weighted linear regression on all cluster centers
    slope, intercept, r_squared, std_err = compute_weighted_regression(centers_df)
    return centers_df, slope, intercept, r_squared, std_err

