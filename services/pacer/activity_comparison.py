"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd
from haversine import haversine
from streamlit.logger import get_logger

from persistence.repositories import RacePacingLinksRepo
from services.planner_service import PlannerService
from services.speed_profile_service import SpeedProfileService

logger = get_logger(__name__)


class ActivityComparison:
    """Comparison helpers between planned race segments and actual activity data."""

    def __init__(
        self,
        load_race: Callable[
            [str], Optional[tuple[str, list[float], pd.DataFrame, Optional[list[float]]]]
        ],
        planner: PlannerService,
        speed_profile: SpeedProfileService,
        links_repo: RacePacingLinksRepo,
    ) -> None:
        self.load_race = load_race
        self.planner = planner
        self.speed_profile = speed_profile
        self.links_repo = links_repo

    def compare_race_segments_with_activity(
        self, race_id: str, activity_timeseries_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Compare planned race segments with actual activity performance."""
        loaded = self.load_race(race_id)
        if not loaded:
            return None

        _race_name, _aid_stations_km, planned_segments_df, _aid_stations_times = loaded

        if activity_timeseries_df.empty or planned_segments_df.empty:
            return None

        if "cumulated_distance" not in activity_timeseries_df.columns:
            if "lat" in activity_timeseries_df.columns and "lon" in activity_timeseries_df.columns:
                preprocessed = self.speed_profile.preprocess_timeseries(activity_timeseries_df.copy())
                if not preprocessed.empty and "cumulated_distance" in preprocessed.columns:
                    activity_timeseries_df = preprocessed
                else:
                    activity_timeseries_df = self.speed_profile.distance(
                        activity_timeseries_df, lat_col="lat", lon_col="lon"
                    )
                    activity_timeseries_df = self.speed_profile.cumulated_distance(
                        activity_timeseries_df, distance_col="distance"
                    )
            else:
                return None

        has_timestamp = "timestamp" in activity_timeseries_df.columns
        if has_timestamp and "cumulated_duration_seconds" not in activity_timeseries_df.columns:
            if activity_timeseries_df["timestamp"].dtype == "object":
                activity_timeseries_df = self.speed_profile.time_from_timestamp(
                    activity_timeseries_df, timestamp_col="timestamp"
                )
            activity_timeseries_df = self.speed_profile.duration(
                activity_timeseries_df, timestamp_col="timestamp"
            )

        has_pace = "paceKmh" in activity_timeseries_df.columns
        if not has_pace and "speed_km_h" in activity_timeseries_df.columns:
            activity_timeseries_df["paceKmh"] = activity_timeseries_df["speed_km_h"]
            has_pace = True

        has_gps = "lat" in activity_timeseries_df.columns and "lon" in activity_timeseries_df.columns
        if not has_gps:
            logger.warning("GPS data required for GPS-based segment matching")
            return None

        comparison_rows = []

        for _, planned_seg in planned_segments_df.iterrows():
            seg_start_km = float(planned_seg["startKm"])
            seg_end_km = float(planned_seg["endKm"])

            actual_start_idx = None
            actual_end_idx = None

            planned_start_lat = planned_seg.get("startLat")
            planned_start_lon = planned_seg.get("startLon")
            planned_end_lat = planned_seg.get("endLat")
            planned_end_lon = planned_seg.get("endLon")

            if (
                has_gps
                and planned_start_lat is not None
                and pd.notna(planned_start_lat)
                and planned_start_lon is not None
                and pd.notna(planned_start_lon)
                and planned_end_lat is not None
                and pd.notna(planned_end_lat)
                and planned_end_lon is not None
                and pd.notna(planned_end_lon)
            ):
                valid_gps_mask = (
                    activity_timeseries_df["lat"].notna() & activity_timeseries_df["lon"].notna()
                )
                valid_gps_df = activity_timeseries_df[valid_gps_mask].copy()

                if not valid_gps_df.empty:
                    start_distances = valid_gps_df.apply(
                        lambda row: haversine(
                            (float(planned_start_lat), float(planned_start_lon)),
                            (float(row["lat"]), float(row["lon"])),
                        ),
                        axis=1,
                    )
                    end_distances = valid_gps_df.apply(
                        lambda row: haversine(
                            (float(planned_end_lat), float(planned_end_lon)),
                            (float(row["lat"]), float(row["lon"])),
                        ),
                        axis=1,
                    )

                    actual_start_idx = valid_gps_df.index[start_distances.idxmin()]
                    actual_end_idx = valid_gps_df.index[end_distances.idxmin()]

                    if actual_end_idx <= actual_start_idx:
                        actual_start_idx = None
                        actual_end_idx = None

            if actual_start_idx is None or actual_end_idx is None:
                seg_points = activity_timeseries_df[
                    (activity_timeseries_df["cumulated_distance"] >= seg_start_km)
                    & (activity_timeseries_df["cumulated_distance"] <= seg_end_km)
                ].copy()

                if seg_points.empty:
                    comparison_rows.append(
                        {
                            "segmentId": int(planned_seg["segmentId"]),
                            "type": planned_seg.get("type", "unknown"),
                            "startKm": seg_start_km,
                            "endKm": seg_end_km,
                            "plannedTimeSec": float(planned_seg.get("timeSec", 0) or 0),
                            "plannedSpeedKmh": float(planned_seg.get("speedKmh", 0) or 0),
                            "plannedSpeedEqKmh": float(planned_seg.get("speedEqKmh", 0) or 0),
                            "actualTimeSec": None,
                            "actualSpeedKmh": None,
                            "actualSpeedEqKmh": None,
                            "timeDeltaSec": None,
                            "speedDeltaKmh": None,
                            "speedEqDeltaKmh": None,
                        }
                    )
                    continue

                actual_start_idx = seg_points.index[0]
                actual_end_idx = seg_points.index[-1]

            actual_seg_mask = (activity_timeseries_df.index >= actual_start_idx) & (
                activity_timeseries_df.index <= actual_end_idx
            )
            seg_points = activity_timeseries_df[actual_seg_mask].copy()

            if seg_points.empty:
                comparison_rows.append(
                    {
                        "segmentId": int(planned_seg["segmentId"]),
                        "type": planned_seg.get("type", "unknown"),
                        "startKm": seg_start_km,
                        "endKm": seg_end_km,
                        "plannedTimeSec": float(planned_seg.get("timeSec", 0) or 0),
                        "plannedSpeedKmh": float(planned_seg.get("speedKmh", 0) or 0),
                        "plannedSpeedEqKmh": float(planned_seg.get("speedEqKmh", 0) or 0),
                        "actualTimeSec": None,
                        "actualSpeedKmh": None,
                        "actualSpeedEqKmh": None,
                        "timeDeltaSec": None,
                        "speedDeltaKmh": None,
                        "speedEqDeltaKmh": None,
                    }
                )
                continue

            actual_distance_km = 0.0
            if has_gps and "lat" in seg_points.columns and "lon" in seg_points.columns:
                valid_points = seg_points[seg_points["lat"].notna() & seg_points["lon"].notna()].copy()
                if len(valid_points) > 1:
                    distances = [
                        haversine(
                            (float(valid_points.iloc[i]["lat"]), float(valid_points.iloc[i]["lon"])),
                            (
                                float(valid_points.iloc[i + 1]["lat"]),
                                float(valid_points.iloc[i + 1]["lon"]),
                            ),
                        )
                        for i in range(len(valid_points) - 1)
                    ]
                    actual_distance_km = sum(distances)
                elif "cumulated_distance" in seg_points.columns:
                    actual_distance_km = (
                        float(seg_points["cumulated_distance"].iloc[-1])
                        - float(seg_points["cumulated_distance"].iloc[0])
                    )
            elif "cumulated_distance" in seg_points.columns:
                actual_distance_km = (
                    float(seg_points["cumulated_distance"].iloc[-1])
                    - float(seg_points["cumulated_distance"].iloc[0])
                )
            else:
                actual_distance_km = seg_end_km - seg_start_km

            actual_time_sec = None
            if has_timestamp and "cumulated_duration_seconds" in seg_points.columns:
                time_start = float(seg_points["cumulated_duration_seconds"].iloc[0])
                time_end = float(seg_points["cumulated_duration_seconds"].iloc[-1])
                actual_time_sec = time_end - time_start

            actual_speed_kmh = None
            if has_pace and actual_time_sec and actual_time_sec > 0:
                avg_pace = seg_points["paceKmh"].mean()
                if pd.notna(avg_pace):
                    actual_speed_kmh = float(avg_pace)
            elif actual_time_sec and actual_time_sec > 0:
                actual_speed_kmh = (actual_distance_km / actual_time_sec) * 3600

            actual_speed_eq_kmh = None
            actual_elev_gain = 0.0
            if "elevationM" in seg_points.columns:
                elev_diffs = seg_points["elevationM"].diff().fillna(0)
                actual_elev_gain = float(elev_diffs[elev_diffs > 0].sum())

            if actual_time_sec and actual_time_sec > 0:
                actual_distance_eq_km = self.planner.compute_distance_eq_km(
                    actual_distance_km, actual_elev_gain
                )
                actual_speed_eq_kmh = (actual_distance_eq_km / actual_time_sec) * 3600

            planned_time_sec = float(planned_seg.get("timeSec", 0) or 0)
            planned_speed_kmh = float(planned_seg.get("speedKmh", 0) or 0)
            planned_speed_eq_kmh = float(planned_seg.get("speedEqKmh", 0) or 0)

            time_delta = actual_time_sec - planned_time_sec if actual_time_sec is not None else None
            speed_delta = actual_speed_kmh - planned_speed_kmh if actual_speed_kmh is not None else None
            speed_eq_delta = (
                actual_speed_eq_kmh - planned_speed_eq_kmh if actual_speed_eq_kmh is not None else None
            )

            actual_start_km = (
                float(seg_points["cumulated_distance"].iloc[0])
                if "cumulated_distance" in seg_points.columns
                else seg_start_km
            )
            actual_end_km = (
                float(seg_points["cumulated_distance"].iloc[-1])
                if "cumulated_distance" in seg_points.columns
                else seg_end_km
            )

            comparison_rows.append(
                {
                    "segmentId": int(planned_seg["segmentId"]),
                    "type": planned_seg.get("type", "unknown"),
                    "startKm": actual_start_km,
                    "endKm": actual_end_km,
                    "plannedTimeSec": planned_time_sec,
                    "plannedSpeedKmh": planned_speed_kmh,
                    "plannedSpeedEqKmh": planned_speed_eq_kmh,
                    "actualTimeSec": actual_time_sec,
                    "actualSpeedKmh": actual_speed_kmh,
                    "actualSpeedEqKmh": actual_speed_eq_kmh,
                    "timeDeltaSec": time_delta,
                    "speedDeltaKmh": speed_delta,
                    "speedEqDeltaKmh": speed_eq_delta,
                }
            )

        return pd.DataFrame(comparison_rows)

    def link_race_to_activity(self, activity_id: str, race_id: str) -> str:
        """Link a race pacing to an activity."""
        existing_links = self.links_repo.list(activityId=activity_id)
        if not existing_links.empty:
            link_id = existing_links.iloc[0]["linkId"]
            self.links_repo.update(link_id, {"raceId": race_id})
            return link_id
        return self.links_repo.create({"activityId": activity_id, "raceId": race_id})

    def unlink_race_from_activity(self, activity_id: str) -> None:
        """Unlink race pacing from an activity."""
        existing_links = self.links_repo.list(activityId=activity_id)
        if not existing_links.empty:
            for _, link in existing_links.iterrows():
                self.links_repo.delete(link["linkId"])

    def get_linked_race_id(self, activity_id: str) -> Optional[str]:
        """Get the race ID linked to an activity."""
        activity_id_str = str(activity_id)
        existing_links = self.links_repo.list(activityId=activity_id_str)
        if not existing_links.empty:
            race_id = existing_links.iloc[0].get("raceId")
            if race_id and pd.notna(race_id) and str(race_id).strip():
                return str(race_id).strip()
        return None
