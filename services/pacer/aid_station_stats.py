"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import pandas as pd


class AidStationStats:
    """Aid station statistics and timing helpers for pacer workflows."""

    def compute_segment_stats_between(
        self, start_km: float, end_km: float, segments_df: pd.DataFrame
    ) -> dict[str, float]:
        """Compute statistics (distance, dist-eq, elevation gain/loss) between two points."""
        stats = {
            "distanceKm": 0.0,
            "distanceEqKm": 0.0,
            "elevGainM": 0.0,
            "elevLossM": 0.0,
        }

        if start_km >= end_km:
            return stats

        for _, seg in segments_df.iterrows():
            seg_start = float(seg.get("startKm", 0))
            seg_end = float(seg.get("endKm", 0))
            seg_distance_eq = float(seg.get("distanceEqKm", 0) or 0)
            seg_elev_gain = float(seg.get("elevGainM", 0) or 0)
            seg_elev_loss = float(seg.get("elevLossM", 0) or 0)

            if seg_end <= start_km or seg_start >= end_km:
                continue

            overlap_start = max(seg_start, start_km)
            overlap_end = min(seg_end, end_km)
            overlap_ratio = (
                (overlap_end - overlap_start) / (seg_end - seg_start)
                if (seg_end - seg_start) > 0
                else 1.0
            )

            stats["distanceKm"] += overlap_end - overlap_start
            stats["distanceEqKm"] += seg_distance_eq * overlap_ratio
            stats["elevGainM"] += seg_elev_gain * overlap_ratio
            stats["elevLossM"] += seg_elev_loss * overlap_ratio

        return stats

    def compute_aid_station_stats(
        self, aid_stations_km: list[float], segments_df: pd.DataFrame
    ) -> list[dict[str, float]]:
        """Compute stats for each aid station segment (from previous or start)."""
        sorted_aid_km = sorted(aid_stations_km)
        stats_list = []

        if sorted_aid_km:
            first_stats = self.compute_segment_stats_between(0.0, sorted_aid_km[0], segments_df)
            first_stats["timeSec"] = self._compute_time_between(0.0, sorted_aid_km[0], segments_df)
            stats_list.append(first_stats)

            for i in range(len(sorted_aid_km) - 1):
                segment_stats = self.compute_segment_stats_between(
                    sorted_aid_km[i], sorted_aid_km[i + 1], segments_df
                )
                segment_stats["timeSec"] = self._compute_time_between(
                    sorted_aid_km[i], sorted_aid_km[i + 1], segments_df
                )
                stats_list.append(segment_stats)

        return stats_list

    def compute_cumulative_stats_at_aid_stations(
        self, aid_stations_km: list[float], segments_df: pd.DataFrame
    ) -> list[dict[str, float]]:
        """Compute cumulative statistics at each aid station (from start)."""
        sorted_aid_km = sorted(aid_stations_km)
        cumulative_stats = []

        for aid_km in sorted_aid_km:
            stats = self.compute_segment_stats_between(0.0, aid_km, segments_df)
            cumulative_stats.append(stats)

        return cumulative_stats

    def compute_aid_station_times(
        self, aid_stations_km: list[float], segments_df: pd.DataFrame
    ) -> list[float]:
        """Compute cumulative time at each aid station."""
        aid_times = []
        for aid_km in sorted(aid_stations_km):
            cumulative_time = 0.0

            for _, seg in segments_df.iterrows():
                seg_start = float(seg.get("startKm", 0))
                seg_end = float(seg.get("endKm", 0))
                seg_time = float(seg.get("timeSec", 0) or 0)

                if seg_end <= aid_km:
                    cumulative_time += seg_time
                elif seg_start < aid_km:
                    seg_distance = seg_end - seg_start
                    if seg_distance > 0:
                        partial_distance = aid_km - seg_start
                        partial_ratio = partial_distance / seg_distance
                        cumulative_time += seg_time * partial_ratio
                    else:
                        cumulative_time += seg_time
                    break
                else:
                    break

            aid_times.append(cumulative_time)

        return aid_times

    def _compute_time_between(
        self, start_km: float, end_km: float, segments_df: pd.DataFrame
    ) -> float:
        """Compute time between two points."""
        cumulative_time = 0.0

        if start_km >= end_km:
            return cumulative_time

        for _, seg in segments_df.iterrows():
            seg_start = float(seg.get("startKm", 0))
            seg_end = float(seg.get("endKm", 0))
            seg_time = float(seg.get("timeSec", 0) or 0)

            if seg_end <= start_km or seg_start >= end_km:
                continue

            if seg_end <= end_km and seg_start >= start_km:
                cumulative_time += seg_time
            elif seg_start < end_km and seg_end > start_km:
                overlap_start = max(seg_start, start_km)
                overlap_end = min(seg_end, end_km)
                seg_distance = seg_end - seg_start
                if seg_distance > 0:
                    overlap_distance = overlap_end - overlap_start
                    partial_ratio = overlap_distance / seg_distance
                    cumulative_time += seg_time * partial_ratio

        return cumulative_time
