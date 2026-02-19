"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from streamlit.logger import get_logger

from services.pacer.segment_merger import SegmentMerger
from services.planner_service import PlannerService
from utils.elevation import compute_avg_grade
from utils.grade_classification import classify_grade_pacer_5cat
from utils.segments import merge_adjacent_same_color, merge_small_segments
from utils.time import compute_segment_time

logger = get_logger(__name__)


class SegmentationService:
    """Segmentation and merging helpers for pacer workflows."""

    def __init__(self, planner: PlannerService) -> None:
        self.planner = planner
        self.merger = SegmentMerger(
            planner=planner,
            compute_avg_grade=compute_avg_grade,
            classify_grade=classify_grade_pacer_5cat,
            recompute_segments=self._recompute_and_reclassify_segments,
        )

    def segment_course(
        self,
        df: pd.DataFrame,
        aid_stations_km: list[float],
        min_seg_len_m: int = 150,
        max_splits_per_long_up: int = 5,
    ) -> pd.DataFrame:
        """Segment course by grade and aid stations."""
        if df.empty or "cumulated_distance" not in df.columns or "grade_ma_10" not in df.columns:
            logger.warning("Insufficient data for segmentation")
            return pd.DataFrame()

        df = df.copy().reset_index(drop=True)

        aid_stations_km = sorted([float(x) for x in aid_stations_km if x > 0])
        max_distance = df["cumulated_distance"].max()
        aid_stations_km = [x for x in aid_stations_km if x <= max_distance]
        aid_stations_km_set = set(aid_stations_km)

        segments_df = self._get_initial_segments_with_mean_elevation(df, aid_stations_km_set)
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))

        segments_df = self._merge_small_segments_with_closest(segments_df, df, aid_stations_km_set)
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))

        segments_df = self._merge_contrary_trend_segments(segments_df, df, aid_stations_km_set)
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))

        segments_df = self._merge_consecutive_same_type_segments(segments_df, df, aid_stations_km_set)
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))

        segments_df = self._merge_small_segments_with_closest(segments_df, df, aid_stations_km_set)
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))

        df = self._rebuild_segments_df_with_types(df, segments_df)
        return self._compute_segment_metrics(df, segments_df, aid_stations_km_set)

    def _get_initial_segments_with_mean_elevation(
        self, df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        """Step 1: Get initial segments with mean elevation gain."""
        df["elevation_delta_per_km"] = (
            df["elevationM_ma_5"].diff() / df["cumulated_distance"].diff().replace(0, np.nan) * 1000
        )
        df["elevation_delta_per_km"] = df["elevation_delta_per_km"].fillna(0)

        window_size = min(100, len(df) // 10)
        if window_size > 1:
            dist_diff = df["cumulated_distance"].diff(window_size).replace(0, np.nan)
            elev_diff = df["elevationM_ma_5"].diff(window_size).abs()
            df["cumulated_elevation_delta_per_km"] = (elev_diff / dist_diff * 1000).fillna(0)
        else:
            df["cumulated_elevation_delta_per_km"] = df["elevation_delta_per_km"].abs()

        df["grade_category"] = df.apply(
            lambda row: classify_grade_pacer_5cat(
                row["grade_ma_10"], row.get("cumulated_elevation_delta_per_km", None)
            ),
            axis=1,
        )

        df["segment"] = (df["grade_category"] != df["grade_category"].shift()).cumsum()

        for aid_km in aid_stations_km_set:
            distances = abs(df["cumulated_distance"] - aid_km)
            nearest_idx = distances.idxmin()
            nearest_dist_m = abs(df.loc[nearest_idx, "cumulated_distance"] - aid_km) * 1000

            if nearest_dist_m <= 25:
                df.loc[nearest_idx:, "segment"] = df.loc[nearest_idx:, "segment"] + df["segment"].max() + 1
                df["segment"] = pd.Categorical(df["segment"]).codes

        if len(df) > 0 and df["cumulated_distance"].max() > 0:
            points_per_m = len(df) / (df["cumulated_distance"].max() * 1000)
            min_size = max(3, int(150 * points_per_m))
        else:
            min_size = 3

        df = merge_small_segments(df, min_size=min_size)
        df = merge_adjacent_same_color(df)

        segments_df = self._aggregate_segments(df, aid_stations_km_set)
        return self._recompute_and_reclassify_segments(segments_df, df, aid_stations_km_set)

    def _merge_small_segments_with_closest(
        self, segments_df: pd.DataFrame, df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        return self.merger.merge_small_segments_with_closest(segments_df, df, aid_stations_km_set)

    def _merge_contrary_trend_segments(
        self, segments_df: pd.DataFrame, df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        return self.merger.merge_contrary_trend_segments(segments_df, df, aid_stations_km_set)

    def _merge_consecutive_same_type_segments(
        self, segments_df: pd.DataFrame, df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        return self.merger.merge_consecutive_same_type_segments(segments_df, df, aid_stations_km_set)

    def _merge_two_segments(self, seg1: pd.Series, seg2: pd.Series, df: pd.DataFrame) -> dict:
        return self.merger.merge_two_segments(seg1, seg2, df)

    def _aggregate_segments(self, df: pd.DataFrame, aid_stations_km_set: set[float]) -> pd.DataFrame:
        """Aggregate points into segment summary."""
        segments = []
        unique_segments_with_positions = []
        for seg_id in df["segment"].unique():
            seg_df = df[df["segment"] == seg_id]
            if not seg_df.empty:
                start_pos = seg_df["cumulated_distance"].min()
                unique_segments_with_positions.append((start_pos, seg_id))

        unique_segments_with_positions.sort(key=lambda x: x[0])
        unique_segments = [seg_id for _, seg_id in unique_segments_with_positions]

        for idx, seg_id in enumerate(unique_segments):
            seg_df = df[df["segment"] == seg_id]
            if seg_df.empty:
                continue

            actual_start = seg_df["cumulated_distance"].min()
            actual_end = seg_df["cumulated_distance"].max()

            if idx == 0:
                start_km = 0.0
            else:
                start_km = actual_start

            if idx == len(unique_segments) - 1:
                end_km = df["cumulated_distance"].max()
            else:
                end_km = actual_end

            end_km = max(end_km, start_km)
            distance_km = end_km - start_km

            elev_gain = seg_df["elevation_difference"].clip(lower=0).sum()
            elev_loss = seg_df["elevation_difference"].clip(upper=0).abs().sum()
            avg_grade = compute_avg_grade(elev_gain, elev_loss, distance_km)

            is_aid_split = False
            for aid_km in aid_stations_km_set:
                if abs(start_km - aid_km) < 0.025 or abs(end_km - aid_km) < 0.025:
                    is_aid_split = True
                    break

            segments.append(
                {
                    "segmentId": int(seg_id),
                    "type": seg_df["grade_category"].iloc[0],
                    "startKm": start_km,
                    "endKm": end_km,
                    "distanceKm": distance_km,
                    "elevGainM": elev_gain,
                    "elevLossM": elev_loss,
                    "avgGrade": avg_grade,
                    "isAidSplit": is_aid_split,
                }
            )

        if len(segments) > 1:
            for idx in range(len(segments) - 1):
                segments[idx]["endKm"] = segments[idx + 1]["startKm"]
                segments[idx]["distanceKm"] = segments[idx]["endKm"] - segments[idx]["startKm"]

        return pd.DataFrame(segments)

    def _rebuild_segments_df_with_types(self, df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
        """Rebuild df with updated segment IDs and types."""
        df = df.copy()
        df["segment"] = -1
        df["grade_category"] = "unknown"

        for _, seg in segments_df.iterrows():
            mask = (df["cumulated_distance"] >= seg["startKm"]) & (df["cumulated_distance"] <= seg["endKm"])
            df.loc[mask, "segment"] = seg["segmentId"]
            df.loc[mask, "grade_category"] = seg["type"]

        return df[df["segment"] >= 0].reset_index(drop=True)

    def _recompute_and_reclassify_segments(
        self, segments_df: pd.DataFrame, df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        """Recompute segment metrics and reclassify types after merging/splitting."""
        result_segments = []

        for _, seg in segments_df.iterrows():
            seg_df = df[
                (df["cumulated_distance"] >= seg["startKm"]) & (df["cumulated_distance"] <= seg["endKm"])
            ]

            if seg_df.empty:
                result_segments.append(seg.to_dict())
                continue

            original_start = seg["startKm"]
            original_end = seg["endKm"]

            distance_km = original_end - original_start
            elev_gain = seg_df["elevation_difference"].clip(lower=0).sum() if not seg_df.empty else 0.0
            elev_loss = seg_df["elevation_difference"].clip(upper=0).abs().sum() if not seg_df.empty else 0.0
            avg_grade = compute_avg_grade(elev_gain, elev_loss, distance_km)

            elevation_delta_per_km = 0.0
            if distance_km > 0:
                elevation_delta_per_km = abs(elev_gain - elev_loss) / distance_km

            new_type = classify_grade_pacer_5cat(avg_grade, elevation_delta_per_km)

            is_aid_split = seg.get("isAidSplit", False)
            if not is_aid_split:
                for aid_km in aid_stations_km_set:
                    if abs(original_start - aid_km) < 0.025 or abs(original_end - aid_km) < 0.025:
                        is_aid_split = True
                        break

            result_segments.append(
                {
                    "segmentId": seg["segmentId"],
                    "type": new_type,
                    "startKm": original_start,
                    "endKm": original_end,
                    "distanceKm": distance_km,
                    "elevGainM": elev_gain,
                    "elevLossM": elev_loss,
                    "avgGrade": avg_grade,
                    "isAidSplit": is_aid_split,
                }
            )

        result_df = pd.DataFrame(result_segments).reset_index(drop=True)
        result_df = result_df.sort_values("startKm").reset_index(drop=True)
        result_df["segmentId"] = range(len(result_df))
        return result_df

    def _compute_segment_metrics(
        self, df: pd.DataFrame, segments_df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        """Compute final segment metrics with distance-equivalent and time."""
        segments = []
        segments_df_sorted = segments_df.sort_values("startKm").reset_index(drop=True)

        for _, seg_row in segments_df_sorted.iterrows():
            seg_id = seg_row["segmentId"]
            start_km = seg_row["startKm"]
            end_km = seg_row["endKm"]

            seg_df = df[(df["cumulated_distance"] >= start_km) & (df["cumulated_distance"] <= end_km)]

            distance_km = end_km - start_km

            start_lat = None
            start_lon = None
            end_lat = None
            end_lon = None

            if not seg_df.empty:
                first_point = seg_df.iloc[0]
                last_point = seg_df.iloc[-1]

                if "lat" in seg_df.columns and "lon" in seg_df.columns:
                    start_lat = float(first_point["lat"]) if pd.notna(first_point["lat"]) else None
                    start_lon = float(first_point["lon"]) if pd.notna(first_point["lon"]) else None
                    end_lat = float(last_point["lat"]) if pd.notna(last_point["lat"]) else None
                    end_lon = float(last_point["lon"]) if pd.notna(last_point["lon"]) else None

                elev_gain = seg_df["elevation_difference"].clip(lower=0).sum()
                elev_loss = seg_df["elevation_difference"].clip(upper=0).abs().sum()
                avg_grade = compute_avg_grade(elev_gain, elev_loss, distance_km)
            else:
                elev_gain = seg_row.get("elevGainM", 0.0)
                elev_loss = seg_row.get("elevLossM", 0.0)
                avg_grade = compute_avg_grade(elev_gain, elev_loss, distance_km)

                if "lat" in df.columns and "lon" in df.columns:
                    start_distances = abs(df["cumulated_distance"] - start_km)
                    end_distances = abs(df["cumulated_distance"] - end_km)
                    start_idx = start_distances.idxmin()
                    end_idx = end_distances.idxmin()

                    if pd.notna(df.loc[start_idx, "lat"]) and pd.notna(df.loc[start_idx, "lon"]):
                        start_lat = float(df.loc[start_idx, "lat"])
                        start_lon = float(df.loc[start_idx, "lon"])
                    if pd.notna(df.loc[end_idx, "lat"]) and pd.notna(df.loc[end_idx, "lon"]):
                        end_lat = float(df.loc[end_idx, "lat"])
                        end_lon = float(df.loc[end_idx, "lon"])

            is_aid_split = seg_row.get("isAidSplit", False)
            if not is_aid_split:
                for aid_km in aid_stations_km_set:
                    if abs(start_km - aid_km) < 0.025 or abs(end_km - aid_km) < 0.025:
                        is_aid_split = True
                        break

            distance_eq_km = self.planner.compute_distance_eq_km(distance_km, elev_gain)
            seg_type = seg_row.get("type", "unknown")

            segment_dict = {
                "segmentId": int(seg_id),
                "type": seg_type,
                "startKm": start_km,
                "endKm": end_km,
                "distanceKm": distance_km,
                "elevGainM": elev_gain,
                "elevLossM": elev_loss,
                "avgGrade": avg_grade,
                "isAidSplit": is_aid_split,
                "distanceEqKm": distance_eq_km,
                "speedEqKmh": 0.0,
                "speedKmh": 0.0,
                "timeSec": 0,
            }

            if start_lat is not None:
                segment_dict["startLat"] = start_lat
            if start_lon is not None:
                segment_dict["startLon"] = start_lon
            if end_lat is not None:
                segment_dict["endLat"] = end_lat
            if end_lon is not None:
                segment_dict["endLon"] = end_lon

            segments.append(segment_dict)

        result_df = pd.DataFrame(segments)
        result_df = result_df.sort_values("startKm").reset_index(drop=True)
        result_df["segmentId"] = range(len(result_df))
        return result_df

    def merge_segments_manually(
        self, segments_df: pd.DataFrame, segment_ids: list[int], metrics_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge multiple adjacent segments manually."""
        if len(segment_ids) < 2:
            return segments_df

        segment_ids = sorted(segment_ids)

        segments_to_merge = segments_df[segments_df["segmentId"].isin(segment_ids)].sort_values("startKm")
        if len(segments_to_merge) != len(segment_ids):
            logger.warning("Some segment IDs not found: %s", segment_ids)
            return segments_df

        for i in range(len(segments_to_merge) - 1):
            current_end = segments_to_merge.iloc[i]["endKm"]
            next_start = segments_to_merge.iloc[i + 1]["startKm"]
            if abs(current_end - next_start) > 0.01:
                logger.warning("Segments %s are not adjacent", segment_ids)
                return segments_df

        merged_seg = segments_to_merge.iloc[0].to_dict()
        for i in range(1, len(segments_to_merge)):
            next_seg = segments_to_merge.iloc[i].to_dict()
            merged_seg = self._merge_two_segments(
                pd.Series(merged_seg), pd.Series(next_seg), metrics_df
            )

        aid_stations_km_set = set()
        if "isAidSplit" in segments_to_merge.columns:
            merged_seg["isAidSplit"] = segments_to_merge["isAidSplit"].any()

        if "speedEqKmh" in segments_to_merge.columns:
            total_dist_eq = segments_to_merge["distanceEqKm"].sum()
            if total_dist_eq > 0:
                merged_seg["speedEqKmh"] = (
                    (segments_to_merge["speedEqKmh"] * segments_to_merge["distanceEqKm"]).sum()
                    / total_dist_eq
                )
            else:
                merged_seg["speedEqKmh"] = (
                    segments_to_merge["speedEqKmh"].iloc[0] if len(segments_to_merge) > 0 else 0.0
                )
        else:
            merged_seg["speedEqKmh"] = 0.0

        if "speedKmh" in segments_to_merge.columns:
            total_dist = segments_to_merge["distanceKm"].sum()
            if total_dist > 0:
                merged_seg["speedKmh"] = (
                    (segments_to_merge["speedKmh"] * segments_to_merge["distanceKm"]).sum()
                    / total_dist
                )
            else:
                merged_seg["speedKmh"] = (
                    segments_to_merge["speedKmh"].iloc[0] if len(segments_to_merge) > 0 else 0.0
                )
        else:
            merged_seg["speedKmh"] = 0.0

        result_df = segments_df[~segments_df["segmentId"].isin(segment_ids)].copy()
        result_df = pd.concat([result_df, pd.DataFrame([merged_seg])], ignore_index=True)

        result_df = result_df.sort_values("startKm").reset_index(drop=True)
        result_df["segmentId"] = range(len(result_df))

        merged_segment_ids_original = sorted(segment_ids)
        first_merged_start = segments_df[segments_df["segmentId"] == merged_segment_ids_original[0]][
            "startKm"
        ].iloc[0]
        merged_positions = result_df[(result_df["startKm"] - first_merged_start).abs() < 0.001].index

        merged_idx_for_speeds = None
        if len(merged_positions) > 0:
            merged_idx_for_speeds = merged_positions[0]

        if len(result_df) > 1:
            for idx in range(len(result_df) - 1):
                current_end = result_df.iloc[idx]["endKm"]
                next_start = result_df.iloc[idx + 1]["startKm"]
                gap = next_start - current_end

                if gap > 0.001:
                    result_df.iloc[idx, result_df.columns.get_loc("endKm")] = next_start
                    result_df.iloc[idx, result_df.columns.get_loc("distanceKm")] = (
                        result_df.iloc[idx]["endKm"] - result_df.iloc[idx]["startKm"]
                    )
                elif gap < -0.001:
                    result_df.iloc[idx + 1, result_df.columns.get_loc("startKm")] = current_end
                    result_df.iloc[idx + 1, result_df.columns.get_loc("distanceKm")] = (
                        result_df.iloc[idx + 1]["endKm"] - result_df.iloc[idx + 1]["startKm"]
                    )

        result_df = self._recompute_and_reclassify_segments(result_df, metrics_df, aid_stations_km_set)
        result_df = self._compute_segment_metrics(metrics_df, result_df, aid_stations_km_set)

        if merged_idx_for_speeds is not None and merged_idx_for_speeds < len(result_df):
            current_start = (
                result_df.iloc[merged_idx_for_speeds]["startKm"]
                if merged_idx_for_speeds < len(result_df)
                else None
            )
            if current_start is not None:
                target_mask = (result_df["startKm"] - current_start).abs() < 0.001
                target_indices = result_df[target_mask].index
                if len(target_indices) > 0:
                    target_idx = target_indices[0]
                    result_df.at[target_idx, "speedEqKmh"] = merged_seg.get("speedEqKmh", 0.0)
                    result_df.at[target_idx, "speedKmh"] = merged_seg.get("speedKmh", 0.0)
                    distance_eq = result_df.at[target_idx, "distanceEqKm"]
                    distance = result_df.at[target_idx, "distanceKm"]
                    speed_eq = result_df.at[target_idx, "speedEqKmh"]
                    speed = result_df.at[target_idx, "speedKmh"]
                    result_df.at[target_idx, "timeSec"] = compute_segment_time(
                        distance_eq, distance, speed_eq, speed
                    )

        result_df = result_df.sort_values("startKm").reset_index(drop=True)
        result_df["segmentId"] = range(len(result_df))
        return result_df
