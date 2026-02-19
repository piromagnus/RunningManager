"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from streamlit.logger import get_logger

from services.planner_service import PlannerService

logger = get_logger(__name__)


class SegmentMerger:
    """Merge and recompute pacer segments."""

    def __init__(
        self,
        planner: PlannerService,
        compute_avg_grade: Callable[[float, float, float], float],
        classify_grade: Callable[[float, float | None], str],
        recompute_segments: Callable[[pd.DataFrame, pd.DataFrame, set[float]], pd.DataFrame],
    ) -> None:
        self.planner = planner
        self.compute_avg_grade = compute_avg_grade
        self.classify_grade = classify_grade
        self.recompute_segments = recompute_segments

    def merge_small_segments_with_closest(
        self, segments_df: pd.DataFrame, df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        """Step 2: Merge small segments (< 1 km-eq) with closest neighbor."""
        if segments_df.empty:
            return segments_df

        segments_df = segments_df.copy()
        max_iterations = len(segments_df)
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            changed = False

            # Compute distance-equivalent for each segment
            segments_df["distanceEqKm"] = segments_df.apply(
                lambda row: self.planner.compute_distance_eq_km(row["distanceKm"], row["elevGainM"]),
                axis=1,
            )

            # Sort by startKm
            segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
            segments_df["segmentId"] = range(len(segments_df))

            processed_indices = set()
            result_segments = []
            i = 0

            while i < len(segments_df):
                if i in processed_indices:
                    i += 1
                    continue

                seg = segments_df.iloc[i]
                distance_eq = seg["distanceEqKm"]
                is_aid_split = seg.get("isAidSplit", False)

                if distance_eq < 1.0 and not is_aid_split:
                    best_match_idx = None
                    best_gap = float("inf")

                    if distance_eq < 0.3:
                        max_gap = 0.5
                    elif distance_eq < 0.7:
                        max_gap = 0.2
                    else:
                        max_gap = 0.1

                    if i > 0 and (i - 1) not in processed_indices:
                        prev_seg = segments_df.iloc[i - 1]
                        prev_is_aid = prev_seg.get("isAidSplit", False)
                        if not prev_is_aid:
                            gap = abs(prev_seg["endKm"] - seg["startKm"])
                            if gap < max_gap and gap < best_gap:
                                best_gap = gap
                                best_match_idx = i - 1

                    if i < len(segments_df) - 1 and (i + 1) not in processed_indices:
                        next_seg = segments_df.iloc[i + 1]
                        next_is_aid = next_seg.get("isAidSplit", False)
                        if not next_is_aid:
                            gap = abs(seg["endKm"] - next_seg["startKm"])
                            if gap < max_gap and gap < best_gap:
                                best_gap = gap
                                best_match_idx = i + 1

                    if best_match_idx is not None:
                        matched_seg = segments_df.iloc[best_match_idx]
                        if best_match_idx < i:
                            merged = self.merge_two_segments(matched_seg, seg, df)
                            if result_segments:
                                result_segments[-1] = merged
                            else:
                                result_segments.append(merged)
                            processed_indices.add(i)
                            processed_indices.add(best_match_idx)
                            i += 1
                        else:
                            merged = self.merge_two_segments(seg, matched_seg, df)
                            result_segments.append(merged)
                            processed_indices.add(i)
                            processed_indices.add(best_match_idx)
                            i += 2
                        changed = True
                        continue

                    if distance_eq < 0.5:
                        best_match_idx = None
                        best_gap = float("inf")

                        if i > 0 and (i - 1) not in processed_indices:
                            prev_seg = segments_df.iloc[i - 1]
                            prev_is_aid = prev_seg.get("isAidSplit", False)
                            if not prev_is_aid:
                                gap = abs(prev_seg["endKm"] - seg["startKm"])
                                if gap < best_gap and gap < 1.0:
                                    best_gap = gap
                                    best_match_idx = i - 1

                        if i < len(segments_df) - 1 and (i + 1) not in processed_indices:
                            next_seg = segments_df.iloc[i + 1]
                            next_is_aid = next_seg.get("isAidSplit", False)
                            if not next_is_aid:
                                gap = abs(seg["endKm"] - next_seg["startKm"])
                                if gap < best_gap and gap < 1.0:
                                    best_gap = gap
                                    best_match_idx = i + 1

                        if best_match_idx is not None:
                            matched_seg = segments_df.iloc[best_match_idx]
                            if best_match_idx < i:
                                merged = self.merge_two_segments(matched_seg, seg, df)
                                if result_segments:
                                    result_segments[-1] = merged
                                else:
                                    result_segments.append(merged)
                                processed_indices.add(i)
                                processed_indices.add(best_match_idx)
                                i += 1
                            else:
                                merged = self.merge_two_segments(seg, matched_seg, df)
                                result_segments.append(merged)
                                processed_indices.add(i)
                                processed_indices.add(best_match_idx)
                                i += 2
                            changed = True
                            continue

                result_segments.append(seg.to_dict())
                processed_indices.add(i)
                i += 1

            if not changed:
                break

            segments_df = pd.DataFrame(result_segments).reset_index(drop=True)
            segments_df = self.recompute_segments(segments_df, df, aid_stations_km_set)

        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))
        return segments_df

    def merge_contrary_trend_segments(
        self, segments_df: pd.DataFrame, df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        """Step 3: Merge small descents in uphill or small uphills in downhill."""
        if len(segments_df) < 3:
            return segments_df

        segments_df = segments_df.copy()
        max_iterations = len(segments_df)
        iteration = 0

        uphill_types = {"steep_up", "run_up"}
        downhill_types = {"steep_down", "down"}

        while iteration < max_iterations:
            iteration += 1
            changed = False

            segments_df["distanceEqKm"] = segments_df.apply(
                lambda row: self.planner.compute_distance_eq_km(row["distanceKm"], row["elevGainM"]),
                axis=1,
            )

            segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
            segments_df["segmentId"] = range(len(segments_df))

            result_segments = []
            i = 0

            while i < len(segments_df):
                if i >= len(segments_df) - 1:
                    result_segments.append(segments_df.iloc[i].to_dict())
                    break

                seg1 = segments_df.iloc[i]
                seg2 = segments_df.iloc[i + 1]

                if seg1.get("isAidSplit", False) or seg2.get("isAidSplit", False):
                    result_segments.append(seg1.to_dict())
                    i += 1
                    continue

                gap = abs(seg1["endKm"] - seg2["startKm"])
                if gap >= 0.1:
                    result_segments.append(seg1.to_dict())
                    i += 1
                    continue

                should_merge = False
                dist_eq1 = seg1["distanceEqKm"]
                dist_eq2 = seg2["distanceEqKm"]

                window_start = max(0, i - 5)
                window_end = min(len(segments_df), i + 6)
                window_segments = segments_df.iloc[window_start:window_end]

                uphill_count = 0
                downhill_count = 0
                for idx in range(len(window_segments)):
                    window_idx = window_start + idx
                    if window_idx == i or window_idx == i + 1:
                        continue
                    seg_type = window_segments.iloc[idx]["type"]
                    if seg_type in uphill_types:
                        uphill_count += 1
                    elif seg_type in downhill_types:
                        downhill_count += 1

                general_trend = None
                if downhill_count > uphill_count + 2:
                    general_trend = "descent"
                elif uphill_count > downhill_count + 2:
                    general_trend = "climb"

                if general_trend == "climb":
                    if seg1["type"] in downhill_types and dist_eq1 < 0.5:
                        should_merge = True
                    elif seg2["type"] in downhill_types and dist_eq2 < 0.5:
                        should_merge = True

                if general_trend == "descent":
                    if seg1["type"] in uphill_types and dist_eq1 < 0.5:
                        should_merge = True
                    elif seg2["type"] in uphill_types and dist_eq2 < 0.5:
                        should_merge = True

                if should_merge:
                    merged = self.merge_two_segments(seg1, seg2, df)
                    result_segments.append(merged)
                    i += 2
                    changed = True
                else:
                    result_segments.append(seg1.to_dict())
                    i += 1

            if not changed:
                break

            segments_df = pd.DataFrame(result_segments).reset_index(drop=True)
            segments_df = self.recompute_segments(segments_df, df, aid_stations_km_set)

        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))
        return segments_df

    def merge_consecutive_same_type_segments(
        self, segments_df: pd.DataFrame, df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        """Step 4: Merge consecutive segments with same type."""
        if len(segments_df) < 2:
            return segments_df

        segments_df = segments_df.copy()
        max_iterations = len(segments_df)
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            changed = False

            segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
            segments_df["segmentId"] = range(len(segments_df))

            result_segments = []
            i = 0

            while i < len(segments_df):
                if i >= len(segments_df) - 1:
                    result_segments.append(segments_df.iloc[i].to_dict())
                    break

                seg1 = segments_df.iloc[i]
                seg2 = segments_df.iloc[i + 1]

                if seg1.get("isAidSplit", False) or seg2.get("isAidSplit", False):
                    result_segments.append(seg1.to_dict())
                    i += 1
                    continue

                gap = abs(seg1["endKm"] - seg2["startKm"])
                max_gap_same_type = 0.2
                if gap >= max_gap_same_type:
                    result_segments.append(seg1.to_dict())
                    i += 1
                    continue

                if seg1["type"] == seg2["type"]:
                    merged = self.merge_two_segments(seg1, seg2, df)
                    result_segments.append(merged)
                    i += 2
                    changed = True
                else:
                    result_segments.append(seg1.to_dict())
                    i += 1

            if not changed:
                break

            segments_df = pd.DataFrame(result_segments).reset_index(drop=True)
            segments_df = self.recompute_segments(segments_df, df, aid_stations_km_set)

        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))
        return segments_df

    def merge_two_segments(self, seg1: pd.Series, seg2: pd.Series, df: pd.DataFrame) -> dict:
        """Merge two adjacent segments and compute combined metrics."""
        seg1_end = seg1["endKm"]
        seg2_start = seg2["startKm"]

        if abs(seg1_end - seg2_start) > 0.01:
            logger.warning(
                "Attempting to merge non-adjacent segments: seg1 ends at %s, seg2 starts at %s",
                seg1_end,
                seg2_start,
            )

        start_km = min(seg1["startKm"], seg2["startKm"])
        end_km = max(seg1["endKm"], seg2["endKm"])

        combined_df = df[(df["cumulated_distance"] >= start_km) & (df["cumulated_distance"] <= end_km)]

        if combined_df.empty:
            distance_km = end_km - start_km
            elev_gain = seg1["elevGainM"] + seg2["elevGainM"]
            elev_loss = seg1["elevLossM"] + seg2["elevLossM"]
            avg_grade = self.compute_avg_grade(elev_gain, elev_loss, distance_km)
            merged_type = seg1["type"] if seg1["distanceKm"] >= seg2["distanceKm"] else seg2["type"]
        else:
            distance_km = end_km - start_km
            elev_gain = combined_df["elevation_difference"].clip(lower=0).sum()
            elev_loss = combined_df["elevation_difference"].clip(upper=0).abs().sum()
            avg_grade = self.compute_avg_grade(elev_gain, elev_loss, distance_km)

            elevation_delta_per_km = 0.0
            if distance_km > 0:
                elevation_delta_per_km = abs(elev_gain - elev_loss) / distance_km

            merged_type = self.classify_grade(avg_grade, elevation_delta_per_km)

        return {
            "segmentId": 0,
            "type": merged_type,
            "startKm": start_km,
            "endKm": end_km,
            "distanceKm": distance_km,
            "elevGainM": elev_gain,
            "elevLossM": elev_loss,
            "avgGrade": avg_grade,
            "isAidSplit": seg1.get("isAidSplit", False) or seg2.get("isAidSplit", False),
        }
