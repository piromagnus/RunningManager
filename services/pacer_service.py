"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Pacer service for race course segmentation and pacing calculations.

Handles time-invariant GPX routes (no timestamps required).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from streamlit.logger import get_logger

from persistence.csv_storage import CsvStorage
from services.planner_service import PlannerService
from services.speed_profile_service import SpeedProfileService
from utils.ids import new_id
from utils.segments import merge_adjacent_same_color, merge_small_segments

logger = get_logger(__name__)


class PacerService:
    """Service for race course segmentation and pacing."""

    def __init__(self, storage: CsvStorage):
        self.storage = storage
        self.planner = PlannerService(storage)
        self.speed_profile = SpeedProfileService(config=None)

    def preprocess_timeseries_for_pacing(self, timeseries_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess timeseries for pacing (time-invariant).

        Handles GPX routes without timestamps. Computes distance and grade
        but skips duration/speed computation.

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
        df = self.speed_profile.distance(df, lat_col="lat", lon_col="lon")

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

        # Apply moving average ONLY to elevation (for noise reduction)
        df = self.speed_profile.moving_average(df, window_size=5, col="elevationM")

        # Compute elevation differences using smoothed elevation
        df = self.speed_profile.elevation(df, elevation_col="elevationM_ma_5")

        # Compute cumulated distance (using original accurate distance)
        df = self.speed_profile.cumulated_distance(df)

        # Apply moving average to distance (for grade calculation smoothing)
        df = self.speed_profile.moving_average(df, window_size=10, col="distance")

        # Compute grade using smoothed distance
        df = self.speed_profile.grade(df, distance_col="distance_ma_10", elevation_col="elevation_difference")

        # Apply moving average to grade
        df = self.speed_profile.moving_average(df, window_size=10, col="grade")

        return df

    def compute_avg_grade(self, elev_gain_m: float, elev_loss_m: float, distance_km: float) -> float:
        """Compute average grade from elevation gain/loss and distance.
        
        Rules:
        - If D+ >= 2 * D-: use D+ / distance
        - If D- >= 2 * D+: use -D- / distance
        - Otherwise: use (D+ - D-) / distance
        
        Args:
            elev_gain_m: Total elevation gain in meters
            elev_loss_m: Total elevation loss in meters
            distance_km: Distance in kilometers
            
        Returns:
            Average grade (decimal, not percentage)
        """
        if distance_km <= 0:
            return 0.0
        
        # If one is at least twice the other, use the dominant one
        if elev_gain_m >= 2 * elev_loss_m:
            # Dominant gain: use D+ / distance
            return elev_gain_m / (distance_km * 1000)
        elif elev_loss_m >= 2 * elev_gain_m:
            # Dominant loss: use -D- / distance
            return -elev_loss_m / (distance_km * 1000)
        else:
            # Similar magnitudes: use net elevation change
            return (elev_gain_m - elev_loss_m) / (distance_km * 1000)

    def classify_grade(
        self, grade: float, cumulated_elevation_delta_per_km: Optional[float] = None
    ) -> Literal["steep_up", "run_up", "flat", "down", "steep_down"]:
        """Classify grade into category.

        Args:
            grade: Grade value (decimal, not percentage)
            cumulated_elevation_delta_per_km: Optional elevation delta per km for flat detection

        Returns:
            Grade category string
        """
        # Check flat first with elevation delta if provided
        if cumulated_elevation_delta_per_km is not None and abs(cumulated_elevation_delta_per_km) < 10.0:
            # Flat if elevation delta < 10m/1km even if grade slightly off
            return "flat"
        
        if grade >= 0.10:
            return "steep_up"
        elif 0.02 <= grade < 0.10:
            return "run_up"
        elif -0.02 < grade < 0.02:
            return "flat"
        elif -0.25 < grade <= -0.02:
            return "down"
        else:  # grade <= -0.25
            return "steep_down"

    def segment_course(
        self,
        df: pd.DataFrame,
        aid_stations_km: list[float],
        min_seg_len_m: int = 150,
        max_splits_per_long_up: int = 5,
    ) -> pd.DataFrame:
        """Segment course by grade and aid stations.

        Simplified 5-step pipeline:
        1. Get segments with mean elevation gain
        2. Merge small segments (< 1 km-eq) with closest neighbor
        3. Merge small descents in uphill or small uphills in downhill
        4. Merge consecutive segments with same type
        5. Final cleanup: merge any remaining small segments (< 1 km-eq)

        Args:
            df: Preprocessed DataFrame with cumulated_distance (km), elevationM_ma_5, grade_ma_10
            aid_stations_km: List of aid station positions in km
            min_seg_len_m: Minimum segment length in meters (unused, kept for compatibility)
            max_splits_per_long_up: Maximum splits for long uphill segments (unused, kept for compatibility)

        Returns:
            DataFrame with segment metrics
        """
        if df.empty or "cumulated_distance" not in df.columns or "grade_ma_10" not in df.columns:
            logger.warning("Insufficient data for segmentation")
            return pd.DataFrame()

        df = df.copy()
        df = df.reset_index(drop=True)

        # Normalize and sort aid stations
        aid_stations_km = sorted([float(x) for x in aid_stations_km if x > 0])
        max_distance = df["cumulated_distance"].max()
        aid_stations_km = [x for x in aid_stations_km if x <= max_distance]
        aid_stations_km_set = set(aid_stations_km)

        # Step 1: Get segments with mean elevation gain
        segments_df = self._get_initial_segments_with_mean_elevation(df, aid_stations_km_set)
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))

        # Step 2: Merge small segments (< 1 km-eq) with closest neighbor
        segments_df = self._merge_small_segments_with_closest(segments_df, df, aid_stations_km_set)
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))

        # Step 3: Merge small descents in uphill or small uphills in downhill
        segments_df = self._merge_contrary_trend_segments(segments_df, df, aid_stations_km_set)
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))

        # Step 4: Merge consecutive segments with same type
        segments_df = self._merge_consecutive_same_type_segments(segments_df, df, aid_stations_km_set)
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))

        # Step 5: Final cleanup - merge any remaining small segments (< 1 km-eq)
        # This catches small segments that may have been created or missed in previous steps
        segments_df = self._merge_small_segments_with_closest(segments_df, df, aid_stations_km_set)
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))

        # Rebuild df with final segment types
        df = self._rebuild_segments_df_with_types(df, segments_df)

        # Compute final segment metrics
        return self._compute_segment_metrics(df, segments_df, aid_stations_km_set)

    def _get_initial_segments_with_mean_elevation(
        self, df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        """Step 1: Get initial segments with mean elevation gain.

        - Classify points by grade
        - Create initial segments (category changes)
        - Force splits at aid stations
        - Aggregate with basic metrics
        - Reclassify types with mean metrics

        Args:
            df: Preprocessed DataFrame with point data
            aid_stations_km_set: Set of aid station positions

        Returns:
            DataFrame with initial segment summaries
        """
        # Classify grade for each point
        df["elevation_delta_per_km"] = (
            df["elevationM_ma_5"].diff() / df["cumulated_distance"].diff().replace(0, np.nan) * 1000
        )
        df["elevation_delta_per_km"] = df["elevation_delta_per_km"].fillna(0)

        # Compute rolling elevation delta per km over ~1km window
        window_size = min(100, len(df) // 10)  # Approximate 1km window
        if window_size > 1:
            dist_diff = df["cumulated_distance"].diff(window_size).replace(0, np.nan)
            elev_diff = df["elevationM_ma_5"].diff(window_size).abs()
            df["cumulated_elevation_delta_per_km"] = (elev_diff / dist_diff * 1000).fillna(0)
        else:
            df["cumulated_elevation_delta_per_km"] = df["elevation_delta_per_km"].abs()

        df["grade_category"] = df.apply(
            lambda row: self.classify_grade(
                row["grade_ma_10"], row.get("cumulated_elevation_delta_per_km", None)
            ),
            axis=1,
        )

        # Create initial segments when category changes
        df["segment"] = (df["grade_category"] != df["grade_category"].shift()).cumsum()

        # Force segment boundaries at aid stations
        for aid_km in aid_stations_km_set:
            distances = abs(df["cumulated_distance"] - aid_km)
            nearest_idx = distances.idxmin()
            nearest_dist_m = abs(df.loc[nearest_idx, "cumulated_distance"] - aid_km) * 1000

            if nearest_dist_m <= 25:
                # Force split: change segment ID at this point
                df.loc[nearest_idx:, "segment"] = df.loc[nearest_idx:, "segment"] + df["segment"].max() + 1
                # Renumber segments
                df["segment"] = pd.Categorical(df["segment"]).codes

        # Initial merge of very small segments and same-color adjacent segments
        if len(df) > 0 and df["cumulated_distance"].max() > 0:
            points_per_m = len(df) / (df["cumulated_distance"].max() * 1000)
            min_size = max(3, int(150 * points_per_m))
        else:
            min_size = 3

        df = merge_small_segments(df, min_size=min_size)
        df = merge_adjacent_same_color(df)

        # Aggregate to segments with basic metrics
        segments_df = self._aggregate_segments(df, aid_stations_km_set)

        # Recompute and reclassify segments with mean metrics
        segments_df = self._recompute_and_reclassify_segments(segments_df, df, aid_stations_km_set)

        return segments_df

    def _merge_small_segments_with_closest(
        self, segments_df: pd.DataFrame, df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        """Step 2: Merge small segments (< 1 km-eq) with closest neighbor.

        Args:
            segments_df: DataFrame with segment summaries
            df: Full DataFrame with point data
            aid_stations_km_set: Set of aid station positions

        Returns:
            Updated segments DataFrame with small segments merged
        """
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
                lambda row: self.planner.compute_distance_eq_km(row["distanceKm"], row["elevGainM"]), axis=1
            )

            # Sort by startKm
            segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
            segments_df["segmentId"] = range(len(segments_df))

            # Use a set to track which segments have been merged/processed
            processed_indices = set()
            result_segments = []
            i = 0

            while i < len(segments_df):
                # Skip if already processed
                if i in processed_indices:
                    i += 1
                    continue

                seg = segments_df.iloc[i]
                distance_eq = seg["distanceEqKm"]
                is_aid_split = seg.get("isAidSplit", False)

                # If segment is small (< 1 km-eq) and not at aid station, merge with closest neighbor
                if distance_eq < 1.0 and not is_aid_split:
                    best_match_idx = None
                    best_gap = float("inf")

                    # Determine max gap tolerance based on segment size
                    # Very small segments (< 0.3 km-eq): allow up to 0.5 km gap
                    # Small segments (0.3-0.7 km-eq): allow up to 0.2 km gap
                    # Larger small segments (0.7-1.0 km-eq): allow up to 0.1 km gap
                    if distance_eq < 0.3:
                        max_gap = 0.5
                    elif distance_eq < 0.7:
                        max_gap = 0.2
                    else:
                        max_gap = 0.1

                    # Check previous segment (only if not already processed)
                    if i > 0 and (i - 1) not in processed_indices:
                        prev_seg = segments_df.iloc[i - 1]
                        prev_is_aid = prev_seg.get("isAidSplit", False)
                        if not prev_is_aid:
                            gap = abs(prev_seg["endKm"] - seg["startKm"])
                            if gap < max_gap and gap < best_gap:
                                best_gap = gap
                                best_match_idx = i - 1

                    # Check next segment (only if not already processed)
                    if i < len(segments_df) - 1 and (i + 1) not in processed_indices:
                        next_seg = segments_df.iloc[i + 1]
                        next_is_aid = next_seg.get("isAidSplit", False)
                        if not next_is_aid:
                            gap = abs(seg["endKm"] - next_seg["startKm"])
                            if gap < max_gap and gap < best_gap:
                                best_gap = gap
                                best_match_idx = i + 1

                    # Merge with best match if found
                    if best_match_idx is not None:
                        matched_seg = segments_df.iloc[best_match_idx]
                        # Determine merge order
                        if best_match_idx < i:
                            # Merging with previous segment
                            merged = self._merge_two_segments(matched_seg, seg, df)
                            # Find and replace the previous segment in result_segments
                            # Since we process in order, it should be the last item
                            if result_segments:
                                result_segments[-1] = merged
                            else:
                                result_segments.append(merged)
                            processed_indices.add(i)
                            processed_indices.add(best_match_idx)
                            i += 1  # Move to next segment
                        else:
                            # Merging with next segment
                            merged = self._merge_two_segments(seg, matched_seg, df)
                            result_segments.append(merged)
                            processed_indices.add(i)
                            processed_indices.add(best_match_idx)
                            i += 2  # Skip both current and next
                        changed = True
                        continue
                    else:
                        # No valid neighbor found within gap tolerance
                        # For very small segments (< 0.5 km-eq), try merging with ANY neighbor
                        # regardless of gap size (up to 1 km) if no closer match exists
                        if distance_eq < 0.5:
                            best_match_idx = None
                            best_gap = float("inf")
                            
                            # Check previous segment with relaxed gap
                            if i > 0 and (i - 1) not in processed_indices:
                                prev_seg = segments_df.iloc[i - 1]
                                prev_is_aid = prev_seg.get("isAidSplit", False)
                                if not prev_is_aid:
                                    gap = abs(prev_seg["endKm"] - seg["startKm"])
                                    if gap < best_gap and gap < 1.0:  # Allow up to 1 km gap
                                        best_gap = gap
                                        best_match_idx = i - 1
                            
                            # Check next segment with relaxed gap
                            if i < len(segments_df) - 1 and (i + 1) not in processed_indices:
                                next_seg = segments_df.iloc[i + 1]
                                next_is_aid = next_seg.get("isAidSplit", False)
                                if not next_is_aid:
                                    gap = abs(seg["endKm"] - next_seg["startKm"])
                                    if gap < best_gap and gap < 1.0:  # Allow up to 1 km gap
                                        best_gap = gap
                                        best_match_idx = i + 1
                            
                            # Merge if found
                            if best_match_idx is not None:
                                matched_seg = segments_df.iloc[best_match_idx]
                                if best_match_idx < i:
                                    # Merging with previous segment
                                    merged = self._merge_two_segments(matched_seg, seg, df)
                                    if result_segments:
                                        result_segments[-1] = merged
                                    else:
                                        result_segments.append(merged)
                                    processed_indices.add(i)
                                    processed_indices.add(best_match_idx)
                                    i += 1
                                else:
                                    # Merging with next segment
                                    merged = self._merge_two_segments(seg, matched_seg, df)
                                    result_segments.append(merged)
                                    processed_indices.add(i)
                                    processed_indices.add(best_match_idx)
                                    i += 2
                                changed = True
                                continue

                # Keep segment as-is
                result_segments.append(seg.to_dict())
                processed_indices.add(i)
                i += 1

            if not changed:
                break  # No more merges possible

            segments_df = pd.DataFrame(result_segments)
            segments_df = segments_df.reset_index(drop=True)

            # Recompute metrics after merging
            segments_df = self._recompute_and_reclassify_segments(segments_df, df, aid_stations_km_set)

        # Final sorting and renumbering
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))
        return segments_df

    def _merge_contrary_trend_segments(
        self, segments_df: pd.DataFrame, df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        """Step 3: Merge small descents in uphill or small uphills in downhill.

        Args:
            segments_df: DataFrame with segment summaries
            df: Full DataFrame with point data
            aid_stations_km_set: Set of aid station positions

        Returns:
            Updated segments DataFrame with contrary trend segments merged
        """
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

            # Compute distance-equivalent for each segment
            segments_df["distanceEqKm"] = segments_df.apply(
                lambda row: self.planner.compute_distance_eq_km(row["distanceKm"], row["elevGainM"]), axis=1
            )

            # Sort by startKm
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

                # Don't merge if either segment is at aid station
                if seg1.get("isAidSplit", False) or seg2.get("isAidSplit", False):
                    result_segments.append(seg1.to_dict())
                    i += 1
                    continue

                # Check adjacency (allow small gaps)
                seg1_end = seg1["endKm"]
                seg2_start = seg2["startKm"]
                gap = abs(seg1_end - seg2_start)
                if gap >= 0.1:
                    result_segments.append(seg1.to_dict())
                    i += 1
                    continue

                should_merge = False
                dist_eq1 = seg1["distanceEqKm"]
                dist_eq2 = seg2["distanceEqKm"]

                # Determine general trend by looking at surrounding segments (5 before, 5 after)
                window_start = max(0, i - 5)
                window_end = min(len(segments_df), i + 6)
                window_segments = segments_df.iloc[window_start:window_end]

                # Count uphill vs downhill segments in window (excluding current pair)
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

                # Determine general trend
                general_trend = None
                if downhill_count > uphill_count + 2:  # Clear descent trend
                    general_trend = "descent"
                elif uphill_count > downhill_count + 2:  # Clear climb trend
                    general_trend = "climb"

                # Case 1: Small descent (< 0.5 km-eq) in a general climb
                if general_trend == "climb":
                    if seg1["type"] in downhill_types and dist_eq1 < 0.5:
                        should_merge = True
                    elif seg2["type"] in downhill_types and dist_eq2 < 0.5:
                        should_merge = True

                # Case 2: Small climb (< 0.5 km-eq) in a general descent
                if general_trend == "descent":
                    if seg1["type"] in uphill_types and dist_eq1 < 0.5:
                        should_merge = True
                    elif seg2["type"] in uphill_types and dist_eq2 < 0.5:
                        should_merge = True

                if should_merge:
                    merged = self._merge_two_segments(seg1, seg2, df)
                    result_segments.append(merged)
                    i += 2
                    changed = True
                else:
                    result_segments.append(seg1.to_dict())
                    i += 1

            if not changed:
                break  # No more merges possible

            segments_df = pd.DataFrame(result_segments)
            segments_df = segments_df.reset_index(drop=True)

            # Recompute metrics after merging
            segments_df = self._recompute_and_reclassify_segments(segments_df, df, aid_stations_km_set)

        # Final sorting and renumbering
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))
        return segments_df

    def _merge_consecutive_same_type_segments(
        self, segments_df: pd.DataFrame, df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        """Step 4: Merge consecutive segments with same type.

        Args:
            segments_df: DataFrame with segment summaries
            df: Full DataFrame with point data
            aid_stations_km_set: Set of aid station positions

        Returns:
            Updated segments DataFrame with consecutive same-type segments merged
        """
        if len(segments_df) < 2:
            return segments_df

        segments_df = segments_df.copy()
        max_iterations = len(segments_df)
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            changed = False

            # Sort by startKm
            segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
            segments_df["segmentId"] = range(len(segments_df))

            result_segments = []
            i = 0

            while i < len(segments_df):
                if i >= len(segments_df) - 1:
                    # Last segment, keep as-is
                    result_segments.append(segments_df.iloc[i].to_dict())
                    break

                seg1 = segments_df.iloc[i]
                seg2 = segments_df.iloc[i + 1]

                # Don't merge if either segment is at aid station
                if seg1.get("isAidSplit", False) or seg2.get("isAidSplit", False):
                    result_segments.append(seg1.to_dict())
                    i += 1
                    continue

                # Check if segments are adjacent (allow small gaps)
                seg1_end = seg1["endKm"]
                seg2_start = seg2["startKm"]
                gap = abs(seg1_end - seg2_start)
                
                # For same-type segments, allow larger gaps (up to 0.2 km) to catch segments
                # that may have small gaps due to recomputation
                max_gap_same_type = 0.2
                if gap >= max_gap_same_type:
                    result_segments.append(seg1.to_dict())
                    i += 1
                    continue

                # Check if segments have the same type
                if seg1["type"] == seg2["type"]:
                    # Merge segments
                    merged = self._merge_two_segments(seg1, seg2, df)
                    result_segments.append(merged)
                    i += 2  # Skip both segments
                    changed = True
                else:
                    result_segments.append(seg1.to_dict())
                    i += 1

            if not changed:
                break  # No more merges possible

            segments_df = pd.DataFrame(result_segments)
            segments_df = segments_df.reset_index(drop=True)

            # Recompute metrics after merging
            segments_df = self._recompute_and_reclassify_segments(segments_df, df, aid_stations_km_set)

        # Final sorting and renumbering
        segments_df = segments_df.sort_values("startKm").reset_index(drop=True)
        segments_df["segmentId"] = range(len(segments_df))
        return segments_df

    def _merge_two_segments(
        self, seg1: pd.Series, seg2: pd.Series, df: pd.DataFrame
    ) -> dict:
        """Merge two adjacent segments and compute combined metrics.

        Args:
            seg1: First segment Series (must be adjacent to seg2)
            seg2: Second segment Series (must be adjacent to seg1)
            df: Full DataFrame with point data

        Returns:
            Dictionary with merged segment metrics
        """
        # Ensure segments are adjacent (seg1 ends where seg2 starts)
        seg1_end = seg1["endKm"]
        seg2_start = seg2["startKm"]
        
        # Verify adjacency (allow small tolerance for floating point)
        if abs(seg1_end - seg2_start) > 0.01:
            logger.warning(
                f"Attempting to merge non-adjacent segments: seg1 ends at {seg1_end}, seg2 starts at {seg2_start}"
            )
        
        # Use seg1 start and seg2 end for merged segment (cover entire range including any gap)
        start_km = min(seg1["startKm"], seg2["startKm"])
        end_km = max(seg1["endKm"], seg2["endKm"])

        # Get combined point data (from seg1 start to seg2 end, covering any gap)
        combined_df = df[(df["cumulated_distance"] >= start_km) & (df["cumulated_distance"] <= end_km)]

        if combined_df.empty:
            # Fallback: sum metrics and use full range
            distance_km = end_km - start_km  # Cover full range including gap
            elev_gain = seg1["elevGainM"] + seg2["elevGainM"]
            elev_loss = seg1["elevLossM"] + seg2["elevLossM"]
            avg_grade = self.compute_avg_grade(elev_gain, elev_loss, distance_km)
            # Determine dominant type
            merged_type = seg1["type"] if seg1["distanceKm"] >= seg2["distanceKm"] else seg2["type"]
        else:
            # Recompute from actual data, but ensure we cover the full range
            # Use full range (start_km to end_km) to cover any gaps
            # But compute metrics only from actual data points
            distance_km = end_km - start_km  # Full range including gap
            elev_gain = combined_df["elevation_difference"].clip(lower=0).sum()
            elev_loss = combined_df["elevation_difference"].clip(upper=0).abs().sum()
            avg_grade = self.compute_avg_grade(elev_gain, elev_loss, distance_km)
            
            # Compute elevation delta per km for flat detection
            elevation_delta_per_km = 0.0
            if distance_km > 0:
                elevation_delta_per_km = abs(elev_gain - elev_loss) / distance_km
            
            # Determine type from combined grade and elevation delta
            merged_type = self.classify_grade(avg_grade, elevation_delta_per_km)
            # Use full range to cover gaps
            # start_km and end_km already set above

        return {
            "segmentId": 0,  # Will be renumbered
            "type": merged_type,
            "startKm": start_km,
            "endKm": end_km,
            "distanceKm": distance_km,
            "elevGainM": elev_gain,
            "elevLossM": elev_loss,
            "avgGrade": avg_grade,
            "isAidSplit": seg1.get("isAidSplit", False) or seg2.get("isAidSplit", False),
        }

    def _aggregate_segments(self, df: pd.DataFrame, aid_stations_km_set: set[float]) -> pd.DataFrame:
        """Aggregate points into segment summary."""
        segments = []
        # Get unique segments and sort by their start position (not by segment ID)
        unique_segments_with_positions = []
        for seg_id in df["segment"].unique():
            seg_df = df[df["segment"] == seg_id]
            if not seg_df.empty:
                start_pos = seg_df["cumulated_distance"].min()
                unique_segments_with_positions.append((start_pos, seg_id))
        
        # Sort by start position
        unique_segments_with_positions.sort(key=lambda x: x[0])
        unique_segments = [seg_id for _, seg_id in unique_segments_with_positions]
        
        for idx, seg_id in enumerate(unique_segments):
            seg_df = df[df["segment"] == seg_id]
            if seg_df.empty:
                continue

            # Get actual min/max from points
            actual_start = seg_df["cumulated_distance"].min()
            actual_end = seg_df["cumulated_distance"].max()
            
            # Determine segment boundaries to ensure continuous coverage (no gaps, no overlaps)
            # Segments must touch exactly: seg[i].endKm == seg[i+1].startKm
            if idx == 0:
                # First segment starts at the beginning
                start_km = 0.0
            else:
                # Start exactly where previous segment ended (will be set after previous segment is processed)
                # For now, use actual_start, we'll fix continuity after all segments are processed
                start_km = actual_start
            
            if idx == len(unique_segments) - 1:
                # Last segment ends at the end
                end_km = df["cumulated_distance"].max()
            else:
                # End exactly where next segment starts (will be set when processing next segment)
                # For now, use actual_end, we'll fix continuity after all segments are processed
                end_km = actual_end
            
            # Ensure end >= start
            end_km = max(end_km, start_km)
            distance_km = end_km - start_km
            
            # Compute metrics from actual points in segment
            elev_gain = seg_df["elevation_difference"].clip(lower=0).sum()
            elev_loss = seg_df["elevation_difference"].clip(upper=0).abs().sum()
            avg_grade = self.compute_avg_grade(elev_gain, elev_loss, distance_km)

            # Check if aid station at boundary
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
        
        # Fix continuity: ensure each segment ends where next starts
        if len(segments) > 1:
            for idx in range(len(segments) - 1):
                # Set current segment end to next segment start
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
        """Recompute segment metrics and reclassify types after merging/splitting.

        Args:
            segments_df: DataFrame with segment summaries
            df: Full DataFrame with point data
            aid_stations_km_set: Set of aid station positions

        Returns:
            Updated segments DataFrame with recomputed metrics and reclassified types
        """
        result_segments = []

        for _, seg in segments_df.iterrows():
            # Get point data for this segment
            seg_df = df[
                (df["cumulated_distance"] >= seg["startKm"]) & (df["cumulated_distance"] <= seg["endKm"])
            ]

            if seg_df.empty:
                # Keep original if no data
                result_segments.append(seg.to_dict())
                continue

            # Preserve original segment boundaries to maintain continuity
            original_start = seg["startKm"]
            original_end = seg["endKm"]
            
            # Use original boundaries to ensure continuity (cover gaps)
            distance_km = original_end - original_start
            elev_gain = seg_df["elevation_difference"].clip(lower=0).sum() if not seg_df.empty else 0.0
            elev_loss = seg_df["elevation_difference"].clip(upper=0).abs().sum() if not seg_df.empty else 0.0
            avg_grade = self.compute_avg_grade(elev_gain, elev_loss, distance_km)

            # Compute elevation delta per km for flat detection
            elevation_delta_per_km = 0.0
            if distance_km > 0:
                elevation_delta_per_km = abs(elev_gain - elev_loss) / distance_km

            # Reclassify segment type based on recomputed metrics
            new_type = self.classify_grade(avg_grade, elevation_delta_per_km)

            # Check if aid station at boundary
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
                    "startKm": original_start,  # Preserve original boundaries
                    "endKm": original_end,  # Preserve original boundaries
                    "distanceKm": distance_km,
                    "elevGainM": elev_gain,
                    "elevLossM": elev_loss,
                    "avgGrade": avg_grade,
                    "isAidSplit": is_aid_split,
                }
            )

        result_df = pd.DataFrame(result_segments)
        result_df = result_df.reset_index(drop=True)
        # Sort by startKm to ensure proper ordering
        result_df = result_df.sort_values("startKm").reset_index(drop=True)
        # Renumber segments after sorting
        result_df["segmentId"] = range(len(result_df))
        return result_df

    def _compute_segment_metrics(
        self, df: pd.DataFrame, segments_df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        """Compute final segment metrics with distance-equivalent and time.
        
        Uses segment boundaries from segments_df to ensure continuity.
        """
        segments = []
        
        # Sort segments_df by startKm to process in order
        segments_df_sorted = segments_df.sort_values("startKm").reset_index(drop=True)

        for idx, seg_row in segments_df_sorted.iterrows():
            seg_id = seg_row["segmentId"]
            start_km = seg_row["startKm"]
            end_km = seg_row["endKm"]
            
            # Get point data for this segment using boundaries from segments_df
            seg_df = df[
                (df["cumulated_distance"] >= start_km) & (df["cumulated_distance"] <= end_km)
            ]

            # Use boundaries from segments_df to ensure continuity
            distance_km = end_km - start_km

            # Compute metrics from actual data points (if any)
            if not seg_df.empty:
                elev_gain = seg_df["elevation_difference"].clip(lower=0).sum()
                elev_loss = seg_df["elevation_difference"].clip(upper=0).abs().sum()
                avg_grade = self.compute_avg_grade(elev_gain, elev_loss, distance_km)
            else:
                # Fallback to segment_df values if no data points
                elev_gain = seg_row.get("elevGainM", 0.0)
                elev_loss = seg_row.get("elevLossM", 0.0)
                avg_grade = self.compute_avg_grade(elev_gain, elev_loss, distance_km)

            # Check if aid station at boundary
            is_aid_split = seg_row.get("isAidSplit", False)
            if not is_aid_split:
                for aid_km in aid_stations_km_set:
                    if abs(start_km - aid_km) < 0.025 or abs(end_km - aid_km) < 0.025:
                        is_aid_split = True
                        break

            # Compute distance-equivalent
            distance_eq_km = self.planner.compute_distance_eq_km(distance_km, elev_gain)

            # Get type from segments_df
            seg_type = seg_row.get("type", "unknown")

            segments.append(
                {
                    "segmentId": int(seg_id),
                    "type": seg_type,
                    "startKm": start_km,  # Use boundaries from segments_df
                    "endKm": end_km,  # Use boundaries from segments_df
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
            )

        result_df = pd.DataFrame(segments)
        # Sort by startKm to ensure proper ordering
        result_df = result_df.sort_values("startKm").reset_index(drop=True)
        # Renumber segments after sorting
        result_df["segmentId"] = range(len(result_df))
        return result_df

    def merge_segments_manually(
        self, segments_df: pd.DataFrame, segment_ids: list[int], metrics_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge multiple adjacent segments manually.

        Args:
            segments_df: DataFrame with segment summaries
            segment_ids: List of segment IDs to merge (must be adjacent and in order)
            metrics_df: Full DataFrame with point data for recomputing metrics

        Returns:
            Updated segments DataFrame with merged segments
        """
        if len(segment_ids) < 2:
            return segments_df

        # Sort segment IDs to ensure order
        segment_ids = sorted(segment_ids)

        # Verify segments are adjacent
        segments_to_merge = segments_df[segments_df["segmentId"].isin(segment_ids)].sort_values("startKm")
        if len(segments_to_merge) != len(segment_ids):
            logger.warning(f"Some segment IDs not found: {segment_ids}")
            return segments_df

        # Check adjacency
        for i in range(len(segments_to_merge) - 1):
            current_end = segments_to_merge.iloc[i]["endKm"]
            next_start = segments_to_merge.iloc[i + 1]["startKm"]
            if abs(current_end - next_start) > 0.01:
                logger.warning(f"Segments {segment_ids} are not adjacent")
                return segments_df

        # Merge segments sequentially
        merged_seg = segments_to_merge.iloc[0].to_dict()
        for i in range(1, len(segments_to_merge)):
            next_seg = segments_to_merge.iloc[i].to_dict()
            merged_seg = self._merge_two_segments(
                pd.Series(merged_seg), pd.Series(next_seg), metrics_df
            )

        # Extract aid stations from original segments if any
        aid_stations_km_set = set()
        if "isAidSplit" in segments_to_merge.columns:
            # Preserve aid split flag if any merged segment was at an aid station
            merged_seg["isAidSplit"] = segments_to_merge["isAidSplit"].any()
        
        # Preserve speed values from original segments (use weighted average by distance)
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
                merged_seg["speedKmh"] = segments_to_merge["speedKmh"].iloc[0] if len(segments_to_merge) > 0 else 0.0
        else:
            merged_seg["speedKmh"] = 0.0
        
        # Remove merged segments and add the new merged one
        result_df = segments_df[~segments_df["segmentId"].isin(segment_ids)].copy()
        result_df = pd.concat([result_df, pd.DataFrame([merged_seg])], ignore_index=True)

        # Sort by startKm and renumber
        result_df = result_df.sort_values("startKm").reset_index(drop=True)
        result_df["segmentId"] = range(len(result_df))

        # Find the merged segment BEFORE fixing continuity (by position since we just sorted)
        # The merged segment should be at the position where segments 0, 1, 2 were
        # Find by checking which segment covers the range from the first merged segment's start
        merged_segment_ids_original = sorted(segment_ids)
        first_merged_start = segments_df[segments_df["segmentId"] == merged_segment_ids_original[0]]["startKm"].iloc[0]
        merged_positions = result_df[
            (result_df["startKm"] - first_merged_start).abs() < 0.001
        ].index
        
        merged_idx_for_speeds = None
        if len(merged_positions) > 0:
            merged_idx_for_speeds = merged_positions[0]

        # Fix continuity: ensure each segment ends where next starts
        # BUT: preserve original boundaries for segments that have valid data
        # Only adjust if there's a gap, not if segments overlap
        if len(result_df) > 1:
            for idx in range(len(result_df) - 1):
                current_end = result_df.iloc[idx]["endKm"]
                next_start = result_df.iloc[idx + 1]["startKm"]
                gap = next_start - current_end
                
                # Only adjust if there's a significant gap (> 0.001 km)
                # Don't modify if segments overlap or are already touching
                if gap > 0.001:
                    # Extend current segment to meet next
                    result_df.iloc[idx, result_df.columns.get_loc("endKm")] = next_start
                    result_df.iloc[idx, result_df.columns.get_loc("distanceKm")] = (
                        result_df.iloc[idx]["endKm"] - result_df.iloc[idx]["startKm"]
                    )
                elif gap < -0.001:
                    # Segments overlap - adjust next segment start
                    result_df.iloc[idx + 1, result_df.columns.get_loc("startKm")] = current_end
                    result_df.iloc[idx + 1, result_df.columns.get_loc("distanceKm")] = (
                        result_df.iloc[idx + 1]["endKm"] - result_df.iloc[idx + 1]["startKm"]
                    )
        
        # Recompute metrics for the merged segment
        result_df = self._recompute_and_reclassify_segments(result_df, metrics_df, aid_stations_km_set)

        # Compute final metrics including distanceEqKm, speedEqKmh, speedKmh, timeSec
        # Note: _compute_segment_metrics expects (points_df, segments_df, aid_stations_km_set)
        result_df = self._compute_segment_metrics(metrics_df, result_df, aid_stations_km_set)
        
        # Restore speed values using the position we found earlier
        if merged_idx_for_speeds is not None and merged_idx_for_speeds < len(result_df):
            # After recomputation, the segment might have moved, so find it again by start position
            # But the segmentId should be preserved or predictable
            # Actually, after _compute_segment_metrics, segmentId is renumbered, so we need to find by startKm
            current_start = (
                result_df.iloc[merged_idx_for_speeds]["startKm"]
                if merged_idx_for_speeds < len(result_df)
                else None
            )
            if current_start is not None:
                # Find segment with this startKm
                target_mask = (result_df["startKm"] - current_start).abs() < 0.001
                target_indices = result_df[target_mask].index
                if len(target_indices) > 0:
                    target_idx = target_indices[0]
                    result_df.at[target_idx, "speedEqKmh"] = merged_seg.get("speedEqKmh", 0.0)
                    result_df.at[target_idx, "speedKmh"] = merged_seg.get("speedKmh", 0.0)
                    # Recompute time with preserved speeds
                    distance_eq = result_df.at[target_idx, "distanceEqKm"]
                    distance = result_df.at[target_idx, "distanceKm"]
                    speed_eq = result_df.at[target_idx, "speedEqKmh"]
                    speed = result_df.at[target_idx, "speedKmh"]
                    result_df.at[target_idx, "timeSec"] = self.compute_segment_time(
                        distance_eq, distance, speed_eq, speed
                    )

        # Final sort and renumber
        result_df = result_df.sort_values("startKm").reset_index(drop=True)
        result_df["segmentId"] = range(len(result_df))

        return result_df

    def compute_segment_time(
        self, distance_eq_km: float, distance_km: float, speed_eq_kmh: float, speed_kmh: float
    ) -> int:
        """Compute segment time from speed.

        Args:
            distance_eq_km: Distance-equivalent in km
            distance_km: Actual distance in km
            speed_eq_kmh: Speed-equivalent in km/h (takes precedence)
            speed_kmh: Speed in km/h

        Returns:
            Time in seconds
        """
        if speed_eq_kmh > 0:
            return int(round(3600 * distance_eq_km / speed_eq_kmh))
        elif speed_kmh > 0:
            return int(round(3600 * distance_km / speed_kmh))
        else:
            return 0

    def aggregate_summary(self, segments_df: pd.DataFrame) -> dict:
        """Aggregate totals from segments.

        Args:
            segments_df: DataFrame with segment metrics

        Returns:
            Dictionary with totals
        """
        if segments_df.empty:
            return {
                "distanceKm": 0.0,
                "distanceEqKm": 0.0,
                "elevGainM": 0.0,
                "elevLossM": 0.0,
                "timeSec": 0,
            }

        return {
            "distanceKm": float(segments_df["distanceKm"].sum()),
            "distanceEqKm": float(segments_df["distanceEqKm"].sum()),
            "elevGainM": float(segments_df["elevGainM"].sum()),
            "elevLossM": float(segments_df["elevLossM"].sum()),
            "timeSec": int(segments_df["timeSec"].sum()),
        }

    def compute_segment_stats_between(
        self, start_km: float, end_km: float, segments_df: pd.DataFrame
    ) -> dict[str, float]:
        """Compute statistics (distance, dist-eq, elevation gain/loss) between two points.

        Args:
            start_km: Start distance in km
            end_km: End distance in km
            segments_df: DataFrame with segment metrics

        Returns:
            Dictionary with distanceKm, distanceEqKm, elevGainM, elevLossM
        """
        stats = {
            "distanceKm": 0.0,
            "distanceEqKm": 0.0,
            "elevGainM": 0.0,
            "elevLossM": 0.0,
        }

        if start_km >= end_km:
            return stats

        # Sum stats from segments between start and end
        for _, seg in segments_df.iterrows():
            seg_start = float(seg.get("startKm", 0))
            seg_end = float(seg.get("endKm", 0))
            seg_distance_eq = float(seg.get("distanceEqKm", 0) or 0)
            seg_elev_gain = float(seg.get("elevGainM", 0) or 0)
            seg_elev_loss = float(seg.get("elevLossM", 0) or 0)

            # Skip segments entirely before start or after end
            if seg_end <= start_km or seg_start >= end_km:
                continue

            # Segment overlaps with the range
            overlap_start = max(seg_start, start_km)
            overlap_end = min(seg_end, end_km)
            overlap_ratio = (overlap_end - overlap_start) / (seg_end - seg_start) if (seg_end - seg_start) > 0 else 1.0

            stats["distanceKm"] += (overlap_end - overlap_start)
            stats["distanceEqKm"] += seg_distance_eq * overlap_ratio
            stats["elevGainM"] += seg_elev_gain * overlap_ratio
            stats["elevLossM"] += seg_elev_loss * overlap_ratio

        return stats

    def compute_aid_station_stats(
        self, aid_stations_km: list[float], segments_df: pd.DataFrame
    ) -> list[dict[str, float]]:
        """Compute statistics for each aid station segment (from previous or start).

        Args:
            aid_stations_km: List of aid station positions in km
            segments_df: DataFrame with segment metrics

        Returns:
            List of dictionaries with distanceKm, distanceEqKm, elevGainM, elevLossM, timeSec for each segment
        """
        sorted_aid_km = sorted(aid_stations_km)
        stats_list = []

        # First segment: from start (0) to first aid station
        if sorted_aid_km:
            first_stats = self.compute_segment_stats_between(0.0, sorted_aid_km[0], segments_df)
            # Add time for this segment
            first_time = self._compute_time_between(0.0, sorted_aid_km[0], segments_df)
            first_stats["timeSec"] = first_time
            stats_list.append(first_stats)

            # Subsequent segments: between aid stations
            for i in range(len(sorted_aid_km) - 1):
                segment_stats = self.compute_segment_stats_between(
                    sorted_aid_km[i], sorted_aid_km[i + 1], segments_df
                )
                # Add time for this segment
                segment_time = self._compute_time_between(sorted_aid_km[i], sorted_aid_km[i + 1], segments_df)
                segment_stats["timeSec"] = segment_time
                stats_list.append(segment_stats)

        return stats_list

    def _compute_time_between(self, start_km: float, end_km: float, segments_df: pd.DataFrame) -> float:
        """Compute time between two points.

        Args:
            start_km: Start distance in km
            end_km: End distance in km
            segments_df: DataFrame with segment metrics

        Returns:
            Time in seconds
        """
        cumulative_time = 0.0

        if start_km >= end_km:
            return cumulative_time

        # Sum times from segments between start and end
        for _, seg in segments_df.iterrows():
            seg_start = float(seg.get("startKm", 0))
            seg_end = float(seg.get("endKm", 0))
            seg_time = float(seg.get("timeSec", 0) or 0)

            # Skip segments entirely before start or after end
            if seg_end <= start_km or seg_start >= end_km:
                continue

            # Segment overlaps with the range
            if seg_end <= end_km and seg_start >= start_km:
                # Full segment included
                cumulative_time += seg_time
            elif seg_start < end_km and seg_end > start_km:
                # Partial segment - interpolate time based on distance ratio
                overlap_start = max(seg_start, start_km)
                overlap_end = min(seg_end, end_km)
                seg_distance = seg_end - seg_start
                if seg_distance > 0:
                    overlap_distance = overlap_end - overlap_start
                    partial_ratio = overlap_distance / seg_distance
                    cumulative_time += seg_time * partial_ratio

        return cumulative_time

    def compute_cumulative_stats_at_aid_stations(
        self, aid_stations_km: list[float], segments_df: pd.DataFrame
    ) -> list[dict[str, float]]:
        """Compute cumulative statistics at each aid station (from start).

        Args:
            aid_stations_km: List of aid station positions in km
            segments_df: DataFrame with segment metrics

        Returns:
            List of dictionaries with cumulative distanceEqKm, elevGainM, elevLossM at each aid station
        """
        sorted_aid_km = sorted(aid_stations_km)
        cumulative_stats = []

        for aid_km in sorted_aid_km:
            stats = self.compute_segment_stats_between(0.0, aid_km, segments_df)
            cumulative_stats.append(stats)

        return cumulative_stats

    def compute_aid_station_times(
        self, aid_stations_km: list[float], segments_df: pd.DataFrame
    ) -> list[float]:
        """Compute cumulative time at each aid station.

        Args:
            aid_stations_km: List of aid station positions in km
            segments_df: DataFrame with segment metrics (must have startKm, endKm, timeSec)

        Returns:
            List of cumulative times in seconds for each aid station
        """
        aid_times = []
        for aid_km in sorted(aid_stations_km):
            cumulative_time = 0.0

            # Sum times from segments up to this aid station
            for _, seg in segments_df.iterrows():
                seg_start = float(seg.get("startKm", 0))
                seg_end = float(seg.get("endKm", 0))
                seg_time = float(seg.get("timeSec", 0) or 0)

                # If segment ends before or at aid station, add full time
                if seg_end <= aid_km:
                    cumulative_time += seg_time
                # If segment starts before aid station but ends after, calculate partial time
                elif seg_start < aid_km:
                    # Interpolate time based on distance ratio
                    seg_distance = seg_end - seg_start
                    if seg_distance > 0:
                        partial_distance = aid_km - seg_start
                        partial_ratio = partial_distance / seg_distance
                        cumulative_time += seg_time * partial_ratio
                    else:
                        cumulative_time += seg_time
                    break
                # If segment starts after aid station, we're done
                else:
                    break

            aid_times.append(cumulative_time)

        return aid_times

    def save_race(
        self,
        race_name: str,
        aid_stations_km: list[float],
        segments_df: pd.DataFrame,
        race_id: Optional[str] = None,
        aid_stations_times: Optional[list[float]] = None,
    ) -> str:
        """Save race pacing data to CSV files.

        Args:
            race_name: Name of the race
            aid_stations_km: List of aid station positions
            segments_df: DataFrame with segment metrics
            race_id: Optional race ID (generates new if None)
            aid_stations_times: Optional list of cumulative times at aid stations (in seconds).
                If None, will be computed from segments_df.

        Returns:
            Race ID
        """
        if race_id is None:
            race_id = new_id()

        race_pacing_dir = self.storage.base_dir / "race_pacing"
        race_pacing_dir.mkdir(parents=True, exist_ok=True)

        # Compute aid station times if not provided
        if aid_stations_times is None:
            aid_stations_times = self.compute_aid_station_times(aid_stations_km, segments_df)

        # Save race metadata (races.csv is in main data folder)
        races_file = self.storage.base_dir / "races.csv"
        races_df = self.storage.read_csv(races_file)

        # Compute segment statistics between aid stations
        segment_stats = self.compute_aid_station_stats(aid_stations_km, segments_df)

        if races_df.empty:
            races_df = pd.DataFrame(
                columns=[
                    "raceId",
                    "name",
                    "createdAt",
                    "aidStationsKm",
                    "aidStationsTimes",
                    "aidStationsDistEq",
                    "aidStationsElevGain",
                    "aidStationsElevLoss",
                ]
            )

        # Ensure columns exist for backward compatibility
        if "aidStationsTimes" not in races_df.columns:
            races_df["aidStationsTimes"] = ""
        if "aidStationsDistEq" not in races_df.columns:
            races_df["aidStationsDistEq"] = ""
        if "aidStationsElevGain" not in races_df.columns:
            races_df["aidStationsElevGain"] = ""
        if "aidStationsElevLoss" not in races_df.columns:
            races_df["aidStationsElevLoss"] = ""

        # Convert aid stations and times to strings
        aid_stations_str = ",".join([str(x) for x in sorted(aid_stations_km)])
        aid_times_str = ",".join([str(int(t)) for t in aid_stations_times])

        # Convert segment statistics to strings
        aid_dist_eq_list = [str(round(seg.get("distanceEqKm", 0.0), 1)) for seg in segment_stats]
        aid_elev_gain_list = [str(int(seg.get("elevGainM", 0.0))) for seg in segment_stats]
        aid_elev_loss_list = [str(int(seg.get("elevLossM", 0.0))) for seg in segment_stats]
        aid_dist_eq_str = ",".join(aid_dist_eq_list)
        aid_elev_gain_str = ",".join(aid_elev_gain_list)
        aid_elev_loss_str = ",".join(aid_elev_loss_list)

        # Update or create race entry
        if "raceId" in races_df.columns and race_id in races_df["raceId"].values:
            races_df.loc[races_df["raceId"] == race_id, "name"] = race_name
            races_df.loc[races_df["raceId"] == race_id, "aidStationsKm"] = aid_stations_str
            races_df.loc[races_df["raceId"] == race_id, "aidStationsTimes"] = aid_times_str
            races_df.loc[races_df["raceId"] == race_id, "aidStationsDistEq"] = aid_dist_eq_str
            races_df.loc[races_df["raceId"] == race_id, "aidStationsElevGain"] = aid_elev_gain_str
            races_df.loc[races_df["raceId"] == race_id, "aidStationsElevLoss"] = aid_elev_loss_str
        else:
            import datetime

            new_row = {
                "raceId": race_id,
                "name": race_name,
                "createdAt": datetime.datetime.now().isoformat(),
                "aidStationsKm": aid_stations_str,
                "aidStationsTimes": aid_times_str,
                "aidStationsDistEq": aid_dist_eq_str,
                "aidStationsElevGain": aid_elev_gain_str,
                "aidStationsElevLoss": aid_elev_loss_str,
            }
            races_df = pd.concat([races_df, pd.DataFrame([new_row])], ignore_index=True)

        self.storage.write_csv(races_file, races_df)

        # Save segments
        segments_file = race_pacing_dir / f"{race_id}_segments.csv"
        self.storage.write_csv(segments_file, segments_df)

        logger.info(f"Saved race {race_id}: {race_name}")
        return race_id

    def load_race(
        self, race_id: str
    ) -> Optional[tuple[str, list[float], pd.DataFrame, Optional[list[float]]]]:
        """Load race pacing data from CSV files.

        Args:
            race_id: Race ID

        Returns:
            Tuple of (race_name, aid_stations_km, segments_df, aid_stations_times) or None if not found.
            aid_stations_times may be None if not saved.
        """
        race_pacing_dir = self.storage.base_dir / "race_pacing"
        races_file = self.storage.base_dir / "races.csv"

        if not races_file.exists():
            return None

        races_df = self.storage.read_csv(races_file)
        if races_df.empty or race_id not in races_df["raceId"].values:
            return None

        race_row = races_df[races_df["raceId"] == race_id].iloc[0]
        race_name = str(race_row["name"])

        # Parse aid stations
        aid_stations_str = str(race_row.get("aidStationsKm", ""))
        aid_stations_km = [float(x.strip()) for x in aid_stations_str.split(",") if x.strip()]

        # Parse aid station times (optional, for backward compatibility)
        aid_stations_times = None
        if "aidStationsTimes" in race_row and pd.notna(race_row.get("aidStationsTimes")):
            aid_times_str = str(race_row["aidStationsTimes"])
            if aid_times_str.strip():
                aid_stations_times = [float(x.strip()) for x in aid_times_str.split(",") if x.strip()]

        # Load segments
        segments_file = race_pacing_dir / f"{race_id}_segments.csv"
        if not segments_file.exists():
            return None

        segments_df = self.storage.read_csv(segments_file)

        return (race_name, aid_stations_km, segments_df, aid_stations_times)

    def list_races(self) -> pd.DataFrame:
        """List all saved races.

        Returns:
            DataFrame with raceId, name, createdAt columns
        """
        races_file = self.storage.base_dir / "races.csv"

        if not races_file.exists():
            return pd.DataFrame(columns=["raceId", "name", "createdAt"])

        races_df = self.storage.read_csv(races_file)
        if races_df.empty:
            return pd.DataFrame(columns=["raceId", "name", "createdAt"])

        return races_df[["raceId", "name", "createdAt"]].copy()

