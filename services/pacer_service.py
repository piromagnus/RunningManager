"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Pacer service for race course segmentation and pacing calculations.

Handles time-invariant GPX routes (no timestamps required).
"""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from streamlit.logger import get_logger

from persistence.csv_storage import CsvStorage
from persistence.repositories import RacePacingLinksRepo
from services.pacer.activity_comparison import ActivityComparison
from services.pacer.aid_station_stats import AidStationStats
from services.pacer.preprocessing import PacerPreprocessor
from services.pacer.race_persistence import RacePersistence
from services.pacer.segmentation import SegmentationService
from services.planner_service import PlannerService
from services.speed_profile_service import SpeedProfileService
from utils.config import Config
from utils.grade_classification import classify_grade_pacer_5cat

logger = get_logger(__name__)


class PacerService:
    """Service for race course segmentation and pacing."""

    def __init__(self, storage: CsvStorage, config: Optional[Config] = None):
        self.storage = storage
        self.planner = PlannerService(storage)
        from utils.config import load_config
        self.config = config or load_config()
        self.speed_profile = SpeedProfileService(config=self.config)
        self.links_repo = RacePacingLinksRepo(storage)
        self.segmentation = SegmentationService(self.planner)
        self.preprocessor = PacerPreprocessor()
        self.aid_station_stats = AidStationStats()
        self.race_persistence = RacePersistence(
            storage=self.storage,
            compute_aid_station_times=self.aid_station_stats.compute_aid_station_times,
            compute_aid_station_stats=self.aid_station_stats.compute_aid_station_stats,
        )
        self.activity_comparison = ActivityComparison(
            load_race=self.race_persistence.load_race,
            planner=self.planner,
            speed_profile=self.speed_profile,
            links_repo=self.links_repo,
        )

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
        return self.preprocessor.preprocess_timeseries_for_pacing(timeseries_df)

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
        return classify_grade_pacer_5cat(grade, cumulated_elevation_delta_per_km)

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
        return self.segmentation.segment_course(
            df,
            aid_stations_km,
            min_seg_len_m=min_seg_len_m,
            max_splits_per_long_up=max_splits_per_long_up,
        )

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
        return self.segmentation._get_initial_segments_with_mean_elevation(df, aid_stations_km_set)

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
        return self.segmentation._merge_small_segments_with_closest(segments_df, df, aid_stations_km_set)

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
        return self.segmentation._merge_contrary_trend_segments(segments_df, df, aid_stations_km_set)

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
        return self.segmentation._merge_consecutive_same_type_segments(segments_df, df, aid_stations_km_set)

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
        return self.segmentation._merge_two_segments(seg1, seg2, df)

    def _aggregate_segments(self, df: pd.DataFrame, aid_stations_km_set: set[float]) -> pd.DataFrame:
        """Aggregate points into segment summary."""
        return self.segmentation._aggregate_segments(df, aid_stations_km_set)

    def _rebuild_segments_df_with_types(self, df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
        """Rebuild df with updated segment IDs and types."""
        return self.segmentation._rebuild_segments_df_with_types(df, segments_df)

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
        return self.segmentation._recompute_and_reclassify_segments(segments_df, df, aid_stations_km_set)

    def _compute_segment_metrics(
        self, df: pd.DataFrame, segments_df: pd.DataFrame, aid_stations_km_set: set[float]
    ) -> pd.DataFrame:
        """Compute final segment metrics with distance-equivalent and time.
        
        Uses segment boundaries from segments_df to ensure continuity.
        """
        return self.segmentation._compute_segment_metrics(df, segments_df, aid_stations_km_set)

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
        return self.segmentation.merge_segments_manually(segments_df, segment_ids, metrics_df)

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
        return self.aid_station_stats.compute_segment_stats_between(start_km, end_km, segments_df)

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
        return self.aid_station_stats.compute_aid_station_stats(aid_stations_km, segments_df)

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
        return self.aid_station_stats.compute_cumulative_stats_at_aid_stations(aid_stations_km, segments_df)

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
        return self.aid_station_stats.compute_aid_station_times(aid_stations_km, segments_df)

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
        return self.race_persistence.save_race(
            race_name,
            aid_stations_km,
            segments_df,
            race_id=race_id,
            aid_stations_times=aid_stations_times,
        )

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
        return self.race_persistence.load_race(race_id)

    def list_races(self) -> pd.DataFrame:
        """List all saved races.

        Returns:
            DataFrame with raceId, name, createdAt columns
        """
        return self.race_persistence.list_races()

    def compare_race_segments_with_activity(
        self, race_id: str, activity_timeseries_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Compare planned race segments with actual activity performance.

        Args:
            race_id: Race ID to load planned segments from
            activity_timeseries_df: Activity timeseries DataFrame with cumulated_distance, timestamp, paceKmh

        Returns:
            DataFrame with comparison metrics for each segment, or None if race not found
        """
        return self.activity_comparison.compare_race_segments_with_activity(
            race_id, activity_timeseries_df
        )

    def link_race_to_activity(self, activity_id: str, race_id: str) -> str:
        """Link a race pacing to an activity.

        Args:
            activity_id: Activity ID
            race_id: Race ID

        Returns:
            Link ID
        """
        return self.activity_comparison.link_race_to_activity(activity_id, race_id)

    def unlink_race_from_activity(self, activity_id: str) -> None:
        """Unlink race pacing from an activity.

        Args:
            activity_id: Activity ID
        """
        self.activity_comparison.unlink_race_from_activity(activity_id)

    def get_linked_race_id(self, activity_id: str) -> Optional[str]:
        """Get the race ID linked to an activity.

        Args:
            activity_id: Activity ID

        Returns:
            Race ID if linked, None otherwise
        """
        return self.activity_comparison.get_linked_race_id(activity_id)

