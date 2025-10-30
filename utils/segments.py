"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Segment merging utilities for elevation plots.

Extracted from graph/elevation.py to keep visualization helpers reusable.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from streamlit.logger import get_logger

logger = get_logger(__name__)


def merge_small_segments(plot_df: pd.DataFrame, min_size: int = 3) -> pd.DataFrame:
    """Merge segments smaller than min_size with adjacent segments.

    Merges small segments with the adjacent segment that has the closest grade value.

    Args:
        plot_df: DataFrame with 'segment' and 'grade_ma_10' columns
        min_size: Minimum segment size (default: 3)

    Returns:
        DataFrame with merged segments
    """
    df = plot_df.copy()
    max_iterations = len(df)  # Safety limit
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        segment_sizes = df.groupby("segment").size()
        small_segments = segment_sizes[segment_sizes < min_size].index.tolist()

        if not small_segments:
            break  # No more small segments to merge

        # Process small segments
        for small_seg_id in small_segments:
            small_seg_mask = df["segment"] == small_seg_id
            small_seg_indices = df[small_seg_mask].index.tolist()

            if not small_seg_indices:
                continue

            # Get the grade value of the small segment (use median)
            small_seg_grades = df.loc[small_seg_indices, "grade_ma_10"]
            small_seg_grade = small_seg_grades.median()

            # Find adjacent segments
            first_idx = small_seg_indices[0]
            last_idx = small_seg_indices[-1]

            # Check previous segment
            prev_seg_id: Optional[int] = None
            if first_idx > 0:
                prev_seg_id = df.loc[first_idx - 1, "segment"]
                if prev_seg_id == small_seg_id:
                    prev_seg_id = None

            # Check next segment
            next_seg_id: Optional[int] = None
            if last_idx < len(df) - 1:
                next_seg_id = df.loc[last_idx + 1, "segment"]
                if next_seg_id == small_seg_id:
                    next_seg_id = None

            # Choose the adjacent segment with closest grade value
            merge_target: Optional[int] = None
            min_grade_diff = float("inf")

            if prev_seg_id is not None:
                prev_seg_mask = df["segment"] == prev_seg_id
                prev_seg_grades = df.loc[prev_seg_mask, "grade_ma_10"]
                prev_seg_grade = prev_seg_grades.median()
                grade_diff = abs(prev_seg_grade - small_seg_grade)
                if grade_diff < min_grade_diff:
                    min_grade_diff = grade_diff
                    merge_target = prev_seg_id

            if next_seg_id is not None:
                next_seg_mask = df["segment"] == next_seg_id
                next_seg_grades = df.loc[next_seg_mask, "grade_ma_10"]
                next_seg_grade = next_seg_grades.median()
                grade_diff = abs(next_seg_grade - small_seg_grade)
                if grade_diff < min_grade_diff:
                    min_grade_diff = grade_diff
                    merge_target = next_seg_id

            # Merge the small segment with the target segment
            if merge_target is not None:
                df.loc[small_seg_mask, "segment"] = merge_target
                # Update grade category to match the merged segment
                merged_seg_mask = df["segment"] == merge_target
                merged_grades = df.loc[merged_seg_mask, "grade_ma_10"]
                merged_grade = merged_grades.median()
                # Attempt to preserve existing category column if present
                if "grade_category" in df.columns:
                    df.loc[small_seg_mask, "grade_category"] = df.loc[merged_seg_mask, "grade_category"].mode().iloc[0]
                logger.debug(
                    f"Merged segment {small_seg_id} (size={len(small_seg_indices)}, "
                    f"grade={small_seg_grade:.3f}) into segment {merge_target} "
                    f"(grade={merged_grade:.3f}, diff={min_grade_diff:.3f})"
                )
            else:
                # No adjacent segment found, skip (shouldn't happen unless it's the only segment)
                logger.warning(f"Could not find adjacent segment to merge segment {small_seg_id}")
                break

    if iteration >= max_iterations:
        logger.warning(f"Reached max iterations ({max_iterations}) while merging small segments")

    return df


def merge_adjacent_same_color(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Merge adjacent segments that have the same grade category (color).

    After merging small segments, adjacent segments with the same color should be
    merged together for a cleaner visualization.

    Args:
        plot_df: DataFrame with 'segment' and 'grade_category' columns

    Returns:
        DataFrame with merged adjacent same-color segments
    """
    df = plot_df.copy()
    max_iterations = len(df)  # Safety limit
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        changed = False

        # Get unique segments sorted by their first occurrence
        segments = df["segment"].unique()
        if len(segments) <= 1:
            break  # Only one segment or no segments

        # Check each segment against the next one
        for i in range(len(segments) - 1):
            seg1_id = segments[i]
            seg2_id = segments[i + 1]

            # Get the indices for both segments
            seg1_mask = df["segment"] == seg1_id
            seg2_mask = df["segment"] == seg2_id

            seg1_indices = df[seg1_mask].index.tolist()
            seg2_indices = df[seg2_mask].index.tolist()

            if not seg1_indices or not seg2_indices:
                continue

            # Check if segments are adjacent (seg1 ends right before seg2 starts)
            seg1_last_idx = seg1_indices[-1]
            seg2_first_idx = seg2_indices[0]

            if seg1_last_idx + 1 != seg2_first_idx:
                continue  # Not adjacent

            # Check if they have the same grade category
            seg1_category = df.loc[seg1_indices[0], "grade_category"]
            seg2_category = df.loc[seg2_indices[0], "grade_category"]

            if seg1_category == seg2_category:
                # Merge seg2 into seg1
                df.loc[seg2_mask, "segment"] = seg1_id
                changed = True
                logger.debug(
                    f"Merged adjacent segments {seg1_id} and {seg2_id} "
                    f"(same category: {seg1_category})"
                )

        if not changed:
            break  # No more merges possible

    if iteration >= max_iterations:
        logger.warning(
            f"Reached max iterations ({max_iterations}) while merging adjacent same-color segments"
        )

    # Renumber segments to be consecutive starting from 0
    unique_segments = df["segment"].unique()
    segment_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_segments))}
    df["segment"] = df["segment"].map(segment_mapping)

    return df


