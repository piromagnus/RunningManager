"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Elevation profile visualization with grade-based coloring.

Handles rendering of elevation profiles and grade histograms.
"""

from __future__ import annotations

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit.logger import get_logger

from utils.segments import merge_adjacent_same_color, merge_small_segments

logger = get_logger(__name__)

# Grade category color mapping
GRADE_COLOR_MAPPING = {
    "grade_lt_neg_0_5": "#001f3f",  # darkblue
    "grade_lt_neg_0_25": "#004d26",  # darkgreen
    "grade_lt_neg_0_05": "#22c55e",  # green
    "grade_neutral": "#d1d5db",  # lightgray
    "grade_lt_0_1": "#eab308",  # yellow
    "grade_lt_0_25": "#f97316",  # orange
    "grade_lt_0_5": "#dc2626",  # red
    "grade_ge_0_5": "#000000",  # black
    "unknown": "#808080",  # gray
}

GRADE_HISTOGRAM_COLORS = GRADE_COLOR_MAPPING


def get_grade_category(grade_val: float) -> str:
    """Map grade value to category for coloring.

    Args:
        grade_val: Grade value (elevation change / distance)

    Returns:
        Category string for grade classification
    """
    if pd.isna(grade_val):
        return "unknown"
    if grade_val < -0.5:
        return "grade_lt_neg_0_5"
    elif -0.5 <= grade_val < -0.25:
        return "grade_lt_neg_0_25"
    elif -0.25 <= grade_val < -0.05:
        return "grade_lt_neg_0_05"
    elif -0.05 <= grade_val < 0.05:
        return "grade_neutral"
    elif 0.05 <= grade_val < 0.1:
        return "grade_lt_0_1"
    elif 0.1 <= grade_val < 0.25:
        return "grade_lt_0_25"
    elif 0.25 <= grade_val < 0.5:
        return "grade_lt_0_5"
    else:
        return "grade_ge_0_5"


def prepare_elevation_plot_data(metrics_df: pd.DataFrame) -> pd.DataFrame | None:
    """Prepare preprocessed metrics DataFrame for elevation plotting.

    Validates and cleans data, adds required columns for plotting.

    Args:
        metrics_df: Preprocessed metrics DataFrame from elevation_preprocessing

    Returns:
        Prepared DataFrame ready for plotting, or None if insufficient data
    """
    required_cols = [
        "cumulated_distance",
        "elevationM_ma_5",
        "grade_ma_10",
        "speed_km_h",
        "hr",
        "cumulated_duration_seconds",
    ]

    # Check which columns exist in df
    available_cols = [col for col in required_cols if col in metrics_df.columns]
    missing_req_cols = [col for col in required_cols if col not in metrics_df.columns]

    logger.debug(f"Required columns: {required_cols}")
    logger.debug(f"Available columns: {available_cols}")
    if missing_req_cols:
        logger.warning(f"Missing required columns: {missing_req_cols}")

    if not available_cols:
        logger.error("No required columns found in DataFrame")
        return None

    # Create plot_df with only available columns
    plot_df = metrics_df[available_cols].copy()

    logger.debug(f"Initial plot_df shape: {plot_df.shape}")

    # Drop rows where essential columns are NaN
    if "cumulated_distance" not in plot_df.columns or "elevationM_ma_5" not in plot_df.columns:
        logger.error("Essential columns (cumulated_distance or elevationM_ma_5) missing")
        return None

    plot_df = plot_df.dropna(subset=["cumulated_distance", "elevationM_ma_5"])
    logger.debug(f"After dropping NaN in essential columns: {plot_df.shape}")

    if plot_df.empty:
        logger.warning("Plot DataFrame is empty after dropping NaN")
        return None

    # Fill NaN values for other columns (don't drop rows)
    if "grade_ma_10" in plot_df.columns:
        plot_df["grade_ma_10"] = plot_df["grade_ma_10"].fillna(0)
    else:
        plot_df["grade_ma_10"] = 0
        logger.debug("Created grade_ma_10 column with default value 0")

    if "speed_km_h" not in plot_df.columns:
        plot_df["speed_km_h"] = 0
    else:
        plot_df["speed_km_h"] = plot_df["speed_km_h"].fillna(0)

    if "hr" not in plot_df.columns:
        plot_df["hr"] = 0
    else:
        plot_df["hr"] = plot_df["hr"].fillna(0)

    if "cumulated_duration_seconds" not in plot_df.columns:
        plot_df["cumulated_duration_seconds"] = 0
    else:
        plot_df["cumulated_duration_seconds"] = plot_df["cumulated_duration_seconds"].fillna(0)

    plot_df = plot_df.reset_index(drop=True)

    logger.debug(f"After filling NaN: {plot_df.shape}, columns: {list(plot_df.columns)}")

    # Calculate Y-axis bounds
    try:
        min_elev = plot_df["elevationM_ma_5"].min()
        max_elev = plot_df["elevationM_ma_5"].max()
        elevation_range = max(max_elev - min_elev, 1)
        padding = max(20, elevation_range * 0.05)
        y_min = min_elev - padding
        y_max = max_elev + padding
        logger.debug(
            "Y-axis bounds with padding: min=%.2f, max=%.2f (range=%.2f, padding=%.2f)",
            y_min,
            y_max,
            elevation_range,
            padding,
        )
        plot_df["_y_min"] = y_min
        plot_df["_y_max"] = y_max
    except Exception as e:
        logger.error(f"Failed to calculate Y-axis bounds: {e}", exc_info=True)
        return None

    # Add grade category for segment-based coloring
    try:
        plot_df["grade_category"] = plot_df["grade_ma_10"].apply(get_grade_category)
        categories = plot_df["grade_category"].unique()
        logger.debug("Grade categories created. Unique categories: %s", categories)
    except Exception as e:
        logger.error(f"Failed to create grade categories: {e}", exc_info=True)
        return None

    # Prepare numeric values for tooltips
    try:
        # Calculate grade percentage for tooltip
        plot_df["grade_pct"] = plot_df["grade_ma_10"] * 100

        # Calculate time in hours for tooltip
        plot_df["time_hours"] = plot_df["cumulated_duration_seconds"] / 3600.0

        # Ensure all numeric columns are properly filled
        plot_df["speed_km_h"] = plot_df["speed_km_h"].fillna(0)
        plot_df["hr"] = plot_df["hr"].fillna(0)
        plot_df["time_hours"] = plot_df["time_hours"].fillna(0)
        plot_df["grade_pct"] = plot_df["grade_pct"].fillna(0)

        logger.debug("Tooltip numeric columns prepared successfully")
    except Exception as e:
        logger.error(f"Failed to prepare tooltip values: {e}", exc_info=True)
        return None

    # Create segments by grouping consecutive rows with the same grade category
    try:
        plot_df["segment"] = (plot_df["grade_category"] != plot_df["grade_category"].shift()).cumsum()
        num_segments = plot_df["segment"].max() + 1 if len(plot_df) > 0 else 0
        logger.debug(f"Created {num_segments} initial segments")

        # Merge small segments (< min_size points) with adjacent segments
        plot_df = merge_small_segments(plot_df, min_size=10)
        num_segments_after_merge = plot_df["segment"].max() + 1 if len(plot_df) > 0 else 0
        logger.debug(f"After merging small segments: {num_segments_after_merge} segments")

        # Merge adjacent segments with the same grade category (color)
        plot_df = merge_adjacent_same_color(plot_df)
        num_segments_final = plot_df["segment"].max() + 1 if len(plot_df) > 0 else 0
        logger.debug(f"After merging adjacent same-color segments: {num_segments_final} segments")
    except Exception as e:
        logger.error(f"Failed to create segments: {e}", exc_info=True)
        return None

    return plot_df


def _merge_small_segments(plot_df: pd.DataFrame, min_size: int = 3) -> pd.DataFrame:
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
            prev_seg_id = None
            if first_idx > 0:
                prev_seg_id = df.loc[first_idx - 1, "segment"]
                if prev_seg_id == small_seg_id:
                    prev_seg_id = None

            # Check next segment
            next_seg_id = None
            if last_idx < len(df) - 1:
                next_seg_id = df.loc[last_idx + 1, "segment"]
                if next_seg_id == small_seg_id:
                    next_seg_id = None

            # Choose the adjacent segment with closest grade value
            merge_target = None
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
                df.loc[small_seg_mask, "grade_category"] = get_grade_category(merged_grade)
                logger.debug(
                    f"Merged segment {small_seg_id} (size={len(small_seg_indices)}, "
                    f"grade={small_seg_grade:.3f}) into segment {merge_target} "
                    f"(grade={merged_grade:.3f}, diff={min_grade_diff:.3f})"
                )
                # Log distribution of segment sizes with 20 bins
                segment_sizes = df.groupby("segment").size()
                logger.debug(f"Segment size distribution (20 bins):\n{segment_sizes.value_counts(bins=20)}")
            else:
                # No adjacent segment found, skip (shouldn't happen unless it's the only segment)
                logger.warning(f"Could not find adjacent segment to merge segment {small_seg_id}")
                break

    if iteration >= max_iterations:
        logger.warning(f"Reached max iterations ({max_iterations}) while merging small segments")

    return df


def _merge_adjacent_same_color(plot_df: pd.DataFrame) -> pd.DataFrame:
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


def render_elevation_profile(plot_df: pd.DataFrame) -> None:
    """Render interactive elevation profile with segment-based grade coloring.

    Args:
        plot_df: Prepared DataFrame from prepare_elevation_plot_data()
    """
    try:
        logger.info("Starting elevation profile plot")
        logger.debug(f"Input DataFrame shape: {plot_df.shape}, columns: {list(plot_df.columns)}")

        y_min = plot_df["_y_min"].iloc[0]
        y_max = plot_df["_y_max"].iloc[0]

        # Build the layered chart with one area per segment
        charts = []

        try:
            # Add area fill for each segment with full tooltips
            for segment_id in plot_df["segment"].unique():
                segment_data = plot_df[plot_df["segment"] == segment_id]
                if not segment_data.empty:
                    if "grade_category" not in segment_data.columns:
                        logger.warning(f"Grade category missing for segment {segment_id}")
                        segment_color = GRADE_COLOR_MAPPING.get("unknown", "#808080")
                    else:
                        segment_category = segment_data["grade_category"].iloc[0]
                        segment_color = GRADE_COLOR_MAPPING.get(
                            segment_category, GRADE_COLOR_MAPPING.get("unknown", "#808080")
                        )

                    area_segment = (
                        alt.Chart(segment_data)
                        .mark_area(opacity=0.4, interpolate="monotone", line=False)
                        .encode(
                            x=alt.X("cumulated_distance:Q", title="Distance", scale=alt.Scale(nice=True)),
                            y=alt.Y(
                                "elevationM_ma_5:Q",
                                title="Elevation",
                                scale=alt.Scale(domain=[y_min, y_max], nice=False),
                            ),
                            y2=alt.datum(y_min),
                            color=alt.value(segment_color),
                            tooltip=[
                                alt.Tooltip("grade_pct:Q", title="Grade", format=".2f"),
                                alt.Tooltip("cumulated_distance:Q", title="Distance (km)", format=".2f"),
                                alt.Tooltip("time_hours:Q", title="Time (h)", format=".2f"),
                                alt.Tooltip("hr:Q", title="HR (bpm)", format=".2f"),
                                alt.Tooltip("speed_km_h:Q", title="Speed (km/h)", format=".2f"),
                                alt.Tooltip("elevationM_ma_5:Q", title="Elevation (m)", format=".2f"),
                            ],
                        )
                    )
                    charts.append(area_segment)

            logger.debug(f"Created {len(charts)} area segments")
            if len(plot_df) > 0:
                logger.debug(
                    "Elevation range for rendering: min=%.2f, max=%.2f, y_min=%.2f, y_max=%.2f",
                    plot_df["elevationM_ma_5"].min(),
                    plot_df["elevationM_ma_5"].max(),
                    y_min,
                    y_max,
                )
        except Exception as e:
            logger.error(f"Failed to create area segments: {e}", exc_info=True)
            st.caption("Erreur lors de la création des segments de zone.")
            return

        # Combine all area segments (elevation profile is the top edge of the areas)
        try:
            combined = alt.layer(*charts).properties(
                width=800,
                height=400,
                title=("Profil d'élévation avec code couleur de pente"),
            )

            logger.debug("Charts combined successfully")
            logger.debug(f"Combined chart type: {type(combined)}")

            st.altair_chart(combined, theme=None, use_container_width=True)
            logger.info("Elevation profile chart rendered successfully")
        except Exception as e:
            logger.error(f"Failed to combine or render charts: {e}", exc_info=True)
            st.error(f"Erreur lors du rendu du graphique: {str(e)}")
            raise

        # Add legend
        st.markdown("**Légende des pentes :**")
        legend_items = [
            ("🔵 < -50%", "Descente très raide"),
            ("🟢 -50% à -25%", "Descente"),
            ("🟢 -25% à -5%", "Descente douce"),
            ("⚪ -5% à +5%", "Terrain plat"),
            ("🟡 +5% à +10%", "Montée douce"),
            ("🟠 +10% à +25%", "Montée"),
            ("🔴 +25% à +50%", "Montée raide"),
            ("⚫ ≥ +50%", "Montée très raide"),
        ]

        cols = st.columns(4)
        for idx, (label, desc) in enumerate(legend_items):
            with cols[idx % 4]:
                st.caption(f"{label}: {desc}")

    except Exception as e:
        logger.error(f"Unexpected error in render_elevation_profile: {e}", exc_info=True)
        st.error(f"Erreur inattendue lors du rendu du profil d'élévation: {str(e)}")
        st.caption("Vérifiez les logs pour plus de détails.")


def render_grade_histogram(metrics_df: pd.DataFrame) -> None:
    """Render a histogram of grade values with color coding.

    Args:
        metrics_df: Preprocessed metrics DataFrame with grade_ma_10 column
    """
    if "grade_ma_10" not in metrics_df.columns:
        st.caption("Pas de données de pente disponibles.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    # Get grade values
    grades = metrics_df["grade_ma_10"].dropna()

    if grades.empty:
        st.caption("Pas de données de pente disponibles.")
        return

    # Plot histogram
    n, bins, patches = ax.hist(grades, bins=50, alpha=0.7, edgecolor="black")

    # Color each bar based on its bin center
    def get_color(grade_val: float) -> str:
        if grade_val < -0.5:
            return GRADE_HISTOGRAM_COLORS["grade_lt_neg_0_5"]
        elif -0.5 <= grade_val < -0.25:
            return GRADE_HISTOGRAM_COLORS["grade_lt_neg_0_25"]
        elif -0.25 <= grade_val < -0.05:
            return GRADE_HISTOGRAM_COLORS["grade_lt_neg_0_05"]
        elif -0.05 <= grade_val < 0.05:
            return GRADE_HISTOGRAM_COLORS["grade_neutral"]
        elif 0.05 <= grade_val < 0.1:
            return GRADE_HISTOGRAM_COLORS["grade_lt_0_1"]
        elif 0.1 <= grade_val < 0.25:
            return GRADE_HISTOGRAM_COLORS["grade_lt_0_25"]
        elif 0.25 <= grade_val < 0.5:
            return GRADE_HISTOGRAM_COLORS["grade_lt_0_5"]
        else:
            return GRADE_HISTOGRAM_COLORS["grade_ge_0_5"]

    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i + 1]) / 2
        color = get_color(bin_center)
        patch.set_facecolor(color)

    ax.set_xlabel("Pente")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des pentes")
    ax.grid(True, alpha=0.3, axis="y")

    # Add vertical lines at grade boundaries
    for boundary in [-0.5, -0.25, -0.05, 0.05, 0.1, 0.25, 0.5]:
        ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

