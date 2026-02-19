"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Pacer segments visualization with elevation profile and grade-based coloring.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.logger import get_logger

from utils.constants import PACER_SEGMENT_COLORS

logger = get_logger(__name__)

def render_pacer_segments(
    metrics_df: pd.DataFrame, segments_df: pd.DataFrame, aid_stations_km: list[float]
) -> None:
    """Render elevation profile with grade-based segments and aid station markers.

    Args:
        metrics_df: Preprocessed metrics DataFrame with cumulated_distance, elevationM_ma_5
        segments_df: DataFrame with segment metrics (segmentId, type, startKm, endKm, etc.)
        aid_stations_km: List of aid station positions in km
    """
    if metrics_df.empty or segments_df.empty:
        st.warning("Données insuffisantes pour afficher le profil.")
        return

    # Use metrics_df directly - don't filter or preprocess
    if "cumulated_distance" not in metrics_df.columns or "elevationM_ma_5" not in metrics_df.columns:
        st.warning("Données insuffisantes pour le graphique.")
        return
    
    # Build plot_df directly from metrics_df - preserve all points
    plot_df = metrics_df[["cumulated_distance", "elevationM_ma_5"]].copy()
    if "grade_ma_10" in metrics_df.columns:
        plot_df["grade_ma_10"] = metrics_df["grade_ma_10"]
    else:
        plot_df["grade_ma_10"] = 0.0
    
    # Remove only truly invalid points (NaN in both distance and elevation)
    plot_df = plot_df.dropna(subset=["cumulated_distance", "elevationM_ma_5"]).reset_index(drop=True)
    if plot_df.empty:
        st.warning("Erreur lors de la préparation des données pour le graphique.")
        return
    
    # Calculate y-axis bounds
    y_min_val = float(plot_df["elevationM_ma_5"].min() - 20)
    y_max_val = float(plot_df["elevationM_ma_5"].max() + 20)
    
    # Calculate x-axis bounds from SEGMENTS, not just plot_df
    # This ensures all segments are visible even if they extend beyond the data points
    x_min_val = float(segments_df["startKm"].min() if not segments_df.empty else plot_df["cumulated_distance"].min())
    x_max_val = float(segments_df["endKm"].max() if not segments_df.empty else plot_df["cumulated_distance"].max())

    # Base chart encoding with explicit x-axis domain from segments
    base = alt.Chart(plot_df).encode(
        x=alt.X(
            "cumulated_distance:Q",
            title="Distance (km)",
            scale=alt.Scale(domain=[x_min_val, x_max_val], nice=True)
        ),
        y=alt.Y("elevationM_ma_5:Q", title="Altitude (m)", scale=alt.Scale(domain=[y_min_val, y_max_val], nice=True)),
    )

    # Render segments EXACTLY as they are in segments_df - NO sorting, NO filtering
    charts = []
    
    logger.info(f"Rendering {len(segments_df)} segments from DataFrame")
    logger.info(f"Segment IDs: {segments_df['segmentId'].tolist() if 'segmentId' in segments_df.columns else 'N/A'}")
    
    # Iterate through segments_df AS IS - no modifications
    segments_rendered = 0
    for idx, seg in segments_df.iterrows():
        # Get segment type and color
        seg_type = seg.get("type", "unknown")
        color = PACER_SEGMENT_COLORS.get(seg_type, "#808080")
        
        start_km = float(seg["startKm"])
        end_km = float(seg["endKm"])
        seg_id = seg.get("segmentId", idx)
        
        # Get points for this segment - use inclusive boundaries
        seg_points = plot_df[
            (plot_df["cumulated_distance"] >= start_km)
            & (plot_df["cumulated_distance"] <= end_km)
        ].copy()

        if seg_points.empty:
            logger.warning(f"Segment {seg_id} has no points - skipping")
            continue

        # Add segment ID to points to prevent Altair from merging adjacent segments
        seg_points["_segment_id"] = seg_id

        logger.debug(f"Segment {seg_id}: {seg_type} ({start_km:.3f}-{end_km:.3f} km), {len(seg_points)} points")
        
        # Create segment chart with unique segment ID to prevent merging
        # Use explicit x-axis domain to ensure all segments are visible
        area_segment = (
            alt.Chart(seg_points)
            .mark_area(opacity=0.4, color=color)
            .encode(
                x=alt.X("cumulated_distance:Q", title="Distance (km)", scale=alt.Scale(domain=[x_min_val, x_max_val])),
                y=alt.Y("elevationM_ma_5:Q", title="Altitude (m)", scale=alt.Scale(domain=[y_min_val, y_max_val])),
                y2=alt.datum(y_min_val),
                # Add segment ID to encoding to force separation
                detail=alt.Detail("_segment_id:N"),
                tooltip=[
                    alt.Tooltip("cumulated_distance:Q", title="Distance", format=".2f"),
                    alt.Tooltip("elevationM_ma_5:Q", title="Altitude", format=".0f"),
                    alt.Tooltip("grade_ma_10:Q", title="Pente", format=".1%"),
                ],
            )
        )
        charts.append(area_segment)
        segments_rendered += 1
    
    logger.info(f"Actually rendered {segments_rendered} segment charts")

    # Add elevation line with explicit x-axis domain
    elevation_line = base.mark_line(color="#000000", strokeWidth=2).encode(
        x=alt.X("cumulated_distance:Q", title="Distance (km)", scale=alt.Scale(domain=[x_min_val, x_max_val])),
        y=alt.Y("elevationM_ma_5:Q", title="Altitude (m)", scale=alt.Scale(domain=[y_min_val, y_max_val])),
        tooltip=[
            alt.Tooltip("cumulated_distance:Q", title="Distance", format=".2f"),
            alt.Tooltip("elevationM_ma_5:Q", title="Altitude", format=".0f"),
        ],
    )
    charts.append(elevation_line)

    # Add aid station markers
    if aid_stations_km:
        aid_data = []
        for idx, aid_km in enumerate(sorted(aid_stations_km), start=1):
            # Find elevation at aid station
            aid_elev = float(plot_df[plot_df["cumulated_distance"] <= aid_km]["elevationM_ma_5"].iloc[-1]) if len(
                plot_df[plot_df["cumulated_distance"] <= aid_km]
            ) > 0 else y_min_val

            aid_data.append({"distance": float(aid_km), "elevation": aid_elev, "label": f"RAV {idx}"})

        aid_df = pd.DataFrame(aid_data)

        aid_rules = (
            alt.Chart(aid_df)
            .mark_rule(strokeWidth=2, strokeDash=[5, 5], color="#3b82f6")
            .encode(
                x=alt.X("distance:Q", title="Distance (km)"),
                tooltip=[
                    alt.Tooltip("distance:Q", title="Distance", format=".2f"),
                    alt.Tooltip("label:N", title="Ravitaillement"),
                ],
            )
        )

        aid_labels = (
            alt.Chart(aid_df)
            .mark_text(align="left", dx=5, dy=-5, fontSize=12, fontWeight="bold", color="#3b82f6")
            .encode(
                x=alt.X("distance:Q", title="Distance (km)"),
                y=alt.Y("elevation:Q", title="Altitude (m)", scale=alt.Scale(domain=[y_min_val, y_max_val])),
                text=alt.Text("label:N"),
            )
        )

        charts.extend([aid_rules, aid_labels])

    # Combine all charts
    try:
        combined = alt.layer(*charts).properties(
            width=800,
            height=400,
            title="Profil d'élévation avec segments de pente",
        )

        st.altair_chart(combined, theme=None, use_container_width=True)
    except Exception as e:
        logger.error(f"Failed to render pacer segments chart: {e}", exc_info=True)
        st.error(f"Erreur lors du rendu du graphique: {str(e)}")

    # Add legend
    st.markdown("**Légende des segments :**")
    legend_items = [
        ("🔴 Montée raide", "≥ 10%"),
        ("🟠 Montée", "2% à 10%"),
        ("⚪ Plat", "< 2% ou < 10m/1km"),
        ("🟢 Descente", "-2% à -25%"),
        ("🟢 Descente raide", "≤ -25%"),
    ]

    cols = st.columns(5)
    for idx, (label, desc) in enumerate(legend_items):
        with cols[idx]:
            st.caption(f"{label}: {desc}")

