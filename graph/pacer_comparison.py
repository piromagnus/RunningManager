"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Comparison visualization for race pacing vs actual performance.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.logger import get_logger

from utils.constants import PACER_SEGMENT_COLORS

logger = get_logger(__name__)


def render_comparison_elevation_profile(
    metrics_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    aid_stations_km: list[float],
) -> None:
    """Render elevation profile with segments showing comparison tooltips.

    Args:
        metrics_df: Preprocessed metrics DataFrame with cumulated_distance, elevationM_ma_5
        segments_df: DataFrame with planned segment metrics
        comparison_df: DataFrame with comparison metrics (planned vs actual) - uses actual distances
        aid_stations_km: List of aid station positions in km
    """
    if metrics_df.empty or comparison_df.empty:
        st.warning("Données insuffisantes pour afficher le profil de comparaison.")
        return

    if "cumulated_distance" not in metrics_df.columns or "elevationM_ma_5" not in metrics_df.columns:
        st.warning("Données insuffisantes pour le graphique.")
        return

    # Build plot_df directly from metrics_df
    plot_df = metrics_df[["cumulated_distance", "elevationM_ma_5"]].copy()
    if "grade_ma_10" in metrics_df.columns:
        plot_df["grade_ma_10"] = metrics_df["grade_ma_10"]
    else:
        plot_df["grade_ma_10"] = 0.0

    plot_df = plot_df.dropna(subset=["cumulated_distance", "elevationM_ma_5"]).reset_index(drop=True)
    if plot_df.empty:
        st.warning("Erreur lors de la préparation des données pour le graphique.")
        return

    # Calculate y-axis bounds
    y_min_val = float(plot_df["elevationM_ma_5"].min() - 20)
    y_max_val = float(plot_df["elevationM_ma_5"].max() + 20)

    # Calculate x-axis bounds from actual segments (comparison_df)
    x_min_val = float(comparison_df["startKm"].min() if not comparison_df.empty else plot_df["cumulated_distance"].min())
    x_max_val = float(comparison_df["endKm"].max() if not comparison_df.empty else plot_df["cumulated_distance"].max())

    # Base chart encoding
    base = alt.Chart(plot_df).encode(
        x=alt.X(
            "cumulated_distance:Q",
            title="Distance (km)",
            scale=alt.Scale(domain=[x_min_val, x_max_val], nice=True)
        ),
        y=alt.Y("elevationM_ma_5:Q", title="Altitude (m)", scale=alt.Scale(domain=[y_min_val, y_max_val], nice=True)),
    )

    # Render segments with comparison tooltips - use actual segments from comparison_df
    charts = []

    for idx, seg in comparison_df.iterrows():
        seg_type = seg.get("type", "unknown")
        color = PACER_SEGMENT_COLORS.get(seg_type, "#808080")

        # Use actual distances from comparison_df (GPS-matched)
        start_km = float(seg["startKm"])
        end_km = float(seg["endKm"])
        seg_id = int(seg.get("segmentId", idx))

        # Get points for this segment using actual distances
        seg_points = plot_df[
            (plot_df["cumulated_distance"] >= start_km)
            & (plot_df["cumulated_distance"] <= end_km)
        ].copy()

        if seg_points.empty:
            continue

        # Add comparison metrics to each point for tooltip
        distance_km = end_km - start_km
        time_delta_sec = float(seg.get("timeDeltaSec", 0) or 0) if pd.notna(seg.get("timeDeltaSec")) else None
        
        # Get actual and planned speeds to compute deltas (matching table logic)
        planned_speed_kmh = float(seg.get("plannedSpeedKmh", 0) or 0)
        actual_speed_kmh = seg.get("actualSpeedKmh")
        planned_speed_eq_kmh = float(seg.get("plannedSpeedEqKmh", 0) or 0)
        actual_speed_eq_kmh = seg.get("actualSpeedEqKmh")
        
        # Compute deltas using same logic as table
        if actual_speed_kmh is not None and planned_speed_kmh > 0:
            speed_delta = float(actual_speed_kmh) - planned_speed_kmh
        else:
            speed_delta = float(seg.get("speedDeltaKmh", 0) or 0) if pd.notna(seg.get("speedDeltaKmh")) else None
        
        if actual_speed_eq_kmh is not None and planned_speed_eq_kmh > 0:
            speed_eq_delta = float(actual_speed_eq_kmh) - planned_speed_eq_kmh
        else:
            speed_eq_delta = float(seg.get("speedEqDeltaKmh", 0) or 0) if pd.notna(seg.get("speedEqDeltaKmh")) else None

        # Convert time delta to minutes for display
        time_delta_min = (time_delta_sec / 60.0) if time_delta_sec is not None else 0.0

        seg_points["_segment_id"] = seg_id
        seg_points["_segment_distance"] = distance_km
        seg_points["_time_delta_min"] = time_delta_min
        seg_points["_speed_delta"] = speed_delta if speed_delta is not None else 0.0
        seg_points["_speed_eq_delta"] = speed_eq_delta if speed_eq_delta is not None else 0.0

        # Create segment chart with enriched tooltip
        area_segment = (
            alt.Chart(seg_points)
            .mark_area(opacity=0.4, color=color)
            .encode(
                x=alt.X("cumulated_distance:Q", title="Distance (km)", scale=alt.Scale(domain=[x_min_val, x_max_val])),
                y=alt.Y("elevationM_ma_5:Q", title="Altitude (m)", scale=alt.Scale(domain=[y_min_val, y_max_val])),
                y2=alt.datum(y_min_val),
                detail=alt.Detail("_segment_id:N"),
                tooltip=[
                    alt.Tooltip("_segment_id:Q", title="Segment", format=".0f"),  # Segment number at top
                    alt.Tooltip("cumulated_distance:Q", title="Distance", format=".2f"),
                    alt.Tooltip("elevationM_ma_5:Q", title="Altitude", format=".0f"),
                    alt.Tooltip("_segment_distance:Q", title="Distance segment", format=".2f"),
                    alt.Tooltip("_time_delta_min:Q", title="Δ Temps", format="+.1f"),
                    alt.Tooltip("_speed_delta:Q", title="Δ Vitesse", format="+.1f"),
                    alt.Tooltip("_speed_eq_delta:Q", title="Δ Vitesse-eq", format="+.1f"),
                ],
            )
        )
        charts.append(area_segment)

    # Add elevation line
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
            aid_elev = (
                float(plot_df[plot_df["cumulated_distance"] <= aid_km]["elevationM_ma_5"].iloc[-1])
                if len(plot_df[plot_df["cumulated_distance"] <= aid_km]) > 0
                else y_min_val
            )
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
            title="Profil d'élévation - Comparaison planifié vs réel",
        )

        st.altair_chart(combined, theme=None, use_container_width=True)
    except Exception as e:
        logger.error(f"Failed to render comparison elevation chart: {e}", exc_info=True)
        st.error(f"Erreur lors du rendu du graphique: {str(e)}")


def render_delta_bar_chart(
    comparison_df: pd.DataFrame,
    planned_segments_df: pd.DataFrame | None = None,
    pacer_service=None,
) -> None:
    """Render bar chart showing deltas (time, speed, speed-eq) for each segment.

    Args:
        comparison_df: DataFrame with comparison metrics
        planned_segments_df: Optional DataFrame with planned segment metrics (for recomputing missing deltas)
        pacer_service: Optional PacerService instance (for recomputing missing deltas)
    """
    if comparison_df.empty:
        return

    # Prepare data for bar chart - use deltas from comparison_df, recompute if missing
    chart_data = []
    for _, row in comparison_df.iterrows():
        seg_id = int(row["segmentId"])
        seg_type = row.get("type", "unknown")
        start_km = float(row["startKm"])
        end_km = float(row["endKm"])
        distance_km = end_km - start_km

        # Convert time delta from seconds to minutes
        time_delta_sec = float(row.get("timeDeltaSec", 0) or 0) if pd.notna(row.get("timeDeltaSec")) else None
        time_delta_min = (time_delta_sec / 60.0) if time_delta_sec is not None else 0.0

        # Use same logic as table: recompute deltas from planned/actual speeds
        # First try to use delta from comparison_df, but recompute if missing using same logic as table
        speed_delta = None
        speed_eq_delta = None
        
        if planned_segments_df is not None and pacer_service is not None:
            planned_seg_mask = planned_segments_df[planned_segments_df["segmentId"] == seg_id]
            planned_seg = planned_seg_mask.iloc[0] if not planned_seg_mask.empty else None
            
            # Get planned values
            planned_time_sec = float(row.get("plannedTimeSec", 0) or 0)
            planned_speed_kmh = float(row.get("plannedSpeedKmh", 0) or 0)
            planned_speed_eq_kmh = float(row.get("plannedSpeedEqKmh", 0) or 0)
            elev_gain = float(planned_seg.get("elevGainM", 0) or 0) if planned_seg is not None else 0.0
            
            # Compute missing planned speed if needed
            if planned_speed_kmh == 0 and planned_time_sec > 0:
                planned_speed_kmh = (distance_km / planned_time_sec) * 3600
            
            # Compute missing planned speed-eq if needed
            if planned_speed_eq_kmh == 0 and planned_time_sec > 0:
                distance_eq = pacer_service.planner.compute_distance_eq_km(distance_km, elev_gain)
                if distance_eq > 0:
                    planned_speed_eq_kmh = (distance_eq / planned_time_sec) * 3600
            
            # Get actual values
            actual_time_sec = row.get("actualTimeSec")
            actual_speed_kmh = row.get("actualSpeedKmh")
            actual_speed_eq_kmh = row.get("actualSpeedEqKmh")
            
            # Compute missing actual speed if needed
            if actual_speed_kmh is None and actual_time_sec and actual_time_sec > 0:
                actual_speed_kmh = (distance_km / actual_time_sec) * 3600
            
            # Compute missing actual speed-eq if needed
            if actual_speed_eq_kmh is None and actual_speed_kmh is not None and actual_time_sec and actual_time_sec > 0:
                # Use planned elevation gain as fallback (same as table logic)
                actual_distance_eq_km = pacer_service.planner.compute_distance_eq_km(distance_km, elev_gain)
                if actual_distance_eq_km > 0:
                    actual_speed_eq_kmh = (actual_distance_eq_km / actual_time_sec) * 3600
            
            # Compute deltas (matching table logic)
            if actual_speed_kmh is not None and planned_speed_kmh > 0:
                speed_delta = float(actual_speed_kmh) - planned_speed_kmh
            else:
                # Fallback to comparison_df value if available
                speed_delta_from_df = row.get("speedDeltaKmh")
                if pd.notna(speed_delta_from_df) and speed_delta_from_df is not None:
                    speed_delta = float(speed_delta_from_df)
            
            if actual_speed_eq_kmh is not None and planned_speed_eq_kmh > 0:
                speed_eq_delta = float(actual_speed_eq_kmh) - planned_speed_eq_kmh
            else:
                # Fallback to comparison_df value if available
                speed_eq_delta_from_df = row.get("speedEqDeltaKmh")
                if pd.notna(speed_eq_delta_from_df) and speed_eq_delta_from_df is not None:
                    speed_eq_delta = float(speed_eq_delta_from_df)
        else:
            # Fallback: use values directly from comparison_df
            speed_delta_from_df = row.get("speedDeltaKmh")
            speed_eq_delta_from_df = row.get("speedEqDeltaKmh")
            speed_delta = float(speed_delta_from_df) if pd.notna(speed_delta_from_df) and speed_delta_from_df is not None else None
            speed_eq_delta = float(speed_eq_delta_from_df) if pd.notna(speed_eq_delta_from_df) and speed_eq_delta_from_df is not None else None

        chart_data.append({
            "segmentId": seg_id,
            "label": f"#{seg_id} ({seg_type})",
            "distance": f"{start_km:.1f}-{end_km:.1f}",
            "timeDelta": time_delta_min,
            "speedDelta": speed_delta if speed_delta is not None else 0.0,
            "speedEqDelta": speed_eq_delta if speed_eq_delta is not None else 0.0,
        })

    chart_df = pd.DataFrame(chart_data)

    # Create bar chart with different metrics
    base = alt.Chart(chart_df).encode(
        x=alt.X("label:N", title="Segment", sort=None),
    )

    # Time delta bars (green for negative, red for positive) - in minutes
    time_bars = base.mark_bar().encode(
        y=alt.Y("timeDelta:Q", title="Δ Temps (min)"),
        color=alt.condition(
            alt.datum.timeDelta < 0,
            alt.value("#22c55e"),  # green for negative (faster)
            alt.value("#dc2626"),  # red for positive (slower)
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Segment"),
            alt.Tooltip("distance:N", title="Distance"),
            alt.Tooltip("timeDelta:Q", title="Δ Temps", format="+.1f"),
        ],
    )

    st.altair_chart(time_bars.properties(width=800, height=300, title="Δ Temps par segment"), theme=None, use_container_width=True)

    # Speed deltas (two charts)
    col1, col2 = st.columns(2)

    with col1:
        speed_bars = base.mark_bar(color="#3b82f6").encode(
            y=alt.Y("speedDelta:Q", title="Δ Vitesse (km/h)"),
            tooltip=[
                alt.Tooltip("label:N", title="Segment"),
                alt.Tooltip("distance:N", title="Distance"),
                alt.Tooltip("speedDelta:Q", title="Δ Vitesse", format="+.1f"),
            ],
        )
        st.altair_chart(speed_bars.properties(width=400, height=300, title="Δ Vitesse"), theme=None, use_container_width=True)

    with col2:
        speed_eq_bars = base.mark_bar(color="#8b5cf6").encode(
            y=alt.Y("speedEqDelta:Q", title="Δ Vitesse-eq (km/h)"),
            tooltip=[
                alt.Tooltip("label:N", title="Segment"),
                alt.Tooltip("distance:N", title="Distance"),
                alt.Tooltip("speedEqDelta:Q", title="Δ Vitesse-eq", format="+.1f"),
            ],
        )
        st.altair_chart(speed_eq_bars.properties(width=400, height=300, title="Δ Vitesse-eq"), theme=None, use_container_width=True)

