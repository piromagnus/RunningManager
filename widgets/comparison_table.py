"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.pacer_service import PacerService
from utils.formatting import fmt_decimal, format_delta_minutes, format_session_duration


def render_comparison_table(
    comparison_df: pd.DataFrame,
    planned_segments_df: pd.DataFrame,
    pacer_service: PacerService,
) -> None:
    """Render comparison table with computed missing values and styled deltas."""
    display_data = []

    for _, row in comparison_df.iterrows():
        seg_id = int(row["segmentId"])
        planned_seg_mask = planned_segments_df[planned_segments_df["segmentId"] == seg_id]
        planned_seg = planned_seg_mask.iloc[0] if not planned_seg_mask.empty else None

        planned_time_sec = float(row.get("plannedTimeSec", 0) or 0)
        planned_speed_kmh = float(row.get("plannedSpeedKmh", 0) or 0)
        planned_speed_eq_kmh = float(row.get("plannedSpeedEqKmh", 0) or 0)
        distance_km = float(row["endKm"]) - float(row["startKm"])

        elev_gain = float(planned_seg.get("elevGainM", 0) or 0) if planned_seg is not None else 0.0

        if planned_speed_kmh == 0 and planned_speed_eq_kmh > 0 and planned_time_sec > 0:
            distance_eq = pacer_service.planner.compute_distance_eq_km(distance_km, elev_gain)
            if distance_eq > 0:
                planned_speed_kmh = (distance_km / planned_time_sec) * 3600

        if planned_speed_eq_kmh == 0 and planned_speed_kmh > 0 and planned_time_sec > 0:
            distance_eq = pacer_service.planner.compute_distance_eq_km(distance_km, elev_gain)
            if distance_eq > 0:
                planned_speed_eq_kmh = (distance_eq / planned_time_sec) * 3600

        actual_time_sec = row.get("actualTimeSec")
        actual_speed_kmh = row.get("actualSpeedKmh")
        actual_speed_eq_kmh = row.get("actualSpeedEqKmh")

        if actual_speed_kmh is None and actual_speed_eq_kmh is not None and actual_time_sec and actual_time_sec > 0:
            distance_eq = pacer_service.planner.compute_distance_eq_km(distance_km, elev_gain)
            if distance_eq > 0:
                actual_speed_kmh = (distance_km / actual_time_sec) * 3600

        if actual_speed_eq_kmh is None and actual_speed_kmh is not None and actual_time_sec and actual_time_sec > 0:
            distance_eq = pacer_service.planner.compute_distance_eq_km(distance_km, elev_gain)
            if distance_eq > 0:
                actual_speed_eq_kmh = (distance_eq / actual_time_sec) * 3600

        planned_time = format_session_duration(int(planned_time_sec)) if planned_time_sec > 0 else "-"
        actual_time = (
            format_session_duration(int(actual_time_sec)) if actual_time_sec is not None else "-"
        )
        time_delta_sec = (
            float(row.get("timeDeltaSec", 0) or 0) if pd.notna(row.get("timeDeltaSec")) else None
        )

        planned_speed_str = fmt_decimal(planned_speed_kmh, 1) if planned_speed_kmh > 0 else "-"
        actual_speed_str = fmt_decimal(actual_speed_kmh, 1) if actual_speed_kmh is not None else "-"

        if actual_speed_kmh is not None and planned_speed_kmh > 0:
            speed_delta = actual_speed_kmh - planned_speed_kmh
        else:
            speed_delta = (
                float(row.get("speedDeltaKmh", 0) or 0) if pd.notna(row.get("speedDeltaKmh")) else None
            )

        planned_speed_eq_str = fmt_decimal(planned_speed_eq_kmh, 1) if planned_speed_eq_kmh > 0 else "-"
        actual_speed_eq_str = (
            fmt_decimal(actual_speed_eq_kmh, 1) if actual_speed_eq_kmh is not None else "-"
        )

        if actual_speed_eq_kmh is not None and planned_speed_eq_kmh > 0:
            speed_eq_delta = actual_speed_eq_kmh - planned_speed_eq_kmh
        else:
            speed_eq_delta = (
                float(row.get("speedEqDeltaKmh", 0) or 0)
                if pd.notna(row.get("speedEqDeltaKmh"))
                else None
            )

        if time_delta_sec is not None:
            time_delta_str = format_delta_minutes(time_delta_sec)
            time_delta_color = "#dc2626" if time_delta_sec > 0 else "#22c55e"
        else:
            time_delta_str = "-"
            time_delta_color = None

        display_data.append(
            {
                "Segment": f"#{seg_id} ({row.get('type', 'unknown')})",
                "Distance": f"{fmt_decimal(row['startKm'], 1)} - {fmt_decimal(row['endKm'], 1)} km",
                "Temps planifié": planned_time,
                "Temps réel": actual_time,
                "Δ Temps": time_delta_str,
                "Vitesse planifiée": f"{planned_speed_str} km/h",
                "Vitesse réelle": f"{actual_speed_str} km/h",
                "Δ Vitesse": (
                    f"{'+' if speed_delta and speed_delta >= 0 else ''}"
                    f"{fmt_decimal(speed_delta, 1) if speed_delta is not None else 0.0} km/h"
                ),
                "Vitesse-eq planifiée": f"{planned_speed_eq_str} km/h",
                "Vitesse-eq réelle": f"{actual_speed_eq_str} km/h",
                "Δ Vitesse-eq": (
                    f"{'+' if speed_eq_delta and speed_eq_delta >= 0 else ''}"
                    f"{fmt_decimal(speed_eq_delta, 1) if speed_eq_delta is not None else 0.0} km/h"
                ),
                "_time_delta_color": time_delta_color,
            }
        )

    comparison_display_df = pd.DataFrame(display_data)

    st.dataframe(
        comparison_display_df.drop(columns=["_time_delta_color"]),
        use_container_width=True,
        hide_index=True,
    )
