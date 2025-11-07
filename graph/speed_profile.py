"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Speed profile visualization chart.

Maximum speed profiles across different time windows for both raw speed and equivalent speed.
"""

from __future__ import annotations

import altair as alt
import pandas as pd


def create_speed_profile_chart(profile_df: pd.DataFrame, chart_width: int = 800) -> alt.Chart:
    """Create speed profile chart showing max speed and max equivalent speed across window sizes.
    
    Args:
        profile_df: DataFrame with columns windowSec, maxSpeedKmh, maxSpeedEqKmh
        chart_width: Chart width in pixels (default: 800)
    
    Returns:
        alt.Chart: Altair chart with two lines (speed and speed-eq) over window sizes
    """
    if profile_df.empty:
        # Return empty chart
        return alt.Chart(pd.DataFrame({"windowSec": [], "value": [], "type": []})).mark_line()
    
    # Prepare data for plotting (convert to long format)
    chart_data = pd.DataFrame({
        "windowSec": profile_df["windowSec"].tolist() * 2,
        "value": (
            profile_df["maxSpeedKmh"].tolist() +
            profile_df["maxSpeedEqKmh"].tolist()
        ),
        "type": (
            ["Vitesse max (km/h)"] * len(profile_df) +
            ["Vitesse‑eq max (km/h)"] * len(profile_df)
        ),
    })
    
    # Create line chart
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "windowSec:Q",
                title="Fenêtre (s)",
                scale=alt.Scale(type="log", base=10),
            ),
            y=alt.Y(
                "value:Q",
                title="Vitesse (km/h)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "type:N",
                title="Type",
                scale=alt.Scale(
                    domain=["Vitesse max (km/h)", "Vitesse‑eq max (km/h)"],
                    range=["#1f77b4", "#ff7f0e"],  # Blue and orange
                ),
            ),
            tooltip=[
                alt.Tooltip("windowSec:Q", title="Fenêtre (s)", format=".0f"),
                alt.Tooltip("value:Q", title="Vitesse (km/h)", format=".2f"),
                alt.Tooltip("type:N", title="Type"),
            ],
        )
        .properties(width=chart_width, height=400)
    )
    
    return chart

