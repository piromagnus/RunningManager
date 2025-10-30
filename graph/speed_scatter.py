"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

SpeedEq scatter chart visualization.

Scatter plot showing SpeedEq vs Duration with HR coloring.
"""

from __future__ import annotations

import altair as alt
import pandas as pd


def create_speedeq_scatter_chart(
    cloud_df: pd.DataFrame, athlete_id: str
) -> alt.Chart:
    """Create SpeedEq scatter chart (SpeedEq vs Duration, colored by HR).

    Args:
        cloud_df: DataFrame with columns: durationH, speedEqKmh, avgHr, activityId, etc.
        athlete_id: Athlete ID for URL generation

    Returns:
        alt.Chart: Altair scatter chart
    """
    cloud_df = cloud_df.copy()
    cloud_df["avgHr"] = pd.to_numeric(cloud_df.get("avgHr"), errors="coerce")
    cloud_df = cloud_df.fillna({"avgHr": 0.0, "name": ""})
    if "activityId" in cloud_df.columns:
        cloud_df["activityId"] = cloud_df["activityId"].astype(str)
        base_page = "Activity"
        cloud_df["activityUrl"] = (
            base_page + "?activityId=" + cloud_df["activityId"] + "&athleteId=" + str(athlete_id)
        )

    return (
        alt.Chart(cloud_df)
        .mark_circle(size=60, opacity=0.85)
        .encode(
            x=alt.X("durationH:Q", title="Durée (h)"),
            y=alt.Y("speedEqKmh:Q", title="Vitesse équivalente (km/h)"),
            color=alt.Color("avgHr:Q", scale=alt.Scale(scheme="yelloworangered"), title="FC moy."),
            href=alt.Href("activityUrl:N"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("name:N", title="Nom"),
                alt.Tooltip("distanceEqKm:Q", title="Dist. équiv. (km)", format=".2f"),
                alt.Tooltip("distanceKm:Q", title="Distance (km)", format=".2f"),
                alt.Tooltip("ascentM:Q", title="D+ (m)", format=".0f"),
                alt.Tooltip("durationH:Q", title="Durée (h)", format=".2f"),
                alt.Tooltip("speedEqKmh:Q", title="SpeedEq (km/h)", format=".2f"),
                alt.Tooltip("avgHr:Q", title="FC moy.", format=".0f"),
            ],
        )
        .properties(height=360)
    )

