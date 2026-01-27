"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Speed profile visualization chart.

Maximum speed profiles across different time windows for both raw speed and equivalent speed.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

WINDOW_LABEL_EXPR = (
    "datum.value >= 3600 ? format(datum.value / 3600, '.1f') + ' h' : "
    "datum.value >= 60 ? format(datum.value / 60, '.0f') + ' min' : "
    "format(datum.value, '.0f') + ' s'"
)


def create_speed_profile_chart(
    profile_df: pd.DataFrame,
    chart_width: int = 800,
    x_domain: tuple[float, float] | None = None,
    tick_values: list[float] | None = None,
) -> alt.Chart:
    """Create speed profile chart showing max speed and max equivalent speed across window sizes.

    Args:
        profile_df: DataFrame with columns windowSec, maxSpeedKmh, maxSpeedEqKmh and
            optional activity reference columns for tooltips.
        chart_width: Chart width in pixels (default: 800)

    Returns:
        alt.Chart: Altair chart with two lines (speed and speed-eq) over window sizes
    """
    if profile_df.empty:
        # Return empty chart
        return alt.Chart(pd.DataFrame({"windowSec": [], "value": [], "type": []})).mark_line()
    
    def _col_or_empty(column_name: str) -> list[str]:
        if column_name in profile_df.columns:
            return profile_df[column_name].fillna("").astype(str).tolist()
        return [""] * len(profile_df)

    speed_activity_ids = _col_or_empty("maxSpeedActivityId")
    speed_activity_names = _col_or_empty("maxSpeedActivityName")
    speed_activity_dates = _col_or_empty("maxSpeedActivityDate")
    speed_eq_activity_ids = _col_or_empty("maxSpeedEqActivityId")
    speed_eq_activity_names = _col_or_empty("maxSpeedEqActivityName")
    speed_eq_activity_dates = _col_or_empty("maxSpeedEqActivityDate")

    # Prepare data for plotting (convert to long format)
    chart_data = pd.DataFrame(
        {
            "windowSec": profile_df["windowSec"].tolist() * 2,
            "value": (
                profile_df["maxSpeedKmh"].tolist()
                + profile_df["maxSpeedEqKmh"].tolist()
            ),
            "type": (
                ["Vitesse max (km/h)"] * len(profile_df)
                + ["Vitesse‑eq max (km/h)"] * len(profile_df)
            ),
            "activityId": speed_activity_ids + speed_eq_activity_ids,
            "activityName": speed_activity_names + speed_eq_activity_names,
            "activityDate": speed_activity_dates + speed_eq_activity_dates,
        }
    )
    
    scale_kwargs = {"type": "log", "base": 10}
    if x_domain:
        scale_kwargs["domain"] = [float(x_domain[0]), float(x_domain[1])]
    if tick_values is None:
        tick_values = (
            pd.to_numeric(profile_df.get("windowSec"), errors="coerce")
            .dropna()
            .unique()
            .tolist()
        )
        tick_values = sorted(tick_values)

    # Create line chart
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "windowSec:Q",
                title="Fenêtre",
                scale=alt.Scale(**scale_kwargs),
                axis=alt.Axis(labelExpr=WINDOW_LABEL_EXPR, values=tick_values),
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
                alt.Tooltip("activityName:N", title="Activité"),
                alt.Tooltip("activityDate:N", title="Date"),
                alt.Tooltip("activityId:N", title="Id"),
            ],
        )
        .properties(width=chart_width, height=400)
    )
    
    return chart


def create_speed_profile_cloud_chart(
    cloud_df: pd.DataFrame,
    chart_width: int = 800,
    speed_type: str = "both",
    x_domain: tuple[float, float] | None = None,
    tick_values: list[float] | None = None,
) -> alt.Chart:
    """Create speed profile cloud chart with all points per window size."""
    if cloud_df.empty:
        return alt.Chart(pd.DataFrame({"windowSec": [], "value": [], "type": []})).mark_circle()

    def _col_or_empty(column_name: str) -> list[str]:
        if column_name in cloud_df.columns:
            return cloud_df[column_name].fillna("").astype(str).tolist()
        return [""] * len(cloud_df)

    def _col_or_nan(column_name: str) -> list[float | None]:
        if column_name in cloud_df.columns:
            return pd.to_numeric(cloud_df[column_name], errors="coerce").tolist()
        return [None] * len(cloud_df)

    activity_ids = _col_or_empty("activityId")
    activity_names = _col_or_empty("activityName")
    activity_dates = _col_or_empty("activityDateStr")

    speed_values = _col_or_nan("maxSpeedKmh")
    speed_eq_values = _col_or_nan("maxSpeedEqKmh")
    hr_values = _col_or_nan("hrAtMaxSpeed")
    hr_eq_values = _col_or_nan("hrAtMaxSpeedEq")

    if speed_type == "eq":
        chart_data = pd.DataFrame(
            {
                "windowSec": cloud_df["windowSec"].tolist(),
                "value": speed_eq_values,
                "type": ["Vitesse‑eq max (km/h)"] * len(cloud_df),
                "activityId": activity_ids,
                "activityName": activity_names,
                "activityDate": activity_dates,
                "hrValue": hr_eq_values,
            }
        )
    elif speed_type == "raw":
        chart_data = pd.DataFrame(
            {
                "windowSec": cloud_df["windowSec"].tolist(),
                "value": speed_values,
                "type": ["Vitesse max (km/h)"] * len(cloud_df),
                "activityId": activity_ids,
                "activityName": activity_names,
                "activityDate": activity_dates,
                "hrValue": hr_values,
            }
        )
    else:
        chart_data = pd.DataFrame(
            {
                "windowSec": cloud_df["windowSec"].tolist() * 2,
                "value": speed_values + speed_eq_values,
                "type": (
                    ["Vitesse max (km/h)"] * len(cloud_df)
                    + ["Vitesse‑eq max (km/h)"] * len(cloud_df)
                ),
                "activityId": activity_ids + activity_ids,
                "activityName": activity_names + activity_names,
                "activityDate": activity_dates + activity_dates,
                "hrValue": hr_values + hr_eq_values,
            }
        )

    scale_kwargs = {"type": "log", "base": 10}
    if x_domain:
        scale_kwargs["domain"] = [float(x_domain[0]), float(x_domain[1])]
    if tick_values is None:
        tick_values = (
            pd.to_numeric(cloud_df.get("windowSec"), errors="coerce")
            .dropna()
            .unique()
            .tolist()
        )
        tick_values = sorted(tick_values)

    chart = (
        alt.Chart(chart_data)
        .mark_circle(size=50, opacity=0.6)
        .encode(
            x=alt.X(
                "windowSec:Q",
                title="Fenêtre",
                scale=alt.Scale(**scale_kwargs),
                axis=alt.Axis(labelExpr=WINDOW_LABEL_EXPR, values=tick_values),
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
                    range=["#1f77b4", "#ff7f0e"],
                ),
            ),
            tooltip=[
                alt.Tooltip("windowSec:Q", title="Fenêtre (s)", format=".0f"),
                alt.Tooltip("value:Q", title="Vitesse (km/h)", format=".2f"),
                alt.Tooltip("type:N", title="Type"),
                alt.Tooltip("activityName:N", title="Activité"),
                alt.Tooltip("activityDate:N", title="Date"),
                alt.Tooltip("hrValue:Q", title="FC", format=".0f"),
                alt.Tooltip("activityId:N", title="Id"),
            ],
        )
        .properties(width=chart_width, height=400)
    )

    return chart

