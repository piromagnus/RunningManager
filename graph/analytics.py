"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Analytics visualization charts.

Charts for planned vs actual metrics comparisons.
"""

from __future__ import annotations

import altair as alt
import pandas as pd


def create_weekly_bar_chart(
    stack_df: pd.DataFrame,
    metric_label: str,
    metric_cfg: dict,
    color_scale: alt.Scale,
    chart_width: int = 860,
) -> alt.Chart:
    """Create weekly stacked bar chart for planned vs actual metrics.

    Args:
        stack_df: DataFrame with weekly segments (planned/actual)
        metric_label: Metric label for Y-axis
        metric_cfg: Metric configuration dict with 'unit' key
        color_scale: Altair color scale for segments
        chart_width: Chart width in pixels (default: 860)

    Returns:
        alt.Chart: Stacked bar chart
    """
    return (
        alt.Chart(stack_df)
        .mark_bar()
        .encode(
            x=alt.X("weekLabel:N", title="Semaine"),
            y=alt.Y("value:Q", title=f"{metric_label} ({metric_cfg['unit']})"),
            color=alt.Color("segment_display:N", scale=color_scale, title=""),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("weekLabel:N", title="Semaine"),
                alt.Tooltip("actual:Q", title=f"Réalisé ({metric_cfg['unit']})", format=".2f"),
                alt.Tooltip("planned:Q", title=f"Planifié ({metric_cfg['unit']})", format=".2f"),
                alt.Tooltip("actualTimeHours:Q", title="Durée (h)", format=".2f"),
                alt.Tooltip("actualDistanceKm:Q", title="Distance (km)", format=".2f"),
                alt.Tooltip("actualDistanceEqKm:Q", title="Dist. équiv. (km)", format=".2f"),
                alt.Tooltip("actualTrimp:Q", title="TRIMP", format=".2f"),
            ],
        )
        .properties(height=400, width=chart_width)
    )


def create_daily_bar_chart(
    day_stack_df: pd.DataFrame,
    metric_label: str,
    metric_cfg: dict,
    color_scale: alt.Scale,
    chart_width: int = 860,
) -> alt.Chart:
    """Create daily stacked bar chart for planned vs actual metrics.

    Args:
        day_stack_df: DataFrame with daily segments (planned/actual)
        metric_label: Metric label for Y-axis
        metric_cfg: Metric configuration dict with 'unit' key
        color_scale: Altair color scale for segments
        chart_width: Chart width in pixels (default: 860)

    Returns:
        alt.Chart: Stacked bar chart
    """
    return (
        alt.Chart(day_stack_df)
        .mark_bar()
        .encode(
            x=alt.X("weekLabel:N", title="Jour"),
            y=alt.Y("value:Q", title=f"{metric_label} ({metric_cfg['unit']})"),
            color=alt.Color("segment_display:N", scale=color_scale, title=""),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("weekLabel:N", title="Jour"),
                alt.Tooltip("segment_display:N", title="Segment"),
                alt.Tooltip("value:Q", title="Valeur", format=".2f"),
                alt.Tooltip("planned:Q", title="Planifié", format=".2f"),
                alt.Tooltip("actual:Q", title="Réalisé", format=".2f"),
                alt.Tooltip("maxValue:Q", title="Max", format=".2f"),
                alt.Tooltip("activity_names:N", title="Activités"),
            ],
        )
        .properties(height=300, width=chart_width)
    )

