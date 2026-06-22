"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Analytics visualization charts.

Charts for planned vs actual metrics comparisons.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from utils.constants import CATEGORY_CHART_COLORS


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


def create_category_breakdown_chart(
    breakdown_df: pd.DataFrame,
    metric_label: str,
    metric_cfg: dict,
    *,
    chart_width: int = 860,
) -> alt.Chart | None:
    """Weekly stacked bars of a metric by activity category (bottom-to-top order)."""
    if breakdown_df.empty:
        return None

    working = breakdown_df.copy()
    working["value"] = pd.to_numeric(working["value"], errors="coerce").fillna(0.0)
    working = working[working["value"] > 0]
    if working.empty or "weekLabel" not in working.columns:
        return None

    stack_meta = (
        working.sort_values("category_order")
        .drop_duplicates("category")[["category", "category_label", "category_order"]]
    )
    legend_labels = stack_meta["category_label"].astype(str).tolist()
    legend_colors = [
        CATEGORY_CHART_COLORS.get(str(cat), "#94a3b8") for cat in stack_meta["category"]
    ]
    unit = metric_cfg.get("unit", "")
    value_title = f"{metric_label} ({unit})" if unit else metric_label

    return (
        alt.Chart(working)
        .mark_bar()
        .encode(
            x=alt.X("weekLabel:N", title="Semaine", sort=None),
            y=alt.Y("value:Q", title=value_title, stack=True),
            color=alt.Color(
                "category_label:N",
                title="Type d'activité",
                scale=alt.Scale(domain=legend_labels, range=legend_colors),
            ),
            order=alt.Order("category_order:Q"),
            tooltip=[
                alt.Tooltip("weekLabel:N", title="Semaine"),
                alt.Tooltip("category_label:N", title="Type"),
                alt.Tooltip("value:Q", title=value_title, format=".2f"),
            ],
        )
        .properties(height=400, width=chart_width)
    )

