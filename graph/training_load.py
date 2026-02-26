"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Training load visualization charts.

Charts showing acute vs chronic training load with bands.
"""

from __future__ import annotations

import altair as alt
import pandas as pd


def create_training_load_chart(
    df: pd.DataFrame,
    metric_key: str,
    metric_cfg: dict,
    *,
    band_lower_ratio: float = 0.75,
    band_upper_ratio: float = 1.5,
    danger_ratio: float | None = None,
) -> alt.Chart:
    """Create a training load chart showing acute vs chronic with bands.

    Args:
        df: DataFrame with date and metric columns
        metric_key: Key identifying the metric (for reference)
        metric_cfg: Configuration dict with keys:
            - "label": Display label for the metric
            - "chronic": Column name for chronic (28-day) load
            - "acute": Column name for acute (7-day) load

    Returns:
        alt.Chart: Altair chart with acute/chronic lines and band
    """
    planned_col = metric_cfg["chronic"]
    acute_col = metric_cfg["acute"]

    working = df[["date", planned_col, acute_col]].copy()
    working[planned_col] = pd.to_numeric(working[planned_col], errors="coerce").fillna(0.0)
    working[acute_col] = pd.to_numeric(working[acute_col], errors="coerce").fillna(0.0)
    working["chronic_lower"] = band_lower_ratio * working[planned_col]
    working["chronic_upper"] = band_upper_ratio * working[planned_col]
    if danger_ratio is not None:
        working["danger_line"] = danger_ratio * working[planned_col]

    base = alt.Chart(working).encode(x=alt.X("date:T", title="Date"))

    fill = base.mark_area(opacity=0.2, color="#2563eb").encode(
        y=alt.Y("chronic_lower:Q", title=metric_cfg["label"]),
        y2="chronic_upper:Q",
    )

    chronic_line = base.mark_line(color="#1d4ed8", strokeWidth=2).encode(y=f"{planned_col}:Q")
    acute_line = base.mark_line(color="#f97316", strokeWidth=2).encode(y=f"{acute_col}:Q")
    layers: list[alt.Chart] = [fill, chronic_line, acute_line]
    if danger_ratio is not None:
        danger_line = base.mark_line(color="#dc2626", strokeWidth=1.8).encode(y="danger_line:Q")
        layers.append(danger_line)

    return (
        alt.layer(*layers)
        .encode(
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip(planned_col, title="Charge chronique", format=".2f"),
                alt.Tooltip(acute_col, title="Charge aiguë", format=".2f"),
            ]
        )
        .properties(height=340)
    )

