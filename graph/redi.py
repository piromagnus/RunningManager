"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import altair as alt
import pandas as pd


def create_workload_ratio_chart(
    df: pd.DataFrame,
    *,
    method_label: str,
    ratio_col: str,
    acute_col: str,
    chronic_col: str,
) -> alt.Chart:
    """Create REDI/EWMA acute:chronic ratio chart with risk-zone references."""
    working = df[["date", ratio_col, acute_col, chronic_col]].copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"])
    working[ratio_col] = pd.to_numeric(working[ratio_col], errors="coerce")
    working[acute_col] = pd.to_numeric(working[acute_col], errors="coerce")
    working[chronic_col] = pd.to_numeric(working[chronic_col], errors="coerce")
    working = working.dropna(subset=[ratio_col, acute_col, chronic_col])

    base = alt.Chart(working).encode(x=alt.X("date:T", title="Date"))

    safe_zone_df = working[["date"]].copy()
    safe_zone_df["zone_lower"] = 0.8
    safe_zone_df["zone_upper"] = 1.3

    safe_zone = (
        alt.Chart(safe_zone_df)
        .mark_area(opacity=0.15, color="#22c55e")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("zone_lower:Q", title="Ratio aiguë / chronique"),
            y2="zone_upper:Q",
        )
    )

    boundary_lines = (
        alt.Chart(pd.DataFrame({"threshold": [0.8, 1.3]}))
        .mark_rule(color="#16a34a", strokeWidth=1.5, strokeDash=[4, 4], opacity=0.8)
        .encode(y="threshold:Q")
    )

    danger_line = (
        alt.Chart(pd.DataFrame({"threshold": [1.5]}))
        .mark_rule(color="#dc2626", strokeWidth=2, strokeDash=[6, 4])
        .encode(y="threshold:Q")
    )

    ratio_line = base.mark_line(color="#2563eb", strokeWidth=2).encode(
        y=alt.Y(f"{ratio_col}:Q", title="Ratio aiguë / chronique"),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip(f"{ratio_col}:Q", title=f"Ratio {method_label}", format=".3f"),
            alt.Tooltip(f"{acute_col}:Q", title=f"{method_label} aiguë", format=".2f"),
            alt.Tooltip(f"{chronic_col}:Q", title=f"{method_label} chronique", format=".2f"),
        ],
    )

    return alt.layer(safe_zone, boundary_lines, danger_line, ratio_line).properties(height=340)


def create_redi_ratio_chart(df: pd.DataFrame) -> alt.Chart:
    """Create REDI ratio chart with default REDI column names."""
    return create_workload_ratio_chart(
        df,
        method_label="REDI",
        ratio_col="redi_ratio",
        acute_col="redi_acute",
        chronic_col="redi_chronic",
    )
