"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

QQ-line charts for activity metrics distributions.
"""

from __future__ import annotations

import statistics
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd


def create_qq_line_chart(
    values: pd.Series,
    *,
    title: str,
    point_color: str = "#3b82f6",
    line_color: str = "#ef4444",
    chart_width: int = 280,
    chart_height: int = 240,
) -> Optional[alt.Chart]:
    """Build a QQ chart (sample quantiles vs normal quantiles) with fitted QQ-line."""
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty or len(series) < 8:
        return None

    sorted_values = np.sort(series.to_numpy(dtype=float))
    std = float(np.std(sorted_values, ddof=1))
    if not np.isfinite(std) or std <= 0:
        return None
    mean = float(np.mean(sorted_values))

    n = len(sorted_values)
    probabilities = (np.arange(1, n + 1, dtype=float) - 0.5) / float(n)
    normal = statistics.NormalDist(mu=0.0, sigma=1.0)
    theoretical_z = np.array([normal.inv_cdf(float(p)) for p in probabilities], dtype=float)
    theoretical_quantiles = mean + std * theoretical_z

    qq_df = pd.DataFrame(
        {
            "theoretical_quantile": theoretical_quantiles,
            "sample_quantile": sorted_values,
        }
    )

    slope, intercept = np.polyfit(theoretical_quantiles, sorted_values, deg=1)
    predicted = slope * theoretical_quantiles + intercept
    sse = float(np.sum((sorted_values - predicted) ** 2))
    sst = float(np.sum((sorted_values - np.mean(sorted_values)) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0 else 1.0

    qq_df["line"] = predicted
    qq_df["equation"] = f"y = {slope:.3f}x + {intercept:.3f} (R²={r2:.3f})"

    domain_min = float(min(qq_df["theoretical_quantile"].min(), qq_df["sample_quantile"].min()) - 1.0)
    domain_max = float(max(qq_df["theoretical_quantile"].max(), qq_df["sample_quantile"].max()) + 1.0)
    if not np.isfinite(domain_min) or not np.isfinite(domain_max) or domain_max <= domain_min:
        domain_min, domain_max = 0.0, 1.0

    points = (
        alt.Chart(qq_df)
        .mark_circle(size=36, color=point_color, opacity=0.8)
        .encode(
            x=alt.X(
                "theoretical_quantile:Q",
                title="Quantiles théoriques",
                scale=alt.Scale(domain=[domain_min, domain_max]),
            ),
            y=alt.Y(
                "sample_quantile:Q",
                title="Quantiles observés",
                scale=alt.Scale(domain=[domain_min, domain_max]),
            ),
            tooltip=[
                alt.Tooltip("theoretical_quantile:Q", title="Quantile théorique", format=".2f"),
                alt.Tooltip("sample_quantile:Q", title="Quantile observé", format=".2f"),
                alt.Tooltip("equation:N", title="QQ-line"),
            ],
        )
    )
    line = (
        alt.Chart(qq_df)
        .mark_line(color=line_color, strokeWidth=2)
        .encode(
            x=alt.X(
                "theoretical_quantile:Q",
                title="Quantiles théoriques",
                scale=alt.Scale(domain=[domain_min, domain_max]),
            ),
            y=alt.Y(
                "line:Q",
                title="Quantiles observés",
                scale=alt.Scale(domain=[domain_min, domain_max]),
            ),
        )
    )

    return (points + line).properties(
        width=chart_width,
        height=chart_height,
        title=f"{title} - y = {slope:.3f}x + {intercept:.3f}",
    )
