"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

HR vs Speed visualization chart.

Cluster-based scatter plot with weighted linear regression.
"""

from __future__ import annotations

from typing import Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd


def create_hr_speed_chart(
    centers_df: pd.DataFrame,
    slope: Optional[float],
    intercept: Optional[float],
    r_squared: Optional[float],
    std_err: Optional[float],
    chart_width: int = 860,
) -> alt.Chart:
    """Create HR vs Speed scatter chart with regression line.

    Args:
        centers_df: DataFrame with cluster centers (hr, speed, hr_std, speed_std, etc.)
        slope: Regression slope (None if no regression)
        intercept: Regression intercept (None if no regression)
        r_squared: R-squared value (None if no regression)
        std_err: Standard error (None if no regression)
        chart_width: Chart width in pixels (default: 860)

    Returns:
        alt.Chart: Altair chart with scatter, error bars, and regression line
    """
    chart_df = centers_df.copy()

    # Prepare data for error bars (uncertainty based on std)
    chart_df["hr_upper"] = chart_df["hr"] + chart_df["hr_std"]
    chart_df["hr_lower"] = chart_df["hr"] - chart_df["hr_std"]
    chart_df["speed_upper"] = chart_df["speed"] + chart_df["speed_std"]
    chart_df["speed_lower"] = chart_df["speed"] - chart_df["speed_std"]

    # Color by cluster
    color_encoding = alt.Color("cluster:O", scale=alt.Scale(scheme="viridis"), title="Cluster")

    # Plot cluster centers (HR on y-axis, Speed on x-axis)
    centers = (
        alt.Chart(chart_df)
        .mark_circle(size=100, opacity=0.8, stroke="black", strokeWidth=2)
        .encode(
            x=alt.X("speed:Q", title="Vitesse (km/h)", scale=alt.Scale(domain=[4, 24])),
            y=alt.Y("hr:Q", title="Fréquence cardiaque (bpm)", scale=alt.Scale(domain=[80, 210])),
            color=color_encoding,
            tooltip=[
                alt.Tooltip("hr:Q", title="FC (bpm)", format=".1f"),
                alt.Tooltip("speed:Q", title="Vitesse (km/h)", format=".1f"),
                alt.Tooltip("hr_std:Q", title="FC std (bpm)", format=".1f"),
                alt.Tooltip("speed_std:Q", title="Vitesse std (km/h)", format=".1f"),
                alt.Tooltip("cluster:O", title="Cluster"),
                alt.Tooltip("name:N", title="Activité"),
                alt.Tooltip("activity_date_str:N", title="Date"),
                alt.Tooltip("count:Q", title="Nombre de points"),
            ],
        )
    )

    # Add horizontal error bars (uncertainty in Speed, now on x-axis)
    speed_error_bars = (
        alt.Chart(chart_df)
        .mark_rule(strokeWidth=2, opacity=0.6)
        .encode(
            x=alt.X("speed_lower:Q", title="Vitesse (km/h)", scale=alt.Scale(domain=[4, 24])),
            x2="speed_upper:Q",
            y=alt.Y("hr:Q", title="Fréquence cardiaque (bpm)", scale=alt.Scale(domain=[80, 210])),
            color=color_encoding,
        )
    )

    # Add vertical error bars (uncertainty in HR, now on y-axis)
    hr_error_bars = (
        alt.Chart(chart_df)
        .mark_rule(strokeWidth=2, opacity=0.6)
        .encode(
            x=alt.X("speed:Q", title="Vitesse (km/h)", scale=alt.Scale(domain=[4, 24])),
            y=alt.Y("hr_lower:Q", title="Fréquence cardiaque (bpm)", scale=alt.Scale(domain=[80, 210])),
            y2="hr_upper:Q",
            color=color_encoding,
        )
    )

    layers: list[alt.Chart] = [speed_error_bars, hr_error_bars, centers]

    # Add regression line if available
    if slope is not None and intercept is not None:
        # Regression: hr = slope * speed + intercept
        speed_range = [4, 24]  # Use fixed range
        regression_data = pd.DataFrame({
            "speed": speed_range,
            "hr": [slope * speed + intercept for speed in speed_range],
        })

        regression_line = (
            alt.Chart(regression_data)
            .mark_line(color="red", strokeWidth=2)
            .encode(
                x=alt.X("speed:Q", scale=alt.Scale(domain=[4, 24])),
                y=alt.Y("hr:Q", scale=alt.Scale(domain=[80, 210]))
            )
        )

        # Add confidence interval
        if std_err is not None:
            regression_data["upper"] = regression_data["hr"] + std_err
            regression_data["lower"] = regression_data["hr"] - std_err

            confidence_band = (
                alt.Chart(regression_data)
                .mark_area(opacity=0.2, color="red")
                .encode(
                    x=alt.X("speed:Q", scale=alt.Scale(domain=[4, 24])),
                    y=alt.Y("upper:Q", scale=alt.Scale(domain=[80, 210])),
                    y2="lower:Q",
                )
            )
            layers.insert(0, confidence_band)

        layers.append(regression_line)

    return alt.layer(*layers).properties(height=500, width=chart_width)


def compute_weighted_regression(
    centers_df: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute weighted linear regression on cluster centers.

    Args:
        centers_df: DataFrame with speed, hr, speed_std columns

    Returns:
        Tuple of (slope, intercept, r_squared, std_err) or (None, None, None, None) if insufficient data
    """
    x_values = centers_df["speed"].values
    y_values = centers_df["hr"].values

    # Compute weights: inverse of speed_std (uncertainty in the independent variable)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    speed_std_values = centers_df["speed_std"].values

    # Weight = 1 / (speed_std + epsilon) - clusters with larger std have less weight
    weights = 1.0 / (speed_std_values + epsilon)

    if len(x_values) < 2:
        return None, None, None, None

    # Weighted least squares: y = slope * x + intercept
    # Build design matrix for [x, 1]
    A = np.vstack([x_values, np.ones(len(x_values))]).T

    # Apply weights
    W = np.diag(weights)
    A_weighted = np.sqrt(W) @ A
    y_weighted = np.sqrt(W) @ y_values

    # Solve weighted least squares: (A^T * W * A) * params = A^T * W * y
    params, residuals, rank, s = np.linalg.lstsq(A_weighted, y_weighted, rcond=None)
    slope, intercept = params

    # Calculate R-squared for weighted regression
    y_pred = slope * x_values + intercept
    ss_res = np.sum(weights * (y_values - y_pred)**2)
    ss_tot = np.sum(weights * (y_values - np.average(y_values, weights=weights))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Calculate weighted standard error
    if len(x_values) > 2:
        dof = len(x_values) - 2  # degrees of freedom
        mse = ss_res / dof if dof > 0 else 0.0
        std_err = np.sqrt(mse)
    else:
        std_err = 0.0

    return slope, intercept, r_squared, std_err

