"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Per-activity HR cluster chart with bidirectional standard deviation bars.
"""

from __future__ import annotations

import math
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor


def create_activity_cluster_chart(
    metrics_ts_df: pd.DataFrame,
    speed_type: str = "speed",
    chart_width: int = 860,
) -> Optional[alt.Chart]:
    """Build per-cluster mean HR/speed chart with horizontal and vertical std bars."""
    if metrics_ts_df is None or metrics_ts_df.empty:
        return None
    speed_col = "speedeq_smooth" if str(speed_type).lower().startswith("speedeq") else "speed_smooth"
    required_columns = {"cluster", "hr_shifted", speed_col}
    if not required_columns.issubset(set(metrics_ts_df.columns)):
        return None

    working = metrics_ts_df[["cluster", "hr_shifted", speed_col]].copy()
    working["cluster"] = pd.to_numeric(working["cluster"], errors="coerce")
    working["hr_shifted"] = pd.to_numeric(working["hr_shifted"], errors="coerce")
    working[speed_col] = pd.to_numeric(working[speed_col], errors="coerce")
    working = working.dropna(subset=["cluster", "hr_shifted", speed_col])
    if working.empty:
        return None

    grouped = (
        working.groupby("cluster", as_index=False)
        .agg(
            count=("cluster", "size"),
            hr_mean=("hr_shifted", "mean"),
            hr_std=("hr_shifted", "std"),
            speed_mean=(speed_col, "mean"),
            speed_std=(speed_col, "std"),
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    if grouped.empty:
        return None

    min_cluster_size = max(1, int(math.ceil(len(working) * 0.05)))
    grouped = grouped[grouped["count"] >= min_cluster_size].copy()
    if grouped.empty:
        return None

    grouped["hr_std"] = pd.to_numeric(grouped["hr_std"], errors="coerce").fillna(0.0)
    grouped["speed_std"] = pd.to_numeric(grouped["speed_std"], errors="coerce").fillna(0.0)
    grouped["hr_lower"] = grouped["hr_mean"] - grouped["hr_std"]
    grouped["hr_upper"] = grouped["hr_mean"] + grouped["hr_std"]
    grouped["speed_lower"] = grouped["speed_mean"] - grouped["speed_std"]
    grouped["speed_upper"] = grouped["speed_mean"] + grouped["speed_std"]
    grouped["cluster_label"] = grouped["cluster"].astype(int).astype(str)
    grouped["is_outlier"] = False

    speed_axis_title = (
        "Vitesse équivalente (km/h)" if speed_col == "speedeq_smooth" else "Vitesse (km/h)"
    )
    x_domain = _axis_domain(grouped["speed_lower"], grouped["speed_upper"])
    y_domain = _axis_domain(grouped["hr_lower"], grouped["hr_upper"])
    grouped, regression_line_df = _fit_regression_with_outliers(grouped, x_domain)

    color = alt.Color("cluster_label:N", title="Cluster")

    x_error = (
        alt.Chart(grouped)
        .mark_rule(strokeWidth=2, opacity=0.65)
        .encode(
            x=alt.X("speed_lower:Q", title=speed_axis_title, scale=alt.Scale(domain=x_domain)),
            x2="speed_upper:Q",
            y=alt.Y("hr_mean:Q", title="Fréquence cardiaque (bpm)", scale=alt.Scale(domain=y_domain)),
            color=color,
        )
    )
    y_error = (
        alt.Chart(grouped)
        .mark_rule(strokeWidth=2, opacity=0.65)
        .encode(
            x=alt.X("speed_mean:Q", title=speed_axis_title, scale=alt.Scale(domain=x_domain)),
            y=alt.Y("hr_lower:Q", title="Fréquence cardiaque (bpm)", scale=alt.Scale(domain=y_domain)),
            y2="hr_upper:Q",
            color=color,
        )
    )
    inlier_points = (
        alt.Chart(grouped[grouped["is_outlier"] == False])  # noqa: E712
        .mark_circle(size=120, stroke="black", strokeWidth=1, opacity=0.9)
        .encode(
            x=alt.X("speed_mean:Q", title=speed_axis_title, scale=alt.Scale(domain=x_domain)),
            y=alt.Y("hr_mean:Q", title="Fréquence cardiaque (bpm)", scale=alt.Scale(domain=y_domain)),
            color=color,
            tooltip=[
                alt.Tooltip("cluster_label:N", title="Cluster"),
                alt.Tooltip("count:Q", title="Points"),
                alt.Tooltip("hr_mean:Q", title="FC moyenne", format=".1f"),
                alt.Tooltip("hr_std:Q", title="FC std", format=".1f"),
                alt.Tooltip("speed_mean:Q", title="Vitesse moyenne", format=".2f"),
                alt.Tooltip("speed_std:Q", title="Vitesse std", format=".2f"),
                alt.Tooltip("is_outlier:N", title="Outlier"),
            ],
        )
    )
    outlier_points = (
        alt.Chart(grouped[grouped["is_outlier"] == True])  # noqa: E712
        .mark_point(
            shape="diamond",
            size=180,
            color="#d62728",
            filled=False,
            stroke="black",
            strokeWidth=1.2,
            opacity=1.0,
        )
        .encode(
            x=alt.X("speed_mean:Q", title=speed_axis_title, scale=alt.Scale(domain=x_domain)),
            y=alt.Y("hr_mean:Q", title="Fréquence cardiaque (bpm)", scale=alt.Scale(domain=y_domain)),
            tooltip=[
                alt.Tooltip("cluster_label:N", title="Cluster"),
                alt.Tooltip("count:Q", title="Points"),
                alt.Tooltip("hr_mean:Q", title="FC moyenne", format=".1f"),
                alt.Tooltip("speed_mean:Q", title="Vitesse moyenne", format=".2f"),
                alt.Tooltip("is_outlier:N", title="Outlier"),
            ],
        )
    )

    layers: list[alt.Chart] = [x_error, y_error, inlier_points, outlier_points]
    regression_title: str | None = None
    if regression_line_df is not None and not regression_line_df.empty:
        equation = str(regression_line_df.iloc[0].get("equation") or "").strip()
        r2 = pd.to_numeric(pd.Series([regression_line_df.iloc[0].get("r2")]), errors="coerce").iloc[0]
        if equation:
            regression_title = (
                f"Régression linéaire: {equation} · R²={float(r2):.3f}" if pd.notna(r2) else equation
            )
        regression_layer = (
            alt.Chart(regression_line_df)
            .mark_line(color="#ef4444", strokeDash=[6, 4], strokeWidth=2)
            .encode(
                x=alt.X("speed_mean:Q", title=speed_axis_title, scale=alt.Scale(domain=x_domain)),
                y=alt.Y(
                    "hr_regression:Q",
                    title="Fréquence cardiaque (bpm)",
                    scale=alt.Scale(domain=y_domain),
                ),
                tooltip=[
                    alt.Tooltip("equation:N", title="Régression"),
                    alt.Tooltip("r2:Q", title="R²", format=".3f"),
                    alt.Tooltip("inliers:Q", title="Inliers"),
                    alt.Tooltip("outliers:Q", title="Outliers"),
                ],
            )
        )
        layers.append(regression_layer)

    chart = alt.layer(*layers).properties(width=chart_width, height=420)
    if regression_title:
        chart = chart.properties(title=regression_title)
    return chart


def _axis_domain(lower_series: pd.Series, upper_series: pd.Series) -> list[float]:
    lower = float(pd.to_numeric(lower_series, errors="coerce").min())
    upper = float(pd.to_numeric(upper_series, errors="coerce").max())
    domain_min = lower - 1.0
    domain_max = upper + 1.0
    if not np.isfinite(domain_min) or not np.isfinite(domain_max):
        return [0.0, 1.0]
    if domain_max <= domain_min:
        return [domain_min - 1.0, domain_max + 1.0]
    return [domain_min, domain_max]


def _fit_regression_with_outliers(
    grouped: pd.DataFrame,
    x_domain: list[float],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    working = grouped.copy()
    if working.empty:
        return working, None

    x_values = pd.to_numeric(working["speed_mean"], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(working["hr_mean"], errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if valid_mask.sum() < 2:
        return working, None

    indices = np.where(valid_mask)[0]
    x = x_values[valid_mask].reshape(-1, 1)
    y = y_values[valid_mask]
    inlier_mask = np.ones(len(y), dtype=bool)

    if len(y) >= 4:
        y_median = float(np.median(y))
        mad = float(np.median(np.abs(y - y_median)))
        residual_threshold = mad if mad > 0 else 1.0
        min_samples = max(2, int(math.ceil(len(y) * 0.5)))
        try:
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=min_samples,
                residual_threshold=residual_threshold,
                random_state=42,
            )
            ransac.fit(x, y)
            if ransac.inlier_mask_ is not None and ransac.inlier_mask_.sum() >= 2:
                inlier_mask = ransac.inlier_mask_
        except Exception:
            inlier_mask = np.ones(len(y), dtype=bool)

    working["is_outlier"] = False
    outlier_indices = indices[~inlier_mask]
    if outlier_indices.size > 0:
        working.loc[outlier_indices, "is_outlier"] = True

    x_fit = x[inlier_mask] if inlier_mask.sum() >= 2 else x
    y_fit = y[inlier_mask] if inlier_mask.sum() >= 2 else y
    if len(y_fit) < 2:
        return working, None

    model = LinearRegression()
    model.fit(x_fit, y_fit)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(x_fit, y_fit))

    x_start = float(x_domain[0])
    x_end = float(x_domain[1])
    line_df = pd.DataFrame(
        {
            "speed_mean": [x_start, x_end],
            "hr_regression": [slope * x_start + intercept, slope * x_end + intercept],
            "equation": [f"HR = {slope:.3f} * v + {intercept:.3f}"] * 2,
            "r2": [r2, r2],
            "inliers": [int(inlier_mask.sum()), int(inlier_mask.sum())],
            "outliers": [int((~inlier_mask).sum()), int((~inlier_mask).sum())],
        }
    )
    return working, line_df
