"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Comparison panel widget for planned vs actual metrics.

Displays comparison cards showing planned, actual, and delta values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

from services.activity_detail_service import PlanComparison
from utils.formatting import format_duration, fmt_decimal, fmt_m


@dataclass
class ComparisonMetric:
    """A comparison metric with planned, actual, and delta values."""

    planned: Optional[float]
    actual: Optional[float]
    delta: Optional[float]


def format_comparison_value(label: str, value: Optional[float]) -> str:
    """Format a comparison value based on its label.

    Args:
        label: Metric label (e.g., "Distance", "Durée", "TRIMP", "D+")
        value: Value to format

    Returns:
        str: Formatted value string or "-" if None
    """
    if value is None:
        return "-"
    if label == "Durée":
        return format_duration(value, include_seconds=False)
    if label == "D+":
        return fmt_m(value)
    precision = 1 if label in {"Distance", "TRIMP"} else 1
    return fmt_decimal(value, precision)


def format_comparison_delta(label: str, value: Optional[float]) -> tuple[str, str]:
    """Format a comparison delta value.

    Args:
        label: Metric label
        value: Delta value

    Returns:
        tuple[str, str]: Tuple of (delta_label, delta_value)
    """
    if value is None:
        return label, "-"
    sign = "+" if value >= 0 else "-"
    magnitude = abs(value)
    if label == "Durée":
        return "Δ", f"{sign}{format_duration(magnitude, include_seconds=False)}"
    if label == "D+":
        return "Δ", f"{sign}{fmt_m(magnitude)}"
    precision = 1 if label in {"Distance", "TRIMP"} else 1
    return "Δ", f"{sign}{fmt_decimal(magnitude, precision)}"


def render_comparison_panel(comparison: PlanComparison, match_score: Optional[float]) -> None:
    """Render comparison panel showing planned vs actual metrics.

    Args:
        comparison: PlanComparison object with metrics
        match_score: Match score for the link (optional)
    """
    if not comparison:
        st.info("Cette activité n'est liée à aucune séance planifiée.")
        return

    st.subheader("Comparaison plan / réalisé")
    if match_score is not None:
        score = fmt_decimal(match_score, 2)
        st.caption(f"Score de correspondance : {score}")

    comparison_cards = [
        ("Distance", comparison.distance),
        ("Durée", comparison.duration),
        ("TRIMP", comparison.trimp),
        ("D+", comparison.ascent),
    ]
    cols = st.columns(len(comparison_cards))
    for (label, metric), col in zip(comparison_cards, cols):
        with col:
            planned_val = format_comparison_value(label, metric.planned)
            actual_val = format_comparison_value(label, metric.actual)
            delta_label, delta_value = format_comparison_delta(label, metric.delta)
            delta_color = "#60ac84" if (metric.delta or 0) >= 0 else "#9e4836"
            indicator = "▲" if (metric.delta or 0) >= 0 else "▼"
            st.markdown(
                f"""
<div style="
    background: rgba(20, 32, 48, 0.75);
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    border: 1px solid rgba(228, 204, 160, 0.22);
    box-shadow: 0 12px 24px rgba(8, 14, 24, 0.28);
">
    <div style="font-size:0.9rem; color:#e4cca0; letter-spacing:0.04em;">{label.upper()}</div>
    <div style="margin-top:0.35rem; color:#f8fafc;">Plan&nbsp;: <strong>{planned_val}</strong></div>
    <div style="color:#f8fafc;">Réel&nbsp;: <strong>{actual_val}</strong></div>
    <div style="margin-top:0.4rem; color:{delta_color}; font-weight:600;">
        {indicator}&nbsp;{delta_label}: {delta_value}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

