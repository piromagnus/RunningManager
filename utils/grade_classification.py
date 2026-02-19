"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def classify_grade_pacer_5cat(
    grade: float, elevation_delta_per_km: Optional[float] = None
) -> str:
    """Classify grade for pacer segmentation (5 categories).

    Args:
        grade: Grade value (decimal, not percentage).
        elevation_delta_per_km: Optional elevation delta per km for flat detection.

    Returns:
        One of: steep_up, run_up, flat, down, steep_down.
    """
    if pd.isna(grade):
        return "flat"

    if elevation_delta_per_km is not None and abs(elevation_delta_per_km) < 10.0:
        return "flat"

    if grade >= 0.10:
        return "steep_up"
    if 0.02 <= grade < 0.10:
        return "run_up"
    if -0.02 < grade < 0.02:
        return "flat"
    if -0.25 < grade <= -0.02:
        return "down"
    return "steep_down"


def classify_grade_elevation_8cat(grade: float) -> str:
    """Classify grade for elevation visualization (8 categories).

    Args:
        grade: Grade value (decimal, not percentage).

    Returns:
        One of: grade_lt_neg_0_5, grade_lt_neg_0_25, grade_lt_neg_0_05,
        grade_neutral, grade_lt_0_1, grade_lt_0_25, grade_lt_0_5, grade_ge_0_5,
        unknown.
    """
    if pd.isna(grade):
        return "unknown"
    if grade < -0.5:
        return "grade_lt_neg_0_5"
    if -0.5 <= grade < -0.25:
        return "grade_lt_neg_0_25"
    if -0.25 <= grade < -0.05:
        return "grade_lt_neg_0_05"
    if -0.05 <= grade < 0.05:
        return "grade_neutral"
    if 0.05 <= grade < 0.1:
        return "grade_lt_0_1"
    if 0.1 <= grade < 0.25:
        return "grade_lt_0_25"
    if 0.25 <= grade < 0.5:
        return "grade_lt_0_5"
    return "grade_ge_0_5"
