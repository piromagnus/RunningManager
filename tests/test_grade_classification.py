"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from utils.grade_classification import (
    classify_grade_elevation_8cat,
    classify_grade_pacer_5cat,
)


def test_classify_grade_pacer_5cat_boundaries() -> None:
    assert classify_grade_pacer_5cat(0.15) == "steep_up"
    assert classify_grade_pacer_5cat(0.10) == "steep_up"
    assert classify_grade_pacer_5cat(0.05) == "run_up"
    assert classify_grade_pacer_5cat(0.02) == "run_up"
    assert classify_grade_pacer_5cat(0.01) == "flat"
    assert classify_grade_pacer_5cat(0.0) == "flat"
    assert classify_grade_pacer_5cat(-0.01) == "flat"
    assert classify_grade_pacer_5cat(-0.02) == "down"
    assert classify_grade_pacer_5cat(-0.10) == "down"
    assert classify_grade_pacer_5cat(-0.25) == "steep_down"
    assert classify_grade_pacer_5cat(-0.30) == "steep_down"


def test_classify_grade_pacer_5cat_flat_with_elevation_delta() -> None:
    assert classify_grade_pacer_5cat(0.03, elevation_delta_per_km=5.0) == "flat"
    assert classify_grade_pacer_5cat(-0.03, elevation_delta_per_km=8.0) == "flat"
    assert classify_grade_pacer_5cat(0.03, elevation_delta_per_km=15.0) == "run_up"


def test_classify_grade_elevation_8cat_boundaries() -> None:
    assert classify_grade_elevation_8cat(-0.6) == "grade_lt_neg_0_5"
    assert classify_grade_elevation_8cat(-0.5) == "grade_lt_neg_0_25"
    assert classify_grade_elevation_8cat(-0.3) == "grade_lt_neg_0_25"
    assert classify_grade_elevation_8cat(-0.25) == "grade_lt_neg_0_05"
    assert classify_grade_elevation_8cat(-0.1) == "grade_lt_neg_0_05"
    assert classify_grade_elevation_8cat(-0.05) == "grade_neutral"
    assert classify_grade_elevation_8cat(0.0) == "grade_neutral"
    assert classify_grade_elevation_8cat(0.05) == "grade_lt_0_1"
    assert classify_grade_elevation_8cat(0.1) == "grade_lt_0_25"
    assert classify_grade_elevation_8cat(0.25) == "grade_lt_0_5"
    assert classify_grade_elevation_8cat(0.5) == "grade_ge_0_5"
