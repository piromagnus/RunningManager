"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Tests for prospective empirical/power-law HRR blend helpers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "predict_race_segment_hrr_adaptive",
    REPO / "scripts" / "predict_race_segment_hrr_adaptive.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_mod)


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "durationMin": [60.0, 180.0, 240.0, 360.0, 480.0],
            "activityCount": [10, 8, 6, 4, 3],
            "targetHrr": [0.80, 0.76, 0.67, 0.62, 0.62],
        }
    )


def test_empirical_hrr_steps_with_duration() -> None:
    w = _windows()
    assert abs(_mod.empirical_hrr_for_duration(3 * 3600, w) - 0.76) < 1e-9
    assert abs(_mod.empirical_hrr_for_duration(10 * 3600, w) - 0.62) < 1e-9
    assert abs(_mod.empirical_hrr_for_duration(30 * 60, w) - 0.80) < 1e-9


def test_blend_keeps_powerlaw_below_four_hours() -> None:
    target, weight = _mod.blend_powerlaw_empirical_hrr(
        powerlaw_hrr=0.78,
        empirical_hrr=0.76,
        predicted_sec=3.2 * 3600,
        w_max=0.70,
    )
    assert weight == 0.0
    assert abs(target - 0.78) < 1e-12


def test_blend_pulls_ultra_toward_empirical() -> None:
    target, weight = _mod.blend_powerlaw_empirical_hrr(
        powerlaw_hrr=0.75,
        empirical_hrr=0.617333,
        predicted_sec=10.5 * 3600,
        w_max=0.70,
    )
    assert abs(weight - 0.70) < 1e-12
    expected = 0.3 * 0.75 + 0.7 * 0.617333
    assert abs(target - expected) < 1e-9
    # Near observed Échappée mean HRR without using race HR.
    assert 0.65 < target < 0.66
