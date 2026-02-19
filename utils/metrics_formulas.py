"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


def compute_trimp_hr_reserve(
    avg_hr: Optional[float],
    duration_sec: Optional[float],
    hr_rest: Optional[float],
    hr_max: Optional[float],
) -> float:
    """Compute TRIMP using HR reserve weighting (Banister-style)."""
    if avg_hr is None or duration_sec is None:
        return 0.0
    if avg_hr <= 0 or duration_sec <= 0:
        return 0.0
    if hr_rest is None or hr_max is None:
        return 0.0
    if hr_max <= hr_rest:
        return 0.0

    hrr = (avg_hr - hr_rest) / (hr_max - hr_rest)
    hrr = max(0.0, min(hrr, 1.2))
    if hrr <= 0:
        return 0.0

    duration_hours = duration_sec / 3600.0
    return duration_hours * hrr * 0.64 * math.exp(1.92 * hrr)


def compute_trimp_hr_reserve_from_profile(
    avg_hr: Optional[float],
    duration_sec: Optional[float],
    hr_profile: Optional[Tuple[float, float]],
) -> float:
    """Compute TRIMP using HR reserve, taking (hr_rest, hr_max) tuple."""
    if hr_profile is None:
        return 0.0
    hr_rest, hr_max = hr_profile
    return compute_trimp_hr_reserve(avg_hr, duration_sec, hr_rest, hr_max)
