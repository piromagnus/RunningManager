"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Elevation-related helpers.
"""

from __future__ import annotations


def compute_avg_grade(elev_gain_m: float, elev_loss_m: float, distance_km: float) -> float:
    """Compute average grade from elevation gain/loss and distance.

    Rules:
    - If D+ >= 2 * D-: use D+ / distance
    - If D- >= 2 * D+: use -D- / distance
    - Otherwise: use (D+ - D-) / distance

    Args:
        elev_gain_m: Total elevation gain in meters
        elev_loss_m: Total elevation loss in meters
        distance_km: Distance in kilometers

    Returns:
        Average grade (decimal, not percentage)
    """
    if distance_km <= 0:
        return 0.0

    if elev_gain_m >= 2 * elev_loss_m:
        return elev_gain_m / (distance_km * 1000)
    if elev_loss_m >= 2 * elev_gain_m:
        return -elev_loss_m / (distance_km * 1000)
    return (elev_gain_m - elev_loss_m) / (distance_km * 1000)
