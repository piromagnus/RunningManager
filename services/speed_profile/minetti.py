"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations


def minetti_energy_cost_walking(grade: float) -> float:
    """Calculate energy cost of walking at a given grade using Minetti et al. (2002)."""
    if grade >= 0.5:
        grade = 0.5
    elif grade <= -0.5:
        grade = -0.5

    return (
        280.5 * grade**5
        - 58.7 * grade**4
        - 76.8 * grade**3
        + 51.9 * grade**2
        + 19.6 * grade
        + 2.5
    )


def minetti_energy_cost_running(grade: float) -> float:
    """Calculate energy cost of running at a given grade using Minetti et al. (2002)."""
    if grade >= 0.5:
        grade = 0.5
    elif grade <= -0.5:
        grade = -0.5

    return (
        155.4 * grade**5
        - 30.4 * grade**4
        - 43.3 * grade**3
        + 46.3 * grade**2
        + 19.5 * grade
        + 3.6
    )
