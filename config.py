"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

METRICS = ["Time", "Distance", "Trimp", "DistEq"]

# Window sizes (in seconds) for maximum speed profile computation
# Used to compute rolling maximum average speeds across different time windows
PROFILE_WINDOW_SIZES = [
    5,10,15, 20, 30, 60, 120, 180, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    4500, 5400, 7200, 9000, 12600, 14400, 18000, 21600, 23400, 27000, 30600, 34200, 36000,
    45000, 54000, 72000, 90000]
