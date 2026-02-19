"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Time helpers for ISO week boundaries and local date.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd


def today_local() -> dt.date:
    return dt.date.today()


def iso_week_start(d: dt.date) -> dt.datetime:
    # Monday is 1, Sunday is 7; convert to 0-based Monday
    weekday = d.isoweekday()  # 1..7
    monday = d - dt.timedelta(days=weekday - 1)
    return dt.datetime.combine(monday, dt.time.min)


def iso_week_end(d: dt.date) -> dt.datetime:
    start = iso_week_start(d)
    end = start + dt.timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
    return end

def to_date(value: object) -> Optional[dt.date]:
    if value in (None, "", "NaT"):
        return None
    try:
        if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
            return value
        parsed = pd.to_datetime(value)
        if pd.isna(parsed):
            return None
        return parsed.date()
    except Exception:
        return None


def parse_timestamp(value: object) -> Optional[pd.Timestamp]:
    """Parse a timestamp value to pandas Timestamp.

    Args:
        value: Timestamp value (string, datetime, date, None, etc.)

    Returns:
        Parsed pandas Timestamp or None if parsing fails
    """
    if value in (None, "", "NaT"):
        return None
    try:
        if isinstance(value, pd.Timestamp):
            return value
        if isinstance(value, dt.datetime):
            return pd.Timestamp(value)
        if isinstance(value, dt.date):
            return pd.Timestamp(dt.datetime.combine(value, dt.time.min))
        parsed = pd.to_datetime(value)
        if pd.isna(parsed):
            return None
        return parsed
    except Exception:
        return None


def ensure_datetime(value: object) -> Optional[dt.datetime]:
    """Ensure a value is a datetime, converting if necessary.

    Args:
        value: Value to convert (datetime, date, string, etc.)

    Returns:
        Datetime object or None if conversion fails
    """
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime()
    if isinstance(parsed, dt.datetime):
        return parsed
    return None


def compute_segment_time(
    distance_eq_km: float, distance_km: float, speed_eq_kmh: float, speed_kmh: float
) -> int:
    """Compute segment time from speed.

    Args:
        distance_eq_km: Distance-equivalent in km
        distance_km: Actual distance in km
        speed_eq_kmh: Speed-equivalent in km/h (takes precedence)
        speed_kmh: Speed in km/h

    Returns:
        Time in seconds
    """
    if speed_eq_kmh > 0:
        return int(round(3600 * distance_eq_km / speed_eq_kmh))
    if speed_kmh > 0:
        return int(round(3600 * distance_km / speed_kmh))
    return 0
