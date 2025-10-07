"""Time helpers for ISO week boundaries and local date."""

from __future__ import annotations

import datetime as dt


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
