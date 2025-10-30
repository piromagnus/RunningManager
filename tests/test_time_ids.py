"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from utils.time import today_local, iso_week_start, iso_week_end
from utils.ids import new_id
import datetime as dt


def test_week_boundaries():
    d = dt.date(2025, 10, 6)  # Monday
    start = iso_week_start(d)
    end = iso_week_end(d)
    assert start.weekday() == 0
    assert end.weekday() == 6


def test_new_id_unique():
    a = new_id()
    b = new_id()
    assert a != b
    assert len(a) > 10
