"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Dashboard state management utilities.

Handles session state persistence for dashboard preferences.
"""

from __future__ import annotations

import datetime as dt
from typing import Tuple

import pandas as pd
import streamlit as st


def get_dashboard_date_range(
    min_date: dt.date,
    max_date: dt.date,
    default_days: int | None = None,
    default_months: int | None = None,
) -> Tuple[dt.datetime, dt.datetime]:
    """Get or initialize dashboard date range from session state.

    Args:
        min_date: Minimum available date
        max_date: Maximum available date
        default_days: Default number of days to show (mutually exclusive with default_months)
        default_months: Default number of months to show (mutually exclusive with default_days)

    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    default_end = max_date
    if default_months is not None:
        default_start = max(min_date, (pd.Timestamp(default_end) - pd.DateOffset(months=default_months)).date())
    elif default_days is not None:
        default_start = max(min_date, default_end - pd.Timedelta(days=default_days).to_pytimedelta())
    else:
        # Fallback to 3 months if neither specified
        default_start = max(min_date, (pd.Timestamp(default_end) - pd.DateOffset(months=3)).date())

    if "dashboard_range" not in st.session_state:
        st.session_state["dashboard_range"] = (
            pd.Timestamp(default_start).to_pydatetime(),
            pd.Timestamp(default_end).to_pydatetime(),
        )

    return st.session_state["dashboard_range"]


def set_dashboard_date_range(start_dt: dt.datetime, end_dt: dt.datetime) -> None:
    """Save dashboard date range to session state.

    Args:
        start_dt: Start datetime
        end_dt: End datetime
    """
    st.session_state["dashboard_range"] = (start_dt, end_dt)


def set_dashboard_date_range_quick(
    max_date: dt.date, days: int | None = None, months: int | None = None, years: int | None = None
) -> None:
    """Set dashboard date range using a quick preset.

    Args:
        max_date: Maximum date (typically today)
        days: Number of days to go back (mutually exclusive with months/years)
        months: Number of months to go back (mutually exclusive with days/years)
        years: Number of years to go back (mutually exclusive with days/months)
    """
    if days is not None:
        start_date = (pd.Timestamp(max_date) - pd.Timedelta(days=days)).to_pydatetime()
    elif months is not None:
        start_date = (pd.Timestamp(max_date) - pd.DateOffset(months=months)).to_pydatetime()
    elif years is not None:
        start_date = (pd.Timestamp(max_date) - pd.DateOffset(years=years)).to_pydatetime()
    else:
        return

    end_date = pd.Timestamp(max_date).to_pydatetime()
    st.session_state["dashboard_range"] = (start_date, end_date)

