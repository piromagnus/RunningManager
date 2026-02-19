"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Session state helpers for Streamlit.
"""

from __future__ import annotations

import streamlit as st

from utils.constants import (
    STATE_COACH_SETTINGS,
    STATE_CURRENT_ATHLETE_ID,
    STATE_OAUTH_STATE,
    STATE_STRAVA_TOKENS_META,
)


def init_session_state() -> None:
    if STATE_CURRENT_ATHLETE_ID not in st.session_state:
        st.session_state[STATE_CURRENT_ATHLETE_ID] = None
    if STATE_COACH_SETTINGS not in st.session_state:
        st.session_state[STATE_COACH_SETTINGS] = {}
    if STATE_OAUTH_STATE not in st.session_state:
        st.session_state[STATE_OAUTH_STATE] = None
    if STATE_STRAVA_TOKENS_META not in st.session_state:
        st.session_state[STATE_STRAVA_TOKENS_META] = {"hasTokens": False}


def get_current_athlete_id():
    return st.session_state.get(STATE_CURRENT_ATHLETE_ID)


def set_current_athlete_id(athlete_id: str | None) -> None:
    st.session_state[STATE_CURRENT_ATHLETE_ID] = athlete_id
