"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Session state helpers for Streamlit.
"""

from __future__ import annotations

import streamlit as st


CURRENT_ATHLETE_ID = "current_athlete_id"
COACH_SETTINGS = "coach_settings"
OAUTH_STATE = "oauth_state"
STRAVA_TOKENS_META = "strava_tokens_meta"


def init_session_state() -> None:
    if CURRENT_ATHLETE_ID not in st.session_state:
        st.session_state[CURRENT_ATHLETE_ID] = None
    if COACH_SETTINGS not in st.session_state:
        st.session_state[COACH_SETTINGS] = {}
    if OAUTH_STATE not in st.session_state:
        st.session_state[OAUTH_STATE] = None
    if STRAVA_TOKENS_META not in st.session_state:
        st.session_state[STRAVA_TOKENS_META] = {"hasTokens": False}


def get_current_athlete_id():
    return st.session_state.get(CURRENT_ATHLETE_ID)


def set_current_athlete_id(athlete_id: str | None) -> None:
    st.session_state[CURRENT_ATHLETE_ID] = athlete_id
