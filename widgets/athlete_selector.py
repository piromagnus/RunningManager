"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Athlete selector widget for Streamlit pages.

Provides a consistent athlete selection interface across multiple pages.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from persistence.repositories import AthletesRepo


def select_athlete(athletes_repo: AthletesRepo) -> Optional[str]:
    """Render athlete selector and return selected athlete ID.

    Args:
        athletes_repo: AthletesRepo instance for querying athletes

    Returns:
        Optional[str]: Selected athlete ID or None if no athletes available
    """
    df = athletes_repo.list()
    if df.empty:
        st.warning("Aucun athlète enregistré.")
        return None
    options = {
        f"{row.get('name') or 'Sans nom'} ({row.get('athleteId')})": row.get("athleteId")
        for _, row in df.iterrows()
    }
    label = st.selectbox("Athlète", list(options.keys()))
    return options.get(label)

