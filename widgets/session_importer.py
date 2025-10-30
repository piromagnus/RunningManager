"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Session importer widget for importing planned sessions into templates.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from utils.helpers import clean_notes, default_template_title


def render_session_importer(
    sessions_for_import: List[Dict[str, Any]],
    state: Dict[str, Any],
) -> None:
    """Render session importer dropdown and handle session import.

    Args:
        sessions_for_import: List of planned sessions available for import
        state: Session state dictionary to update
    """
    session_options = ["Aucune"] + [
        f"{rec.get('date')} • {rec.get('type')} ({rec.get('plannedSessionId')})"
        for rec in sessions_for_import
    ]
    selected_session_label = st.selectbox(
        "Importer une session planifiée",
        session_options,
        index=0,
        key="creator-import-session",
    )
    if selected_session_label != "Aucune":
        idx = session_options.index(selected_session_label) - 1
        base_session = sessions_for_import[idx]
        state["sessionPayload"] = {
            "type": base_session.get("type"),
            "plannedDistanceKm": base_session.get("plannedDistanceKm"),
            "plannedDurationSec": base_session.get("plannedDurationSec"),
            "plannedAscentM": base_session.get("plannedAscentM"),
            "targetType": base_session.get("targetType"),
            "targetLabel": base_session.get("targetLabel"),
            "notes": base_session.get("notes"),
            "stepEndMode": base_session.get("stepEndMode"),
            "stepsJson": base_session.get("stepsJson"),
        }
        state["templateTitle"] = default_template_title(base_session)
        state["templateNotes"] = clean_notes(base_session.get("notes"))
        state["templateId"] = None
        st.session_state["creator-template-title"] = state["templateTitle"]
        st.session_state["creator-template-notes"] = state["templateNotes"]
        st.session_state["creator-session-notes"] = state["templateNotes"]
        st.rerun()

