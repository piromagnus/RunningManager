"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Template selector widget for choosing existing session templates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from utils.helpers import clean_notes


def render_template_selector(
    templates: List[Dict[str, Any]],
    selected_template_id: Optional[str],
    state: Dict[str, Any],
) -> Optional[str]:
    """Render template selector dropdown and handle template selection.

    Args:
        templates: List of available templates
        selected_template_id: Currently selected template ID
        state: Session state dictionary to update

    Returns:
        Selected template ID or None
    """
    template_map = {t.get("templateId"): t for t in templates}
    template_labels = ["Créer un nouveau modèle"] + [
        f"{t.get('title') or 'Sans titre'} ({t.get('templateId')})" for t in templates
    ]
    selected_label = st.selectbox(
        "Modèle existant", template_labels, index=0, key="creator-template-select"
    )
    new_selected_template_id = None
    if selected_label != "Créer un nouveau modèle":
        selected_index = template_labels.index(selected_label) - 1
        new_selected_template_id = templates[selected_index].get("templateId")

    if new_selected_template_id != state.get("templateId"):
        if new_selected_template_id:
            tpl = template_map.get(new_selected_template_id)
            if tpl:
                state["templateId"] = new_selected_template_id
                state["templateTitle"] = tpl.get("title") or ""
                state["templateNotes"] = clean_notes(tpl.get("notes"))
                state["sessionPayload"] = tpl.get("payload") or {}
                st.session_state["creator-template-title"] = state["templateTitle"]
                st.session_state["creator-template-notes"] = state["templateNotes"]
                st.session_state["creator-session-notes"] = clean_notes(
                    state["sessionPayload"].get("notes")
                )
            else:
                state["sessionPayload"] = {}
                state["templateTitle"] = ""
                state["templateNotes"] = ""
                state["templateId"] = None
                st.session_state["creator-template-title"] = ""
                st.session_state["creator-template-notes"] = ""
                st.session_state["creator-session-notes"] = ""
        else:
            state["sessionPayload"] = {}
            state["templateTitle"] = ""
            state["templateNotes"] = ""
            state["templateId"] = None
            st.session_state["creator-template-title"] = ""
            st.session_state["creator-template-notes"] = ""
            st.session_state["creator-session-notes"] = ""
        st.rerun()

    return new_selected_template_id

