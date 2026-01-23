"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Template action buttons widget for saving, scheduling, and deleting templates.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict

import streamlit as st

from services.metrics_service import MetricsComputationService
from services.session_templates_service import SessionTemplatesService


def render_template_actions(
    athlete_id: str | None,
    template_title: str,
    template_notes: str,
    session_type: str,
    payload_for_save: Dict[str, Any],
    session_notes: str,
    state: Dict[str, Any],
    templates_service: SessionTemplatesService,
    list_templates_cache_clear_func,
    metrics_service: MetricsComputationService | None = None,
) -> None:
    """Render action buttons for template management.

    Args:
        athlete_id: Athlete ID for template operations
        template_title: Template title
        template_notes: Template notes
        session_type: Session type
        payload_for_save: Payload dictionary ready for saving
        session_notes: Session notes
        state: Session state dictionary
        templates_service: SessionTemplatesService instance
        list_templates_cache_clear_func: Function to clear templates cache
        metrics_service: Optional MetricsComputationService instance
    """
    col_save, col_save_new, col_schedule, col_delete = st.columns([1, 1, 1, 1])
    with col_save:
        if st.button("💾 Enregistrer le modèle", key="creator-save-template"):
            if not template_title.strip():
                st.error("Le titre du modèle est requis.")
            elif not athlete_id:
                st.error("Sélectionnez un athlète.")
            else:
                try:
                    if state.get("templateId"):
                        templates_service.update(
                            state["templateId"],
                            title=template_title.strip(),
                            base_type=session_type,
                            payload=payload_for_save,
                            notes=template_notes,
                        )
                        st.session_state["planner_templates_refresh"] = True
                        st.success("Modèle mis à jour.")
                    else:
                        new_id = templates_service.create(
                            athlete_id=athlete_id,
                            title=template_title.strip(),
                            base_type=session_type,
                            payload=payload_for_save,
                            notes=template_notes,
                        )
                        state["templateId"] = new_id
                        st.session_state["planner_templates_refresh"] = True
                        st.success(f"Nouveau modèle enregistré ({new_id}).")
                    list_templates_cache_clear_func()
                    state["sessionPayload"] = dict(payload_for_save)
                except Exception as exc:
                    st.error(f"Impossible d'enregistrer le modèle: {exc}")
    with col_save_new:
        if st.button("🆕 Enregistrer sous un nouveau modèle", key="creator-save-new"):
            if not template_title.strip():
                st.error("Le titre du modèle est requis.")
            elif not athlete_id:
                st.error("Sélectionnez un athlète.")
            else:
                try:
                    new_id = templates_service.create(
                        athlete_id=athlete_id,
                        title=template_title.strip(),
                        base_type=session_type,
                        payload=payload_for_save,
                        notes=template_notes,
                    )
                    state["templateId"] = new_id
                    st.session_state["planner_templates_refresh"] = True
                    st.success(f"Copie enregistrée ({new_id}).")
                    list_templates_cache_clear_func()
                    state["sessionPayload"] = dict(payload_for_save)
                except Exception as exc:
                    st.error(f"Création échouée: {exc}")
    with col_schedule:
        schedule_date = st.date_input(
            "Date de planification",
            value=state.get("date") if isinstance(state.get("date"), dt.date) else dt.date.today(),
            key="creator-schedule-date",
        )
        if st.button("🗓️ Enregistrer & planifier", key="creator-save-schedule"):
            if not template_title.strip():
                st.error("Le titre du modèle est requis.")
            elif not athlete_id:
                st.error("Sélectionnez un athlète.")
            else:
                try:
                    template_id = state.get("templateId")
                    if template_id:
                        templates_service.update(
                            template_id,
                            title=template_title.strip(),
                            base_type=session_type,
                            payload=payload_for_save,
                            notes=template_notes,
                        )
                    else:
                        template_id = templates_service.create(
                            athlete_id=athlete_id,
                            title=template_title.strip(),
                            base_type=session_type,
                            payload=payload_for_save,
                            notes=template_notes,
                        )
                        state["templateId"] = template_id
                    templates_service.apply_to_calendar(
                        template_id, athlete_id, schedule_date, notes=session_notes
                    )
                    if metrics_service:
                        metrics_service.recompute_planned_for_athlete(str(athlete_id))
                    list_templates_cache_clear_func()
                    st.session_state["planner_templates_refresh"] = True
                    st.success("Modèle planifié et appliqué au calendrier.")
                    state["sessionPayload"] = dict(payload_for_save)
                except Exception as exc:
                    st.error(f"Impossible de planifier le modèle: {exc}")
    with col_delete:
        if state.get("templateId"):
            if st.button("🗑️ Supprimer le modèle", key="creator-delete"):
                try:
                    templates_service.delete(state["templateId"])
                    list_templates_cache_clear_func()
                    st.session_state["planner_templates_refresh"] = True
                    st.success("Modèle supprimé.")
                    state.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Suppression échouée: {exc}")

