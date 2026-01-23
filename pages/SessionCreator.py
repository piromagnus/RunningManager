"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo, PlannedSessionsRepo
from services.metrics_service import MetricsComputationService
from services.planner_service import PlannerService
from services.session_templates_service import SessionTemplatesService
from utils.config import load_config
from utils.formatting import fmt_decimal, set_locale
from utils.helpers import clean_notes, default_template_title
from utils.styling import apply_theme
from widgets.session_forms import (
    render_fundamental_form,
    render_interval_form,
    render_long_run_form,
    render_race_form,
)
from widgets.session_importer import render_session_importer
from widgets.template_actions import render_template_actions
from widgets.template_selector import render_template_selector

st.set_page_config(page_title="Running Manager - Session Creator", layout="wide")
apply_theme()
st.title("Session Template Creator")

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
ath_repo = AthletesRepo(storage)
sessions_repo = PlannedSessionsRepo(storage)
planner = PlannerService(storage)
templates_service = SessionTemplatesService(storage)
metrics_service = MetricsComputationService(storage)


@st.cache_data(ttl=10)
def list_templates(athlete_id: Optional[str]) -> List[Dict[str, Any]]:
    return templates_service.list(athlete_id=athlete_id) if athlete_id else templates_service.list()


@st.cache_data(ttl=10)
def list_sessions(athlete_id: Optional[str]) -> List[Dict[str, Any]]:
    if not athlete_id:
        return []
    df = sessions_repo.list(athleteId=athlete_id)
    if df.empty:
        return []
    df = df.sort_values("date", ascending=False)
    return df.head(50).to_dict(orient="records")

def _render_session_form(
    athlete_id: Optional[str],
    session_type: str,
    payload: Dict[str, Any],
    planner: PlannerService,
) -> Tuple[Dict[str, Any], Optional[float]]:
    """Render session form based on session type.

    Args:
        athlete_id: Athlete ID for planning calculations
        session_type: Type of session (FUNDAMENTAL_ENDURANCE, LONG_RUN, RACE, INTERVAL_SIMPLE)
        payload: Existing session payload data
        planner: PlannerService instance for calculations

    Returns:
        Tuple of (form_result_dict, distance_eq_preview)
    """
    session_type = (session_type or "FUNDAMENTAL_ENDURANCE").upper()
    if session_type == "FUNDAMENTAL_ENDURANCE":
        return render_fundamental_form(athlete_id, payload, planner)
    if session_type == "LONG_RUN":
        return render_long_run_form(athlete_id, payload, planner)
    if session_type == "RACE":
        return render_race_form(athlete_id, payload, planner)
    if session_type == "INTERVAL_SIMPLE":
        interval_result = render_interval_form(athlete_id, payload, planner)
        return interval_result, None
    st.warning(f"Type inconnu {session_type}, utilisation de FUNDAMENTAL_ENDURANCE.")
    return render_fundamental_form(athlete_id, payload, planner)


def _session_payload_for_save(
    session_type: str, form_data: Dict[str, Any], notes: str
) -> Dict[str, Any]:
    payload = {
        "type": session_type,
        "plannedDistanceKm": form_data.get("plannedDistanceKm"),
        "plannedDurationSec": form_data.get("plannedDurationSec"),
        "plannedAscentM": form_data.get("plannedAscentM"),
        "targetType": form_data.get("targetType"),
        "targetLabel": form_data.get("targetLabel"),
        "raceName": form_data.get("raceName"),
        "notes": notes,
        "stepEndMode": form_data.get("stepEndMode"),
        "stepsJson": form_data.get("stepsJson"),
    }
    return payload


state = st.session_state.setdefault("session_creator_state", {})
state.setdefault("templateTitleSource", "auto")
if "sessionPayload" not in state:
    base_type_default = state.get("baseType") or "FUNDAMENTAL_ENDURANCE"
    state["sessionPayload"] = {"type": base_type_default}

prefill = st.session_state.pop("session_creator_prefill", None)
if prefill:
    if isinstance(prefill.get("date"), str):
        try:
            prefill["date"] = dt.date.fromisoformat(prefill["date"])
        except Exception:
            prefill["date"] = dt.date.today()
    state.update(prefill)

ath_df = ath_repo.list()
athlete_options = (
    {
        f"{r.get('name') or 'Unnamed'} ({r.get('athleteId')})": r.get("athleteId")
        for _, r in ath_df.iterrows()
    }
    if not ath_df.empty
    else {}
)
ath_label = st.selectbox("Athlete", list(athlete_options.keys())) if athlete_options else None
athlete_id = athlete_options.get(ath_label) if ath_label else state.get("athleteId")
state["athleteId"] = athlete_id

templates = list_templates(athlete_id)
render_template_selector(templates, state.get("templateId"), state)

sessions_for_import = list_sessions(athlete_id)
render_session_importer(sessions_for_import, state)

session_payload = state.get("sessionPayload") or {}
session_type = (
    session_payload.get("type") or state.get("baseType") or "FUNDAMENTAL_ENDURANCE"
).upper()
base_type_options = ["FUNDAMENTAL_ENDURANCE", "LONG_RUN", "RACE", "INTERVAL_SIMPLE"]
session_type = st.selectbox(
    "Type de session",
    base_type_options,
    index=base_type_options.index(session_type) if session_type in base_type_options else 0,
    key="creator-session-type",
)
state["baseType"] = session_type

template_title_default = state.get("templateTitle") or default_template_title(
    {"type": session_type}
)
template_title = st.text_input(
    "Titre du modèle", value=template_title_default, key="creator-template-title"
)
state["templateTitle"] = template_title
template_notes_default = state.get("templateNotes") or ""
template_notes = st.text_area(
    "Notes du modèle", value=template_notes_default, key="creator-template-notes"
)
state["templateNotes"] = template_notes

session_notes_default = clean_notes(session_payload.get("notes"))
session_notes = st.text_area(
    "Notes de session", value=session_notes_default, key="creator-session-notes"
)

form_result, distance_eq_preview = _render_session_form(
    athlete_id, session_type, session_payload, planner
)

if distance_eq_preview is not None:
    st.caption(f"Distance-eq courante ≈ {fmt_decimal(distance_eq_preview, 1)} km")

payload_for_save = _session_payload_for_save(session_type, form_result, session_notes)

render_template_actions(
    athlete_id,
    template_title,
    template_notes,
    session_type,
    payload_for_save,
    session_notes,
    state,
    templates_service,
    list_templates.clear,
    metrics_service,
)

if st.button("↩︎ Retour au planner", key="creator-back"):
    st.switch_page("pages/Planner.py")
