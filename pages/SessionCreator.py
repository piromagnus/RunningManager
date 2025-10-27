from __future__ import annotations

import json
import datetime as dt
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from utils.config import load_config
from utils.formatting import fmt_decimal, fmt_m, set_locale
from utils.styling import apply_theme
from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo, PlannedSessionsRepo
from services.planner_service import PlannerService
from services.session_templates_service import SessionTemplatesService
from ui.interval_editor import render_interval_editor


st.set_page_config(page_title="Running Manager - Session Creator", layout="wide")
apply_theme()
st.title("Session Template Creator")

st.markdown(
    """
    <style>
    .rm-card {
      padding: 10px 12px;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 8px;
      margin-bottom: 10px;
      background: rgba(0,0,0,0.15);
    }
    .rm-loop-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 8px 10px; margin-bottom: 10px; }
    .rm-interval-action { background: rgba(255,255,255,0.02); border-radius: 6px; padding: 6px 8px; margin-bottom: 6px; }
    .rm-interval-editor .stNumberInput label,
    .rm-interval-editor .stSelectbox label,
    .rm-interval-editor .stTextInput label { font-size: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
ath_repo = AthletesRepo(storage)
sessions_repo = PlannedSessionsRepo(storage)
planner = PlannerService(storage)
templates_service = SessionTemplatesService(storage)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def _format_duration(seconds: int) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h{minutes:02d}"
    return f"{minutes} min"


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


def _default_template_title(session: Dict[str, Any]) -> str:
    raw_type = str(session.get("type") or "Session").replace("_", " ")
    title_part = raw_type.title()
    date_part = str(session.get("date") or "")
    return f"{title_part} {date_part}".strip()


def _clean_notes(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _render_fundamental_form(
    athlete_id: Optional[str],
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[float]]:
    planned_distance_km = payload.get("plannedDistanceKm")
    planned_duration_sec = payload.get("plannedDurationSec")
    planned_ascent_m = payload.get("plannedAscentM")
    target_type = payload.get("targetType") or "pace"
    target_label = payload.get("targetLabel") or None

    mode_options = ["distance", "duration"]
    distance_present = _coerce_float(planned_distance_km, 0.0) > 0
    duration_present = _coerce_int(planned_duration_sec, 0) > 0
    default_index = 0 if distance_present or not duration_present else 1
    mode_choice = st.radio(
        "Mode de saisie",
        mode_options,
        index=default_index,
        format_func=lambda x: "Distance + D+" if x == "distance" else "Durée + D+ mini",
        horizontal=True,
        key="creator-fundamental-mode",
    )

    distance_eq_preview = None
    result: Dict[str, Any] = {}
    athlete_ref = str(athlete_id or "")

    if mode_choice == "distance":
        distance_input = st.number_input(
            "Distance planifiée (km)",
            min_value=0.0,
            value=_coerce_float(planned_distance_km, 0.0),
            step=0.1,
            key="creator-fundamental-distance",
        )
        ascent_input = st.number_input(
            "Ascension planifiée (m)",
            min_value=0,
            value=_coerce_int(planned_ascent_m, 0),
            step=50,
            key="creator-fundamental-ascent",
        )
        derived = planner.derive_from_distance(athlete_ref, distance_input, ascent_input)
        result["plannedDistanceKm"] = derived["distanceKm"]
        result["plannedDurationSec"] = derived["durationSec"]
        result["plannedAscentM"] = int(ascent_input)
        distance_eq_preview = derived["distanceEqKm"]
        st.caption(
            f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km • Durée estimée ≈ {_format_duration(result['plannedDurationSec'])}"
        )
    else:
        duration_input = st.number_input(
            "Durée planifiée (sec)",
            min_value=0,
            value=_coerce_int(planned_duration_sec, 3600),
            step=300,
            key="creator-fundamental-duration",
        )
        ascent_input = st.number_input(
            "Ascension minimale (m)",
            min_value=0,
            value=_coerce_int(planned_ascent_m, 0),
            step=50,
            key="creator-fundamental-min-ascent",
        )
        derived = planner.derive_from_duration(athlete_ref, int(duration_input), ascent_input)
        result["plannedDistanceKm"] = derived["distanceKm"]
        result["plannedDurationSec"] = derived["durationSec"]
        result["plannedAscentM"] = int(ascent_input)
        distance_eq_preview = derived["distanceEqKm"]
        st.caption(
            f"Distance estimée ≈ {fmt_decimal(result['plannedDistanceKm'], 1)} km • Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km"
        )

    target_options = ["none", "hr", "pace"]
    selection = target_type if target_type in target_options else "pace"
    chosen = st.selectbox(
        "Cible",
        target_options,
        index=target_options.index(selection),
        key="creator-fundamental-target",
    )
    if chosen == "none":
        result["targetType"] = None
        result["targetLabel"] = None
    else:
        result["targetType"] = chosen
        result["targetLabel"] = "Fundamental"
        st.caption("Seuil fixé automatiquement sur Fundamental.")

    return result, distance_eq_preview


def _render_long_run_form(
    athlete_id: Optional[str],
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[float]]:
    planned_distance_km = payload.get("plannedDistanceKm")
    planned_duration_sec = payload.get("plannedDurationSec")
    planned_ascent_m = payload.get("plannedAscentM")
    target_type = payload.get("targetType")
    target_label = payload.get("targetLabel")

    mode_options = ["distance", "duration"]
    distance_present = _coerce_float(planned_distance_km, 0.0) > 0
    duration_present = _coerce_int(planned_duration_sec, 0) > 0
    default_index = 0 if distance_present or not duration_present else 1
    mode_choice = st.radio(
        "Mode de saisie",
        mode_options,
        index=default_index,
        format_func=lambda x: "Distance + D+" if x == "distance" else "Durée + D+ mini",
        horizontal=True,
        key="creator-long-mode",
    )

    distance_eq_preview = None
    result: Dict[str, Any] = {}
    athlete_ref = str(athlete_id or "")

    if mode_choice == "distance":
        distance_input = st.number_input(
            "Distance planifiée (km)",
            min_value=0.0,
            value=_coerce_float(planned_distance_km, 0.0),
            step=0.5,
            key="creator-long-distance",
        )
        ascent_input = st.number_input(
            "Ascension planifiée (m)",
            min_value=0,
            value=_coerce_int(planned_ascent_m, 500),
            step=50,
            key="creator-long-ascent",
        )
        derived = planner.derive_from_distance(athlete_ref, distance_input, ascent_input)
        result["plannedDistanceKm"] = derived["distanceKm"]
        result["plannedDurationSec"] = derived["durationSec"]
        result["plannedAscentM"] = int(ascent_input)
        distance_eq_preview = derived["distanceEqKm"]
        st.caption(
            f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km • Durée estimée ≈ {_format_duration(result['plannedDurationSec'])}"
        )
    else:
        duration_input = st.number_input(
            "Durée planifiée (sec)",
            min_value=0,
            value=_coerce_int(planned_duration_sec, 7200),
            step=300,
            key="creator-long-duration",
        )
        ascent_input = st.number_input(
            "Ascension minimale (m)",
            min_value=0,
            value=_coerce_int(planned_ascent_m, 500),
            step=50,
            key="creator-long-min-ascent",
        )
        derived = planner.derive_from_duration(athlete_ref, int(duration_input), ascent_input)
        result["plannedDistanceKm"] = derived["distanceKm"]
        result["plannedDurationSec"] = derived["durationSec"]
        result["plannedAscentM"] = int(ascent_input)
        distance_eq_preview = derived["distanceEqKm"]
        st.caption(
            f"Distance estimée ≈ {fmt_decimal(result['plannedDistanceKm'], 1)} km • Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km"
        )

    target_type = target_type if isinstance(target_type, str) else "none"
    target_options = ["none", "hr", "pace"]
    target_choice = st.selectbox(
        "Cible",
        target_options,
        index=(0 if target_type not in target_options else target_options.index(target_type)),
        key="creator-long-target",
    )
    if target_choice in ("hr", "pace"):
        names = planner.list_threshold_names(str(athlete_id or "")) if athlete_id else ["Fundamental", "Threshold 30", "Threshold 60"]
        idx = names.index(target_label) if target_label in names else 0
        selected_label = st.selectbox("Seuil", names, index=idx, key="creator-long-threshold")
        result["targetType"] = target_choice
        result["targetLabel"] = selected_label
    else:
        result["targetType"] = None
        result["targetLabel"] = None

    return result, distance_eq_preview


def _render_race_form(
    athlete_id: Optional[str],
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[float]]:
    planned_distance_km = _coerce_float(payload.get("plannedDistanceKm"), 0.0)
    planned_duration_sec = _coerce_int(payload.get("plannedDurationSec"), 0)
    planned_ascent_m = _coerce_int(payload.get("plannedAscentM"), 0)

    distance_input = st.number_input(
        "Distance (km)",
        min_value=0.0,
        value=planned_distance_km,
        step=0.1,
        key="creator-race-distance",
    )
    ascent_input = st.number_input(
        "Ascension (m)",
        min_value=0,
        value=planned_ascent_m,
        step=50,
        key="creator-race-ascent",
    )
    target_time_input = st.number_input(
        "Temps cible (sec)",
        min_value=0,
        value=planned_duration_sec,
        step=60,
        key="creator-race-duration",
    )

    result = {
        "plannedDistanceKm": float(distance_input),
        "plannedAscentM": int(ascent_input),
        "plannedDurationSec": int(target_time_input),
        "targetType": "race",
        "targetLabel": None,
    }
    distance_eq_preview = planner.compute_distance_eq_km(result["plannedDistanceKm"], result["plannedAscentM"])
    st.caption(
        f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km • "
        f"Temps cible ≈ {_format_duration(result['plannedDurationSec'])}"
    )
    return result, distance_eq_preview


def _render_interval_form(
    athlete_id: Optional[str],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    thr_names = planner.list_threshold_names(str(athlete_id or "")) if athlete_id else ["Threshold 60", "Threshold 30", "Fundamental", "MVA", "Max speed"]
    serialized_steps = render_interval_editor("creator", payload.get("stepsJson"), thr_names)
    step_end_mode_default = payload.get("stepEndMode") or "auto"
    step_end_mode = st.selectbox(
        "Mode de fin",
        ["auto", "lap"],
        index=(["auto", "lap"].index(step_end_mode_default) if step_end_mode_default in ("auto", "lap") else 0),
        key="creator-interval-end-mode",
    )
    planned_duration_sec = planner.estimate_interval_duration_sec(serialized_steps)
    planned_distance_km = planner.estimate_interval_distance_km(str(athlete_id or ""), serialized_steps)
    planned_ascent_m = planner.estimate_interval_ascent_m(serialized_steps)
    distance_eq_preview = planner.compute_distance_eq_km(planned_distance_km, planned_ascent_m)
    st.caption(
        f"Durée ≈ {_format_duration(planned_duration_sec)} • "
        f"Distance ≈ {fmt_decimal(planned_distance_km, 1)} km • "
        f"D+ ≈ {fmt_m(planned_ascent_m)} • "
        f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km"
    )
    return {
        "stepsJson": json.dumps(serialized_steps, ensure_ascii=False, separators=(",", ":")),
        "stepEndMode": step_end_mode,
        "plannedDurationSec": planned_duration_sec,
        "plannedDistanceKm": planned_distance_km,
        "plannedAscentM": planned_ascent_m,
        "targetType": None,
        "targetLabel": None,
    }


def _render_session_form(
    athlete_id: Optional[str],
    session_type: str,
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[float]]:
    session_type = (session_type or "FUNDAMENTAL_ENDURANCE").upper()
    if session_type == "FUNDAMENTAL_ENDURANCE":
        return _render_fundamental_form(athlete_id, payload)
    if session_type == "LONG_RUN":
        return _render_long_run_form(athlete_id, payload)
    if session_type == "RACE":
        return _render_race_form(athlete_id, payload)
    if session_type == "INTERVAL_SIMPLE":
        interval_result = _render_interval_form(athlete_id, payload)
        return interval_result, None
    st.warning(f"Type inconnu {session_type}, utilisation de FUNDAMENTAL_ENDURANCE.")
    return _render_fundamental_form(athlete_id, payload)


def _session_payload_for_save(session_type: str, form_data: Dict[str, Any], notes: str) -> Dict[str, Any]:
    payload = {
        "type": session_type,
        "plannedDistanceKm": form_data.get("plannedDistanceKm"),
        "plannedDurationSec": form_data.get("plannedDurationSec"),
        "plannedAscentM": form_data.get("plannedAscentM"),
        "targetType": form_data.get("targetType"),
        "targetLabel": form_data.get("targetLabel"),
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
template_map = {t.get("templateId"): t for t in templates}
template_labels = ["Créer un nouveau modèle"] + [
    f"{t.get('title') or 'Sans titre'} ({t.get('templateId')})" for t in templates
]
selected_label = st.selectbox("Modèle existant", template_labels, index=0, key="creator-template-select")
selected_template_id = None
if selected_label != "Créer un nouveau modèle":
    selected_index = template_labels.index(selected_label) - 1
    selected_template_id = templates[selected_index].get("templateId")

if selected_template_id != state.get("templateId"):
    if selected_template_id:
        tpl = template_map.get(selected_template_id)
        if tpl:
            state["templateId"] = selected_template_id
            state["templateTitle"] = tpl.get("title") or ""
            state["templateNotes"] = _clean_notes(tpl.get("notes"))
            state["sessionPayload"] = tpl.get("payload") or {}
            st.session_state["creator-template-title"] = state["templateTitle"]
            st.session_state["creator-template-notes"] = state["templateNotes"]
            st.session_state["creator-session-notes"] = _clean_notes(state["sessionPayload"].get("notes"))
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

sessions_for_import = list_sessions(athlete_id)
session_options = ["Aucune"] + [
    f"{rec.get('date')} • {rec.get('type')} ({rec.get('plannedSessionId')})" for rec in sessions_for_import
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
    state["templateTitle"] = _default_template_title(base_session)
    state["templateNotes"] = _clean_notes(base_session.get("notes"))
    state["templateId"] = None
    st.session_state["creator-template-title"] = state["templateTitle"]
    st.session_state["creator-template-notes"] = state["templateNotes"]
    st.session_state["creator-session-notes"] = state["templateNotes"]
    st.rerun()

session_payload = state.get("sessionPayload") or {}
session_type = (session_payload.get("type") or state.get("baseType") or "FUNDAMENTAL_ENDURANCE").upper()
base_type_options = ["FUNDAMENTAL_ENDURANCE", "LONG_RUN", "RACE", "INTERVAL_SIMPLE"]
session_type = st.selectbox(
    "Type de session",
    base_type_options,
    index=base_type_options.index(session_type) if session_type in base_type_options else 0,
    key="creator-session-type",
)
state["baseType"] = session_type

template_title_default = state.get("templateTitle") or _default_template_title({"type": session_type})
template_title = st.text_input("Titre du modèle", value=template_title_default, key="creator-template-title")
state["templateTitle"] = template_title
template_notes_default = state.get("templateNotes") or ""
template_notes = st.text_area("Notes du modèle", value=template_notes_default, key="creator-template-notes")
state["templateNotes"] = template_notes

session_notes_default = _clean_notes(session_payload.get("notes"))
session_notes = st.text_area("Notes de session", value=session_notes_default, key="creator-session-notes")

form_result, distance_eq_preview = _render_session_form(athlete_id, session_type, session_payload)

if distance_eq_preview is not None:
    st.caption(f"Distance-eq courante ≈ {fmt_decimal(distance_eq_preview, 1)} km")

payload_for_save = _session_payload_for_save(session_type, form_result, session_notes)

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
                list_templates.clear()
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
                list_templates.clear()
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
                templates_service.apply_to_calendar(template_id, athlete_id, schedule_date, notes=session_notes)
                list_templates.clear()
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
                list_templates.clear()
                st.session_state["planner_templates_refresh"] = True
                st.success("Modèle supprimé.")
                state.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Suppression échouée: {exc}")

if st.button("↩︎ Retour au planner", key="creator-back"):
    st.switch_page("pages/Planner.py")
