from __future__ import annotations

import json
import datetime as dt
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from utils.config import load_config
from utils.formatting import fmt_decimal, fmt_m, set_locale
from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo, PlannedSessionsRepo
from services.planner_service import PlannerService
from services.session_templates_service import SessionTemplatesService


st.set_page_config(page_title="Running Manager - Session Creator", layout="wide")
st.title("Session Template Creator")

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
    # sort most recent first
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


def _render_interval_form(
    athlete_id: Optional[str],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    thr_names = planner.list_threshold_names(str(athlete_id or "")) if athlete_id else ["Threshold 60", "Threshold 30", "Fundamental", "MVA", "Max speed"]
    steps_json = payload.get("stepsJson")
    parsed: Dict[str, Any] = {}
    if steps_json:
        if isinstance(steps_json, dict):
            parsed = steps_json
        else:
            try:
                parsed = json.loads(steps_json)
            except Exception:
                parsed = {}
    warmup_default = int(parsed.get("warmupSec") or 600)
    warmup = int(st.number_input("Warm-up (sec)", min_value=0, value=warmup_default, step=60, key="creator-interval-warmup"))
    cooldown_default = int(parsed.get("cooldownSec") or 600)
    cooldown = int(
        st.number_input("Cool-down (sec)", min_value=0, value=cooldown_default, step=60, key="creator-interval-cooldown")
    )
    mode = st.radio(
        "Interval build mode",
        ["Simple repeats", "Loops"],
        index=(0 if "repeats" in parsed else 1 if "loops" in parsed else 0),
        horizontal=True,
        key="creator-interval-mode",
    )
    if mode == "Simple repeats":
        repeats = parsed.get("repeats", [])
        num_rep = st.number_input("Repeats", min_value=1, value=max(1, len(repeats) or 6), step=1, key="creator-interval-repeats")
        rows = []
        for idx in range(int(num_rep)):
            st.markdown(f"Repeat {idx + 1}")
            cur = repeats[idx] if idx < len(repeats) else {}
            w = int(
                st.number_input(
                    f"Work (sec) #{idx + 1}",
                    min_value=10,
                    value=int(cur.get("workSec") or 60),
                    step=10,
                    key=f"creator-interval-work-{idx}",
                )
            )
            r = int(
                st.number_input(
                    f"Recovery (sec) #{idx + 1}",
                    min_value=10,
                    value=int(cur.get("recoverSec") or 60),
                    step=10,
                    key=f"creator-interval-recover-{idx}",
                )
            )
            tt_default = cur.get("targetType") or "hr"
            tt = st.selectbox(
                f"Target type #{idx + 1}",
                ["hr", "pace", "sensation"],
                index=(["hr", "pace", "sensation"].index(tt_default)),
                key=f"creator-interval-target-type-{idx}",
            )
            if tt in ("hr", "pace"):
                tl_index = (
                    thr_names.index(cur.get("targetLabel", thr_names[0]))
                    if cur.get("targetLabel") in thr_names
                    else 0
                )
                tl = st.selectbox(
                    f"Target label #{idx + 1}",
                    thr_names,
                    index=tl_index,
                    key=f"creator-interval-target-label-{idx}",
                )
            else:
                tl = st.text_input(
                    f"Sensation label #{idx + 1}",
                    value=str(cur.get("targetLabel") or ""),
                    key=f"creator-interval-sensation-{idx}",
                )
            rows.append({"workSec": w, "recoverSec": r, "targetType": tt, "targetLabel": tl})
        steps = {"warmupSec": warmup, "repeats": rows}
    else:
        loops = parsed.get("loops", [])
        loop_count = st.number_input(
            "Number of loops",
            min_value=1,
            value=max(1, len(loops) or 1),
            key="creator-interval-loops-count",
        )
        between = int(
            st.number_input(
                "Between-loop recovery (sec)",
                min_value=0,
                value=int(parsed.get("betweenLoopRecoverSec") or 0),
                step=10,
                key="creator-interval-between",
            )
        )
        built_loops = []
        for li in range(int(loop_count)):
            st.markdown(f"Loop {li + 1}")
            cur_loop = loops[li] if li < len(loops) else {}
            actions = cur_loop.get("actions") or []
            repeats = int(
                st.number_input(
                    f"Loop repeats #{li + 1}",
                    min_value=1,
                    value=int(cur_loop.get("repeats") or 1),
                    key=f"creator-interval-loop-repeats-{li}",
                )
            )
            act_count = st.number_input(
                f"Actions in loop #{li + 1}",
                min_value=1,
                value=max(1, len(actions) or 2),
                key=f"creator-interval-actions-{li}",
            )
            built_actions = []
            for ai in range(int(act_count)):
                st.markdown(f"- Action {ai + 1}")
                cur_act = actions[ai] if ai < len(actions) else {}
                kind = st.selectbox(
                    f"Kind #{li + 1}.{ai + 1}",
                    ["run", "recovery"],
                    index=(0 if (cur_act.get("kind") or "run") == "run" else 1),
                    key=f"creator-interval-kind-{li}-{ai}",
                )
                sec = int(
                    st.number_input(
                        f"Seconds #{li + 1}.{ai + 1}",
                        min_value=5,
                        value=int(cur_act.get("sec") or 60),
                        step=5,
                        key=f"creator-interval-sec-{li}-{ai}",
                    )
                )
                tt_default = cur_act.get("targetType") or "hr"
                tt = st.selectbox(
                    f"Target type #{li + 1}.{ai + 1}",
                    ["hr", "pace", "sensation"],
                    index=(["hr", "pace", "sensation"].index(tt_default)),
                    key=f"creator-interval-target-type-{li}-{ai}",
                )
                if tt in ("hr", "pace"):
                    tl_index = (
                        thr_names.index(cur_act.get("targetLabel", thr_names[0]))
                        if cur_act.get("targetLabel") in thr_names
                        else 0
                    )
                    tl = st.selectbox(
                        f"Target label #{li + 1}.{ai + 1}",
                        thr_names,
                        index=tl_index,
                        key=f"creator-interval-target-label-{li}-{ai}",
                    )
                else:
                    tl = st.text_input(
                        f"Sensation label #{li + 1}.{ai + 1}",
                        value=str(cur_act.get("targetLabel") or ""),
                        key=f"creator-interval-sensation-{li}-{ai}",
                    )
                ascend = int(
                    st.number_input(
                        f"Ascend gain (m) #{li + 1}.{ai + 1} (opt)",
                        min_value=0,
                        value=int(cur_act.get("ascendM") or 0),
                        step=10,
                        key=f"creator-interval-asc-{li}-{ai}",
                    )
                )
                descend = int(
                    st.number_input(
                        f"Descent loss (m) #{li + 1}.{ai + 1} (opt)",
                        min_value=0,
                        value=int(cur_act.get("descendM") or 0),
                        step=10,
                        key=f"creator-interval-desc-{li}-{ai}",
                    )
                )
                built_actions.append(
                    {
                        "kind": kind,
                        "sec": sec,
                        "targetType": tt,
                        "targetLabel": tl,
                        "ascendM": ascend,
                        "descendM": descend,
                    }
                )
            built_loops.append({"repeats": repeats, "actions": built_actions})
        steps = {"warmupSec": warmup, "loops": built_loops, "betweenLoopRecoverSec": between}
    result["stepsJson"] = json.dumps({**steps, "cooldownSec": cooldown})
    step_end_mode = payload.get("stepEndMode") or "auto"
    result["stepEndMode"] = st.selectbox(
        "Interval end mode",
        ["auto", "lap"],
        index=(["auto", "lap"].index(step_end_mode) if step_end_mode in ("auto", "lap") else 0),
        key="creator-interval-end-mode",
    )
    result["plannedDurationSec"] = planner.estimate_interval_duration_sec(json.loads(result["stepsJson"]))
    result["plannedDistanceKm"] = planner.estimate_interval_distance_km(str(athlete_id or ""), json.loads(result["stepsJson"]))
    result["plannedAscentM"] = planner.estimate_interval_ascent_m(json.loads(result["stepsJson"]))
    result["targetType"] = None
    result["targetLabel"] = None
    return result


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
base_type_options = ["FUNDAMENTAL_ENDURANCE", "LONG_RUN", "INTERVAL_SIMPLE"]
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
elif session_type == "INTERVAL_SIMPLE":
    st.caption(
        f"Durée estimée ≈ {_format_duration(form_result.get('plannedDurationSec', 0))} • "
        f"Distance estimée ≈ {fmt_decimal(form_result.get('plannedDistanceKm'), 1)} km • "
        f"Ascent ≈ {fmt_m(form_result.get('plannedAscentM'))}"
    )

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
