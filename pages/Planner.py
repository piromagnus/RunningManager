"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo, PlannedSessionsRepo, ThresholdsRepo
from services.metrics_service import MetricsComputationService
from services.planner_service import PlannerService
from services.session_templates_service import SessionTemplatesService
from services.templates_service import TemplatesService
from ui.interval_editor import render_interval_editor
from utils.coercion import coerce_float, coerce_int
from utils.config import load_config
from utils.formatting import fmt_decimal, fmt_m, format_session_duration, set_locale
from utils.helpers import clean_optional, default_template_title
from utils.styling import apply_theme
from utils.time import iso_week_end, iso_week_start
from utils.ui_helpers import get_dialog_factory
from widgets.week_view import render_week_view

st.set_page_config(page_title="Running Manager - Planner", layout="wide")
apply_theme()
st.title("Planner")

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
sessions_repo = PlannedSessionsRepo(storage)
ath_repo = AthletesRepo(storage)
thr_repo = ThresholdsRepo(storage)
tmpl = TemplatesService(storage)
planner = PlannerService(storage)
session_templates = SessionTemplatesService(storage)
metrics_service = MetricsComputationService(storage)




def _reset_planner_state() -> None:
    planner_state = st.session_state.setdefault(
        "planner_state", {"form": {}, "source": None, "template_context": {}}
    )
    planner_state["form"] = {}
    planner_state["source"] = None
    planner_state["template_context"] = {}
    for key in list(st.session_state.keys()):
        if key.startswith("planner-interval-editor-"):
            st.session_state.pop(key, None)


def _apply_planner_prefill(
    planner_state: Dict[str, Any],
    source: str,
    *,
    date_value: dt.date,
    session_type: str,
    payload: Dict[str, Any],
    force: bool = False,
    template_context: Optional[Dict[str, Any]] = None,
) -> None:
    if not force and planner_state.get("source") == source and planner_state.get("form"):
        return

    planned_distance = float(coerce_float(payload.get("plannedDistanceKm"), 0.0))
    planned_duration = int(coerce_int(payload.get("plannedDurationSec"), 0))
    planned_ascent = int(coerce_int(payload.get("plannedAscentM"), 0))
    target_type = str(payload.get("targetType") or "").lower()
    if target_type not in ("hr", "pace"):
        target_type = "none"

    form = {
        "date": date_value,
        "type": session_type,
        "notes": clean_optional(payload.get("notes")),
        "plannedDistanceKm": planned_distance,
        "plannedDurationSec": planned_duration,
        "plannedAscentM": planned_ascent,
        "targetType": target_type,
        "targetLabel": clean_optional(payload.get("targetLabel"))
        if target_type in ("hr", "pace")
        else None,
        "mode": "distance" if planned_distance > 0 or planned_duration <= 0 else "duration",
        "stepEndMode": payload.get("stepEndMode")
        or ("auto" if session_type == "INTERVAL_SIMPLE" else None),
        "stepsJson": payload.get("stepsJson") or "",
        "templateTitle": clean_optional(payload.get("templateTitle")),
        "raceName": clean_optional(payload.get("raceName")),
    }
    planner_state["form"] = form
    planner_state["source"] = source
    context = dict(template_context or {})
    if "original_title" in context:
        context["original_title"] = clean_optional(context.get("original_title"))
    context.setdefault("prompt_ack", False)
    planner_state["template_context"] = context




def _should_prompt_template_save(form: Dict[str, Any], planner_state: Dict[str, Any]) -> bool:
    context = (planner_state or {}).get("template_context") or {}
    if not context or context.get("prompt_ack"):
        return False
    kind = context.get("kind")
    if kind not in {"template", "edit"}:
        return False
    original = clean_optional(context.get("original_title"))
    current = clean_optional(form.get("templateTitle"))
    if not original or not current:
        return False
    return original != current


def _render_template_prompt_if_needed() -> None:
    prompt_open = st.session_state.get("planner_template_prompt_open")
    prompt_data = st.session_state.get("planner_template_prompt_data")
    if not prompt_open or not prompt_data:
        return

    factory = get_dialog_factory()
    if not factory:
        st.info(
            "La séance a été renommée. Utilisez l'onglet Session Creator pour créer un modèle si besoin.",
            icon="ℹ️",
        )
        st.session_state.pop("planner_template_prompt_open", None)
        st.session_state.pop("planner_template_prompt_data", None)
        return

    title = prompt_data.get("template_title") or "Nouvelle séance"

    @factory("Enregistrer comme modèle ?")
    def _dialog() -> None:
        st.write(
            (
                "Vous avez renommé la séance en **{name}**. Souhaitez-vous enregistrer ce contenu comme modèle ?"
            ).format(name=title)
        )
        actions = st.columns(2)
        with actions[0]:
            if st.button("Créer un modèle", type="primary", key="planner-prompt-create-template"):
                prefill = {
                    "athleteId": prompt_data.get("athlete_id"),
                    "date": prompt_data.get("date"),
                    "templateTitle": prompt_data.get("template_title"),
                    "templateNotes": prompt_data.get("template_notes"),
                    "templateTitleSource": "manual",
                    "sessionPayload": prompt_data.get("session_payload") or {},
                }
                st.session_state["session_creator_prefill"] = prefill
                st.session_state.pop("planner_template_prompt_open", None)
                st.session_state.pop("planner_template_prompt_data", None)
                st.switch_page("pages/SessionCreator.py")
        with actions[1]:
            if st.button("Continuer", key="planner-prompt-continue"):
                st.session_state.pop("planner_template_prompt_open", None)
                st.session_state.pop("planner_template_prompt_data", None)


def build_session_row(
    form: Dict[str, Any],
    athlete_id: str,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    date_value = form.get("date")
    if isinstance(date_value, dt.date):
        date_str = date_value.isoformat()
    else:
        date_str = clean_optional(date_value)
    row = {
        "athleteId": athlete_id,
        "date": date_str,
        "type": form.get("type"),
        "plannedDistanceKm": form.get("plannedDistanceKm"),
        "plannedDurationSec": form.get("plannedDurationSec"),
        "plannedAscentM": form.get("plannedAscentM"),
        "targetType": None
        if form.get("targetType") in (None, "", "none")
        else form.get("targetType"),
        "targetLabel": form.get("targetLabel"),
        "notes": clean_optional(form.get("notes")),
        "templateTitle": clean_optional(form.get("templateTitle")),
        "raceName": clean_optional(form.get("raceName")),
        "stepEndMode": form.get("stepEndMode"),
        "stepsJson": form.get("stepsJson"),
    }
    if overrides:
        row.update(overrides)
    return row


@st.cache_data(ttl=5)
def get_sessions_df_cached(athlete_id: str):
    return sessions_repo.list(athleteId=athlete_id)


@st.cache_data(ttl=60)
def get_threshold_names_cached(athlete_id: str):
    return planner.list_threshold_names(athlete_id)


@st.cache_data(ttl=5)
def get_session_templates_cached(athlete_id: str):
    return session_templates.list(athlete_id=athlete_id)


if st.session_state.pop("planner_templates_refresh", False):
    get_session_templates_cached.clear()

planner_state = st.session_state.setdefault(
    "planner_state", {"form": {}, "source": None, "template_context": {}}
)
planner_state.setdefault("template_context", {})

_render_template_prompt_if_needed()

st.subheader("Select Athlete and Week")
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
athlete_id = athlete_options.get(ath_label) if ath_label else None

today = dt.date.today()
if "planner_week_date" not in st.session_state:
    st.session_state["planner_week_date"] = today
if "planner_week_picker" not in st.session_state:
    st.session_state["planner_week_picker"] = st.session_state["planner_week_date"]


def _update_week_from_picker() -> None:
    st.session_state["planner_week_date"] = st.session_state.get("planner_week_picker", today)


nav_prev, nav_picker, nav_next = st.columns([1, 2, 1])
with nav_prev:
    if st.button("← Semaine précédente", key="planner-prev-week"):
        new_date = st.session_state["planner_week_date"] - dt.timedelta(days=7)
        st.session_state["planner_week_date"] = new_date
        st.session_state["planner_week_picker"] = new_date
        st.rerun()
with nav_picker:
    st.date_input(
        "Week (pick any day)",
        value=st.session_state["planner_week_picker"],
        key="planner_week_picker",
        on_change=_update_week_from_picker,
    )
with nav_next:
    if st.button("Semaine suivante →", key="planner-next-week"):
        new_date = st.session_state["planner_week_date"] + dt.timedelta(days=7)
        st.session_state["planner_week_date"] = new_date
        st.session_state["planner_week_picker"] = new_date
        st.rerun()

selected_week_date = st.session_state["planner_week_date"]
week_start = iso_week_start(selected_week_date)
week_end = iso_week_end(selected_week_date)
st.caption(f"Week range: {week_start.date()} → {week_end.date()}")

st.divider()
st.subheader("Session editor")
edit_ctx = st.session_state.get("planner_edit")
with st.expander("Create/Edit session", expanded=bool(edit_ctx)):
    mode = (edit_ctx or {}).get("mode")
    existing = None
    default_date = week_start.date()
    payload_source = "planner:default"
    base_payload: Dict[str, Any] = {}
    typ = "FUNDAMENTAL_ENDURANCE"
    template_context: Dict[str, Any] = {}

    if mode == "edit":
        sid = (edit_ctx or {}).get("plannedSessionId")
        existing = sessions_repo.get(sid) if sid else None
        if existing:
            try:
                default_date = dt.date.fromisoformat(existing.get("date"))
            except Exception:
                default_date = week_start.date()
            typ = (existing.get("type") or "FUNDAMENTAL_ENDURANCE").upper()
            base_payload = existing
            payload_source = f"edit:{sid}"
            template_context = {
                "kind": "edit",
                "original_title": existing.get("templateTitle"),
            }
    elif mode == "create":
        requested_date = (edit_ctx or {}).get("date")
        if requested_date:
            try:
                default_date = dt.date.fromisoformat(requested_date)
            except Exception:
                default_date = week_start.date()

    if not existing:
        template_records = get_session_templates_cached(athlete_id) if athlete_id else []
        base_options: List[Dict[str, Any]] = [
            {"kind": "type", "value": "FUNDAMENTAL_ENDURANCE", "label": "Endurance fondamentale"},
            {"kind": "type", "value": "LONG_RUN", "label": "Sortie longue"},
            {"kind": "type", "value": "RACE", "label": "Course"},
        ]
        for tpl in template_records:
            label = tpl.get("title") or tpl.get("templateId")
            base_options.append(
                {
                    "kind": "template",
                    "value": tpl.get("templateId"),
                    "label": f"Modèle • {label}",
                    "template": tpl,
                }
            )
        base_choice = st.selectbox(
            "Type ou modèle",
            list(range(len(base_options))),
            index=0,
            format_func=lambda idx: base_options[idx]["label"],
            key="planner_base_selector",
        )
        chosen = base_options[base_choice]
        if hasattr(default_date, "isoformat"):
            source_date_token = default_date.isoformat()
        else:
            source_date_token = str(default_date)

        if chosen["kind"] == "type":
            typ = str(chosen["value"]).upper()
            base_payload = {}
            payload_source = f"create:type:{typ}:{source_date_token}"
            template_context = {}
        else:
            tpl_record = chosen.get("template") or session_templates.get(
                str(chosen.get("value") or "")
            )
            payload = dict((tpl_record or {}).get("payload") or {})
            payload["type"] = (
                payload.get("type") or (tpl_record or {}).get("baseType") or "FUNDAMENTAL_ENDURANCE"
            ).upper()
            if tpl_record:
                payload.setdefault("templateTitle", tpl_record.get("title") or "")
            base_payload = payload
            typ = payload.get("type")
            payload_source = f"create:template:{chosen.get('value')}:{source_date_token}"
            template_context = {
                "kind": "template",
                "template_id": chosen.get("value"),
                "original_title": payload.get("templateTitle") or (tpl_record or {}).get("title"),
            }
        col_template_btn, _ = st.columns([1, 3])
        with col_template_btn:
            if st.button("Créer un modèle", key="planner-create-template"):
                st.session_state["session_creator_prefill"] = {
                    "date": str(default_date),
                    "athleteId": athlete_id,
                }
                st.switch_page("pages/SessionCreator.py")

    _apply_planner_prefill(
        planner_state,
        payload_source,
        date_value=default_date,
        session_type=typ,
        payload=base_payload or {},
        force=mode == "edit",
        template_context=template_context,
    )

    form = planner_state.setdefault("form", {})

    date_value = st.date_input("Date", value=form.get("date", default_date))
    form["date"] = date_value

    if existing:
        type_options = ["FUNDAMENTAL_ENDURANCE", "LONG_RUN", "RACE", "INTERVAL_SIMPLE"]
        if typ not in type_options:
            type_options.append(typ)
        typ = st.selectbox(
            "Type",
            type_options,
            index=type_options.index(form.get("type", typ)),
        )
        form["type"] = typ
    else:
        form["type"] = typ

    title_placeholder = default_template_title({"type": typ, "date": date_value})
    session_title = st.text_input(
        "Titre de la séance",
        value=str(form.get("templateTitle") or ""),
        placeholder=title_placeholder,
    )
    form["templateTitle"] = session_title

    notes = st.text_area("Notes", value=form.get("notes", ""))
    form["notes"] = notes

    planned_distance_km = float(form.get("plannedDistanceKm", 0.0))
    planned_duration_sec = int(form.get("plannedDurationSec", 0))
    planned_ascent_m = int(form.get("plannedAscentM", 0))
    target_type = form.get("targetType", "none")
    target_label = form.get("targetLabel")
    step_end_mode = form.get("stepEndMode")
    steps_json = form.get("stepsJson") or ""

    if typ == "FUNDAMENTAL_ENDURANCE":
        step_end_mode = None
        steps_json = None
        mode_options = ["distance", "duration"]
        current_mode = form.get("mode", "distance")
        if current_mode not in mode_options:
            current_mode = "distance"
        mode_choice = st.radio(
            "Mode de saisie",
            mode_options,
            index=mode_options.index(current_mode),
            format_func=lambda x: "Distance + D+" if x == "distance" else "Durée + D+ mini",
            horizontal=True,
        )
        form["mode"] = mode_choice
        distance_eq_preview = None

        if mode_choice == "distance":
            distance_input = st.number_input(
                "Distance planifiée (km)",
                min_value=0.0,
                value=float(planned_distance_km),
                step=0.1,
            )
            ascent_input = st.number_input(
                "Ascension planifiée (m)",
                min_value=0,
                value=int(planned_ascent_m),
                step=50,
            )
            derived = planner.derive_from_distance(
                str(athlete_id or ""), distance_input, ascent_input
            )
            planned_distance_km = derived["distanceKm"]
            planned_duration_sec = derived["durationSec"]
            planned_ascent_m = int(ascent_input)
            distance_eq_preview = derived["distanceEqKm"]
            st.caption(
                (
                    f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km • "
                    f"Durée estimée ≈ {format_session_duration(planned_duration_sec)}"
                )
            )
        else:
            duration_input = st.number_input(
                "Durée planifiée (sec)",
                min_value=0,
                value=int(planned_duration_sec) if planned_duration_sec else 3600,
                step=300,
            )
            ascent_input = st.number_input(
                "Ascension minimale (m)",
                min_value=0,
                value=int(planned_ascent_m),
                step=50,
            )
            derived = planner.derive_from_duration(
                str(athlete_id or ""), int(duration_input), ascent_input
            )
            planned_distance_km = derived["distanceKm"]
            planned_duration_sec = derived["durationSec"]
            planned_ascent_m = int(ascent_input)
            distance_eq_preview = derived["distanceEqKm"]
            st.caption(
                (
                    f"Distance estimée ≈ {fmt_decimal(planned_distance_km, 1)} km • "
                    f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km"
                )
            )

        target_options = ["none", "hr", "pace"]
        current_target = target_type if target_type in target_options else "none"
        target_type = st.selectbox(
            "Cible",
            target_options,
            index=target_options.index(current_target),
        )
        if target_type == "none":
            target_label = None
        else:
            target_label = "Fundamental"
            st.caption("Seuil fixé automatiquement sur Fundamental.")
        form["targetType"] = target_type
        form["targetLabel"] = target_label

    elif typ == "LONG_RUN":
        step_end_mode = None
        steps_json = None
        mode_options = ["distance", "duration"]
        current_mode = form.get("mode", "distance")
        if current_mode not in mode_options:
            current_mode = "distance"
        mode_choice = st.radio(
            "Mode de saisie",
            mode_options,
            index=mode_options.index(current_mode),
            format_func=lambda x: "Distance + D+" if x == "distance" else "Durée + D+ mini",
            horizontal=True,
        )
        form["mode"] = mode_choice
        distance_eq_preview = None

        if mode_choice == "distance":
            distance_input = st.number_input(
                "Distance planifiée (km)",
                min_value=0.0,
                value=float(planned_distance_km),
                step=0.5,
            )
            ascent_input = st.number_input(
                "Ascension planifiée (m)",
                min_value=0,
                value=int(planned_ascent_m) if planned_ascent_m else 500,
                step=50,
            )
            derived = planner.derive_from_distance(
                str(athlete_id or ""), distance_input, ascent_input
            )
            planned_distance_km = derived["distanceKm"]
            planned_duration_sec = derived["durationSec"]
            planned_ascent_m = int(ascent_input)
            distance_eq_preview = derived["distanceEqKm"]
            st.caption(
                (
                    f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km • "
                    f"Durée estimée ≈ {format_session_duration(planned_duration_sec)}"
                )
            )
        else:
            duration_input = st.number_input(
                "Durée planifiée (sec)",
                min_value=0,
                value=int(planned_duration_sec) if planned_duration_sec else 7200,
                step=300,
            )
            ascent_input = st.number_input(
                "Ascension minimale (m)",
                min_value=0,
                value=int(planned_ascent_m) if planned_ascent_m else 500,
                step=50,
            )
            derived = planner.derive_from_duration(
                str(athlete_id or ""), int(duration_input), ascent_input
            )
            planned_distance_km = derived["distanceKm"]
            planned_duration_sec = derived["durationSec"]
            planned_ascent_m = int(ascent_input)
            distance_eq_preview = derived["distanceEqKm"]
            st.caption(
                (
                    f"Distance estimée ≈ {fmt_decimal(planned_distance_km, 1)} km • "
                    f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km"
                )
            )

        target_type = st.selectbox(
            "Cible",
            ["none", "hr", "pace"],
            index=(
                0
                if target_type not in ["none", "hr", "pace"]
                else ["none", "hr", "pace"].index(target_type)
            ),
        )
        if target_type in ("hr", "pace"):
            names = (
                get_threshold_names_cached(athlete_id)
                if athlete_id
                else ["Fundamental", "Threshold 30", "Threshold 60"]
            )
            idx = names.index(target_label) if target_label in names else 0
            target_label = st.selectbox("Seuil", names, index=idx)
        else:
            target_label = None
        form["targetType"] = target_type
        form["targetLabel"] = target_label

    elif typ == "RACE":
        form["mode"] = None
        step_end_mode = None
        steps_json = None
        distance_input = st.number_input(
            "Distance (km)",
            min_value=0.0,
            value=float(planned_distance_km),
            step=0.1,
        )
        ascent_input = st.number_input(
            "Ascension (m)",
            min_value=0,
            value=int(planned_ascent_m),
            step=50,
        )
        race_name_input = st.text_input(
            "Nom de la course",
            value=form.get("raceName", ""),
        )
        target_time = st.number_input(
            "Temps cible (sec)",
            min_value=0,
            value=int(planned_duration_sec),
            step=60,
        )
        planned_distance_km = float(distance_input)
        planned_ascent_m = int(ascent_input)
        planned_duration_sec = int(target_time)
        form["raceName"] = race_name_input
        distance_eq_preview = planner.compute_distance_eq_km(planned_distance_km, planned_ascent_m)
        st.caption(
            f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km • "
            f"Temps cible ≈ {format_session_duration(planned_duration_sec)}"
        )
        target_type = "race"
        target_label = None
        form["targetType"] = target_type
        form["targetLabel"] = target_label

    elif typ == "INTERVAL_SIMPLE":
        form["targetType"] = None
        form["targetLabel"] = None
        thr_names = (
            get_threshold_names_cached(athlete_id)
            if athlete_id
            else [
                "Threshold 60",
                "Threshold 30",
                "Fundamental",
                "MVA",
                "Max speed",
            ]
        )
        serialized_steps = render_interval_editor("planner", steps_json, thr_names)
        step_end_mode = st.selectbox(
            "Mode de fin",
            ["auto", "lap"],
            index=(["auto", "lap"].index(form.get("stepEndMode") or "auto")),
        )
        planned_duration_sec = planner.estimate_interval_duration_sec(serialized_steps)
        planned_distance_km = planner.estimate_interval_distance_km(
            str(athlete_id or ""), serialized_steps
        )
        planned_ascent_m = planner.estimate_interval_ascent_m(serialized_steps)
        distance_eq_preview = planner.compute_distance_eq_km(planned_distance_km, planned_ascent_m)
        st.caption(
            f"Durée ≈ {format_session_duration(planned_duration_sec)} • "
            f"Distance ≈ {fmt_decimal(planned_distance_km, 1)} km • "
            f"D+ ≈ {fmt_m(planned_ascent_m)} • "
            f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km"
        )
        steps_json = json.dumps(serialized_steps, ensure_ascii=False, separators=(",", ":"))
    else:
        form["mode"] = None

    form["plannedDistanceKm"] = planned_distance_km
    form["plannedDurationSec"] = planned_duration_sec
    form["plannedAscentM"] = planned_ascent_m
    form["targetType"] = target_type
    form["targetLabel"] = target_label
    form["stepEndMode"] = step_end_mode
    form["stepsJson"] = steps_json
    if typ != "RACE":
        form["raceName"] = ""

    col_save, col_cancel, col_delete = st.columns(3)
    with col_save:
        if st.button("💾 Save", help="Save session"):
            if not athlete_id:
                st.error("Please select an athlete")
            else:
                should_prompt = _should_prompt_template_save(form, planner_state)
                row = build_session_row(form, athlete_id)
                session_payload = {
                    "type": row.get("type"),
                    "plannedDistanceKm": row.get("plannedDistanceKm"),
                    "plannedDurationSec": row.get("plannedDurationSec"),
                    "plannedAscentM": row.get("plannedAscentM"),
                    "targetType": row.get("targetType"),
                    "targetLabel": row.get("targetLabel"),
                    "notes": row.get("notes"),
                    "raceName": row.get("raceName"),
                    "stepEndMode": row.get("stepEndMode"),
                    "stepsJson": row.get("stepsJson"),
                }
                sid = existing["plannedSessionId"] if existing else None
                if existing:
                    sessions_repo.update(existing["plannedSessionId"], row)
                    st.success("Session updated")
                else:
                    sid = sessions_repo.create(row)
                    st.success(f"Session added: {sid}")
                metrics_service.recompute_planned_for_athlete(str(athlete_id))
                if should_prompt:
                    planner_state.get("template_context", {})["prompt_ack"] = True
                    st.session_state["planner_template_prompt_data"] = {
                        "athlete_id": athlete_id,
                        "date": row.get("date"),
                        "template_title": clean_optional(form.get("templateTitle")),
                        "template_notes": clean_optional(form.get("notes")),
                        "session_payload": session_payload,
                        "planned_session_id": str(sid) if sid else None,
                    }
                    st.session_state["planner_template_prompt_open"] = True
                get_sessions_df_cached.clear()
                st.session_state["planner_edit"] = None
                _reset_planner_state()
                st.rerun()
    with col_cancel:
        if st.button("✖️ Cancel", help="Cancel editing"):
            st.session_state["planner_edit"] = None
            _reset_planner_state()
            st.rerun()
    with col_delete:
        if existing and st.button("🗑️ Delete", help="Delete session"):
            sessions_repo.delete(existing["plannedSessionId"])
            metrics_service.recompute_planned_for_athlete(str(athlete_id))
            get_sessions_df_cached.clear()
            st.session_state["planner_edit"] = None
            _reset_planner_state()
            st.rerun()


st.divider()
week_records, df_in_week = render_week_view(
    athlete_id=athlete_id,
    week_start=week_start,
    week_end=week_end,
    sessions_repo=sessions_repo,
    planner=planner,
    session_templates=session_templates,
    get_sessions_df_cached=get_sessions_df_cached,
    get_session_templates_cached=get_session_templates_cached,
)

st.divider()
st.subheader("Week templates")
if athlete_id:
    col1, col2 = st.columns(2)
    with col1:
        tmpl_name = st.text_input("Template name", value=f"Week {week_start.date()}")
        if st.button("Save current week as template"):
            if not week_records:
                st.warning("No sessions in the selected week to save.")
            else:
                tid = tmpl.save_week_template(
                    athlete_id, week_records, week_start.date(), tmpl_name
                )
                st.success(f"Saved template: {tid}")
                st.rerun()
    with col2:
        templates = tmpl.list(athlete_id=athlete_id)
        options = {
            f"{t.get('name')} ({t.get('templateId')})": t.get("templateId") for t in templates
        }
        if options:
            sel = st.selectbox("Available templates", list(options.keys()))
            clear_before_apply = st.checkbox(
                "Clear current week before applying",
                value=False,
                key="planner-clear-week",
            )
            if sel and st.button("Apply to this week"):
                if (
                    clear_before_apply
                    and df_in_week is not None
                    and hasattr(df_in_week, "empty")
                    and not df_in_week.empty
                ):
                    for _, row in df_in_week.iterrows():
                        sessions_repo.delete(str(row.get("plannedSessionId")))
                tmpl.apply_week_template(athlete_id, options[sel], week_start.date(), sessions_repo)
                metrics_service.recompute_planned_for_athlete(str(athlete_id))
                get_sessions_df_cached.clear()
                st.success("Template applied")
                st.rerun()
        else:
            st.caption("No templates saved yet.")
