from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
import math

import streamlit as st
from typing import Any, Dict, List, Optional

from utils.config import load_config
from utils.formatting import set_locale, fmt_decimal, fmt_m
from utils.styling import apply_theme
from utils.time import iso_week_start, iso_week_end
from persistence.csv_storage import CsvStorage
from persistence.repositories import PlannedSessionsRepo, AthletesRepo, ThresholdsRepo
from services.templates_service import TemplatesService
from services.planner_service import PlannerService
from services.planner_presenter import build_card_view_model, build_empty_state_placeholder
from services.session_templates_service import SessionTemplatesService
from ui.interval_editor import render_interval_editor


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


def _format_week_duration(seconds: int) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    return f"{hours}h{minutes:02d}"


def _format_session_duration(seconds: int) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h{minutes:02d}"
    return f"{minutes} min"


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


def _default_template_title(session: Dict[str, Any]) -> str:
    raw_type = str(session.get("type") or "Session").replace("_", " ")
    title_part = raw_type.title()
    date_part = str(session.get("date") or "")
    return f"{title_part} {date_part}".strip()


def _clean_optional(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _reset_planner_state() -> None:
    planner_state = st.session_state.setdefault("planner_state", {"form": {}, "source": None})
    planner_state["form"] = {}
    planner_state["source"] = None
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
) -> None:
    if not force and planner_state.get("source") == source and planner_state.get("form"):
        return

    planned_distance = float(_coerce_float(payload.get("plannedDistanceKm"), 0.0))
    planned_duration = int(_coerce_int(payload.get("plannedDurationSec"), 0))
    planned_ascent = int(_coerce_int(payload.get("plannedAscentM"), 0))
    target_type = str(payload.get("targetType") or "").lower()
    if target_type not in ("hr", "pace"):
        target_type = "none"

    form = {
        "date": date_value,
        "type": session_type,
        "notes": _clean_optional(payload.get("notes")),
        "plannedDistanceKm": planned_distance,
        "plannedDurationSec": planned_duration,
        "plannedAscentM": planned_ascent,
        "targetType": target_type,
        "targetLabel": _clean_optional(payload.get("targetLabel")) if target_type in ("hr", "pace") else None,
        "mode": "distance" if planned_distance > 0 or planned_duration <= 0 else "duration",
        "stepEndMode": payload.get("stepEndMode") or ("auto" if session_type == "INTERVAL_SIMPLE" else None),
        "stepsJson": payload.get("stepsJson") or "",
    }
    planner_state["form"] = form
    planner_state["source"] = source


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
        date_str = _clean_optional(date_value)
    row = {
        "athleteId": athlete_id,
        "date": date_str,
        "type": form.get("type"),
        "plannedDistanceKm": form.get("plannedDistanceKm"),
        "plannedDurationSec": form.get("plannedDurationSec"),
        "plannedAscentM": form.get("plannedAscentM"),
        "targetType": None if form.get("targetType") in (None, "", "none") else form.get("targetType"),
        "targetLabel": form.get("targetLabel"),
        "notes": _clean_optional(form.get("notes")),
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

planner_state = st.session_state.setdefault("planner_state", {"form": {}, "source": None})

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
    .rm-card-header { font-size: 0.95rem; font-weight: 600; margin-bottom: 4px; }
    .rm-card-meta { font-size: 0.8rem; color: rgba(255,255,255,0.7); margin-bottom: 4px; }
    .rm-card-section { background: rgba(255,255,255,0.04); border-radius: 6px; padding: 6px 8px; margin-top: 6px; }
    .rm-card-section-title { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; margin-bottom: 2px; }
    .rm-card-section-body { font-size: 0.8rem; line-height: 1.3; }
    .rm-card-actions { display: flex; gap: 6px; margin-top: 6px; }
    .rm-loop-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 8px 10px; margin-bottom: 10px; }
    .rm-interval-action { background: rgba(255,255,255,0.02); border-radius: 6px; padding: 6px 8px; margin-bottom: 6px; }
    .rm-interval-editor .stNumberInput label,
    .rm-interval-editor .stSelectbox label,
    .rm-interval-editor .stTextInput label { font-size: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
        else:
            tpl_record = chosen.get("template") or session_templates.get(str(chosen.get("value") or ""))
            payload = dict((tpl_record or {}).get("payload") or {})
            payload["type"] = (payload.get("type") or (tpl_record or {}).get("baseType") or "FUNDAMENTAL_ENDURANCE").upper()
            base_payload = payload
            typ = payload.get("type")
            payload_source = f"create:template:{chosen.get('value')}:{source_date_token}"
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
            derived = planner.derive_from_distance(str(athlete_id or ""), distance_input, ascent_input)
            planned_distance_km = derived["distanceKm"]
            planned_duration_sec = derived["durationSec"]
            planned_ascent_m = int(ascent_input)
            distance_eq_preview = derived["distanceEqKm"]
            st.caption(
                f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km • Durée estimée ≈ {_format_session_duration(planned_duration_sec)}"
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
            derived = planner.derive_from_duration(str(athlete_id or ""), int(duration_input), ascent_input)
            planned_distance_km = derived["distanceKm"]
            planned_duration_sec = derived["durationSec"]
            planned_ascent_m = int(ascent_input)
            distance_eq_preview = derived["distanceEqKm"]
            st.caption(
                f"Distance estimée ≈ {fmt_decimal(planned_distance_km, 1)} km • Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km"
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
            derived = planner.derive_from_distance(str(athlete_id or ""), distance_input, ascent_input)
            planned_distance_km = derived["distanceKm"]
            planned_duration_sec = derived["durationSec"]
            planned_ascent_m = int(ascent_input)
            distance_eq_preview = derived["distanceEqKm"]
            st.caption(
                f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km • Durée estimée ≈ {_format_session_duration(planned_duration_sec)}"
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
            derived = planner.derive_from_duration(str(athlete_id or ""), int(duration_input), ascent_input)
            planned_distance_km = derived["distanceKm"]
            planned_duration_sec = derived["durationSec"]
            planned_ascent_m = int(ascent_input)
            distance_eq_preview = derived["distanceEqKm"]
            st.caption(
                f"Distance estimée ≈ {fmt_decimal(planned_distance_km, 1)} km • Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km"
            )

        target_type = st.selectbox(
            "Cible",
            ["none", "hr", "pace"],
            index=(0 if target_type not in ["none", "hr", "pace"] else ["none", "hr", "pace"].index(target_type)),
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
        target_time = st.number_input(
            "Temps cible (sec)",
            min_value=0,
            value=int(planned_duration_sec),
            step=60,
        )
        planned_distance_km = float(distance_input)
        planned_ascent_m = int(ascent_input)
        planned_duration_sec = int(target_time)
        distance_eq_preview = planner.compute_distance_eq_km(planned_distance_km, planned_ascent_m)
        st.caption(
            f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km • "
            f"Temps cible ≈ {_format_session_duration(planned_duration_sec)}"
        )
        target_type = "race"
        target_label = None
        form["targetType"] = target_type
        form["targetLabel"] = target_label

    elif typ == "INTERVAL_SIMPLE":
        form["targetType"] = None
        form["targetLabel"] = None
        thr_names = get_threshold_names_cached(athlete_id) if athlete_id else [
            "Threshold 60",
            "Threshold 30",
            "Fundamental",
            "MVA",
            "Max speed",
        ]
        serialized_steps = render_interval_editor("planner", steps_json, thr_names)
        step_end_mode = st.selectbox(
            "Mode de fin",
            ["auto", "lap"],
            index=(["auto", "lap"].index(form.get("stepEndMode") or "auto")),
        )
        planned_duration_sec = planner.estimate_interval_duration_sec(serialized_steps)
        planned_distance_km = planner.estimate_interval_distance_km(str(athlete_id or ""), serialized_steps)
        planned_ascent_m = planner.estimate_interval_ascent_m(serialized_steps)
        distance_eq_preview = planner.compute_distance_eq_km(planned_distance_km, planned_ascent_m)
        st.caption(
            f"Durée ≈ {_format_session_duration(planned_duration_sec)} • "
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

    col_save, col_cancel, col_delete = st.columns(3)
    with col_save:
        if st.button("💾 Save", help="Save session"):
            if not athlete_id:
                st.error("Please select an athlete")
            else:
                row = build_session_row(form, athlete_id)
                if existing:
                    sessions_repo.update(existing["plannedSessionId"], row)
                    st.success("Session updated")
                else:
                    sid = sessions_repo.create(row)
                    st.success(f"Session added: {sid}")
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
            get_sessions_df_cached.clear()
            st.session_state["planner_edit"] = None
            _reset_planner_state()
            st.rerun()


st.divider()
st.subheader("Week view")
week_records: List[Dict[str, Any]] = []
df_in_week = None

if athlete_id:
    df = sessions_repo.list(athleteId=athlete_id)
    df_in_week = (
        df[(df["date"] >= str(week_start.date())) & (df["date"] <= str(week_end.date()))]
        if not df.empty
        else df
    )
    week_records = df_in_week.to_dict(orient="records") if not df_in_week.empty else []
    records_by_day: Dict[str, List[Dict[str, Any]]] = {}
    for rec in week_records:
        date_key = str(rec.get("date"))
        records_by_day.setdefault(date_key, []).append(rec)

    cols = st.columns([1, 1, 1, 1, 1, 1, 1])
    for i in range(7):
        day = week_start.date() + dt.timedelta(days=i)
        day_key = str(day)
        with cols[i]:
            st.markdown(f"**{day.strftime('%a')}**\n\n{day}")
            day_items = records_by_day.get(day_key, [])
            if day_items:
                for record in day_items:
                    sid = record.get("plannedSessionId")
                    session_type = record.get("type")
                    duration = record.get("plannedDurationSec")
                    if isinstance(duration, float) and math.isnan(duration):
                        duration = 0
                    try:
                        duration = int(float(duration)) if duration not in (None, "") else 0
                    except Exception:
                        duration = 0
                    distance = record.get("plannedDistanceKm")
                    if isinstance(distance, float) and math.isnan(distance):
                        distance = None
                    ascent = record.get("plannedAscentM")
                    if isinstance(ascent, float) and math.isnan(ascent):
                        ascent = None
                    target_type = record.get("targetType") if isinstance(record.get("targetType"), str) else None
                    target_label = record.get("targetLabel") if isinstance(record.get("targetLabel"), str) else None
                    step_mode = record.get("stepEndMode") if isinstance(record.get("stepEndMode"), str) else None

                    estimated_distance = planner.estimate_session_distance_km(athlete_id, record)
                    distance_eq = planner.compute_session_distance_eq(athlete_id, record)

                    model = build_card_view_model(
                        {
                            "plannedSessionId": sid,
                            "type": session_type,
                            "plannedDurationSec": duration,
                            "plannedDistanceKm": distance,
                            "plannedAscentM": ascent,
                            "targetType": target_type,
                            "targetLabel": target_label,
                            "stepEndMode": step_mode,
                            "stepsJson": record.get("stepsJson"),
                        },
                        estimated_distance_km=estimated_distance,
                        distance_eq_km=distance_eq,
                    )
                    st.markdown("<div class='rm-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rm-card-header'>{model['header']}</div>", unsafe_allow_html=True)
                    meta = model.get("meta") or []
                    if meta:
                        st.markdown(
                            f"<div class='rm-card-meta'>{' • '.join(meta)}</div>",
                            unsafe_allow_html=True,
                        )
                    for section in model.get("sections", []):
                        lines_html = "<br/>".join(section.lines)
                        st.markdown(
                            "<div class='rm-card-section'>"
                            f"<div class='rm-card-section-title'>{section.title}</div>"
                            f"<div class='rm-card-section-body'>{lines_html}</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    action_cols = st.columns([1] * len(model["actions"])) if model["actions"] else []
                    for col, action in zip(action_cols, model["actions"]):
                        key = f"{action.action}-{action.session_id}"
                        with col:
                            if st.button(action.icon, key=key, help=action.label):
                                if action.action == "edit":
                                    st.session_state["planner_edit"] = {"mode": "edit", "plannedSessionId": sid}
                                    st.rerun()
                                elif action.action == "save-template":
                                    session_for_template = dict(record)
                                    if not session_for_template.get("athleteId"):
                                        session_for_template["athleteId"] = athlete_id
                                    try:
                                        title_hint = _default_template_title(session_for_template)
                                        template_id = session_templates.create_from_session(
                                            session_for_template,
                                            title=title_hint,
                                        )
                                        get_session_templates_cached.clear()
                                        st.session_state["planner_flash"] = (
                                            "success",
                                            f"Modèle enregistré ({title_hint}) – id {template_id}",
                                        )
                                    except Exception as exc:
                                        st.session_state["planner_flash"] = (
                                            "error",
                                            f"Impossible de créer le modèle: {exc}",
                                        )
                                    st.rerun()
                                elif action.action == "delete":
                                    sessions_repo.delete(str(sid))
                                    get_sessions_df_cached.clear()
                                    st.rerun()
                                elif action.action == "view":
                                    st.session_state["session_view_sid"] = str(sid)
                                    st.switch_page("pages/Session.py")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                placeholder = build_empty_state_placeholder(day.strftime("%A"))
                st.caption(f"{placeholder['message']} {placeholder['cta']}")
            if st.button("＋", key=f"add-{day}", help="Add session"):
                st.session_state["planner_edit"] = {"mode": "create", "date": str(day)}
                st.session_state.pop("planner_base_selector", None)
                st.rerun()

    totals = planner.compute_weekly_totals(athlete_id, week_records)
    totals_caption = (
        "Totals — "
        f"{_format_week_duration(totals['timeSec'])} • "
        f"{fmt_decimal(totals['distanceKm'], 1)} km • "
        f"DEQ {fmt_decimal(totals.get('distanceEqKm', 0.0), 1)} km • "
        f"{fmt_m(totals['ascentM'])}"
    )
    st.caption(totals_caption)
    flash = st.session_state.pop("planner_flash", None)
    if flash:
        level, message = flash
        if level == "success":
            st.success(message)
        elif level == "error":
            st.error(message)
else:
    st.info("Select an athlete to view sessions.")

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
                tid = tmpl.save_week_template(athlete_id, week_records, week_start.date(), tmpl_name)
                st.success(f"Saved template: {tid}")
                st.rerun()
    with col2:
        templates = tmpl.list(athlete_id=athlete_id)
        options = {
            f"{t.get('name')} ({t.get('templateId')})": t.get("templateId")
            for t in templates
        }
        if options:
            sel = st.selectbox("Available templates", list(options.keys()))
            clear_before_apply = st.checkbox(
                "Clear current week before applying",
                value=False,
                key="planner-clear-week",
            )
            if sel and st.button("Apply to this week"):
                if clear_before_apply and df_in_week is not None and hasattr(df_in_week, "empty") and not df_in_week.empty:
                    for _, row in df_in_week.iterrows():
                        sessions_repo.delete(str(row.get("plannedSessionId")))
                tmpl.apply_week_template(athlete_id, options[sel], week_start.date(), sessions_repo)
                get_sessions_df_cached.clear()
                st.success("Template applied")
                st.rerun()
        else:
            st.caption("No templates saved yet.")
