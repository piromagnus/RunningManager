from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
import math

import streamlit as st
from typing import Any, Dict, List

from utils.config import load_config
from utils.formatting import set_locale, fmt_decimal, fmt_m
from utils.time import iso_week_start, iso_week_end
from persistence.csv_storage import CsvStorage
from persistence.repositories import PlannedSessionsRepo, AthletesRepo, ThresholdsRepo
from services.templates_service import TemplatesService
from services.planner_service import PlannerService
from services.planner_presenter import build_card_view_model, build_empty_state_placeholder


st.set_page_config(page_title="Running Manager - Planner", layout="wide")
st.title("Planner")

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
sessions_repo = PlannedSessionsRepo(storage)
ath_repo = AthletesRepo(storage)
thr_repo = ThresholdsRepo(storage)
tmpl = TemplatesService(storage)
planner = PlannerService(storage)


def _format_week_duration(seconds: int) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    return f"{hours}h{minutes:02d}"

# Caching helpers
@st.cache_data(ttl=5)
def get_sessions_df_cached(athlete_id: str):
    return sessions_repo.list(athleteId=athlete_id)

@st.cache_data(ttl=60)
def get_threshold_names_cached(athlete_id: str):
    return planner.list_threshold_names(athlete_id)

# Simple CSS widening and card styling
st.markdown(
    """
    <style>
    .rm-card {
      padding: 8px 10px;
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 6px;
      margin-bottom: 8px;
    }
    .rm-top { font-size: 0.9rem; line-height: 1.2; }
    .rm-actions { display: flex; gap: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


st.subheader("Select Athlete and Week")
ath_df = ath_repo.list()
athlete_options = (
    {
        f"{r.get('name') or 'Unnamed'} ({r.get('athleteId')})": r.get('athleteId')
        for _, r in ath_df.iterrows()
    }
    if not ath_df.empty
    else {}
)
ath_label = st.selectbox("Athlete", list(athlete_options.keys())) if athlete_options else None
athlete_id = athlete_options.get(ath_label) if ath_label else None

today = dt.date.today()
week_monday = st.date_input("Week (pick any day)", value=today)
week_start = iso_week_start(week_monday)
week_end = iso_week_end(week_monday)
st.caption(f"Week range: {week_start.date()} → {week_end.date()}")

st.divider()
st.subheader("Session editor")
edit_ctx = st.session_state.get("planner_edit")
with st.expander("Create/Edit session", expanded=bool(edit_ctx)):
    mode = (edit_ctx or {}).get("mode")
    existing = None
    default_date = week_start.date()
    if mode == "edit":
        sid = (edit_ctx or {}).get("plannedSessionId")
        existing = sessions_repo.get(sid) if sid else None
        if existing:
            default_date = dt.date.fromisoformat(existing.get("date"))
    elif mode == "create":
        d = (edit_ctx or {}).get("date")
        if d:
            try:
                default_date = dt.date.fromisoformat(d)
            except Exception:
                default_date = week_start.date()

    date = st.date_input("Date", value=default_date)
    if existing:
        typ = existing.get("type")
    else:
        typ = st.selectbox("Type", ["FUNDAMENTAL_ENDURANCE", "LONG_RUN", "INTERVAL_SIMPLE"], index=0)

    notes = st.text_area("Notes", value=(existing.get("notes") if existing else ""))

    target_type = existing.get("targetType") if existing else None
    target_label = existing.get("targetLabel") if existing else None
    planned_distance_km = existing.get("plannedDistanceKm") if existing else None
    planned_duration_sec = existing.get("plannedDurationSec") if existing else None
    planned_ascent_m = existing.get("plannedAscentM") if existing else None
    step_end_mode = existing.get("stepEndMode") if existing else None
    steps_json = existing.get("stepsJson") if existing else None

    if typ == "FUNDAMENTAL_ENDURANCE":
        planned_duration_sec = int(st.number_input("Duration (sec)", min_value=0, value=int(planned_duration_sec or 3600), step=300))
        target_type = st.selectbox(
            "Target type",
            ["none", "hr", "pace"],
            index=(0 if not target_type else ["none", "hr", "pace"].index(target_type)),
        )
        if target_type in ("hr", "pace"):
            names = get_threshold_names_cached(athlete_id) if athlete_id else ["Fundamental", "Threshold 30", "Threshold 60"]
            idx = names.index(target_label) if target_label in names else 0
            target_label = st.selectbox("Threshold", names, index=idx)
        planned_ascent_m = int(st.number_input("Planned ascent (m)", min_value=0, value=int(planned_ascent_m or 0), step=50))
        est_km = planner.estimate_km(athlete_id, planned_duration_sec) if athlete_id else None
        if est_km is not None:
            st.caption(f"Estimated km ≈ {est_km:.1f}")
            # Persist estimated distance for Fundamental Endurance
            planned_distance_km = round(est_km, 1)

    elif typ == "LONG_RUN":
        planned_distance_km = float(st.number_input("Distance (km)", min_value=0.0, value=float(planned_distance_km or 20.0), step=1.0))
        planned_duration_sec = int(st.number_input("Time (sec, optional)", min_value=0, value=int(planned_duration_sec or 0), step=300)) or None
        planned_ascent_m = int(st.number_input("Min ascent (m)", min_value=0, value=int(planned_ascent_m or 500), step=50))
        target_type = st.selectbox(
            "Target type",
            ["none", "hr", "pace"],
            index=(0 if not target_type else ["none", "hr", "pace"].index(target_type)),
        )
        if target_type in ("hr", "pace"):
            names = get_threshold_names_cached(athlete_id) if athlete_id else ["Fundamental", "Threshold 30", "Threshold 60"]
            idx = names.index(target_label) if target_label in names else 0
            target_label = st.selectbox("Threshold", names, index=idx)

    elif typ == "INTERVAL_SIMPLE":
        warmup = 600
        cooldown = 600
        mode = st.radio("Interval build mode", ["Simple repeats", "Loops"], index=0, horizontal=True)
        thr_names = get_threshold_names_cached(athlete_id) if athlete_id else ["Threshold 60", "Threshold 30", "Fundamental", "MVA", "Max speed"]
        parsed = {}
        if steps_json:
            try:
                parsed = json.loads(steps_json)
            except Exception:
                parsed = {}
        warmup = int(st.number_input("Warm-up (sec)", min_value=0, value=int(parsed.get("warmupSec", warmup)), step=60))
        if mode == "Simple repeats":
            repeats = parsed.get("repeats", [])
            num_rep = st.number_input("Repeats", min_value=1, value=max(1, len(repeats) or 6), step=1)
            rows = []
            for idx in range(int(num_rep)):
                st.markdown(f"Repeat {idx+1}")
                cur = repeats[idx] if idx < len(repeats) else {}
                w = int(st.number_input(f"Work (sec) #{idx+1}", min_value=10, value=int(cur.get("workSec", 60)), step=10))
                r = int(st.number_input(f"Recovery (sec) #{idx+1}", min_value=10, value=int(cur.get("recoverSec", 60)), step=10))
                tt = st.selectbox(
                    f"Target type #{idx+1}",
                    ["hr", "pace", "sensation"],
                    index=( ["hr", "pace", "sensation"].index(cur.get("targetType", "hr")) ),
                )
                if tt in ("hr", "pace"):
                    tl_index = (
                        thr_names.index(cur.get("targetLabel", thr_names[0]))
                        if cur.get("targetLabel") in thr_names
                        else 0
                    )
                    tl = st.selectbox(
                        f"Target label #{idx+1}",
                        thr_names,
                        index=tl_index,
                    )
                else:
                    tl = st.text_input(f"Sensation label #{idx+1}", value=str(cur.get("targetLabel") or ""))
                rows.append({"workSec": w, "recoverSec": r, "targetType": tt, "targetLabel": tl})
            steps = {"warmupSec": warmup, "repeats": rows}
        else:
            # Loops mode
            loops = parsed.get("loops", [])
            loop_count = st.number_input("Number of loops", min_value=1, value=max(1, len(loops) or 1))
            between = int(st.number_input("Between-loop recovery (sec)", min_value=0, value=int(parsed.get("betweenLoopRecoverSec") or 0), step=10))
            built_loops = []
            for li in range(int(loop_count)):
                st.markdown(f"Loop {li+1}")
                cur_loop = loops[li] if li < len(loops) else {}
                actions = cur_loop.get("actions", [])
                repeats = int(
                    st.number_input(
                        f"Loop repeats #{li+1}",
                        min_value=1,
                        value=int(cur_loop.get("repeats") or 1),
                        key=f"lrep-{li}",
                    )
                )
                act_count = st.number_input(
                    f"Actions in loop #{li+1}",
                    min_value=1,
                    value=max(1, len(actions) or 2),
                    key=f"lacts-{li}",
                )
                built_actions = []
                for ai in range(int(act_count)):
                    st.markdown(f"- Action {ai+1}")
                    cur_act = actions[ai] if ai < len(actions) else {}
                    kind = st.selectbox(f"Kind #{li+1}.{ai+1}", ["run", "recovery"], index=(0 if (cur_act.get("kind") or "run") == "run" else 1), key=f"k-{li}-{ai}")
                    sec = int(st.number_input(f"Seconds #{li+1}.{ai+1}", min_value=5, value=int(cur_act.get("sec") or 60), step=5, key=f"s-{li}-{ai}"))
                    tt = st.selectbox(
                        f"Target type #{li+1}.{ai+1}",
                        ["hr", "pace", "sensation"],
                        index=( ["hr", "pace", "sensation"].index(cur_act.get("targetType") or "hr") ),
                        key=f"tt-{li}-{ai}",
                    )
                    if tt in ("hr", "pace"):
                        tl_index = (
                            thr_names.index(cur_act.get("targetLabel", thr_names[0]))
                            if cur_act.get("targetLabel") in thr_names
                            else 0
                        )
                        tl = st.selectbox(
                            f"Target label #{li+1}.{ai+1}",
                            thr_names,
                            index=tl_index,
                            key=f"tl-{li}-{ai}",
                        )
                    else:
                        tl = st.text_input(f"Sensation label #{li+1}.{ai+1}", value=str(cur_act.get("targetLabel") or ""), key=f"tls-{li}-{ai}")
                    ascend = int(
                        st.number_input(
                            f"Ascend gain (m) #{li+1}.{ai+1} (opt)",
                            min_value=0,
                            value=int(cur_act.get("ascendM") or 0),
                            step=10,
                            key=f"asc-{li}-{ai}",
                        )
                    )
                    descend = int(
                        st.number_input(
                            f"Descent loss (m) #{li+1}.{ai+1} (opt)",
                            min_value=0,
                            value=int(cur_act.get("descendM") or 0),
                            step=10,
                            key=f"des-{li}-{ai}",
                        )
                    )
                    built_actions.append({"kind": kind, "sec": sec, "targetType": tt, "targetLabel": tl, "ascendM": ascend, "descendM": descend})
                built_loops.append({"repeats": repeats, "actions": built_actions})
            steps = {"warmupSec": warmup, "loops": built_loops, "betweenLoopRecoverSec": between}
        cooldown = int(st.number_input("Cool-down (sec)", min_value=0, value=int(parsed.get("cooldownSec", cooldown)), step=60))
        step_end_mode = st.selectbox("Interval end mode", ["auto", "lap"], index=( ["auto","lap"].index(step_end_mode or "auto") ))
        steps["cooldownSec"] = cooldown
        steps_json = json.dumps(steps)
        planned_duration_sec = planner.estimate_interval_duration_sec(steps)
        planned_distance_km = planner.estimate_interval_distance_km(athlete_id, steps) if athlete_id else None

    col_save, col_cancel, col_delete = st.columns(3)
    with col_save:
        if st.button("💾 Save", help="Save session"):
            if not athlete_id:
                st.error("Please select an athlete")
            else:
                row = {
                    "athleteId": athlete_id,
                    "date": str(date),
                    "type": typ,
                    "plannedDistanceKm": planned_distance_km,
                    "plannedDurationSec": planned_duration_sec,
                    "plannedAscentM": planned_ascent_m,
                    "targetType": None if target_type == "none" else target_type,
                    "targetLabel": target_label,
                    "notes": notes,
                    "stepEndMode": step_end_mode,
                    "stepsJson": steps_json,
                }
                if existing:
                    sessions_repo.update(existing["plannedSessionId"], row)
                    st.success("Session updated")
                else:
                    sid = sessions_repo.create(row)
                    st.success(f"Session added: {sid}")
                st.session_state["planner_edit"] = None
                st.rerun()
    with col_cancel:
        if st.button("✖️ Cancel", help="Cancel editing"):
            st.session_state["planner_edit"] = None
            st.rerun()
    with col_delete:
        if existing and st.button("🗑️ Delete", help="Delete session"):
            sessions_repo.delete(existing["plannedSessionId"])
            st.session_state["planner_edit"] = None
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

                    model = build_card_view_model(
                        {
                            "plannedSessionId": sid,
                            "type": session_type,
                            "plannedDurationSec": duration,
                            "plannedDistanceKm": distance,
                            "targetType": target_type,
                            "targetLabel": target_label,
                            "stepEndMode": step_mode,
                        },
                        estimated_distance_km=estimated_distance,
                    )
                    st.markdown("<div class='rm-card'>", unsafe_allow_html=True)
                    badges = " | ".join(model["badges"])
                    suffix = f" | {badges}" if badges else ""
                    st.markdown(
                        f"<div class='rm-top'>{model['header']}{suffix}</div>",
                        unsafe_allow_html=True,
                    )
                    c1, c2, c3 = st.columns([1, 1, 1])
                    for col, action in zip((c1, c2, c3), model["actions"]):
                        key = f"{action.action}-{action.session_id}"
                        with col:
                            if st.button(action.icon, key=key, help=action.label):
                                if action.action == "edit":
                                    st.session_state["planner_edit"] = {"mode": "edit", "plannedSessionId": sid}
                                    st.rerun()
                                elif action.action == "delete":
                                    sessions_repo.delete(str(sid))
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
                st.rerun()

    totals = planner.compute_weekly_totals(athlete_id, week_records)
    totals_caption = (
        f"Totals — {_format_week_duration(totals['timeSec'])} • {fmt_decimal(totals['distanceKm'], 1)} km • {fmt_m(totals['ascentM'])}"
    )
    st.caption(totals_caption)
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
            f"{t.get('name')} ({t.get('templateId')})": t.get('templateId')
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
                st.success("Template applied")
                st.rerun()
        else:
            st.caption("No templates saved yet.")
