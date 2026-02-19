"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Callable

import pandas as pd
import streamlit as st

from services.planner_presenter import build_card_view_model, build_empty_state_placeholder
from utils.formatting import format_week_duration, fmt_decimal, fmt_m
from utils.helpers import default_template_title

if TYPE_CHECKING:
    from persistence.repositories import PlannedSessionsRepo
    from services.planner_service import PlannerService
    from services.session_templates_service import SessionTemplatesService


def render_week_view(
    *,
    athlete_id: Optional[str],
    week_start: dt.datetime,
    week_end: dt.datetime,
    sessions_repo: "PlannedSessionsRepo",
    planner: "PlannerService",
    session_templates: "SessionTemplatesService",
    get_sessions_df_cached: Callable[[], pd.DataFrame],
    get_session_templates_cached: Callable[[], pd.DataFrame],
) -> tuple[list[Dict[str, Any]], Optional[pd.DataFrame]]:
    """Render the planner week view and return records for further use."""
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
                        target_type = (
                            record.get("targetType")
                            if isinstance(record.get("targetType"), str)
                            else None
                        )
                        target_label = (
                            record.get("targetLabel")
                            if isinstance(record.get("targetLabel"), str)
                            else None
                        )
                        step_mode = (
                            record.get("stepEndMode")
                            if isinstance(record.get("stepEndMode"), str)
                            else None
                        )

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
                        st.markdown(
                            f"<div class='rm-card-header'>{model['header']}</div>",
                            unsafe_allow_html=True,
                        )
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
                        action_cols = (
                            st.columns([1] * len(model["actions"])) if model["actions"] else []
                        )
                        for col, action in zip(action_cols, model["actions"]):
                            key = f"{action.action}-{action.session_id}"
                            with col:
                                if st.button(action.icon, key=key, help=action.label):
                                    if action.action == "edit":
                                        st.session_state["planner_edit"] = {
                                            "mode": "edit",
                                            "plannedSessionId": sid,
                                        }
                                        st.rerun()
                                    elif action.action == "save-template":
                                        session_for_template = dict(record)
                                        if not session_for_template.get("athleteId"):
                                            session_for_template["athleteId"] = athlete_id
                                        try:
                                            title_hint = default_template_title(session_for_template)
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
            f"{format_week_duration(totals['timeSec'])} • "
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

    return week_records, df_in_week
