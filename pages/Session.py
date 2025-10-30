"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from utils.config import load_config
from utils.formatting import fmt_decimal, fmt_m, set_locale
from utils.styling import apply_theme
from persistence.csv_storage import CsvStorage
from persistence.repositories import PlannedSessionsRepo
from services.planner_service import PlannerService
from services.interval_utils import describe_action, normalize_steps


st.set_page_config(page_title="Running Manager - Session")
apply_theme()
st.title("Session details")

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
repo = PlannedSessionsRepo(storage)
planner = PlannerService(storage)

sid = st.session_state.get("session_view_sid")
if not sid:
    st.warning("No session selected.")
    st.page_link("pages/Planner.py", label="← Back to planner")
    st.stop()

row = repo.get(sid)
if not row:
    st.error("Session not found.")
    st.page_link("pages/Planner.py", label="← Back to planner")
    st.stop()

st.subheader("Overview")
st.write({k: v for k, v in row.items() if k not in ("stepsJson",)})

# Estimated km for Fundamental when duration is present
if row.get("type") == "FUNDAMENTAL_ENDURANCE" and row.get("plannedDurationSec"):
    athlete_id = row.get("athleteId")
    try:
        est = planner.estimate_km(athlete_id, int(row.get("plannedDurationSec")))
        if est is not None:
            st.caption(f"Estimated km ≈ {est:.1f}")
    except Exception:
        pass

if row.get("stepsJson"):
    st.subheader("Interval steps")
    try:
        data = json.loads(row.get("stepsJson"))
        try:
            est_dur = planner.estimate_interval_duration_sec(data)
            est_km = planner.estimate_interval_distance_km(row.get("athleteId"), data)
            est_asc = planner.estimate_interval_ascent_m(data)
            st.caption(
                f"Durée ≈ {_format_session_duration(est_dur)} • "
                f"Distance ≈ {fmt_decimal(est_km, 1)} km • "
                f"D+ ≈ {fmt_m(est_asc)}"
            )
        except Exception:
            pass
        steps = normalize_steps(data)
        if steps["preBlocks"]:
            st.markdown("**Avant**")
            for block in steps["preBlocks"]:
                st.markdown(f"- {describe_action(block)}")
        for li, loop in enumerate(steps["loops"], 1):
            repeats = int(loop.get("repeats") or 1)
            st.markdown(f"**Boucle {li} ×{repeats}**")
            for action in loop.get("actions") or []:
                st.markdown(f"- {describe_action(action)}")
        between = steps.get("betweenBlock")
        if between and int(between.get("sec") or 0) > 0:
            st.markdown("**Entre boucles**")
            st.markdown(f"- {describe_action(between)}")
        if steps["postBlocks"]:
            st.markdown("**Après**")
            for block in steps["postBlocks"]:
                st.markdown(f"- {describe_action(block)}")
        st.expander("Raw steps JSON").json(data)
    except Exception:
        st.text(row.get("stepsJson"))

col1, col2 = st.columns(2)
with col1:
    if st.button("← Back"):
        st.switch_page("pages/Planner.py")
with col2:
    if st.button("✏️ Edit"):
        st.session_state["planner_edit"] = {"mode": "edit", "plannedSessionId": sid}
        st.switch_page("pages/Planner.py")
