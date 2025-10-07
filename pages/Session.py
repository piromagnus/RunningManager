from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from utils.config import load_config
from utils.formatting import set_locale
from persistence.csv_storage import CsvStorage
from persistence.repositories import PlannedSessionsRepo
from services.planner_service import PlannerService


st.set_page_config(page_title="Running Manager - Session")
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
        # Derived totals
        try:
            est_dur = planner.estimate_interval_duration_sec(data)
            est_km = planner.estimate_interval_distance_km(row.get("athleteId"), data)
            h = est_dur // 3600
            m = (est_dur % 3600) // 60
            st.caption(f"Interval totals: {h}h{m:02d} • est≈{est_km:.1f} km")
        except Exception:
            pass
        # Readable breakdown
        if "repeats" in data:
            reps = data.get("repeats") or []
            for i, rep in enumerate(reps, 1):
                st.write(f"Repeat {i}: work={rep.get('workSec')}s, rec={rep.get('recoverSec')}s, target={rep.get('targetType')} {rep.get('targetLabel')}")
        else:
            loops = data.get("loops") or []
            between = int(data.get("betweenLoopRecoverSec") or 0)
            for li, loop in enumerate(loops, 1):
                st.write(f"Loop {li} ×{int(loop.get('repeats') or 1)} (between {between}s)")
                for ai, act in enumerate(loop.get("actions") or [], 1):
                    st.write(f"- Action {ai}: {act.get('kind')} {int(act.get('sec') or 0)}s, target={act.get('targetType')} {act.get('targetLabel')}, asc={int(act.get('ascendM') or 0)}m, desc={int(act.get('descendM') or 0)}m")
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
