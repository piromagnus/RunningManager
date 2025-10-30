"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import streamlit as st
from pathlib import Path

from utils.config import load_config
from utils.formatting import set_locale
from utils.styling import apply_theme
from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo, ThresholdsRepo
from utils.ids import new_id

DEFAULT_HR_REST = 60.0
DEFAULT_HR_MAX = 190.0

st.set_page_config(page_title="Running Manager - Athlete")
apply_theme()
st.title("Athlete")

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
ath_repo = AthletesRepo(storage)
thr_repo = ThresholdsRepo(storage)

st.subheader("Athletes")

# Load existing athletes
athletes_df = ath_repo.list()
athlete_options = []
id_by_label = {}
for _, row in athletes_df.iterrows():
    label = f"{row.get('name') or 'Unnamed'} ({row.get('athleteId')})"
    athlete_options.append(label)
    id_by_label[label] = row.get("athleteId")

selected_label = None
if athlete_options:
    selected_label = st.selectbox("Select athlete", athlete_options, index=0)
    selected_athlete_id = id_by_label.get(selected_label)
else:
    st.info("No athletes yet. Use 'Add athlete' to create one.")
    selected_athlete_id = None

# Add athlete flow (button opens a small form)
if "adding_athlete" not in st.session_state:
    st.session_state["adding_athlete"] = False

if st.button("Add athlete"):
    st.session_state["adding_athlete"] = True

if st.session_state.get("adding_athlete"):
    with st.form("add-athlete-form"):
        name = st.text_input("Athlete name", value="Runner 1")
        units = st.selectbox("Units", ["metric"], index=0)
        submitted = st.form_submit_button("Save")
        if submitted:
            aid = new_id()
            ath_repo.create(
                {
                    "athleteId": aid,
                    "coachId": "coach-1",
                    "name": name,
                    "thresholdsProfileId": "default",
                    "units": units,
                    "hrRest": DEFAULT_HR_REST,
                    "hrMax": DEFAULT_HR_MAX,
                }
            )
            st.session_state["adding_athlete"] = False
            st.success("Athlete added")

st.subheader("Thresholds (manual entries)")
if selected_athlete_id is None:
    st.warning("Select an athlete to add/view thresholds.")
else:
    athlete_record = ath_repo.get(selected_athlete_id) or {}
    with st.form("athlete-details-form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            name_value = st.text_input("Name", value=str(athlete_record.get("name") or ""))
        with col_b:
            units_value = st.selectbox("Units", ["metric"], index=0)
        with col_c:
            hr_rest_value = st.number_input(
                "Resting HR",
                min_value=30.0,
                max_value=120.0,
                value=float(athlete_record.get("hrRest") or DEFAULT_HR_REST),
                step=1.0,
            )
        hr_max_value = st.number_input(
            "Max HR",
            min_value=60.0,
            max_value=240.0,
            value=float(athlete_record.get("hrMax") or DEFAULT_HR_MAX),
            step=1.0,
        )
        if st.form_submit_button("Save athlete info"):
            ath_repo.update(
                selected_athlete_id,
                {
                    "name": name_value,
                    "units": units_value,
                    "hrRest": hr_rest_value,
                    "hrMax": hr_max_value,
                },
            )
            st.success("Athlete information updated")

    # Show existing thresholds for the selected athlete
    thr_df = thr_repo.list(athleteId=selected_athlete_id)
    if not thr_df.empty:
        st.write("Existing thresholds (edit inline):")
        edited = st.data_editor(
            thr_df,
            use_container_width=True,
            disabled=["thresholdId", "athleteId"],
            hide_index=True,
        )
        if st.button("Save threshold edits"):
            try:
                # Persist row-by-row based on thresholdId
                for _, row in edited.iterrows():
                    tid = row.get("thresholdId")
                    if not tid:
                        continue
                    thr_repo.update(str(tid), row.to_dict())
                st.success("Threshold changes saved")
            except Exception as e:
                st.error(f"Failed to save edits: {e}")
    else:
        st.info("No thresholds yet for this athlete.")

    with st.form("add-threshold"):
        col1, col2 = st.columns(2)
        with col1:
            th_name = st.selectbox(
                "Name", ["Fundamental", "Threshold 60", "Threshold 30", "MVA", "Max speed"]
            )
            hr_min = st.number_input("HR min", min_value=0, max_value=250, value=0)
            hr_max = st.number_input("HR max", min_value=0, max_value=250, value=150)
        with col2:
            pace_min = st.number_input(
                "Pace flat min (km/h)", min_value=0.0, max_value=30.0, value=8.0, step=0.1
            )
            pace_max = st.number_input(
                "Pace flat max (km/h)", min_value=0.0, max_value=30.0, value=12.0, step=0.1
            )
            ascend_min = st.number_input(
                "Ascent rate min (m/h)", min_value=0, max_value=10000, value=0
            )
            ascend_max = st.number_input(
                "Ascent rate max (m/h)", min_value=0, max_value=10000, value=0
            )
        submitted = st.form_submit_button("Add threshold")
        if submitted:
            tid = new_id()
            thr_repo.create(
                {
                    "thresholdId": tid,
                    "athleteId": selected_athlete_id,
                    "name": th_name,
                    "hrMin": hr_min,
                    "hrMax": hr_max,
                    "paceFlatKmhMin": pace_min,
                    "paceFlatKmhMax": pace_max,
                    "ascentRateMPerHMin": ascend_min,
                    "ascentRateMPerHMax": ascend_max,
                }
            )
            st.success("Threshold added")
