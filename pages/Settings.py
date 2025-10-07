import streamlit as st
from pathlib import Path

from utils.config import load_config
from utils.formatting import set_locale
from persistence.csv_storage import CsvStorage
from persistence.repositories import SettingsRepo


st.set_page_config(page_title="Running Manager - Settings")
st.title("Settings")

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
settings_repo = SettingsRepo(storage)


st.subheader("Coach Settings")
units = st.selectbox("Units", ["metric"], index=0, help="Metric units only in MVP")
distance_eq = st.number_input(
    "Distance-eq factor (km per meter ascent)", min_value=0.0, max_value=0.1, value=0.01, step=0.001,
    help="Default: 0.01 (100 m ascent = 1.0 km)"
)

if st.button("Save Settings"):
    settings_repo.update("coach-1", {"coachId": "coach-1", "units": units, "distanceEqFactor": distance_eq})
    st.success("Settings saved")

with st.expander("Integrations (info)"):
    st.write("Strava OAuth and Garmin login flows will be available in their pages.")
