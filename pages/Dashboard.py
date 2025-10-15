from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from config import METRICS as CONFIG_METRICS
from utils.config import load_config
from utils.formatting import set_locale
from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo


st.set_page_config(page_title="Running Manager - Dashboard", layout="wide")
st.title("Dashboard")

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
ath_repo = AthletesRepo(storage)


def _select_athlete() -> str | None:
    df = ath_repo.list()
    if df.empty:
        st.warning("Aucun athlète enregistré.")
        return None
    options = {
        f"{row.get('name') or 'Sans nom'} ({row.get('athleteId')})": row.get("athleteId")
        for _, row in df.iterrows()
    }
    label = st.selectbox("Athlète", list(options.keys()))
    return options.get(label)


def _load_daily_metrics(athlete_id: str) -> pd.DataFrame:
    df = storage.read_csv("daily_metrics.csv")
    if df.empty:
        return df
    df = df[df.get("athleteId") == athlete_id]
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    return df.reset_index(drop=True)


def _training_load_chart(df: pd.DataFrame, metric_key: str, metric_cfg: dict) -> alt.Chart:
    planned_col = metric_cfg["chronic"]
    acute_col = metric_cfg["acute"]

    working = df[["date", planned_col, acute_col]].copy()
    working["chronic_lower"] = 0.75 * working[planned_col]
    working["chronic_upper"] = 1.5 * working[planned_col]

    base = alt.Chart(working).encode(x=alt.X("date:T", title="Date"))

    fill = base.mark_area(opacity=0.2, color="#3b82f6").encode(
        y=alt.Y("chronic_lower:Q", title=metric_cfg["label"]),
        y2="chronic_upper:Q",
    )

    chronic_line = base.mark_line(color="#1d4ed8", strokeWidth=2).encode(y=f"{planned_col}:Q")
    acute_line = base.mark_line(color="#f97316", strokeWidth=2).encode(y=f"{acute_col}:Q")

    return (fill + chronic_line + acute_line).properties(height=340, width="container")


athlete_id = _select_athlete()
if not athlete_id:
    st.stop()

daily_metrics = _load_daily_metrics(athlete_id)
if daily_metrics.empty:
    st.info("Aucune donnée journalière disponible pour cet athlète.")
    st.stop()

metric_definitions = {
    "Time": {
        "label": "Temps (s)",
        "acute": "acuteTimeSec",
        "chronic": "chronicTimeSec",
    },
    "Distance": {
        "label": "Distance (km)",
        "acute": "acuteDistanceKm",
        "chronic": "chronicDistanceKm",
    },
    "DistEq": {
        "label": "Distance équivalente (km)",
        "acute": "acuteDistanceEqKm",
        "chronic": "chronicDistanceEqKm",
    },
    "Trimp": {
        "label": "TRIMP",
        "acute": "acuteTrimp",
        "chronic": "chronicTrimp",
    },
}

available_metrics = [m for m in CONFIG_METRICS if m in metric_definitions]
selected_metric = st.selectbox("Métrique", available_metrics, index=available_metrics.index("DistEq") if "DistEq" in available_metrics else 0)

chart = _training_load_chart(daily_metrics, selected_metric, metric_definitions[selected_metric])
st.altair_chart(chart, use_container_width=True)
