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
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    numeric_columns = [
        "distanceKm",
        "distanceEqKm",
        "timeSec",
        "trimp",
        "ascentM",
        "acuteDistanceKm",
        "chronicDistanceKm",
        "acuteDistanceEqKm",
        "chronicDistanceEqKm",
        "acuteTimeSec",
        "chronicTimeSec",
        "acuteTrimp",
        "chronicTrimp",
        "acuteAscentM",
        "chronicAscentM",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df = df.sort_values("date")
    return df.reset_index(drop=True)


def _training_load_chart(df: pd.DataFrame, metric_key: str, metric_cfg: dict) -> alt.Chart:
    planned_col = metric_cfg["chronic"]
    acute_col = metric_cfg["acute"]

    working = df[["date", planned_col, acute_col]].copy()
    working[planned_col] = pd.to_numeric(working[planned_col], errors="coerce").fillna(0.0)
    working[acute_col] = pd.to_numeric(working[acute_col], errors="coerce").fillna(0.0)
    working["chronic_lower"] = 0.75 * working[planned_col]
    working["chronic_upper"] = 1.5 * working[planned_col]

    base = alt.Chart(working).encode(x=alt.X("date:T", title="Date"))

    fill = base.mark_area(opacity=0.2, color="#2563eb").encode(
        y=alt.Y("chronic_lower:Q", title=metric_cfg["label"]),
        y2="chronic_upper:Q",
    )

    chronic_line = base.mark_line(color="#1d4ed8", strokeWidth=2).encode(y=f"{planned_col}:Q")
    acute_line = base.mark_line(color="#f97316", strokeWidth=2).encode(y=f"{acute_col}:Q")

    return (
        fill
        + chronic_line
        + acute_line
    ).encode(
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip(planned_col, title="Charge chronique", format=".2f"),
            alt.Tooltip(acute_col, title="Charge aiguë", format=".2f"),
        ]
    ).properties(height=340, width="container")


athlete_id = _select_athlete()
if not athlete_id:
    st.stop()

daily_metrics = _load_daily_metrics(athlete_id)
if daily_metrics.empty:
    st.info("Aucune donnée journalière disponible pour cet athlète.")
    st.stop()

# --- Date range controls (same behavior as Analytics) ---
min_date = daily_metrics["date"].min().date()
max_date = daily_metrics["date"].max().date()

default_end = max_date
default_start = max(min_date, default_end - pd.Timedelta(days=28).to_pytimedelta())

if "dashboard_range" not in st.session_state:
    st.session_state["dashboard_range"] = (
        pd.Timestamp(default_start).to_pydatetime(),
        pd.Timestamp(default_end).to_pydatetime(),
    )

col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
if col_btn1.button("7 jours"):
    st.session_state["dashboard_range"] = (
        (pd.Timestamp(max_date) - pd.Timedelta(days=7)).to_pydatetime(),
        pd.Timestamp(max_date).to_pydatetime(),
    )
if col_btn2.button("28 jours"):
    st.session_state["dashboard_range"] = (
        (pd.Timestamp(max_date) - pd.Timedelta(days=28)).to_pydatetime(),
        pd.Timestamp(max_date).to_pydatetime(),
    )
if col_btn3.button("3 mois"):
    st.session_state["dashboard_range"] = (
        (pd.Timestamp(max_date) - pd.DateOffset(months=3)).to_pydatetime(),
        pd.Timestamp(max_date).to_pydatetime(),
    )
if col_btn4.button("1 an"):
    st.session_state["dashboard_range"] = (
        (pd.Timestamp(max_date) - pd.DateOffset(years=1)).to_pydatetime(),
        pd.Timestamp(max_date).to_pydatetime(),
    )

start_dt, end_dt = st.slider(
    "Période",
    min_value=pd.Timestamp(min_date).to_pydatetime(),
    max_value=pd.Timestamp(max_date).to_pydatetime(),
    value=(
        pd.to_datetime(st.session_state["dashboard_range"][0]).to_pydatetime(),
        pd.to_datetime(st.session_state["dashboard_range"][1]).to_pydatetime(),
    ),
    format="YYYY-MM-DD",
)
st.session_state["dashboard_range"] = (start_dt, end_dt)

start_date = pd.Timestamp(start_dt).date()
end_date = pd.Timestamp(end_dt).date()

# Filter daily metrics to selected date range
mask = (daily_metrics["date"].dt.normalize() >= pd.Timestamp(start_date)) & (
    daily_metrics["date"].dt.normalize() <= pd.Timestamp(end_date)
)
daily_metrics = daily_metrics[mask]

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
    "Ascent": {
        "label": "Dénivelé positif (m)",
        "acute": "acuteAscentM",
        "chronic": "chronicAscentM",
    },
}

available_metrics: list[str] = [m for m in CONFIG_METRICS if m in metric_definitions]
for metric_key in metric_definitions.keys():
    if metric_key not in available_metrics:
        available_metrics.append(metric_key)
selected_metric = st.selectbox("Métrique", available_metrics, index=available_metrics.index("DistEq") if "DistEq" in available_metrics else 0)

chart = _training_load_chart(daily_metrics, selected_metric, metric_definitions[selected_metric])
st.altair_chart(chart, use_container_width=True)
