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

# Global activity category filter (applies to all charts below)
cat_options = ["RUN", "TRAIL_RUN", "HIKE"]
selected_cats = st.multiselect(
    "Types d'activités",
    options=cat_options,
    default=cat_options,
    help="Filtre appliqué aux graphiques ci-dessous.",
)

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
# Shared tabs for the two charts
tab_charge, tab_speed = st.tabs(["Charge", "SpeedEq"])

with tab_charge:
    st.subheader("Charge d'entraînement")
    selected_metric = st.selectbox(
        "Métrique",
        available_metrics,
        index=available_metrics.index("DistEq") if "DistEq" in available_metrics else 0,
        help="Sélectionne la métrique pour le graphique de charge."
    )
    # Recompute acute/chronic from activities for selected categories
    acts_path = storage.base_dir / "activities_metrics.csv"
    if acts_path.exists():
        acts_df_all = pd.read_csv(acts_path)
    else:
        acts_df_all = pd.DataFrame()
    if acts_df_all.empty:
        chart = _training_load_chart(daily_metrics, selected_metric, metric_definitions[selected_metric])
        st.altair_chart(chart, use_container_width=True)
    else:
        acts_df = acts_df_all[acts_df_all.get("athleteId") == athlete_id].copy()
        # Apply categories filter
        if selected_cats:
            acts_df["category"] = acts_df.get("category", pd.Series(dtype=str)).astype(str).str.upper()
            acts_df = acts_df[acts_df["category"].isin([c.upper() for c in selected_cats])]
        # Build per-day values for selected metric
        col_map = {
            "Time": "timeSec",
            "Distance": "distanceKm",
            "DistEq": "distanceEqKm",
            "Trimp": "trimp",
            "Ascent": "ascentM",
        }
        value_col = col_map.get(selected_metric, "distanceEqKm")
        acts_df["date"] = pd.to_datetime(acts_df.get("startDate"), errors="coerce").dt.normalize()
        # Include a buffer window before start_date to stabilize rolling windows
        buffer_start = pd.Timestamp(start_date) - pd.Timedelta(days=50)
        acts_df = acts_df[(acts_df["date"] >= buffer_start) & (acts_df["date"] <= pd.Timestamp(end_date))]
        per_day = (
            acts_df.groupby("date", as_index=False)[value_col].sum().rename(columns={value_col: "value"})
            if value_col in acts_df.columns
            else pd.DataFrame(columns=["date", "value"])
        )
        # Ensure continuous date index
        all_days = pd.DataFrame({
            "date": pd.date_range(start=buffer_start, end=pd.Timestamp(end_date), freq="D")
        })
        series = all_days.merge(per_day, on="date", how="left").fillna({"value": 0.0})
        # Rolling sums (acute: 7d, chronic: 42d)
        series["acute"] = series["value"].rolling(window=7, min_periods=1).sum()
        series["chronic"] = series["value"].rolling(window=42, min_periods=1).sum()
        # Slice to selected range only
        series = series[(series["date"] >= pd.Timestamp(start_date)) & (series["date"] <= pd.Timestamp(end_date))]
        # Build a minimal df with expected columns for the selected metric
        working_for_chart = pd.DataFrame({"date": series["date"]})
        col_names = {
            "Time": ("chronicTimeSec", "acuteTimeSec"),
            "Distance": ("chronicDistanceKm", "acuteDistanceKm"),
            "DistEq": ("chronicDistanceEqKm", "acuteDistanceEqKm"),
            "Trimp": ("chronicTrimp", "acuteTrimp"),
            "Ascent": ("chronicAscentM", "acuteAscentM"),
        }
        chronic_name, acute_name = col_names[selected_metric]
        working_for_chart[chronic_name] = series["chronic"].astype(float)
        working_for_chart[acute_name] = series["acute"].astype(float)
        chart = _training_load_chart(working_for_chart, selected_metric, metric_definitions[selected_metric])
        st.altair_chart(chart, use_container_width=True)

# --- Nuage d'activités: SpeedEq (km/h) vs Durée (h), couleur = FC moyenne ---
with tab_speed:
    st.subheader("Vitesse équivalente des activités")

    acts_path = storage.base_dir / "activities_metrics.csv"
    if acts_path.exists():
        acts_df = pd.read_csv(acts_path)
    else:
        acts_df = pd.DataFrame()

    if acts_df.empty:
        st.info("Aucune activité disponible pour le nuage SpeedEq.")
    else:
        # Everything below renders inside the SpeedEq tab
        acts_df = acts_df[acts_df.get("athleteId") == athlete_id].copy()
    # Keep only dates within selected range
    acts_df["date"] = pd.to_datetime(acts_df.get("startDate"), errors="coerce").dt.normalize()
    mask = (acts_df["date"] >= pd.Timestamp(start_date)) & (acts_df["date"] <= pd.Timestamp(end_date))
    acts_df = acts_df[mask]
    # Apply category filter
    if selected_cats:
        acts_df["category"] = acts_df.get("category", pd.Series(dtype=str)).astype(str).str.upper()
        acts_df = acts_df[acts_df["category"].isin([c.upper() for c in selected_cats])]
    # Compute SpeedEq = DistEq / duration (km/h)
    dist_eq = pd.to_numeric(acts_df.get("distanceEqKm"), errors="coerce").fillna(0.0)
    time_sec = pd.to_numeric(acts_df.get("timeSec"), errors="coerce").fillna(0.0)
    with pd.option_context('mode.use_inf_as_na', True):
        speed_eq_kmh = (dist_eq / time_sec.replace(0, pd.NA)) * 3600.0
    acts_df["speedEqKmh"] = speed_eq_kmh.fillna(0.0)
    acts_df["durationH"] = (time_sec / 3600.0).astype(float)
    # Join Strava name from activities.csv for tooltips
    acts_info_path = storage.base_dir / "activities.csv"
    if acts_info_path.exists():
        acts_info = pd.read_csv(acts_info_path, usecols=["activityId", "name"]) if "activityId" in pd.read_csv(acts_info_path, nrows=0).columns else pd.DataFrame()
    else:
        acts_info = pd.DataFrame()
    if not acts_info.empty and "activityId" in acts_df.columns:
        a_left = acts_df.copy()
        a_left["activityId"] = a_left["activityId"].astype(str)
        acts_info["activityId"] = acts_info["activityId"].astype(str)
        acts_df = a_left.merge(acts_info, on="activityId", how="left")
    # Prepare chart dataframe (include activityId for link building)
    keep_cols = [
        "activityId",
        "date",
        "durationH",
        "speedEqKmh",
        "avgHr",
        "distanceEqKm",
        "distanceKm",
        "ascentM",
        "name",
    ]
    cloud = acts_df[[c for c in keep_cols if c in acts_df.columns]].copy()
    cloud["avgHr"] = pd.to_numeric(cloud.get("avgHr"), errors="coerce")
    cloud = cloud.fillna({"avgHr": 0.0, "name": ""})
    if "activityId" in cloud.columns:
        cloud["activityId"] = cloud["activityId"].astype(str)
        # Link to Activities page with query param for selection
        cloud["activityUrl"] = "Activities?activityId=" + cloud["activityId"]

    cloud_chart = (
        alt.Chart(cloud)
        .mark_circle(size=60, opacity=0.85)
        .encode(
            x=alt.X("durationH:Q", title="Durée (h)"),
            y=alt.Y("speedEqKmh:Q", title="Vitesse équivalente (km/h)"),
            color=alt.Color("avgHr:Q", scale=alt.Scale(scheme="yelloworangered"), title="FC moy."),
            href=alt.Href("activityUrl:N"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("name:N", title="Nom"),
                alt.Tooltip("distanceEqKm:Q", title="Dist. équiv. (km)", format=".2f"),
                alt.Tooltip("distanceKm:Q", title="Distance (km)", format=".2f"),
                alt.Tooltip("ascentM:Q", title="D+ (m)", format=".0f"),
                alt.Tooltip("durationH:Q", title="Durée (h)", format=".2f"),
                alt.Tooltip("speedEqKmh:Q", title="SpeedEq (km/h)", format=".2f"),
                alt.Tooltip("avgHr:Q", title="FC moy.", format=".0f"),
            ],
        )
        .properties(height=360, width="container")
    )
    st.altair_chart(cloud_chart, use_container_width=True)
