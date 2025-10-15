from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from config import METRICS as CONFIG_METRICS
from utils.config import load_config
from utils.formatting import fmt_decimal, set_locale
from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo, SettingsRepo
from services.analytics_service import AnalyticsService


st.set_page_config(page_title="Running Manager - Analytics", layout="wide")
st.title("Analytics")

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
ath_repo = AthletesRepo(storage)
settings_repo = SettingsRepo(storage)
analytics = AnalyticsService(storage)

CATEGORY_OPTIONS = {
    "RUN": "Course",
    "TRAIL_RUN": "Trail",
    "HIKE": "Randonnée",
    "RIDE": "Cyclisme",
}

METRIC_CONFIG = {
    "Time": {
        "planned_col": "plannedTimeSec",
        "category_suffix": "TimeSec",
        "transform": analytics.seconds_to_hours,
        "unit": "heures",
    },
    "Distance": {
        "planned_col": "plannedDistanceKm",
        "category_suffix": "DistanceKm",
        "transform": None,
        "unit": "km",
    },
    "Trimp": {
        "planned_col": "plannedTrimp",
        "category_suffix": "Trimp",
        "transform": None,
        "unit": "TRIMP",
    },
    "DistEq": {
        "planned_col": "plannedDistanceEqKm",
        "category_suffix": "DistanceEqKm",
        "transform": None,
        "unit": "km équivalent",
    },
}


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


def _load_saved_activity_types() -> list[str]:
    settings = settings_repo.get("coach-1") or {}
    raw = settings.get("analyticsActivityTypes") or "[]"
    try:
        values = json.loads(raw)
        if isinstance(values, list):
            return [v for v in values if v in CATEGORY_OPTIONS]
    except json.JSONDecodeError:
        pass
    return []


athlete_id = _select_athlete()
if not athlete_id:
    st.stop()

with st.expander("Filtrer les activités incluses"):
    default_types = _load_saved_activity_types() or [key for key in CATEGORY_OPTIONS.keys() if key != "RIDE"]
    selected_types = st.multiselect(
        "Inclure les types d'activités suivants",
        options=list(CATEGORY_OPTIONS.keys()),
        default=default_types,
        format_func=lambda key: CATEGORY_OPTIONS[key],
    )
    if not selected_types:
        st.warning("Au moins un type doit être sélectionné. Toutes les activités Run/Trail/Hike seront incluses.")
        selected_types = list(CATEGORY_OPTIONS.keys())
    if st.button("Enregistrer ces types pour l'analytics"):
        settings_repo.update("coach-1", {"analyticsActivityTypes": json.dumps(selected_types)})
        st.success("Préférence sauvegardée.")

metric_label_default = CONFIG_METRICS.index("Distance") if "Distance" in CONFIG_METRICS else 0
metric_label = st.selectbox("Métrique", CONFIG_METRICS, index=metric_label_default)
metric_cfg = METRIC_CONFIG[metric_label]

df = analytics.load_weekly_metrics(athlete_id=athlete_id)
if df.empty:
    st.info("Aucune métrique hebdomadaire disponible.")
    st.stop()

working_df = df.copy()
planned_raw = pd.to_numeric(working_df[metric_cfg["planned_col"]], errors="coerce").fillna(0.0)

metrics_path = storage.base_dir / "activities_metrics.csv"
if metrics_path.exists():
    activities_metrics_df = pd.read_csv(metrics_path)
else:
    activities_metrics_df = pd.DataFrame()

value_column_map = {
    "Time": "timeSec",
    "Distance": "distanceKm",
    "DistEq": "distanceEqKm",
    "Trimp": "trimp",
}

if not activities_metrics_df.empty:
    activities_metrics_df = activities_metrics_df[activities_metrics_df["athleteId"] == athlete_id]
    activities_metrics_df["category"] = activities_metrics_df["category"].astype(str).str.upper()
    if selected_types:
        activities_metrics_df = activities_metrics_df[
            activities_metrics_df["category"].isin(selected_types)
        ]
    activities_metrics_df["date"] = pd.to_datetime(
        activities_metrics_df["startDate"], errors="coerce"
    )
    activities_metrics_df = activities_metrics_df.dropna(subset=["date"])
    activities_metrics_df["isoYear"] = activities_metrics_df["date"].dt.isocalendar().year
    activities_metrics_df["isoWeek"] = activities_metrics_df["date"].dt.isocalendar().week
    actual_agg = (
        activities_metrics_df.groupby(["isoYear", "isoWeek"], as_index=False)[
            value_column_map[metric_label]
        ]
        .sum()
        .rename(columns={value_column_map[metric_label]: "actualMetricValue"})
    )
else:
    actual_agg = pd.DataFrame(columns=["isoYear", "isoWeek", "actualMetricValue"])

working_df = working_df.merge(actual_agg, on=["isoYear", "isoWeek"], how="left")
working_df["actualMetricValue"] = working_df["actualMetricValue"].fillna(0.0)
actual_raw = working_df["actualMetricValue"]

if metric_cfg["transform"]:
    planned_values = planned_raw.apply(metric_cfg["transform"])
    actual_values = actual_raw.apply(metric_cfg["transform"])
else:
    planned_values = planned_raw
    actual_values = actual_raw

planned_col_name = "planned_metric"
actual_col_name = "actual_metric"
working_df[planned_col_name] = planned_values
working_df[actual_col_name] = actual_values

stack_df = analytics.build_planned_vs_actual_segments(
    working_df,
    planned_column=planned_col_name,
    actual_column=actual_col_name,
    metric_key=metric_label.lower(),
)

if stack_df.empty:
    st.info("Pas encore de données à comparer.")
    st.stop()

color_scale = alt.Scale(
    domain=["Réalisé", "Au-dessus du plan", "Plan manquant"],
    range=["#3b82f6", "#16a34a", "#f97316"],
)

chart = (
    alt.Chart(stack_df)
    .mark_bar()
    .encode(
        x=alt.X("weekLabel:N", title="Semaine"),
        y=alt.Y("value:Q", title=f"{metric_label} ({metric_cfg['unit']})"),
        color=alt.Color("segment:N", scale=color_scale, title=""),
        order=alt.Order("order:Q"),
        tooltip=[
            alt.Tooltip("weekLabel:N", title="Semaine"),
            alt.Tooltip("segment:N", title="Segment"),
            alt.Tooltip("value:Q", title="Valeur", format=".2f"),
            alt.Tooltip("planned:Q", title="Planifié", format=".2f"),
            alt.Tooltip("actual:Q", title="Réalisé", format=".2f"),
            alt.Tooltip("maxValue:Q", title="Max", format=".2f"),
        ],
    )
    .properties(height=400, width="container")
)

st.altair_chart(chart, use_container_width=True)

st.caption(
    "Le segment 'Réalisé' représente la partie communément couverte. "
    "La portion 'Au-dessus du plan' s'affiche lorsque la charge réalisée dépasse le plan. "
    "La portion 'Plan manquant' matérialise l'écart restant par rapport au plan."
)

with st.expander("Détail hebdomadaire"):
    summary_df = stack_df[["weekLabel", "planned", "actual", "maxValue"]].drop_duplicates()
    summary_df = summary_df.sort_values("weekLabel")
    summary_df["Planifié"] = summary_df["planned"].map(lambda v: fmt_decimal(v, 1))
    summary_df["Réalisé"] = summary_df["actual"].map(lambda v: fmt_decimal(v, 1))
    summary_df["Max"] = summary_df["maxValue"].map(lambda v: fmt_decimal(v, 1))
    st.dataframe(
        summary_df[["weekLabel", "Planifié", "Réalisé", "Max"]].rename(columns={"weekLabel": "Semaine"}),
        use_container_width=True,
    )
