"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from config import METRICS as CONFIG_METRICS
from graph.hr_speed import create_hr_speed_chart
from graph.speed_scatter import create_speedeq_scatter_chart
from graph.training_load import create_training_load_chart
from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo
from services.analytics_service import AnalyticsService
from services.dashboard_data_service import load_daily_metrics, load_hr_speed_data
from services.speed_profile_service import SpeedProfileService
from utils.config import load_config
from utils.dashboard_state import (
    get_dashboard_date_range,
    set_dashboard_date_range,
    set_dashboard_date_range_quick,
)
from utils.formatting import set_locale
from utils.styling import apply_theme
from widgets.athlete_selector import select_athlete

st.set_page_config(page_title="Running Manager - Dashboard", layout="wide")
apply_theme()
st.title("Dashboard")

CHART_WIDTH = 1100

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
ath_repo = AthletesRepo(storage)
analytics = AnalyticsService(storage)
speed_profile_service = SpeedProfileService(cfg)

athlete_id = select_athlete(ath_repo)
if not athlete_id:
    st.stop()

daily_metrics = load_daily_metrics(storage, athlete_id)
if daily_metrics.empty:
    st.info("Aucune donnée journalière disponible pour cet athlète.")
    st.stop()

# --- Date range controls (same behavior as Analytics) ---
min_plan, max_plan = analytics.planned_date_bounds(athlete_id)

min_candidates = [daily_metrics["date"].min().date()]
if min_plan:
    min_candidates.append(min_plan)
min_date = min(min_candidates)

max_candidates = [daily_metrics["date"].max().date(), pd.Timestamp.today().date()]
if max_plan:
    max_candidates.append(max_plan)
max_date = max(max_candidates)

default_end = max_date
default_start = max(min_date, (default_end - pd.DateOffset(months=3)).date())

start_dt, end_dt = get_dashboard_date_range(min_date, max_date, default_months=3)

col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
if col_btn1.button("7 jours"):
    set_dashboard_date_range_quick(max_date, days=7)
    st.rerun()
if col_btn2.button("28 jours"):
    set_dashboard_date_range_quick(max_date, days=28)
    st.rerun()
if col_btn3.button("3 mois"):
    set_dashboard_date_range_quick(max_date, months=3)
    st.rerun()
if col_btn4.button("6 mois"):
    set_dashboard_date_range_quick(max_date, months=6)
    st.rerun()
if col_btn5.button("1 an"):
    set_dashboard_date_range_quick(max_date, years=1)
    st.rerun()

start_dt, end_dt = st.slider(
    "Période",
    min_value=pd.Timestamp(min_date).to_pydatetime(),
    max_value=pd.Timestamp(max_date).to_pydatetime(),
    value=(start_dt, end_dt),
    format="YYYY-MM-DD",
)
set_dashboard_date_range(start_dt, end_dt)

start_date = pd.Timestamp(start_dt).date()
end_date = pd.Timestamp(end_dt).date()

# Filter daily metrics to selected date range
mask = (daily_metrics["date"].dt.normalize() >= pd.Timestamp(start_date)) & (
    daily_metrics["date"].dt.normalize() <= pd.Timestamp(end_date)
)
daily_metrics = daily_metrics[mask]

# Global activity category filter (applies to all charts below)
cat_options = ["RUN", "TRAIL_RUN", "HIKE", "RIDE", "BACKCOUNTRY_SKI"]
default_cat_selection = [cat for cat in cat_options if cat != "RIDE"]
selected_cats = st.multiselect(
    "Types d'activités",
    options=cat_options,
    default=default_cat_selection,
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
tab_charge, tab_speed, tab_hr_speed = st.tabs(["Charge", "SpeedEq", "FC vs Vitesse"])

with tab_charge:
    st.subheader("Charge d'entraînement")
    selected_metric = st.selectbox(
        "Métrique",
        available_metrics,
        index=available_metrics.index("DistEq") if "DistEq" in available_metrics else 0,
        help="Sélectionne la métrique pour le graphique de charge.",
    )
    # Recompute acute/chronic including planned values
    col_names = {
        "Time": ("chronicTimeSec", "acuteTimeSec"),
        "Distance": ("chronicDistanceKm", "acuteDistanceKm"),
        "DistEq": ("chronicDistanceEqKm", "acuteDistanceEqKm"),
        "Trimp": ("chronicTrimp", "acuteTrimp"),
        "Ascent": ("chronicAscentM", "acuteAscentM"),
    }

    if selected_metric not in col_names:
        chart = create_training_load_chart(
            daily_metrics, selected_metric, metric_definitions[selected_metric]
        )
        chart = chart.properties(width=CHART_WIDTH)
        st.altair_chart(chart, use_container_width=False)
    else:
        chronic_name, acute_name = col_names[selected_metric]

        buffer_start_date = (pd.Timestamp(start_date) - pd.Timedelta(days=50)).date()
        if buffer_start_date < min_date:
            buffer_start_date = min_date
        range_end_date = max(end_date, max_plan or end_date)

        daily_range_df = analytics.daily_range(
            athlete_id=athlete_id,
            metric_label=selected_metric,
            selected_types=selected_cats or None,
            start_date=buffer_start_date,
            end_date=range_end_date,
        )

        if daily_range_df.empty:
            chart = create_training_load_chart(
                daily_metrics, selected_metric, metric_definitions[selected_metric]
            )
            chart = chart.properties(width=CHART_WIDTH)
            st.altair_chart(chart, use_container_width=False)
        else:
            daily_range_df = daily_range_df.copy()
            daily_range_df["date"] = pd.to_datetime(daily_range_df["date"], errors="coerce")
            daily_range_df = daily_range_df.dropna(subset=["date"])
            daily_range_df = daily_range_df.sort_values("date")
            daily_range_df["actual_value"] = pd.to_numeric(
                daily_range_df.get("actual_value"), errors="coerce"
            ).fillna(0.0)
            daily_range_df["planned_value"] = pd.to_numeric(
                daily_range_df.get("planned_value"), errors="coerce"
            ).fillna(0.0)

            start_ts = pd.Timestamp(start_date)
            range_end_ts = pd.Timestamp(range_end_date)
            today_ts = pd.Timestamp.today().normalize()

            daily_range_df["actual_acute"] = (
                daily_range_df["actual_value"].rolling(window=7, min_periods=1).mean()
            )
            daily_range_df["actual_chronic"] = (
                daily_range_df["actual_value"].rolling(window=28, min_periods=1).mean()
            )

            combined_for_plan = np.where(
                daily_range_df["date"] <= today_ts,
                daily_range_df["actual_value"],
                daily_range_df["planned_value"],
            )
            combined_for_plan = pd.to_numeric(combined_for_plan, errors="coerce")
            daily_range_df["planned_acute"] = (
                pd.Series(combined_for_plan).rolling(window=7, min_periods=1).mean().to_numpy()
            )
            daily_range_df["planned_chronic"] = (
                pd.Series(combined_for_plan).rolling(window=28, min_periods=1).mean().to_numpy()
            )

            daily_range_df.loc[
                daily_range_df["date"] <= today_ts, ["planned_chronic", "planned_acute"]
            ] = np.nan

            plot_df_base = daily_range_df[
                (daily_range_df["date"] >= start_ts) & (daily_range_df["date"] <= range_end_ts)
            ].copy()
            plot_df_base.loc[
                plot_df_base["date"] > today_ts, ["actual_chronic", "actual_acute"]
            ] = np.nan

            band_df = plot_df_base.dropna(subset=["actual_chronic"]).copy()
            band_df["lower"] = 0.75 * band_df["actual_chronic"]
            band_df["upper"] = 1.5 * band_df["actual_chronic"]

            planned_band_df = plot_df_base[
                (plot_df_base["date"] > today_ts) & plot_df_base["planned_chronic"].notna()
            ].copy()
            if not planned_band_df.empty:
                planned_band_df["lower"] = 0.75 * planned_band_df["planned_chronic"]
                planned_band_df["upper"] = 1.5 * planned_band_df["planned_chronic"]

            plot_rows: list[dict] = []
            for _, row in plot_df_base.iterrows():
                date_val = row["date"]
                is_future = bool(date_val > today_ts)
                actual_chronic_val = row.get("actual_chronic")
                actual_acute_val = row.get("actual_acute")
                planned_chronic_val = row.get("planned_chronic")
                planned_acute_val = row.get("planned_acute")

                if pd.notna(actual_chronic_val):
                    plot_rows.append(
                        {
                            "date": date_val,
                            "value": float(actual_chronic_val),
                            "series": "Charge chronique (réalisé)",
                            "is_planned": 0,
                            "is_future": int(is_future),
                        }
                    )
                if pd.notna(actual_acute_val):
                    plot_rows.append(
                        {
                            "date": date_val,
                            "value": float(actual_acute_val),
                            "series": "Charge aiguë (réalisé)",
                            "is_planned": 0,
                            "is_future": int(is_future),
                        }
                    )
                if pd.notna(planned_chronic_val):
                    plot_rows.append(
                        {
                            "date": date_val,
                            "value": float(planned_chronic_val),
                            "series": "Charge chronique (planifié)",
                            "is_planned": 1,
                            "is_future": int(is_future),
                        }
                    )
                if pd.notna(planned_acute_val):
                    plot_rows.append(
                        {
                            "date": date_val,
                            "value": float(planned_acute_val),
                            "series": "Charge aiguë (planifié)",
                            "is_planned": 1,
                            "is_future": int(is_future),
                        }
                    )

            plot_df = pd.DataFrame(plot_rows)

            if plot_df.empty:
                chart = create_training_load_chart(
                    daily_metrics, selected_metric, metric_definitions[selected_metric]
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                color_scale = alt.Scale(
                    domain=[
                        "Charge chronique (réalisé)",
                        "Charge aiguë (réalisé)",
                        "Charge chronique (planifié)",
                        "Charge aiguë (planifié)",
                    ],
                    range=["#1d4ed8", "#f97316", "#60a5fa", "#fb923c"],
                )

                stroke_dash = alt.condition(
                    (alt.datum.is_planned == 1) & (alt.datum.is_future == 1),
                    alt.value([6, 3]),
                    alt.value([1, 0]),
                )

                line_chart = (
                    alt.Chart(plot_df)
                    .mark_line(strokeWidth=2)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("value:Q", title=metric_definitions[selected_metric]["label"]),
                        color=alt.Color("series:N", scale=color_scale, title=""),
                        strokeDash=stroke_dash,
                        detail=["series", "is_future"],
                        tooltip=[
                            alt.Tooltip("date:T", title="Date"),
                            alt.Tooltip("series:N", title="Série"),
                            alt.Tooltip("value:Q", title="Valeur", format=".2f"),
                        ],
                    )
                )

                layers: list[alt.Chart] = []
                if not band_df.empty:
                    actual_fill = (
                        alt.Chart(band_df)
                        .mark_area(opacity=0.18, color="#2563eb")
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("lower:Q", title=metric_definitions[selected_metric]["label"]),
                            y2="upper:Q",
                        )
                    )
                    layers.append(actual_fill)
                if "planned_band_df" in locals() and not planned_band_df.empty:
                    planned_fill = (
                        alt.Chart(planned_band_df)
                        .mark_area(opacity=0.18, color="#93c5fd")
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("lower:Q", title=metric_definitions[selected_metric]["label"]),
                            y2="upper:Q",
                        )
                    )
                    layers.append(planned_fill)

                layers.append(line_chart)
                chart = alt.layer(*layers).properties(height=340)

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
    mask = (acts_df["date"] >= pd.Timestamp(start_date)) & (
        acts_df["date"] <= pd.Timestamp(end_date)
    )
    acts_df = acts_df[mask]
    # Apply category filter
    if selected_cats:
        acts_df["category"] = acts_df.get("category", pd.Series(dtype=str)).astype(str).str.upper()
        acts_df = acts_df[acts_df["category"].isin([c.upper() for c in selected_cats])]
    # Compute SpeedEq = DistEq / duration (km/h)
    dist_eq = pd.to_numeric(acts_df.get("distanceEqKm"), errors="coerce").fillna(0.0)
    time_sec = pd.to_numeric(acts_df.get("timeSec"), errors="coerce").fillna(0.0)
    with pd.option_context("mode.use_inf_as_na", True):
        speed_eq_kmh = (dist_eq / time_sec.replace(0, pd.NA)) * 3600.0
    acts_df["speedEqKmh"] = speed_eq_kmh.fillna(0.0)
    acts_df["durationH"] = (time_sec / 3600.0).astype(float)
    # Join Strava name from activities.csv for tooltips
    acts_info_path = storage.base_dir / "activities.csv"
    if acts_info_path.exists():
        acts_info = (
            pd.read_csv(acts_info_path, usecols=["activityId", "name"])
            if "activityId" in pd.read_csv(acts_info_path, nrows=0).columns
            else pd.DataFrame()
        )
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
        base_page = "Activity"
        cloud["activityUrl"] = (
            base_page + "?activityId=" + cloud["activityId"] + "&athleteId=" + str(athlete_id)
        )

    cloud_chart = create_speedeq_scatter_chart(cloud, athlete_id)
    st.altair_chart(cloud_chart, use_container_width=True)

# --- HR vs Speed Graph (Cluster-based) ---
with tab_hr_speed:
    st.subheader("Relation FC vs Vitesse")

    @st.cache_data(ttl=3600)
    def _cached_load_hr_speed_data(
        athlete_id: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        categories: List[str],
    ) -> Tuple[pd.DataFrame, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Cached wrapper for load_hr_speed_data."""
        return load_hr_speed_data(
            storage, speed_profile_service, cfg, athlete_id, start_date, end_date, categories
        )

    # Load data
    hr_speed_df, slope, intercept, r_squared, std_err = _cached_load_hr_speed_data(
        athlete_id, start_date, end_date, selected_cats
    )

    if hr_speed_df.empty:
        st.info("Aucune donnée de timeseries disponible pour la période sélectionnée.")
    else:
        chart = create_hr_speed_chart(hr_speed_df, slope, intercept, r_squared, std_err, CHART_WIDTH)
        if slope is not None and intercept is not None and r_squared is not None:
            formula_text = f"FC = {slope:.2f} × Vitesse + {intercept:.2f}\nR² = {r_squared:.3f}"
            st.caption(formula_text)
        st.altair_chart(chart, use_container_width=True)
