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

from graph.hr_speed import create_hr_speed_chart
from graph.hr_zones import build_zone_colors, render_zone_borders_chart, render_zone_speed_evolution
from graph.redi import create_workload_ratio_chart
from graph.speed_profile import create_speed_profile_chart, create_speed_profile_cloud_chart
from graph.speed_scatter import create_speedeq_scatter_chart
from graph.training_load import create_training_load_chart
from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo
from services.analytics_service import AnalyticsService
from services.dashboard_data_service import (
    load_aggregated_speed_profile,
    load_daily_metrics,
    load_hr_speed_data,
    load_speed_profile_cloud,
)
from services.hr_zones_service import HrZonesService
from services.speed_profile_service import SpeedProfileService
from services.timeseries_service import TimeseriesService
from utils.config import load_config
from utils.constants import CATEGORY_ORDER, CHART_WIDTH_DASHBOARD
from utils.constants import METRICS as CONFIG_METRICS
from utils.dashboard_state import (
    get_dashboard_date_range,
    set_dashboard_date_range,
    set_dashboard_date_range_quick,
)
from utils.formatting import set_locale
from utils.redi import compute_ewma, compute_redi
from utils.styling import apply_theme
from widgets.athlete_selector import select_athlete

st.set_page_config(page_title="Running Manager - Dashboard", layout="wide")
apply_theme()
st.title("Dashboard")

CHART_WIDTH = CHART_WIDTH_DASHBOARD

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
ath_repo = AthletesRepo(storage)
analytics = AnalyticsService(storage)
speed_profile_service = SpeedProfileService(cfg)
ts_service = TimeseriesService(cfg)
hr_zones_service = HrZonesService(storage, ts_service, speed_profile_service)

athlete_id = select_athlete(ath_repo)
if not athlete_id:
    st.stop()

daily_metrics = load_daily_metrics(storage, athlete_id)
if daily_metrics.empty:
    st.info("Aucune donnée journalière disponible pour cet athlète.")
    st.stop()
daily_metrics_all = daily_metrics.copy()

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
cat_options = list(CATEGORY_ORDER)
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

raw_metric_columns = {
    "Time": "timeSec",
    "Distance": "distanceKm",
    "DistEq": "distanceEqKm",
    "Trimp": "trimp",
    "Ascent": "ascentM",
}


def _compute_acute_chronic(
    values: pd.Series | np.ndarray,
    method: str,
    *,
    lambda_acute: float = 0.25,
    lambda_chronic: float = 0.07,
) -> tuple[np.ndarray, np.ndarray]:
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    method_key = str(method).upper()

    if method_key == "REDI":
        acute_values = compute_redi(numeric_values, lam=lambda_acute)
        chronic_values = compute_redi(numeric_values, lam=lambda_chronic)
    elif method_key == "EWMA":
        acute_values = compute_ewma(numeric_values, lam=lambda_acute)
        chronic_values = compute_ewma(numeric_values, lam=lambda_chronic)
    else:
        series_values = pd.Series(numeric_values)
        acute_values = series_values.rolling(window=7, min_periods=1).mean().to_numpy()
        chronic_values = series_values.rolling(window=28, min_periods=1).mean().to_numpy()

    return acute_values, chronic_values


def _stateful_radio(
    *,
    label: str,
    options: list[str],
    key: str,
    default: str,
    help_text: str | None = None,
) -> str:
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default if default in options else options[0]
    return st.radio(
        label,
        options=options,
        key=key,
        horizontal=True,
        help=help_text,
    )


available_metrics: list[str] = [m for m in CONFIG_METRICS if m in metric_definitions]
for metric_key in metric_definitions.keys():
    if metric_key not in available_metrics:
        available_metrics.append(metric_key)
# Shared tabs for the charts
(
    tab_charge,
    tab_redi,
    tab_speed,
    tab_hr_speed,
    tab_hr_shift,
    tab_speed_profile,
    tab_speed_profile_cloud,
    tab_hr_borders,
    tab_zone_speed,
) = st.tabs(
    [
        "Charge",
        "REDI",
        "SpeedEq",
        "FC vs Vitesse",
        "Décalage HR",
        "Profil de vitesse",
        "Nuage de vitesse max",
        "Zones HR",
        "Vitesse par zone",
    ]
)

with tab_charge:
    st.subheader("Charge d'entraînement")
    selected_metric = _stateful_radio(
        label="Métrique",
        options=available_metrics,
        key="dashboard_charge_metric",
        default="DistEq",
        help_text="Sélectionne la métrique pour le graphique de charge.",
    )
    selected_ratio_method = _stateful_radio(
        label="Méthode de ratio",
        options=["REDI", "EWMA", "ACWR"],
        key="dashboard_charge_ratio_method",
        default="REDI",
        help_text="Définit la méthode de calcul des charges aiguë/chronique. REDI par défaut.",
    )
    # Recompute acute/chronic including planned values
    col_names = {
        "Time": ("chronicTimeSec", "acuteTimeSec"),
        "Distance": ("chronicDistanceKm", "acuteDistanceKm"),
        "DistEq": ("chronicDistanceEqKm", "acuteDistanceEqKm"),
        "Trimp": ("chronicTrimp", "acuteTrimp"),
        "Ascent": ("chronicAscentM", "acuteAscentM"),
    }

    def _build_charge_chart(base_df: pd.DataFrame) -> alt.Chart:
        workload_col = raw_metric_columns.get(selected_metric)
        if workload_col and workload_col in base_df.columns:
            working_df = base_df[["date", workload_col]].copy()
            working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce")
            working_df = working_df.dropna(subset=["date"]).sort_values("date")
            working_df[workload_col] = pd.to_numeric(working_df[workload_col], errors="coerce").fillna(0.0)
            acute_values, chronic_values = _compute_acute_chronic(
                working_df[workload_col], selected_ratio_method
            )
            working_df["acute_custom"] = acute_values
            working_df["chronic_custom"] = chronic_values
            method_cfg = {
                "label": metric_definitions[selected_metric]["label"],
                "acute": "acute_custom",
                "chronic": "chronic_custom",
            }
            return create_training_load_chart(
                working_df,
                selected_metric,
                method_cfg,
                band_lower_ratio=0.8,
                band_upper_ratio=1.3,
                danger_ratio=1.5,
            )
        return create_training_load_chart(
            base_df,
            selected_metric,
            metric_definitions[selected_metric],
            band_lower_ratio=0.8,
            band_upper_ratio=1.3,
            danger_ratio=1.5,
        )

    if selected_metric not in col_names:
        chart = _build_charge_chart(daily_metrics)
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
            chart = _build_charge_chart(daily_metrics)
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

            actual_acute_values, actual_chronic_values = _compute_acute_chronic(
                daily_range_df["actual_value"], selected_ratio_method
            )
            daily_range_df["actual_acute"] = actual_acute_values
            daily_range_df["actual_chronic"] = actual_chronic_values

            combined_for_plan = np.where(
                daily_range_df["date"] <= today_ts,
                daily_range_df["actual_value"],
                daily_range_df["planned_value"],
            )
            combined_for_plan = pd.to_numeric(combined_for_plan, errors="coerce")
            planned_acute_values, planned_chronic_values = _compute_acute_chronic(
                combined_for_plan, selected_ratio_method
            )
            daily_range_df["planned_acute"] = planned_acute_values
            daily_range_df["planned_chronic"] = planned_chronic_values

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
            band_df["lower"] = 0.8 * band_df["actual_chronic"]
            band_df["upper"] = 1.3 * band_df["actual_chronic"]
            band_df["danger"] = 1.5 * band_df["actual_chronic"]

            planned_band_df = plot_df_base[
                (plot_df_base["date"] > today_ts) & plot_df_base["planned_chronic"].notna()
            ].copy()
            if not planned_band_df.empty:
                planned_band_df["lower"] = 0.8 * planned_band_df["planned_chronic"]
                planned_band_df["upper"] = 1.3 * planned_band_df["planned_chronic"]
                planned_band_df["danger"] = 1.5 * planned_band_df["planned_chronic"]

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
                chart = _build_charge_chart(daily_metrics)
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
                    actual_danger = (
                        alt.Chart(band_df)
                        .mark_line(color="#dc2626", strokeWidth=1.8)
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("danger:Q", title=metric_definitions[selected_metric]["label"]),
                        )
                    )
                    layers.append(actual_danger)
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
                    planned_danger = (
                        alt.Chart(planned_band_df)
                        .mark_line(color="#fca5a5", strokeWidth=1.6, strokeDash=[6, 3])
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("danger:Q", title=metric_definitions[selected_metric]["label"]),
                        )
                    )
                    layers.append(planned_danger)

                layers.append(line_chart)
                chart = alt.layer(*layers).properties(height=340)

                st.altair_chart(chart, use_container_width=True)


with tab_redi:
    st.subheader("Ratios ACWR, EWMA & REDI")
    selected_redi_metric = _stateful_radio(
        label="Métrique",
        options=available_metrics,
        key="dashboard_redi_metric",
        default="DistEq",
        help_text="Sélectionne la métrique utilisée pour les indices ACWR/EWMA/REDI.",
    )

    acute_col, chronic_col = st.columns(2)
    lambda_acute = acute_col.slider(
        "Lambda aiguë",
        min_value=0.05,
        max_value=0.5,
        value=0.25,
        step=0.01,
        key="dashboard_redi_lambda_acute",
        help="Par défaut 0.25 (équivalent EWMA N=7).",
    )
    lambda_chronic = chronic_col.slider(
        "Lambda chronique",
        min_value=0.01,
        max_value=0.15,
        value=0.07,
        step=0.005,
        key="dashboard_redi_lambda_chronic",
        help="Par défaut 0.07 (équivalent EWMA N=28).",
    )

    if lambda_chronic >= lambda_acute:
        st.info(
            "Conseil: utilisez une lambda aiguë supérieure à la lambda chronique "
            "pour une meilleure séparation des signaux."
        )

    workload_column = raw_metric_columns.get(selected_redi_metric)
    if not workload_column or workload_column not in daily_metrics_all.columns:
        st.info("Métrique indisponible pour les calculs ACWR/EWMA/REDI.")
    else:
        redi_df = daily_metrics_all[["date", workload_column]].copy()
        redi_df["date"] = pd.to_datetime(redi_df["date"], errors="coerce")
        redi_df = redi_df.dropna(subset=["date"]).sort_values("date")
        redi_df[workload_column] = pd.to_numeric(redi_df[workload_column], errors="coerce").fillna(0.0)

        workloads = redi_df[workload_column].to_numpy(dtype=float)

        acwr_acute_values, acwr_chronic_values = _compute_acute_chronic(workloads, "ACWR")
        redi_df["acwr_acute"] = acwr_acute_values
        redi_df["acwr_chronic"] = acwr_chronic_values
        redi_df["redi_acute"] = compute_redi(workloads, lam=lambda_acute)
        redi_df["redi_chronic"] = compute_redi(workloads, lam=lambda_chronic)
        redi_df["ewma_acute"] = compute_ewma(workloads, lam=lambda_acute)
        redi_df["ewma_chronic"] = compute_ewma(workloads, lam=lambda_chronic)

        redi_df["acwr_ratio"] = np.divide(
            redi_df["acwr_acute"],
            redi_df["acwr_chronic"],
            out=np.full(redi_df.shape[0], np.nan),
            where=redi_df["acwr_chronic"] > 0,
        )
        redi_df["redi_ratio"] = np.divide(
            redi_df["redi_acute"],
            redi_df["redi_chronic"],
            out=np.full(redi_df.shape[0], np.nan),
            where=redi_df["redi_chronic"] > 0,
        )
        redi_df["ewma_ratio"] = np.divide(
            redi_df["ewma_acute"],
            redi_df["ewma_chronic"],
            out=np.full(redi_df.shape[0], np.nan),
            where=redi_df["ewma_chronic"] > 0,
        )

        ratio_mask = (redi_df["date"].dt.normalize() >= pd.Timestamp(start_date)) & (
            redi_df["date"].dt.normalize() <= pd.Timestamp(end_date)
        )
        ratio_plot_df = redi_df[ratio_mask].copy()
        ratio_plot_df = ratio_plot_df.replace([np.inf, -np.inf], np.nan)

        if ratio_plot_df.empty:
            st.info("Aucune donnée disponible pour les indices ACWR/EWMA/REDI.")
        else:
            acwr_chart = create_workload_ratio_chart(
                ratio_plot_df,
                method_label="ACWR",
                ratio_col="acwr_ratio",
                acute_col="acwr_acute",
                chronic_col="acwr_chronic",
            )
            redi_chart = create_workload_ratio_chart(
                ratio_plot_df,
                method_label="REDI",
                ratio_col="redi_ratio",
                acute_col="redi_acute",
                chronic_col="redi_chronic",
            )
            ewma_chart = create_workload_ratio_chart(
                ratio_plot_df,
                method_label="EWMA",
                ratio_col="ewma_ratio",
                acute_col="ewma_acute",
                chronic_col="ewma_chronic",
            )

            chart_col_acwr, chart_col_ewma, chart_col_redi = st.columns(3)
            with chart_col_acwr:
                st.caption("ACWR (aiguë / chronique)")
                st.altair_chart(acwr_chart, use_container_width=True)
            with chart_col_ewma:
                st.caption("EWMA (aiguë / chronique)")
                st.altair_chart(ewma_chart, use_container_width=True)
            with chart_col_redi:
                st.caption("REDI (aiguë / chronique)")
                st.altair_chart(redi_chart, use_container_width=True)

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

with tab_hr_shift:
    st.subheader("Evolution du décalage HR par activité")
    acts_path = storage.base_dir / "activities_metrics.csv"
    shifts_df = pd.read_csv(acts_path) if acts_path.exists() else pd.DataFrame()
    if shifts_df.empty:
        st.info("Aucune donnée d'activité disponible.")
    else:
        required_cols = {"activityId", "athleteId", "startDate", "hrSpeedShift"}
        if not required_cols.issubset(set(shifts_df.columns)):
            st.info("Les colonnes nécessaires au suivi du décalage HR sont absentes.")
        else:
            shifts_df = shifts_df[shifts_df["athleteId"].astype(str) == str(athlete_id)].copy()
            shifts_df["date"] = pd.to_datetime(shifts_df["startDate"], errors="coerce")
            shifts_df = shifts_df.dropna(subset=["date"])
            shifts_df = shifts_df[
                (shifts_df["date"].dt.date >= start_date) & (shifts_df["date"].dt.date <= end_date)
            ]
            if selected_cats:
                allowed = {str(cat).upper() for cat in selected_cats}
                if "category" in shifts_df.columns:
                    shifts_df["category"] = shifts_df["category"].astype(str).str.upper()
                    shifts_df = shifts_df[shifts_df["category"].isin(allowed)]
            if shifts_df.empty:
                st.info("Aucune donnée de décalage HR sur la période sélectionnée.")
            else:
                shifts_df["hrSpeedShift"] = pd.to_numeric(shifts_df["hrSpeedShift"], errors="coerce")
                shifts_df = shifts_df.dropna(subset=["hrSpeedShift"]).copy()
                if shifts_df.empty:
                    st.info("Aucune valeur de décalage HR disponible sur la période sélectionnée.")
                else:
                    shifts_df["distanceKm"] = pd.to_numeric(
                        shifts_df.get("distanceKm"), errors="coerce"
                    ).fillna(0.0)
                    shifts_df["distanceEqKm"] = pd.to_numeric(
                        shifts_df.get("distanceEqKm"), errors="coerce"
                    ).fillna(0.0)
                    shifts_df["timeSec"] = pd.to_numeric(
                        shifts_df.get("timeSec"), errors="coerce"
                    ).fillna(0.0)
                    shifts_df["avgHr"] = pd.to_numeric(shifts_df.get("avgHr"), errors="coerce")
                    shifts_df["trimp"] = pd.to_numeric(shifts_df.get("trimp"), errors="coerce")
                    shifts_df["ascentM"] = pd.to_numeric(shifts_df.get("ascentM"), errors="coerce")
                    with pd.option_context("mode.use_inf_as_na", True):
                        shifts_df["speedKmh"] = (
                            shifts_df["distanceKm"] / shifts_df["timeSec"].replace(0, pd.NA) * 3600.0
                        )
                        shifts_df["speedEqKmh"] = (
                            shifts_df["distanceEqKm"] / shifts_df["timeSec"].replace(0, pd.NA) * 3600.0
                        )
                    shifts_df["category"] = (
                        shifts_df.get("category", pd.Series("OTHER", index=shifts_df.index))
                        .astype(str)
                        .str.upper()
                    )

                    acts_info_path = storage.base_dir / "activities.csv"
                    acts_info = pd.DataFrame()
                    if acts_info_path.exists():
                        acts_info = (
                            pd.read_csv(acts_info_path, usecols=["activityId", "name"])
                            if "activityId" in pd.read_csv(acts_info_path, nrows=0).columns
                            else pd.DataFrame()
                        )
                    if not acts_info.empty:
                        shifts_df["activityId"] = shifts_df["activityId"].astype(str)
                        acts_info["activityId"] = acts_info["activityId"].astype(str)
                        shifts_df = shifts_df.merge(acts_info, on="activityId", how="left")
                    shifts_df["name"] = shifts_df.get("name", pd.Series(index=shifts_df.index)).fillna("")
                    shifts_df["activityLabel"] = shifts_df.apply(
                        lambda row: row["name"] if str(row["name"]).strip() else str(row["activityId"]),
                        axis=1,
                    )
                    shifts_df["dateLabel"] = shifts_df["date"].dt.strftime("%Y-%m-%d")
                    shifts_df = shifts_df.sort_values(["date", "activityId"]).reset_index(drop=True)

                    line = (
                        alt.Chart(shifts_df)
                        .mark_line(color="#1d4ed8", strokeWidth=2)
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("hrSpeedShift:Q", title="Décalage HR (échantillons)"),
                        )
                    )
                    points = (
                        alt.Chart(shifts_df)
                        .mark_circle(size=70, opacity=0.9)
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("hrSpeedShift:Q", title="Décalage HR (échantillons)"),
                            color=alt.Color("category:N", title="Type"),
                            tooltip=[
                                alt.Tooltip("activityLabel:N", title="Activité"),
                                alt.Tooltip("dateLabel:N", title="Date"),
                                alt.Tooltip("category:N", title="Type"),
                                alt.Tooltip("hrSpeedShift:Q", title="HR shift", format=".0f"),
                                alt.Tooltip("avgHr:Q", title="FC moy (bpm)", format=".1f"),
                                alt.Tooltip("speedKmh:Q", title="Vitesse (km/h)", format=".2f"),
                                alt.Tooltip("speedEqKmh:Q", title="Vitesse eq (km/h)", format=".2f"),
                                alt.Tooltip("distanceKm:Q", title="Distance (km)", format=".2f"),
                                alt.Tooltip("distanceEqKm:Q", title="Dist. eq (km)", format=".2f"),
                                alt.Tooltip("timeSec:Q", title="Temps (s)", format=".0f"),
                                alt.Tooltip("ascentM:Q", title="D+ (m)", format=".0f"),
                                alt.Tooltip("trimp:Q", title="TRIMP", format=".1f"),
                            ],
                        )
                    )
                    st.altair_chart(
                        alt.layer(line, points).properties(height=340, width=CHART_WIDTH),
                        use_container_width=True,
                    )

# --- Max Speed Profile ---
with tab_speed_profile:
    st.subheader("Profil de vitesse max")

    @st.cache_data(ttl=3600)
    def _cached_load_speed_profile(
        athlete_id: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        categories: List[str],
    ) -> pd.DataFrame:
        """Cached wrapper for load_aggregated_speed_profile."""
        return load_aggregated_speed_profile(
            storage, speed_profile_service, cfg, athlete_id, start_date, end_date, categories
        )

    profile_df = _cached_load_speed_profile(athlete_id, start_date, end_date, selected_cats)

    if profile_df.empty:
        st.info("Aucune donnée de profil de vitesse disponible pour la période sélectionnée.")
    else:
        window_min = float(profile_df["windowSec"].min())
        window_max = float(profile_df["windowSec"].max())
        domain_min = max(1.0, window_min - 1.0)
        chart = create_speed_profile_chart(
            profile_df, CHART_WIDTH, x_domain=(domain_min, window_max)
        )
        st.altair_chart(chart, use_container_width=True)

# --- Max Speed Profile Cloud ---
with tab_speed_profile_cloud:
    st.subheader("Nuage de vitesse max")
    show_speed_eq = st.checkbox(
        "Afficher la vitesse équivalente",
        value=False,
        help="Décochez pour afficher la vitesse brute.",
    )

    @st.cache_data(ttl=3600)
    def _cached_load_speed_profile_cloud(
        athlete_id: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        categories: List[str],
    ) -> pd.DataFrame:
        """Cached wrapper for load_speed_profile_cloud."""
        return load_speed_profile_cloud(
            storage, speed_profile_service, cfg, athlete_id, start_date, end_date, categories
        )

    cloud_df = _cached_load_speed_profile_cloud(athlete_id, start_date, end_date, selected_cats)

    if cloud_df.empty:
        st.info("Aucune donnée de nuage de vitesse disponible pour la période sélectionnée.")
    else:
        window_options = sorted(cloud_df["windowSec"].dropna().unique().tolist())
        if not window_options:
            st.info("Aucune fenêtre disponible pour le nuage de vitesse.")
            st.stop()

        def _format_window_label(window_sec: float) -> str:
            if window_sec >= 3600:
                return f"{window_sec / 3600:.1f} h"
            if window_sec >= 60:
                return f"{window_sec / 60:.0f} min"
            return f"{window_sec:.0f} s"

        min_window, max_window = st.select_slider(
            "Fenêtres",
            options=window_options,
            value=(window_options[0], window_options[-1]),
            format_func=_format_window_label,
        )
        cloud_df = cloud_df[
            (cloud_df["windowSec"] >= min_window) & (cloud_df["windowSec"] <= max_window)
        ]
        if cloud_df.empty:
            st.info("Aucune donnée dans la plage de fenêtres sélectionnée.")
            st.stop()

        speed_type = "eq" if show_speed_eq else "raw"
        domain_min = max(1.0, float(min_window) - 1.0)
        chart = create_speed_profile_cloud_chart(
            cloud_df, CHART_WIDTH, speed_type=speed_type, x_domain=(domain_min, max_window)
        )
        st.altair_chart(chart, use_container_width=True)

with tab_hr_borders:
    st.subheader("Evolution des frontières de zones HR")
    acts_path = storage.base_dir / "activities_metrics.csv"
    metrics_df = pd.read_csv(acts_path) if acts_path.exists() else pd.DataFrame()
    if metrics_df.empty or "athleteId" not in metrics_df.columns:
        st.info("Aucune donnée de frontière de zone HR disponible.")
    else:
        metrics_df = metrics_df[metrics_df["athleteId"].astype(str) == str(athlete_id)].copy()
        if metrics_df.empty or "startDate" not in metrics_df.columns:
            st.info("Aucune donnée de frontière de zone HR disponible.")
        else:
            metrics_df["startDate"] = pd.to_datetime(metrics_df["startDate"], errors="coerce")
            metrics_df = metrics_df.dropna(subset=["startDate"])
            metrics_df = metrics_df[
                (metrics_df["startDate"].dt.date >= start_date)
                & (metrics_df["startDate"].dt.date <= end_date)
            ]
            if selected_cats and "category" in metrics_df.columns:
                allowed = {str(cat).upper() for cat in selected_cats}
                metrics_df["category"] = metrics_df["category"].astype(str).str.upper()
                metrics_df = metrics_df[metrics_df["category"].isin(allowed)]
            border_cols = [
                col
                for col in metrics_df.columns
                if col.startswith("hrZone_z") and col.endswith("_upper")
            ]
            if metrics_df.empty or not border_cols:
                st.info("Aucune frontière HR calculée sur la période sélectionnée.")
            else:
                input_df = metrics_df[["startDate", *border_cols]].copy()
                input_df["startDate"] = input_df["startDate"].dt.strftime("%Y-%m-%d")
                chart = render_zone_borders_chart(input_df, chart_width=CHART_WIDTH)
                if chart is None:
                    st.info("Aucune frontière HR calculée sur la période sélectionnée.")
                else:
                    st.altair_chart(chart, use_container_width=True)

with tab_zone_speed:
    st.subheader("Evolution de la vitesse par zone HR")
    metric_label = st.radio(
        "Métrique",
        options=["Vitesse", "Vitesse équivalente"],
        index=0,
        horizontal=True,
        key="dashboard_zone_speed_metric",
    )
    view_label = st.radio(
        "Vue",
        options=["Agrégation hebdomadaire", "Par activité"],
        index=0,
        horizontal=True,
        key="dashboard_zone_speed_view",
    )
    metric_key = "speed" if metric_label == "Vitesse" else "speedeq"

    if view_label == "Agrégation hebdomadaire":
        evolution_df = hr_zones_service.build_zone_speed_evolution(
            athlete_id=athlete_id,
            start_date=start_date,
            end_date=end_date,
            categories=selected_cats,
        )
        chart = render_zone_speed_evolution(
            evolution_df,
            metric=metric_key,
            chart_width=CHART_WIDTH,
        )
        if chart is None:
            st.info("Aucune donnée de vitesse par zone disponible sur la période.")
        else:
            st.altair_chart(chart, use_container_width=True)
    else:
        points_df = hr_zones_service.build_activity_zone_speed_points(
            athlete_id=athlete_id,
            start_date=start_date,
            end_date=end_date,
            categories=selected_cats,
        )
        field = "avg_speed_kmh" if metric_key == "speed" else "avg_speedeq_kmh"
        y_title = "Vitesse (km/h)" if metric_key == "speed" else "Vitesse équivalente (km/h)"
        if points_df.empty or field not in points_df.columns:
            st.info("Aucune donnée de vitesse par zone disponible sur la période.")
        else:
            points_df = points_df.copy()
            points_df["date"] = pd.to_datetime(points_df["date"], errors="coerce")
            points_df[field] = pd.to_numeric(points_df[field], errors="coerce")
            points_df = points_df.dropna(subset=["date", "zone_label", field]).sort_values("date")
            if points_df.empty:
                st.info("Aucune donnée de vitesse par zone disponible sur la période.")
            else:
                zones = points_df["zone_label"].astype(str).unique().tolist()
                zone_domain = sorted(zones, key=lambda label: int(label[1:]))
                zone_colors = build_zone_colors(len(zone_domain))
                chart = (
                    alt.Chart(points_df)
                    .mark_circle(size=65, opacity=0.85)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y(f"{field}:Q", title=y_title),
                        color=alt.Color(
                            "zone_label:N",
                            title="Zone",
                            scale=alt.Scale(domain=zone_domain, range=zone_colors),
                        ),
                        tooltip=[
                            alt.Tooltip("date:T", title="Date"),
                            alt.Tooltip("activityId:N", title="Activité"),
                            alt.Tooltip("zone_label:N", title="Zone"),
                            alt.Tooltip(f"{field}:Q", title=y_title, format=".2f"),
                            alt.Tooltip("time_seconds:Q", title="Temps (s)", format=".0f"),
                        ],
                    )
                    .properties(height=320, width=CHART_WIDTH)
                )
                st.altair_chart(chart, use_container_width=True)
