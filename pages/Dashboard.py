from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional

from config import METRICS as CONFIG_METRICS
from utils.config import load_config
from utils.formatting import set_locale
from utils.styling import apply_theme
from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo
from services.analytics_service import AnalyticsService
from services.speed_profile_service import SpeedProfileService
from services.timeseries_service import TimeseriesService


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
timeseries_service = TimeseriesService(cfg)


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
        (fill + chronic_line + acute_line)
        .encode(
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip(planned_col, title="Charge chronique", format=".2f"),
                alt.Tooltip(acute_col, title="Charge aiguë", format=".2f"),
            ]
        )
        .properties(height=340)
    )


athlete_id = _select_athlete()
if not athlete_id:
    st.stop()

daily_metrics = _load_daily_metrics(athlete_id)
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
cat_options = ["RUN", "TRAIL_RUN", "HIKE", "RIDE"]
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
        chart = _training_load_chart(
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
            chart = _training_load_chart(
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
                chart = _training_load_chart(
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
        .properties(height=360)
    )
    st.altair_chart(cloud_chart, use_container_width=True)

# --- HR vs Speed Graph (Cluster-based) ---
with tab_hr_speed:
    st.subheader("Relation FC vs Vitesse")

    @st.cache_data(ttl=3600)
    def _load_hr_speed_data(athlete_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp, categories: List[str], min_cluster_percent: float = 5.0) -> Tuple[pd.DataFrame, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Load and preprocess HR vs Speed data for all activities in date range.
        
        Returns cluster centers with their standard deviations instead of all data points.
        
        Args:
            min_cluster_percent: Minimum percentage of total points a cluster must contain to be included (default: 5.0%)
        """
        # Get activities in date range
        acts_df = storage.read_csv("activities_metrics.csv")
        if acts_df.empty:
            return pd.DataFrame(), None, None, None, None

        acts_df = acts_df[acts_df.get("athleteId") == athlete_id].copy()
        acts_df["date"] = pd.to_datetime(acts_df.get("startDate"), errors="coerce").dt.normalize()
        mask = (acts_df["date"] >= pd.Timestamp(start_date)) & (acts_df["date"] <= pd.Timestamp(end_date))
        acts_df = acts_df[mask]

        # Apply category filter
        if categories:
            acts_df["category"] = acts_df.get("category", pd.Series(dtype=str)).astype(str).str.upper()
            acts_df = acts_df[acts_df["category"].isin([c.upper() for c in categories])]

        # Get activities info to check for timeseries
        activities_info = storage.read_csv("activities.csv")
        if not activities_info.empty:
            activities_info = activities_info[activities_info.get("athleteId") == athlete_id].copy()
            activities_info["activityId"] = activities_info["activityId"].astype(str)
            acts_df["activityId"] = acts_df["activityId"].astype(str)
            acts_df = acts_df.merge(
                activities_info[["activityId", "hasTimeseries"]],
                on="activityId",
                how="left",
            )
            # Filter to activities with timeseries
            acts_df = acts_df[acts_df.get("hasTimeseries", pd.Series(dtype=bool)) == True]
        else:
            # If no activities info, check if timeseries file exists
            acts_df = acts_df.copy()
            acts_df["activityId"] = acts_df["activityId"].astype(str)
            acts_df["hasTimeseries"] = acts_df["activityId"].apply(
                lambda aid: (cfg.timeseries_dir / f"{aid}.csv").exists()
            )
            acts_df = acts_df[acts_df["hasTimeseries"] == True]

        if acts_df.empty:
            return pd.DataFrame(), None, None, None, None

        # Collect all HR and Speed data
        all_data = []

        for _, row in acts_df.iterrows():
            activity_id = str(row.get("activityId", ""))
            if not activity_id:
                continue

            # Try to load precomputed metrics_ts first
            metrics_ts = speed_profile_service.load_metrics_ts(activity_id)
            if metrics_ts is None or metrics_ts.empty:
                # Process timeseries if not precomputed
                result = speed_profile_service.process_timeseries(activity_id, strategy="cluster")
                if result is None:
                    continue
                # Save for future use
                speed_profile_service.save_metrics_ts(activity_id, result)
                # Extract data from result
                if result.hr_shifted is not None and result.speed_smooth is not None:
                    df_act = pd.DataFrame({
                        "hr": result.hr_shifted.values,
                        "speed": result.speed_smooth.values,
                        "activityId": activity_id,
                    })
                    if result.clusters is not None and len(result.clusters) == len(df_act):
                        df_act["cluster"] = result.clusters
                    all_data.append(df_act)
            else:
                # Use precomputed data
                if "hr_shifted" in metrics_ts.columns and "speed_smooth" in metrics_ts.columns:
                    df_act = pd.DataFrame({
                        "hr": metrics_ts["hr_shifted"].values,
                        "speed": metrics_ts["speed_smooth"].values,
                        "activityId": activity_id,
                    })
                    if "cluster" in metrics_ts.columns and len(metrics_ts["cluster"]) == len(df_act):
                        df_act["cluster"] = metrics_ts["cluster"].values
                    all_data.append(df_act)

        if not all_data:
            return pd.DataFrame(), None, None, None, None

        # Compute cluster centers for each activity separately
        n_clusters = cfg.n_cluster
        cluster_centers_list = []
        
        for df_act in all_data:
            activity_id = df_act["activityId"].iloc[0]
            df_act = df_act.dropna(subset=["hr", "speed"])
            
            if df_act.empty:
                continue
            
            total_points = len(df_act)
            min_points_per_cluster = max(1, int(total_points * min_cluster_percent / 100.0))
            min_allowed_points = max(min_points_per_cluster, 20)  # At least 20 points or percentage threshold
            
            # Prepare data for clustering (use individual activity's data)
            X = df_act[["hr", "speed"]].values
            
            if len(X) < n_clusters:
                continue
            
            # Perform KMeans clustering for this activity
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            df_act["cluster"] = clusters
            
            # Compute cluster centers and standard deviations for this activity
            # Filter out clusters with less than min_cluster_percent of total points or less than 20 points
            for cluster_id in range(n_clusters):
                cluster_data = df_act[df_act["cluster"] == cluster_id]
                cluster_count = len(cluster_data)
                
                # Skip clusters that are too small (outliers)
                if cluster_count < min_allowed_points:
                    continue
                
                if cluster_count > 0:
                    hr_mean = cluster_data["hr"].mean()
                    speed_mean = cluster_data["speed"].mean()
                    
                    # Skip clusters with mean speed less than 6 km/h
                    if speed_mean < 6:
                        continue
                    
                    hr_std = cluster_data["hr"].std()
                    speed_std = cluster_data["speed"].std()
                    
                    cluster_centers_list.append({
                        "hr": hr_mean,
                        "speed": speed_mean,
                        "hr_std": hr_std if not pd.isna(hr_std) else 0.0,
                        "speed_std": speed_std if not pd.isna(speed_std) else 0.0,
                        "cluster": cluster_id,
                        "activityId": activity_id,
                        "count": cluster_count,
                    })
        
        if not cluster_centers_list:
            return pd.DataFrame(), None, None, None, None
        
        centers_df = pd.DataFrame(cluster_centers_list)
        
        # Load activity names and dates from activities.csv
        activities_info = storage.read_csv("activities.csv")
        if not activities_info.empty:
            activities_info = activities_info[activities_info.get("athleteId") == athlete_id].copy()
            activities_info["activityId"] = activities_info["activityId"].astype(str)
            centers_df["activityId"] = centers_df["activityId"].astype(str)
            
            # Merge name and startTime
            merge_cols = ["activityId"]
            if "name" in activities_info.columns:
                merge_cols.append("name")
            if "startTime" in activities_info.columns:
                merge_cols.append("startTime")
            
            centers_df = centers_df.merge(
                activities_info[merge_cols],
                on="activityId",
                how="left",
            )
            
            # Format date for display
            if "startTime" in centers_df.columns:
                centers_df["activity_date"] = pd.to_datetime(centers_df["startTime"], errors="coerce")
                centers_df["activity_date_str"] = centers_df["activity_date"].dt.strftime("%Y-%m-%d")
            else:
                centers_df["activity_date_str"] = ""
            
            # Ensure name exists
            if "name" not in centers_df.columns:
                centers_df["name"] = ""
            centers_df["name"] = centers_df["name"].fillna("").astype(str)
            
            # Ensure date_str exists
            if "activity_date_str" not in centers_df.columns:
                centers_df["activity_date_str"] = ""
            centers_df["activity_date_str"] = centers_df["activity_date_str"].fillna("").astype(str)
        else:
            # If no activities info, create empty columns
            centers_df["name"] = ""
            centers_df["activity_date_str"] = ""
        
        # Compute weighted linear regression on all cluster centers
        # Using speed as x (independent) and hr as y (dependent)
        # Weights are inversely proportional to the standard deviation
        x_values = centers_df["speed"].values
        y_values = centers_df["hr"].values
        
        # Compute weights: inverse of speed_std (uncertainty in the independent variable)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        speed_std_values = centers_df["speed_std"].values
        
        # Weight = 1 / (speed_std + epsilon) - clusters with larger std have less weight
        weights = 1.0 / (speed_std_values + epsilon)

        if len(x_values) < 2:
            return centers_df, None, None, None, None

        # Weighted least squares: y = slope * x + intercept
        # Build design matrix for [x, 1]
        A = np.vstack([x_values, np.ones(len(x_values))]).T
        
        # Apply weights
        W = np.diag(weights)
        A_weighted = np.sqrt(W) @ A
        y_weighted = np.sqrt(W) @ y_values
        
        # Solve weighted least squares: (A^T * W * A) * params = A^T * W * y
        params, residuals, rank, s = np.linalg.lstsq(A_weighted, y_weighted, rcond=None)
        slope, intercept = params
        
        # Calculate R-squared for weighted regression
        y_pred = slope * x_values + intercept
        ss_res = np.sum(weights * (y_values - y_pred)**2)
        ss_tot = np.sum(weights * (y_values - np.average(y_values, weights=weights))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Calculate weighted standard error
        if len(x_values) > 2:
            dof = len(x_values) - 2  # degrees of freedom
            mse = ss_res / dof if dof > 0 else 0.0
            std_err = np.sqrt(mse)
        else:
            std_err = 0.0

        return centers_df, slope, intercept, r_squared, std_err

    # Load data
    hr_speed_df, slope, intercept, r_squared, std_err = _load_hr_speed_data(
        athlete_id, start_date, end_date, selected_cats
    )

    if hr_speed_df.empty:
        st.info("Aucune donnée de timeseries disponible pour la période sélectionnée.")
    else:
        # Create scatter plot of cluster centers with error bars
        chart_df = hr_speed_df.copy()

        # Prepare data for error bars (uncertainty based on std)
        chart_df["hr_upper"] = chart_df["hr"] + chart_df["hr_std"]
        chart_df["hr_lower"] = chart_df["hr"] - chart_df["hr_std"]
        chart_df["speed_upper"] = chart_df["speed"] + chart_df["speed_std"]
        chart_df["speed_lower"] = chart_df["speed"] - chart_df["speed_std"]

        # Color by cluster
        color_encoding = alt.Color("cluster:O", scale=alt.Scale(scheme="viridis"), title="Cluster")

        # Plot cluster centers (HR on y-axis, Speed on x-axis)
        centers = (
            alt.Chart(chart_df)
            .mark_circle(size=100, opacity=0.8, stroke="black", strokeWidth=2)
            .encode(
                x=alt.X("speed:Q", title="Vitesse (km/h)", scale=alt.Scale(domain=[4, 24])),
                y=alt.Y("hr:Q", title="Fréquence cardiaque (bpm)", scale=alt.Scale(domain=[80, 210])),
                color=color_encoding,
                tooltip=[
                    alt.Tooltip("hr:Q", title="FC (bpm)", format=".1f"),
                    alt.Tooltip("speed:Q", title="Vitesse (km/h)", format=".1f"),
                    alt.Tooltip("hr_std:Q", title="FC std (bpm)", format=".1f"),
                    alt.Tooltip("speed_std:Q", title="Vitesse std (km/h)", format=".1f"),
                    alt.Tooltip("cluster:O", title="Cluster"),
                    alt.Tooltip("name:N", title="Activité"),
                    alt.Tooltip("activity_date_str:N", title="Date"),
                    alt.Tooltip("count:Q", title="Nombre de points"),
                ],
            )
        )

        # Add horizontal error bars (uncertainty in Speed, now on x-axis)
        speed_error_bars = (
            alt.Chart(chart_df)
            .mark_rule(strokeWidth=2, opacity=0.6)
            .encode(
                x=alt.X("speed_lower:Q", title="Vitesse (km/h)", scale=alt.Scale(domain=[4, 24])),
                x2="speed_upper:Q",
                y=alt.Y("hr:Q", title="Fréquence cardiaque (bpm)", scale=alt.Scale(domain=[80, 210])),
                color=color_encoding,
            )
        )

        # Add vertical error bars (uncertainty in HR, now on y-axis)
        hr_error_bars = (
            alt.Chart(chart_df)
            .mark_rule(strokeWidth=2, opacity=0.6)
            .encode(
                x=alt.X("speed:Q", title="Vitesse (km/h)", scale=alt.Scale(domain=[4, 24])),
                y=alt.Y("hr_lower:Q", title="Fréquence cardiaque (bpm)", scale=alt.Scale(domain=[80, 210])),
                y2="hr_upper:Q",
                color=color_encoding,
            )
        )

        # Add regression line if available
        if slope is not None and intercept is not None:
            # Regression: hr = slope * speed + intercept
            speed_range = [4, 24]  # Use fixed range
            regression_data = pd.DataFrame({
                "speed": speed_range,
                "hr": [slope * speed + intercept for speed in speed_range],
            })

            regression_line = (
                alt.Chart(regression_data)
                .mark_line(color="red", strokeWidth=2)
                .encode(
                    x=alt.X("speed:Q", scale=alt.Scale(domain=[4, 24])),
                    y=alt.Y("hr:Q", scale=alt.Scale(domain=[80, 210]))
                )
            )

            # Add confidence interval
            if std_err is not None:
                regression_data["upper"] = regression_data["hr"] + std_err
                regression_data["lower"] = regression_data["hr"] - std_err

                confidence_band = (
                    alt.Chart(regression_data)
                    .mark_area(opacity=0.2, color="red")
                    .encode(
                        x=alt.X("speed:Q", scale=alt.Scale(domain=[4, 24])),
                        y=alt.Y("upper:Q", scale=alt.Scale(domain=[80, 210])),
                        y2="lower:Q",
                    )
                )

                chart = alt.layer(confidence_band, speed_error_bars, hr_error_bars, centers, regression_line)
            else:
                chart = alt.layer(speed_error_bars, hr_error_bars, centers, regression_line)

            formula_text = f"FC = {slope:.2f} × Vitesse + {intercept:.2f}\nR² = {r_squared:.3f}"
            st.caption(formula_text)
        else:
            chart = alt.layer(speed_error_bars, hr_error_bars, centers)

        chart = chart.properties(height=500, width=CHART_WIDTH)
        st.altair_chart(chart, use_container_width=True)
