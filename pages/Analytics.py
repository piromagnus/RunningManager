"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from graph.analytics import create_daily_bar_chart, create_weekly_bar_chart
from graph.hr_zones import build_zone_colors
from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo, SettingsRepo
from services.analytics_service import AnalyticsService
from services.hr_zones_service import HrZonesService
from services.speed_profile_service import SpeedProfileService
from services.timeseries_service import TimeseriesService
from utils.config import load_config
from utils.constants import (
    CATEGORY_LABELS_FR,
    CHART_WIDTH_DEFAULT,
    METRIC_CONFIG,
)
from utils.constants import (
    METRICS as CONFIG_METRICS,
)
from utils.formatting import fmt_decimal, set_locale
from utils.styling import apply_theme
from widgets.athlete_selector import select_athlete

st.set_page_config(page_title="Running Manager - Analytics", layout="wide")
apply_theme()
st.title("Analytics")

CHART_WIDTH = CHART_WIDTH_DEFAULT

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
ath_repo = AthletesRepo(storage)
settings_repo = SettingsRepo(storage)
analytics = AnalyticsService(storage)
settings_values = settings_repo.get("coach-1") or {}

CATEGORY_OPTIONS = CATEGORY_LABELS_FR


def _to_int_setting(
    raw: object,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    parsed = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        value = default
    else:
        try:
            value = int(parsed)
        except Exception:
            value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


zone_count_setting = _to_int_setting(
    settings_values.get("hrZoneCount"),
    default=5,
    minimum=2,
    maximum=5,
)
zone_window_days_setting = _to_int_setting(
    settings_values.get("hrZoneWindowDays"),
    default=90,
    minimum=1,
)
ts_service = TimeseriesService(cfg)
speed_profile_service = SpeedProfileService(cfg)
hr_zones_service = HrZonesService(
    storage=storage,
    ts_service=ts_service,
    speed_profile_service=speed_profile_service,
    zone_count=zone_count_setting,
    window_days=zone_window_days_setting,
)


def _resolve_metric_config() -> dict[str, dict]:
    resolved = {}
    for key, cfg in METRIC_CONFIG.items():
        transform = cfg.get("transform")
        resolved_transform = analytics.seconds_to_hours if transform == "seconds_to_hours" else None
        resolved[key] = {**cfg, "transform": resolved_transform}
    return resolved


METRIC_DEFINITIONS = _resolve_metric_config()

def _load_saved_activity_types() -> list[str]:
    """Load allowed activity categories for analytics from settings.

    Be lenient with persisted values: the column may contain JSON string,
    an actual list, or NaN/float due to CSV typing. Only return valid keys.
    """
    settings = settings_values
    raw = settings.get("analyticsActivityTypes")

    # Already a list (robustness if another writer stored native list)
    if isinstance(raw, list):
        return [v for v in raw if v in CATEGORY_OPTIONS]

    # If it's a string-like, try JSON decoding
    if isinstance(raw, (str, bytes)):
        text = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
        text = text.strip()
        if not text:
            return []
        try:
            values = json.loads(text)
            if isinstance(values, list):
                return [v for v in values if v in CATEGORY_OPTIONS]
        except (json.JSONDecodeError, TypeError, ValueError):
            return []

    # Any other type (including float/NaN) → return empty selection
    return []


def _build_session_zone_distribution(
    activities_df: pd.DataFrame,
) -> pd.DataFrame:
    if activities_df.empty or "activityId" not in activities_df.columns:
        return pd.DataFrame()

    working = activities_df.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
    else:
        working["date"] = pd.to_datetime(working.get("startDate"), errors="coerce")
    working = working.dropna(subset=["activityId", "date"]).sort_values("date")
    if working.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for activity in working.itertuples(index=False):
        activity_id = str(activity.activityId)
        summary = hr_zones_service.load_zone_summary(activity_id)
        if summary is None or summary.empty:
            continue

        activity_date = pd.to_datetime(getattr(activity, "date", None), errors="coerce")
        if pd.isna(activity_date):
            continue
        date_label = activity_date.strftime("%Y-%m-%d")
        session_label = f"{date_label} · {activity_id}"
        for zone_row in summary.itertuples(index=False):
            zone = int(zone_row.zone)
            time_seconds_raw = pd.to_numeric(pd.Series([zone_row.time_seconds]), errors="coerce").iloc[0]
            time_seconds = float(time_seconds_raw) if pd.notna(time_seconds_raw) else 0.0
            rows.append(
                {
                    "sessionDate": activity_date.normalize(),
                    "sessionLabel": session_label,
                    "zone": zone,
                    "zone_label": f"Z{zone}",
                    "time_seconds": time_seconds,
                }
            )

    if not rows:
        return pd.DataFrame()

    session_df = pd.DataFrame(rows).sort_values(["sessionDate", "zone"]).reset_index(drop=True)
    session_df["time_minutes"] = session_df["time_seconds"] / 60.0
    total_seconds = session_df.groupby("sessionLabel")["time_seconds"].transform("sum")
    session_df["pct_time"] = 0.0
    valid_total_mask = total_seconds > 0
    session_df.loc[valid_total_mask, "pct_time"] = (
        session_df.loc[valid_total_mask, "time_seconds"] / total_seconds[valid_total_mask] * 100.0
    )
    return session_df


def _render_hr_zone_stacked_chart(
    zone_df: pd.DataFrame,
    *,
    x_col: str,
    x_title: str,
    absolute: bool,
) -> alt.Chart | None:
    if zone_df.empty:
        return None

    working = zone_df.copy()
    if "zone_label" not in working.columns and "zone" in working.columns:
        working["zone"] = pd.to_numeric(working["zone"], errors="coerce")
        working["zone_label"] = working["zone"].map(
            lambda value: f"Z{int(value)}" if pd.notna(value) else None
        )
    if "zone_label" not in working.columns:
        return None

    value_col = "time_minutes" if absolute else "pct_time"
    if value_col not in working.columns:
        return None
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce").fillna(0.0)
    working = working.dropna(subset=[x_col, "zone_label"])
    if working.empty:
        return None

    labels = working["zone_label"].dropna().astype(str).unique().tolist()
    zone_domain = sorted(labels, key=lambda label: int(label[1:]) if label.startswith("Z") else 999)
    zone_range = build_zone_colors(len(zone_domain))

    y_title = "Temps (min)" if absolute else "% du temps"
    value_format = ".1f" if absolute else ".2f"
    return (
        alt.Chart(working)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:N", title=x_title),
            y=alt.Y(f"{value_col}:Q", title=y_title),
            color=alt.Color(
                "zone_label:N",
                title="Zone",
                scale=alt.Scale(domain=zone_domain, range=zone_range),
            ),
            order=alt.Order("zone:Q"),
            tooltip=[
                alt.Tooltip(f"{x_col}:N", title=x_title),
                alt.Tooltip("zone_label:N", title="Zone"),
                alt.Tooltip(f"{value_col}:Q", title=y_title, format=value_format),
                alt.Tooltip("time_minutes:Q", title="Temps (min)", format=".1f"),
                alt.Tooltip("pct_time:Q", title="% du temps", format=".2f"),
            ],
        )
        .properties(height=320, width=CHART_WIDTH)
    )


athlete_id = select_athlete(ath_repo)
if not athlete_id:
    st.stop()

with st.expander("Filtrer les activités incluses"):
    default_types = _load_saved_activity_types() or [
        key for key in CATEGORY_OPTIONS.keys() if key != "RIDE"
    ]
    selected_types = st.multiselect(
        "Inclure les types d'activités suivants",
        options=list(CATEGORY_OPTIONS.keys()),
        default=default_types,
        format_func=lambda key: CATEGORY_OPTIONS[key],
    )
    if not selected_types:
        st.warning(
            "Au moins un type doit être sélectionné. "
            "Toutes les activités Run/Trail/Hike seront incluses.",
        )
        selected_types = list(CATEGORY_OPTIONS.keys())
    if st.button("Enregistrer ces types pour l'analytics"):
        settings_repo.update("coach-1", {"analyticsActivityTypes": json.dumps(selected_types)})
        st.success("Préférence sauvegardée.")

# --- Date range controls ---
today = pd.Timestamp.today().normalize().date()
min_act, max_act = analytics.activity_date_bounds(athlete_id)
min_plan, max_plan = analytics.planned_date_bounds(athlete_id)

min_candidates = [d for d in (min_act, min_plan) if d]
max_candidates = [d for d in (max_act, max_plan) if d]

min_date = min(min_candidates) if min_candidates else today
max_date = max(max_candidates) if max_candidates else today

default_end = max_date
# Default to last 28 days within available bounds
default_start = max(min_date, (default_end - dt.timedelta(days=90)))

# Maintain selection in session state
if "analytics_range" not in st.session_state:
    st.session_state["analytics_range"] = (
        pd.Timestamp(default_start).to_pydatetime(),
        pd.Timestamp(default_end).to_pydatetime(),
    )

col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
if col_btn1.button("7 jours"):
    st.session_state["analytics_range"] = (
        (pd.Timestamp(max_date) - pd.Timedelta(days=7)).to_pydatetime(),
        pd.Timestamp(max_date).to_pydatetime(),
    )
if col_btn2.button("28 jours"):
    st.session_state["analytics_range"] = (
        (pd.Timestamp(max_date) - pd.Timedelta(days=28)).to_pydatetime(),
        pd.Timestamp(max_date).to_pydatetime(),
    )
if col_btn3.button("3 mois"):
    st.session_state["analytics_range"] = (
        (pd.Timestamp(max_date) - pd.DateOffset(months=3)).to_pydatetime(),
        pd.Timestamp(max_date).to_pydatetime(),
    )
if col_btn4.button("1 an"):
    st.session_state["analytics_range"] = (
        (pd.Timestamp(max_date) - pd.DateOffset(years=1)).to_pydatetime(),
        pd.Timestamp(max_date).to_pydatetime(),
    )

start_dt, end_dt = st.slider(
    "Période",
    min_value=pd.Timestamp(min_date).to_pydatetime(),
    max_value=pd.Timestamp(max_date).to_pydatetime(),
    value=(
        pd.to_datetime(st.session_state["analytics_range"][0]).to_pydatetime(),
        pd.to_datetime(st.session_state["analytics_range"][1]).to_pydatetime(),
    ),
    format="YYYY-MM-DD",
)
st.session_state["analytics_range"] = (start_dt, end_dt)

start_date = pd.Timestamp(start_dt).date()
end_date = pd.Timestamp(end_dt).date()
plan_range_end = end_date
if max_plan and max_plan > plan_range_end:
    plan_range_end = max_plan

metric_label_default = CONFIG_METRICS.index("DistEq") if "DistEq" in CONFIG_METRICS else 0
metric_label = st.selectbox("Métrique", CONFIG_METRICS, index=metric_label_default)
metric_cfg = METRIC_DEFINITIONS[metric_label]

# Precompute daily range dataframe (full days with zeros) for fallback/derivations
value_column_map = {
    "Time": "timeSec",
    "Distance": "distanceKm",
    "DistEq": "distanceEqKm",
    "Trimp": "trimp",
}
daily_df = analytics.daily_range(
    athlete_id=athlete_id,
    metric_label=metric_label,
    selected_types=selected_types,
    start_date=start_date,
    end_date=plan_range_end,
)
if metric_cfg["transform"]:
    daily_df["planned_metric"] = daily_df["planned_value"].apply(metric_cfg["transform"])  # type: ignore[index]
    daily_df["actual_metric"] = daily_df["actual_value"].apply(metric_cfg["transform"])  # type: ignore[index]
else:
    daily_df["planned_metric"] = daily_df["planned_value"]
    daily_df["actual_metric"] = daily_df["actual_value"]

df = analytics.load_weekly_metrics(athlete_id=athlete_id)
if df.empty:
    st.info("Aucune métrique hebdomadaire disponible.")
    st.stop()

working_df = df.copy()
weekly_df = analytics.weekly_range(
    athlete_id=athlete_id,
    metric_label=metric_label,
    selected_types=selected_types,
    start_date=start_date,
    end_date=plan_range_end,
)
weeks_df = weekly_df[["weekStart", "isoYear", "isoWeek"]].rename(
    columns={"weekStart": "weekStartDate"}
)
planned_raw = pd.to_numeric(weekly_df.get("planned_value"), errors="coerce").fillna(0.0)
actual_raw = pd.to_numeric(weekly_df.get("actual_value"), errors="coerce").fillna(0.0)

metrics_path = storage.base_dir / "activities_metrics.csv"
if metrics_path.exists():
    activities_metrics_raw = pd.read_csv(metrics_path)
else:
    activities_metrics_raw = pd.DataFrame()

# Initialize defaults
activities_metrics_df = pd.DataFrame()
acts_all = pd.DataFrame()
actual_agg = pd.DataFrame(columns=["isoYear", "isoWeek", "actualMetricValue"])

if not activities_metrics_raw.empty:
    # Keep a copy for link-based rescheduling (no category filter)
    acts_all = activities_metrics_raw[activities_metrics_raw["athleteId"] == athlete_id].copy()
    acts_all["date"] = pd.to_datetime(acts_all["startDate"], errors="coerce").dt.normalize()

    # Now apply category filters for charting
    activities_metrics_df = activities_metrics_raw[
        activities_metrics_raw["athleteId"] == athlete_id
    ].copy()
    activities_metrics_df["category"] = activities_metrics_df["category"].astype(str).str.upper()
    if selected_types:
        activities_metrics_df = activities_metrics_df[
            activities_metrics_df["category"].isin(selected_types)
        ]
    activities_metrics_df["date"] = pd.to_datetime(
        activities_metrics_df["startDate"], errors="coerce"
    )
    activities_metrics_df = activities_metrics_df.dropna(subset=["date"])
    # Date range filter for weekly actual aggregation
    mask_range = (activities_metrics_df["date"].dt.normalize() >= pd.Timestamp(start_date)) & (
        activities_metrics_df["date"].dt.normalize() <= pd.Timestamp(end_date)
    )
    activities_metrics_df = activities_metrics_df[mask_range]
    activities_metrics_df["isoYear"] = activities_metrics_df["date"].dt.isocalendar().year
    activities_metrics_df["isoWeek"] = activities_metrics_df["date"].dt.isocalendar().week
    actual_agg = (
        activities_metrics_df.groupby(["isoYear", "isoWeek"], as_index=False)[
            value_column_map[metric_label]
        ]
        .sum()
        .rename(columns={value_column_map[metric_label]: "actualMetricValue"})
    )
    # Ensure join keys are ints on both sides to avoid merge mismatches
    actual_agg["isoYear"] = (
        pd.to_numeric(actual_agg["isoYear"], errors="coerce").fillna(0).astype(int)
    )
    actual_agg["isoWeek"] = (
        pd.to_numeric(actual_agg["isoWeek"], errors="coerce").fillna(0).astype(int)
    )
    working_df["isoYear"] = (
        pd.to_numeric(working_df["isoYear"], errors="coerce").fillna(0).astype(int)
    )
    working_df["isoWeek"] = (
        pd.to_numeric(working_df["isoWeek"], errors="coerce").fillna(0).astype(int)
    )

# --- Dataset summary for selected athlete and filters (range-limited) ---
st.subheader("Résumé des données")
# Totals over the selected range (may be zero if no activities)
num_acts = int(activities_metrics_df.shape[0]) if not activities_metrics_df.empty else 0
total_km = (
    pd.to_numeric(activities_metrics_df.get("distanceKm"), errors="coerce").fillna(0.0).sum()
    if not activities_metrics_df.empty
    else 0.0
)
total_time_sec = (
    pd.to_numeric(activities_metrics_df.get("timeSec"), errors="coerce").fillna(0.0).sum()
    if not activities_metrics_df.empty
    else 0.0
)
total_time_h = analytics.seconds_to_hours(total_time_sec)
total_ascent_m = (
    pd.to_numeric(activities_metrics_df.get("ascentM"), errors="coerce").fillna(0.0).sum()
    if not activities_metrics_df.empty
    else 0.0
)
total_disteq_km = (
    pd.to_numeric(activities_metrics_df.get("distanceEqKm"), errors="coerce").fillna(0.0).sum()
    if not activities_metrics_df.empty
    else 0.0
)

period_label = f"{start_date.isoformat()} → {end_date.isoformat()}"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Activités sélectionnées", f"{num_acts}")
c2.metric("Distance (km)", fmt_decimal(float(total_km), 1))
c3.metric("Durée (h)", fmt_decimal(float(total_time_h), 1))
c4.metric("D+ (m)", fmt_decimal(float(total_ascent_m), 0))
c5.metric("Dist. équiv. (km)", fmt_decimal(float(total_disteq_km), 1))
st.caption(f"Période sélectionnée: {period_label}")

# Build weekly segments directly from the grid (all weeks in selected range)
if metric_cfg["transform"]:
    planned_values = planned_raw.apply(metric_cfg["transform"])  # type: ignore[index]
    actual_values = actual_raw.apply(metric_cfg["transform"])  # type: ignore[index]
else:
    planned_values = planned_raw
    actual_values = actual_raw

weekly_input = pd.DataFrame(
    {
        "athleteId": athlete_id,
        "isoYear": weeks_df["isoYear"].astype(int),
        "isoWeek": weeks_df["isoWeek"].astype(int),
        "weekLabel": pd.to_datetime(weeks_df["weekStartDate"]).dt.strftime("%Y-%m-%d"),
        "planned_metric": planned_values,
        "actual_metric": actual_values,
    }
)

stack_df = analytics.build_planned_vs_actual_segments(
    weekly_input,
    planned_column="planned_metric",
    actual_column="actual_metric",
    metric_key=metric_label.lower(),
)

# No fallback: weekly chart strictly uses weekly metrics aligned to the selected weeks grid

if stack_df.empty:
    st.info("Pas encore de données à comparer.")
    st.stop()

today_ts = pd.Timestamp.today().normalize()
stack_df["weekDate"] = pd.to_datetime(stack_df["weekLabel"], errors="coerce")
stack_df["segment_display"] = stack_df["segment"]
future_week_mask = (stack_df["segment"] == "Plan manquant") & (stack_df["weekDate"] > today_ts)
stack_df.loc[future_week_mask, "segment_display"] = "Plan à venir"

color_scale = alt.Scale(
    domain=["Réalisé", "Au-dessus du plan", "Plan manquant", "Plan à venir"],
    range=["#3b82f6", "#16a34a", "#f97316", "#facc15"],
)

weekly_actual_metrics = pd.DataFrame(
    columns=[
        "weekLabel",
        "actualTimeHours",
        "actualDistanceKm",
        "actualDistanceEqKm",
        "actualTrimp",
    ]
)
if not activities_metrics_df.empty:
    weekly_metrics = activities_metrics_df.copy()
    if not weekly_metrics.empty:
        weekly_metrics["date"] = weekly_metrics["date"].dt.normalize()
        weekly_group = weekly_metrics.groupby(["isoYear", "isoWeek"], as_index=False).agg(
            timeSec=("timeSec", "sum"),
            distanceKm=("distanceKm", "sum"),
            distanceEqKm=("distanceEqKm", "sum"),
            trimp=("trimp", "sum"),
            weekDate=("date", "min"),
        )
        weekly_group["weekLabel"] = weekly_group["weekDate"].dt.strftime("%Y-%m-%d")
        weekly_actual_metrics = weekly_group[
            ["weekLabel", "timeSec", "distanceKm", "distanceEqKm", "trimp"]
        ].rename(
            columns={
                "timeSec": "actualTimeHours",
                "distanceKm": "actualDistanceKm",
                "distanceEqKm": "actualDistanceEqKm",
                "trimp": "actualTrimp",
            }
        )
        weekly_actual_metrics["actualTimeHours"] = weekly_actual_metrics["actualTimeHours"].apply(
            lambda v: float(analytics.seconds_to_hours(float(v)))
        )
        for col in ("actualDistanceKm", "actualDistanceEqKm", "actualTrimp"):
            weekly_actual_metrics[col] = pd.to_numeric(
                weekly_actual_metrics[col], errors="coerce"
            ).fillna(0.0)

stack_with_actuals = stack_df.merge(weekly_actual_metrics, on="weekLabel", how="left")
for col in ("actualTimeHours", "actualDistanceKm", "actualDistanceEqKm", "actualTrimp"):
    stack_with_actuals[col] = pd.to_numeric(stack_with_actuals.get(col), errors="coerce").fillna(
        0.0
    )

chart = create_weekly_bar_chart(
    stack_with_actuals, metric_label, metric_cfg, color_scale, CHART_WIDTH
)
st.altair_chart(chart, use_container_width=False)

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

# --- Daily per-day planned vs actual (bar), keeping same color code ---

# Load planned metrics for daily aggregation (full dataset; filtering in service)
planned_metrics_path = storage.base_dir / "planned_metrics.csv"
if planned_metrics_path.exists():
    planned_metrics_df = pd.read_csv(planned_metrics_path)
else:
    planned_metrics_df = pd.DataFrame()


# Prepare daily planned and actual series
def _to_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


value_column = value_column_map[metric_label]

daily_df = analytics.daily_range(
    athlete_id=athlete_id,
    metric_label=metric_label,
    selected_types=selected_types,
    start_date=start_date,
    end_date=end_date,
)

# Build continuous date index to leave empty space for days without activities
# daily_df already includes all days in range

# Apply optional transform (e.g., seconds -> hours)
if metric_cfg["transform"]:
    daily_df["planned_metric"] = daily_df["planned_value"].apply(metric_cfg["transform"])  # type: ignore[index]
    daily_df["actual_metric"] = daily_df["actual_value"].apply(metric_cfg["transform"])  # type: ignore[index]
else:
    daily_df["planned_metric"] = daily_df["planned_value"]
    daily_df["actual_metric"] = daily_df["actual_value"]

# Shape for segment builder
if not daily_df.empty:
    shaped = pd.DataFrame(
        {
            "athleteId": athlete_id,
            "isoYear": daily_df["date"].dt.isocalendar().year.astype(int),
            "isoWeek": daily_df["date"].dt.isocalendar().week.astype(int),
            "weekLabel": daily_df["date"].dt.strftime("%Y-%m-%d"),
            "planned_metric": daily_df["planned_metric"],
            "actual_metric": daily_df["actual_metric"],
            "activity_names": daily_df["activity_names"].fillna(""),
        }
    )
    day_stack_df = analytics.build_planned_vs_actual_segments(
        shaped,
        planned_column="planned_metric",
        actual_column="actual_metric",
        metric_key=metric_label.lower(),
    )

    if "activity_names" in shaped.columns and not day_stack_df.empty:
        activity_names_lookup = shaped[["weekLabel", "activity_names"]].drop_duplicates()
        day_stack_df = day_stack_df.merge(activity_names_lookup, on="weekLabel", how="left")
        actual_segments = day_stack_df["segment"].isin(["Réalisé", "Au-dessus du plan"])
        day_stack_df.loc[~actual_segments, "activity_names"] = ""
        day_stack_df["activity_names"] = day_stack_df["activity_names"].fillna("")

    if not day_stack_df.empty:
        day_stack_df["date_dt"] = pd.to_datetime(day_stack_df["weekLabel"], errors="coerce")
        day_stack_df["segment_display"] = day_stack_df["segment"]
        future_day_mask = (day_stack_df["segment"] == "Plan manquant") & (
            day_stack_df["date_dt"] > today_ts
        )
        day_stack_df.loc[future_day_mask, "segment_display"] = "Plan à venir"
else:
    day_stack_df = pd.DataFrame(
        columns=[
            "weekLabel",
            "segment",
            "value",
            "planned",
            "actual",
            "maxValue",
            "order",
            "activity_names",
        ]
    )  # minimal columns

if not day_stack_df.empty:
    day_chart = create_daily_bar_chart(
        day_stack_df, metric_label, metric_cfg, color_scale, CHART_WIDTH
    )
    st.altair_chart(day_chart, use_container_width=False)
else:
    st.info("Aucune donnée quotidienne à afficher pour ce filtre.")
    st.dataframe(
        summary_df[["weekLabel", "Planifié", "Réalisé", "Max"]].rename(
            columns={"weekLabel": "Semaine"}
        ),
        use_container_width=True,
    )

st.subheader("Temps passé dans chaque zone HR")
hr_zone_view_col, hr_zone_scale_col = st.columns(2)
hr_zone_view = hr_zone_view_col.radio(
    "Agrégation",
    options=["Hebdomadaire", "Par session"],
    horizontal=True,
    key="analytics_hr_zone_view",
)
hr_zone_scale = hr_zone_scale_col.radio(
    "Affichage",
    options=["Absolu", "Normalisé"],
    horizontal=True,
    key="analytics_hr_zone_scale",
)
hr_zone_absolute = hr_zone_scale == "Absolu"

if hr_zone_view == "Hebdomadaire":
    weekly_zone_df = hr_zones_service.build_weekly_zone_data(
        athlete_id=athlete_id,
        start_date=start_date,
        end_date=end_date,
        categories=selected_types,
    )
    if weekly_zone_df.empty:
        st.info("Aucune donnée de zone HR disponible pour la période sélectionnée.")
    else:
        weekly_zone_df = weekly_zone_df.copy().sort_values(["weekStartDate", "zone"])
        weekly_zone_df["x_label"] = weekly_zone_df["weekLabel"]
        zone_chart = _render_hr_zone_stacked_chart(
            weekly_zone_df,
            x_col="x_label",
            x_title="Semaine",
            absolute=hr_zone_absolute,
        )
        if zone_chart is None:
            st.info("Aucune donnée de zone HR disponible pour la période sélectionnée.")
        else:
            st.altair_chart(zone_chart, use_container_width=False)
else:
    session_zone_df = _build_session_zone_distribution(activities_metrics_df)
    if session_zone_df.empty:
        st.info("Aucune donnée de zone HR disponible pour les sessions sélectionnées.")
    else:
        session_zone_df = session_zone_df.copy().sort_values(["sessionDate", "zone"])
        session_zone_df["x_label"] = session_zone_df["sessionLabel"]
        zone_chart = _render_hr_zone_stacked_chart(
            session_zone_df,
            x_col="x_label",
            x_title="Session",
            absolute=hr_zone_absolute,
        )
        if zone_chart is None:
            st.info("Aucune donnée de zone HR disponible pour les sessions sélectionnées.")
        else:
            st.altair_chart(zone_chart, use_container_width=False)
