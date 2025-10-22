from __future__ import annotations

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from utils.config import Config, load_config
from utils.formatting import fmt_decimal, fmt_km, fmt_m, set_locale
from utils.styling import apply_theme
from persistence.csv_storage import CsvStorage
from services.activity_detail_service import ActivityDetail, ActivityDetailService
from services.timeseries_service import TimeseriesService


st.set_page_config(page_title="Running Manager - Activité", layout="wide")
apply_theme()


MAP_STYLES = [
    ("OpenStreetMap (sans clé)", "open-street-map", False),
    ("Mapbox Light", "mapbox://styles/mapbox/light-v11", True),
    ("Mapbox Dark", "mapbox://styles/mapbox/dark-v11", True),
    ("Mapbox Outdoors", "mapbox://styles/mapbox/outdoors-v12", True),
    ("Mapbox Satellite", "mapbox://styles/mapbox/satellite-streets-v12", True),
]


def _format_duration(seconds):
    if seconds in (None, "", float("nan")):
        return "-"
    total = int(float(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}"
    if minutes:
        return f"{minutes}m{secs:02d}"
    return f"{secs}s"


def main() -> None:
    cfg = load_config()
    set_locale("fr_FR")
    storage = CsvStorage(cfg.data_dir)
    ts_service = TimeseriesService(cfg)
    detail_service = ActivityDetailService(storage, cfg, ts_service)
    if cfg.mapbox_token:
        pdk.settings.mapbox_api_key = cfg.mapbox_token

    params = st.query_params

    def _first(value):
        if isinstance(value, list):
            return value[0] if value else None
        return value

    activity_id = _first(params.get("activityId")) or st.session_state.get("activity_view_id")
    athlete_id = _first(params.get("athleteId")) or st.session_state.get("activity_view_athlete")

    st.page_link("pages/Activities.py", label="← Retour au flux")

    if not activity_id:
        st.warning("Aucune activité sélectionnée.")
        st.stop()

    try:
        detail = detail_service.get_detail(athlete_id, str(activity_id))
    except Exception as exc:
        st.error(f"Impossible de charger l'activité : {exc}")
        st.stop()

    st.title(detail.title or "Activité")
    if detail.description:
        st.caption(detail.description)

    _render_summary(detail)
    _render_link_panel(detail)

    st.subheader("Timeseries")
    _render_timeseries(ts_service, detail.activity_id)

    st.subheader("Trace sur la carte")
    map_style = _select_map_style(cfg)
    _render_map(detail, map_style)


def _render_summary(detail: ActivityDetail) -> None:
    summary = detail.summary
    metrics = [
        ("Distance", fmt_km(summary.distance_km) if summary.distance_km is not None else "-"),
        ("Durée", _format_duration(summary.moving_sec)),
        ("D+", fmt_m(summary.ascent_m) if summary.ascent_m is not None else "-"),
        ("FC moy", fmt_decimal(summary.avg_hr, 0) if summary.avg_hr is not None else "-"),
        ("TRIMP", fmt_decimal(summary.trimp, 1) if summary.trimp is not None else "-"),
        (
            "Dist. équiv.",
            fmt_decimal(summary.distance_eq_km, 1) if summary.distance_eq_km is not None else "-",
        ),
    ]
    cols = st.columns(len(metrics))
    for (label, value), col in zip(metrics, cols):
        with col:
            st.metric(label, value)


def _select_map_style(cfg: Config) -> str:
    available_styles = [style for style in MAP_STYLES if cfg.mapbox_token or not style[2]]
    labels = [style[0] for style in available_styles]
    default_label = st.session_state.get("activity_map_style_label", labels[0])
    if default_label not in labels:
        default_label = labels[0]
    selection = st.selectbox(
        "Fond de carte",
        labels,
        index=labels.index(default_label),
        help="Ajoutez MAPBOX_TOKEN dans `.env` pour débloquer les fonds Mapbox supplémentaires.",
    )
    st.session_state["activity_map_style_label"] = selection
    return next(style for label, style, _ in available_styles if label == selection)


def _render_link_panel(detail: ActivityDetail) -> None:
    if not detail.linked or not detail.comparison:
        st.info("Cette activité n'est liée à aucune séance planifiée.")
        return

    comp = detail.comparison
    st.subheader("Comparaison plan / réalisé")
    score = fmt_decimal(detail.match_score, 2) if detail.match_score is not None else "n/a"
    st.caption(f"Score de correspondance : {score}")

    comparison_cards = [
        ("Distance", comp.distance),
        ("Durée", comp.duration),
        ("TRIMP", comp.trimp),
        ("D+", comp.ascent),
    ]
    cols = st.columns(len(comparison_cards))
    for (label, metric), col in zip(comparison_cards, cols):
        with col:
            planned_val = _format_comparison_value(label, metric.planned)
            actual_val = _format_comparison_value(label, metric.actual)
            delta_label, delta_value = _format_comparison_delta(label, metric.delta)
            delta_color = "#60ac84" if (metric.delta or 0) >= 0 else "#9e4836"
            indicator = "▲" if (metric.delta or 0) >= 0 else "▼"
            st.markdown(
                f"""
<div style="
    background: rgba(20, 32, 48, 0.75);
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    border: 1px solid rgba(228, 204, 160, 0.22);
    box-shadow: 0 12px 24px rgba(8, 14, 24, 0.28);
">
    <div style="font-size:0.9rem; color:#e4cca0; letter-spacing:0.04em;">{label.upper()}</div>
    <div style="margin-top:0.35rem; color:#f8fafc;">Plan&nbsp;: <strong>{planned_val}</strong></div>
    <div style="color:#f8fafc;">Réel&nbsp;: <strong>{actual_val}</strong></div>
    <div style="margin-top:0.4rem; color:{delta_color}; font-weight:600;">
        {indicator}&nbsp;{delta_label}: {delta_value}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

def _format_comparison_value(label: str, value) -> str:
    if value is None:
        return "-"
    if label == "Durée":
        return _format_duration(value)
    if label == "D+":
        return fmt_m(value)
    precision = 1 if label in {"Distance", "TRIMP"} else 1
    return fmt_decimal(value, precision)


def _format_comparison_delta(label: str, value) -> str:
    if value is None:
        return label, "-"
    sign = "+" if value >= 0 else "-"
    magnitude = abs(value)
    if label == "Durée":
        return "Δ", f"{sign}{_format_duration(magnitude)}"
    if label == "D+":
        return "Δ", f"{sign}{fmt_m(magnitude)}"
    precision = 1 if label in {"Distance", "TRIMP"} else 1
    return "Δ", f"{sign}{fmt_decimal(magnitude, precision)}"


def _render_timeseries(ts_service: TimeseriesService, activity_id: str) -> None:
    df = ts_service.load(activity_id)
    if df is None or df.empty:
        st.caption("Pas de données timeseries pour cette activité.")
        return
    df = df.copy()
    if "timestamp" not in df.columns:
        st.caption("Timeseries incomplète.")
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        st.caption("Timeseries incomplète.")
        return
    df = df.sort_values("timestamp")
    start = df["timestamp"].iloc[0]
    df["minutes"] = (df["timestamp"] - start).dt.total_seconds() / 60.0

    charts = []
    if "hr" in df.columns and df["hr"].notna().any():
        charts.append(
            alt.Chart(df)
            .mark_line(color="#ef4444")
            .encode(x=alt.X("minutes:Q", title="Temps (min)"), y=alt.Y("hr:Q", title="FC (bpm)"))
            .properties(height=180)
        )
    if "paceKmh" in df.columns and df["paceKmh"].notna().any():
        charts.append(
            alt.Chart(df)
            .mark_line(color="#3b82f6")
            .encode(
                x=alt.X("minutes:Q", title="Temps (min)"),
                y=alt.Y("paceKmh:Q", title="Vitesse (km/h)"),
            )
            .properties(height=180)
        )
    if "elevationM" in df.columns and df["elevationM"].notna().any():
        charts.append(
            alt.Chart(df)
            .mark_area(color="#10b981", opacity=0.4)
            .encode(
                x=alt.X("minutes:Q", title="Temps (min)"),
                y=alt.Y("elevationM:Q", title="Altitude (m)"),
            )
            .properties(height=160)
        )
    if not charts:
        st.caption("Pas de séries exploitables (FC, vitesse ou altitude).")
        return
    for chart in charts:
        st.altair_chart(chart, use_container_width=True)


def _render_map(detail: ActivityDetail, map_style: str) -> None:
    if detail.map_path:
        path_coords = [[point.lon, point.lat] for point in detail.map_path]
        data = [{"path": path_coords}]
        initial = pdk.ViewState(
            latitude=detail.map_path[0].lat,
            longitude=detail.map_path[0].lon,
            zoom=13,
            pitch=0,
        )
        layer = pdk.Layer(
            "PathLayer",
            data=data,
            get_path="path",
            get_color=[255, 99, 71],
            width_scale=20,
            width_min_pixels=3,
        )
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=initial,
                map_style=map_style,
            )
        )
    else:
        st.caption(detail.map_notice or "Aucune donnée de trace disponible.")


main()
