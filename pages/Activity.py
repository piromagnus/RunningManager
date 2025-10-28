from __future__ import annotations

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit.logger import get_logger

from utils.config import load_config, redact
from utils.formatting import fmt_decimal, fmt_km, fmt_m, set_locale
from utils.styling import apply_theme
from persistence.csv_storage import CsvStorage
from services.activity_detail_service import ActivityDetail, ActivityDetailService
from services.timeseries_service import TimeseriesService


logger = get_logger(__name__)

st.set_page_config(page_title="Running Manager - Activité", layout="wide")
apply_theme()

CHART_WIDTH = 860

BASE_MAP_STYLES = [
    {
        "label": "Carto clair (défaut)",
        "provider": "carto",
        "style": "light",
    },
    {
        "label": "Carto sombre",
        "provider": "carto",
        "style": "dark",
    },
]

MAPBOX_STYLES = [
    {
        "label": "Mapbox Light",
        "provider": "mapbox",
        "style": "mapbox://styles/mapbox/light-v11",
    },
    {
        "label": "Mapbox Dark",
        "provider": "mapbox",
        "style": "mapbox://styles/mapbox/dark-v11",
    },
    {
        "label": "Mapbox Outdoors",
        "provider": "mapbox",
        "style": "mapbox://styles/mapbox/outdoors-v12",
    },
    {
        "label": "Mapbox Satellite",
        "provider": "mapbox",
        "style": "mapbox://styles/mapbox/satellite-streets-v12",
    },
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
    cfg = st.session_state.get("app_config")
    if cfg is None:
        cfg = load_config()
        st.session_state["app_config"] = cfg
    mapbox_token = st.session_state.get("mapbox_token")
    if mapbox_token is None:
        mapbox_token = cfg.mapbox_token
        if mapbox_token:
            st.session_state["mapbox_token"] = mapbox_token
    set_locale("fr_FR")
    storage = CsvStorage(cfg.data_dir)
    ts_service = TimeseriesService(cfg)
    detail_service = ActivityDetailService(storage, cfg, ts_service)

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
    map_choice = _select_map_style(mapbox_token)
    _render_map(detail, map_choice, mapbox_token)


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


def _select_map_style(mapbox_token: str | None) -> dict:
    styles = list(BASE_MAP_STYLES)
    if mapbox_token:
        styles.extend(MAPBOX_STYLES)
    labels = [style["label"] for style in styles]
    default_label = st.session_state.get("activity_map_style_label", labels[0])
    if default_label not in labels:
        default_label = labels[0]
    selection = st.selectbox(
        "Fond de carte",
        labels,
        index=labels.index(default_label),
        help=(
            "Ajoutez MAPBOX_TOKEN dans `.env` pour débloquer les fonds Mapbox supplémentaires."
            if not mapbox_token
            else ""
        ),
    )
    st.session_state["activity_map_style_label"] = selection
    return next(style for style in styles if style["label"] == selection)


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
        st.altair_chart(chart)


def _build_map_deck(
    detail: ActivityDetail, map_choice: dict, mapbox_token: str | None
) -> pdk.Deck | None:
    logger.info(
        "Building map deck for activity %s with map choice %s and mapbox token %s",
        detail.activity_id,
        map_choice,
        mapbox_token,
    )
    if not detail.map_path:
        logger.debug("Skipping map render: no path for activity %s", detail.activity_id)
        return None

    path_coords = [[point.lon, point.lat] for point in detail.map_path]
    if not path_coords:
        logger.debug("Skipping map render: empty path for activity %s", detail.activity_id)
        return None

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
    provider = map_choice.get("provider", "carto")
    style = map_choice.get("style", "light")
    import os

    logger.debug("MAPBOX_API_KEY: %s", os.getenv("MAPBOX_API_KEY"))
    logger.debug(
        "Building map deck for activity %s with provider=%s style=%s points=%d",
        detail.activity_id,
        provider,
        style,
        len(path_coords),
    )
    deck_kwargs: dict = {
        "layers": [layer],
        "initial_view_state": initial,
        "map_provider": provider,
        "map_style": style,
        "tooltip": False,
    }
    if provider == "mapbox":
        if not mapbox_token:
            logger.warning(
                "Mapbox style requested without token for activity %s (style=%s)",
                detail.activity_id,
                style,
            )
            return None
        redacted_token = redact(mapbox_token)
        logger.debug(
            "Applying Mapbox token for activity %s (style=%s, token=%s)",
            detail.activity_id,
            style,
            redacted_token,
        )
        # current_token = getattr(pdk.settings, "mapbox_api_key", None)
        # if current_token != mapbox_token:
        #     pdk.settings.mapbox_api_key = mapbox_token
        #     logger.info(
        #         "Updated pydeck Mapbox token for activity %s (token=%s)",
        #         detail.activity_id,
        #         redacted_token,
        #     )
        # deck_kwargs["api_keys"] = {"mapbox": mapbox_token}
    deck = pdk.Deck(**deck_kwargs)
    # if provider == "mapbox" and mapbox_token:
    #     deck.mapbox_key = mapbox_token
    #     logger.debug(
    #         "Set deck.mapbox_key manually for activity %s (token=%s)",
    #         detail.activity_id,
    #         redact(mapbox_token),
    #     )
    return deck


def _render_map(detail: ActivityDetail, map_choice: dict, mapbox_token: str | None) -> None:
    provider = map_choice.get("provider")
    if provider == "mapbox":
        redacted = redact(mapbox_token)
        st.caption(
            f"Fond de carte Mapbox : {map_choice.get('label')} (jeton {redacted or 'absent'})"
        )
    else:
        st.caption(f"Fond de carte Carto : {map_choice.get('label')}")
    deck = _build_map_deck(detail, map_choice, mapbox_token)
    if deck is None:
        logger.debug(
            "No deck generated for activity %s with provider %s",
            detail.activity_id,
            map_choice.get("provider"),
        )
        st.caption(detail.map_notice or "Aucune donnée de trace disponible.")
        return
    deck_token = getattr(deck, "mapbox_key", None)
    logger.debug(
        "Deck mapbox key attribute for activity %s: %s",
        detail.activity_id,
        redact(deck_token),
    )
    st.pydeck_chart(deck)


if __name__ == "__main__":
    main()
