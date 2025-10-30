"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit.logger import get_logger

from graph.elevation import prepare_elevation_plot_data, render_elevation_profile, render_grade_histogram
from graph.timeseries import render_timeseries_charts
from persistence.csv_storage import CsvStorage
from services.activity_detail_service import ActivityDetail, ActivityDetailService
from services.timeseries_service import TimeseriesService
from utils.config import load_config, redact
from utils.elevation_preprocessing import preprocess_for_elevation_profile
from utils.formatting import fmt_decimal, fmt_km, fmt_m, format_duration, set_locale
from utils.styling import apply_theme
from widgets.comparison_panel import render_comparison_panel

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
    render_comparison_panel(detail.comparison, detail.match_score)

    st.subheader("Timeseries")
    charts = render_timeseries_charts(ts_service, detail.activity_id)
    if charts:
        for chart in charts:
            st.altair_chart(chart)
        # Render elevation profile if we have GPS data
        df = ts_service.load(detail.activity_id)
        if df is not None and not df.empty and "lat" in df.columns and "lon" in df.columns:
            _render_elevation_profile(df)
    else:
        st.caption("Pas de données timeseries pour cette activité.")

    st.subheader("Trace sur la carte")
    map_choice = _select_map_style(mapbox_token)
    _render_map(detail, map_choice, mapbox_token)


def _render_summary(detail: ActivityDetail) -> None:
    summary = detail.summary
    metrics = [
        ("Distance", fmt_km(summary.distance_km) if summary.distance_km is not None else "-"),
        ("Durée", format_duration(summary.moving_sec, include_seconds=True)),
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




def _render_elevation_profile(df: pd.DataFrame) -> None:
    """Render elevation profile with grade-based color coding and histogram."""
    # Preprocess timeseries to compute grade metrics
    metrics_df = preprocess_for_elevation_profile(df)

    if metrics_df is None or metrics_df.empty:
        st.caption("Données insuffisantes pour le profil d'élévation détaillé.")
        return

    # Prepare data for plotting
    plot_df = prepare_elevation_plot_data(metrics_df)
    if plot_df is None or plot_df.empty:
        st.caption("Données insuffisantes pour le profil d'élévation détaillé.")
        return

    # Render the elevation profile visualization
    try:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Profil d'élévation avec pente (Interactif)**")
            render_elevation_profile(plot_df)
        with col2:
            st.markdown("**Distribution des pentes**")
            render_grade_histogram(metrics_df)
    except Exception as e:
        logger.warning(f"Failed to render elevation profile: {e}", exc_info=True)
        st.caption("Erreur lors du rendu du profil d'élévation.")


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
