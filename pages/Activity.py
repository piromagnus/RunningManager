"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit.logger import get_logger

from graph.elevation import prepare_elevation_plot_data, render_elevation_profile, render_grade_histogram
from graph.pacer_comparison import render_comparison_elevation_profile, render_delta_bar_chart
from graph.speed_profile import create_speed_profile_chart
from graph.timeseries import render_timeseries_charts
from persistence.csv_storage import CsvStorage
from persistence.repositories import ActivitiesRepo
from services.activity_detail_service import ActivityDetail, ActivityDetailService
from services.pacer_service import PacerService
from services.speed_profile_service import SpeedProfileService
from services.timeseries_service import TimeseriesService
from utils.config import load_config, redact
from utils.constants import BASE_MAP_STYLES, CHART_WIDTH_DEFAULT, MAPBOX_STYLES
from utils.elevation_preprocessing import preprocess_for_elevation_profile
from utils.formatting import (
    fmt_decimal,
    fmt_km,
    fmt_m,
    format_duration,
    set_locale,
)
from utils.styling import apply_theme
from utils.ui_helpers import get_dialog_factory
from widgets.comparison_panel import render_comparison_panel
from widgets.comparison_table import render_comparison_table

logger = get_logger(__name__)

st.set_page_config(page_title="Running Manager - Activité", layout="wide")
apply_theme()

CHART_WIDTH = CHART_WIDTH_DEFAULT


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
    speed_profile_service = SpeedProfileService(cfg)
    detail_service = ActivityDetailService(storage, cfg, ts_service)
    pacer_service = PacerService(storage, cfg)
    activities_repo = ActivitiesRepo(storage)

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

    # Race pacing link section
    # Ensure activity_id is a string for consistency
    activity_id_str = str(activity_id)
    linked_race_id = _render_race_pacing_link(
        activity_id_str, athlete_id, activities_repo, pacer_service, ts_service, detail
    )

    # Create tabs: Classic timeseries and Comparison (if pacing linked)
    if linked_race_id:
        tab1, tab2 = st.tabs(["📊 Time series", "🎯 Comparaison pacing vs réel"])
        
        with tab1:
            _render_timeseries_section(ts_service, speed_profile_service, detail.activity_id)
            _render_speed_profile_section(speed_profile_service, detail.activity_id)
        
        with tab2:
            _render_pacing_comparison(activity_id_str, linked_race_id, pacer_service, ts_service)
    else:
        # No pacing linked - show classic timeseries
        _render_timeseries_section(ts_service, speed_profile_service, detail.activity_id)
        _render_speed_profile_section(speed_profile_service, detail.activity_id)

    st.subheader("Trace sur la carte")
    map_choice = _select_map_style(mapbox_token)
    _render_map(detail, map_choice, mapbox_token)


def _render_race_pacing_link(
    activity_id: str,
    athlete_id: str | None,
    activities_repo: ActivitiesRepo,
    pacer_service: PacerService,
    ts_service: TimeseriesService,
    detail: ActivityDetail,
) -> str | None:
    """Render race pacing link button with popup dialog.
    
    Returns:
        Linked race ID if linked, None otherwise
    """
    # Get linked race ID from race-pacing-link.csv
    # Ensure activity_id is a string
    activity_id_str = str(activity_id)
    linked_race_id = pacer_service.get_linked_race_id(activity_id_str)
    races_df = pacer_service.list_races()

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔗 Lier un pacing", key=f"link_pacing_button_{activity_id}"):
            # Open dialog
            factory = get_dialog_factory()
            if not factory:
                st.warning("Version de Streamlit sans prise en charge des dialogues.")
                return linked_race_id

            @factory("Lier un pacing de course")
            def _dialog() -> None:
                if races_df.empty:
                    st.info("Aucun pacing de course disponible. Créez-en un dans la page 'Race Pacing'.")
                    return

                race_options = ["--- Aucun pacing ---"] + [
                    f"{row['name']} ({row['raceId']})" for _, row in races_df.iterrows()
                ]

                selected_idx = st.selectbox(
                    "Sélectionner un pacing",
                    range(len(race_options)),
                    format_func=lambda x: race_options[x],
                    index=0 if not linked_race_id else next(
                        (i for i, opt in enumerate(race_options) if linked_race_id in opt), 0
                    ),
                )

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✅ Lier", type="primary"):
                        if selected_idx > 0:
                            selected_race_id = race_options[selected_idx].split("(")[-1].rstrip(")")
                            try:
                                pacer_service.link_race_to_activity(activity_id_str, selected_race_id)
                                st.success("Pacing lié avec succès!")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to link race: {e}", exc_info=True)
                                st.error(f"Erreur lors de la liaison: {str(e)}")
                        else:
                            # Unlink
                            pacer_service.unlink_race_from_activity(activity_id_str)
                            st.success("Pacing délié!")
                            st.rerun()

                with col_btn2:
                    if st.button("Annuler"):
                        pass  # Dialog will close

            _dialog()

    with col2:
        if linked_race_id:
            # Get race name
            loaded_race = pacer_service.load_race(linked_race_id)
            if loaded_race:
                race_name, _, _, _ = loaded_race
                st.success(f"✅ Pacing lié: **{race_name}**")
                if st.button("Délier", key=f"unlink_pacing_button_{activity_id_str}"):
                    pacer_service.unlink_race_from_activity(activity_id_str)
                    st.success("Pacing délié!")
                    st.rerun()
        else:
            st.info("Aucun pacing lié. Cliquez sur le bouton pour en lier un.")

    return linked_race_id


def _render_pacing_comparison(
    activity_id: str,
    race_id: str,
    pacer_service: PacerService,
    ts_service: TimeseriesService,
) -> None:
    """Render pacing comparison tab with elevation profile, bar charts, and comparison table."""
    timeseries_df = ts_service.load(activity_id)
    if timeseries_df is None or timeseries_df.empty:
        st.warning("Les données timeseries sont nécessaires pour comparer les segments planifiés avec les réels.")
        return

    # Get planned race data
    loaded_race = pacer_service.load_race(race_id)
    if not loaded_race:
        st.error("Impossible de charger le pacing de course.")
        return

    race_name, aid_stations_km, planned_segments_df, aid_stations_times = loaded_race

    # Preprocess timeseries for comparison
    metrics_df = pacer_service.preprocess_timeseries_for_pacing(timeseries_df)
    if metrics_df.empty:
        st.warning("Impossible de prétraiter les données timeseries.")
        return

    # Compare segments
    comparison_df = pacer_service.compare_race_segments_with_activity(race_id, timeseries_df)
    if comparison_df is None or comparison_df.empty:
        st.warning("Impossible de comparer les segments. Vérifiez que les données timeseries sont disponibles.")
        return

    st.subheader(f"📊 Comparaison: {race_name}")

    # Render elevation profile with comparison tooltips
    render_comparison_elevation_profile(metrics_df, planned_segments_df, comparison_df, aid_stations_km)

    # Render delta bar charts
    st.subheader("📈 Deltas par segment")
    render_delta_bar_chart(comparison_df, planned_segments_df, pacer_service)

    # Comparison table with computed missing values
    st.subheader("📋 Tableau de comparaison détaillé")
    render_comparison_table(comparison_df, planned_segments_df, pacer_service)


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


def _render_timeseries_section(
    ts_service: TimeseriesService,
    speed_profile_service: SpeedProfileService,
    activity_id: str,
) -> None:
    st.subheader("Timeseries")
    charts = render_timeseries_charts(ts_service, activity_id, speed_profile_service)
    if charts:
        _render_timeseries_grid(charts)
        df = ts_service.load(activity_id)
        if df is not None and not df.empty and "lat" in df.columns and "lon" in df.columns:
            _render_elevation_profile(df, speed_profile_service, activity_id)
    else:
        st.caption("Pas de données timeseries pour cette activité.")


def _render_timeseries_grid(charts: dict[str, object]) -> None:
    col1, col2 = st.columns(2)
    with col1:
        hr_chart = charts.get("hr")
        if hr_chart is not None:
            st.altair_chart(hr_chart, use_container_width=True)
    with col2:
        speed_chart = charts.get("speed")
        if speed_chart is not None:
            st.altair_chart(speed_chart, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        elevation_chart = charts.get("elevation")
        if elevation_chart is not None:
            st.altair_chart(elevation_chart, use_container_width=True)
    with col4:
        speed_eq_chart = charts.get("speed_eq")
        if speed_eq_chart is not None:
            st.altair_chart(speed_eq_chart, use_container_width=True)
        elif "speed" in charts:
            st.caption("SpeedEq indisponible (données GPS insuffisantes).")


def _render_speed_profile_section(
    speed_profile_service: SpeedProfileService,
    activity_id: str,
) -> None:
    st.subheader("Profils de vitesse maximale")
    profile_df = speed_profile_service.load_speed_profile(activity_id)
    if profile_df is None:
        try:
            speed_profile_service.compute_and_store_speed_profile(activity_id)
            profile_df = speed_profile_service.load_speed_profile(activity_id)
        except Exception as exc:
            logger.exception("Failed to compute speed profile: %s", exc)
            profile_df = None

    if profile_df is not None and not profile_df.empty:
        chart = create_speed_profile_chart(profile_df, chart_width=CHART_WIDTH)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("Impossible de charger ou calculer le profil de vitesse pour cette activité.")


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




def _render_elevation_profile(
    df: pd.DataFrame,
    speed_profile_service: SpeedProfileService | None = None,
    activity_id: str | None = None,
) -> None:
    """Render elevation profile with grade-based color coding and histogram."""
    # Preprocess timeseries to compute grade metrics (uses cache if available)
    metrics_df = preprocess_for_elevation_profile(df, speed_profile_service, activity_id)

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
