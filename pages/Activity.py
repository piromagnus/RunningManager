from __future__ import annotations

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit.logger import get_logger

from persistence.csv_storage import CsvStorage
from services.activity_detail_service import ActivityDetail, ActivityDetailService
from services.speed_profile_service import SpeedProfileService
from services.timeseries_service import TimeseriesService
from utils.config import load_config, redact
from utils.formatting import fmt_decimal, fmt_km, fmt_m, set_locale
from utils.styling import apply_theme

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

    # Render elevation profile with grade-based color coding
    _render_elevation_profile(df)


def _render_elevation_profile(df: pd.DataFrame) -> None:
    """Render elevation profile with grade-based color coding and histogram."""
    # Preprocess timeseries to compute grade metrics
    sp = SpeedProfileService(config=None)
    
    # Check if we have GPS data for preprocessing
    if "lat" not in df.columns or "lon" not in df.columns:
        st.caption("Données GPS manquantes pour le profil d'élévation détaillé.")
        return
    
    if df["lat"].isna().all() or df["lon"].isna().all():
        st.caption("Données GPS insuffisantes pour le profil d'élévation détaillé.")
        return
    
    try:
        metrics_df = sp.preprocess_timeseries(df)
    except Exception as e:
        logger.warning(f"Failed to preprocess timeseries for elevation profile: {e}")
        st.caption("Impossible de calculer le profil d'élévation détaillé.")
        return
    
    if metrics_df.empty:
        st.caption("Données insuffisantes pour le profil d'élévation détaillé.")
        return
    
    # Apply moving average to grade for smoothing
    metrics_df = sp.moving_average(metrics_df, window_size=10, col="grade")
    
    # Render the elevation profile visualization
    try:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Profil d'élévation avec pente (Interactif)**")
            _plot_interactive_elevation(metrics_df)
        with col2:
            st.markdown("**Distribution des pentes**")
            _plot_grade_histogram(metrics_df)
    except Exception as e:
        logger.warning(f"Failed to render elevation profile: {e}")
        st.caption("Erreur lors du rendu du profil d'élévation.")


def _get_grade_category(grade_val):
    """Map grade value to category for coloring."""
    if pd.isna(grade_val):
        return "unknown"
    if grade_val < -0.5:
        return "grade_lt_neg_0_5"
    elif -0.5 <= grade_val < -0.25:
        return "grade_lt_neg_0_25"
    elif -0.25 <= grade_val < -0.05:
        return "grade_lt_neg_0_05"
    elif -0.05 <= grade_val < 0.05:
        return "grade_neutral"
    elif 0.05 <= grade_val < 0.1:
        return "grade_lt_0_1"
    elif 0.1 <= grade_val < 0.25:
        return "grade_lt_0_25"
    elif 0.25 <= grade_val < 0.5:
        return "grade_lt_0_5"
    else:
        return "grade_ge_0_5"


def _plot_interactive_elevation(df: pd.DataFrame) -> None:
    """Plot interactive elevation profile with segment-based grade coloring."""
    try:
        logger.info("Starting elevation profile plot")
        logger.debug(f"Input DataFrame shape: {df.shape}, columns: {list(df.columns)}")
        
        COLOR_MAPPING = {
            'grade_lt_neg_0_5': '#001f3f',    # darkblue
            'grade_lt_neg_0_25': '#004d26',   # darkgreen
            'grade_lt_neg_0_05': '#22c55e',   # green
            'grade_neutral': '#d1d5db',       # lightgray
            'grade_lt_0_1': '#eab308',        # yellow
            'grade_lt_0_25': '#f97316',       # orange
            'grade_lt_0_5': '#dc2626',        # red
            'grade_ge_0_5': '#000000',        # black
            'unknown': '#808080'              # gray
        }
        
        # Prepare data with formatted values - ensure all required columns exist
        required_cols = ['cumulated_distance', 'elevationM_ma_5', 'grade_ma_10', 
                         'speed_km_h', 'hr', 'cumulated_duration_seconds']
        
        # Check which columns exist in df
        available_cols = [col for col in required_cols if col in df.columns]
        missing_req_cols = [col for col in required_cols if col not in df.columns]
        
        logger.debug(f"Required columns: {required_cols}")
        logger.debug(f"Available columns: {available_cols}")
        if missing_req_cols:
            logger.warning(f"Missing required columns: {missing_req_cols}")
        
        if not available_cols:
            logger.error("No required columns found in DataFrame")
            st.caption("Données insuffisantes pour le profil (colonnes manquantes).")
            return
        
        # Create plot_df with only available columns
        plot_df = df[available_cols].copy()
        
        logger.debug(f"Initial plot_df shape: {plot_df.shape}")
        
        # Drop rows where essential columns are NaN
        if 'cumulated_distance' not in plot_df.columns or 'elevationM_ma_5' not in plot_df.columns:
            logger.error("Essential columns (cumulated_distance or elevationM_ma_5) missing")
            st.caption("Données insuffisantes pour le profil (colonnes essentielles manquantes).")
            return
        
        plot_df = plot_df.dropna(subset=['cumulated_distance', 'elevationM_ma_5'])
        logger.debug(f"After dropping NaN in essential columns: {plot_df.shape}")
        
        if plot_df.empty:
            logger.warning("Plot DataFrame is empty after dropping NaN")
            st.caption(
                "Données insuffisantes pour le profil "
                "(toutes les lignes supprimées après nettoyage)."
            )
            return
        
        # Fill NaN values for other columns (don't drop rows)
        if 'grade_ma_10' in plot_df.columns:
            plot_df['grade_ma_10'] = plot_df['grade_ma_10'].fillna(0)
        else:
            plot_df['grade_ma_10'] = 0
            logger.debug("Created grade_ma_10 column with default value 0")
        
        if 'speed_km_h' not in plot_df.columns:
            plot_df['speed_km_h'] = 0
        else:
            plot_df['speed_km_h'] = plot_df['speed_km_h'].fillna(0)
        
        if 'hr' not in plot_df.columns:
            plot_df['hr'] = 0
        else:
            plot_df['hr'] = plot_df['hr'].fillna(0)
        
        if 'cumulated_duration_seconds' not in plot_df.columns:
            plot_df['cumulated_duration_seconds'] = 0
        else:
            plot_df['cumulated_duration_seconds'] = plot_df['cumulated_duration_seconds'].fillna(0)
        
        plot_df = plot_df.reset_index(drop=True)
        
        logger.debug(f"After filling NaN: {plot_df.shape}, columns: {list(plot_df.columns)}")
        
        # Calculate Y-axis bounds
        try:
            min_elev = plot_df['elevationM_ma_5'].min()
            max_elev = plot_df['elevationM_ma_5'].max()
            elevation_range = max(max_elev - min_elev, 1)
            padding = max(20, elevation_range * 0.05)
            y_min = min_elev - padding
            y_max = max_elev + padding
            logger.debug(
                "Y-axis bounds with padding: min=%.2f, max=%.2f (range=%.2f, padding=%.2f)",
                y_min,
                y_max,
                elevation_range,
                padding,
            )
        except Exception as e:
            logger.error(f"Failed to calculate Y-axis bounds: {e}", exc_info=True)
            st.caption("Erreur lors du calcul des limites de l'axe Y.")
            return
        
        # Add grade category for segment-based coloring
        try:
            plot_df['grade_category'] = plot_df['grade_ma_10'].apply(_get_grade_category)
            categories = plot_df['grade_category'].unique()
            logger.debug("Grade categories created. Unique categories: %s", categories)
        except Exception as e:
            logger.error(f"Failed to create grade categories: {e}", exc_info=True)
            st.caption("Erreur lors de la création des catégories de pente.")
            return
        
        # Prepare numeric values for tooltips (Altair will format them)
        # We'll use format strings in Tooltip() instead of pre-formatted strings
        try:
            # Calculate grade percentage for tooltip
            plot_df['grade_pct'] = plot_df['grade_ma_10'] * 100
            
            # Calculate time in hours for tooltip
            plot_df['time_hours'] = plot_df['cumulated_duration_seconds'] / 3600.0
            
            # Ensure all numeric columns are properly filled
            plot_df['speed_km_h'] = plot_df['speed_km_h'].fillna(0)
            plot_df['hr'] = plot_df['hr'].fillna(0)
            plot_df['time_hours'] = plot_df['time_hours'].fillna(0)
            plot_df['grade_pct'] = plot_df['grade_pct'].fillna(0)
            
            logger.debug("Tooltip numeric columns prepared successfully")
            if len(plot_df) > 0:
                sample_row = plot_df.iloc[0]
                logger.debug(
                    (
                        "Sample tooltip values (first row): distance=%.2f km, "
                        "elevation=%.2f m, grade=%.2f%%, time=%.2f h, "
                        "hr=%.2f bpm, speed=%.2f km/h"
                    ),
                    sample_row['cumulated_distance'],
                    sample_row['elevationM_ma_5'],
                    sample_row['grade_pct'],
                    sample_row['time_hours'],
                    sample_row['hr'],
                    sample_row['speed_km_h'],
                )
        except Exception as e:
            logger.error(f"Failed to prepare tooltip values: {e}", exc_info=True)
            st.caption("Erreur lors de la préparation des valeurs de tooltip.")
            return
        
        # Create segments by grouping consecutive rows with the same grade category
        try:
            plot_df['segment'] = (
                plot_df['grade_category'] != plot_df['grade_category'].shift()
            ).cumsum()
            num_segments = plot_df['segment'].max() + 1 if len(plot_df) > 0 else 0
            logger.debug(f"Created {num_segments} segments")
        except Exception as e:
            logger.error(f"Failed to create segments: {e}", exc_info=True)
            st.caption("Erreur lors de la création des segments.")
            return
        
        # Build the layered chart with one area per segment
        charts = []
        
        try:
            # Add area fill for each segment with full tooltips
            for segment_id in plot_df['segment'].unique():
                segment_data = plot_df[plot_df['segment'] == segment_id]
                if not segment_data.empty:
                    if 'grade_category' not in segment_data.columns:
                        logger.warning(f"Grade category missing for segment {segment_id}")
                        segment_color = COLOR_MAPPING.get('unknown', '#808080')
                    else:
                        segment_category = segment_data['grade_category'].iloc[0]
                        segment_color = COLOR_MAPPING.get(
                            segment_category,
                            COLOR_MAPPING.get('unknown', '#808080'),
                        )
                    
                    area_segment = alt.Chart(segment_data).mark_area(
                        opacity=0.4,
                        interpolate='monotone'
                    ).encode(
                        x=alt.X(
                            'cumulated_distance:Q',
                            title='Distance',
                            scale=alt.Scale(nice=True)
                        ),
                        y=alt.Y(
                            'elevationM_ma_5:Q',
                            title='Elevation',
                            scale=alt.Scale(domain=[y_min, y_max], nice=False)
                        ),
                        y2=alt.datum(y_min),
                        color=alt.value(segment_color),
                        tooltip=[
                            alt.Tooltip('grade_pct:Q', title='Grade', format='.2f'),
                            alt.Tooltip(
                                'cumulated_distance:Q',
                                title='Distance (km)',
                                format='.2f',
                            ),
                            alt.Tooltip('time_hours:Q', title='Time (h)', format='.2f'),
                            alt.Tooltip('hr:Q', title='HR (bpm)', format='.2f'),
                            alt.Tooltip('speed_km_h:Q', title='Speed (km/h)', format='.2f'),
                            alt.Tooltip(
                                'elevationM_ma_5:Q',
                                title='Elevation (m)',
                                format='.2f',
                            ),
                        ]
                    )
                    charts.append(area_segment)
            
            logger.debug(f"Created {len(charts)} area segments")
            if len(plot_df) > 0:
                logger.debug(
                    "Elevation range for rendering: min=%.2f, max=%.2f, "
                    "y_min=%.2f, y_max=%.2f",
                    plot_df['elevationM_ma_5'].min(),
                    plot_df['elevationM_ma_5'].max(),
                    y_min,
                    y_max,
                )
        except Exception as e:
            logger.error(f"Failed to create area segments: {e}", exc_info=True)
            st.caption("Erreur lors de la création des segments de zone.")
            return
        
        # Combine all area segments (elevation profile is the top edge of the areas)
        try:
            combined = alt.layer(*charts).properties(
                width=800,
                height=400,
                title=(
                    "Profil d'élévation avec code couleur de pente "
                ),
            )
            
            logger.debug("Charts combined successfully")
            logger.debug(f"Combined chart type: {type(combined)}")
            
            st.altair_chart(combined, theme=None, use_container_width=True)
            logger.info("Elevation profile chart rendered successfully")
        except Exception as e:
            logger.error(f"Failed to combine or render charts: {e}", exc_info=True)
            st.error(f"Erreur lors du rendu du graphique: {str(e)}")
            raise
        
        # Add legend
        st.markdown("**Légende des pentes :**")
        legend_items = [
            ("🔵 < -50%", "Descente très raide"),
            ("🟢 -50% à -25%", "Descente"),
            ("🟢 -25% à -5%", "Descente douce"),
            ("⚪ -5% à +5%", "Terrain plat"),
            ("🟡 +5% à +10%", "Montée douce"),
            ("🟠 +10% à +25%", "Montée"),
            ("🔴 +25% à +50%", "Montée raide"),
            ("⚫ ≥ +50%", "Montée très raide"),
        ]
        
        cols = st.columns(4)
        for idx, (label, desc) in enumerate(legend_items):
            with cols[idx % 4]:
                st.caption(f"{label}: {desc}")
    
    except Exception as e:
        logger.error(f"Unexpected error in _plot_interactive_elevation: {e}", exc_info=True)
        st.error(f"Erreur inattendue lors du rendu du profil d'élévation: {str(e)}")
        st.caption("Vérifiez les logs pour plus de détails.")


def _plot_grade_histogram(df: pd.DataFrame) -> None:
    """Plot a histogram of grade values with color coding."""
    import matplotlib.pyplot as plt
    
    COLOR_PALETTE = {
        'grade_lt_neg_0_5': '#001f3f',    # darkblue
        'grade_lt_neg_0_25': '#004d26',   # darkgreen
        'grade_lt_neg_0_05': '#22c55e',   # green
        'grade_neutral': '#d1d5db',       # lightgray
        'grade_lt_0_1': '#eab308',        # yellow
        'grade_lt_0_25': '#f97316',       # orange
        'grade_lt_0_5': '#dc2626',        # red
        'grade_ge_0_5': '#000000'         # black
    }
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Get grade values
    grades = df['grade_ma_10'].dropna()
    
    if grades.empty:
        st.caption("Pas de données de pente disponibles.")
        return
    
    # Plot histogram
    n, bins, patches = ax.hist(grades, bins=50, alpha=0.7, edgecolor='black')
    
    # Color each bar based on its bin center
    def get_color(grade_val):
        if grade_val < -0.5:
            return COLOR_PALETTE['grade_lt_neg_0_5']
        elif -0.5 <= grade_val < -0.25:
            return COLOR_PALETTE['grade_lt_neg_0_25']
        elif -0.25 <= grade_val < -0.05:
            return COLOR_PALETTE['grade_lt_neg_0_05']
        elif -0.05 <= grade_val < 0.05:
            return COLOR_PALETTE['grade_neutral']
        elif 0.05 <= grade_val < 0.1:
            return COLOR_PALETTE['grade_lt_0_1']
        elif 0.1 <= grade_val < 0.25:
            return COLOR_PALETTE['grade_lt_0_25']
        elif 0.25 <= grade_val < 0.5:
            return COLOR_PALETTE['grade_lt_0_5']
        else:
            return COLOR_PALETTE['grade_ge_0_5']
    
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        color = get_color(bin_center)
        patch.set_facecolor(color)
    
    ax.set_xlabel('Pente')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution des pentes')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add vertical lines at grade boundaries
    for boundary in [-0.5, -0.25, -0.05, 0.05, 0.1, 0.25, 0.5]:
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


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
