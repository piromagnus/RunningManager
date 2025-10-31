"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Race Pacing page for GPX import, course segmentation, and pacing plan.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.logger import get_logger

from graph.pacer_segments import render_pacer_segments
from persistence.csv_storage import CsvStorage
from services.pacer_service import PacerService
from utils.config import load_config
from utils.formatting import fmt_decimal, format_session_duration, set_locale
from utils.gpx_parser import parse_gpx_to_timeseries
from utils.styling import apply_theme

logger = get_logger(__name__)

st.set_page_config(page_title="Running Manager - Race Pacing", layout="wide")
apply_theme()
st.title("Race Pacing")

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
pacer_service = PacerService(storage)

# Initialize session state
if "race_pacing_race_id" not in st.session_state:
    st.session_state["race_pacing_race_id"] = None
if "race_pacing_race_name" not in st.session_state:
    st.session_state["race_pacing_race_name"] = ""
if "race_pacing_timeseries" not in st.session_state:
    st.session_state["race_pacing_timeseries"] = None
if "race_pacing_metrics" not in st.session_state:
    st.session_state["race_pacing_metrics"] = None
if "race_pacing_aid_km" not in st.session_state:
    st.session_state["race_pacing_aid_km"] = []
if "race_pacing_segments" not in st.session_state:
    st.session_state["race_pacing_segments"] = None

# Load existing races dropdown
races_df = pacer_service.list_races()
if not races_df.empty:
    race_options = ["--- Nouvelle course ---"] + [
        f"{row['name']} ({row['raceId']})" for _, row in races_df.iterrows()
    ]
    selected_race_idx = st.selectbox(
        "Charger une course existante",
        range(len(race_options)),
        format_func=lambda x: race_options[x]
    )

    if selected_race_idx > 0:
        selected_race_name = race_options[selected_race_idx]
        race_id = selected_race_name.split("(")[-1].rstrip(")")
        
        # Only load if this is a different race than what's already loaded
        current_race_id = st.session_state.get("race_pacing_race_id")
        if race_id != current_race_id:
            loaded_race = pacer_service.load_race(race_id)

            if loaded_race:
                race_name, aid_stations_km, segments_df, aid_stations_times = loaded_race
                st.session_state["race_pacing_race_id"] = race_id
                st.session_state["race_pacing_race_name"] = race_name
                st.session_state["race_pacing_aid_km"] = aid_stations_km
                st.session_state["race_pacing_segments"] = segments_df
                # Store aid station times if available (for display)
                if aid_stations_times:
                    st.session_state["race_pacing_aid_times"] = aid_stations_times
                else:
                    # Clear aid times if not available
                    st.session_state.pop("race_pacing_aid_times", None)
                st.rerun()

# Race name input
# Use race_id in key to force widget update when loading a different race
race_name_key = f"race_name_input_{st.session_state.get('race_pacing_race_id', 'new')}"
race_name = st.text_input(
    "Nom de la course",
    value=st.session_state.get("race_pacing_race_name", ""),
    key=race_name_key
)
if race_name != st.session_state.get("race_pacing_race_name", ""):
    st.session_state["race_pacing_race_name"] = race_name

# GPX upload
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("GPX de la course", type=["gpx"], key="gpx_uploader")

    if uploaded_file is not None:
        try:
            # Preserve aid stations before processing GPX
            preserved_aid_km = st.session_state.get("race_pacing_aid_km", []).copy()
            
            gpx_bytes = uploaded_file.read()
            timeseries_df = parse_gpx_to_timeseries(gpx_bytes)

            if timeseries_df.empty:
                st.error("Erreur: Le fichier GPX ne contient pas assez de points de trace (< 100 points).")
            else:
                st.session_state["race_pacing_timeseries"] = timeseries_df

                # Preprocess for pacing
                metrics_df = pacer_service.preprocess_timeseries_for_pacing(timeseries_df)
                if metrics_df.empty:
                    st.error("Erreur lors du prétraitement des données GPX.")
                else:
                    st.session_state["race_pacing_metrics"] = metrics_df
                    # Restore preserved aid stations after GPX load
                    st.session_state["race_pacing_aid_km"] = preserved_aid_km
                    st.success(f"GPX chargé: {len(timeseries_df)} points")

        except Exception as e:
            logger.error(f"Failed to parse GPX: {e}", exc_info=True)
            st.error(f"Erreur lors du parsing du GPX: {str(e)}")

# Aid stations input
with col2:
    st.subheader("Ravitaillements (km)")
    # Use race_id in key to force widget update when loading a different race
    # Also ensures aid stations are preserved when GPX is loaded
    aid_input_key = f"aid_stations_input_{st.session_state.get('race_pacing_race_id', 'new')}"
    aid_input = st.text_area(
        "Positions en km (une par ligne)",
        value="\n".join([str(x) for x in st.session_state.get("race_pacing_aid_km", [])]),
        height=150,
        key=aid_input_key,
    )

    # Parse aid stations
    aid_stations_km = []
    for line in aid_input.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                km = float(line)
                if km > 0:
                    aid_stations_km.append(km)
            except ValueError:
                pass

    aid_stations_km = sorted(set(aid_stations_km))
    st.session_state["race_pacing_aid_km"] = aid_stations_km

    if aid_stations_km:
        st.caption(f"{len(aid_stations_km)} ravitaillement(s) configuré(s)")
        
        # Button to apply aid stations and segment
        if st.button("Appliquer les ravitaillements", key="apply_aid_stations_button"):
            metrics_available = (
                st.session_state["race_pacing_metrics"] is not None
                and not st.session_state["race_pacing_metrics"].empty
            )
            if metrics_available:
                try:
                    segments_df = pacer_service.segment_course(
                        st.session_state["race_pacing_metrics"],
                        st.session_state["race_pacing_aid_km"]
                    )
                    if not segments_df.empty:
                        st.session_state["race_pacing_segments"] = segments_df
                        st.success(
                            f"Ravitaillements appliqués. Parcours segmenté en {len(segments_df)} segments"
                        )
                        st.rerun()
                    else:
                        st.error("Erreur lors de la segmentation du parcours.")
                except Exception as e:
                    logger.error(f"Failed to segment course: {e}", exc_info=True)
                    st.error(f"Erreur lors de la segmentation: {str(e)}")
            else:
                st.warning("Veuillez d'abord charger un fichier GPX.")

# Process segmentation if we have metrics and aid stations
if st.session_state["race_pacing_metrics"] is not None and not st.session_state["race_pacing_metrics"].empty:
    metrics_df = st.session_state["race_pacing_metrics"]

    if st.button("Segmenter le parcours", key="segment_button"):
        try:
            segments_df = pacer_service.segment_course(metrics_df, st.session_state["race_pacing_aid_km"])
            if not segments_df.empty:
                st.session_state["race_pacing_segments"] = segments_df
                st.success(f"Parcours segmenté en {len(segments_df)} segments")
            else:
                st.error("Erreur lors de la segmentation du parcours.")
        except Exception as e:
            logger.error(f"Failed to segment course: {e}", exc_info=True)
            st.error(f"Erreur lors de la segmentation: {str(e)}")

# Display chart and segments editor
if st.session_state["race_pacing_segments"] is not None and not st.session_state["race_pacing_segments"].empty:
    segments_df = st.session_state["race_pacing_segments"].copy()

    # Render elevation profile with segments
    if st.session_state["race_pacing_metrics"] is not None:
        render_pacer_segments(
            st.session_state["race_pacing_metrics"], segments_df, st.session_state["race_pacing_aid_km"]
        )

    st.subheader("Éditeur de segments")

    # Manual merge section
    with st.expander("🔀 Fusionner des segments manuellement", expanded=False):
        st.caption("Sélectionnez deux segments ou plus adjacents à fusionner")
        
        # Initialize selection state
        if "merge_selection" not in st.session_state:
            st.session_state["merge_selection"] = []
        
        # Initialize merge counter to force checkbox recreation after merge
        if "merge_counter" not in st.session_state:
            st.session_state["merge_counter"] = 0
        
        # Create checkboxes for segment selection in columns
        num_cols = min(6, len(segments_df))
        merge_cols = st.columns(num_cols)
        selected_segments = []
        
        for idx, row in segments_df.iterrows():
            col_idx = idx % num_cols
            seg_id = int(row["segmentId"])
            seg_type = row.get("type", "unknown")
            seg_start = row.get("startKm", 0)
            seg_end = row.get("endKm", 0)
            
            with merge_cols[col_idx]:
                label = f"#{seg_id}: {seg_type} ({seg_start:.1f}-{seg_end:.1f} km)"
                # Include merge_counter in key to force recreation after merge
                checkbox_key = f"merge_checkbox_{seg_id}_{idx}_{st.session_state['merge_counter']}"
                is_selected = st.checkbox(
                    label,
                    value=seg_id in st.session_state.get("merge_selection", []),
                    key=checkbox_key
                )
                
                if is_selected:
                    selected_segments.append(seg_id)
        
        st.session_state["merge_selection"] = selected_segments
        
        if len(selected_segments) >= 2:
            # Verify segments are adjacent
            selected_df = segments_df[segments_df["segmentId"].isin(selected_segments)].sort_values("startKm")
            is_adjacent = True
            for i in range(len(selected_df) - 1):
                current_end = selected_df.iloc[i]["endKm"]
                next_start = selected_df.iloc[i + 1]["startKm"]
                if abs(current_end - next_start) > 0.01:
                    is_adjacent = False
                    break
            
            if is_adjacent:
                merge_button_label = f"Fusionner {len(selected_segments)} segments"
                if st.button(merge_button_label, key="merge_segments_button"):
                    try:
                        merged_df = pacer_service.merge_segments_manually(
                            segments_df,
                            selected_segments,
                            st.session_state["race_pacing_metrics"]
                        )
                        st.session_state["race_pacing_segments"] = merged_df
                        # Clear selection and increment counter to force checkbox recreation
                        st.session_state["merge_selection"] = []
                        st.session_state["merge_counter"] = st.session_state.get("merge_counter", 0) + 1
                        st.success(f"{len(selected_segments)} segments fusionnés avec succès!")
                        # Force rerun to update checkboxes
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Failed to merge segments: {e}", exc_info=True)
                        st.error(f"Erreur lors de la fusion: {str(e)}")
                        # Clear selection even on error
                        st.session_state["merge_selection"] = []
            else:
                st.warning(
                    "⚠️ Les segments sélectionnés doivent être adjacents pour être fusionnés."
                )
        elif len(selected_segments) == 1:
            st.info("ℹ️ Sélectionnez au moins 2 segments adjacents pour fusionner.")
        else:
            st.info("ℹ️ Sélectionnez des segments à fusionner en cochant les cases ci-dessus.")

    # Prepare editable DataFrame - keep numeric values for editor
    editable_cols = ["speedEqKmh", "speedKmh"]
    read_only_cols = [
        "segmentId", "type", "startKm", "endKm", "distanceKm",
        "elevGainM", "elevLossM", "distanceEqKm", "isAidSplit", "timeSec"
    ]

    # Ensure all columns exist
    available_cols = [col for col in read_only_cols + editable_cols if col in segments_df.columns]
    display_df = segments_df[available_cols].copy()

    # Add avgGrade if available (convert to percentage for display)
    if "avgGrade" in segments_df.columns:
        display_df["avgGrade"] = segments_df["avgGrade"] * 100  # Convert to percentage

    # Editor config
    column_config = {}
    if "segmentId" in display_df.columns:
        column_config["segmentId"] = st.column_config.NumberColumn("ID", disabled=True, format="%d")
    if "type" in display_df.columns:
        column_config["type"] = st.column_config.TextColumn("Type", disabled=True)
    if "startKm" in display_df.columns:
        column_config["startKm"] = st.column_config.NumberColumn("Début (km)", disabled=True, format="%.2f")
    if "endKm" in display_df.columns:
        column_config["endKm"] = st.column_config.NumberColumn("Fin (km)", disabled=True, format="%.2f")
    if "distanceKm" in display_df.columns:
        column_config["distanceKm"] = st.column_config.NumberColumn("Distance (km)", disabled=True, format="%.2f")
    if "distanceEqKm" in display_df.columns:
        column_config["distanceEqKm"] = st.column_config.NumberColumn("Dist. Eq. (km)", disabled=True, format="%.2f")
    if "elevGainM" in display_df.columns:
        column_config["elevGainM"] = st.column_config.NumberColumn("D+ (m)", disabled=True, format="%.0f")
    if "elevLossM" in display_df.columns:
        column_config["elevLossM"] = st.column_config.NumberColumn("D- (m)", disabled=True, format="%.0f")
    if "avgGrade" in display_df.columns:
        column_config["avgGrade"] = st.column_config.NumberColumn("Pente (%)", disabled=True, format="%.1f")
    if "isAidSplit" in display_df.columns:
        column_config["isAidSplit"] = st.column_config.CheckboxColumn("RAV", disabled=True)
    if "speedEqKmh" in display_df.columns:
        column_config["speedEqKmh"] = st.column_config.NumberColumn(
            "Vitesse Eq. (km/h)", min_value=0.0, step=0.1, format="%.1f"
        )
    if "speedKmh" in display_df.columns:
        column_config["speedKmh"] = st.column_config.NumberColumn(
            "Vitesse (km/h)", min_value=0.0, step=0.1, format="%.1f"
        )
    if "timeSec" in display_df.columns:
        column_config["timeSec"] = st.column_config.NumberColumn("Temps (s)", disabled=True, format="%d")

    # Speed by type configuration section
    if not segments_df.empty and "type" in segments_df.columns:
        with st.expander("⚙️ Définir les vitesses par type de segment", expanded=False):
            st.caption(
                "Configurez les vitesses pour chaque type de segment. "
                "Cliquez sur 'Appliquer' pour mettre à jour tous les segments correspondants."
            )
            
            # Get unique segment types
            unique_types = sorted(segments_df["type"].unique().tolist())
            
            # Initialize session state for speed by type if not exists
            if "speed_by_type" not in st.session_state:
                st.session_state["speed_by_type"] = {}
            
            # Create form for each type
            type_configs = {}
            for seg_type in unique_types:
                st.markdown(f"**{seg_type}**")
                col_type1, col_type2, col_type3 = st.columns([2, 2, 1])
                
                with col_type1:
                    speed_mode_key = f"speed_mode_{seg_type}"
                    speed_mode = st.selectbox(
                        "Type de vitesse",
                        ["speedEqKmh", "speedKmh"],
                        index=0 if st.session_state.get(speed_mode_key, "speedEqKmh") == "speedEqKmh" else 1,
                        key=speed_mode_key,
                        label_visibility="collapsed"
                    )
                    st.caption("Vitesse Eq." if speed_mode == "speedEqKmh" else "Vitesse")
                
                with col_type2:
                    speed_value_key = f"speed_value_{seg_type}"
                    # Get current average speed for this type or default to 0
                    type_segments = segments_df[segments_df["type"] == seg_type]
                    if speed_mode == "speedEqKmh":
                        default_speed = float(type_segments["speedEqKmh"].mean()) if not type_segments.empty else 0.0
                    else:
                        default_speed = float(type_segments["speedKmh"].mean()) if not type_segments.empty else 0.0
                    
                    speed_value = st.number_input(
                        "Valeur (km/h)",
                        min_value=0.0,
                        step=0.1,
                        value=float(st.session_state.get(speed_value_key, default_speed)),
                        key=speed_value_key,
                        label_visibility="collapsed"
                    )
                    st.caption(f"{speed_value:.1f} km/h")
                
                with col_type3:
                    count = len(type_segments)
                    st.caption(f"{count} segment{'s' if count > 1 else ''}")
                
                type_configs[seg_type] = {
                    "mode": speed_mode,
                    "value": speed_value
                }
            
            # Button to apply speeds by type
            if st.button("✅ Appliquer les vitesses par type", key="apply_speeds_by_type_button", type="primary"):
                if not segments_df.empty:
                    updated_segments = segments_df.copy()
                    
                    # Apply speeds to segments based on their type
                    for seg_type, config in type_configs.items():
                        type_mask = updated_segments["type"] == seg_type
                        if config["mode"] == "speedEqKmh":
                            updated_segments.loc[type_mask, "speedEqKmh"] = config["value"]
                        else:
                            updated_segments.loc[type_mask, "speedKmh"] = config["value"]
                    
                    # Recompute all times based on updated speeds
                    for idx, row in updated_segments.iterrows():
                        speed_eq = float(row.get("speedEqKmh", 0) or 0)
                        speed = float(row.get("speedKmh", 0) or 0)
                        distance_eq = float(row.get("distanceEqKm", 0) or 0)
                        distance = float(row.get("distanceKm", 0) or 0)
                        
                        time_sec = pacer_service.compute_segment_time(distance_eq, distance, speed_eq, speed)
                        updated_segments.at[idx, "timeSec"] = time_sec
                    
                    st.session_state["race_pacing_segments"] = updated_segments
                    st.success("Vitesses appliquées et temps recalculés pour tous les segments!")
                    st.rerun()

    edited_df = st.data_editor(display_df, column_config=column_config, hide_index=True, key="segments_editor")

    # Button to update speeds from editor and recompute all times
    # IMPORTANT: No automatic updates happen when editing - only when button is clicked
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("🔄 Recalculer les temps", key="recompute_times_button", type="primary"):
            if not edited_df.empty and not segments_df.empty:
                # Use edited_df as the source of truth - it contains ALL current values from editor
                updated_segments = segments_df.copy()
                
                # Copy all speed values from edited_df (which has all user edits)
                for idx in range(min(len(edited_df), len(updated_segments))):
                    updated_segments.at[idx, "speedEqKmh"] = float(edited_df.iloc[idx].get("speedEqKmh", 0) or 0)
                    updated_segments.at[idx, "speedKmh"] = float(edited_df.iloc[idx].get("speedKmh", 0) or 0)
                
                # Recompute all times based on updated speeds
                for idx, row in updated_segments.iterrows():
                    speed_eq = float(row.get("speedEqKmh", 0) or 0)
                    speed = float(row.get("speedKmh", 0) or 0)
                    distance_eq = float(row.get("distanceEqKm", 0) or 0)
                    distance = float(row.get("distanceKm", 0) or 0)
                    
                    time_sec = pacer_service.compute_segment_time(distance_eq, distance, speed_eq, speed)
                    updated_segments.at[idx, "timeSec"] = time_sec
                
                st.session_state["race_pacing_segments"] = updated_segments
                st.success("Temps recalculés pour tous les segments!")
                st.rerun()

    # Display expected times at aid stations
    aid_stations_km = st.session_state.get("race_pacing_aid_km", [])
    if aid_stations_km and not segments_df.empty:
        st.subheader("⏱️ Temps attendus aux ravitaillements")
        
        # Calculate cumulative time up to each aid station using service method
        aid_times_sec = pacer_service.compute_aid_station_times(aid_stations_km, segments_df)
        
        # Calculate statistics for each segment between aid stations
        segment_stats = pacer_service.compute_aid_station_stats(aid_stations_km, segments_df)
        
        # Store computed times in session state
        st.session_state["race_pacing_aid_times"] = aid_times_sec
        
        sorted_aid_km = sorted(aid_stations_km)
        
        # Compute cumulative stats at each aid station
        cumulative_stats = pacer_service.compute_cumulative_stats_at_aid_stations(aid_stations_km, segments_df)
        
        # Display aid station times with stats since last aid station and cumulative stats
        if aid_times_sec:
            num_cols = min(5, len(aid_times_sec))
            aid_cols = st.columns(num_cols)
            for idx, (aid_km, aid_time) in enumerate(zip(sorted_aid_km, aid_times_sec)):
                col_idx = idx % num_cols
                with aid_cols[col_idx]:
                    # Get stats for this segment (from previous aid station or start)
                    if idx < len(segment_stats):
                        seg_stats = segment_stats[idx]
                        dist_eq = seg_stats.get("distanceEqKm", 0.0)
                        elev_gain = seg_stats.get("elevGainM", 0.0)
                        elev_loss = seg_stats.get("elevLossM", 0.0)
                        
                        # Add cumulative stats if available
                        cum_dist_eq = 0.0
                        cum_elev_gain = 0.0
                        cum_elev_loss = 0.0
                        if idx < len(cumulative_stats):
                            cum_stats = cumulative_stats[idx]
                            cum_dist_eq = cum_stats.get("distanceEqKm", 0.0)
                            cum_elev_gain = cum_stats.get("elevGainM", 0.0)
                            cum_elev_loss = cum_stats.get("elevLossM", 0.0)
                        
                        # Create custom metric card with all info inside
                        depuis_text = (
                            f"Depuis dernier RAV: Dist-eq: {fmt_decimal(dist_eq, 1)} km | "
                            f"D+: {fmt_decimal(elev_gain, 0)} m | D-: {fmt_decimal(elev_loss, 0)} m"
                        )
                        cumule_text = (
                            f"<strong style='color: #e4cca0;'>Cumulé depuis départ:</strong> "
                            f"Dist-eq: {fmt_decimal(cum_dist_eq, 1)} km | "
                            f"D+: {fmt_decimal(cum_elev_gain, 0)} m | "
                            f"D-: {fmt_decimal(cum_elev_loss, 0)} m"
                        )
                        metric_html = f"""
                        <div style="
                            background: rgba(12, 20, 33, 0.55);
                            border-radius: 16px;
                            padding: 0.75rem 1.1rem;
                            border: 1px solid rgba(228, 204, 160, 0.18);
                            box-shadow: 0 18px 30px rgba(8, 14, 24, 0.38);
                        ">
                            <div style="font-size: 0.875rem; color: #e4cca0; margin-bottom: 0.5rem;">
                                RAV {idx + 1}
                            </div>
                            <div style="font-size: 2rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.5rem;">
                                {format_session_duration(int(aid_time))}
                            </div>
                            <div style="
                                display: flex; align-items: center; gap: 0.5rem;
                                margin-bottom: 0.75rem; flex-wrap: wrap;
                            ">
                                <span style="color: #22c55e; font-weight: 500; font-size: 0.9rem;">
                                    → {fmt_decimal(aid_km, 1)} km
                                </span>
                                <span style="font-size: 0.7rem; color: #d4acb4; margin-left: auto;">
                                    {depuis_text}
                                </span>
                            </div>
                            <div style="
                                font-size: 0.7rem; color: #d4acb4;
                                border-top: 1px solid rgba(228, 204, 160, 0.15);
                                padding-top: 0.5rem;
                            ">
                                {cumule_text}
                            </div>
                        </div>
                        """
                        st.markdown(metric_html, unsafe_allow_html=True)
                    else:
                        st.metric(
                            f"RAV {idx + 1}",
                            format_session_duration(int(aid_time)),
                            delta=f"→ {fmt_decimal(aid_km, 1)} km"
                        )
        
        # Display table with statistics between each aid station
        st.subheader("📊 Statistiques entre ravitaillements")
        if segment_stats:
            table_data = []
            for idx, seg_stats in enumerate(segment_stats):
                if idx == 0:
                    from_label = "Départ"
                    to_label = f"RAV 1 ({fmt_decimal(sorted_aid_km[0], 1)} km)"
                else:
                    from_label = f"RAV {idx} ({fmt_decimal(sorted_aid_km[idx - 1], 1)} km)"
                    to_label = f"RAV {idx + 1} ({fmt_decimal(sorted_aid_km[idx], 1)} km)"
                
                seg_time = seg_stats.get("timeSec", 0.0)
                table_data.append({
                    "De": from_label,
                    "À": to_label,
                    "Distance (km)": fmt_decimal(seg_stats.get("distanceKm", 0.0), 2),
                    "Dist-eq (km)": fmt_decimal(seg_stats.get("distanceEqKm", 0.0), 2),
                    "D+ (m)": fmt_decimal(seg_stats.get("elevGainM", 0.0), 0),
                    "D- (m)": fmt_decimal(seg_stats.get("elevLossM", 0.0), 0),
                    "Temps": format_session_duration(int(seg_time)),
                })
            
            # Add final segment to end if there are segments after last aid station
            if sorted_aid_km:
                final_km = sorted_aid_km[-1]
                total_distance = float(segments_df["endKm"].max() if not segments_df.empty else 0.0)
                if total_distance > final_km:
                    final_stats = pacer_service.compute_segment_stats_between(
                        final_km, total_distance, segments_df
                    )
                    final_time = pacer_service._compute_time_between(final_km, total_distance, segments_df)
                    table_data.append({
                        "De": f"RAV {len(sorted_aid_km)} ({fmt_decimal(final_km, 1)} km)",
                        "À": f"Arrivée ({fmt_decimal(total_distance, 1)} km)",
                        "Distance (km)": fmt_decimal(final_stats.get("distanceKm", 0.0), 2),
                        "Dist-eq (km)": fmt_decimal(final_stats.get("distanceEqKm", 0.0), 2),
                        "D+ (m)": fmt_decimal(final_stats.get("elevGainM", 0.0), 0),
                        "D- (m)": fmt_decimal(final_stats.get("elevLossM", 0.0), 0),
                        "Temps": format_session_duration(int(final_time)),
                    })
            
            stats_df = pd.DataFrame(table_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        st.divider()

    # Display summary
    summary = pacer_service.aggregate_summary(segments_df)
    col_sum1, col_sum2, col_sum3, col_sum4, col_sum5 = st.columns(5)
    with col_sum1:
        st.metric("Distance", fmt_decimal(summary["distanceKm"], 1) + " km")
    with col_sum2:
        st.metric("Dist. Eq.", fmt_decimal(summary["distanceEqKm"], 1) + " km")
    with col_sum3:
        st.metric("D+", fmt_decimal(summary["elevGainM"], 0) + " m")
    with col_sum4:
        st.metric("D-", fmt_decimal(summary["elevLossM"], 0) + " m")
    with col_sum5:
        st.metric("Temps total", format_session_duration(summary["timeSec"]))

    # Save button
    if st.button("Enregistrer la course", key="save_race_button"):
        if not race_name or not race_name.strip():
            st.warning("Veuillez entrer un nom de course.")
        else:
            try:
                # Get computed aid station times (or compute if not available)
                aid_times = st.session_state.get("race_pacing_aid_times")
                if aid_times is None and aid_stations_km and not segments_df.empty:
                    aid_times = pacer_service.compute_aid_station_times(
                        st.session_state["race_pacing_aid_km"], segments_df
                    )
                
                race_id = pacer_service.save_race(
                    race_name.strip(),
                    st.session_state["race_pacing_aid_km"],
                    segments_df,
                    st.session_state.get("race_pacing_race_id"),
                    aid_stations_times=aid_times,
                )
                st.session_state["race_pacing_race_id"] = race_id
                st.success(f"Course '{race_name}' enregistrée avec succès!")
                st.rerun()
            except Exception as e:
                logger.error(f"Failed to save race: {e}", exc_info=True)
                st.error(f"Erreur lors de l'enregistrement: {str(e)}")

