"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Session form widgets for creating and editing session templates.

Provides reusable form components for different session types.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import streamlit as st

from services.planner_service import PlannerService
from ui.interval_editor import render_interval_editor
from utils.coercion import coerce_float, coerce_int
from utils.formatting import fmt_decimal, fmt_m, format_session_duration


def render_fundamental_form(
    athlete_id: Optional[str],
    payload: Dict[str, Any],
    planner: PlannerService,
) -> Tuple[Dict[str, Any], Optional[float]]:
    """Render form for fundamental endurance session type.

    Args:
        athlete_id: Athlete ID for planning calculations
        payload: Existing session payload data
        planner: PlannerService instance for calculations

    Returns:
        Tuple of (form_result_dict, distance_eq_preview)
    """
    planned_distance_km = payload.get("plannedDistanceKm")
    planned_duration_sec = payload.get("plannedDurationSec")
    planned_ascent_m = payload.get("plannedAscentM")
    target_type = payload.get("targetType") or "pace"

    mode_options = ["distance", "duration"]
    distance_present = coerce_float(planned_distance_km, 0.0) > 0
    duration_present = coerce_int(planned_duration_sec, 0) > 0
    default_index = 0 if distance_present or not duration_present else 1
    mode_choice = st.radio(
        "Mode de saisie",
        mode_options,
        index=default_index,
        format_func=lambda x: "Distance + D+" if x == "distance" else "Durée + D+ mini",
        horizontal=True,
        key="creator-fundamental-mode",
    )

    distance_eq_preview = None
    result: Dict[str, Any] = {}
    athlete_ref = str(athlete_id or "")

    if mode_choice == "distance":
        distance_input = st.number_input(
            "Distance planifiée (km)",
            min_value=0.0,
            value=coerce_float(planned_distance_km, 0.0),
            step=0.1,
            key="creator-fundamental-distance",
        )
        ascent_input = st.number_input(
            "Ascension planifiée (m)",
            min_value=0,
            value=coerce_int(planned_ascent_m, 0),
            step=50,
            key="creator-fundamental-ascent",
        )
        derived = planner.derive_from_distance(athlete_ref, distance_input, ascent_input)
        result["plannedDistanceKm"] = derived["distanceKm"]
        result["plannedDurationSec"] = derived["durationSec"]
        result["plannedAscentM"] = int(ascent_input)
        distance_eq_preview = derived["distanceEqKm"]
        st.caption(
            f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km • "
            f"Durée estimée ≈ {format_session_duration(result['plannedDurationSec'])}"
        )
    else:
        duration_input = st.number_input(
            "Durée planifiée (sec)",
            min_value=0,
            value=coerce_int(planned_duration_sec, 3600),
            step=300,
            key="creator-fundamental-duration",
        )
        ascent_input = st.number_input(
            "Ascension minimale (m)",
            min_value=0,
            value=coerce_int(planned_ascent_m, 0),
            step=50,
            key="creator-fundamental-min-ascent",
        )
        derived = planner.derive_from_duration(athlete_ref, int(duration_input), ascent_input)
        result["plannedDistanceKm"] = derived["distanceKm"]
        result["plannedDurationSec"] = derived["durationSec"]
        result["plannedAscentM"] = int(ascent_input)
        distance_eq_preview = derived["distanceEqKm"]
        st.caption(
            f"Distance estimée ≈ {fmt_decimal(result['plannedDistanceKm'], 1)} km • "
            f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km"
        )

    target_options = ["none", "hr", "pace"]
    selection = target_type if target_type in target_options else "pace"
    chosen = st.selectbox(
        "Cible",
        target_options,
        index=target_options.index(selection),
        key="creator-fundamental-target",
    )
    if chosen == "none":
        result["targetType"] = None
        result["targetLabel"] = None
    else:
        result["targetType"] = chosen
        result["targetLabel"] = "Fundamental"
        st.caption("Seuil fixé automatiquement sur Fundamental.")

    return result, distance_eq_preview


def render_long_run_form(
    athlete_id: Optional[str],
    payload: Dict[str, Any],
    planner: PlannerService,
) -> Tuple[Dict[str, Any], Optional[float]]:
    """Render form for long run session type.

    Args:
        athlete_id: Athlete ID for planning calculations
        payload: Existing session payload data
        planner: PlannerService instance for calculations

    Returns:
        Tuple of (form_result_dict, distance_eq_preview)
    """
    planned_distance_km = payload.get("plannedDistanceKm")
    planned_duration_sec = payload.get("plannedDurationSec")
    planned_ascent_m = payload.get("plannedAscentM")
    target_type = payload.get("targetType")
    target_label = payload.get("targetLabel")

    mode_options = ["distance", "duration"]
    distance_present = coerce_float(planned_distance_km, 0.0) > 0
    duration_present = coerce_int(planned_duration_sec, 0) > 0
    default_index = 0 if distance_present or not duration_present else 1
    mode_choice = st.radio(
        "Mode de saisie",
        mode_options,
        index=default_index,
        format_func=lambda x: "Distance + D+" if x == "distance" else "Durée + D+ mini",
        horizontal=True,
        key="creator-long-mode",
    )

    distance_eq_preview = None
    result: Dict[str, Any] = {}
    athlete_ref = str(athlete_id or "")

    if mode_choice == "distance":
        distance_input = st.number_input(
            "Distance planifiée (km)",
            min_value=0.0,
            value=coerce_float(planned_distance_km, 0.0),
            step=0.5,
            key="creator-long-distance",
        )
        ascent_input = st.number_input(
            "Ascension planifiée (m)",
            min_value=0,
            value=coerce_int(planned_ascent_m, 500),
            step=50,
            key="creator-long-ascent",
        )
        derived = planner.derive_from_distance(athlete_ref, distance_input, ascent_input)
        result["plannedDistanceKm"] = derived["distanceKm"]
        result["plannedDurationSec"] = derived["durationSec"]
        result["plannedAscentM"] = int(ascent_input)
        distance_eq_preview = derived["distanceEqKm"]
        st.caption(
            f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km • "
            f"Durée estimée ≈ {format_session_duration(result['plannedDurationSec'])}"
        )
    else:
        duration_input = st.number_input(
            "Durée planifiée (sec)",
            min_value=0,
            value=coerce_int(planned_duration_sec, 7200),
            step=300,
            key="creator-long-duration",
        )
        ascent_input = st.number_input(
            "Ascension minimale (m)",
            min_value=0,
            value=coerce_int(planned_ascent_m, 500),
            step=50,
            key="creator-long-min-ascent",
        )
        derived = planner.derive_from_duration(athlete_ref, int(duration_input), ascent_input)
        result["plannedDistanceKm"] = derived["distanceKm"]
        result["plannedDurationSec"] = derived["durationSec"]
        result["plannedAscentM"] = int(ascent_input)
        distance_eq_preview = derived["distanceEqKm"]
        st.caption(
            f"Distance estimée ≈ {fmt_decimal(result['plannedDistanceKm'], 1)} km • "
            f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km"
        )

    target_type = target_type if isinstance(target_type, str) else "none"
    target_options = ["none", "hr", "pace"]
    target_choice = st.selectbox(
        "Cible",
        target_options,
        index=(0 if target_type not in target_options else target_options.index(target_type)),
        key="creator-long-target",
    )
    if target_choice in ("hr", "pace"):
        names = (
            planner.list_threshold_names(str(athlete_id or ""))
            if athlete_id
            else ["Fundamental", "Threshold 30", "Threshold 60"]
        )
        idx = names.index(target_label) if target_label in names else 0
        selected_label = st.selectbox("Seuil", names, index=idx, key="creator-long-threshold")
        result["targetType"] = target_choice
        result["targetLabel"] = selected_label
    else:
        result["targetType"] = None
        result["targetLabel"] = None

    return result, distance_eq_preview


def render_race_form(
    athlete_id: Optional[str],
    payload: Dict[str, Any],
    planner: PlannerService,
) -> Tuple[Dict[str, Any], Optional[float]]:
    """Render form for race session type.

    Args:
        athlete_id: Athlete ID (unused but kept for consistency)
        payload: Existing session payload data
        planner: PlannerService instance for calculations

    Returns:
        Tuple of (form_result_dict, distance_eq_preview)
    """
    planned_distance_km = coerce_float(payload.get("plannedDistanceKm"), 0.0)
    planned_duration_sec = coerce_int(payload.get("plannedDurationSec"), 0)
    planned_ascent_m = coerce_int(payload.get("plannedAscentM"), 0)
    race_name = payload.get("raceName") or ""

    distance_input = st.number_input(
        "Distance (km)",
        min_value=0.0,
        value=planned_distance_km,
        step=0.1,
        key="creator-race-distance",
    )
    ascent_input = st.number_input(
        "Ascension (m)",
        min_value=0,
        value=planned_ascent_m,
        step=50,
        key="creator-race-ascent",
    )
    race_name_input = st.text_input(
        "Nom de la course",
        value=str(race_name),
        key="creator-race-name",
    )
    target_time_input = st.number_input(
        "Temps cible (sec)",
        min_value=0,
        value=planned_duration_sec,
        step=60,
        key="creator-race-duration",
    )

    result = {
        "plannedDistanceKm": float(distance_input),
        "plannedAscentM": int(ascent_input),
        "plannedDurationSec": int(target_time_input),
        "targetType": "race",
        "targetLabel": None,
        "raceName": race_name_input.strip(),
    }
    distance_eq_preview = planner.compute_distance_eq_km(
        result["plannedDistanceKm"], result["plannedAscentM"]
    )
    st.caption(
        f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km • "
        f"Temps cible ≈ {format_session_duration(result['plannedDurationSec'])}"
    )
    return result, distance_eq_preview


def render_interval_form(
    athlete_id: Optional[str],
    payload: Dict[str, Any],
    planner: PlannerService,
) -> Dict[str, Any]:
    """Render form for interval session type.

    Args:
        athlete_id: Athlete ID for threshold names
        payload: Existing session payload data
        planner: PlannerService instance for calculations

    Returns:
        Form result dictionary
    """
    thr_names = (
        planner.list_threshold_names(str(athlete_id or ""))
        if athlete_id
        else ["Threshold 60", "Threshold 30", "Fundamental", "MVA", "Max speed"]
    )
    serialized_steps = render_interval_editor("creator", payload.get("stepsJson"), thr_names)
    step_end_mode_default = payload.get("stepEndMode") or "auto"
    step_end_mode = st.selectbox(
        "Mode de fin",
        ["auto", "lap"],
        index=(
            ["auto", "lap"].index(step_end_mode_default)
            if step_end_mode_default in ("auto", "lap")
            else 0
        ),
        key="creator-interval-end-mode",
    )
    planned_duration_sec = planner.estimate_interval_duration_sec(serialized_steps)
    planned_distance_km = planner.estimate_interval_distance_km(
        str(athlete_id or ""), serialized_steps
    )
    planned_ascent_m = planner.estimate_interval_ascent_m(serialized_steps)
    distance_eq_preview = planner.compute_distance_eq_km(planned_distance_km, planned_ascent_m)
    st.caption(
        f"Durée ≈ {format_session_duration(planned_duration_sec)} • "
        f"Distance ≈ {fmt_decimal(planned_distance_km, 1)} km • "
        f"D+ ≈ {fmt_m(planned_ascent_m)} • "
        f"Distance-eq ≈ {fmt_decimal(distance_eq_preview, 1)} km"
    )
    return {
        "stepsJson": json.dumps(serialized_steps, ensure_ascii=False, separators=(",", ":")),
        "stepEndMode": step_end_mode,
        "plannedDurationSec": planned_duration_sec,
        "plannedDistanceKm": planned_distance_km,
        "plannedAscentM": planned_ascent_m,
        "targetType": None,
        "targetLabel": None,
    }

