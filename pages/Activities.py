from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from utils.config import load_config
from utils.formatting import fmt_decimal, fmt_km, fmt_m, fmt_speed_kmh, set_locale
from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo, ThresholdsRepo
from services.lap_metrics_service import LapMetricsService
from services.linking_service import LinkingService
from services.timeseries_service import TimeseriesService


st.set_page_config(page_title="Running Manager - Activities", layout="wide")
st.title("Activités")


def _trigger_rerun() -> None:
    rerun = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun:
        rerun()


def _format_duration(seconds: Optional[float]) -> str:
    if seconds in (None, "") or pd.isna(seconds):
        return "-"
    total = int(float(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}"
    if minutes:
        return f"{minutes}m{secs:02d}"
    return f"{secs}s"


cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
ath_repo = AthletesRepo(storage)
thresholds_repo = ThresholdsRepo(storage)
link_service = LinkingService(storage)
ts_service = TimeseriesService(cfg)
lap_metrics_service = LapMetricsService(storage, cfg)
_activities_metrics_df = storage.read_csv("activities_metrics.csv")
if not _activities_metrics_df.empty:
    _activities_metrics_df["activityId"] = _activities_metrics_df["activityId"].astype(str)
    ACTIVITY_METRICS_MAP = _activities_metrics_df.set_index("activityId").to_dict(orient="index")
else:
    ACTIVITY_METRICS_MAP = {}

_planned_metrics_df = storage.read_csv("planned_metrics.csv")
if not _planned_metrics_df.empty:
    _planned_metrics_df["plannedSessionId"] = _planned_metrics_df["plannedSessionId"].astype(str)
    PLANNED_METRICS_MAP = _planned_metrics_df.set_index("plannedSessionId").to_dict(orient="index")
else:
    PLANNED_METRICS_MAP = {}
_RAW_CACHE: Dict[str, Optional[Dict[str, object]]] = {}
ALLOWED_TYPES = {"run", "hike", "trailrun", "trail running"}
ALLOWED_TYPES_NORMALIZED = {t.replace(" ", "").lower() for t in ALLOWED_TYPES}


ath_df = ath_repo.list()
if ath_df.empty:
    st.warning("Aucun athlète disponible.")
    st.stop()

ath_options = {
    f"{row.get('name') or 'Sans nom'} ({row.get('athleteId')})": row.get("athleteId")
    for _, row in ath_df.iterrows()
}
ath_label = st.selectbox("Athlète", list(ath_options.keys()))
athlete_id = ath_options.get(ath_label)

if not athlete_id:
    st.stop()

athlete_row = ath_df[ath_df["athleteId"] == athlete_id].iloc[0]
thresholds_df = thresholds_repo.list(athleteId=athlete_id)

activities = storage.read_csv("activities.csv")
if activities.empty:
    st.info("Aucune activité enregistrée pour le moment.")
    st.stop()

activities = activities[activities["athleteId"] == athlete_id]
if activities.empty:
    st.info("Aucune activité pour cet athlète.")
    st.stop()

activities["startTime"] = pd.to_datetime(activities["startTime"], errors="coerce")
activities = activities.sort_values("startTime", ascending=False).reset_index(drop=True)


def _load_raw_detail(row: pd.Series) -> Optional[Dict[str, object]]:
    path_str = row.get("rawJsonPath")
    if not path_str:
        return None
    path = Path(cfg.data_dir) / Path(str(path_str))
    key = str(path.resolve())
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]
    if not path.exists():
        _RAW_CACHE[key] = None
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        data = None
    _RAW_CACHE[key] = data
    return data


def _activity_type(row: pd.Series) -> Optional[str]:
    detail = _load_raw_detail(row)
    if not detail:
        return None
    type_value = detail.get("type")
    if not type_value and "sport_type" in detail:
        type_value = detail.get("sport_type")
    if isinstance(type_value, str):
        return type_value
    return None


def _activity_name(row: pd.Series) -> str:
    detail = _load_raw_detail(row)
    if not detail:
        return ""
    name = detail.get("name")
    return str(name) if name else ""


unlinked_df = link_service.unlinked_activities(athlete_id)
if not unlinked_df.empty:
    unlinked_df = unlinked_df[
        unlinked_df.apply(
            lambda r: (_activity_type(r) or "").lower().replace(" ", "") in ALLOWED_TYPES_NORMALIZED,
            axis=1,
        )
    ].reset_index(drop=True)
linked_df = link_service.linked_activities(athlete_id)
if not linked_df.empty:
    linked_df = linked_df[
        linked_df.apply(
            lambda r: (_activity_type(r) or "").lower().replace(" ", "") in ALLOWED_TYPES_NORMALIZED,
            axis=1,
        )
    ].reset_index(drop=True)


def _activity_label(row: pd.Series) -> str:
    date = row.get("startTime")
    if pd.notna(date):
        date = pd.to_datetime(date).strftime("%Y-%m-%d")
    else:
        date = "???"
    dist = row.get("distanceKm")
    distance_label = fmt_km(float(dist)) if dist not in (None, "") and not pd.isna(dist) else ""
    title = _activity_name(row) or row.get("source") or "activité"
    return f"{date} • {distance_label} • {title}"


def _planned_label(row: pd.Series) -> str:
    date = row.get("date") or row.get("plannedDate")
    if date:
        try:
            date = pd.to_datetime(date).strftime("%Y-%m-%d")
        except Exception:
            date = str(date)
    else:
        date = "???"
    typ = row.get("type") or row.get("plannedType") or ""
    dist = row.get("plannedDistanceKm")
    dist_lbl = fmt_decimal(float(dist), 1) + " km" if dist not in (None, "") and not pd.isna(dist) else ""
    dur = row.get("plannedDurationSec")
    dur_lbl = _format_duration(dur) if dur not in (None, "") and not pd.isna(dur) else ""
    parts = [date, typ, dist_lbl, dur_lbl]
    return " • ".join([p for p in parts if p])


def _render_summary(activity: pd.Series) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Distance", fmt_km(_coerce_float(activity.get("distanceKm"))))
    with col2:
        st.metric("Temps en mouvement", _format_duration(activity.get("movingSec")))
    with col3:
        st.metric("D+", fmt_m(_coerce_float(activity.get("ascentM"))))
    with col4:
        st.metric("FC moyenne", fmt_decimal(_coerce_float(activity.get("avgHr")), 0))
    name = _activity_name(activity)
    if name:
        st.caption(f"Nom Strava : {name}")


def _coerce_float(value: object) -> Optional[float]:
    try:
        if value in (None, "", "NaN"):
            return None
        return float(value)
    except Exception:
        return None


def _coerce_int(value: object) -> Optional[int]:
    try:
        if value in (None, "", "NaN"):
            return None
        return int(float(value))
    except Exception:
        return None


ATHLETE_HR_REST = _coerce_float(athlete_row.get("hrRest"))
ATHLETE_HR_MAX = _coerce_float(athlete_row.get("hrMax"))


def _fmt_optional(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "-"
    return fmt_decimal(float(value), digits)


def _format_duration_delta(actual_sec: Optional[float], planned_sec: Optional[float]) -> str:
    if actual_sec is None or planned_sec is None:
        return "-"
    delta = float(actual_sec) - float(planned_sec)
    sign = "+" if delta >= 0 else "-"
    formatted = _format_duration(abs(delta))
    return f"{sign}{formatted}"


def _compute_trimp(duration_sec: Optional[float], avg_hr: Optional[float]) -> Optional[float]:
    if (
        duration_sec in (None, "", "NaN")
        or avg_hr in (None, "", "NaN")
        or ATHLETE_HR_REST in (None, "", "NaN")
        or ATHLETE_HR_MAX in (None, "", "NaN")
    ):
        return None
    duration_sec = float(duration_sec)
    avg_hr = float(avg_hr)
    hr_rest = float(ATHLETE_HR_REST)
    hr_max = float(ATHLETE_HR_MAX)
    if hr_max <= hr_rest or duration_sec <= 0 or avg_hr <= hr_rest:
        return None
    hr_reserve = (avg_hr - hr_rest) / (hr_max - hr_rest)
    if hr_reserve <= 0:
        return None
    duration_hours = duration_sec / 3600.0
    factor = 0.64 * math.exp(1.92 * hr_reserve)
    return duration_hours * hr_reserve * factor


def _find_threshold(label: Optional[str]) -> Optional[pd.Series]:
    if thresholds_df.empty:
        return None
    target = label or "Fundamental"
    hit = thresholds_df[thresholds_df["name"] == target]
    if hit.empty and target != "Fundamental":
        hit = thresholds_df[thresholds_df["name"] == "Fundamental"]
    if hit.empty:
        return None
    return hit.iloc[0]


def _threshold_mid_hr(threshold_row: pd.Series) -> Optional[float]:
    if threshold_row is None:
        return None
    hr_min = _coerce_float(threshold_row.get("hrMin"))
    hr_max = _coerce_float(threshold_row.get("hrMax"))
    if hr_min is None or hr_max is None or hr_max <= 0:
        return None
    if hr_min <= 0 and hr_max > 0:
        return hr_max
    return (hr_min + hr_max) / 2.0


def _threshold_avg_pace(threshold_row: pd.Series) -> Optional[float]:
    if threshold_row is None:
        return None
    pace_min = _coerce_float(threshold_row.get("paceFlatKmhMin"))
    pace_max = _coerce_float(threshold_row.get("paceFlatKmhMax"))
    if pace_min and pace_max:
        return (pace_min + pace_max) / 2.0
    if pace_min:
        return pace_min
    if pace_max:
        return pace_max
    return None
def _render_timeseries(activity_id: str) -> None:
    ts_df = ts_service.load(activity_id)
    if ts_df is None:
        st.info("Pas de données timeseries pour cette activité.")
        return
    if "timestamp" not in ts_df.columns:
        st.info("Timeseries incomplète.")
        return
    ts_df = ts_df.copy()
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"], errors="coerce")
    ts_df = ts_df.dropna(subset=["timestamp"])
    if ts_df.empty:
        st.info("Timeseries incomplète.")
        return
    ts_df = ts_df.sort_values("timestamp")
    start = ts_df["timestamp"].iloc[0]
    ts_df["minutes"] = (ts_df["timestamp"] - start).dt.total_seconds() / 60.0

    if "hr" in ts_df.columns and ts_df["hr"].notna().any():
        hr_chart = (
            alt.Chart(ts_df)
            .mark_line(color="#ef4444")
            .encode(x=alt.X("minutes:Q", title="Temps (min)"), y=alt.Y("hr:Q", title="FC (bpm)"))
            .properties(height=180)
        )
        st.altair_chart(hr_chart, use_container_width=True)

    if "paceKmh" in ts_df.columns and ts_df["paceKmh"].notna().any():
        pace_chart = (
            alt.Chart(ts_df)
            .mark_line(color="#3b82f6")
            .encode(
                x=alt.X("minutes:Q", title="Temps (min)"),
                y=alt.Y("paceKmh:Q", title="Vitesse (km/h)"),
            )
            .properties(height=180)
        )
        st.altair_chart(pace_chart, use_container_width=True)

    if "elevationM" in ts_df.columns and ts_df["elevationM"].notna().any():
        elevation_chart = (
            alt.Chart(ts_df)
            .mark_area(color="#10b981", opacity=0.4)
            .encode(
                x=alt.X("minutes:Q", title="Temps (min)"),
                y=alt.Y("elevationM:Q", title="Altitude (m)"),
            )
            .properties(height=160)
        )
        st.altair_chart(elevation_chart, use_container_width=True)


def _format_lap_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "#",
                "Phase",
                "Segment",
                "Distance (km)",
                "Dist. équiv. (km)",
                "Durée",
                "TRIMP",
                "FC moy",
                "FC max",
                "% FC max",
                "Vitesse (km/h)",
            ]
        )

    def _series(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series([None] * len(df), index=df.index)

    def _format_numeric(series: pd.Series, digits: int) -> List[str]:
        return series.map(lambda v: _fmt_optional(v, digits)).tolist()

    lap_numbers = pd.to_numeric(_series("lapIndex"), errors="coerce").fillna(0).astype(int).tolist()
    labels = _series("label").fillna("Recovery").astype(str).tolist()
    names = _series("name").fillna("").astype(str).tolist()
    splits = _series("split").fillna("").astype(str).tolist()
    segments: List[str] = []
    for name_value, split_value in zip(names, splits):
        label = name_value.strip()
        if not label:
            label = split_value.strip()
        segments.append(label)

    distance_km = _format_numeric(_series("distanceKm"), 2)
    distance_eq_km = _format_numeric(_series("distanceEqKm"), 2)
    durations = _series("timeSec").map(_format_duration).tolist()
    trimp = _format_numeric(_series("trimp"), 1)
    avg_hr = _format_numeric(_series("avgHr"), 0)
    max_hr = _format_numeric(_series("maxHr"), 0)
    hr_percent = _format_numeric(_series("hrPercentMax"), 0)
    speeds = _format_numeric(_series("avgSpeedKmh"), 1)

    return pd.DataFrame(
        {
            "#": lap_numbers,
            "Phase": labels,
            "Segment": segments,
            "Distance (km)": distance_km,
            "Dist. équiv. (km)": distance_eq_km,
            "Durée": durations,
            "TRIMP": trimp,
            "FC moy": avg_hr,
            "FC max": max_hr,
            "% FC max": hr_percent,
            "Vitesse (km/h)": speeds,
        }
    )


def _render_laps(activity_id: str) -> None:
    lap_df = lap_metrics_service.load(activity_id)
    if lap_df is None:
        st.caption("Aucune donnée d'intervalles disponible.")
        return
    if lap_df.empty:
        st.caption("Aucun intervalle détecté pour cette activité.")
        return
    lap_df = lap_df.sort_values("lapIndex").reset_index(drop=True)
    table = _format_lap_table(lap_df)
    st.dataframe(table, use_container_width=True, hide_index=True)


def _suggestions_table(suggestions: List[Dict[str, object]]) -> None:
    if not suggestions:
        st.caption("Aucune suggestion disponible.")
        return
    df = pd.DataFrame(suggestions)
    df["score"] = df["score"].map(lambda v: fmt_decimal(_coerce_float(v), 2))
    if "plannedDistanceKm" in df.columns:
        df["plannedDistanceKm"] = df["plannedDistanceKm"].map(
            lambda v: fmt_decimal(_coerce_float(v), 1) if _coerce_float(v) is not None else ""
        )
    if "plannedDurationSec" in df.columns:
        df["plannedDurationSec"] = df["plannedDurationSec"].map(
            lambda v: _format_duration(_coerce_int(v))
        )
    st.dataframe(
        df.rename(
            columns={
                "plannedSessionId": "Session",
                "date": "Date",
                "type": "Type",
                "plannedDistanceKm": "Distance planifiée (km)",
                "plannedDurationSec": "Durée planifiée",
                "score": "Score",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_comparison(selected_row: pd.Series) -> None:
    planned_id = str(selected_row.get("plannedSessionId") or "")
    planned_row = link_service.sessions.get(planned_id) if planned_id else None

    activity_metrics = ACTIVITY_METRICS_MAP.get(str(selected_row.get("activityId") or ""))
    planned_metrics = PLANNED_METRICS_MAP.get(planned_id) if planned_id else None

    planned_distance = _coerce_float(
        selected_row.get("plannedDistanceKm")
        if selected_row.get("plannedDistanceKm") not in (None, "")
        else (planned_row or {}).get("plannedDistanceKm")
    )
    actual_distance = _coerce_float(selected_row.get("distanceKm"))

    planned_distance_eq = _coerce_float(
        (planned_metrics or {}).get("distanceEqKm")
        if planned_metrics and planned_metrics.get("distanceEqKm") not in (None, "")
        else (planned_row or {}).get("plannedDistanceEqKm")
    )
    actual_distance_eq = _coerce_float(
        (activity_metrics or {}).get("distanceEqKm")
    )

    planned_duration = _coerce_float(
        selected_row.get("plannedDurationSec")
        if selected_row.get("plannedDurationSec") not in (None, "")
        else (planned_row or {}).get("plannedDurationSec")
    )
    actual_duration = _coerce_float(selected_row.get("movingSec") or selected_row.get("elapsedSec"))

    target_label = (
        selected_row.get("targetLabel")
        if selected_row.get("targetLabel") not in (None, "")
        else (planned_row or {}).get("targetLabel")
    )
    threshold_row = _find_threshold(target_label)
    if planned_duration is None and planned_distance is not None:
        pace = _threshold_avg_pace(threshold_row)
        if pace and pace > 0:
            planned_duration = (planned_distance / pace) * 3600.0

    planned_hr = _threshold_mid_hr(threshold_row)
    planned_trimp = _coerce_float((planned_metrics or {}).get("trimp"))
    if planned_trimp is None:
        planned_trimp = _compute_trimp(planned_duration, planned_hr)
    actual_trimp = _coerce_float((activity_metrics or {}).get("trimp"))
    if actual_trimp is None:
        actual_trimp = _compute_trimp(actual_duration, _coerce_float(selected_row.get("avgHr")))

    rows = [
        {
            "Métrique": "Distance (km)",
            "Planifié": _fmt_optional(planned_distance, 1),
            "Réalisé": _fmt_optional(actual_distance, 1),
            "Écart": _fmt_optional(
                (actual_distance - planned_distance) if (actual_distance is not None and planned_distance is not None) else None,
                1,
            ),
        },
        {
            "Métrique": "Distance équivalente (km)",
            "Planifié": _fmt_optional(planned_distance_eq, 1),
            "Réalisé": _fmt_optional(actual_distance_eq, 1),
            "Écart": _fmt_optional(
                (actual_distance_eq - planned_distance_eq)
                if (actual_distance_eq is not None and planned_distance_eq is not None)
                else None,
                1,
            ),
        },
        {
            "Métrique": "Durée",
            "Planifié": _format_duration(planned_duration) if planned_duration is not None else "-",
            "Réalisé": _format_duration(actual_duration) if actual_duration is not None else "-",
            "Écart": _format_duration_delta(actual_duration, planned_duration),
        },
        {
            "Métrique": "TRIMP",
            "Planifié": _fmt_optional(planned_trimp, 1),
            "Réalisé": _fmt_optional(actual_trimp, 1),
            "Écart": _fmt_optional(
                (actual_trimp - planned_trimp) if (actual_trimp is not None and planned_trimp is not None) else None,
                1,
            ),
        },
    ]
    st.table(pd.DataFrame(rows))


tab_unlinked, tab_linked = st.tabs(["Activités non liées", "Activités liées"])


with tab_unlinked:
    st.subheader("Activités non liées")
    if unlinked_df.empty:
        st.info("Toutes les activités disponibles sont déjà liées.")
    else:
        activity_ids = unlinked_df["activityId"].astype(str).tolist()
        labels = {aid: _activity_label(unlinked_df.iloc[idx]) for idx, aid in enumerate(activity_ids)}
        selected_id = st.selectbox(
            "Sélectionnez une activité",
            activity_ids,
            format_func=lambda aid: labels[aid],
        )
        selected_row = unlinked_df[unlinked_df["activityId"].astype(str) == selected_id].iloc[0]
        _render_summary(selected_row)
        st.markdown("#### Détails")
        start_raw = selected_row.get("startTime")
        start_label = ""
        if pd.notna(start_raw):
            try:
                start_label = pd.to_datetime(start_raw).strftime("%Y-%m-%d %H:%M")
            except Exception:
                start_label = str(start_raw)
        moving_total = _format_duration(selected_row.get("elapsedSec"))
        moving_sec = _coerce_float(selected_row.get("movingSec"))
        distance_km = _coerce_float(selected_row.get("distanceKm"))
        avg_speed = None
        if moving_sec and moving_sec > 0 and distance_km:
            avg_speed = distance_km / (moving_sec / 3600.0)
        st.write(
            {
                "Date de départ": start_label,
                "Source": selected_row.get("source"),
                "Type": _activity_type(selected_row) or "-",
                "Nom Strava": _activity_name(selected_row) or "-",
                "Durée totale": moving_total,
                "Vitesse moyenne": fmt_speed_kmh(avg_speed),
            }
        )
        st.markdown("#### Timeseries")
        _render_timeseries(selected_id)
        st.markdown("#### Intervalles")
        _render_laps(selected_id)

        suggestions = link_service.suggest_for_activity(athlete_id, selected_row)
        st.markdown("#### Suggestions")
        _suggestions_table(suggestions)

        planned_df = link_service.available_planned_sessions(athlete_id)
        if planned_df.empty:
            st.info("Aucune séance planifiée disponible pour établir un lien.")
        else:
            planned_ids = planned_df["plannedSessionId"].astype(str).tolist()
            planned_labels = {
                sid: _planned_label(planned_df.iloc[idx]) for idx, sid in enumerate(planned_ids)
            }
            default_selection = None
            if suggestions:
                first = suggestions[0]["plannedSessionId"]
                if first in planned_labels:
                    default_selection = first
            default_index = planned_ids.index(default_selection) if default_selection in planned_ids else 0

            with st.form(f"link-form-{selected_id}"):
                chosen_session = st.selectbox(
                    "Séance planifiée",
                    planned_ids,
                    index=default_index,
                    format_func=lambda sid: planned_labels[sid],
                )
                rpe_value = st.slider("RPE (1-10)", min_value=1, max_value=10, value=5)
                comment_value = st.text_input("Commentaire (optionnel)")
                submitted = st.form_submit_button("Lier l'activité")
                if submitted:
                    try:
                        link_service.create_link(
                            athlete_id,
                            selected_id,
                            chosen_session,
                            rpe=rpe_value,
                            comments=comment_value,
                        )
                        st.success("Activité liée avec succès.")
                        _trigger_rerun()
                    except ValueError as exc:
                        st.error(str(exc))


with tab_linked:
    st.subheader("Activités liées")
    if linked_df.empty:
        st.info("Aucune activité liée pour le moment.")
    else:
        activity_ids = linked_df["activityId"].astype(str).tolist()
        labels = {aid: _activity_label(linked_df.iloc[idx]) for idx, aid in enumerate(activity_ids)}
        selected_id = st.selectbox(
            "Activité liée",
            activity_ids,
            format_func=lambda aid: labels[aid],
        )
        selected_row = linked_df[linked_df["activityId"].astype(str) == selected_id].iloc[0]
        _render_summary(selected_row)
        st.markdown("#### Détails")
        planned_info = {
            "Session planifiée": _planned_label(selected_row),
            "Type activité": _activity_type(selected_row) or "-",
            "Score de correspondance": fmt_decimal(_coerce_float(selected_row.get("matchScore")), 2),
        }
        st.write(planned_info)

        st.markdown("#### Comparaison plan vs réalisé")
        _render_comparison(selected_row)

        st.markdown("#### Timeseries")
        _render_timeseries(selected_id)

        current_rpe = _coerce_int(selected_row.get("rpe")) or 5
        current_comment = selected_row.get("comments") or ""
        link_id = str(selected_row.get("linkId"))
        with st.form(f"edit-link-{link_id}"):
            new_rpe = st.slider("RPE (1-10)", min_value=1, max_value=10, value=int(current_rpe))
            new_comment = st.text_input("Commentaire", value=current_comment)
            submitted = st.form_submit_button("Mettre à jour la liaison")
            if submitted:
                link_service.update_link(link_id, rpe=new_rpe, comments=new_comment)
                st.success("Liaison mise à jour.")
                _trigger_rerun()

        if st.button("Supprimer la liaison", key=f"remove-{link_id}", type="secondary"):
            link_service.delete_link(link_id)
            st.success("Liaison supprimée.")
            _trigger_rerun()
