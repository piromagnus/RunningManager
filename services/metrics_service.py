"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Metrics computation pipeline for activities, weekly, and daily aggregates.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from persistence.csv_storage import CsvStorage
from persistence.repositories import (
    ActivitiesMetricsRepo,
    ActivitiesRepo,
    AthletesRepo,
    DailyMetricsRepo,
    PlannedMetricsRepo,
    PlannedSessionsRepo,
    SettingsRepo,
    WeeklyMetricsRepo,
)
from services.interval_utils import normalize_steps
from services.planner_service import PlannerService
from services.speed_profile_service import SpeedProfileService
from utils.coercion import safe_float, safe_int
from utils.config import load_config
from utils.time import iso_week_start, to_date

PRIMARY_CATEGORIES = {"RUN", "TRAIL_RUN", "HIKE", "RIDE"}
TRAINING_LOAD_CATEGORIES = {"RUN", "TRAIL_RUN", "HIKE"}




@dataclass
class MetricsComputationService:
    storage: CsvStorage

    def __post_init__(self) -> None:
        self.activities = ActivitiesRepo(self.storage)
        self.activity_metrics = ActivitiesMetricsRepo(self.storage)
        self.sessions = PlannedSessionsRepo(self.storage)
        self.planned_metrics = PlannedMetricsRepo(self.storage)
        self.settings = SettingsRepo(self.storage)
        self.weekly_repo = WeeklyMetricsRepo(self.storage)
        self.daily_repo = DailyMetricsRepo(self.storage)
        self.athletes = AthletesRepo(self.storage)
        self.planner = PlannerService(self.storage)
        self.config = load_config()
        self.speed_profile_service = SpeedProfileService(self.config)
        self._bike_eq_cache: Optional[Tuple[float, float, float]] = None

    # ------------------------------------------------------------------
    # Public API
    def recompute_all(self, athlete_id: Optional[str] = None) -> None:
        athlete_ids = list(self._target_athletes(athlete_id))
        self._recompute_for_athletes(athlete_ids, replace_all=athlete_id is None)

    def recompute_athlete(self, athlete_id: str) -> None:
        self._recompute_for_athletes([athlete_id], replace_all=False)

    def recompute_for_activities(self, activity_ids: Sequence[str]) -> None:
        if not activity_ids:
            return
        df = self.activities.list()
        if df.empty or "activityId" not in df.columns:
            return
        activity_ids = {str(aid) for aid in activity_ids}
        subset = df[df["activityId"].astype(str).isin(activity_ids)]
        if subset.empty or "athleteId" not in subset.columns:
            return
        athlete_ids = sorted(set(subset["athleteId"].astype(str)))
        self._recompute_for_athletes(athlete_ids, replace_all=False)

    # ------------------------------------------------------------------
    # Internal helpers
    def _target_athletes(self, athlete_id: Optional[str]) -> Iterable[str]:
        if athlete_id:
            return [athlete_id]
        ids: set[str] = set()
        df = self.athletes.list()
        if not df.empty and "athleteId" in df.columns:
            ids.update(df["athleteId"].astype(str))
        for repo in (self.activities, self.sessions):
            rdf = repo.list()
            if not rdf.empty and "athleteId" in rdf.columns:
                ids.update(rdf["athleteId"].astype(str))
        return sorted(ids)

    def _recompute_for_athletes(self, athlete_ids: Sequence[str], *, replace_all: bool) -> None:
        if not athlete_ids and not replace_all:
            return
        activity_frames: List[pd.DataFrame] = []
        planned_frames: List[pd.DataFrame] = []
        weekly_frames: List[pd.DataFrame] = []
        daily_frames: List[pd.DataFrame] = []

        hr_profiles = self._load_hr_profiles()

        for athlete_id in athlete_ids:
            activities_df = self.activities.list(athleteId=athlete_id)
            planned_df = self.sessions.list(athleteId=athlete_id)

            metrics_df = self._compute_activity_metrics(
                athlete_id=athlete_id,
                activities_df=activities_df,
                hr_profile=hr_profiles.get(athlete_id),
            )
            activity_frames.append(metrics_df)

            planned_metrics_df = self._compute_planned_metrics(
                athlete_id=athlete_id,
                planned_df=planned_df,
                hr_profile=hr_profiles.get(athlete_id),
            )
            planned_frames.append(planned_metrics_df)

            weekly_frames.append(
                self._build_weekly_metrics(
                    athlete_id=athlete_id,
                    actual_metrics_df=metrics_df,
                    planned_metrics_df=planned_metrics_df,
                )
            )
            daily_frames.append(
                self._build_daily_metrics(
                    athlete_id=athlete_id,
                    actual_metrics_df=metrics_df,
                )
            )

        athletes_set = set(athlete_ids)
        self._persist_frame(self.activity_metrics, activity_frames, athletes_set, replace_all)
        self._persist_frame(self.planned_metrics, planned_frames, athletes_set, replace_all)
        self._persist_frame(self.weekly_repo, weekly_frames, athletes_set, replace_all)
        self._persist_frame(self.daily_repo, daily_frames, athletes_set, replace_all)

    def _persist_frame(
        self,
        repo: ActivitiesMetricsRepo | PlannedMetricsRepo | WeeklyMetricsRepo | DailyMetricsRepo,
        new_frames: List[pd.DataFrame],
        athlete_ids: set[str],
        replace_all: bool,
    ) -> None:
        new_df = (
            pd.concat(new_frames, ignore_index=True)
            if new_frames
            else pd.DataFrame(columns=repo.headers)
        )
        existing = repo.list()
        if replace_all or existing.empty:
            merged = new_df if not new_df.empty else pd.DataFrame(columns=repo.headers)
        else:
            existing = existing[~existing.get("athleteId", "").astype(str).isin(athlete_ids)]
            merged = (
                pd.concat([existing, new_df], ignore_index=True)
                if not new_df.empty
                else existing.reset_index(drop=True)
            )
        merged = merged[repo.headers] if not merged.empty else pd.DataFrame(columns=repo.headers)
        self.storage.write_csv(repo.file_name, merged)

    # ------------------------------------------------------------------
    # Activity metrics -------------------------------------------------
    def _compute_activity_metrics(
        self,
        athlete_id: str,
        activities_df: pd.DataFrame,
        hr_profile: Optional[Tuple[float, float]],
    ) -> pd.DataFrame:
        if activities_df.empty:
            return pd.DataFrame(columns=self.activity_metrics.headers)

        rows: List[Dict[str, object]] = []
        for _, row in activities_df.iterrows():
            activity_id = str(row.get("activityId") or "")
            start = to_date(row.get("startTime"))
            if not activity_id or not start:
                continue
            distance_km = safe_float(row.get("distanceKm"))
            ascent_m = safe_float(row.get("ascentM"))
            time_sec = float(safe_int(row.get("movingSec")))
            if time_sec <= 0:
                time_sec = float(safe_int(row.get("elapsedSec")))
            avg_hr = safe_float(row.get("avgHr"))
            trimp = self._compute_trimp(avg_hr, time_sec, hr_profile)
            sport_type, category = self._resolve_activity_category(row)
            if category == "RIDE":
                distance_eq_km = self._compute_bike_distance_eq(activity_id, distance_km, ascent_m)
            else:
                distance_eq_km = self.planner.compute_distance_eq_km(distance_km, ascent_m)

            # Compute HR speed shift if timeseries exists
            hr_speed_shift = ""
            has_timeseries = row.get("hasTimeseries")
            if has_timeseries:
                try:
                    ts_path = self.storage.base_dir / "timeseries" / f"{activity_id}.csv"
                    if ts_path.exists():
                        ts_df = pd.read_csv(ts_path)
                        # Check for required HR column
                        if not ts_df.empty and "hr" in ts_df.columns:
                            # Try preprocessing (requires lat/lon for distance-based speed)
                            # If GPS data missing, fall back to using paceKmh directly
                            if "lat" in ts_df.columns and "lon" in ts_df.columns:
                                ts_df = self.speed_profile_service.preprocess_timeseries(ts_df)
                                if not ts_df.empty and "hr" in ts_df.columns and "speed_km_h" in ts_df.columns:
                                    offset, _ = self.speed_profile_service.compute_hr_speed_shift(ts_df)
                                    hr_speed_shift = int(offset)
                            elif "paceKmh" in ts_df.columns:
                                # Use paceKmh directly - it's already speed in km/h
                                # Create a temporary speed_km_h column for compatibility
                                ts_df = ts_df.copy()
                                ts_df["speed_km_h"] = ts_df["paceKmh"]
                                if not ts_df.empty and "hr" in ts_df.columns:
                                    offset, _ = self.speed_profile_service.compute_hr_speed_shift(ts_df)
                                    hr_speed_shift = int(offset)
                except Exception:
                    hr_speed_shift = ""

            rows.append(
                {
                    "activityId": activity_id,
                    "athleteId": athlete_id,
                    "startDate": str(start),
                    "sportType": sport_type,
                    "category": category,
                    "source": row.get("source") or "",
                    "distanceKm": distance_km,
                    "timeSec": time_sec,
                    "ascentM": ascent_m,
                    "distanceEqKm": distance_eq_km,
                    "trimp": trimp,
                    "avgHr": avg_hr if avg_hr > 0 else "",
                    "hrSpeedShift": hr_speed_shift,
                }
            )

        df = pd.DataFrame(rows, columns=self.activity_metrics.headers if rows else None)
        if df.empty:
            return pd.DataFrame(columns=self.activity_metrics.headers)
        return df[self.activity_metrics.headers]

    def _compute_planned_metrics(
        self,
        athlete_id: str,
        planned_df: pd.DataFrame,
        hr_profile: Optional[Tuple[float, float]],
    ) -> pd.DataFrame:
        if planned_df.empty:
            return pd.DataFrame(columns=self.planned_metrics.headers)

        rows: List[Dict[str, object]] = []
        for _, row in planned_df.iterrows():
            session_id = str(row.get("plannedSessionId") or "")
            if not session_id:
                continue
            date = to_date(row.get("date"))
            if not date:
                continue
            ascent = safe_float(row.get("plannedAscentM"))
            distance = safe_float(row.get("plannedDistanceKm"))
            duration = safe_int(row.get("plannedDurationSec"))

            if duration <= 0 and distance > 0:
                derived = self.planner.derive_from_distance(athlete_id, distance, ascent)
                duration = derived["durationSec"]
                distance = derived["distanceKm"]
            elif distance <= 0 and duration > 0:
                derived = self.planner.derive_from_duration(athlete_id, duration, ascent)
                distance = derived["distanceKm"]
            distance_eq = self.planner.compute_session_distance_eq(
                athlete_id,
                {
                    "type": row.get("type"),
                    "plannedDistanceKm": distance,
                    "plannedDurationSec": duration,
                    "plannedAscentM": ascent,
                },
            )
            planned_trimp = self._compute_planned_trimp(
                athlete_id=athlete_id,
                session=row,
                duration_sec=duration,
                hr_profile=hr_profile,
            )
            rows.append(
                {
                    "plannedSessionId": session_id,
                    "athleteId": athlete_id,
                    "date": str(date),
                    "type": row.get("type") or "",
                    "timeSec": float(duration),
                    "distanceKm": float(distance),
                    "distanceEqKm": float(distance_eq or 0.0),
                    "trimp": planned_trimp,
                }
            )
        df = pd.DataFrame(rows, columns=self.planned_metrics.headers if rows else None)
        if df.empty:
            return pd.DataFrame(columns=self.planned_metrics.headers)
        return df[self.planned_metrics.headers]

    def _load_hr_profiles(self) -> Dict[str, Tuple[float, float]]:
        profiles: Dict[str, Tuple[float, float]] = {}
        df = self.athletes.list()
        if df.empty:
            return profiles
        for _, row in df.iterrows():
            athlete_id = str(row.get("athleteId") or "")
            if not athlete_id:
                continue
            hr_rest = safe_float(row.get("hrRest"))
            hr_max = safe_float(row.get("hrMax"))
            if hr_max <= hr_rest or hr_max <= 0:
                continue
            profiles[athlete_id] = (hr_rest, hr_max)
        return profiles

    def _resolve_activity_category(self, row: pd.Series) -> Tuple[str, str]:
        raw_sport = row.get("sportType")
        sport_type = str(raw_sport).strip() if raw_sport is not None else ""
        if sport_type.lower() in {"", "nan", "none"}:
            sport_type = ""
        raw_path = row.get("rawJsonPath")
        if not sport_type and isinstance(raw_path, (str, bytes)):
            raw_path_str = (
                raw_path.decode() if isinstance(raw_path, (bytes, bytearray)) else raw_path
            )
            raw_path_str = raw_path_str.strip()
            if raw_path_str:
                sport_type = self._raw_activity_sport(raw_path_str)
        if not sport_type:
            fallback = row.get("type")
            if fallback is not None:
                fallback_str = str(fallback).strip()
                if fallback_str.lower() not in {"", "nan", "none"}:
                    sport_type = fallback_str
            if not sport_type:
                sport_type = ""
        category = self._normalize_category(sport_type)
        return sport_type or "", category

    def _raw_activity_sport(self, raw_path: str) -> str:
        path = self.storage.base_dir / Path(raw_path)
        try:
            with path.open("r", encoding="utf-8") as fh:
                detail = json.load(fh)
        except Exception:
            return ""
        return str(detail.get("sport_type") or detail.get("type") or "")

    @staticmethod
    def _normalize_category(sport_type: str) -> str:
        st_lower = sport_type.lower()
        if st_lower in {"run", "running"}:
            return "RUN"
        if st_lower in {"trailrun", "trail_running", "trail run"}:
            return "TRAIL_RUN"
        if st_lower in {"hike", "hiking"}:
            return "HIKE"
        if st_lower in {"ride", "cycling", "bike", "biking"}:
            return "RIDE"
        return "OTHER"

    @staticmethod
    def _compute_trimp(
        avg_hr: float, time_sec: float, hr_profile: Optional[Tuple[float, float]]
    ) -> float:
        # TODO : Add a gender based TRIMP formula if needed
        if not hr_profile or avg_hr <= 0 or time_sec <= 0:
            return 0.0
        hr_rest, hr_max = hr_profile
        if hr_max <= hr_rest:
            return 0.0
        hrr = (avg_hr - hr_rest) / (hr_max - hr_rest)
        hrr = max(0.0, min(hrr, 1.2))
        if hrr <= 0:
            return 0.0
        duration_hours = time_sec / 3600.0
        return duration_hours * hrr * 0.64 * math.exp(1.92 * hrr)

    # ------------------------------------------------------------------
    # Weekly metrics ---------------------------------------------------
    def _build_weekly_metrics(
        self,
        athlete_id: str,
        actual_metrics_df: pd.DataFrame,
        planned_metrics_df: pd.DataFrame,
    ) -> pd.DataFrame:
        planned_agg = self._group_metrics_by_week(planned_metrics_df, "date")
        actual_agg = self._group_metrics_by_week(actual_metrics_df, "startDate")

        if planned_agg.empty and actual_agg.empty:
            return pd.DataFrame(columns=self.weekly_repo.headers)

        weeks = pd.concat(
            [planned_agg[["isoYear", "isoWeek"]], actual_agg[["isoYear", "isoWeek"]]]
        ).drop_duplicates()
        records: List[Dict[str, object]] = []

        for _, wk in weeks.iterrows():
            year = int(wk["isoYear"])
            week = int(wk["isoWeek"])
            week_start = iso_week_start(dt.date.fromisocalendar(year, week, 1)).date()
            week_end = week_start + dt.timedelta(days=6)

            planned_row = planned_agg[
                (planned_agg["isoYear"] == year) & (planned_agg["isoWeek"] == week)
            ].head(1)
            actual_row = actual_agg[
                (actual_agg["isoYear"] == year) & (actual_agg["isoWeek"] == week)
            ].head(1)

            planned_time = float(planned_row["timeSec"].iloc[0]) if not planned_row.empty else 0.0
            planned_distance = (
                float(planned_row["distanceKm"].iloc[0]) if not planned_row.empty else 0.0
            )
            planned_distance_eq = (
                float(planned_row["distanceEqKm"].iloc[0]) if not planned_row.empty else 0.0
            )
            planned_trimp = float(planned_row["trimp"].iloc[0]) if not planned_row.empty else 0.0
            num_planned = int(planned_row["count"].iloc[0]) if not planned_row.empty else 0

            actual_time = float(actual_row["timeSec"].iloc[0]) if not actual_row.empty else 0.0
            actual_distance = (
                float(actual_row["distanceKm"].iloc[0]) if not actual_row.empty else 0.0
            )
            actual_distance_eq = (
                float(actual_row["distanceEqKm"].iloc[0]) if not actual_row.empty else 0.0
            )
            actual_trimp = float(actual_row["trimp"].iloc[0]) if not actual_row.empty else 0.0
            num_actual = int(actual_row["count"].iloc[0]) if not actual_row.empty else 0

            adherence = 0.0
            if planned_distance_eq > 0:
                adherence = round(min(actual_distance_eq / planned_distance_eq * 100.0, 999.0), 2)

            records.append(
                {
                    "athleteId": athlete_id,
                    "isoYear": year,
                    "isoWeek": week,
                    "weekStartDate": str(week_start),
                    "weekEndDate": str(week_end),
                    "plannedTimeSec": planned_time,
                    "actualTimeSec": actual_time,
                    "plannedDistanceKm": planned_distance,
                    "plannedDistanceEqKm": planned_distance_eq,
                    "actualDistanceKm": actual_distance,
                    "actualDistanceEqKm": actual_distance_eq,
                    "plannedTrimp": planned_trimp,
                    "actualTrimp": actual_trimp,
                    "intenseTimeSec": 0.0,
                    "easyTimeSec": actual_time,
                    "numPlannedSessions": num_planned,
                    "numActualSessions": num_actual,
                    "adherencePct": adherence,
                }
            )

        df = pd.DataFrame(records, columns=self.weekly_repo.headers if records else None)
        if df.empty:
            return pd.DataFrame(columns=self.weekly_repo.headers)
        df = df.sort_values(["isoYear", "isoWeek"]).reset_index(drop=True)
        return df[self.weekly_repo.headers]

    # ------------------------------------------------------------------
    # Daily metrics ----------------------------------------------------
    def _build_daily_metrics(
        self,
        athlete_id: str,
        actual_metrics_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if actual_metrics_df.empty:
            return pd.DataFrame(columns=self.daily_repo.headers)

        filtered = actual_metrics_df.copy()
        filtered = filtered[
            filtered["category"].astype(str).str.upper().isin(TRAINING_LOAD_CATEGORIES)
        ]
        filtered["date"] = pd.to_datetime(filtered["startDate"], errors="coerce")
        filtered = filtered.dropna(subset=["date"])
        if filtered.empty:
            return pd.DataFrame(columns=self.daily_repo.headers)

        grouped = (
            filtered.groupby(filtered["date"].dt.date)[
                ["distanceKm", "distanceEqKm", "timeSec", "trimp", "ascentM"]
            ]
            .sum()
            .sort_index()
        )
        if grouped.empty:
            return pd.DataFrame(columns=self.daily_repo.headers)

        grouped.index = pd.to_datetime(grouped.index)
        full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
        grouped = grouped.reindex(full_index, fill_value=0.0)

        grouped["acuteDistanceKm"] = grouped["distanceKm"].rolling(window=7, min_periods=1).mean()
        grouped["chronicDistanceKm"] = (
            grouped["distanceKm"].rolling(window=28, min_periods=1).mean()
        )
        grouped["acuteTimeSec"] = grouped["timeSec"].rolling(window=7, min_periods=1).mean()
        grouped["chronicTimeSec"] = grouped["timeSec"].rolling(window=28, min_periods=1).mean()
        grouped["acuteDistanceEqKm"] = (
            grouped["distanceEqKm"].rolling(window=7, min_periods=1).mean()
        )
        grouped["chronicDistanceEqKm"] = (
            grouped["distanceEqKm"].rolling(window=28, min_periods=1).mean()
        )
        grouped["acuteTrimp"] = grouped["trimp"].rolling(window=7, min_periods=1).mean()
        grouped["chronicTrimp"] = grouped["trimp"].rolling(window=28, min_periods=1).mean()
        grouped["acuteAscentM"] = grouped["ascentM"].rolling(window=7, min_periods=1).mean()
        grouped["chronicAscentM"] = grouped["ascentM"].rolling(window=28, min_periods=1).mean()

        records = []
        for date, values in grouped.iterrows():
            records.append(
                {
                    "dailyId": f"{athlete_id}-{date.date()}",
                    "athleteId": athlete_id,
                    "date": str(date.date()),
                    "distanceKm": float(values["distanceKm"]),
                    "timeSec": float(values["timeSec"]),
                    "distanceEqKm": float(values["distanceEqKm"]),
                    "trimp": float(values["trimp"]),
                    "ascentM": float(values["ascentM"]),
                    "acuteDistanceKm": float(values["acuteDistanceKm"]),
                    "chronicDistanceKm": float(values["chronicDistanceKm"]),
                    "acuteTimeSec": float(values["acuteTimeSec"]),
                    "chronicTimeSec": float(values["chronicTimeSec"]),
                    "acuteDistanceEqKm": float(values["acuteDistanceEqKm"]),
                    "chronicDistanceEqKm": float(values["chronicDistanceEqKm"]),
                    "acuteTrimp": float(values["acuteTrimp"]),
                    "chronicTrimp": float(values["chronicTrimp"]),
                    "acuteAscentM": float(values["acuteAscentM"]),
                    "chronicAscentM": float(values["chronicAscentM"]),
                }
            )
        df = pd.DataFrame(records, columns=self.daily_repo.headers if records else None)
        if df.empty:
            return pd.DataFrame(columns=self.daily_repo.headers)
        return df[self.daily_repo.headers]

    @staticmethod
    def _group_metrics_by_week(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "isoYear",
                    "isoWeek",
                    "timeSec",
                    "distanceKm",
                    "distanceEqKm",
                    "trimp",
                    "count",
                ]
            )
        working = df.copy()
        working["date"] = pd.to_datetime(working[date_column], errors="coerce")
        working = working.dropna(subset=["date"])
        if working.empty:
            return pd.DataFrame(
                columns=[
                    "isoYear",
                    "isoWeek",
                    "timeSec",
                    "distanceKm",
                    "distanceEqKm",
                    "trimp",
                    "count",
                ]
            )
        working["isoYear"] = working["date"].dt.isocalendar().year
        working["isoWeek"] = working["date"].dt.isocalendar().week
        grouped = (
            working.groupby(["isoYear", "isoWeek"])
            .agg(
                timeSec=("timeSec", "sum"),
                distanceKm=("distanceKm", "sum"),
                distanceEqKm=("distanceEqKm", "sum"),
                trimp=("trimp", "sum"),
                count=("timeSec", "count"),
            )
            .reset_index()
        )
        return grouped

    def _compute_planned_trimp(
        self,
        athlete_id: str,
        session: pd.Series,
        duration_sec: int,
        hr_profile: Optional[Tuple[float, float]],
    ) -> float:
        if hr_profile is None or duration_sec <= 0:
            return 0.0
        segments = self._planned_segments(athlete_id, session, hr_profile)
        total = 0.0
        for seg_duration, avg_hr in segments:
            if seg_duration <= 0:
                continue
            total += self._compute_trimp(avg_hr, seg_duration, hr_profile)
        if total <= 0:
            target_hr = self._avg_hr_for_target(
                athlete_id,
                session.get("targetType"),
                session.get("targetLabel"),
                hr_profile,
            )
            total = self._compute_trimp(target_hr, duration_sec, hr_profile)
        return total

    def _bike_eq_factors(self) -> Tuple[float, float, float]:
        if self._bike_eq_cache is not None:
            return self._bike_eq_cache
        settings = self.settings.get("coach-1") or {}
        if "bikeEqDistance" in settings:
            dist = max(float(safe_float(settings.get("bikeEqDistance"))), 0.0)
        else:
            dist = 0.3
        if "bikeEqAscent" in settings:
            asc = max(float(safe_float(settings.get("bikeEqAscent"))), 0.0)
        else:
            asc = 0.02
        if "bikeEqDescent" in settings:
            desc = max(float(safe_float(settings.get("bikeEqDescent"))), 0.0)
        else:
            desc = 0.0
        self._bike_eq_cache = (float(dist), float(asc), float(desc))
        return self._bike_eq_cache

    def _compute_bike_distance_eq(
        self, activity_id: str, distance_km: float, ascent_m: float
    ) -> float:
        dist_f, asc_f, desc_f = self._bike_eq_factors()
        distance = max(float(distance_km or 0.0), 0.0)
        ascent = max(float(ascent_m or 0.0), 0.0)
        descent = 0.0
        if desc_f > 0 and activity_id:
            ts_path = self.storage.base_dir / "timeseries" / f"{activity_id}.csv"
            if ts_path.exists():
                try:
                    ts = pd.read_csv(ts_path)
                    if "elevationM" in ts.columns:
                        elevation = pd.to_numeric(ts["elevationM"], errors="coerce").ffill()
                        diffs = elevation.diff()
                        if diffs is not None and not diffs.empty:
                            descent = float((-diffs[diffs < 0].sum()) if (diffs < 0).any() else 0.0)
                except Exception:
                    descent = 0.0
        return distance * dist_f + ascent * asc_f + descent * desc_f

    def _planned_segments(
        self,
        athlete_id: str,
        session: pd.Series,
        hr_profile: Optional[Tuple[float, float]],
    ) -> List[Tuple[int, float]]:
        segments: List[Tuple[int, float]] = []
        fundamental_hr = self._avg_hr_for_target(athlete_id, None, "Fundamental", hr_profile)
        duration = safe_int(session.get("plannedDurationSec"))
        steps_json = session.get("stepsJson")
        if not steps_json:
            avg_hr = self._avg_hr_for_target(
                athlete_id,
                session.get("targetType"),
                session.get("targetLabel"),
                hr_profile,
            )
            if duration > 0:
                segments.append((duration, avg_hr))
            return segments

        try:
            steps = json.loads(steps_json)
        except Exception:
            if duration > 0:
                segments.append((duration, fundamental_hr))
            return segments

        normalised = normalize_steps(steps)

        def _segment_hr(block: Dict[str, Any]) -> float:
            kind = (block.get("kind") or "recovery").lower()
            if kind == "run":
                return self._avg_hr_for_target(
                    athlete_id,
                    block.get("targetType"),
                    block.get("targetLabel"),
                    hr_profile,
                )
            return fundamental_hr

        for block in normalised["preBlocks"]:
            sec = safe_int(block.get("sec"))
            if sec > 0:
                segments.append((sec, _segment_hr(block)))

        between = normalised.get("betweenBlock")
        between_sec = safe_int((between or {}).get("sec"))
        between_hr = _segment_hr(between) if between and between_sec > 0 else fundamental_hr

        loops = normalised["loops"] or []
        loop_count = len(loops)
        for loop_index, loop in enumerate(loops):
            actions = loop.get("actions") or []
            repeats = max(1, safe_int(loop.get("repeats")) or 1)
            for rep_index in range(repeats):
                for action in actions:
                    sec = safe_int(action.get("sec"))
                    if sec <= 0:
                        continue
                    segments.append((sec, _segment_hr(action)))
            if between_sec > 0 and loop_count > 1 and loop_index < loop_count - 1:
                segments.append((between_sec, between_hr))

        for block in normalised["postBlocks"]:
            sec = safe_int(block.get("sec"))
            if sec > 0:
                segments.append((sec, _segment_hr(block)))
        return segments

    def _avg_hr_for_target(
        self,
        athlete_id: str,
        target_type: Optional[str],
        target_label: Optional[str],
        hr_profile: Optional[Tuple[float, float]],
    ) -> float:
        label = target_label or ""
        if target_type == "sensation":
            label = "Fundamental"
        if target_type in {"hr", "pace"} and not label:
            label = target_label or "Fundamental"
        threshold = None
        if label:
            threshold = self.planner.resolve_threshold_target(athlete_id, label)
        if threshold:
            hr_min = safe_float(threshold.get("hrMin"))
            hr_max = safe_float(threshold.get("hrMax"))
            if hr_max > 0 and hr_min > 0:
                return (hr_min + hr_max) / 2.0
        if hr_profile:
            hr_rest, hr_max = hr_profile
            if hr_max > hr_rest:
                return hr_rest + 0.6 * (hr_max - hr_rest)
        return 0.0
