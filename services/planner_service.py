from __future__ import annotations

import statistics
from dataclasses import dataclass
import json
import math
from typing import Any, Dict, List, Optional

from persistence.csv_storage import CsvStorage
from persistence.repositories import (
    ThresholdsRepo,
    ActivitiesRepo,
    PlannedSessionsRepo,
    SettingsRepo,
)


@dataclass
class PlannerService:
    storage: CsvStorage

    def __post_init__(self) -> None:
        self.thresholds = ThresholdsRepo(self.storage)
        self.activities = ActivitiesRepo(self.storage)
        self.sessions = PlannedSessionsRepo(self.storage)
        self.settings = SettingsRepo(self.storage)
        self._distance_eq_factor_cache: Optional[float] = None

    def list_threshold_names(self, athlete_id: str) -> List[str]:
        df = self.thresholds.list(athleteId=athlete_id)
        if df.empty:
            return []
        names = df["name"].dropna().astype(str).tolist()
        # Remove any legacy combined name if present
        names = [n for n in names if n != "Threshold 60/30"]
        # Prefer order: Fundamental, Threshold 30, Threshold 60, others
        preferred = ["Fundamental", "Threshold 30", "Threshold 60"]
        rest = [n for n in names if n not in preferred]
        ordered = [n for n in preferred if n in names] + sorted(rest)
        return ordered

    def resolve_threshold_target(self, athlete_id: str, target_label: str) -> Optional[Dict[str, Any]]:
        df = self.thresholds.list(athleteId=athlete_id)
        if df.empty:
            return None
        hit = df[df["name"] == target_label]
        if hit.empty:
            return None
        return hit.iloc[0].to_dict()

    def recent_easy_pace(self, athlete_id: str) -> Optional[float]:
        df = self.activities.list(athleteId=athlete_id)
        if df.empty:
            return None
        # Sort by startTime descending if available
        if "startTime" in df.columns:
            df = df.sort_values("startTime", ascending=False)
        paces = []
        for _, r in df.head(4).iterrows():
            dist = r.get("distanceKm") or 0
            mov = r.get("movingSec") or 0
            if mov and dist:
                pace = dist / (mov / 3600.0)
                if pace > 0:
                    paces.append(pace)
        if not paces:
            return None
        return statistics.median(paces)

    def estimate_km(self, athlete_id: str, duration_sec: int) -> float:
        pace_kmh: Optional[float] = None
        thr = self.resolve_threshold_target(athlete_id, "Fundamental")
        if thr:
            try:
                pmin = float(thr.get("paceFlatKmhMin") or 0)
                pmax = float(thr.get("paceFlatKmhMax") or 0)
                if pmin > 0 and pmax > 0:
                    pace_kmh = (pmin + pmax) / 2.0
            except Exception:
                pace_kmh = None
        if pace_kmh is None:
            pace_kmh = self.recent_easy_pace(athlete_id)
        if pace_kmh is None:
            pace_kmh = 9.0
        return (duration_sec / 3600.0) * pace_kmh

    # --- New helpers for interval estimation ---
    def _fundamental_pace_kmh(self, athlete_id: str) -> float:
        thr = self.resolve_threshold_target(athlete_id, "Fundamental")
        if thr:
            try:
                pmin = float(thr.get("paceFlatKmhMin") or 0)
                pmax = float(thr.get("paceFlatKmhMax") or 0)
                if pmin > 0 and pmax > 0:
                    return (pmin + pmax) / 2.0
            except Exception:
                pass
        pace = self.recent_easy_pace(athlete_id)
        return pace if pace is not None else 9.0

    def _threshold_pace_kmh(self, athlete_id: str, label: Optional[str]) -> Optional[float]:
        if not label:
            return None
        thr = self.resolve_threshold_target(athlete_id, label)
        if not thr:
            return None
        try:
            pmin = float(thr.get("paceFlatKmhMin") or 0)
            pmax = float(thr.get("paceFlatKmhMax") or 0)
            if pmin > 0 and pmax > 0:
                return (pmin + pmax) / 2.0
        except Exception:
            return None
        return None

    def estimate_interval_duration_sec(self, steps: Dict[str, Any]) -> int:
        warm = int(steps.get("warmupSec") or 0)
        cool = int(steps.get("cooldownSec") or 0)
        # Back-compat simple repeats
        if "repeats" in steps:
            reps = steps.get("repeats") or []
            total_rep = 0
            for rep in reps:
                total_rep += int(rep.get("workSec") or 0) + int(rep.get("recoverSec") or 0)
            return warm + total_rep + cool
        # New loops structure
        loops = steps.get("loops") or []
        between = int(steps.get("betweenLoopRecoverSec") or 0)
        total = warm + cool
        for loop in loops:
            actions = loop.get("actions") or []
            repeats = int(loop.get("repeats") or 1)
            one_loop = 0
            for act in actions:
                one_loop += int(act.get("sec") or 0)
            total += repeats * one_loop
            total += max(0, repeats - 1) * between
        return total

    def estimate_interval_distance_km(self, athlete_id: str, steps: Dict[str, Any]) -> float:
        fundamental_pace = self._fundamental_pace_kmh(athlete_id)
        dist_km = 0.0
        # Warmup
        warm = int(steps.get("warmupSec") or 0)
        dist_km += (warm / 3600.0) * fundamental_pace
        # Back-compat simple repeats
        if "repeats" in steps:
            reps = steps.get("repeats") or []
            for rep in reps:
                work = int(rep.get("workSec") or 0)
                rec = int(rep.get("recoverSec") or 0)
                tt = rep.get("targetType")
                tl = rep.get("targetLabel")
                pace_for_work = None
                if tt in ("pace", "hr"):
                    pace_for_work = self._threshold_pace_kmh(athlete_id, tl)
                if pace_for_work is None:
                    pace_for_work = fundamental_pace
                dist_km += (work / 3600.0) * pace_for_work
                dist_km += (rec / 3600.0) * fundamental_pace
        else:
            # New loops structure
            loops = steps.get("loops") or []
            between = int(steps.get("betweenLoopRecoverSec") or 0)
            for loop in loops:
                actions = loop.get("actions") or []
                repeats = int(loop.get("repeats") or 1)
                one_loop_km = 0.0
                for act in actions:
                    sec = int(act.get("sec") or 0)
                    kind = (act.get("kind") or "recovery").lower()
                    tt = act.get("targetType")
                    tl = act.get("targetLabel")
                    pace = fundamental_pace
                    if kind == "run" and tt in ("pace", "hr"):
                        pace = self._threshold_pace_kmh(athlete_id, tl) or fundamental_pace
                    # asc/desc targets are informational; distance estimate still based on pace
                    one_loop_km += (sec / 3600.0) * pace
                # add between-loop recovery (at fundamental)
                dist_km += repeats * one_loop_km
                dist_km += max(0, repeats - 1) * ((between / 3600.0) * fundamental_pace)
        # Cooldown
        cool = int(steps.get("cooldownSec") or 0)
        dist_km += (cool / 3600.0) * fundamental_pace
        return dist_km

    def estimate_interval_ascent_m(self, steps: Dict[str, Any]) -> int:
        # Only loops/actions currently carry explicit ascend targets
        total = 0
        loops = steps.get("loops") or []
        for loop in loops:
            repeats = int(loop.get("repeats") or 1)
            actions = loop.get("actions") or []
            loop_asc = 0
            for act in actions:
                loop_asc += int(act.get("ascendM") or 0)
            total += repeats * loop_asc
        return int(total)

    def _parse_steps(self, steps_json: Any) -> Optional[Dict[str, Any]]:
        if isinstance(steps_json, dict):
            return steps_json
        if not steps_json:
            return None
        if isinstance(steps_json, str):
            try:
                return json.loads(steps_json)
            except Exception:
                return None
        return None

    def estimate_session_distance_km(self, athlete_id: str, session: Dict[str, Any]) -> Optional[float]:
        # Prefer explicit planned distance
        dist = session.get("plannedDistanceKm")
        if dist not in (None, ""):
            try:
                dist_value = float(dist)
                if math.isnan(dist_value):
                    raise ValueError
                return dist_value
            except Exception:
                pass
        session_type = (session.get("type") or "").upper()
        if session_type == "FUNDAMENTAL_ENDURANCE":
            duration = session.get("plannedDurationSec")
            if duration not in (None, ""):
                try:
                    return self.estimate_km(athlete_id, int(float(duration)))
                except Exception:
                    return None
        if session_type == "INTERVAL_SIMPLE":
            steps = self._parse_steps(session.get("stepsJson"))
            if steps:
                return self.estimate_interval_distance_km(athlete_id, steps)
        return None

    def estimate_session_ascent_m(self, athlete_id: str, session: Dict[str, Any]) -> int:
        ascent = session.get("plannedAscentM")
        if ascent not in (None, ""):
            try:
                ascent_value = float(ascent)
                if math.isnan(ascent_value):
                    raise ValueError
                return int(ascent_value)
            except Exception:
                pass
        session_type = (session.get("type") or "").upper()
        if session_type == "INTERVAL_SIMPLE":
            steps = self._parse_steps(session.get("stepsJson"))
            if steps:
                return self.estimate_interval_ascent_m(steps)
        return 0

    def compute_weekly_totals(self, athlete_id: str, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_time = 0
        total_dist = 0.0
        total_ascent = 0
        for sess in sessions:
            duration = sess.get("plannedDurationSec")
            try:
                if duration not in (None, ""):
                    total_time += int(duration)
            except Exception:
                pass
            est_dist = self.estimate_session_distance_km(athlete_id, sess)
            if est_dist is not None:
                total_dist += est_dist
            total_ascent += self.estimate_session_ascent_m(athlete_id, sess)
        return {
            "timeSec": int(total_time),
            "distanceKm": float(total_dist),
            "ascentM": int(total_ascent),
        }

    # --- Distance equivalent helpers ---
    def _distance_eq_factor(self) -> float:
        if self._distance_eq_factor_cache is not None:
            return self._distance_eq_factor_cache
        row = self.settings.get("coach-1") or {}
        try:
            factor = float(row.get("distanceEqFactor", 0.01))
        except Exception:
            factor = 0.01
        if factor < 0:
            factor = 0.0
        self._distance_eq_factor_cache = factor
        return factor

    def compute_distance_eq_km(self, distance_km: float, ascent_m: float) -> float:
        distance = max(float(distance_km or 0.0), 0.0)
        ascent = max(float(ascent_m or 0.0), 0.0)
        return distance + ascent * self._distance_eq_factor()

    def derive_from_distance(
        self,
        athlete_id: str,
        distance_km: float,
        ascent_m: float,
    ) -> Dict[str, float]:
        eq_km = self.compute_distance_eq_km(distance_km, ascent_m)
        pace = max(self._fundamental_pace_kmh(athlete_id), 0.1)
        duration_sec = int(round((eq_km / pace) * 3600))
        return {
            "distanceKm": max(float(distance_km or 0.0), 0.0),
            "distanceEqKm": eq_km,
            "durationSec": max(duration_sec, 0),
        }

    def derive_from_duration(
        self,
        athlete_id: str,
        duration_sec: int,
        ascent_m: float,
    ) -> Dict[str, float]:
        duration = max(int(duration_sec or 0), 0)
        pace = max(self._fundamental_pace_kmh(athlete_id), 0.1)
        distance_eq_km = (duration / 3600.0) * pace
        ascent = max(float(ascent_m or 0.0), 0.0)
        distance_km = max(distance_eq_km - ascent * self._distance_eq_factor(), 0.0)
        return {
            "distanceKm": distance_km,
            "distanceEqKm": distance_eq_km,
            "durationSec": duration,
        }

    def compute_session_distance_eq(
        self,
        athlete_id: str,
        session: Dict[str, Any],
    ) -> Optional[float]:
        distance = session.get("plannedDistanceKm")
        ascent = session.get("plannedAscentM") or 0
        try:
            if distance not in (None, ""):
                return self.compute_distance_eq_km(float(distance), float(ascent or 0))
        except Exception:
            pass
        session_type = (session.get("type") or "").upper()
        if session_type in {"FUNDAMENTAL_ENDURANCE", "LONG_RUN"}:
            duration = session.get("plannedDurationSec")
            if duration not in (None, ""):
                derived = self.derive_from_duration(
                    athlete_id,
                    int(float(duration)),
                    float(ascent or 0),
                )
                return derived["distanceEqKm"]
        if session_type == "INTERVAL_SIMPLE":
            distance_est = self.estimate_session_distance_km(athlete_id, session)
            if distance_est is not None:
                return self.compute_distance_eq_km(distance_est, float(ascent or 0))
        return None
