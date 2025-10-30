"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from persistence.csv_storage import CsvStorage
from persistence.repositories import (
    ActivitiesMetricsRepo,
    ActivitiesRepo,
    LinksRepo,
    PlannedMetricsRepo,
    PlannedSessionsRepo,
)
from services.timeseries_service import TimeseriesService
from utils.coercion import safe_float_optional, safe_int_optional
from utils.config import Config
from utils.time import parse_timestamp


def _decode_polyline(encoded: str) -> List[tuple[float, float]]:
    if not encoded:
        return []
    index = 0
    length = len(encoded)
    lat = 0
    lon = 0
    coordinates: List[tuple[float, float]] = []

    while index < length:
        shift = 0
        result = 0
        while True:
            if index >= length:
                return []
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        delta = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += delta

        shift = 0
        result = 0
        while True:
            if index >= length:
                return []
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        delta = ~(result >> 1) if (result & 1) else (result >> 1)
        lon += delta

        coordinates.append((lat / 1e5, lon / 1e5))

    return coordinates


@dataclass(frozen=True)
class ActivitySummary:
    distance_km: Optional[float]
    moving_sec: Optional[int]
    elapsed_sec: Optional[int]
    ascent_m: Optional[float]
    avg_hr: Optional[float]
    trimp: Optional[float]
    distance_eq_km: Optional[float]


@dataclass(frozen=True)
class ComparisonMetric:
    actual: Optional[float]
    planned: Optional[float]
    delta: Optional[float]


@dataclass(frozen=True)
class PlanComparison:
    distance: ComparisonMetric
    duration: ComparisonMetric
    trimp: ComparisonMetric
    ascent: ComparisonMetric


@dataclass(frozen=True)
class MapPoint:
    lat: float
    lon: float


@dataclass(frozen=True)
class ActivityDetail:
    activity_id: str
    athlete_id: str
    title: str
    start_time: Optional[pd.Timestamp]
    description: str
    summary: ActivitySummary
    linked: bool
    planned_session_id: Optional[str]
    match_score: Optional[float]
    comparison: Optional[PlanComparison]
    map_path: Optional[List[MapPoint]]
    map_notice: Optional[str]
    raw_detail: Optional[Dict[str, object]]


class ActivityDetailService:
    def __init__(
        self,
        storage: CsvStorage,
        config: Config,
        timeseries_service: Optional[TimeseriesService] = None,
    ):
        self.storage = storage
        self.config = config
        self.activities = ActivitiesRepo(storage)
        self.activity_metrics = ActivitiesMetricsRepo(storage)
        self.links = LinksRepo(storage)
        self.planned_sessions = PlannedSessionsRepo(storage)
        self.planned_metrics = PlannedMetricsRepo(storage)
        self.timeseries = timeseries_service or TimeseriesService(config)

    def get_detail(self, athlete_id: Optional[str], activity_id: str) -> ActivityDetail:
        activity = self.activities.get(activity_id)
        if not activity:
            raise ValueError("Activity not found")
        activity_athlete_id = str(activity.get("athleteId") or "")
        resolved_athlete_id = str(athlete_id or activity_athlete_id)
        if athlete_id and activity_athlete_id and activity_athlete_id != str(athlete_id):
            raise ValueError("Activity does not belong to athlete")

        metrics_row = self._activity_metrics_row(activity_id, resolved_athlete_id)
        link_row = self._link_row(activity_id)
        planned_session = None
        planned_metrics_row = None
        match_score: Optional[float] = None
        planned_session_id: Optional[str] = None
        if link_row is not None:
            planned_session_id = str(link_row.get("plannedSessionId"))
            match_score = safe_float_optional(link_row.get("matchScore"))
            planned_session = self.planned_sessions.get(planned_session_id)
            planned_metrics_row = self._planned_metrics_row(planned_session_id, resolved_athlete_id)

        raw_detail = self._load_raw_detail(activity.get("rawJsonPath"))
        title = self._resolve_title(activity, raw_detail)
        description = self._resolve_description(raw_detail)
        start_time = parse_timestamp(activity.get("startTime"))

        summary = ActivitySummary(
            distance_km=safe_float_optional(activity.get("distanceKm")),
            moving_sec=safe_int_optional(activity.get("movingSec")),
            elapsed_sec=safe_int_optional(activity.get("elapsedSec")),
            ascent_m=safe_float_optional(activity.get("ascentM")),
            avg_hr=safe_float_optional(activity.get("avgHr")),
            trimp=safe_float_optional((metrics_row or {}).get("trimp")),
            distance_eq_km=safe_float_optional((metrics_row or {}).get("distanceEqKm")),
        )

        comparison = None
        if planned_session:
            comparison = self._build_comparison(summary, planned_session, planned_metrics_row)

        map_path, map_notice = self._build_map_path(activity_id, activity)

        return ActivityDetail(
            activity_id=str(activity.get("activityId")),
            athlete_id=resolved_athlete_id,
            title=title,
            start_time=start_time,
            description=description,
            summary=summary,
            linked=planned_session_id is not None,
            planned_session_id=planned_session_id,
            match_score=match_score,
            comparison=comparison,
            map_path=map_path,
            map_notice=map_notice,
            raw_detail=raw_detail,
        )

    def _activity_metrics_row(
        self, activity_id: str, athlete_id: str
    ) -> Optional[Dict[str, object]]:
        metrics = self.activity_metrics.list(athleteId=athlete_id)
        if metrics.empty:
            return None
        metrics = metrics.copy()
        metrics["activityId"] = metrics["activityId"].astype(str)
        hits = metrics[metrics["activityId"] == str(activity_id)]
        if hits.empty:
            return None
        return hits.iloc[0].to_dict()

    def _planned_metrics_row(
        self, planned_session_id: str, athlete_id: str
    ) -> Optional[Dict[str, object]]:
        metrics = self.planned_metrics.list(athleteId=athlete_id)
        if metrics.empty:
            return None
        metrics = metrics.copy()
        metrics["plannedSessionId"] = metrics["plannedSessionId"].astype(str)
        hits = metrics[metrics["plannedSessionId"] == str(planned_session_id)]
        if hits.empty:
            return None
        return hits.iloc[0].to_dict()

    def _link_row(self, activity_id: str) -> Optional[Dict[str, object]]:
        links = self.links.list()
        if links.empty:
            return None
        links = links.copy()
        links["activityId"] = links["activityId"].astype(str)
        hits = links[links["activityId"] == str(activity_id)]
        if hits.empty:
            return None
        return hits.iloc[0].to_dict()

    def _load_raw_detail(self, path_value: Optional[str]) -> Optional[Dict[str, object]]:
        if not path_value:
            return None
        path = Path(path_value)
        if not path.is_absolute():
            path = (self.config.data_dir / path).resolve()
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    @staticmethod
    def _resolve_title(activity: Dict[str, object], raw_detail: Optional[Dict[str, object]]) -> str:
        if raw_detail and raw_detail.get("name"):
            return str(raw_detail["name"])
        if activity.get("name"):
            return str(activity["name"])
        source = activity.get("source") or "Activité"
        activity_id = activity.get("activityId") or ""
        return f"{source} {activity_id}".strip()

    @staticmethod
    def _resolve_description(raw_detail: Optional[Dict[str, object]]) -> str:
        if not raw_detail:
            return ""
        description = raw_detail.get("description") or raw_detail.get("private_note")
        return str(description or "")

    def _build_comparison(
        self,
        summary: ActivitySummary,
        planned_session: Dict[str, object],
        planned_metrics_row: Optional[Dict[str, object]],
    ) -> PlanComparison:
        planned_distance = safe_float_optional(planned_session.get("plannedDistanceKm"))
        planned_duration = safe_int_optional(planned_session.get("plannedDurationSec"))
        planned_ascent = safe_float_optional(planned_session.get("plannedAscentM"))
        planned_trimp = safe_float_optional((planned_metrics_row or {}).get("trimp"))

        return PlanComparison(
            distance=self._comparison_metric(summary.distance_km, planned_distance),
            duration=self._comparison_metric(summary.moving_sec, planned_duration),
            trimp=self._comparison_metric(summary.trimp, planned_trimp),
            ascent=self._comparison_metric(summary.ascent_m, planned_ascent),
        )

    @staticmethod
    def _comparison_metric(actual: Optional[float], planned: Optional[float]) -> ComparisonMetric:
        if actual is None or planned is None:
            return ComparisonMetric(actual=actual, planned=planned, delta=None)
        return ComparisonMetric(actual=actual, planned=planned, delta=actual - planned)

    def _build_map_path(
        self,
        activity_id: str,
        activity: Dict[str, object],
    ) -> tuple[Optional[List[MapPoint]], Optional[str]]:
        poly = activity.get("polyline")
        if poly:
            coords = _decode_polyline(str(poly))
            if coords:
                return ([MapPoint(lat=lat, lon=lon) for lat, lon in coords], None)

        # fallback to timeseries
        df = self.timeseries.load(activity_id) if self.timeseries else None
        if df is None or df.empty:
            return None, "Aucune donnée de trace disponible."

        working = df.copy()
        lat_col = None
        lon_col = None
        for candidate in ("lat", "latitude"):
            if candidate in working.columns:
                lat_col = candidate
                break
        for candidate in ("lon", "lng", "longitude"):
            if candidate in working.columns:
                lon_col = candidate
                break
        if not lat_col or not lon_col:
            return None, "Aucune donnée de trace disponible."

        working = working.dropna(subset=[lat_col, lon_col])
        if working.empty:
            return None, "Aucune donnée de trace disponible."
        path = [
            MapPoint(lat=float(row[lat_col]), lon=float(row[lon_col]))
            for _, row in working.iterrows()
        ]
        return path, None
