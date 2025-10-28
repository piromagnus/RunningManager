from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from persistence.csv_storage import CsvStorage
from persistence.repositories import (
    ActivitiesMetricsRepo,
    ActivitiesRepo,
    LinksRepo,
    PlannedSessionsRepo,
)


def _coerce_str(value: object) -> Optional[str]:
    try:
        if value in (None, "", "NaN"):
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _coerce_float(value: object) -> Optional[float]:
    try:
        if value in (None, "", "NaN"):
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _coerce_int(value: object) -> Optional[int]:
    try:
        if value in (None, "", "NaN"):
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _parse_timestamp(value: object) -> Optional[pd.Timestamp]:
    if value in (None, "", "NaT"):
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


@dataclass(frozen=True)
class ActivityFeedItem:
    activity_id: str
    athlete_id: str
    name: str
    sport_type: str
    start_time: Optional[pd.Timestamp]
    distance_km: Optional[float]
    ascent_m: Optional[float]
    avg_hr: Optional[float]
    moving_sec: Optional[int]
    elapsed_sec: Optional[int]
    trimp: Optional[float]
    distance_eq_km: Optional[float]
    linked: bool
    match_score: Optional[float]
    planned_session_id: Optional[str]
    planned_session_type: Optional[str]
    planned_session_template_title: Optional[str]
    planned_session_race_name: Optional[str]
    planned_session_notes: Optional[str]


@dataclass(frozen=True)
class PlannedSessionCard:
    planned_session_id: str
    date: dt.date
    session_type: str
    template_title: str
    race_name: Optional[str]
    notes: Optional[str]
    planned_distance_km: Optional[float]
    planned_duration_sec: Optional[int]
    planned_ascent_m: Optional[float]
    target_label: str


class ActivityFeedService:
    def __init__(self, storage: CsvStorage):
        self.storage = storage
        self.activities = ActivitiesRepo(storage)
        self.activity_metrics = ActivitiesMetricsRepo(storage)
        self.links = LinksRepo(storage)
        self.planned_sessions = PlannedSessionsRepo(storage)

    def available_sport_types(self, athlete_id: str) -> List[str]:
        activities = self.activities.list(athleteId=athlete_id)
        if activities.empty or "sportType" not in activities.columns:
            return []
        values = activities["sportType"].astype(str).str.strip().str.upper()
        return sorted({value for value in values if value})

    def get_feed(
        self,
        athlete_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> List[ActivityFeedItem]:
        activities = self.activities.list(athleteId=athlete_id)
        if activities.empty:
            return []

        activities = activities.copy()
        activities["activityId"] = activities["activityId"].astype(str)
        activities["startTime"] = activities["startTime"].map(_parse_timestamp)

        metrics = self.activity_metrics.list(athleteId=athlete_id)
        if not metrics.empty:
            metrics = metrics[["activityId", "distanceEqKm", "trimp", "timeSec"]].copy()
            metrics["activityId"] = metrics["activityId"].astype(str)
            metrics = metrics.rename(
                columns={
                    "distanceEqKm": "activityDistanceEqKm",
                    "trimp": "activityTrimp",
                    "timeSec": "activityTimeSec",
                }
            )
        links = self._links_df()

        merged = activities
        if not metrics.empty:
            merged = merged.merge(metrics, on="activityId", how="left")
        if not links.empty:
            merged = merged.merge(
                links[["activityId", "plannedSessionId", "matchScore"]],
                on="activityId",
                how="left",
            )

        planned_lookup = self.planned_sessions.list()
        if not planned_lookup.empty:
            planned_lookup = planned_lookup[
                [
                    "plannedSessionId",
                    "type",
                    "templateTitle",
                    "raceName",
                    "notes",
                ]
            ].copy()
            planned_lookup["plannedSessionId"] = planned_lookup["plannedSessionId"].astype(str)
            planned_lookup = planned_lookup.rename(
                columns={
                    "type": "plannedType",
                    "templateTitle": "plannedTemplateTitle",
                    "raceName": "plannedRaceName",
                    "notes": "plannedNotes",
                }
            )
            merged = merged.merge(planned_lookup, on="plannedSessionId", how="left")

        merged = merged.sort_values("startTime", ascending=False, na_position="last")
        if offset:
            merged = merged.iloc[offset:]
        if limit:
            merged = merged.iloc[:limit]

        feed: List[ActivityFeedItem] = []
        for _, row in merged.iterrows():
            activity_id = str(row["activityId"])
            name = row.get("name") or row.get("source") or activity_id
            sport_type = str(row.get("sportType") or "").strip()
            feed.append(
                ActivityFeedItem(
                    activity_id=activity_id,
                    athlete_id=str(athlete_id),
                    name=str(name),
                    sport_type=sport_type,
                    start_time=row.get("startTime"),
                    distance_km=_coerce_float(row.get("distanceKm")),
                    ascent_m=_coerce_float(row.get("ascentM")),
                    avg_hr=_coerce_float(row.get("avgHr")),
                    moving_sec=_coerce_int(row.get("movingSec")),
                    elapsed_sec=_coerce_int(row.get("elapsedSec")),
                    trimp=_coerce_float(row.get("activityTrimp")),
                    distance_eq_km=_coerce_float(row.get("activityDistanceEqKm")),
                    linked=not pd.isna(row.get("matchScore")),
                    match_score=_coerce_float(row.get("matchScore")),
                    planned_session_id=str(row.get("plannedSessionId"))
                    if not pd.isna(row.get("plannedSessionId"))
                    else None,
                    planned_session_type=str(row.get("plannedType") or "")
                    if not pd.isna(row.get("plannedType"))
                    else None,
                    planned_session_template_title=_coerce_str(row.get("plannedTemplateTitle")),
                    planned_session_race_name=_coerce_str(row.get("plannedRaceName")),
                    planned_session_notes=_coerce_str(row.get("plannedNotes")),
                )
            )
        return feed

    def get_unlinked_planned_sessions(
        self,
        athlete_id: str,
        *,
        reference_date: Optional[dt.date] = None,
        lookback_days: int = 7,
        lookahead_days: int = 14,
        max_items: int = 8,
    ) -> List[PlannedSessionCard]:
        sessions = self.planned_sessions.list(athleteId=athlete_id)
        if sessions.empty:
            return []

        links = self._links_df()
        if not links.empty:
            taken = set(links["plannedSessionId"])
            sessions = sessions[~sessions["plannedSessionId"].astype(str).isin(taken)]
        if sessions.empty:
            return []

        sessions = sessions.copy()
        sessions["plannedSessionId"] = sessions["plannedSessionId"].astype(str)
        sessions["date"] = pd.to_datetime(sessions["date"], errors="coerce").dt.date
        sessions = sessions.dropna(subset=["date"])

        if reference_date is None:
            reference_date = pd.Timestamp.utcnow().date()

        def _within_window(value: dt.date) -> bool:
            delta = (value - reference_date).days
            if lookback_days is not None and delta < -lookback_days:
                return False
            if lookahead_days is not None and delta > lookahead_days:
                return False
            return True

        sessions = sessions[sessions["date"].map(_within_window)]
        if sessions.empty:
            return []

        sessions["delta"] = sessions["date"].map(lambda d: (d - reference_date).days)
        sessions["is_future"] = sessions["delta"] >= 0
        sessions["abs_delta"] = sessions["delta"].abs()

        sessions = sessions.sort_values(
            by=["is_future", "abs_delta", "date"], ascending=[True, True, True]
        )
        if max_items:
            sessions = sessions.iloc[:max_items]

        cards: List[PlannedSessionCard] = []
        for _, row in sessions.iterrows():
            cards.append(
                PlannedSessionCard(
                    planned_session_id=str(row["plannedSessionId"]),
                    date=row["date"],
                    session_type=str(row.get("type") or ""),
                    template_title=str(row.get("templateTitle") or ""),
                    race_name=_coerce_str(row.get("raceName")),
                    notes=_coerce_str(row.get("notes")),
                    planned_distance_km=_coerce_float(row.get("plannedDistanceKm")),
                    planned_duration_sec=_coerce_int(row.get("plannedDurationSec")),
                    planned_ascent_m=_coerce_float(row.get("plannedAscentM")),
                    target_label=str(row.get("targetLabel") or ""),
                )
            )
        return cards

    def _links_df(self) -> pd.DataFrame:
        df = self.links.list()
        if df.empty:
            return df
        df = df.copy()
        for col in ("activityId", "plannedSessionId"):
            if col in df.columns:
                df[col] = df[col].astype(str)
        df["matchScore"] = df.get("matchScore")
        return df
