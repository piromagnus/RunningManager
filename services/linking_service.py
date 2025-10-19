from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
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
from services.metrics_service import MetricsComputationService
from utils.ids import new_id


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


def _ensure_datetime(value: object) -> Optional[dt.datetime]:
    if value in (None, "", "NaT"):
        return None
    try:
        if isinstance(value, dt.datetime):
            return value
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


@dataclass
class LinkingService:
    storage: CsvStorage

    def __post_init__(self) -> None:
        self.activities = ActivitiesRepo(self.storage)
        self.sessions = PlannedSessionsRepo(self.storage)
        self.links = LinksRepo(self.storage)
        self.activity_metrics = ActivitiesMetricsRepo(self.storage)
        self.planned_metrics = PlannedMetricsRepo(self.storage)

    def _links_df(self) -> pd.DataFrame:
        df = self.links.list()
        if "rpe(1-10)" in df.columns:
            df = df.rename(columns={"rpe(1-10)": "rpe"})
        for col in ("activityId", "plannedSessionId", "linkId"):
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df

    def unlinked_activities(self, athlete_id: str) -> pd.DataFrame:
        activities = self.activities.list(athleteId=athlete_id)
        if activities.empty:
            return activities
        activities = self._with_activity_metrics(activities)
        links = self._links_df()
        if not links.empty:
            linked_ids = set(links["activityId"])
            activities = activities[
                ~activities["activityId"].astype(str).isin(linked_ids)
            ]
        activities = activities.sort_values("startTime", ascending=False)
        return activities.reset_index(drop=True)

    def linked_activities(self, athlete_id: str) -> pd.DataFrame:
        activities = self.activities.list(athleteId=athlete_id)
        links = self._links_df()
        if activities.empty or links.empty:
            return pd.DataFrame()
        merged = links.merge(
            activities.assign(activityId=activities["activityId"].astype(str)),
            on="activityId",
            how="inner",
            suffixes=("_link", ""),
        )
        planned = self.sessions.list(athleteId=athlete_id)
        if not planned.empty:
            planned = planned.rename(
                columns={
                    "date": "plannedDate",
                    "type": "plannedType",
                }
            )
            merged = merged.merge(
                planned[
                    [
                        "plannedSessionId",
                        "plannedDate",
                        "plannedType",
                        "plannedDistanceKm",
                        "plannedDurationSec",
                        "targetType",
                        "targetLabel",
                        "plannedAscentM",
                    ]
                ],
                on="plannedSessionId",
                how="left",
            )
        merged = self._with_activity_metrics(merged)
        merged = self._with_planned_metrics(merged)
        merged = merged.sort_values("startTime", ascending=False)
        return merged.reset_index(drop=True)

    def available_planned_sessions(self, athlete_id: str) -> pd.DataFrame:
        planned = self.sessions.list(athleteId=athlete_id)
        if planned.empty:
            return planned
        links = self._links_df()
        if not links.empty and "plannedSessionId" in links.columns:
            taken = set(links["plannedSessionId"])
            planned = planned[
                ~planned["plannedSessionId"].astype(str).isin(taken)
            ]
        planned = planned.sort_values("date")
        return planned.reset_index(drop=True)

    def _with_activity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        working = df.copy()
        working["activityId"] = working["activityId"].astype(str)
        metrics = self.activity_metrics.list()
        if metrics.empty:
            for col in ("activityDistanceEqKm", "activityTimeSec", "activityTrimp"):
                if col not in working.columns:
                    working[col] = None
            return working
        metrics = metrics[
            ["activityId", "distanceEqKm", "timeSec", "trimp"]
        ].copy()
        metrics["activityId"] = metrics["activityId"].astype(str)
        metrics = metrics.rename(
            columns={
                "distanceEqKm": "activityDistanceEqKm",
                "timeSec": "activityTimeSec",
                "trimp": "activityTrimp",
            }
        )
        merged = working.merge(metrics, on="activityId", how="left")
        for col in ("activityDistanceEqKm", "activityTimeSec", "activityTrimp"):
            if col not in merged.columns:
                merged[col] = None
        return merged

    def _with_planned_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "plannedSessionId" not in df.columns:
            return df
        working = df.copy()
        working["plannedSessionId"] = working["plannedSessionId"].astype(str)
        metrics = self.planned_metrics.list()
        if metrics.empty:
            for col in ("plannedMetricDistanceEqKm", "plannedMetricTimeSec", "plannedMetricTrimp"):
                if col not in working.columns:
                    working[col] = None
            return working
        metrics = metrics[
            ["plannedSessionId", "distanceEqKm", "timeSec", "trimp"]
        ].copy()
        metrics["plannedSessionId"] = metrics["plannedSessionId"].astype(str)
        metrics = metrics.rename(
            columns={
                "distanceEqKm": "plannedMetricDistanceEqKm",
                "timeSec": "plannedMetricTimeSec",
                "trimp": "plannedMetricTrimp",
            }
        )
        merged = working.merge(metrics, on="plannedSessionId", how="left")
        for col in ("plannedMetricDistanceEqKm", "plannedMetricTimeSec", "plannedMetricTrimp"):
            if col not in merged.columns:
                merged[col] = None
        return merged

    def suggest_for_activity(
        self,
        athlete_id: str,
        activity_row: pd.Series,
        window_days: int = 7,
    ) -> List[Dict[str, object]]:
        planned = self.available_planned_sessions(athlete_id)
        if planned.empty:
            return []
        suggestions: List[Dict[str, object]] = []
        activity_date = _ensure_datetime(activity_row.get("startTime"))
        activity_distance = _coerce_float(activity_row.get("distanceKm"))
        activity_duration = _coerce_int(
            activity_row.get("movingSec") or activity_row.get("elapsedSec")
        )

        for _, row in planned.iterrows():
            score = self._compute_match_score(
                activity_distance,
                activity_duration,
                activity_date,
                row,
                window_days,
            )
            if score <= 0:
                continue
            suggestions.append(
                {
                    "plannedSessionId": str(row["plannedSessionId"]),
                    "date": row.get("date"),
                    "type": row.get("type"),
                    "plannedDistanceKm": row.get("plannedDistanceKm"),
                    "plannedDurationSec": row.get("plannedDurationSec"),
                    "score": round(score, 3),
                }
            )
        suggestions.sort(key=lambda item: item["score"], reverse=True)
        return suggestions[:5]

    def _compute_match_score(
        self,
        activity_distance: Optional[float],
        activity_duration: Optional[int],
        activity_date: Optional[dt.datetime],
        planned_row: pd.Series,
        window_days: int,
    ) -> float:
        planned_distance = _coerce_float(planned_row.get("plannedDistanceKm"))
        planned_duration = _coerce_int(planned_row.get("plannedDurationSec"))
        planned_date = None
        if planned_row.get("date"):
            planned_date = pd.to_datetime(planned_row.get("date"), utc=True)

        distance_component = 0.0
        if activity_distance is not None and planned_distance not in (None, 0):
            diff = abs(activity_distance - planned_distance)
            base = max(planned_distance, 1.0)
            distance_component = max(0.0, 1.0 - min(diff / base, 1.0))

        duration_component = 0.0
        if activity_duration is not None and planned_duration not in (None, 0):
            diff = abs(activity_duration - planned_duration)
            base = max(planned_duration, 1)
            duration_component = max(0.0, 1.0 - min(diff / base, 1.0))

        date_component = 0.0
        if activity_date and planned_date and window_days > 0:
            delta_days = abs((planned_date.date() - activity_date.date()).days)
            date_component = max(
                0.0, 1.0 - min(delta_days / float(window_days), 1.0)
            )

        score = 0.4 * distance_component + 0.4 * duration_component + 0.2 * date_component
        return max(0.0, min(score, 1.0))

    def create_link(
        self,
        athlete_id: str,
        activity_id: str,
        planned_session_id: str,
        rpe: Optional[int] = None,
        comments: str = "",
        window_days: int = 7,
    ) -> str:
        existing = self._links_df()
        if not existing.empty and activity_id in set(existing["activityId"]):
            raise ValueError("Activity already linked")

        activities = self.activities.list(athleteId=athlete_id)
        activity = activities[activities["activityId"].astype(str) == str(activity_id)]
        if activity.empty:
            raise ValueError("Activity not found for athlete")
        activity_row = activity.iloc[0]

        planned = self.sessions.get(planned_session_id)
        if not planned:
            raise ValueError("Planned session not found")

        score = self._compute_match_score(
            _coerce_float(activity_row.get("distanceKm")),
            _coerce_int(activity_row.get("movingSec") or activity_row.get("elapsedSec")),
            _ensure_datetime(activity_row.get("startTime")),
            pd.Series(planned),
            window_days,
        )

        payload = {
            "linkId": new_id(),
            "plannedSessionId": str(planned_session_id),
            "activityId": str(activity_id),
            "matchScore": round(score, 3) if score > 0 else "",
            "rpe(1-10)": rpe if rpe is not None else "",
            "comments": comments or "",
        }
        link_id = self.links.create(payload)
        MetricsComputationService(self.storage).recompute_athlete(athlete_id)
        return link_id

    def update_link(self, link_id: str, *, rpe: Optional[int], comments: str) -> None:
        updates: Dict[str, object] = {
            "rpe(1-10)": rpe if rpe is not None else "",
            "comments": comments or "",
        }
        self.links.update(link_id, updates)

    def delete_link(self, link_id: str) -> None:
        self.links.delete(link_id)
