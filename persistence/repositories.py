"""Repository layer for CSV-backed storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from persistence.csv_storage import CsvStorage
from utils.ids import new_id


def _ensure_headers(df: pd.DataFrame, headers: List[str]) -> pd.DataFrame:
    for h in headers:
        if h not in df.columns:
            df[h] = pd.Series(dtype="object")
    return df[headers]


@dataclass
class BaseRepo:
    storage: CsvStorage
    file_name: str
    headers: List[str]
    id_column: str

    def _path(self) -> Path:
        return self.storage.base_dir / self.file_name

    def list(self, **filters: Any) -> pd.DataFrame:
        df = self.storage.read_csv(self.file_name)
        df = _ensure_headers(df, self.headers)
        for k, v in filters.items():
            if k in df.columns:
                df = df[df[k] == v]
        return df.reset_index(drop=True)

    def get(self, entity_id: str) -> Optional[Dict[str, Any]]:
        df = self.list()
        if self.id_column not in df.columns:
            return None
        working = df.copy()
        try:
            working[self.id_column] = working[self.id_column].astype(str)
        except Exception:
            working[self.id_column] = working[self.id_column].apply(lambda v: str(v))
        target = str(entity_id)
        hit = working[working[self.id_column] == target]
        if hit.empty:
            return None
        record = hit.iloc[0].to_dict()
        return record

    def create(self, row: Dict[str, Any]) -> str:
        if self.id_column not in row or not row[self.id_column]:
            row[self.id_column] = new_id()
        # Ensure full column ordering
        df = pd.DataFrame([row])
        df = _ensure_headers(df, self.headers)
        self.storage.append_row(self.file_name, df.iloc[0].to_dict(), self.headers)
        return str(row[self.id_column])

    def update(self, entity_id: str, updates: Dict[str, Any]) -> None:
        row = {self.id_column: entity_id}
        row.update(updates)
        self.storage.upsert(self.file_name, [self.id_column], row)

    def delete(self, entity_id: str) -> None:
        df = self.storage.read_csv(self.file_name)
        if df.empty or self.id_column not in df.columns:
            return
        try:
            df[self.id_column] = df[self.id_column].astype(str)
        except Exception:
            df[self.id_column] = df[self.id_column].apply(lambda v: str(v))
        df = df[df[self.id_column] != str(entity_id)]
        df = _ensure_headers(df, self.headers)
        self.storage.write_csv(self.file_name, df)


class ActivitiesRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "activities.csv",
            [
                "activityId",
                "athleteId",
                "source",
                "sportType",
                "name",
                "startTime",
                "distanceKm",
                "elapsedSec",
                "movingSec",
                "ascentM",
                "avgHr",
                "maxHr",
                "hasTimeseries",
                "polyline",
                "rawJsonPath",
            ],
            id_column="activityId",
        )

    def _migrate_headers_if_needed(self) -> None:
        path = self._path()
        if not path.exists():
            return
        df = self.storage.read_csv(self.file_name)
        # If any header missing, rewrite file with full headers and empty defaults
        if any(h not in df.columns for h in self.headers):
            df = _ensure_headers(df, self.headers)
            self.storage.write_csv(self.file_name, df)

    def create(self, row: Dict[str, Any]) -> str:
        # Ensure file headers are migrated before appending
        self._migrate_headers_if_needed()
        return super().create(row)


class PlannedSessionsRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "planned_sessions.csv",
            [
                "plannedSessionId",
                "athleteId",
                "date",
                "type",
                "plannedDistanceKm",
                "plannedDurationSec",
                "plannedAscentM",
                "targetType",
                "targetLabel",
                "notes",
                "stepEndMode",
                "stepsJson",
            ],
            id_column="plannedSessionId",
        )

    def _migrate_headers_if_needed(self) -> None:
        path = self._path()
        if not path.exists():
            return
        df = self.storage.read_csv(self.file_name)
        # If any header missing, rewrite file with full headers and empty defaults
        if any(h not in df.columns for h in self.headers):
            df = _ensure_headers(df, self.headers)
            self.storage.write_csv(self.file_name, df)

    def create(self, row: Dict[str, Any]) -> str:
        # Ensure file headers are migrated before appending
        self._migrate_headers_if_needed()
        return super().create(row)


class LinksRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "links.csv",
            [
                "linkId",
                "plannedSessionId",
                "activityId",
                "matchScore",
                "rpe(1-10)",
                "comments",
            ],
            id_column="linkId",
        )


class MetricsRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "metrics.csv",
            [
                "periodStart",
                "periodEnd",
                "distanceKm",
                "timeSec",
                "ascentM",
                "distanceEqKm",
                "intenseTimeSec",
                "easyTimeSec",
                "numSessions",
                "adherencePct",
            ],
            id_column="periodStart",  # combined with end logically; simple primary for now
        )


class WeeklyMetricsRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "weekly_metrics.csv",
            [
                "athleteId",
                "isoYear",
                "isoWeek",
                "weekStartDate",
                "weekEndDate",
                "plannedTimeSec",
                "actualTimeSec",
                "plannedDistanceKm",
                "plannedDistanceEqKm",
                "actualDistanceKm",
                "actualDistanceEqKm",
                "plannedTrimp",
                "actualTrimp",
                "intenseTimeSec",
                "easyTimeSec",
                "numPlannedSessions",
                "numActualSessions",
                "adherencePct",
            ],
            id_column="weekStartDate",
        )


class DailyMetricsRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "daily_metrics.csv",
            [
                "dailyId",
                "athleteId",
                "date",
                "distanceKm",
                "timeSec",
                "distanceEqKm",
                "trimp",
                "ascentM",
                "acuteDistanceKm",
                "chronicDistanceKm",
                "acuteTimeSec",
                "chronicTimeSec",
                "acuteDistanceEqKm",
                "chronicDistanceEqKm",
                "acuteTrimp",
                "chronicTrimp",
                "acuteAscentM",
                "chronicAscentM",
            ],
            id_column="dailyId",
        )


class ThresholdsRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "thresholds.csv",
            [
                "thresholdId",
                "athleteId",
                "name",
                "hrMin",
                "hrMax",
                "paceFlatKmhMin",
                "paceFlatKmhMax",
                "ascentRateMPerHMin",
                "ascentRateMPerHMax",
            ],
            id_column="thresholdId",
        )


class GoalsRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "goals.csv",
            [
                "goalId",
                "athleteId",
                "date",
                "distanceKm",
                "ascentM",
                "terrain",
                "priority",
                "targetTimeSec",
                "isRace",
            ],
            id_column="goalId",
        )


class TemplatesRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "templates.csv",
            [
                "templateId",
                "athleteId",
                "name",
                "stepsJson",
            ],
            id_column="templateId",
        )


class SessionTemplatesRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "session_templates.csv",
            [
                "templateId",
                "athleteId",
                "title",
                "baseType",
                "payloadJson",
                "notes",
                "lastUsedAt",
            ],
            id_column="templateId",
        )


class ActivitiesMetricsRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "activities_metrics.csv",
            [
                "activityId",
                "athleteId",
                "startDate",
                "sportType",
                "category",
                "source",
                "distanceKm",
                "timeSec",
                "ascentM",
                "distanceEqKm",
                "trimp",
                "avgHr",
            ],
            id_column="activityId",
        )


class PlannedMetricsRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "planned_metrics.csv",
            [
                "plannedSessionId",
                "athleteId",
                "date",
                "type",
                "timeSec",
                "distanceKm",
                "distanceEqKm",
                "trimp",
            ],
            id_column="plannedSessionId",
        )


class AthletesRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "athlete.csv",
            [
                "athleteId",
                "coachId",
                "name",
                "thresholdsProfileId",
                "units",
                "hrRest",
                "hrMax",
            ],
            id_column="athleteId",
        )


class SettingsRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "settings.csv",
            [
                "coachId",
                "units",
                "distanceEqFactor",
                "stravaSyncDays",
                "analyticsActivityTypes",
                "bikeEqDistance",   # multiplier for bike distance contribution
                "bikeEqAscent",     # multiplier for bike ascent contribution
                "bikeEqDescent",    # multiplier for bike descent contribution
            ],
            id_column="coachId",
        )


class TokensRepo(BaseRepo):
    def __init__(self, storage: CsvStorage):
        super().__init__(
            storage,
            "tokens.csv",
            [
                "athleteId",
                "provider",
                "accessTokenEnc",
                "refreshTokenEnc",
                "expiresAt",
            ],
            id_column="athleteId",  # together with provider acts as composite; use upsert with both
        )
