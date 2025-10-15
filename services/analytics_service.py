from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

from persistence.csv_storage import CsvStorage
from persistence.repositories import WeeklyMetricsRepo


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


@dataclass
class AnalyticsService:
    storage: CsvStorage

    def __post_init__(self) -> None:
        self.weekly_repo = WeeklyMetricsRepo(self.storage)

    def load_weekly_metrics(self, athlete_id: Optional[str] = None) -> pd.DataFrame:
        df = self.weekly_repo.list(athleteId=athlete_id) if athlete_id else self.weekly_repo.list()
        if df.empty:
            return df
        numeric_cols: List[str] = [
            "isoYear",
            "isoWeek",
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
        ]
        df = _to_numeric(df, numeric_cols)
        df["isoYear"] = df["isoYear"].astype(int, errors="ignore")
        df["isoWeek"] = df["isoWeek"].astype(int, errors="ignore")
        if "weekStartDate" in df.columns:
            df["weekStartDate"] = pd.to_datetime(df["weekStartDate"], errors="coerce")
        if "weekEndDate" in df.columns:
            df["weekEndDate"] = pd.to_datetime(df["weekEndDate"], errors="coerce")
        df["weekLabel"] = (
            df["isoYear"].astype(int).astype(str)
            + "-W"
            + df["isoWeek"].astype(int).astype(str).str.zfill(2)
        )
        return df

    @staticmethod
    def seconds_to_hours(seconds: float | int | None) -> float:
        if not seconds or seconds <= 0:
            return 0.0
        return float(seconds) / 3600.0

    @staticmethod
    def compute_trimp(duration_sec: float, intensity_factor: float) -> float:
        """
        Compute TRIMP using duration expressed in hours rather than raw seconds.

        Parameters
        ----------
        duration_sec: float
            Session duration in seconds.
        intensity_factor: float
            Aggregated intensity multiplier (e.g. HR reserve weighting).

        Returns
        -------
        float
            TRIMP score.
        """
        if duration_sec <= 0 or intensity_factor <= 0:
            return 0.0
        hours = AnalyticsService.seconds_to_hours(duration_sec)
        return hours * intensity_factor

    def build_planned_vs_actual_segments(
        self,
        df: pd.DataFrame,
        *,
        planned_column: str,
        actual_column: str,
        metric_key: str,
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "athleteId",
                    "weekLabel",
                    "segment",
                    "value",
                    "planned",
                    "actual",
                    "maxValue",
                    "metric",
                    "order",
                ]
            )
        rows: List[Dict[str, object]] = []
        for _, row in df.iterrows():
            planned = float(row.get(planned_column) or 0.0)
            actual = float(row.get(actual_column) or 0.0)
            base = min(planned, actual)
            above = max(actual - planned, 0.0)
            shortfall = max(planned - actual, 0.0)
            max_value = max(planned, actual)
            common = {
                "athleteId": row.get("athleteId"),
                "isoYear": int(row.get("isoYear") or 0),
                "isoWeek": int(row.get("isoWeek") or 0),
                "weekLabel": row.get("weekLabel"),
                "planned": planned,
                "actual": actual,
                "maxValue": max_value,
                "metric": metric_key,
            }

            if max_value == 0:
                rows.append({**common, "segment": "Réalisé", "value": 0.0, "order": 0})
                continue

            if base > 0:
                rows.append({**common, "segment": "Réalisé", "value": base, "order": 0})
            else:
                rows.append({**common, "segment": "Réalisé", "value": 0.0, "order": 0})
            if above > 0:
                rows.append({**common, "segment": "Au-dessus du plan", "value": above, "order": 1})
            if shortfall > 0:
                rows.append({**common, "segment": "Plan manquant", "value": shortfall, "order": 2})
        return pd.DataFrame(rows)
