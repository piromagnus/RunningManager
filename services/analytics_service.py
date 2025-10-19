from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from persistence.csv_storage import CsvStorage
from persistence.repositories import WeeklyMetricsRepo, LinksRepo


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
        self.links_repo = LinksRepo(self.storage)

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
        # Prefer the actual first-day-of-week date as label if available
        if "weekStartDate" in df.columns and df["weekStartDate"].notna().any():
            df["weekLabel"] = df["weekStartDate"].dt.strftime("%Y-%m-%d")
        else:
            # Fallback to ISO year-week label
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

    # ------------------------
    # Daily range API
    def activity_date_bounds(self, athlete_id: str) -> Tuple[Optional[dt.date], Optional[dt.date]]:
        """Return (min_date, max_date) for activities of an athlete.

        Falls back to None if no activities exist. Dates are naive (local) dates.
        """
        path = self.storage.base_dir / "activities_metrics.csv"
        if not path.exists():
            return None, None
        df = pd.read_csv(path)
        if df.empty:
            return None, None
        df = df[df.get("athleteId") == athlete_id]
        if df.empty:
            return None, None
        dates = pd.to_datetime(df.get("startDate"), errors="coerce").dropna().dt.normalize()
        if dates.empty:
            return None, None
        return dates.min().date(), dates.max().date()

    def daily_range(
        self,
        *,
        athlete_id: str,
        metric_label: str,
        selected_types: Optional[Sequence[str]],
        start_date: dt.date,
        end_date: dt.date,
    ) -> pd.DataFrame:
        """Compute daily planned and actual values within [start_date, end_date].

        - Actuals come from activities_metrics.csv filtered by athlete and selected types.
        - Planned values come from planned_metrics.csv rescheduled by links to the
          day of performed activity when linked; otherwise remain on planned date.

        Returns a DataFrame with columns: date, planned_value, actual_value.
        Includes all days in the range even if values are zero.
        """
        # Map metric label to columns
        value_column_map = {
            "Time": "timeSec",
            "Distance": "distanceKm",
            "DistEq": "distanceEqKm",
            "Trimp": "trimp",
        }
        value_col = value_column_map.get(metric_label, "distanceKm")

        # Actuals
        acts_path = self.storage.base_dir / "activities_metrics.csv"
        if acts_path.exists():
            acts_df = pd.read_csv(acts_path)
        else:
            acts_df = pd.DataFrame()
        if not acts_df.empty:
            acts_df = acts_df[acts_df.get("athleteId") == athlete_id]
            if selected_types:
                acts_df["category"] = acts_df.get("category", pd.Series(dtype=str)).astype(str).str.upper()
                acts_df = acts_df[acts_df["category"].isin([s.upper() for s in selected_types])]
            acts_df["date"] = pd.to_datetime(acts_df.get("startDate"), errors="coerce").dt.normalize()
            mask = (acts_df["date"] >= pd.Timestamp(start_date)) & (acts_df["date"] <= pd.Timestamp(end_date))
            acts_df = acts_df[mask]
            actual_daily = (
                acts_df.groupby("date", as_index=False)[value_col].sum().rename(columns={value_col: "actual_value"})
            )
        else:
            actual_daily = pd.DataFrame(columns=["date", "actual_value"])

        # Planned (rescheduled by links)
        pm_path = self.storage.base_dir / "planned_metrics.csv"
        if pm_path.exists():
            pm_df = pd.read_csv(pm_path)
        else:
            pm_df = pd.DataFrame()

        if not pm_df.empty:
            pm_df = pm_df[pm_df.get("athleteId") == athlete_id]
            # Effective date mapping via links
            links_df = self.links_repo.list()
            if not links_df.empty:
                # Build mapping plannedSessionId -> performed date
                links_df = links_df.copy()
                links_df["plannedSessionId"] = links_df.get("plannedSessionId").astype(str)
                links_df["activityId"] = links_df.get("activityId").astype(str)

                # Map activityId -> activity date for athlete
                if not acts_df.empty:
                    act_dates = acts_df[["activityId", "date"]].copy()
                    act_dates["activityId"] = act_dates["activityId"].astype(str)
                    links_df = links_df.merge(act_dates, on="activityId", how="left")
                    link_dates = (
                        links_df.dropna(subset=["date"]).groupby("plannedSessionId", as_index=False)["date"].min()
                    )
                else:
                    link_dates = pd.DataFrame(columns=["plannedSessionId", "date"])
            else:
                link_dates = pd.DataFrame(columns=["plannedSessionId", "date"])

            # Planned effective date
            pm_df = pm_df.copy()
            pm_df["planned_date"] = pd.to_datetime(pm_df.get("date"), errors="coerce").dt.normalize()
            if not link_dates.empty and "plannedSessionId" in pm_df.columns:
                pm_df["plannedSessionId"] = pm_df["plannedSessionId"].astype(str)
                pm_df = pm_df.merge(
                    link_dates.rename(columns={"date": "effective_date"}),
                    on="plannedSessionId",
                    how="left",
                )
                pm_df["effective_date"] = pd.to_datetime(pm_df.get("effective_date"), errors="coerce").dt.normalize()
                pm_df["date"] = pm_df["effective_date"].fillna(pm_df["planned_date"])
            else:
                pm_df["date"] = pm_df["planned_date"]

            if value_col in pm_df.columns:
                mask = (pm_df["date"] >= pd.Timestamp(start_date)) & (pm_df["date"] <= pd.Timestamp(end_date))
                pm_df = pm_df[mask]
                planned_daily = (
                    pm_df.groupby("date", as_index=False)[value_col].sum().rename(columns={value_col: "planned_value"})
                )
            else:
                planned_daily = pd.DataFrame(columns=["date", "planned_value"])
        else:
            planned_daily = pd.DataFrame(columns=["date", "planned_value"])

        # Build continuous day index and merge
        if start_date and end_date and start_date <= end_date:
            all_days = pd.DataFrame(
                {"date": pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="D")}
            )
        else:
            all_days = pd.DataFrame(columns=["date"])
        daily = all_days.merge(planned_daily, on="date", how="left").merge(actual_daily, on="date", how="left")
        daily["planned_value"] = pd.to_numeric(daily.get("planned_value"), errors="coerce").fillna(0.0)
        daily["actual_value"] = pd.to_numeric(daily.get("actual_value"), errors="coerce").fillna(0.0)
        return daily
