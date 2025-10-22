from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from persistence.csv_storage import CsvStorage
from persistence.repositories import WeeklyMetricsRepo, LinksRepo, SettingsRepo


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
        self.settings_repo = SettingsRepo(self.storage)

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

        activity_names_by_date = pd.DataFrame(columns=["date", "activity_names"])

        if not acts_df.empty:
            acts_df = acts_df[acts_df.get("athleteId") == athlete_id].copy()
            if selected_types:
                acts_df["category"] = acts_df.get("category", pd.Series(dtype=str)).astype(str).str.upper()
                acts_df = acts_df[acts_df["category"].isin([s.upper() for s in selected_types])]
            acts_df["date"] = pd.to_datetime(acts_df.get("startDate"), errors="coerce").dt.normalize()
            mask = (acts_df["date"] >= pd.Timestamp(start_date)) & (acts_df["date"] <= pd.Timestamp(end_date))
            acts_df = acts_df[mask]

            # Attach activity names from activities.csv when available
            activities_path = self.storage.base_dir / "activities.csv"
            if not acts_df.empty and activities_path.exists():
                activities_df = pd.read_csv(activities_path)
                if {"activityId", "name"}.issubset(activities_df.columns):
                    name_map = activities_df[["activityId", "name"]].copy()
                    name_map["activityId"] = name_map["activityId"].astype(str)
                    acts_df["activityId"] = acts_df.get("activityId", pd.Series(dtype=str)).astype(str)
                    acts_df = acts_df.merge(name_map, on="activityId", how="left")
            acts_df["activity_name"] = acts_df.get("name", pd.Series(dtype=str)).fillna("").astype(str)

            actual_daily = (
                acts_df.groupby("date", as_index=False)[value_col].sum().rename(columns={value_col: "actual_value"})
            )

            if "activity_name" in acts_df.columns:
                activity_names_by_date = (
                    acts_df.groupby("date", as_index=False)["activity_name"]
                    .agg(lambda values: "\n".join(dict.fromkeys(str(v).strip() for v in values if str(v).strip())))
                    .rename(columns={"activity_name": "activity_names"})
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
        if not activity_names_by_date.empty:
            daily = daily.merge(activity_names_by_date, on="date", how="left")
        else:
            daily["activity_names"] = ""
        daily["planned_value"] = pd.to_numeric(daily.get("planned_value"), errors="coerce").fillna(0.0)
        daily["actual_value"] = pd.to_numeric(daily.get("actual_value"), errors="coerce").fillna(0.0)
        daily["activity_names"] = daily.get("activity_names", pd.Series(dtype=str)).fillna("").astype(str)
        return daily

    # ------------------------
    # Weekly range API (recomputed from included activities)
    def weekly_range(
        self,
        *,
        athlete_id: str,
        metric_label: str,
        selected_types: Optional[Sequence[str]],
        start_date: dt.date,
        end_date: dt.date,
    ) -> pd.DataFrame:
        """Recompute weekly planned vs actual for a given date range.

        - Actuals: from activities_metrics.csv filtered by athlete and selected categories.
          For DistEq metric, bike (RIDE) activities use bike equivalence factors from settings.
          Descent contribution defaults to 0 unless a timeseries is available and a non-zero
          descent factor is configured (best-effort computation).
        - Planned: from planned_metrics.csv by planned date (not rescheduled), summed per ISO week.
        - Returns DataFrame with columns: weekStart, isoYear, isoWeek, planned_value, actual_value.
        """
        # Build weeks grid
        grid: List[dt.date] = []
        cur = (start_date - dt.timedelta(days=start_date.isoweekday() - 1))
        while cur <= end_date:
            grid.append(cur)
            cur = cur + dt.timedelta(days=7)
        weeks_df = pd.DataFrame({"weekStart": pd.to_datetime(grid)})
        iso = weeks_df["weekStart"].dt.isocalendar()
        weeks_df["isoYear"] = iso.year.astype(int)
        weeks_df["isoWeek"] = iso.week.astype(int)

        # Planned
        pm_path = self.storage.base_dir / "planned_metrics.csv"
        if pm_path.exists():
            pm_df = pd.read_csv(pm_path)
        else:
            pm_df = pd.DataFrame()
        planned_col_map = {
            "Time": "timeSec",
            "Distance": "distanceKm",
            "DistEq": "distanceEqKm",
            "Trimp": "trimp",
        }
        planned_col = planned_col_map.get(metric_label, "distanceKm")
        if not pm_df.empty:
            pm_df = pm_df[pm_df.get("athleteId") == athlete_id]
            pm_df["date"] = pd.to_datetime(pm_df.get("date"), errors="coerce").dt.normalize()
            pm_df = pm_df[(pm_df["date"] >= pd.Timestamp(start_date)) & (pm_df["date"] <= pd.Timestamp(end_date))]
            pm_df["isoYear"] = pm_df["date"].dt.isocalendar().year.astype(int)
            pm_df["isoWeek"] = pm_df["date"].dt.isocalendar().week.astype(int)
            if planned_col in pm_df.columns:
                planned_weekly = (
                    pm_df.groupby(["isoYear", "isoWeek"], as_index=False)[planned_col].sum()
                )
            else:
                planned_weekly = pd.DataFrame(columns=["isoYear", "isoWeek", planned_col])
        else:
            planned_weekly = pd.DataFrame(columns=["isoYear", "isoWeek", planned_col])

        # Actuals (with bike DistEq override)
        acts_path = self.storage.base_dir / "activities_metrics.csv"
        if acts_path.exists():
            acts_df = pd.read_csv(acts_path)
        else:
            acts_df = pd.DataFrame()
        actual_col_map = {
            "Time": "timeSec",
            "Distance": "distanceKm",
            "DistEq": "distanceEqKm",
            "Trimp": "trimp",
        }
        actual_col = actual_col_map.get(metric_label, "distanceKm")

        bike_eq = self._load_bike_eq_factors()

        if not acts_df.empty:
            acts_df = acts_df[acts_df.get("athleteId") == athlete_id].copy()
            if selected_types:
                acts_df["category"] = acts_df.get("category", pd.Series(dtype=str)).astype(str).str.upper()
                acts_df = acts_df[acts_df["category"].isin([s.upper() for s in selected_types])]
            acts_df["date"] = pd.to_datetime(acts_df.get("startDate"), errors="coerce").dt.normalize()
            acts_df = acts_df[(acts_df["date"] >= pd.Timestamp(start_date)) & (acts_df["date"] <= pd.Timestamp(end_date))]
            # If DistEq and category RIDE, override per-activity value
            if metric_label == "DistEq":
                acts_df["_value"] = acts_df.apply(
                    lambda r: self._bike_disteq_value(r, bike_eq)
                    if str(r.get("category", "")).upper() == "RIDE"
                    else float(r.get("distanceEqKm") or 0.0),
                    axis=1,
                )
                value_series = acts_df["_value"].astype(float)
            else:
                value_series = pd.to_numeric(acts_df.get(actual_col), errors="coerce").fillna(0.0)

            acts_df["isoYear"] = acts_df["date"].dt.isocalendar().year.astype(int)
            acts_df["isoWeek"] = acts_df["date"].dt.isocalendar().week.astype(int)
            actual_weekly = (
                acts_df.groupby(["isoYear", "isoWeek"], as_index=False).agg(actual_value=(value_series.name if metric_label != "DistEq" else "_value", "sum"))
            )
            if metric_label != "DistEq":
                # recompute with correct name if not DistEq path
                actual_weekly = (
                    acts_df.groupby(["isoYear", "isoWeek"], as_index=False)[actual_col].sum().rename(columns={actual_col: "actual_value"})
                )
        else:
            actual_weekly = pd.DataFrame(columns=["isoYear", "isoWeek", "actual_value"])

        # Join onto weeks grid
        out = weeks_df.merge(
            planned_weekly.rename(columns={planned_col: "planned_value"}),
            on=["isoYear", "isoWeek"],
            how="left",
        ).merge(
            actual_weekly,
            on=["isoYear", "isoWeek"],
            how="left",
        )
        out["planned_value"] = pd.to_numeric(out.get("planned_value"), errors="coerce").fillna(0.0)
        out["actual_value"] = pd.to_numeric(out.get("actual_value"), errors="coerce").fillna(0.0)
        return out

    def _load_bike_eq_factors(self) -> Tuple[float, float, float]:
        settings = self.settings_repo.get("coach-1") or {}
        dist = self._safe_float(settings.get("bikeEqDistance"), 0.3)
        asc = self._safe_float(settings.get("bikeEqAscent"), 0.02)
        desc = self._safe_float(settings.get("bikeEqDescent"), 0.0)
        return dist, asc, desc

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        try:
            if value in (None, "", "NaN"):
                return default
            return float(value)
        except Exception:
            return default

    def _bike_disteq_value(self, row: pd.Series, factors: Tuple[float, float, float]) -> float:
        dist_f, asc_f, desc_f = factors
        distance_km = self._safe_float(row.get("distanceKm"), 0.0)
        ascent_m = self._safe_float(row.get("ascentM"), 0.0)
        descent_m = 0.0
        if desc_f > 0:
            # Optional: best-effort compute descent from timeseries if available
            aid = str(row.get("activityId") or "")
            if aid:
                ts_path = self.storage.base_dir / "timeseries" / f"{aid}.csv"
                if ts_path.exists():
                    try:
                        ts = pd.read_csv(ts_path)
                        if "elevationM" in ts.columns:
                            diffs = pd.to_numeric(ts["elevationM"], errors="coerce").fillna(method="ffill").diff()
                            descent_m = float((-diffs[diffs < 0].sum()) if not diffs.empty else 0.0)
                    except Exception:
                        descent_m = 0.0
        return distance_km * dist_f + ascent_m * asc_f - descent_m * desc_f
