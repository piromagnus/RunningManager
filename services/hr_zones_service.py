"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from persistence.csv_storage import CsvStorage
from services.speed_profile_service import SpeedProfileService
from services.timeseries_service import TimeseriesService

HR_SMOOTHING_WINDOW = 30
MAX_PERSISTED_BORDERS = 4
BORDER_COLUMNS = [f"hrZone_z{i}_upper" for i in range(1, MAX_PERSISTED_BORDERS + 1)]
ZONE_METADATA_COLUMNS = [
    "hrZone_gmm_sample_count",
    "hrZone_gmm_fit_date",
    "hrZone_zone_count",
]


def _to_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _to_int(value: object, default: int) -> int:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    try:
        return int(parsed)
    except Exception:
        return default


@dataclass
class HrZonesService:
    storage: CsvStorage
    ts_service: TimeseriesService
    speed_profile_service: Optional[SpeedProfileService] = None
    zone_count: int = 5
    window_days: int = 90
    fit_activity_types: Sequence[str] = ("RUN", "TRAIL_RUN")

    def __post_init__(self) -> None:
        # Borders are persisted in fixed columns z1..z4, so we cap to 5 zones.
        self.zone_count = max(2, min(_to_int(self.zone_count, 5), MAX_PERSISTED_BORDERS + 1))
        self.window_days = max(1, _to_int(self.window_days, 90))
        normalized = [str(item).strip().upper() for item in self.fit_activity_types if str(item).strip()]
        self.fit_activity_types = tuple(dict.fromkeys(normalized))

    @property
    def _zones_dir(self) -> str:
        return "hr_zones"

    def compute_zones(self, activity_id: str) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
        """Compute per-sample HR zones with GMM and per-zone summary for one activity."""
        computed = self._compute_zones_with_metadata(activity_id)
        if computed is None:
            return None
        return computed["zones_ts_df"], computed["summary_df"]

    def compute_and_store_borders(
        self,
        activity_id: str,
        *,
        athlete_id: Optional[str] = None,
        activity_date: Optional[dt.date] = None,
        hr_speed_shift: Optional[object] = None,
    ) -> Optional[dict[str, object]]:
        computed = self._compute_zones_with_metadata(
            activity_id,
            athlete_id=athlete_id,
            activity_date=activity_date,
            hr_speed_shift=hr_speed_shift,
        )
        if computed is None:
            return None
        self.save_zone_summary(activity_id, computed["summary_df"])
        return self._build_borders_payload(
            borders=computed["borders"],
            zone_count_used=computed["zone_count_used"],
            sample_count=computed["sample_count"],
            fit_date=computed["fit_date"],
        )

    def save_zone_summary(self, activity_id: str, summary_df: pd.DataFrame) -> None:
        relative = f"{self._zones_dir}/{activity_id}.csv"
        self.storage.write_csv(relative, summary_df)

    def load_zone_summary(self, activity_id: str) -> Optional[pd.DataFrame]:
        relative = f"{self._zones_dir}/{activity_id}.csv"
        df = self.storage.read_csv(relative)
        if df.empty:
            return None
        df = _to_numeric(
            df,
            [
                "zone",
                "hr_mean",
                "hr_min",
                "hr_max",
                "time_seconds",
                "avg_speed_kmh",
                "avg_speedeq_kmh",
                "pct_time",
            ],
        )
        return self._ensure_all_zones(df)

    def load_zone_borders(self, activity_id: str) -> Optional[dict[str, object]]:
        metrics_df = self.storage.read_csv("activities_metrics.csv")
        if metrics_df.empty or "activityId" not in metrics_df.columns:
            return None
        match = metrics_df[metrics_df["activityId"].astype(str) == str(activity_id)]
        if match.empty:
            return None
        row = match.iloc[0]
        zone_count = _to_int(row.get("hrZone_zone_count"), self.zone_count)
        zone_count = max(2, min(zone_count, MAX_PERSISTED_BORDERS + 1))
        expected_borders = min(zone_count - 1, MAX_PERSISTED_BORDERS)
        borders: list[float] = []
        for idx in range(1, expected_borders + 1):
            raw = row.get(f"hrZone_z{idx}_upper")
            parsed = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
            if pd.notna(parsed):
                borders.append(float(parsed))
        if len(borders) != expected_borders:
            return None
        if any(borders[i] >= borders[i + 1] for i in range(len(borders) - 1)):
            return None
        return {"zone_count": zone_count, "borders": borders}

    def has_valid_zone_borders(self, activity_id: str) -> bool:
        return self.load_zone_borders(activity_id) is not None

    def get_or_compute_zones(self, activity_id: str) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
        """Return per-sample zone assignments from persisted borders and summary."""
        summary = self.load_zone_summary(activity_id)
        if summary is None or summary.empty:
            return None
        borders = self.load_zone_borders(activity_id)
        if borders is None:
            return None
        ts_df = self._prepare_activity_timeseries(activity_id)
        if ts_df is None:
            return None
        working = ts_df.dropna(subset=["hr_smooth"]).copy()
        if working.empty:
            return None
        zone_series = self._assign_zones_from_borders(
            working["hr_smooth"],
            borders["borders"],  # type: ignore[index]
        )
        working["zone"] = pd.to_numeric(zone_series, errors="coerce")
        working = working.dropna(subset=["zone"]).copy()
        if working.empty:
            return None
        working["zone"] = working["zone"].astype(int)
        working["cluster"] = working["zone"] - 1
        working["zone_label"] = working["zone"].map(lambda z: f"Z{z}")
        working = self._compute_durations(working)
        return working, summary

    def backfill_all_borders(self, athlete_id: Optional[str] = None) -> int:
        metrics_df = self.storage.read_csv("activities_metrics.csv")
        if metrics_df.empty or "activityId" not in metrics_df.columns:
            return 0
        metrics_df = self._ensure_border_columns(metrics_df)
        working = metrics_df.copy()
        if athlete_id:
            mask = working.get("athleteId", "").astype(str) == str(athlete_id)
            target_df = working[mask].copy()
        else:
            target_df = working.copy()
        if target_df.empty:
            return 0
        target_df["sort_date"] = pd.to_datetime(target_df.get("startDate"), errors="coerce")
        target_df = target_df.sort_values(["sort_date", "activityId"])
        updated = 0
        for idx, row in target_df.iterrows():
            activity_id = str(row.get("activityId") or "").strip()
            if not activity_id:
                continue
            row_athlete_id = str(row.get("athleteId") or "").strip() or None
            row_date = pd.to_datetime(row.get("startDate"), errors="coerce")
            payload = self.compute_and_store_borders(
                activity_id,
                athlete_id=row_athlete_id,
                activity_date=row_date.date() if pd.notna(row_date) else None,
                hr_speed_shift=row.get("hrSpeedShift"),
            )
            if payload is None:
                continue
            for key, value in payload.items():
                working.at[idx, key] = value
            updated += 1
        self.storage.write_csv("activities_metrics.csv", working)
        return updated

    def build_weekly_zone_data(
        self,
        athlete_id: str,
        start_date: dt.date,
        end_date: dt.date,
        categories: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        """Aggregate weekly time spent in each zone for selected activity filters."""
        activities_df = self._load_filtered_activities(
            athlete_id=athlete_id,
            start_date=start_date,
            end_date=end_date,
            categories=categories,
        )
        if activities_df.empty:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for row in activities_df.itertuples(index=False):
            activity_id = str(row.activityId)
            summary = self.load_zone_summary(activity_id)
            if summary is None or summary.empty:
                continue

            week_start = row.date - dt.timedelta(days=row.date.isoweekday() - 1)
            for summary_row in summary.itertuples(index=False):
                rows.append(
                    {
                        "weekStartDate": week_start,
                        "weekLabel": week_start.strftime("%Y-%m-%d"),
                        "zone": int(summary_row.zone),
                        "zone_label": f"Z{int(summary_row.zone)}",
                        "time_seconds": float(summary_row.time_seconds or 0.0),
                    }
                )

        if not rows:
            return pd.DataFrame()

        weekly_df = pd.DataFrame(rows)
        weekly_df = (
            weekly_df.groupby(["weekStartDate", "weekLabel", "zone", "zone_label"], as_index=False)[
                "time_seconds"
            ]
            .sum()
            .sort_values(["weekStartDate", "zone"])
            .reset_index(drop=True)
        )
        weekly_df["time_minutes"] = weekly_df["time_seconds"] / 60.0
        weekly_total = weekly_df.groupby("weekStartDate")["time_seconds"].transform("sum")
        weekly_df["pct_time"] = np.where(
            weekly_total > 0,
            (weekly_df["time_seconds"] / weekly_total) * 100.0,
            0.0,
        )
        return weekly_df

    def build_zone_speed_evolution(
        self,
        athlete_id: str,
        start_date: dt.date,
        end_date: dt.date,
        categories: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        """Build weekly weighted mean speed and speed-eq by zone over time."""
        activities_df = self._load_filtered_activities(
            athlete_id=athlete_id,
            start_date=start_date,
            end_date=end_date,
            categories=categories,
        )
        if activities_df.empty:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for activity in activities_df.itertuples(index=False):
            activity_id = str(activity.activityId)
            summary = self.load_zone_summary(activity_id)
            if summary is None or summary.empty:
                continue

            week_start = activity.date - dt.timedelta(days=activity.date.isoweekday() - 1)
            for zone_row in summary.itertuples(index=False):
                time_seconds = float(zone_row.time_seconds or 0.0)
                if time_seconds <= 0:
                    continue
                avg_speed = pd.to_numeric(pd.Series([zone_row.avg_speed_kmh]), errors="coerce").iloc[0]
                avg_speedeq = pd.to_numeric(
                    pd.Series([zone_row.avg_speedeq_kmh]),
                    errors="coerce",
                ).iloc[0]
                speed_weight_seconds = time_seconds if pd.notna(avg_speed) else 0.0
                speedeq_weight_seconds = time_seconds if pd.notna(avg_speedeq) else 0.0
                rows.append(
                    {
                        "weekStartDate": week_start,
                        "weekLabel": week_start.strftime("%Y-%m-%d"),
                        "zone": int(zone_row.zone),
                        "zone_label": f"Z{int(zone_row.zone)}",
                        "time_seconds": time_seconds,
                        "speed_weight_seconds": speed_weight_seconds,
                        "speedeq_weight_seconds": speedeq_weight_seconds,
                        "speed_contrib": (float(avg_speed) * speed_weight_seconds)
                        if pd.notna(avg_speed)
                        else 0.0,
                        "speedeq_contrib": (float(avg_speedeq) * speedeq_weight_seconds)
                        if pd.notna(avg_speedeq)
                        else 0.0,
                    }
                )

        if not rows:
            return pd.DataFrame()

        grouped = (
            pd.DataFrame(rows)
            .groupby(["weekStartDate", "weekLabel", "zone", "zone_label"], as_index=False)
            .agg(
                time_seconds=("time_seconds", "sum"),
                speed_weight_seconds=("speed_weight_seconds", "sum"),
                speedeq_weight_seconds=("speedeq_weight_seconds", "sum"),
                speed_contrib=("speed_contrib", "sum"),
                speedeq_contrib=("speedeq_contrib", "sum"),
            )
            .sort_values(["weekStartDate", "zone"])
            .reset_index(drop=True)
        )

        grouped["avg_speed_kmh"] = np.where(
            grouped["speed_weight_seconds"] > 0,
            grouped["speed_contrib"] / grouped["speed_weight_seconds"],
            np.nan,
        )
        grouped["avg_speedeq_kmh"] = np.where(
            grouped["speedeq_weight_seconds"] > 0,
            grouped["speedeq_contrib"] / grouped["speedeq_weight_seconds"],
            np.nan,
        )
        grouped["date"] = pd.to_datetime(grouped["weekStartDate"], errors="coerce")
        return grouped

    def build_activity_zone_speed_points(
        self,
        athlete_id: str,
        start_date: dt.date,
        end_date: dt.date,
        categories: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        activities_df = self._load_filtered_activities(
            athlete_id=athlete_id,
            start_date=start_date,
            end_date=end_date,
            categories=categories,
        )
        if activities_df.empty:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for activity in activities_df.itertuples(index=False):
            activity_id = str(activity.activityId)
            summary = self.load_zone_summary(activity_id)
            if summary is None or summary.empty:
                continue
            for zone_row in summary.itertuples(index=False):
                rows.append(
                    {
                        "activityId": activity_id,
                        "date": pd.to_datetime(activity.date, errors="coerce"),
                        "zone": int(zone_row.zone),
                        "zone_label": f"Z{int(zone_row.zone)}",
                        "time_seconds": float(zone_row.time_seconds or 0.0),
                        "avg_speed_kmh": pd.to_numeric(
                            pd.Series([zone_row.avg_speed_kmh]),
                            errors="coerce",
                        ).iloc[0],
                        "avg_speedeq_kmh": pd.to_numeric(
                            pd.Series([zone_row.avg_speedeq_kmh]),
                            errors="coerce",
                        ).iloc[0],
                    }
                )
        if not rows:
            return pd.DataFrame()
        points_df = pd.DataFrame(rows)
        points_df = points_df.dropna(subset=["date"]).sort_values(["date", "zone"]).reset_index(drop=True)
        return points_df

    def _load_filtered_activities(
        self,
        athlete_id: str,
        start_date: dt.date,
        end_date: dt.date,
        categories: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        df = self.storage.read_csv("activities_metrics.csv")
        if df.empty:
            return pd.DataFrame()

        needed_cols = {"activityId", "athleteId", "startDate", "category"}
        if not needed_cols.issubset(set(df.columns)):
            return pd.DataFrame()

        filtered = df[df["athleteId"].astype(str) == str(athlete_id)].copy()
        filtered["date"] = pd.to_datetime(filtered["startDate"], errors="coerce").dt.date
        filtered = filtered.dropna(subset=["date"])
        filtered = filtered[
            (filtered["date"] >= start_date)
            & (filtered["date"] <= end_date)
        ]

        if categories:
            allowed = {str(c).upper() for c in categories}
            filtered["category"] = filtered["category"].astype(str).str.upper()
            filtered = filtered[filtered["category"].isin(allowed)]

        return filtered[["activityId", "date"]].drop_duplicates().reset_index(drop=True)

    def _compute_zones_with_metadata(
        self,
        activity_id: str,
        *,
        athlete_id: Optional[str] = None,
        activity_date: Optional[dt.date] = None,
        hr_speed_shift: Optional[object] = None,
    ) -> Optional[dict[str, object]]:
        ts_df = self._prepare_activity_timeseries(activity_id)
        if ts_df is None:
            return None
        working = ts_df.dropna(subset=["hr_smooth"]).copy()
        if working.empty:
            return None
        context_athlete_id = athlete_id
        context_activity_date = activity_date
        if context_athlete_id is None or context_activity_date is None:
            loaded_athlete_id, loaded_activity_date = self._load_activity_context(activity_id)
            context_athlete_id = context_athlete_id or loaded_athlete_id
            context_activity_date = context_activity_date or loaded_activity_date

        reference_hr = self._build_reference_hr_samples(
            activity_id=activity_id,
            fallback_hr=working["hr_smooth"],
            athlete_id=context_athlete_id,
            activity_date=context_activity_date,
        )
        fitted = self._fit_gmm_on_hr(reference_hr)
        if fitted is None:
            return None

        gmm, zones, sorted_means = fitted
        hr_values = working["hr_smooth"].to_numpy(dtype=float).reshape(-1, 1)
        components = gmm.predict(hr_values)
        working["cluster"] = components.astype(int)
        working["zone"] = pd.Series([zones[c] for c in components], index=working.index).astype(int)
        working["zone_label"] = working["zone"].map(lambda z: f"Z{z}")
        working = self._compute_durations(working)

        resolved_shift = (
            hr_speed_shift if hr_speed_shift is not None else self._load_hr_speed_shift(activity_id)
        )
        summary = self._build_zone_summary(working, hr_speed_shift=resolved_shift)
        sample_count = int(reference_hr.dropna().shape[0])
        fit_date = context_activity_date or dt.date.today()
        return {
            "zones_ts_df": working,
            "summary_df": summary,
            "borders": self._compute_zone_borders(sorted_means),
            "zone_count_used": int(len(sorted_means)),
            "sample_count": sample_count,
            "fit_date": fit_date,
        }

    def _prepare_activity_timeseries(self, activity_id: str) -> Optional[pd.DataFrame]:
        ts_df = self.ts_service.load(activity_id)
        if ts_df is None or ts_df.empty:
            return None
        if "timestamp" not in ts_df.columns or "hr" not in ts_df.columns:
            return None
        ts_df = ts_df.copy()
        ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"], errors="coerce")
        ts_df = _to_numeric(ts_df, ["hr", "paceKmh"])
        ts_df = ts_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if ts_df.empty or ts_df["hr"].dropna().empty:
            return None
        ts_df["minutes"] = (ts_df["timestamp"] - ts_df["timestamp"].iloc[0]).dt.total_seconds() / 60.0
        ts_df["hr_smooth"] = ts_df["hr"].rolling(
            window=HR_SMOOTHING_WINDOW, min_periods=1, center=True
        ).mean()
        ts_df["speed_kmh"] = ts_df["paceKmh"] if "paceKmh" in ts_df.columns else np.nan
        self._attach_speedeq(activity_id, ts_df)
        return ts_df

    def _compute_durations(self, zones_df: pd.DataFrame) -> pd.DataFrame:
        working = zones_df.copy()
        working["duration_seconds"] = (
            working["timestamp"].diff().dt.total_seconds().clip(lower=0.0).fillna(0.0)
        )
        positive_durations = working.loc[working["duration_seconds"] > 0, "duration_seconds"]
        default_step = float(positive_durations.median()) if not positive_durations.empty else 1.0
        if not working.empty:
            working.iloc[0, working.columns.get_loc("duration_seconds")] = default_step
        return working

    def _load_activity_context(self, activity_id: str) -> tuple[Optional[str], Optional[dt.date]]:
        metrics_df = self.storage.read_csv("activities_metrics.csv")
        if metrics_df.empty:
            return None, None
        if "activityId" not in metrics_df.columns:
            return None, None

        match = metrics_df[metrics_df["activityId"].astype(str) == str(activity_id)]
        if match.empty:
            return None, None

        athlete_id = str(match.iloc[0].get("athleteId") or "").strip() or None
        activity_date = pd.to_datetime(match.iloc[0].get("startDate"), errors="coerce")
        return athlete_id, (activity_date.date() if pd.notna(activity_date) else None)

    def _build_reference_hr_samples(
        self,
        *,
        activity_id: str,
        fallback_hr: pd.Series,
        athlete_id: Optional[str],
        activity_date: Optional[dt.date],
    ) -> pd.Series:
        if not athlete_id or activity_date is None:
            return fallback_hr.dropna()

        window_start = activity_date - dt.timedelta(days=self.window_days)
        activities_df = self.storage.read_csv("activities_metrics.csv")
        if activities_df.empty:
            return fallback_hr.dropna()
        required_cols = {"activityId", "athleteId", "startDate"}
        if not required_cols.issubset(set(activities_df.columns)):
            return fallback_hr.dropna()

        activities_df = activities_df[activities_df["athleteId"].astype(str) == str(athlete_id)].copy()
        activities_df["date"] = pd.to_datetime(activities_df["startDate"], errors="coerce").dt.date
        activities_df = activities_df.dropna(subset=["date"])
        activities_df = activities_df[
            (activities_df["date"] >= window_start) & (activities_df["date"] <= activity_date)
        ]
        if self.fit_activity_types and "category" in activities_df.columns:
            allowed = {str(cat).upper() for cat in self.fit_activity_types}
            activities_df["category"] = activities_df["category"].astype(str).str.upper()
            activities_df = activities_df[activities_df["category"].isin(allowed)]
        if activities_df.empty:
            return fallback_hr.dropna()

        hr_samples: list[pd.Series] = []
        for aid in activities_df["activityId"].astype(str).drop_duplicates().tolist():
            series = self._load_activity_hr_smooth(aid)
            if not series.empty:
                hr_samples.append(series)

        if not hr_samples:
            return fallback_hr.dropna()

        monthly_hr = pd.concat(hr_samples, ignore_index=True).dropna()
        if monthly_hr.empty:
            return fallback_hr.dropna()

        # Ensure current activity contributes to fitting if missing from activities_metrics.
        if str(activity_id) not in activities_df["activityId"].astype(str).tolist():
            monthly_hr = pd.concat([monthly_hr, fallback_hr.dropna()], ignore_index=True)
        return monthly_hr

    def _load_activity_hr_smooth(self, activity_id: str) -> pd.Series:
        metrics_df = self.ts_service.load_metrics_ts(activity_id)
        if metrics_df is not None and not metrics_df.empty:
            if "hr_smooth" in metrics_df.columns:
                hr_smooth = pd.to_numeric(metrics_df["hr_smooth"], errors="coerce").dropna()
                if not hr_smooth.empty:
                    return hr_smooth
            if "hr" in metrics_df.columns:
                hr_values = pd.to_numeric(metrics_df["hr"], errors="coerce")
                hr_smooth = hr_values.rolling(
                    window=HR_SMOOTHING_WINDOW,
                    min_periods=1,
                    center=True,
                ).mean()
                hr_smooth = hr_smooth.dropna()
                if not hr_smooth.empty:
                    return hr_smooth

        raw_df = self.ts_service.load(activity_id)
        if raw_df is None or raw_df.empty or "hr" not in raw_df.columns:
            return pd.Series(dtype=float)
        hr_values = pd.to_numeric(raw_df["hr"], errors="coerce")
        hr_smooth = hr_values.rolling(window=HR_SMOOTHING_WINDOW, min_periods=1, center=True).mean()
        return hr_smooth.dropna()

    def _fit_gmm_on_hr(
        self,
        hr_series: pd.Series,
    ) -> Optional[tuple[GaussianMixture, dict[int, int], np.ndarray]]:
        values = hr_series.dropna().to_numpy(dtype=float).reshape(-1, 1)
        if values.shape[0] < 6:
            return None

        unique_count = int(np.unique(np.round(values.flatten(), 1)).size)
        max_components = min(self.zone_count, values.shape[0], unique_count)
        if max_components < 2:
            return None

        for n_components in range(max_components, 1, -1):
            try:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(values)
                means = gmm.means_.flatten()
                order = np.argsort(means)
                zone_map = {int(component): int(rank + 1) for rank, component in enumerate(order)}
                return gmm, zone_map, np.sort(means)
            except Exception:
                continue
        return None

    def _attach_speedeq(self, activity_id: str, ts_df: pd.DataFrame) -> None:
        if "speed_kmh" in ts_df.columns:
            speed_fallback = pd.to_numeric(ts_df["speed_kmh"], errors="coerce")
        else:
            speed_fallback = pd.Series(np.nan, index=ts_df.index)

        metrics_df = self.ts_service.load_metrics_ts(activity_id)

        if metrics_df is None or metrics_df.empty or "speedeq_smooth" not in metrics_df.columns:
            ts_df["speedeq_smooth"] = speed_fallback
            return

        speedeq = pd.to_numeric(metrics_df["speedeq_smooth"], errors="coerce").reset_index(drop=True)
        ts_df["speedeq_smooth"] = speed_fallback
        min_len = min(len(ts_df), len(speedeq))
        if min_len > 0:
            fallback_slice = speed_fallback.iloc[:min_len].reset_index(drop=True)
            speedeq_slice = speedeq.iloc[:min_len]
            ts_df.loc[: min_len - 1, "speedeq_smooth"] = (
                speedeq_slice.where(speedeq_slice.notna(), fallback_slice).to_numpy()
            )

    def _build_zone_summary(self, zones_df: pd.DataFrame, hr_speed_shift: Optional[object]) -> pd.DataFrame:
        working = zones_df.copy()
        shift_steps = _to_int(hr_speed_shift, 0)
        if "speed_kmh" in working.columns:
            speed_aligned = pd.to_numeric(working["speed_kmh"], errors="coerce")
        else:
            speed_aligned = pd.Series(np.nan, index=working.index, dtype="float64")
        if "speedeq_smooth" in working.columns:
            speedeq_aligned = pd.to_numeric(working["speedeq_smooth"], errors="coerce")
        else:
            speedeq_aligned = pd.Series(np.nan, index=working.index, dtype="float64")
        if shift_steps != 0:
            speed_aligned = speed_aligned.shift(-shift_steps)
            speedeq_aligned = speedeq_aligned.shift(-shift_steps)
        working["speed_aligned_kmh"] = speed_aligned
        working["speedeq_aligned_kmh"] = speedeq_aligned

        grouped = (
            working.groupby("zone", as_index=False)
            .agg(
                hr_mean=("hr_smooth", "mean"),
                hr_min=("hr_smooth", "min"),
                hr_max=("hr_smooth", "max"),
                time_seconds=("duration_seconds", "sum"),
                avg_speed_kmh=("speed_aligned_kmh", "mean"),
                avg_speedeq_kmh=("speedeq_aligned_kmh", "mean"),
            )
            .sort_values("zone")
        )
        grouped = self._ensure_all_zones(grouped)
        total = float(grouped["time_seconds"].sum())
        grouped["pct_time"] = np.where(
            total > 0,
            grouped["time_seconds"] / total * 100.0,
            0.0,
        )
        grouped["zone"] = grouped["zone"].astype(int)
        grouped["zone_label"] = grouped["zone"].map(lambda z: f"Z{int(z)}")
        return grouped[
            [
                "zone",
                "zone_label",
                "hr_mean",
                "hr_min",
                "hr_max",
                "time_seconds",
                "avg_speed_kmh",
                "avg_speedeq_kmh",
                "pct_time",
            ]
        ].sort_values("zone")

    def _ensure_all_zones(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        base = pd.DataFrame({"zone": list(range(1, self.zone_count + 1))})
        summary_df = summary_df.copy()
        merged = base.merge(summary_df, on="zone", how="left")
        merged["time_seconds"] = pd.to_numeric(merged.get("time_seconds"), errors="coerce").fillna(0.0)
        for col in ("hr_mean", "hr_min", "hr_max", "avg_speed_kmh", "avg_speedeq_kmh", "pct_time"):
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")
            else:
                merged[col] = np.nan
        if "zone_label" not in merged.columns:
            merged["zone_label"] = merged["zone"].map(lambda z: f"Z{int(z)}")
        else:
            merged["zone_label"] = merged["zone"].map(lambda z: f"Z{int(z)}")
        return merged.sort_values("zone").reset_index(drop=True)

    def _load_hr_speed_shift(self, activity_id: str) -> int:
        metrics_df = self.storage.read_csv("activities_metrics.csv")
        if metrics_df.empty or "activityId" not in metrics_df.columns:
            return 0
        match = metrics_df[metrics_df["activityId"].astype(str) == str(activity_id)]
        if match.empty:
            return 0
        return _to_int(match.iloc[0].get("hrSpeedShift"), 0)

    def _compute_zone_borders(self, sorted_means: np.ndarray) -> list[float]:
        if sorted_means.size < 2:
            return []
        return [
            float((sorted_means[idx] + sorted_means[idx + 1]) / 2.0)
            for idx in range(sorted_means.size - 1)
        ]

    def _build_borders_payload(
        self,
        *,
        borders: list[float],
        zone_count_used: int,
        sample_count: int,
        fit_date: dt.date,
    ) -> dict[str, object]:
        payload: dict[str, object] = {col: "" for col in BORDER_COLUMNS}
        for idx, border in enumerate(borders[:MAX_PERSISTED_BORDERS], start=1):
            payload[f"hrZone_z{idx}_upper"] = float(border)
        payload["hrZone_gmm_sample_count"] = int(sample_count)
        payload["hrZone_gmm_fit_date"] = str(fit_date)
        payload["hrZone_zone_count"] = int(max(2, min(zone_count_used, MAX_PERSISTED_BORDERS + 1)))
        return payload

    def _assign_zones_from_borders(self, hr_series: pd.Series, borders: Sequence[float]) -> pd.Series:
        if not borders:
            return pd.Series(np.nan, index=hr_series.index, dtype="float64")
        bins = [-np.inf, *list(borders), np.inf]
        bucket = pd.cut(hr_series, bins=bins, labels=False, include_lowest=True)
        return pd.to_numeric(bucket, errors="coerce") + 1

    def _ensure_border_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in [*BORDER_COLUMNS, *ZONE_METADATA_COLUMNS]:
            if col not in out.columns:
                out[col] = pd.Series(dtype="object")
        return out
