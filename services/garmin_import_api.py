"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Python API integration for importing Garmin Connect activities.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd
from garminconnect import Garmin

from persistence.csv_storage import CsvStorage
from persistence.repositories import ActivitiesRepo
from services.metrics_service import MetricsComputationService
from utils.config import Config

LOGGER = logging.getLogger(__name__)
GARMIN_PROVIDER = "garmin"
PAGE_SIZE = 100


class GarminClient(Protocol):
    """Subset of Garmin Connect's client used by this integration."""

    def login(self, token_store_file: str) -> Any: ...

    def get_activities(
        self, start: int = 0, limit: int = 20, activitytype: str | None = None
    ) -> list[dict[str, Any]] | dict[str, Any]: ...

    def get_activity(self, activity_id: str) -> dict[str, Any]: ...

    def get_activity_details(self, activity_id: str) -> dict[str, Any]: ...


@dataclass
class GarminImportService:
    """Import Garmin Connect activities into RunningManager's CSV data model."""

    storage: CsvStorage
    config: Config
    client_factory: Callable[..., GarminClient] = Garmin
    last_sync_stats: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.activities = ActivitiesRepo(self.storage)

    @property
    def is_ready(self) -> bool:
        return bool(self.config.garmin_email and self.config.garmin_password)

    def sync_last_n_days(self, athlete_id: str, days: int, mfa_code: str | None = None) -> list[str]:
        """Fetch new activities from Garmin and recompute RunningManager metrics."""
        if not athlete_id.strip():
            raise ValueError("athlete_id must not be empty")
        if days <= 0:
            raise ValueError("days must be positive")

        client = self._authenticated_client(mfa_code)
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
        existing_ids = self._existing_activity_ids()
        imported_ids: list[str] = []
        imported_with_timeseries: list[str] = []
        invalid_date_count = 0
        offset = 0

        while True:
            page = client.get_activities(start=offset, limit=PAGE_SIZE)
            if not isinstance(page, list):
                raise RuntimeError("Garmin Connect returned an invalid activity list")
            if not page:
                break

            reached_cutoff = False
            for summary in page:
                started_at = self._activity_datetime(summary)
                if started_at is None:
                    invalid_date_count += 1
                    LOGGER.warning("Skipping Garmin activity with an invalid start time")
                    continue
                if started_at < cutoff:
                    reached_cutoff = True
                    continue

                activity_id = self._activity_id(summary)
                if not activity_id or activity_id in existing_ids:
                    continue

                provider_activity = client.get_activity(activity_id)
                if not isinstance(provider_activity, dict):
                    raise RuntimeError(f"Garmin returned invalid activity data for {activity_id}")
                activity_summary = {**summary, **provider_activity}
                provider_details = client.get_activity_details(activity_id)
                if not isinstance(provider_details, dict):
                    raise RuntimeError(f"Garmin returned invalid timeseries data for {activity_id}")

                normalized = self._normalize_activity(activity_summary)
                raw_path = self._save_raw_activity(activity_id, normalized, provider_details)
                has_timeseries = self._save_timeseries(
                    activity_id,
                    provider_details,
                    normalized.get("start_date"),
                )
                self.activities.create(
                    self._map_activity_row(
                        normalized,
                        athlete_id,
                        has_timeseries,
                        raw_path,
                    )
                )
                existing_ids.add(activity_id)
                imported_ids.append(activity_id)
                if has_timeseries:
                    imported_with_timeseries.append(activity_id)

            offset += len(page)
            if len(page) < PAGE_SIZE or reached_cutoff:
                break

        if imported_ids:
            MetricsComputationService(self.storage, config=self.config).recompute_for_activities(imported_ids)

        self.last_sync_stats = {
            "days": days,
            "downloaded_count": len(imported_ids),
            "timeseries_count": len(imported_with_timeseries),
            "invalid_date_count": invalid_date_count,
            "downloaded_ids": imported_ids,
        }
        return imported_ids

    def _authenticated_client(self, mfa_code: str | None = None) -> GarminClient:
        if not self.config.garmin_email or not self.config.garmin_password:
            raise RuntimeError("Set GARMIN_EMAIL and GARMIN_PASSWORD in the RunningManager .env file.")
        if self.config.garmin_domain not in {"garmin.com", "garmin.cn"}:
            raise RuntimeError("GARMIN_DOMAIN must be either garmin.com or garmin.cn.")
        token_store = self.config.garmin_token_store or (self.config.data_dir / "garmin_tokens")
        token_store.parent.mkdir(parents=True, exist_ok=True)
        client_options: dict[str, Any] = {
            "email": self.config.garmin_email,
            "password": self.config.garmin_password,
            "is_cn": self.config.garmin_domain == "garmin.cn",
        }
        if mfa_code:
            client_options["prompt_mfa"] = lambda: mfa_code
        client = self.client_factory(**client_options)
        client.login(str(token_store))
        return client

    def _existing_activity_ids(self) -> set[str]:
        activities_df = self.activities.list()
        if activities_df.empty or "activityId" not in activities_df.columns:
            return set()
        return set(activities_df["activityId"].dropna().astype(str))

    def _save_raw_activity(
        self,
        activity_id: str,
        normalized: dict[str, Any],
        provider_details: dict[str, Any],
    ) -> Path:
        raw_dir = self.config.data_dir / "raw" / GARMIN_PROVIDER
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{activity_id}.json"
        payload = {
            **normalized,
            "garminActivityDetails": provider_details,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def _save_timeseries(
        self,
        activity_id: str,
        details: dict[str, Any],
        start_time: Any,
    ) -> bool:
        descriptors = details.get("metricDescriptors")
        detail_metrics = details.get("activityDetailMetrics")
        if not isinstance(descriptors, list) or not isinstance(detail_metrics, list):
            return False

        metric_indexes = {
            str(descriptor.get("key")): index
            for index, descriptor in enumerate(descriptors)
            if isinstance(descriptor, dict) and descriptor.get("key")
        }
        timestamp_index = metric_indexes.get("directTimestamp")
        if timestamp_index is None:
            return False

        base_time = self._parse_datetime(start_time)
        rows: list[dict[str, Any]] = []
        for sample in detail_metrics:
            if not isinstance(sample, dict) or not isinstance(sample.get("metrics"), list):
                continue
            values = sample["metrics"]
            if timestamp_index >= len(values):
                continue
            timestamp = self._sample_datetime(values[timestamp_index], base_time)
            if timestamp is None:
                continue

            speed = self._metric_value(values, metric_indexes, "directSpeed")
            rows.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "hr": self._metric_value(values, metric_indexes, "directHeartRate"),
                    "paceKmh": speed * 3.6 if speed is not None else None,
                    "elevationM": self._metric_value(values, metric_indexes, "directElevation"),
                    "cadence": self._first_metric_value(
                        values,
                        metric_indexes,
                        "directRunCadence",
                        "directCadence",
                    ),
                    "lat": self._metric_value(values, metric_indexes, "directLatitude"),
                    "lon": self._metric_value(values, metric_indexes, "directLongitude"),
                }
            )

        if not rows:
            return False
        path = self.config.timeseries_dir / f"{activity_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        return True

    @staticmethod
    def _activity_id(activity: dict[str, Any]) -> str:
        value = activity.get("activityId") or activity.get("activity_id") or activity.get("id")
        try:
            activity_id = int(value)
        except (TypeError, ValueError):
            return ""
        return str(activity_id) if activity_id > 0 else ""

    @classmethod
    def _activity_datetime(cls, activity: dict[str, Any]) -> dt.datetime | None:
        return cls._parse_datetime(
            activity.get("startTimeGMT") or activity.get("startTime") or activity.get("startTimeLocal")
        )

    @staticmethod
    def _parse_datetime(value: Any) -> dt.datetime | None:
        if isinstance(value, dt.datetime):
            parsed = value
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            timestamp = float(value)
            if abs(timestamp) > 100_000_000_000:
                timestamp /= 1000
            try:
                parsed = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        elif isinstance(value, str) and value.strip():
            try:
                parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)

    @classmethod
    def _sample_datetime(
        cls,
        value: Any,
        base_time: dt.datetime | None,
    ) -> dt.datetime | None:
        if isinstance(value, str):
            parsed = cls._parse_datetime(value)
            if parsed is not None:
                return parsed
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            timestamp = float(value)
            if abs(timestamp) > 100_000_000_000:
                timestamp /= 1000
                try:
                    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                except (OverflowError, OSError, ValueError):
                    return None
            if timestamp > 1_000_000_000:
                try:
                    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)
                except (OverflowError, OSError, ValueError):
                    return None
            if base_time is not None:
                return base_time + dt.timedelta(seconds=timestamp)
        return None

    @staticmethod
    def _metric_value(
        values: list[Any],
        metric_indexes: dict[str, int],
        key: str,
    ) -> float | None:
        index = metric_indexes.get(key)
        if index is None or index >= len(values):
            return None
        try:
            number = float(values[index])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        if key in {"directHeartRate", "directSpeed", "directRunCadence", "directCadence"} and number < 0:
            return None
        return number

    @classmethod
    def _first_metric_value(
        cls,
        values: list[Any],
        metric_indexes: dict[str, int],
        *keys: str,
    ) -> float | None:
        for key in keys:
            value = cls._metric_value(values, metric_indexes, key)
            if value is not None:
                return value
        return None

    @classmethod
    def _normalize_activity(cls, activity: dict[str, Any]) -> dict[str, Any]:
        activity_type = activity.get("activityTypeDTO")
        type_key = activity_type.get("typeKey") if isinstance(activity_type, dict) else None
        raw_sport = str(type_key or activity.get("sportType") or activity.get("activityType") or "")
        normalized_sport = cls._normalize_sport_type(raw_sport)
        distance = cls._first_number(activity, "distance", "distanceInMeters")
        elevation = cls._first_number(activity, "elevationGain", "totalElevationGain", "total_elevation_gain")
        elapsed = cls._first_number(activity, "duration", "elapsed_time")
        moving = cls._first_number(activity, "movingDuration", "moving_time")
        local_start = activity.get("startTimeLocal") or activity.get("start_date_local")
        gmt_start = activity.get("startTimeGMT") or activity.get("start_date") or local_start
        average_hr = cls._first_number(activity, "averageHR", "averageHeartRate", "average_heartrate")
        max_hr = cls._first_number(activity, "maxHR", "maxHeartRate", "max_heartrate")
        name = activity.get("activityName") or activity.get("name") or ""

        return {
            "id": cls._activity_id(activity),
            "name": str(name),
            "sport_type": normalized_sport,
            "start_date": gmt_start,
            "start_date_local": local_start,
            "distance": distance,
            "elapsed_time": elapsed,
            "moving_time": moving if moving is not None else elapsed,
            "total_elevation_gain": elevation,
            "average_heartrate": average_hr,
            "max_heartrate": max_hr,
            "map": {},
        }

    @staticmethod
    def _first_number(activity: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = activity.get(key)
            if value is None or value == "":
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
        return None

    @staticmethod
    def _normalize_sport_type(value: str) -> str:
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        categories = {
            "running": "RUN",
            "run": "RUN",
            "trail_running": "TRAIL_RUN",
            "trail_run": "TRAIL_RUN",
            "hiking": "HIKE",
            "walking": "HIKE",
            "cycling": "RIDE",
            "road_biking": "RIDE",
            "mountain_biking": "RIDE",
            "cross_country_skiing": "BACKCOUNTRY_SKI",
            "backcountry_skiing": "BACKCOUNTRY_SKI",
        }
        return categories.get(key, value.strip().upper())

    def _map_activity_row(
        self,
        activity: dict[str, Any],
        athlete_id: str,
        has_timeseries: bool,
        raw_path: Path,
    ) -> dict[str, Any]:
        distance_m = activity.get("distance")
        return {
            "activityId": str(activity["id"]),
            "athleteId": athlete_id,
            "source": GARMIN_PROVIDER,
            "sportType": activity.get("sport_type", ""),
            "name": activity.get("name", ""),
            "startTime": activity.get("start_date_local") or activity.get("start_date"),
            "distanceKm": float(distance_m) / 1000 if distance_m is not None else None,
            "elapsedSec": activity.get("elapsed_time"),
            "movingSec": activity.get("moving_time"),
            "ascentM": activity.get("total_elevation_gain"),
            "avgHr": activity.get("average_heartrate"),
            "maxHr": activity.get("max_heartrate"),
            "hasTimeseries": has_timeseries,
            "polyline": None,
            "rawJsonPath": self._relative_to_data_dir(raw_path),
        }

    def _relative_to_data_dir(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.config.data_dir))
        except ValueError:
            return str(path)
