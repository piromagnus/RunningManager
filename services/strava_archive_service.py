"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Import Strava GDPR “Download Your Data” ZIP archives into the same storage
layout as API sync (activities.csv, raw/strava/{id}.json, timeseries/{id}.csv).
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Union

import pandas as pd

from persistence.csv_storage import CsvStorage
from services.strava_service import StravaService
from utils.config import Config
from utils.strava_track_parser import parse_strava_track_file

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]
ZipSource = Union[bytes, BinaryIO, Path, str]


@dataclass
class StravaArchiveService:
    """Load official Strava GDPR export ZIPs into app storage."""

    storage: CsvStorage
    config: Config
    strava: Optional[StravaService] = None

    def __post_init__(self) -> None:
        self.strava = self.strava or StravaService(storage=self.storage, config=self.config)

    def import_strava_archive(
        self,
        zip_source: ZipSource,
        athlete_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Import activities from a Strava GDPR ZIP.

        Uses Strava Activity ID as ``activityId`` so later API sync merges instead
        of duplicating. Fills missing artifacts/fields only.

        Returns counts: imported, merged, already_complete, missing_file,
        parse_failed, touched_ids.
        """
        stats: Dict[str, Any] = {
            "imported": 0,
            "merged": 0,
            "already_complete": 0,
            "missing_file": 0,
            "parse_failed": 0,
            "touched_ids": [],
        }
        touched: List[str] = []
        touched_with_ts: set[str] = set()
        start_dates: list[dt.date] = []

        with self._open_zip(zip_source) as zf:
            activities_csv_name = self._find_activities_csv(zf)
            if not activities_csv_name:
                raise RuntimeError(
                    "Archive Strava invalide: fichier activities.csv introuvable."
                )
            rows = self._read_activities_csv(zf, activities_csv_name)
            if not rows:
                raise RuntimeError(
                    "Archive Strava invalide: aucune Activity ID dans activities.csv."
                )

            total = len(rows)
            for index, row in enumerate(rows, start=1):
                activity_id = str(row.get("activity_id") or "").strip()
                name = str(row.get("name") or activity_id)
                if progress_callback is not None:
                    progress_callback(index, total, name)
                if not activity_id:
                    continue

                result = self._import_one_activity(
                    zf=zf,
                    athlete_id=athlete_id,
                    meta=row,
                )
                status = result["status"]
                if status == "imported":
                    stats["imported"] += 1
                    touched.append(activity_id)
                elif status == "merged":
                    stats["merged"] += 1
                    touched.append(activity_id)
                else:
                    stats["already_complete"] += 1

                if result.get("missing_file"):
                    stats["missing_file"] += 1
                if result.get("parse_failed"):
                    stats["parse_failed"] += 1

                if result.get("has_timeseries") and status in {"imported", "merged"}:
                    touched_with_ts.add(activity_id)
                if result.get("start_date") is not None and status in {"imported", "merged"}:
                    start_dates.append(result["start_date"])

        # Deduplicate while preserving order
        touched = list(dict.fromkeys(touched))
        stats["touched_ids"] = touched
        if touched:
            assert self.strava is not None
            self.strava._apply_sync_metrics(  # noqa: SLF001 — shared metrics path
                athlete_id=athlete_id,
                created_rows=touched,
                created_rows_with_ts=touched_with_ts,
                created_start_dates=start_dates,
            )
        return stats

    def _import_one_activity(
        self,
        *,
        zf: zipfile.ZipFile,
        athlete_id: str,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        assert self.strava is not None
        activity_id = str(meta["activity_id"])
        detail = self._detail_from_csv_row(meta)
        filename = str(meta.get("filename") or "").strip()

        had_ts = self.strava._timeseries_exists(activity_id)  # noqa: SLF001

        merged_detail, raw_path, raw_changed = self.strava.merge_and_save_raw(
            activity_id, detail
        )

        wrote_ts = False
        parse_failed = False
        missing_file = False
        if not had_ts:
            if not filename:
                missing_file = True
            else:
                track_member = self._resolve_track_member(zf, filename)
                if track_member is None:
                    missing_file = True
                    LOGGER.warning(
                        "Track file missing in archive for activity %s (%s)",
                        activity_id,
                        filename,
                    )
                else:
                    try:
                        track_bytes = zf.read(track_member)
                        ts_df = parse_strava_track_file(track_bytes, filename=track_member)
                        if ts_df is None or ts_df.empty:
                            parse_failed = True
                            LOGGER.warning(
                                "Failed to parse track for activity %s (%s)",
                                activity_id,
                                filename,
                            )
                        else:
                            wrote_ts = self.strava.save_timeseries_dataframe(
                                activity_id, ts_df
                            )
                    except Exception:
                        parse_failed = True
                        LOGGER.exception(
                            "Error parsing track for activity %s (%s)",
                            activity_id,
                            filename,
                        )

        has_timeseries = self.strava._timeseries_exists(activity_id)  # noqa: SLF001
        _, created, row_merged = self.strava.upsert_activity_row_from_detail(
            athlete_id=athlete_id,
            detail=merged_detail,
            has_timeseries=has_timeseries,
            raw_path=raw_path,
        )

        changed = created or row_merged or raw_changed or wrote_ts
        if created:
            status = "imported"
        elif changed:
            status = "merged"
        else:
            status = "already_complete"

        return {
            "status": status,
            "missing_file": missing_file,
            "parse_failed": parse_failed,
            "has_timeseries": has_timeseries,
            "start_date": self._start_date_from_detail(merged_detail),
        }

    @staticmethod
    def _open_zip(zip_source: ZipSource) -> zipfile.ZipFile:
        if isinstance(zip_source, (str, Path)):
            return zipfile.ZipFile(zip_source, "r")
        if isinstance(zip_source, bytes):
            return zipfile.ZipFile(io.BytesIO(zip_source), "r")
        return zipfile.ZipFile(zip_source, "r")

    @staticmethod
    def _find_activities_csv(zf: zipfile.ZipFile) -> Optional[str]:
        candidates = [
            name
            for name in zf.namelist()
            if name.lower().endswith("activities.csv") and not name.endswith("/")
        ]
        if not candidates:
            return None
        # Prefer root-level activities.csv
        for name in candidates:
            if name.lower() == "activities.csv" or name.lower().endswith("/activities.csv"):
                if name.count("/") <= 1:
                    return name
        return candidates[0]

    def _read_activities_csv(
        self, zf: zipfile.ZipFile, member_name: str
    ) -> List[Dict[str, Any]]:
        raw = zf.read(member_name)
        text = raw.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            return []
        field_map = {self._norm_header(h): h for h in reader.fieldnames if h}
        id_key = field_map.get("activity id") or field_map.get("activityid")
        if not id_key:
            return []

        rows: List[Dict[str, Any]] = []
        for raw_row in reader:
            activity_id = str(raw_row.get(id_key) or "").strip()
            if not activity_id:
                continue
            rows.append(
                {
                    "activity_id": activity_id,
                    "name": self._cell(raw_row, field_map, "activity name", "name"),
                    "activity_type": self._cell(
                        raw_row, field_map, "activity type", "type", "sport type"
                    ),
                    "activity_date": self._cell(
                        raw_row, field_map, "activity date", "date"
                    ),
                    "elapsed_time": self._cell(
                        raw_row, field_map, "elapsed time", "elapsed time (s)"
                    ),
                    "moving_time": self._cell(
                        raw_row, field_map, "moving time", "moving time (s)"
                    ),
                    "distance": self._cell(
                        raw_row, field_map, "distance", "distance (m)", "distance (km)"
                    ),
                    "elevation_gain": self._cell(
                        raw_row,
                        field_map,
                        "elevation gain",
                        "elevation gain (m)",
                        "total elevation gain",
                    ),
                    "avg_hr": self._cell(
                        raw_row,
                        field_map,
                        "average heart rate",
                        "average heartrate",
                        "avg heart rate",
                        "avg hr",
                    ),
                    "max_hr": self._cell(
                        raw_row,
                        field_map,
                        "max heart rate",
                        "maximum heart rate",
                        "max hr",
                    ),
                    "filename": self._cell(raw_row, field_map, "filename", "file name"),
                    "_distance_header": (
                        "km"
                        if any(k.endswith("(km)") for k in field_map)
                        and "distance (km)" in field_map
                        else "m"
                    ),
                }
            )
        return rows

    @staticmethod
    def _norm_header(value: str) -> str:
        return " ".join(str(value).strip().lower().replace("_", " ").split())

    @staticmethod
    def _cell(
        row: Dict[str, Any], field_map: Dict[str, str], *aliases: str
    ) -> Optional[str]:
        for alias in aliases:
            key = field_map.get(alias)
            if key is None:
                continue
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    def _detail_from_csv_row(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        activity_id = str(meta["activity_id"])
        start_iso = self._parse_activity_date(meta.get("activity_date"))
        distance_m = self._parse_distance_m(
            meta.get("distance"), unit_hint=str(meta.get("_distance_header") or "m")
        )
        sport = str(meta.get("activity_type") or "").strip()
        detail: Dict[str, Any] = {
            "id": int(activity_id) if activity_id.isdigit() else activity_id,
            "name": meta.get("name") or "",
            "sport_type": sport,
            "type": sport,
            "start_date": start_iso,
            "start_date_local": start_iso,
            "distance": distance_m,
            "elapsed_time": self._parse_int(meta.get("elapsed_time")),
            "moving_time": self._parse_int(meta.get("moving_time")),
            "total_elevation_gain": self._parse_float(meta.get("elevation_gain")),
            "average_heartrate": self._parse_float(meta.get("avg_hr")),
            "max_heartrate": self._parse_float(meta.get("max_hr")),
            "laps": [],
            "map": {},
        }
        return detail

    @staticmethod
    def _parse_activity_date(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime().isoformat().replace("+00:00", "Z")

    @staticmethod
    def _parse_distance_m(value: Optional[str], *, unit_hint: str = "m") -> Optional[float]:
        if value is None or str(value).strip() == "":
            return None
        try:
            number = float(str(value).replace(",", ""))
        except ValueError:
            return None
        # Official export Distance is meters; some docs label km
        if unit_hint == "km" or (number > 0 and number < 1000 and "km" in unit_hint):
            # Heuristic: values under 1000 with km header are km
            if unit_hint == "km":
                return number * 1000.0
        return number

    @staticmethod
    def _parse_int(value: Optional[str]) -> Optional[int]:
        if value is None or str(value).strip() == "":
            return None
        try:
            return int(float(str(value).replace(",", "")))
        except ValueError:
            return None

    @staticmethod
    def _parse_float(value: Optional[str]) -> Optional[float]:
        if value is None or str(value).strip() == "":
            return None
        try:
            return float(str(value).replace(",", ""))
        except ValueError:
            return None

    @staticmethod
    def _start_date_from_detail(detail: Dict[str, Any]) -> Optional[dt.date]:
        try:
            start = pd.to_datetime(
                detail.get("start_date_local") or detail.get("start_date"),
                errors="coerce",
            )
            if pd.notna(start):
                return start.date()
        except Exception:
            return None
        return None

    def _resolve_track_member(self, zf: zipfile.ZipFile, filename: str) -> Optional[str]:
        names = zf.namelist()
        cleaned = filename.lstrip("./")
        candidates = [
            cleaned,
            cleaned.replace("\\", "/"),
            f"activities/{Path(cleaned).name}",
            Path(cleaned).name,
        ]
        name_set = set(names)
        for candidate in candidates:
            if candidate in name_set:
                return candidate
        # Case-insensitive / suffix match on basename
        base = Path(cleaned).name.lower()
        for name in names:
            if Path(name).name.lower() == base:
                return name
            if Path(name).name.lower() == f"{base}.gz" or base.endswith(".gz") and Path(
                name
            ).name.lower() == base:
                return name
        return None
