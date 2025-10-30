"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Per-activity lap metrics computation and storage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo
from services.planner_service import PlannerService
from utils.config import Config
from utils.coercion import safe_float_optional, safe_int_optional


@dataclass
class LapMetricsService:
    storage: CsvStorage
    config: Config

    def __post_init__(self) -> None:
        self._athletes = AthletesRepo(self.storage)
        self._planner = PlannerService(self.storage)

    # ------------------------------------------------------------------
    # Public API
    def compute_and_store(self, athlete_id: str, detail: Dict[str, object]) -> Optional[Path]:
        """Compute lap metrics for a Strava activity and persist them under laps/{activity}.csv."""
        activity_id = str(detail.get("id") or "").strip()
        if not activity_id:
            return None

        laps = detail.get("laps") or []
        if not isinstance(laps, list):
            laps = []

        hr_profile = self._hr_profile(athlete_id)
        rows: List[Dict[str, object]] = []

        for idx, lap in enumerate(laps, start=1):
            if not isinstance(lap, dict):
                continue
            record = self._build_record(idx, lap, hr_profile)
            if record:
                rows.append(record)

        df = self._to_frame(rows)
        path = self.config.laps_dir / f"{activity_id}.csv"
        df.to_csv(path, index=False)
        return path

    def load(self, activity_id: str) -> Optional[pd.DataFrame]:
        """Load lap metrics for an activity if available."""
        path = self.config.laps_dir / f"{activity_id}.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty:
            return df
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    def _hr_profile(self, athlete_id: str) -> Optional[Tuple[float, float]]:
        row = self._athletes.get(str(athlete_id))
        if not row:
            return None
        hr_rest = safe_float_optional(row.get("hrRest"))
        hr_max = safe_float_optional(row.get("hrMax"))
        if hr_rest is None or hr_max is None:
            return None
        if hr_max <= hr_rest or hr_max <= 0:
            return None
        return float(hr_rest), float(hr_max)

    def _build_record(
        self,
        index: int,
        lap: Dict[str, object],
        hr_profile: Optional[Tuple[float, float]],
    ) -> Optional[Dict[str, object]]:
        elapsed = safe_float_optional(lap.get("elapsed_time"))
        moving = safe_float_optional(lap.get("moving_time"))
        if moving is None or moving <= 0:
            moving = elapsed
        if moving is None or moving <= 0:
            return None

        distance_m = safe_float_optional(lap.get("distance"))
        ascent_m = safe_float_optional(lap.get("total_elevation_gain"))
        avg_hr = safe_float_optional(lap.get("average_heartrate"))
        max_hr = safe_float_optional(lap.get("max_heartrate"))
        avg_speed_kmh = self._average_speed_kmh(lap, distance_m, moving)
        distance_km = (distance_m / 1000.0) if distance_m is not None else None
        distance_eq_km = (
            self._planner.compute_distance_eq_km(distance_km, ascent_m or 0.0)
            if distance_km is not None
            else None
        )

        hr_reserve_ratio = self._hr_reserve_ratio(avg_hr, hr_profile)
        hr_percent_max = self._hr_percent_max(avg_hr, hr_profile)
        trimp = self._compute_trimp(avg_hr, moving, hr_profile)
        label = self._classify_lap(hr_reserve_ratio, avg_hr, avg_speed_kmh)

        return {
            "lapIndex": index,
            "split": lap.get("split") or lap.get("lap_index") or index,
            "name": lap.get("name") or "",
            "startTime": lap.get("start_date") or lap.get("start_date_local") or "",
            "distanceKm": round(distance_km, 3) if distance_km is not None else None,
            "timeSec": round(moving, 1) if moving is not None else None,
            "elapsedSec": round(elapsed, 1) if elapsed is not None else None,
            "avgSpeedKmh": round(avg_speed_kmh, 2) if avg_speed_kmh is not None else None,
            "ascentM": round(ascent_m, 1) if ascent_m is not None else None,
            "distanceEqKm": round(distance_eq_km, 3) if distance_eq_km is not None else None,
            "avgHr": round(avg_hr, 1) if avg_hr is not None else None,
            "maxHr": round(max_hr, 1) if max_hr is not None else None,
            "hrReserveRatio": round(hr_reserve_ratio, 3) if hr_reserve_ratio is not None else None,
            "hrPercentMax": round(hr_percent_max, 1) if hr_percent_max is not None else None,
            "trimp": round(trimp, 2) if trimp is not None else None,
            "label": label,
        }

    @staticmethod
    def _average_speed_kmh(
        lap: Dict[str, object], distance_m: Optional[float], moving_sec: Optional[float]
    ) -> Optional[float]:
        speed = safe_float_optional(lap.get("average_speed"))
        if speed is not None:
            return max(speed * 3.6, 0.0)
        if distance_m is None or moving_sec in (None, 0):
            return None
        return max((distance_m / 1000.0) / (moving_sec / 3600.0), 0.0)

    @staticmethod
    def _hr_reserve_ratio(
        avg_hr: Optional[float], hr_profile: Optional[Tuple[float, float]]
    ) -> Optional[float]:
        if hr_profile is None or avg_hr is None:
            return None
        hr_rest, hr_max = hr_profile
        if avg_hr <= hr_rest or hr_max <= hr_rest:
            return None
        ratio = (avg_hr - hr_rest) / (hr_max - hr_rest)
        return max(0.0, min(ratio, 1.5))

    @staticmethod
    def _hr_percent_max(
        avg_hr: Optional[float], hr_profile: Optional[Tuple[float, float]]
    ) -> Optional[float]:
        if hr_profile is None or avg_hr is None:
            return None
        _, hr_max = hr_profile
        if hr_max <= 0:
            return None
        return max((avg_hr / hr_max) * 100.0, 0.0)

    @staticmethod
    def _compute_trimp(
        avg_hr: Optional[float],
        moving_sec: Optional[float],
        hr_profile: Optional[Tuple[float, float]],
    ) -> Optional[float]:
        if hr_profile is None or avg_hr is None or moving_sec in (None, 0):
            return None
        hr_rest, hr_max = hr_profile
        if hr_max <= hr_rest or avg_hr <= hr_rest:
            return None
        hr_ratio = (avg_hr - hr_rest) / (hr_max - hr_rest)
        hr_ratio = max(0.0, min(hr_ratio, 1.2))
        if hr_ratio <= 0:
            return None
        duration_hours = moving_sec / 3600.0
        return duration_hours * hr_ratio * 0.64 * math.exp(1.92 * hr_ratio)

    @staticmethod
    def _classify_lap(
        hr_ratio: Optional[float],
        avg_hr: Optional[float],
        avg_speed_kmh: Optional[float],
    ) -> str:
        if hr_ratio is not None:
            return "Run" if hr_ratio >= 0.5 else "Recovery"
        if avg_hr is not None and avg_hr >= 110:
            return "Run"
        if avg_speed_kmh is not None and avg_speed_kmh >= 8.0:
            return "Run"
        return "Recovery"

    @staticmethod
    def _to_frame(rows: List[Dict[str, object]]) -> pd.DataFrame:
        columns = [
            "lapIndex",
            "split",
            "name",
            "startTime",
            "distanceKm",
            "timeSec",
            "elapsedSec",
            "avgSpeedKmh",
            "ascentM",
            "distanceEqKm",
            "avgHr",
            "maxHr",
            "hrReserveRatio",
            "hrPercentMax",
            "trimp",
            "label",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(rows)
        return df[columns]
