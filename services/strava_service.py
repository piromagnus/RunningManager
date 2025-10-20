"""Strava OAuth2 integration and manual sync service."""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd
import requests

from persistence.csv_storage import CsvStorage
from persistence.repositories import ActivitiesRepo, TokensRepo
from services.lap_metrics_service import LapMetricsService
from services.metrics_service import MetricsComputationService
from utils.config import Config
from utils.crypto import decrypt_text, encrypt_text, get_fernet

LOGGER = logging.getLogger(__name__)

API_BASE = "https://www.strava.com/api/v3"
TOKEN_URL = "https://www.strava.com/oauth/token"
AUTH_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = f"{API_BASE}/oauth/token"
STRAVA_PROVIDER = "strava"
DEFAULT_SCOPE = "activity:read,activity:read_all"
STREAM_KEYS = "time,distance,altitude,heartrate,cadence,latlng,velocity_smooth"


@dataclass
class StravaService:
    storage: CsvStorage
    config: Config
    session: Optional[requests.Session] = None
    now_fn: Callable[[], dt.datetime] = field(
        default_factory=lambda: (lambda: dt.datetime.now(dt.timezone.utc))
    )
    max_retries: int = 3

    def __post_init__(self) -> None:
        self.session = self.session or requests.Session()
        self.activities = ActivitiesRepo(self.storage)
        self.tokens = TokensRepo(self.storage)
        self.lap_metrics = LapMetricsService(self.storage, self.config)
        self._fernet = None

    # --- Public API -------------------------------------------------
    def authorization_url(self, state: str, approval_prompt: str = "auto") -> str:
        if not self.config.strava_client_id or not self.config.strava_redirect_uri:
            raise RuntimeError("STRAVA_CLIENT_ID and STRAVA_REDIRECT_URI must be configured")
        params = {
            "client_id": self.config.strava_client_id,
            "response_type": "code",
            "redirect_uri": self.config.strava_redirect_uri,
            "scope": DEFAULT_SCOPE,
            "state": state,
            "approval_prompt": approval_prompt,
        }
        query = "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
        return f"{AUTH_URL}?{query}"

    def exchange_code(self, athlete_id: str, code: str) -> Dict[str, Any]:
        payload = {
            "client_id": self._require_client_id(),
            "client_secret": self._require_client_secret(),
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self._require_redirect_uri(),
        }
        response = self._request_json("POST", TOKEN_URL, data=payload)
        tokens = self._extract_tokens(response)
        self._save_tokens(athlete_id, tokens)
        print(f"[Strava] Token exchange succeeded for athlete {athlete_id}")
        return {
            "access_token": tokens["access_token"],
            "expires_at": tokens["expires_at"],
        }

    def sync_last_n_days(self, athlete_id: str, days: int) -> List[str]:
        if days <= 0:
            raise ValueError("days must be positive")
        tokens = self._ensure_access_token(athlete_id)
        now = self.now_fn()
        after_ts = int((now - dt.timedelta(days=days)).timestamp())
        existing_ids = self._existing_activity_ids()
        existing_raw_ids = self._existing_raw_ids()
        imported: List[str] = []

        for summary in self._iter_recent_activities(tokens["access_token"], after_ts):
            activity_id = str(summary.get("id"))
            if (
                not activity_id
                or activity_id in existing_ids
                or activity_id in existing_raw_ids
            ):
                continue
            detail = self._get_activity(tokens["access_token"], activity_id)
            if not detail:
                continue
            raw_path = self._save_raw_activity(detail)
            self.lap_metrics.compute_and_store(athlete_id, detail)
            streams = self._get_streams(tokens["access_token"], activity_id)
            has_timeseries = self._save_timeseries(activity_id, detail, streams)
            row = self._map_activity_row(
                detail=detail,
                athlete_id=athlete_id,
                has_timeseries=has_timeseries,
                raw_path=raw_path,
            )
            self.activities.create(row)
            existing_ids.add(activity_id)
            existing_raw_ids.add(activity_id)
            imported.append(activity_id)
        if imported:
            MetricsComputationService(self.storage).recompute_for_activities(imported)
        return imported

    def sync_last_14_days(self, athlete_id: str) -> List[str]:
        return self.sync_last_n_days(athlete_id, 14)

    def rebuild_from_cache(self, athlete_id: str) -> List[str]:
        raw_dir = self.config.raw_strava_dir
        headers = self.activities.headers
        rows: List[Dict[str, Any]] = []
        if raw_dir.exists():
            for path in sorted(raw_dir.glob("*.json")):
                try:
                    with path.open("r", encoding="utf-8") as fh:
                        detail = json.load(fh)
                except Exception:
                    continue
                activity_id = str(detail.get("id") or path.stem)
                has_timeseries = (self.config.timeseries_dir / f"{activity_id}.csv").exists()
                self.lap_metrics.compute_and_store(athlete_id, detail)
                row = self._map_activity_row(detail, athlete_id, has_timeseries, path)
                rows.append(row)
        df = pd.DataFrame(rows, columns=headers if rows else None)
        if df.empty:
            df = pd.DataFrame(columns=headers)
        else:
            df = df[headers]
        self.activities.storage.write_csv(self.activities.file_name, df)
        return [str(r.get("activityId")) for r in rows if r.get("activityId")]

    # --- OAuth helpers ----------------------------------------------
    def _ensure_access_token(self, athlete_id: str) -> Dict[str, Any]:
        tokens = self._load_tokens(athlete_id)
        now_ts = int(self.now_fn().timestamp())
        refresh_cutoff = tokens["expires_at"] - 300
        if now_ts >= refresh_cutoff:
            LOGGER.info("Refreshing Strava token for athlete %s", athlete_id)
            refreshed = self._refresh_token(tokens["refresh_token"])
            self._save_tokens(athlete_id, refreshed)
            tokens = refreshed
        return tokens

    def _refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        payload = {
            "client_id": self._require_client_id(),
            "client_secret": self._require_client_secret(),
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        response = self._request_json("POST", TOKEN_URL, data=payload)
        tokens = self._extract_tokens(response)
        if "refresh_token" not in tokens:
            tokens["refresh_token"] = refresh_token
        print("[Strava] Token refreshed")
        return tokens

    # --- API calls --------------------------------------------------
    def _iter_recent_activities(self, access_token: str, after_ts: int) -> Iterable[Dict[str, Any]]:
        page = 1
        while True:
            params = {
                "after": after_ts,
                "per_page": 200,
                "page": page,
            }
            data = self._request_json(
                "GET",
                f"{API_BASE}/athlete/activities",
                headers=self._auth_headers(access_token),
                params=params,
            )
            if not data:
                break
            if not isinstance(data, list):
                raise RuntimeError("Unexpected response from Strava activities endpoint")
            for item in data:
                yield item
            if len(data) < 200:
                break
            page += 1

    def _get_activity(self, access_token: str, activity_id: str) -> Dict[str, Any]:
        return self._request_json(
            "GET",
            f"{API_BASE}/activities/{activity_id}",
            headers=self._auth_headers(access_token),
        )

    def _get_streams(self, access_token: str, activity_id: str) -> Dict[str, Any]:
        params = {"keys": STREAM_KEYS, "key_by_type": "true"}
        return self._request_json(
            "GET",
            f"{API_BASE}/activities/{activity_id}/streams",
            headers=self._auth_headers(access_token),
            params=params,
        )

    # --- Persistence helpers ---------------------------------------
    def _existing_activity_ids(self) -> set[str]:
        df = self.activities.list()
        if df.empty:
            return set()
        return set(df["activityId"].astype(str))

    def _existing_raw_ids(self) -> set[str]:
        raw_dir = self.config.raw_strava_dir
        if not raw_dir.exists():
            return set()
        return {path.stem for path in raw_dir.glob("*.json")}

    def _save_raw_activity(self, detail: Dict[str, Any]) -> Path:
        activity_id = str(detail.get("id"))
        path = self.config.raw_strava_dir / f"{activity_id}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(detail, fh, ensure_ascii=False)
        return path

    def _save_timeseries(
        self,
        activity_id: str,
        detail: Dict[str, Any],
        streams: Dict[str, Any],
    ) -> bool:
        time_stream = self._stream_data(streams, "time")
        if not time_stream:
            return False

        start_iso = detail.get("start_date") or detail.get("start_date_local")
        try:
            base_dt = dt.datetime.fromisoformat(str(start_iso).replace("Z", "+00:00"))
        except Exception:
            base_dt = None

        hr_stream = self._stream_data(streams, "heartrate")
        vel_stream = self._stream_data(streams, "velocity_smooth")
        alt_stream = self._stream_data(streams, "altitude")
        cad_stream = self._stream_data(streams, "cadence")
        latlng_stream = self._stream_data(streams, "latlng")

        records: List[Dict[str, Any]] = []
        for idx, offset in enumerate(time_stream):
            ts_value: Optional[str]
            if base_dt is not None:
                ts_value = (base_dt + dt.timedelta(seconds=float(offset))).isoformat()
            else:
                ts_value = str(offset)
            lat: Optional[float] = None
            lon: Optional[float] = None
            if latlng_stream and idx < len(latlng_stream):
                pair = latlng_stream[idx]
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    lat = pair[0]
                    lon = pair[1]
            records.append(
                {
                    "timestamp": ts_value,
                    "hr": self._value_at(hr_stream, idx),
                    "paceKmh": self._pace_from_velocity(self._value_at(vel_stream, idx)),
                    "elevationM": self._value_at(alt_stream, idx),
                    "cadence": self._value_at(cad_stream, idx),
                    "lat": lat,
                    "lon": lon,
                }
            )
        if not records:
            return False
        df = pd.DataFrame(records)
        path = self.config.timeseries_dir / f"{activity_id}.csv"
        df.to_csv(path, index=False)
        return True

    # --- Mapping helpers -------------------------------------------
    def _map_activity_row(
        self,
        detail: Dict[str, Any],
        athlete_id: str,
        has_timeseries: bool,
        raw_path: Path,
    ) -> Dict[str, Any]:
        distance_m = detail.get("distance")
        ascent_m = detail.get("total_elevation_gain")
        row = {
            "activityId": str(detail.get("id")),
            "athleteId": athlete_id,
            "source": STRAVA_PROVIDER,
            "startTime": detail.get("start_date_local") or detail.get("start_date"),
            "distanceKm": ((float(distance_m) / 1000.0) if distance_m else None),
            "elapsedSec": detail.get("elapsed_time"),
            "movingSec": detail.get("moving_time"),
            "ascentM": float(ascent_m) if ascent_m is not None else None,
            "avgHr": detail.get("average_heartrate"),
            "maxHr": detail.get("max_heartrate"),
            "hasTimeseries": has_timeseries,
            "polyline": self._polyline(detail),
            "rawJsonPath": self._relative_to_data_dir(raw_path),
        }
        return row

    # --- Token helpers ---------------------------------------------
    def _require_client_id(self) -> str:
        if not self.config.strava_client_id:
            raise RuntimeError("STRAVA_CLIENT_ID is not configured")
        return self.config.strava_client_id

    def _require_client_secret(self) -> str:
        if not self.config.strava_client_secret:
            raise RuntimeError("STRAVA_CLIENT_SECRET is not configured")
        return self.config.strava_client_secret

    def _require_redirect_uri(self) -> str:
        if not self.config.strava_redirect_uri:
            raise RuntimeError("STRAVA_REDIRECT_URI is not configured")
        return self.config.strava_redirect_uri

    def _get_fernet(self):  # type: ignore[return-any]
        if self._fernet is None:
            self._fernet = get_fernet(self.config.encryption_key)
        return self._fernet

    def _save_tokens(self, athlete_id: str, tokens: Dict[str, Any]) -> None:
        fernet = self._get_fernet()
        row = {
            "athleteId": athlete_id,
            "provider": STRAVA_PROVIDER,
            "accessTokenEnc": encrypt_text(fernet, tokens["access_token"]),
            "refreshTokenEnc": encrypt_text(fernet, tokens["refresh_token"]),
            "expiresAt": int(tokens["expires_at"]),
        }
        self.tokens.storage.upsert(
            self.tokens.file_name,
            ["athleteId", "provider"],
            row,
        )

    def _load_tokens(self, athlete_id: str) -> Dict[str, Any]:
        df = self.tokens.list(athleteId=athlete_id)
        if df.empty:
            raise RuntimeError("No Strava tokens found for athlete")
        df = df[df["provider"] == STRAVA_PROVIDER]
        if df.empty:
            raise RuntimeError("No Strava tokens found for athlete")
        row = df.iloc[0]
        fernet = self._get_fernet()
        access = decrypt_text(fernet, str(row["accessTokenEnc"]))
        refresh = decrypt_text(fernet, str(row["refreshTokenEnc"]))
        expires = int(row.get("expiresAt") or 0)
        if not expires:
            raise RuntimeError("Stored Strava token is missing expiry")
        return {
            "access_token": access,
            "refresh_token": refresh,
            "expires_at": expires,
        }

    def _extract_tokens(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        access = payload.get("access_token")
        refresh = payload.get("refresh_token")
        expires = payload.get("expires_at")
        if not access or not refresh or not expires:
            raise RuntimeError("Strava token response missing required fields")
        return {
            "access_token": str(access),
            "refresh_token": str(refresh),
            "expires_at": int(expires),
        }

    # --- Generic helpers -------------------------------------------
    def _request_json(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        for attempt in range(self.max_retries):
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_payload,
                timeout=30,
            )
            if response.status_code == 429 and attempt < self.max_retries - 1:
                retry_after = int(response.headers.get("Retry-After", "1"))
                time.sleep(min(retry_after, 60))
                continue
            if 200 <= response.status_code < 300:
                if response.content and response.content.strip():
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise RuntimeError("Strava API returned non-JSON response") from exc
                return {}
            if response.status_code == 429:
                raise RuntimeError("Strava API rate limit exceeded")
            raise RuntimeError(f"Strava API error {response.status_code}: {response.text}")
        raise RuntimeError("Strava API rate limited; retries exhausted")

    def _auth_headers(self, access_token: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {access_token}"}

    @staticmethod
    def _stream_data(streams: Dict[str, Any], key: str) -> List[Any]:
        entry = streams.get(key)
        if isinstance(entry, dict):
            data = entry.get("data")
        else:
            data = entry
        return data or []

    @staticmethod
    def _value_at(stream: List[Any], index: int) -> Optional[Any]:
        if not stream or index >= len(stream):
            return None
        return stream[index]

    @staticmethod
    def _pace_from_velocity(velocity: Optional[Any]) -> Optional[float]:
        try:
            if velocity is None:
                return None
            return float(velocity) * 3.6
        except Exception:
            return None

    @staticmethod
    def _polyline(detail: Dict[str, Any]) -> Optional[str]:
        mapping = detail.get("map")
        if isinstance(mapping, dict):
            poly = mapping.get("summary_polyline")
            if poly:
                return str(poly)
        return None

    def _relative_to_data_dir(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.config.data_dir))
        except ValueError:
            return str(path)
