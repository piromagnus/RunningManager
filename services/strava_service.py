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
import portalocker
from urllib.parse import urlparse
import requests

from persistence.csv_storage import CsvStorage
from persistence.repositories import ActivitiesRepo, TokensRepo
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

# Strava documented rate limits
RATE_LIMIT_PER_15_MIN = 100
DAILY_LIMIT = 1000


@dataclass
class StravaService:
    storage: CsvStorage
    config: Config
    session: Optional[requests.Session] = None
    now_fn: Callable[[], dt.datetime] = field(
        default_factory=lambda: (lambda: dt.datetime.now(dt.timezone.utc))
    )
    max_retries: int = 3
    # Exposes a summary of the last sync for UI/UX (non-persistent)
    last_sync_stats: Dict[str, Any] = field(default_factory=dict)
    # Last known rate status parsed from headers/log (non-persistent)
    last_rate_status: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session = self.session or requests.Session()
        self.activities = ActivitiesRepo(self.storage)
        self.tokens = TokensRepo(self.storage)
        self._fernet = None
        self._rate_log_path = self.storage.base_dir / "strava_api_log.json"

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

    def preview_sync_last_n_days(self, athlete_id: str, days: int) -> Dict[str, Any]:
        """Preview sync cost and counts without downloading details/streams.

        Returns a dict containing:
        - total_in_range: total activities returned by the listing endpoint
        - already_cached: number with existing raw JSON
        - missing_raw: number lacking raw JSON (would require download)
        - est_detail_requests: ~ missing_raw * 2 (detail + streams)
        - est_page_requests: ~ ceil(total_in_range / 200)
        - est_total_requests: est_detail_requests + est_page_requests
        - est_15min_windows_needed: ceil(est_total_requests / 100)
        - est_additional_waits: max(0, est_15min_windows_needed - 1), capped at 9
        - est_hits_daily_limit: True if est_total_requests > 1000
        """
        if days <= 0:
            raise ValueError("days must be positive")
        tokens = self._ensure_access_token(athlete_id)
        now = self.now_fn()
        after_ts = int((now - dt.timedelta(days=days)).timestamp())
        existing_raw_ids = self._existing_raw_ids()

        total = 0
        missing = 0
        # Iterate all summaries once to count
        for summary in self._iter_recent_activities(tokens["access_token"], after_ts):
            total += 1
            sid = str(summary.get("id") or "")
            if sid and sid not in existing_raw_ids:
                missing += 1

        est_page_requests = (total + 199) // 200
        est_detail_requests = missing * 2
        est_total_requests = est_detail_requests + est_page_requests
        windows_needed = (est_total_requests + RATE_LIMIT_PER_15_MIN - 1) // RATE_LIMIT_PER_15_MIN
        additional_waits = max(0, windows_needed - 1)
        additional_waits = min(additional_waits, 9)  # soft cap (per-day window)
        hits_daily = est_total_requests > DAILY_LIMIT

        preview = {
            "total_in_range": total,
            "already_cached": total - missing,
            "missing_raw": missing,
            "est_detail_requests": est_detail_requests,
            "est_page_requests": est_page_requests,
            "est_total_requests": est_total_requests,
            "est_15min_windows_needed": windows_needed,
            "est_additional_waits": additional_waits,
            "est_hits_daily_limit": hits_daily,
        }
        # Keep for UI usage
        self.last_sync_stats = {**self.last_sync_stats, "preview": preview}
        return preview

    def sync_last_n_days(self, athlete_id: str, days: int) -> List[str]:
        """Sync activities over the last ``days`` and update metrics.

        Behavior:
        - Always list recent activities from Strava (after ``now - days``).
        - For each activity in that window:
          - If raw JSON is missing, fetch detail + streams, persist both, and create
            the activity row in ``activities.csv``.
          - If raw JSON already exists, do not call the network again (cache hit),
            but ensure there's an entry in ``activities.csv`` built from the cache
            if it is currently missing.
        - After creating any new ``activities.csv`` rows, recompute metrics only
          for the affected athlete(s) via ``MetricsComputationService`` (which in
          turn updates activity, daily and weekly metrics incrementally).

        Returns the list of activity IDs newly downloaded from Strava during this
        call (i.e., cache misses). Activities created from cache are not counted
        in the returned list to keep backward-compatible semantics.
        """
        if days <= 0:
            raise ValueError("days must be positive")

        tokens = self._ensure_access_token(athlete_id)
        now = self.now_fn()
        after_ts = int((now - dt.timedelta(days=days)).timestamp())

        existing_ids = self._existing_activity_ids()
        existing_raw_ids = self._existing_raw_ids()

        imported_from_api: List[str] = []  # strictly newly downloaded raw
        created_rows: List[str] = []  # activity rows newly created (from API or cache)

        for summary in self._iter_recent_activities(tokens["access_token"], after_ts):
            activity_id = str(summary.get("id"))
            if not activity_id:
                continue

            has_raw = activity_id in existing_raw_ids
            raw_path = self.config.raw_strava_dir / f"{activity_id}.json"
            has_row = activity_id in existing_ids

            detail: Dict[str, Any] | None = None
            has_timeseries = (self.config.timeseries_dir / f"{activity_id}.csv").exists()

            if not has_raw:
                # Cache miss: fetch detail + streams and persist both
                detail = self._get_activity(tokens["access_token"], activity_id)
                if not detail:
                    continue
                raw_path = self._save_raw_activity(detail)
                streams = self._get_streams(tokens["access_token"], activity_id)
                has_timeseries = self._save_timeseries(activity_id, detail, streams)
                existing_raw_ids.add(activity_id)
                imported_from_api.append(activity_id)
            else:
                # Cache hit: reuse cached raw to build activities row if needed
                if not has_row:
                    try:
                        with raw_path.open("r", encoding="utf-8") as fh:
                            detail = json.load(fh)
                    except Exception:
                        detail = None

            # Ensure we have an activities.csv row if it does not exist yet
            if not has_row and detail is not None:
                row = self._map_activity_row(
                    detail=detail,
                    athlete_id=athlete_id,
                    has_timeseries=has_timeseries,
                    raw_path=raw_path,
                )
                self.activities.create(row)
                existing_ids.add(activity_id)
                created_rows.append(activity_id)

        # Recompute metrics for newly added activities (from API or cache)
        if created_rows:
            MetricsComputationService(self.storage).recompute_for_activities(created_rows)

        # Record stats for UI
        created_from_cache = [aid for aid in created_rows if aid not in imported_from_api]
        self.last_sync_stats = {
            "days": int(days),
            "downloaded_count": len(imported_from_api),
            "created_rows_count": len(created_rows),
            "created_from_cache_count": len(created_from_cache),
            "downloaded_ids": list(imported_from_api),
            "created_from_cache_ids": created_from_cache,
        }

        return imported_from_api

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
        sport_type = str(detail.get("sport_type") or detail.get("type") or "")
        title = detail.get("name")
        row = {
            "activityId": str(detail.get("id")),
            "athleteId": athlete_id,
            "source": STRAVA_PROVIDER,
            "sportType": sport_type,
            "name": str(title) if title is not None else "",
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
            # Log and parse rate headers best-effort
            try:
                self._log_api_call(method, url, response.status_code, response.headers)
                self.last_rate_status = self._rate_status_from_headers(response.headers)
            except Exception:
                pass
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

    # --- Rate-limit logging and status --------------------------------
    def _log_api_call(self, method: str, url: str, status: int, headers: Dict[str, str]) -> None:
        entry = {
            "ts": self.now_fn().isoformat(),
            "method": method,
            "endpoint": self._endpoint_for_log(url),
            "status": int(status),
        }
        usage = headers.get("X-RateLimit-Usage") or headers.get("x-ratelimit-usage")
        limit = headers.get("X-RateLimit-Limit") or headers.get("x-ratelimit-limit")
        if usage:
            try:
                parts = [int(p.strip()) for p in str(usage).split(",") if p.strip()]
                if len(parts) >= 2:
                    entry["usage_short"] = parts[0]
                    entry["usage_daily"] = parts[1]
            except Exception:
                pass
        if limit:
            try:
                parts = [int(p.strip()) for p in str(limit).split(",") if p.strip()]
                if len(parts) >= 2:
                    entry["limit_short"] = parts[0]
                    entry["limit_daily"] = parts[1]
            except Exception:
                pass
        # Append JSON line under lock; keep recent ~500 entries to avoid growth
        path = self._rate_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(path), timeout=10, flags=portalocker.LOCK_EX):
            entries: list[dict] = []
            if path.exists() and path.stat().st_size > 0:
                try:
                    entries = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(entries, list):
                        entries = []
                except Exception:
                    entries = []
            entries.append(entry)
            if len(entries) > 500:
                entries = entries[-500:]
            path.write_text(json.dumps(entries), encoding="utf-8")

    @staticmethod
    def _endpoint_for_log(url: str) -> str:
        try:
            parsed = urlparse(url)
            # keep only path, mask tokens in path (none expected), drop query entirely
            return parsed.path
        except Exception:
            return url

    def _rate_status_from_headers(self, headers: Dict[str, str]) -> Dict[str, Any]:
        now = self.now_fn()
        limit = headers.get("X-RateLimit-Limit") or headers.get("x-ratelimit-limit")
        usage = headers.get("X-RateLimit-Usage") or headers.get("x-ratelimit-usage")
        short_limit = DAILY_LIMIT
        daily_limit = DAILY_LIMIT
        short_used = None
        daily_used = None
        try:
            if limit:
                parts = [int(p.strip()) for p in str(limit).split(",") if p.strip()]
                if len(parts) >= 2:
                    short_limit, daily_limit = parts[0], parts[1]
            if usage:
                parts = [int(p.strip()) for p in str(usage).split(",") if p.strip()]
                if len(parts) >= 2:
                    short_used, daily_used = parts[0], parts[1]
        except Exception:
            pass
        # Compute wait until next 15-min window (approx) if capped
        def seconds_to_next_quarter(dt_now: dt.datetime) -> int:
            # next boundary at minute in {0,15,30,45}
            minutes = (dt_now.minute // 15 + 1) * 15
            hour = dt_now.hour
            day = dt_now.date()
            if minutes >= 60:
                minutes -= 60
                hour = (hour + 1) % 24
                if hour == 0:
                    day = (dt_now + dt.timedelta(days=1)).date()
            next_dt = dt.datetime(dt_now.year, dt_now.month, dt_now.day, hour, minutes, 0, tzinfo=dt_now.tzinfo)
            return max(0, int((next_dt - dt_now).total_seconds()))

        wait = 0
        if short_used is not None and short_used >= short_limit:
            wait = seconds_to_next_quarter(now)
        return {
            "short_used": short_used,
            "short_limit": short_limit,
            "daily_used": daily_used,
            "daily_limit": daily_limit,
            "wait_seconds": wait,
            "as_of": now.isoformat(),
        }

    def get_rate_status(self) -> Dict[str, Any]:
        """Return current rate status using last headers or from local log as fallback."""
        if self.last_rate_status:
            return self.last_rate_status
        # Fallback: compute usage counts from log entries
        path = self._rate_log_path
        now = self.now_fn()
        short_used = 0
        daily_used = 0
        short_limit = RATE_LIMIT_PER_15_MIN
        daily_limit = DAILY_LIMIT
        if path.exists() and path.stat().st_size > 0:
            try:
                entries = json.loads(path.read_text(encoding="utf-8"))
                # count entries in last 15 minutes and day
                for e in reversed(entries):
                    ts = e.get("ts")
                    if not ts:
                        continue
                    try:
                        ts_dt = dt.datetime.fromisoformat(str(ts))
                        if ts_dt.tzinfo is None:
                            ts_dt = ts_dt.replace(tzinfo=dt.timezone.utc)
                    except Exception:
                        continue
                    if (now - ts_dt) <= dt.timedelta(minutes=15):
                        short_used += 1
                    if (now - ts_dt) <= dt.timedelta(days=1):
                        daily_used += 1
                    # Load limits if present in entry
                    if "limit_short" in e:
                        try:
                            short_limit = int(e["limit_short"])
                        except Exception:
                            pass
                    if "limit_daily" in e:
                        try:
                            daily_limit = int(e["limit_daily"])
                        except Exception:
                            pass
            except Exception:
                pass
        # approximate wait
        wait = 0
        if short_used >= short_limit:
            # reuse same helper logic
            wait = self._rate_status_from_headers({}).get("wait_seconds", 0)  # type: ignore[arg-type]
        status = {
            "short_used": short_used,
            "short_limit": short_limit,
            "daily_used": daily_used,
            "daily_limit": daily_limit,
            "wait_seconds": wait,
            "as_of": now.isoformat(),
        }
        self.last_rate_status = status
        return status

    def get_rate_log(self, limit: int = 5) -> List[Dict[str, Any]]:
        path = self._rate_log_path
        if not path.exists() or path.stat().st_size == 0:
            return []
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                return []
            return list(reversed(entries[-limit:]))
        except Exception:
            return []

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
