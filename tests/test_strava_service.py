"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
from cryptography.fernet import Fernet

from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo
from services.strava_service import API_BASE, TOKEN_URL, StravaService
from utils.config import Config
from utils.crypto import decrypt_text, get_fernet


class FakeResponse:
    def __init__(
        self, status_code: int, payload: Any = None, headers: Optional[Dict[str, str]] = None
    ):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        text = json.dumps(self._payload)
        self._text = text
        self.content = text.encode()

    def json(self) -> Any:
        return self._payload

    @property
    def text(self) -> str:
        return self._text


class FakeSession:
    def __init__(self, responses: List[Tuple[str, str, FakeResponse]]):
        self._queue = responses
        self.calls: List[Tuple[str, str, Dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if not self._queue:
            raise AssertionError(f"Unexpected request {method} {url}")
        expected_method, expected_url, response = self._queue.pop(0)
        assert expected_method == method
        assert expected_url == url
        return response

    @property
    def empty(self) -> bool:
        return not self._queue


@pytest.fixture
def config(tmp_path: Path) -> Config:
    key = Fernet.generate_key().decode()
    timeseries_dir = tmp_path / "timeseries"
    raw_dir = tmp_path / "raw" / "strava"
    laps_dir = tmp_path / "laps"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    laps_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        strava_client_id="1234",
        strava_client_secret="top-secret",
        strava_redirect_uri="http://localhost/callback",
        data_dir=tmp_path,
        encryption_key=key,
        timeseries_dir=timeseries_dir,
        raw_strava_dir=raw_dir,
        laps_dir=laps_dir,
        mapbox_token=None,
    )


@pytest.fixture
def storage(config: Config) -> CsvStorage:
    return CsvStorage(base_dir=config.data_dir)


def test_authorization_url_contains_expected_query(storage: CsvStorage, config: Config) -> None:
    service = StravaService(storage=storage, config=config, session=FakeSession([]))
    url = service.authorization_url(state="abc123")
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "www.strava.com"
    qs = parse_qs(parsed.query)
    assert qs["client_id"] == [config.strava_client_id]
    assert qs["redirect_uri"] == [config.strava_redirect_uri]
    assert qs["scope"] == ["activity:read,activity:read_all"]
    assert qs["state"] == ["abc123"]


def test_exchange_code_encrypts_tokens(storage: CsvStorage, config: Config) -> None:
    responses = [
        (
            "POST",
            TOKEN_URL,
            FakeResponse(
                200,
                {
                    "access_token": "access-1",
                    "refresh_token": "refresh-1",
                    "expires_at": int(dt.datetime.now(dt.timezone.utc).timestamp()) + 3600,
                },
            ),
        )
    ]
    session = FakeSession(responses)
    service = StravaService(storage=storage, config=config, session=session)

    service.exchange_code("athlete-1", "code-xyz")
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == TOKEN_URL
    assert kwargs["data"]["redirect_uri"] == config.strava_redirect_uri

    df = service.tokens.list(athleteId="athlete-1")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["provider"] == "strava"
    assert row["accessTokenEnc"] != "access-1"
    fernet = get_fernet(config.encryption_key)
    assert decrypt_text(fernet, row["accessTokenEnc"]) == "access-1"
    assert decrypt_text(fernet, row["refreshTokenEnc"]) == "refresh-1"
    assert session.empty


def test_sync_last_14_days_imports_activity(storage: CsvStorage, config: Config) -> None:
    now = dt.datetime(2024, 1, 20, tzinfo=dt.timezone.utc)
    expires_at = int((now + dt.timedelta(hours=1)).timestamp())
    activity_id = "987654"
    AthletesRepo(storage).create(
        {
            "athleteId": "athlete-1",
            "coachId": "coach-1",
            "name": "Test",
            "hrRest": 50,
            "hrMax": 190,
        }
    )
    responses = [
        (
            "POST",
            TOKEN_URL,
            FakeResponse(
                200,
                {
                    "access_token": "token-abc",
                    "refresh_token": "refresh-abc",
                    "expires_at": expires_at,
                },
            ),
        ),
        (
            "GET",
            f"{API_BASE}/athlete/activities",
            FakeResponse(
                200,
                [
                    {
                        "id": int(activity_id),
                        "start_date": "2024-01-19T07:00:00Z",
                        "start_date_local": "2024-01-19T08:00:00+01:00",
                    }
                ],
            ),
        ),
        (
            "GET",
            f"{API_BASE}/activities/{activity_id}",
            FakeResponse(
                200,
                {
                    "id": int(activity_id),
                    "distance": 10000.0,
                    "elapsed_time": 4000,
                    "moving_time": 3900,
                    "total_elevation_gain": 250.5,
                    "average_heartrate": 142.2,
                    "max_heartrate": 165.0,
                    "start_date": "2024-01-19T07:00:00Z",
                    "start_date_local": "2024-01-19T08:00:00+01:00",
                    "map": {"summary_polyline": "abcd"},
                    "laps": [
                        {
                            "lap_index": 1,
                            "elapsed_time": 320,
                            "moving_time": 310,
                            "distance": 1500.0,
                            "total_elevation_gain": 30.0,
                            "average_heartrate": 150.0,
                            "max_heartrate": 165.0,
                            "average_speed": 4.5,
                        },
                        {
                            "lap_index": 2,
                            "elapsed_time": 180,
                            "moving_time": 175,
                            "distance": 400.0,
                            "total_elevation_gain": 5.0,
                            "average_heartrate": 110.0,
                            "max_heartrate": 125.0,
                            "average_speed": 2.0,
                        },
                    ],
                },
            ),
        ),
        (
            "GET",
            f"{API_BASE}/activities/{activity_id}/streams",
            FakeResponse(
                200,
                {
                    "time": {"data": [0, 10, 20]},
                    "heartrate": {"data": [120, 130, 140]},
                    "velocity_smooth": {"data": [3.0, 3.5, 4.0]},
                    "altitude": {"data": [200.0, 201.5, 203.0]},
                    "cadence": {"data": [80, 82, 84]},
                    "latlng": {"data": [[48.1, 2.3], [48.1001, 2.3001], [48.1002, 2.3002]]},
                },
            ),
        ),
    ]
    session = FakeSession(responses)
    service = StravaService(
        storage=storage,
        config=config,
        session=session,
        now_fn=lambda: now,
    )

    service.exchange_code("athlete-1", "auth-code")
    _, _, first_kwargs = session.calls[0]
    assert first_kwargs["data"]["redirect_uri"] == config.strava_redirect_uri
    imported = service.sync_last_14_days("athlete-1")
    assert imported == [activity_id]

    activities_path = storage.base_dir / "activities.csv"
    df = pd.read_csv(activities_path)
    assert df.shape[0] == 1
    row = df.iloc[0]
    assert str(row["activityId"]) == activity_id
    assert bool(row["hasTimeseries"]) is True
    assert row["rawJsonPath"] == f"raw/strava/{activity_id}.json"

    raw_path = config.raw_strava_dir / f"{activity_id}.json"
    assert raw_path.exists()
    with raw_path.open() as fh:
        data = json.load(fh)
    assert data["id"] == int(activity_id)

    ts_path = config.timeseries_dir / f"{activity_id}.csv"
    assert ts_path.exists()
    ts_df = pd.read_csv(ts_path)
    assert list(ts_df.columns) == [
        "timestamp",
        "hr",
        "paceKmh",
        "elevationM",
        "cadence",
        "lat",
        "lon",
    ]
    assert pytest.approx(ts_df.iloc[1]["paceKmh"], rel=1e-3) == 12.6

    laps_path = config.laps_dir / f"{activity_id}.csv"
    assert laps_path.exists()
    laps_df = pd.read_csv(laps_path)
    assert laps_df.shape[0] == 2
    assert set(laps_df["label"]) == {"Run", "Recovery"}

    assert session.empty

    # Run sync again with cached activity to ensure no duplicate writes
    repeat_session = FakeSession(
        [
            (
                "GET",
                f"{API_BASE}/athlete/activities",
                FakeResponse(
                    200,
                    [
                        {
                            "id": int(activity_id),
                            "start_date": "2024-01-19T07:00:00Z",
                        }
                    ],
                ),
            )
        ]
    )
    service.session = repeat_session
    again = service.sync_last_14_days("athlete-1")
    assert again == []
    assert repeat_session.empty


def test_sync_skips_cached_raw(storage: CsvStorage, config: Config) -> None:
    now = dt.datetime(2024, 2, 10, tzinfo=dt.timezone.utc)
    expires_at = int((now + dt.timedelta(hours=1)).timestamp())
    activity_id = "4321"

    exchange_session = FakeSession(
        [
            (
                "POST",
                TOKEN_URL,
                FakeResponse(
                    200,
                    {
                        "access_token": "token-xyz",
                        "refresh_token": "refresh-xyz",
                        "expires_at": expires_at,
                    },
                ),
            )
        ]
    )
    service = StravaService(
        storage=storage,
        config=config,
        session=exchange_session,
        now_fn=lambda: now,
    )

    service.exchange_code("athlete-1", "auth")
    assert exchange_session.empty

    raw_path = config.raw_strava_dir / f"{activity_id}.json"
    raw_path.write_text("{}", encoding="utf-8")

    fetch_session = FakeSession(
        [
            (
                "GET",
                f"{API_BASE}/athlete/activities",
                FakeResponse(200, [{"id": int(activity_id)}]),
            )
        ]
    )
    service.session = fetch_session
    imported = service.sync_last_n_days("athlete-1", 7)
    assert imported == []
    assert fetch_session.empty


def test_rebuild_from_cache(storage: CsvStorage, config: Config) -> None:
    service = StravaService(storage=storage, config=config, session=FakeSession([]))
    detail = {
        "id": 1010,
        "start_date_local": "2024-03-10T08:00:00",
        "distance": 15000.0,
        "elapsed_time": 4000,
        "moving_time": 3900,
        "total_elevation_gain": 320.0,
        "average_heartrate": 140.0,
        "max_heartrate": 165.0,
        "map": {"summary_polyline": "abcd"},
    }
    raw_path = config.raw_strava_dir / "1010.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(detail), encoding="utf-8")
    ts_path = config.timeseries_dir / "1010.csv"
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    ts_path.write_text("timestamp\n", encoding="utf-8")

    rebuilt = service.rebuild_from_cache("athlete-1")
    assert rebuilt == ["1010"]

    df = storage.read_csv("activities.csv")
    assert df.shape[0] == 1
    row = df.iloc[0]
    assert str(row["activityId"]) == "1010"
    assert pytest.approx(float(row["distanceKm"]), rel=1e-5) == 15.0
    assert bool(row["hasTimeseries"]) is True
