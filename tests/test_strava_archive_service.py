"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Tests for Strava GDPR archive import and sync merge enrichment.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest
from cryptography.fernet import Fernet

from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo
from services.strava_archive_service import StravaArchiveService
from services.strava_service import API_BASE, StravaService
from utils.config import Config
from utils.strava_merge import merge_fill_empty, raw_needs_enrichment
from utils.strava_track_parser import parse_strava_track_file


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
    metrics_ts_dir = tmp_path / "metrics_ts"
    speed_profile_dir = tmp_path / "speed_profil"
    for path in (timeseries_dir, raw_dir, laps_dir, metrics_ts_dir, speed_profile_dir):
        path.mkdir(parents=True, exist_ok=True)
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
        metrics_ts_dir=metrics_ts_dir,
        speed_profile_dir=speed_profile_dir,
        n_cluster=5,
    )


@pytest.fixture
def storage(config: Config) -> CsvStorage:
    return CsvStorage(base_dir=config.data_dir)


def _sample_gpx(*, points: int = 5, with_hr: bool = True) -> bytes:
    trkpts = []
    for i in range(points):
        lat = 45.0 + i * 0.001
        lon = 5.0 + i * 0.001
        ele = 100 + i
        hr = 140 + i
        ts = f"2024-01-19T08:00:{i:02d}Z"
        hr_xml = (
            f"<extensions><gpxtpx:TrackPointExtension xmlns:gpxtpx="
            f'"http://www.garmin.com/xmlschemas/TrackPointExtension/v1">'
            f"<gpxtpx:hr>{hr}</gpxtpx:hr></gpxtpx:TrackPointExtension></extensions>"
            if with_hr
            else ""
        )
        trkpts.append(
            f'<trkpt lat="{lat}" lon="{lon}"><ele>{ele}</ele>'
            f"<time>{ts}</time>{hr_xml}</trkpt>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
        f"<trk><trkseg>{''.join(trkpts)}</trkseg></trk></gpx>"
    ).encode()


def _build_archive_zip(
    *,
    activity_id: str = "11277623453",
    filename: str = "activities/11277623453.gpx",
    include_track: bool = True,
) -> bytes:
    # Quote date fields that contain commas (official Strava export quotes them)
    csv_content = (
        "Activity ID,Activity Date,Activity Name,Activity Type,"
        "Elapsed Time,Moving Time,Distance,Elevation Gain,"
        "Average Heart Rate,Max Heart Rate,Filename\n"
        f'{activity_id},"Jan 19, 2024, 8:00:00 AM",Morning Run,Run,'
        f"3600,3500,10000,120,142,165,{filename}\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("activities.csv", csv_content)
        if include_track:
            zf.writestr(filename, _sample_gpx(points=8))
    return buffer.getvalue()


def test_parse_gpx_activity_includes_hr() -> None:
    df = parse_strava_track_file(_sample_gpx(points=8), filename="a.gpx")
    assert not df.empty
    assert "hr" in df.columns
    assert df["hr"].notna().any()
    assert df["lat"].notna().all()


def test_merge_fill_empty_keeps_non_empty() -> None:
    existing = {"id": 1, "laps": [], "map": {}, "name": "A"}
    incoming = {
        "id": 1,
        "laps": [{"lap_index": 1}],
        "map": {"summary_polyline": "abc"},
        "name": "",
    }
    merged = merge_fill_empty(existing, incoming)
    assert merged["name"] == "A"
    assert merged["laps"] == [{"lap_index": 1}]
    assert merged["map"]["summary_polyline"] == "abc"


def test_raw_needs_enrichment_for_archive_stub() -> None:
    detail = {"id": 1, "laps": [], "map": {}}
    assert raw_needs_enrichment(detail, has_timeseries=True) is True
    assert raw_needs_enrichment(detail, has_timeseries=False) is True
    complete = {
        "id": 1,
        "laps": [{"lap_index": 1}],
        "map": {"summary_polyline": "xyz"},
    }
    assert raw_needs_enrichment(complete, has_timeseries=True) is False


def test_archive_import_creates_artifacts(
    storage: CsvStorage, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    AthletesRepo(storage).create(
        {
            "athleteId": "athlete-1",
            "coachId": "coach-1",
            "name": "Test",
            "hrRest": 50,
            "hrMax": 190,
        }
    )
    service = StravaArchiveService(storage=storage, config=config)
    monkeypatch.setattr(
        service.strava,
        "_apply_sync_metrics",
        lambda **kwargs: None,
    )

    activity_id = "11277623453"
    result = service.import_strava_archive(
        _build_archive_zip(activity_id=activity_id),
        "athlete-1",
    )
    assert result["imported"] == 1
    assert result["merged"] == 0
    assert activity_id in result["touched_ids"]

    row = service.strava.activities.get(activity_id)
    assert row is not None
    assert str(row["activityId"]) == activity_id
    assert row["source"] == "strava"
    assert row["sportType"] == "Run"

    raw_path = config.raw_strava_dir / f"{activity_id}.json"
    assert raw_path.exists()
    detail = json.loads(raw_path.read_text(encoding="utf-8"))
    assert str(detail["id"]) == activity_id

    ts_path = config.timeseries_dir / f"{activity_id}.csv"
    assert ts_path.exists()
    ts_df = pd.read_csv(ts_path)
    assert not ts_df.empty
    assert list(ts_df.columns) == [
        "timestamp",
        "hr",
        "paceKmh",
        "elevationM",
        "cadence",
        "lat",
        "lon",
    ]


def test_archive_import_no_duplicate_on_second_pass(
    storage: CsvStorage, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    AthletesRepo(storage).create(
        {
            "athleteId": "athlete-1",
            "coachId": "coach-1",
            "name": "Test",
            "hrRest": 50,
            "hrMax": 190,
        }
    )
    service = StravaArchiveService(storage=storage, config=config)
    monkeypatch.setattr(service.strava, "_apply_sync_metrics", lambda **kwargs: None)
    zip_bytes = _build_archive_zip()

    first = service.import_strava_archive(zip_bytes, "athlete-1")
    second = service.import_strava_archive(zip_bytes, "athlete-1")

    assert first["imported"] == 1
    assert second["imported"] == 0
    assert second["already_complete"] == 1
    assert second["merged"] == 0

    df = service.strava.activities.list()
    assert len(df) == 1


def test_archive_import_merges_missing_timeseries(
    storage: CsvStorage, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    AthletesRepo(storage).create(
        {
            "athleteId": "athlete-1",
            "coachId": "coach-1",
            "name": "Test",
            "hrRest": 50,
            "hrMax": 190,
        }
    )
    activity_id = "11277623453"
    strava = StravaService(storage=storage, config=config)
    monkeypatch.setattr(strava, "_apply_sync_metrics", lambda **kwargs: None)

    # Seed incomplete activity (row + raw, no timeseries)
    detail = {
        "id": int(activity_id),
        "name": "Morning Run",
        "sport_type": "Run",
        "type": "Run",
        "start_date": "2024-01-19T08:00:00Z",
        "distance": 10000.0,
        "elapsed_time": 3600,
        "moving_time": 3500,
        "total_elevation_gain": 120.0,
        "average_heartrate": 142.0,
        "max_heartrate": 165.0,
        "laps": [],
        "map": {},
    }
    raw_path = strava._save_raw_activity(detail)
    strava.activities.create(
        strava._map_activity_row(
            detail=detail,
            athlete_id="athlete-1",
            has_timeseries=False,
            raw_path=raw_path,
        )
    )
    assert not strava._timeseries_exists(activity_id)

    archive = StravaArchiveService(storage=storage, config=config, strava=strava)
    result = archive.import_strava_archive(
        _build_archive_zip(activity_id=activity_id),
        "athlete-1",
    )
    assert result["imported"] == 0
    assert result["merged"] == 1
    assert strava._timeseries_exists(activity_id)
    assert len(strava.activities.list()) == 1
    row = strava.activities.get(activity_id)
    assert bool(row["hasTimeseries"]) is True


def test_sync_merges_incomplete_archive_raw(
    storage: CsvStorage, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = dt.datetime(2024, 1, 20, tzinfo=dt.timezone.utc)
    expires_at = int((now + dt.timedelta(hours=1)).timestamp())
    activity_id = "11277623453"
    AthletesRepo(storage).create(
        {
            "athleteId": "athlete-1",
            "coachId": "coach-1",
            "name": "Test",
            "hrRest": 50,
            "hrMax": 190,
        }
    )

    # Incomplete archive-like raw (empty laps/polyline, no timeseries)
    incomplete = {
        "id": int(activity_id),
        "name": "Morning Run",
        "sport_type": "Run",
        "distance": 10000.0,
        "elapsed_time": 3600,
        "moving_time": 3500,
        "total_elevation_gain": 120.0,
        "average_heartrate": 142.0,
        "max_heartrate": 165.0,
        "start_date": "2024-01-19T07:00:00Z",
        "start_date_local": "2024-01-19T08:00:00+01:00",
        "laps": [],
        "map": {},
    }
    service = StravaService(
        storage=storage,
        config=config,
        session=FakeSession([]),
        now_fn=lambda: now,
    )
    monkeypatch.setattr(service, "_apply_sync_metrics", lambda **kwargs: None)
    raw_path = service._save_raw_activity(incomplete)
    service.activities.create(
        service._map_activity_row(
            detail=incomplete,
            athlete_id="athlete-1",
            has_timeseries=False,
            raw_path=raw_path,
        )
    )

    # Store tokens without exchange
    from utils.crypto import encrypt_text, get_fernet

    fernet = get_fernet(config.encryption_key)
    service.tokens.storage.upsert(
        service.tokens.file_name,
        ["athleteId", "provider"],
        {
            "athleteId": "athlete-1",
            "provider": "strava",
            "accessTokenEnc": encrypt_text(fernet, "token-abc"),
            "refreshTokenEnc": encrypt_text(fernet, "refresh-abc"),
            "expiresAt": expires_at,
        },
    )

    responses = [
        (
            "GET",
            f"{API_BASE}/athlete/activities",
            FakeResponse(
                200,
                [{"id": int(activity_id), "start_date": "2024-01-19T07:00:00Z"}],
            ),
        ),
        (
            "GET",
            f"{API_BASE}/activities/{activity_id}",
            FakeResponse(
                200,
                {
                    "id": int(activity_id),
                    "name": "Morning Run",
                    "distance": 10000.0,
                    "elapsed_time": 3600,
                    "moving_time": 3500,
                    "total_elevation_gain": 120.0,
                    "average_heartrate": 142.0,
                    "max_heartrate": 165.0,
                    "start_date": "2024-01-19T07:00:00Z",
                    "start_date_local": "2024-01-19T08:00:00+01:00",
                    "map": {"summary_polyline": "enriched_poly"},
                    "laps": [
                        {
                            "lap_index": 1,
                            "elapsed_time": 3600,
                            "moving_time": 3500,
                            "distance": 10000.0,
                            "total_elevation_gain": 120.0,
                            "average_heartrate": 142.0,
                            "max_heartrate": 165.0,
                            "average_speed": 2.8,
                        }
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
                    "time": {"data": [0, 10]},
                    "heartrate": {"data": [120, 130]},
                    "velocity_smooth": {"data": [3.0, 3.5]},
                    "altitude": {"data": [200.0, 201.5]},
                    "cadence": {"data": [80, 82]},
                    "latlng": {"data": [[48.1, 2.3], [48.1001, 2.3001]]},
                },
            ),
        ),
    ]
    service.session = FakeSession(responses)
    imported = service.sync_last_n_days("athlete-1", 14)

    assert imported == [activity_id]
    assert len(service.activities.list()) == 1
    detail = json.loads((config.raw_strava_dir / f"{activity_id}.json").read_text())
    assert detail["map"]["summary_polyline"] == "enriched_poly"
    assert detail["laps"]
    assert service._timeseries_exists(activity_id)
    row = service.activities.get(activity_id)
    assert row["polyline"] == "enriched_poly"
    assert bool(row["hasTimeseries"]) is True
    assert service.session.empty
