"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from persistence.csv_storage import CsvStorage
from services.hr_zones_service import HrZonesService
from services.timeseries_service import TimeseriesService
from utils.config import Config


def _build_cfg(tmp_path: Path) -> Config:
    data_dir = tmp_path
    timeseries_dir = data_dir / "timeseries"
    raw_dir = data_dir / "raw" / "strava"
    laps_dir = data_dir / "laps"
    metrics_ts_dir = data_dir / "metrics_ts"
    speed_profile_dir = data_dir / "speed_profil"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    laps_dir.mkdir(parents=True, exist_ok=True)
    metrics_ts_dir.mkdir(parents=True, exist_ok=True)
    speed_profile_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        strava_client_id=None,
        strava_client_secret=None,
        strava_redirect_uri=None,
        data_dir=data_dir,
        encryption_key=None,
        timeseries_dir=timeseries_dir,
        raw_strava_dir=raw_dir,
        laps_dir=laps_dir,
        mapbox_token=None,
        metrics_ts_dir=metrics_ts_dir,
        speed_profile_dir=speed_profile_dir,
        n_cluster=5,
        hr_zone_count=5,
        hr_zone_window_days=90,
        hr_zone_fit_activity_types=("RUN", "TRAIL_RUN"),
    )


def _write_activity_timeseries(cfg: Config, activity_id: str, *, include_metrics_ts: bool = True) -> None:
    timestamps = pd.date_range("2025-01-01T08:00:00Z", periods=300, freq="s")
    hr = np.concatenate(
        [
            np.full(60, 120.0),
            np.full(60, 135.0),
            np.full(60, 150.0),
            np.full(60, 165.0),
            np.full(60, 180.0),
        ]
    )
    pace = np.concatenate(
        [
            np.full(60, 8.0),
            np.full(60, 9.0),
            np.full(60, 10.0),
            np.full(60, 11.0),
            np.full(60, 12.0),
        ]
    )
    pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "hr": hr,
            "paceKmh": pace,
        }
    ).to_csv(cfg.timeseries_dir / f"{activity_id}.csv", index=False)

    if include_metrics_ts:
        speedeq = np.linspace(8.5, 12.5, 300)
        pd.DataFrame({"speedeq_smooth": speedeq}).to_csv(
            cfg.metrics_ts_dir / f"{activity_id}.csv",
            index=False,
        )


def _write_constant_hr_timeseries(
    cfg: Config,
    activity_id: str,
    *,
    hr: float,
    pace_kmh: float,
    points: int = 180,
) -> None:
    timestamps = pd.date_range("2025-02-01T08:00:00Z", periods=points, freq="s")
    pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "hr": [hr] * points,
            "paceKmh": [pace_kmh] * points,
        }
    ).to_csv(cfg.timeseries_dir / f"{activity_id}.csv", index=False)


def _write_activities_metrics(storage: CsvStorage) -> None:
    storage.write_csv(
        "activities_metrics.csv",
        pd.DataFrame(
            [
                {
                    "activityId": "act-1",
                    "athleteId": "ath-1",
                    "startDate": "2025-01-01",
                    "category": "RUN",
                    "hrSpeedShift": 0,
                },
                {
                    "activityId": "act-2",
                    "athleteId": "ath-1",
                    "startDate": "2025-01-08",
                    "category": "RIDE",
                    "hrSpeedShift": 0,
                },
            ]
        ),
    )


def _apply_borders_payload(storage: CsvStorage, activity_id: str, payload: dict[str, object]) -> None:
    metrics_df = storage.read_csv("activities_metrics.csv")
    if metrics_df.empty or "activityId" not in metrics_df.columns:
        return
    mask = metrics_df["activityId"].astype(str) == str(activity_id)
    for key, value in payload.items():
        if key not in metrics_df.columns:
            metrics_df[key] = pd.Series(dtype="object")
        metrics_df.loc[mask, key] = value
    storage.write_csv("activities_metrics.csv", metrics_df)


def test_compute_and_store_borders_then_load_activity_zones(tmp_path):
    cfg = _build_cfg(tmp_path)
    storage = CsvStorage(cfg.data_dir)
    ts_service = TimeseriesService(cfg)
    service = HrZonesService(
        storage=storage,
        ts_service=ts_service,
        zone_count=cfg.hr_zone_count,
        window_days=cfg.hr_zone_window_days,
    )
    _write_activity_timeseries(cfg, "act-1")
    _write_activities_metrics(storage)

    payload = service.compute_and_store_borders(
        "act-1",
        athlete_id="ath-1",
        activity_date=dt.date(2025, 1, 1),
        hr_speed_shift=0,
    )
    assert payload is not None
    assert "hrZone_z1_upper" in payload
    assert "hrZone_gmm_sample_count" in payload
    assert "hrZone_zone_count" in payload
    _apply_borders_payload(storage, "act-1", payload)

    loaded = service.get_or_compute_zones("act-1")
    assert loaded is not None
    zones_df, summary_df = loaded

    assert "cluster" in zones_df.columns
    assert "zone" in zones_df.columns
    assert "zone_label" in zones_df.columns
    assert "duration_seconds" in zones_df.columns
    assert len(summary_df) == 5
    assert set(summary_df["zone"].astype(int).tolist()) == {1, 2, 3, 4, 5}
    assert abs(float(summary_df["pct_time"].sum()) - 100.0) < 1e-6
    assert summary_df["avg_speed_kmh"].notna().any()
    assert summary_df["avg_speedeq_kmh"].notna().any()

    stored = storage.read_csv("hr_zones/act-1.csv")
    assert not stored.empty
    assert len(stored) == 5


def test_compute_zones_uses_rolling_window_history_for_constant_activity(tmp_path):
    cfg = _build_cfg(tmp_path)
    storage = CsvStorage(cfg.data_dir)
    ts_service = TimeseriesService(cfg)
    service_long_window = HrZonesService(storage=storage, ts_service=ts_service, window_days=90)
    service_short_window = HrZonesService(storage=storage, ts_service=ts_service, window_days=30)

    _write_activity_timeseries(cfg, "act-reference", include_metrics_ts=False)
    _write_constant_hr_timeseries(cfg, "act-target", hr=132.0, pace_kmh=9.5)
    storage.write_csv(
        "activities_metrics.csv",
        pd.DataFrame(
            [
                {
                    "activityId": "act-reference",
                    "athleteId": "ath-1",
                    "startDate": "2025-01-15",
                    "category": "RUN",
                    "hrSpeedShift": 0,
                },
                {
                    "activityId": "act-target",
                    "athleteId": "ath-1",
                    "startDate": "2025-03-01",
                    "category": "RUN",
                    "hrSpeedShift": 0,
                },
            ]
        ),
    )

    computed = service_long_window.compute_zones("act-target")
    assert computed is not None
    zones_df, summary_df = computed
    assert "cluster" in zones_df.columns
    assert zones_df["cluster"].notna().all()
    assert zones_df["zone"].notna().all()
    assert summary_df["time_seconds"].sum() > 0
    assert service_short_window.compute_zones("act-target") is None


def test_hr_zone_fit_activity_types_filters_reference_samples(tmp_path):
    cfg = _build_cfg(tmp_path)
    storage = CsvStorage(cfg.data_dir)
    ts_service = TimeseriesService(cfg)
    service_default = HrZonesService(
        storage=storage,
        ts_service=ts_service,
        fit_activity_types=("RUN", "TRAIL_RUN"),
    )
    service_with_ride = HrZonesService(
        storage=storage,
        ts_service=ts_service,
        fit_activity_types=("RUN", "TRAIL_RUN", "RIDE"),
    )

    _write_activity_timeseries(cfg, "act-target", include_metrics_ts=False)
    _write_activity_timeseries(cfg, "act-reference-ride", include_metrics_ts=False)
    storage.write_csv(
        "activities_metrics.csv",
        pd.DataFrame(
            [
                {
                    "activityId": "act-reference-ride",
                    "athleteId": "ath-1",
                    "startDate": "2025-01-20",
                    "category": "RIDE",
                    "hrSpeedShift": 0,
                },
                {
                    "activityId": "act-target",
                    "athleteId": "ath-1",
                    "startDate": "2025-02-01",
                    "category": "RUN",
                    "hrSpeedShift": 0,
                },
            ]
        ),
    )

    payload_default = service_default.compute_and_store_borders(
        "act-target",
        athlete_id="ath-1",
        activity_date=dt.date(2025, 2, 1),
        hr_speed_shift=0,
    )
    payload_with_ride = service_with_ride.compute_and_store_borders(
        "act-target",
        athlete_id="ath-1",
        activity_date=dt.date(2025, 2, 1),
        hr_speed_shift=0,
    )
    assert payload_default is not None
    assert payload_with_ride is not None
    assert int(payload_with_ride["hrZone_gmm_sample_count"]) > int(
        payload_default["hrZone_gmm_sample_count"]
    )


def test_compute_zones_always_has_speed_eq_value(tmp_path):
    cfg = _build_cfg(tmp_path)
    storage = CsvStorage(cfg.data_dir)
    ts_service = TimeseriesService(cfg)
    service = HrZonesService(storage=storage, ts_service=ts_service)

    _write_activity_timeseries(cfg, "act-no-metrics", include_metrics_ts=False)
    computed = service.compute_zones("act-no-metrics")
    assert computed is not None
    _, summary_df = computed
    assert summary_df["avg_speedeq_kmh"].notna().any()
    assert np.allclose(
        summary_df["avg_speedeq_kmh"].fillna(0.0).to_numpy(),
        summary_df["avg_speed_kmh"].fillna(0.0).to_numpy(),
    )


def test_compute_zones_returns_none_without_hr(tmp_path):
    cfg = _build_cfg(tmp_path)
    storage = CsvStorage(cfg.data_dir)
    ts_service = TimeseriesService(cfg)
    service = HrZonesService(storage=storage, ts_service=ts_service)

    pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=50, freq="s").astype(str),
            "hr": [np.nan] * 50,
            "paceKmh": [10.0] * 50,
        }
    ).to_csv(cfg.timeseries_dir / "act-no-hr.csv", index=False)

    assert service.compute_zones("act-no-hr") is None
    assert service.get_or_compute_zones("act-no-hr") is None


def test_weekly_zone_data_and_speed_evolution(tmp_path):
    cfg = _build_cfg(tmp_path)
    storage = CsvStorage(cfg.data_dir)
    ts_service = TimeseriesService(cfg)
    service = HrZonesService(storage=storage, ts_service=ts_service)

    _write_activity_timeseries(cfg, "act-1")
    _write_activity_timeseries(cfg, "act-2")
    _write_activities_metrics(storage)
    payload_1 = service.compute_and_store_borders(
        "act-1",
        athlete_id="ath-1",
        activity_date=dt.date(2025, 1, 1),
        hr_speed_shift=0,
    )
    payload_2 = service.compute_and_store_borders(
        "act-2",
        athlete_id="ath-1",
        activity_date=dt.date(2025, 1, 8),
        hr_speed_shift=0,
    )
    assert payload_1 is not None
    assert payload_2 is not None
    _apply_borders_payload(storage, "act-1", payload_1)
    _apply_borders_payload(storage, "act-2", payload_2)

    weekly_run_only = service.build_weekly_zone_data(
        athlete_id="ath-1",
        start_date=pd.Timestamp("2025-01-01").date(),
        end_date=pd.Timestamp("2025-01-15").date(),
        categories=["RUN"],
    )
    assert not weekly_run_only.empty
    assert weekly_run_only["weekLabel"].nunique() == 1
    assert set(weekly_run_only["zone"].astype(int).tolist()) == {1, 2, 3, 4, 5}
    assert weekly_run_only["time_seconds"].sum() > 0

    weekly_all = service.build_weekly_zone_data(
        athlete_id="ath-1",
        start_date=pd.Timestamp("2025-01-01").date(),
        end_date=pd.Timestamp("2025-01-15").date(),
        categories=None,
    )
    assert weekly_all["weekLabel"].nunique() == 2

    evolution = service.build_zone_speed_evolution(
        athlete_id="ath-1",
        start_date=pd.Timestamp("2025-01-01").date(),
        end_date=pd.Timestamp("2025-01-15").date(),
        categories=None,
    )
    assert not evolution.empty
    assert {"avg_speed_kmh", "avg_speedeq_kmh", "zone", "date"}.issubset(evolution.columns)
    assert evolution["avg_speed_kmh"].notna().any()

    points = service.build_activity_zone_speed_points(
        athlete_id="ath-1",
        start_date=pd.Timestamp("2025-01-01").date(),
        end_date=pd.Timestamp("2025-01-15").date(),
        categories=None,
    )
    assert not points.empty
    assert {"activityId", "zone", "zone_label", "date"}.issubset(points.columns)


def test_speed_alignment_uses_hr_shift_value(tmp_path):
    cfg = _build_cfg(tmp_path)
    storage = CsvStorage(cfg.data_dir)
    ts_service = TimeseriesService(cfg)
    service = HrZonesService(storage=storage, ts_service=ts_service)

    timestamps = pd.date_range("2025-01-20T08:00:00Z", periods=240, freq="s")
    hr = np.concatenate([np.full(120, 125.0), np.full(120, 172.0)])
    pace = np.concatenate([np.full(120, 8.0), np.full(120, 15.0)])
    pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "hr": hr,
            "paceKmh": pace,
        }
    ).to_csv(cfg.timeseries_dir / "act-shift.csv", index=False)

    storage.write_csv(
        "activities_metrics.csv",
        pd.DataFrame(
            [
                {
                    "activityId": "act-shift",
                    "athleteId": "ath-1",
                    "startDate": "2025-01-20",
                    "category": "RUN",
                    "hrSpeedShift": 0,
                }
            ]
        ),
    )

    payload_no_shift = service.compute_and_store_borders(
        "act-shift",
        athlete_id="ath-1",
        activity_date=dt.date(2025, 1, 20),
        hr_speed_shift=0,
    )
    assert payload_no_shift is not None
    summary_no_shift = service.load_zone_summary("act-shift")
    assert summary_no_shift is not None

    payload_shifted = service.compute_and_store_borders(
        "act-shift",
        athlete_id="ath-1",
        activity_date=dt.date(2025, 1, 20),
        hr_speed_shift=30,
    )
    assert payload_shifted is not None
    summary_shifted = service.load_zone_summary("act-shift")
    assert summary_shifted is not None

    merged = summary_no_shift.merge(
        summary_shifted,
        on="zone",
        suffixes=("_no_shift", "_shifted"),
    )
    speed_delta = (
        pd.to_numeric(merged["avg_speed_kmh_no_shift"], errors="coerce")
        - pd.to_numeric(merged["avg_speed_kmh_shifted"], errors="coerce")
    ).abs()
    assert (speed_delta > 1e-6).any()


def test_backfill_all_borders_updates_activities_metrics(tmp_path):
    cfg = _build_cfg(tmp_path)
    storage = CsvStorage(cfg.data_dir)
    ts_service = TimeseriesService(cfg)
    service = HrZonesService(storage=storage, ts_service=ts_service)

    _write_activity_timeseries(cfg, "act-1")
    _write_activity_timeseries(cfg, "act-2")
    _write_activities_metrics(storage)

    updated = service.backfill_all_borders()
    assert updated == 2

    metrics_df = storage.read_csv("activities_metrics.csv")
    assert not metrics_df.empty
    for col in ("hrZone_gmm_sample_count", "hrZone_zone_count", "hrZone_z1_upper"):
        parsed = pd.to_numeric(metrics_df[col], errors="coerce")
        assert parsed.notna().sum() == 2

    zone_1 = storage.read_csv("hr_zones/act-1.csv")
    zone_2 = storage.read_csv("hr_zones/act-2.csv")
    assert not zone_1.empty
    assert not zone_2.empty
