"""
Configuration loading utilities.

Loads environment variables from `.env`, validates required values, and ensures
data directories exist. Secrets are never logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from streamlit.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Config:
    strava_client_id: Optional[str]
    strava_client_secret: Optional[str]
    strava_redirect_uri: Optional[str]
    data_dir: Path
    encryption_key: Optional[str]
    timeseries_dir: Path
    raw_strava_dir: Path
    laps_dir: Path
    mapbox_token: Optional[str]
    metrics_ts_dir: Path
<<<<<<< HEAD
=======
    n_cluster: int
>>>>>>> feat-speed-profiling-XJGA8


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """Load configuration from environment and provision directories."""
    # project_root = Path(__file__).resolve().parents[1]
    load_dotenv(find_dotenv(), override=True)
    # load_dotenv(project_root / ".env.local", override=False)

    data_dir_str = os.getenv("DATA_DIR", "./data")
    data_dir = Path(data_dir_str).expanduser().resolve()

    strava_client_id = os.getenv("STRAVA_CLIENT_ID")
    strava_client_secret = os.getenv("STRAVA_CLIENT_SECRET")
    strava_redirect_uri = os.getenv("STRAVA_REDIRECT_URI")
    logger.debug("STRAVA_CLIENT_ID: %s", strava_client_id)
    encryption_key = os.getenv("ENCRYPTION_KEY")
    mapbox_token = os.getenv("MAPBOX_API_KEY")
    logger.debug("MAPBOX_API_KEY: %s", mapbox_token)

    timeseries_dir = data_dir / "timeseries"
    raw_strava_dir = data_dir / "raw" / "strava"
    laps_dir = data_dir / "laps"
    metrics_ts_dir = data_dir / "metrics_ts"
<<<<<<< HEAD
=======

    # Load n_cluster with default of 5
    n_cluster_str = os.getenv("N_CLUSTER", "5")
    try:
        n_cluster = int(n_cluster_str)
    except (ValueError, TypeError):
        n_cluster = 5
>>>>>>> feat-speed-profiling-XJGA8

    _ensure_dir(data_dir)
    _ensure_dir(timeseries_dir)
    _ensure_dir(raw_strava_dir)
    _ensure_dir(laps_dir)
    _ensure_dir(metrics_ts_dir)

    return Config(
        strava_client_id=strava_client_id,
        strava_client_secret=strava_client_secret,
        strava_redirect_uri=strava_redirect_uri,
        data_dir=data_dir,
        encryption_key=encryption_key,
        timeseries_dir=timeseries_dir,
        raw_strava_dir=raw_strava_dir,
        laps_dir=laps_dir,
        mapbox_token=mapbox_token,
        metrics_ts_dir=metrics_ts_dir,
<<<<<<< HEAD
=======
        n_cluster=n_cluster,
>>>>>>> feat-speed-profiling-XJGA8
    )


def redact(value: Optional[str], keep_last: int = 4) -> str:
    """Return a redacted string suitable for logs (never log raw secrets)."""
    if not value:
        return ""
    if len(value) <= keep_last:
        return "***"
    return "***" + value[-keep_last:]


def have_crypto_key(cfg: Config) -> bool:
    return bool(cfg.encryption_key)
