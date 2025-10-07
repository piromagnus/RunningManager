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

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    strava_client_id: Optional[str]
    strava_client_secret: Optional[str]
    strava_redirect_uri: Optional[str]
    data_dir: Path
    encryption_key: Optional[str]
    timeseries_dir: Path
    raw_strava_dir: Path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """Load configuration from environment and provision directories."""
    load_dotenv(override=False)

    data_dir_str = os.getenv("DATA_DIR", "./data")
    data_dir = Path(data_dir_str).expanduser().resolve()

    strava_client_id = os.getenv("STRAVA_CLIENT_ID")
    strava_client_secret = os.getenv("STRAVA_CLIENT_SECRET")
    strava_redirect_uri = os.getenv("STRAVA_REDIRECT_URI")
    encryption_key = os.getenv("ENCRYPTION_KEY")

    timeseries_dir = data_dir / "timeseries"
    raw_strava_dir = data_dir / "raw" / "strava"

    _ensure_dir(data_dir)
    _ensure_dir(timeseries_dir)
    _ensure_dir(raw_strava_dir)

    return Config(
        strava_client_id=strava_client_id,
        strava_client_secret=strava_client_secret,
        strava_redirect_uri=strava_redirect_uri,
        data_dir=data_dir,
        encryption_key=encryption_key,
        timeseries_dir=timeseries_dir,
        raw_strava_dir=raw_strava_dir,
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


