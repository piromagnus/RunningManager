"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Parse Strava GDPR export track files (FIT / GPX / TCX, optionally .gz)
into the app timeseries schema used by Strava API sync.
"""

from __future__ import annotations

import gzip
import io
import logging
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from lxml import etree

LOGGER = logging.getLogger(__name__)

TIMESERIES_COLUMNS = [
    "timestamp",
    "hr",
    "paceKmh",
    "elevationM",
    "cadence",
    "lat",
    "lon",
]

# Semicircles → degrees for FIT GPS
_SEMICIRCLES_TO_DEGREES = 180.0 / (2**31)


def parse_strava_track_file(
    data: Union[bytes, Path],
    *,
    filename: Optional[str] = None,
) -> pd.DataFrame:
    """Parse a Strava export track into timeseries columns.

    Returns an empty DataFrame on failure. Output columns match API sync:
    ``timestamp, hr, paceKmh, elevationM, cadence, lat, lon``.
    """
    raw_bytes, name = _read_bytes(data, filename)
    raw_bytes, name = _maybe_gunzip(raw_bytes, name)
    lower = name.lower()
    try:
        if lower.endswith(".fit"):
            df = _parse_fit(raw_bytes)
        elif lower.endswith(".gpx"):
            df = _parse_gpx_activity(raw_bytes)
        elif lower.endswith(".tcx"):
            df = _parse_tcx(raw_bytes)
        else:
            LOGGER.warning("Unsupported Strava track extension for %s", name)
            return pd.DataFrame(columns=TIMESERIES_COLUMNS)
    except Exception:
        LOGGER.exception("Failed to parse Strava track file %s", name)
        return pd.DataFrame(columns=TIMESERIES_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=TIMESERIES_COLUMNS)
    return _normalize_timeseries(df)


def _read_bytes(data: Union[bytes, Path], filename: Optional[str]) -> tuple[bytes, str]:
    if isinstance(data, Path):
        return data.read_bytes(), filename or data.name
    name = filename or "activity.bin"
    return data, name


def _maybe_gunzip(raw_bytes: bytes, name: str) -> tuple[bytes, str]:
    lower = name.lower()
    if lower.endswith(".gz"):
        try:
            raw_bytes = gzip.decompress(raw_bytes)
        except Exception:
            LOGGER.exception("Failed to gunzip track file %s", name)
            raise
        name = name[:-3]
        return raw_bytes, name
    # Some exports store gzip without .gz in Filename but magic header
    if raw_bytes[:2] == b"\x1f\x8b":
        try:
            raw_bytes = gzip.decompress(raw_bytes)
            if lower.endswith(".gz"):
                name = name[:-3]
        except Exception:
            LOGGER.warning("Gzip magic detected but decompress failed for %s; using raw", name)
    return raw_bytes, name


def _parse_fit(raw_bytes: bytes) -> pd.DataFrame:
    try:
        from fitparse import FitFile
    except ImportError:
        LOGGER.error("fitparse is required to parse FIT files from Strava archives")
        return pd.DataFrame()

    fitfile = FitFile(io.BytesIO(raw_bytes))
    rows: list[dict[str, Any]] = []
    prev_dist_m: Optional[float] = None
    prev_ts: Optional[pd.Timestamp] = None

    for record in fitfile.get_messages("record"):
        fields = {f.name: f.value for f in record if f.value is not None}
        ts = fields.get("timestamp")
        lat_raw = fields.get("position_lat")
        lon_raw = fields.get("position_long")
        lat = (float(lat_raw) * _SEMICIRCLES_TO_DEGREES) if lat_raw is not None else None
        lon = (float(lon_raw) * _SEMICIRCLES_TO_DEGREES) if lon_raw is not None else None
        # Filter invalid FIT coordinates
        if lat is not None and (lat < -90 or lat > 90):
            lat = None
        if lon is not None and (lon < -180 or lon > 180):
            lon = None

        speed_ms = fields.get("enhanced_speed")
        if speed_ms is None:
            speed_ms = fields.get("speed")
        pace_kmh: Optional[float] = None
        if speed_ms is not None:
            try:
                pace_kmh = float(speed_ms) * 3.6
            except (TypeError, ValueError):
                pace_kmh = None

        dist_m = fields.get("distance")
        ts_parsed = pd.to_datetime(ts, utc=True, errors="coerce") if ts is not None else pd.NaT
        if pace_kmh is None and dist_m is not None and prev_dist_m is not None:
            if pd.notna(ts_parsed) and prev_ts is not None and pd.notna(prev_ts):
                dt_sec = (ts_parsed - prev_ts).total_seconds()
                if dt_sec > 0:
                    pace_kmh = ((float(dist_m) - float(prev_dist_m)) / dt_sec) * 3.6

        elev = fields.get("enhanced_altitude")
        if elev is None:
            elev = fields.get("altitude")

        rows.append(
            {
                "timestamp": ts_parsed.isoformat() if pd.notna(ts_parsed) else None,
                "hr": fields.get("heart_rate"),
                "paceKmh": pace_kmh,
                "elevationM": elev,
                "cadence": fields.get("cadence"),
                "lat": lat,
                "lon": lon,
            }
        )
        if dist_m is not None:
            prev_dist_m = float(dist_m)
        if pd.notna(ts_parsed):
            prev_ts = ts_parsed

    return pd.DataFrame(rows)


def _parse_gpx_activity(raw_bytes: bytes) -> pd.DataFrame:
    """Parse activity GPX including optional HR/cadence extensions.

    Unlike route GPX parsing, does not require 100 points.
    """
    try:
        root = etree.fromstring(raw_bytes)
    except etree.XMLSyntaxError:
        LOGGER.warning("Invalid GPX XML in Strava archive track")
        return pd.DataFrame()

    # Local-name lookups work across namespaces
    trkpts = root.xpath(".//*[local-name()='trkpt']")
    if not trkpts:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for trkpt in trkpts:
        try:
            lat = float(trkpt.get("lat"))
            lon = float(trkpt.get("lon"))
        except (TypeError, ValueError):
            continue

        elev = _first_text(trkpt, "ele")
        time_text = _first_text(trkpt, "time")
        hr = _first_text(trkpt, "hr")
        cad = _first_text(trkpt, "cad")
        # Some exports use gpxtpx:hr / cadence under extensions
        if hr is None:
            hr = _first_text(trkpt, "heartrate")
        if cad is None:
            cad = _first_text(trkpt, "cadence")

        rows.append(
            {
                "timestamp": time_text,
                "hr": _to_float(hr),
                "paceKmh": None,
                "elevationM": _to_float(elev),
                "cadence": _to_float(cad),
                "lat": lat,
                "lon": lon,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = _derive_pace_from_latlon(df)
    return df


def _parse_tcx(raw_bytes: bytes) -> pd.DataFrame:
    try:
        root = etree.fromstring(raw_bytes)
    except etree.XMLSyntaxError:
        LOGGER.warning("Invalid TCX XML in Strava archive track")
        return pd.DataFrame()

    trackpoints = root.xpath(".//*[local-name()='Trackpoint']")
    if not trackpoints:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for tp in trackpoints:
        time_text = _first_text(tp, "Time")
        lat = _first_text(tp, "LatitudeDegrees")
        lon = _first_text(tp, "LongitudeDegrees")
        elev = _first_text(tp, "AltitudeMeters")
        hr = _first_text(tp, "Value")  # HeartRateBpm/Value — may also match other Values
        # Prefer HeartRateBpm/Value specifically
        hr_nodes = tp.xpath(".//*[local-name()='HeartRateBpm']/*[local-name()='Value']")
        if hr_nodes and hr_nodes[0].text:
            hr = hr_nodes[0].text
        else:
            hr = None
        cad = _first_text(tp, "Cadence")
        speed = _first_text(tp, "Speed")  # m/s in some TCX
        pace_kmh = None
        if speed is not None:
            pace_kmh = _to_float(speed)
            if pace_kmh is not None and pace_kmh < 50:
                # Likely m/s
                pace_kmh = pace_kmh * 3.6

        dist = _first_text(tp, "DistanceMeters")
        rows.append(
            {
                "timestamp": time_text,
                "hr": _to_float(hr),
                "paceKmh": pace_kmh,
                "elevationM": _to_float(elev),
                "cadence": _to_float(cad),
                "lat": _to_float(lat),
                "lon": _to_float(lon),
                "_distanceM": _to_float(dist),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if df["paceKmh"].isna().all():
        df = _derive_pace_from_distance(df)
    else:
        df = df.drop(columns=["_distanceM"], errors="ignore")
    if df["paceKmh"].isna().all():
        df = _derive_pace_from_latlon(df)
    return df


def _derive_pace_from_distance(df: pd.DataFrame) -> pd.DataFrame:
    if "_distanceM" not in df.columns:
        return df
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    dist = pd.to_numeric(out["_distanceM"], errors="coerce")
    pace: list[Optional[float]] = [None]
    for i in range(1, len(out)):
        dt_sec = (ts.iloc[i] - ts.iloc[i - 1]).total_seconds() if pd.notna(ts.iloc[i]) and pd.notna(
            ts.iloc[i - 1]
        ) else None
        if dt_sec and dt_sec > 0 and pd.notna(dist.iloc[i]) and pd.notna(dist.iloc[i - 1]):
            pace.append(((float(dist.iloc[i]) - float(dist.iloc[i - 1])) / dt_sec) * 3.6)
        else:
            pace.append(None)
    out["paceKmh"] = pace
    return out.drop(columns=["_distanceM"], errors="ignore")


def _derive_pace_from_latlon(df: pd.DataFrame) -> pd.DataFrame:
    if "lat" not in df.columns or "lon" not in df.columns:
        return df
    out = df.copy()
    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    else:
        return out
    lat = pd.to_numeric(out["lat"], errors="coerce")
    lon = pd.to_numeric(out["lon"], errors="coerce")
    pace: list[Optional[float]] = [None]
    for i in range(1, len(out)):
        dt_sec = (ts.iloc[i] - ts.iloc[i - 1]).total_seconds() if pd.notna(ts.iloc[i]) and pd.notna(
            ts.iloc[i - 1]
        ) else None
        if (
            dt_sec
            and dt_sec > 0
            and pd.notna(lat.iloc[i])
            and pd.notna(lat.iloc[i - 1])
            and pd.notna(lon.iloc[i])
            and pd.notna(lon.iloc[i - 1])
        ):
            dist_km = _haversine_km(
                float(lat.iloc[i - 1]),
                float(lon.iloc[i - 1]),
                float(lat.iloc[i]),
                float(lon.iloc[i]),
            )
            pace.append((dist_km / dt_sec) * 3600.0)
        else:
            pace.append(None)
    if out["paceKmh"].isna().all() if "paceKmh" in out.columns else True:
        out["paceKmh"] = pace
    return out


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import asin, cos, radians, sin, sqrt

    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def _normalize_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TIMESERIES_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[TIMESERIES_COLUMNS]
    if out["timestamp"].notna().any():
        ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.assign(timestamp=ts)
        out = out[out["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)
        out["timestamp"] = out["timestamp"].map(
            lambda v: v.isoformat() if pd.notna(v) else None
        )
    # Drop rows with no useful sensor/GPS data
    useful = out[["hr", "elevationM", "lat", "lon", "paceKmh", "cadence"]].notna().any(axis=1)
    out = out[useful].reset_index(drop=True)
    return out


def _first_text(node: Any, local_name: str) -> Optional[str]:
    found = node.xpath(f".//*[local-name()='{local_name}']")
    if not found:
        return None
    text = found[0].text
    if text is None:
        return None
    text = text.strip()
    return text or None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
