"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

GPX file parser for route and track data.

Handles both timestamped tracks and time-invariant routes (waypoints only).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from lxml import etree
from streamlit.logger import get_logger

logger = get_logger(__name__)


def parse_gpx_to_timeseries(gpx_bytes: bytes) -> pd.DataFrame:
    """Parse GPX file into timeseries DataFrame.

    Extracts track points (trkpt) with lat/lon/elevation.
    Timestamps are optional - parser works for route-only GPX files.

    Args:
        gpx_bytes: Raw GPX file content as bytes

    Returns:
        DataFrame with columns: lat, lon, elevationM (and optional timestamp)
        Empty DataFrame if parsing fails or < 100 points
    """
    try:
        root = etree.fromstring(gpx_bytes)
    except etree.XMLSyntaxError as e:
        logger.warning(f"Invalid GPX XML: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Failed to parse GPX: {e}", exc_info=True)
        return pd.DataFrame()

    # GPX namespace
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}

    # Find all track points
    trkpts = root.xpath(".//gpx:trkpt", namespaces=ns)
    if not trkpts:
        logger.debug("No track points found in GPX")
        return pd.DataFrame()

    rows = []
    for trkpt in trkpts:
        try:
            lat = trkpt.get("lat")
            lon = trkpt.get("lon")
            if lat is None or lon is None:
                continue

            lat_val = float(lat)
            lon_val = float(lon)

            # Extract elevation if present
            ele_elem = trkpt.find(".//gpx:ele", namespaces=ns)
            elevation_val: Optional[float] = None
            if ele_elem is not None and ele_elem.text:
                try:
                    elevation_val = float(ele_elem.text)
                except (ValueError, TypeError):
                    pass

            # Extract timestamp if present (optional)
            time_elem = trkpt.find(".//gpx:time", namespaces=ns)
            timestamp_val: Optional[str] = None
            if time_elem is not None and time_elem.text:
                timestamp_val = time_elem.text.strip()

            rows.append(
                {
                    "lat": lat_val,
                    "lon": lon_val,
                    "elevationM": elevation_val,
                    "timestamp": timestamp_val,
                }
            )
        except (ValueError, TypeError) as e:
            logger.debug(f"Skipping invalid track point: {e}")
            continue

    if len(rows) < 100:
        logger.warning(f"Insufficient track points: {len(rows)} < 100")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Drop duplicates based on lat/lon
    df = df.drop_duplicates(subset=["lat", "lon"], keep="first").reset_index(drop=True)

    # Forward-fill sparse elevation
    if "elevationM" in df.columns:
        df["elevationM"] = df["elevationM"].ffill().bfill()

    # If timestamps exist, ensure they're monotonic (but not required)
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        # Convert to datetime for validation
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            # Drop rows with invalid timestamps but keep the rest
            df = df[df["timestamp"].notna()].reset_index(drop=True)
            # Ensure monotonic if timestamps present
            if len(df) > 1:
                df = df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            logger.debug(f"Timestamp processing failed, treating as route-only: {e}")
            # Remove timestamp column if processing fails
            df = df.drop(columns=["timestamp"])

    # Remove timestamp column if all NaN (route-only GPX)
    if "timestamp" in df.columns and df["timestamp"].isna().all():
        df = df.drop(columns=["timestamp"])

    logger.debug(f"Parsed GPX: {len(df)} points, columns: {list(df.columns)}")
    return df

