"""Export activity route data as GPX 1.1."""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from lxml import etree

GPX_NAMESPACE = "http://www.topografix.com/GPX/1/1"


def build_gpx_export(
    route_df: pd.DataFrame,
    *,
    name: str,
) -> Optional[bytes]:
    """Build a GPX track from route points, preserving elevation and timestamps."""
    if route_df.empty:
        return None

    lat_col = next((column for column in ("lat", "latitude") if column in route_df), None)
    lon_col = next((column for column in ("lon", "lng", "longitude") if column in route_df), None)
    if lat_col is None or lon_col is None:
        return None

    root = etree.Element(
        etree.QName(GPX_NAMESPACE, "gpx"),
        nsmap={None: GPX_NAMESPACE},
        version="1.1",
        creator="Running Manager",
    )
    track = etree.SubElement(root, etree.QName(GPX_NAMESPACE, "trk"))
    etree.SubElement(track, etree.QName(GPX_NAMESPACE, "name")).text = name
    track_segment = etree.SubElement(track, etree.QName(GPX_NAMESPACE, "trkseg"))
    ele_col = next(
        (column for column in ("elevationM", "ele", "altitude") if column in route_df),
        None,
    )
    time_col = "timestamp" if "timestamp" in route_df else None

    for _, row in route_df.iterrows():
        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(lat)
            or not math.isfinite(lon)
            or not -90 <= lat <= 90
            or not -180 <= lon <= 180
        ):
            continue

        point = etree.SubElement(
            track_segment,
            etree.QName(GPX_NAMESPACE, "trkpt"),
            lat=f"{lat:.7f}",
            lon=f"{lon:.7f}",
        )
        if ele_col is not None:
            try:
                elevation = float(row[ele_col])
            except (TypeError, ValueError):
                elevation = math.nan
            if math.isfinite(elevation):
                etree.SubElement(point, etree.QName(GPX_NAMESPACE, "ele")).text = (
                    f"{elevation:.2f}"
                )
        if time_col is not None and pd.notna(row[time_col]):
            timestamp = pd.to_datetime(row[time_col], errors="coerce", utc=True)
            if pd.notna(timestamp):
                etree.SubElement(point, etree.QName(GPX_NAMESPACE, "time")).text = (
                    timestamp.isoformat().replace("+00:00", "Z")
                )

    if len(track_segment) < 2:
        return None
    return etree.tostring(
        root,
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
    )
