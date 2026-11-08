"""Tests for activity GPX export."""

from __future__ import annotations

import pandas as pd
from lxml import etree

from utils.gpx_export import GPX_NAMESPACE, build_gpx_export


def test_build_gpx_export_preserves_route_elevation_and_timestamps():
    route = pd.DataFrame(
        [
            {
                "lat": 45.0,
                "lon": 5.0,
                "elevationM": 100.5,
                "timestamp": "2025-01-05T07:00:00+01:00",
            },
            {
                "lat": 45.001,
                "lon": 5.001,
                "elevationM": 101.5,
                "timestamp": "2025-01-05T07:00:10+01:00",
            },
        ]
    )

    exported = build_gpx_export(route, name="Morning Run")

    assert exported is not None
    root = etree.fromstring(exported)
    points = root.xpath(".//gpx:trkpt", namespaces={"gpx": GPX_NAMESPACE})
    assert root.findtext(f"{{{GPX_NAMESPACE}}}trk/{{{GPX_NAMESPACE}}}name") == "Morning Run"
    assert len(points) == 2
    assert points[0].findtext(f"{{{GPX_NAMESPACE}}}ele") == "100.50"
    assert points[0].findtext(f"{{{GPX_NAMESPACE}}}time") == "2025-01-05T06:00:00Z"


def test_build_gpx_export_skips_invalid_coordinates_and_requires_two_points():
    one_valid_point = pd.DataFrame(
        [
            {"lat": 45.0, "lon": 5.0},
            {"lat": 91.0, "lon": 5.0},
        ]
    )

    assert build_gpx_export(one_valid_point, name="Short route") is None
    assert build_gpx_export(pd.DataFrame(), name="Empty route") is None
