"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Tests for GPX parser.
"""

from __future__ import annotations

import pandas as pd
import pytest

from utils.gpx_parser import parse_gpx_to_timeseries


def test_parse_gpx_minimal_valid():
    """Test parsing a minimal valid GPX with 5 points."""
    gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <trkseg>
      <trkpt lat="45.0" lon="5.0">
        <ele>100</ele>
      </trkpt>
      <trkpt lat="45.1" lon="5.1">
        <ele>110</ele>
      </trkpt>
      <trkpt lat="45.2" lon="5.2">
        <ele>120</ele>
      </trkpt>
      <trkpt lat="45.3" lon="5.3">
        <ele>130</ele>
      </trkpt>
      <trkpt lat="45.4" lon="5.4">
        <ele>140</ele>
      </trkpt>
    </trkseg>
  </trk>
</gpx>"""
    
    # Should return empty because < 100 points
    result = parse_gpx_to_timeseries(gpx_content.encode())
    assert result.empty


def test_parse_gpx_with_many_points():
    """Test parsing GPX with sufficient points."""
    # Generate GPX with 150 points
    points = []
    for i in range(150):
        lat = 45.0 + i * 0.01
        lon = 5.0 + i * 0.01
        ele = 100 + i * 0.5
        points.append(f'<trkpt lat="{lat}" lon="{lon}"><ele>{ele}</ele></trkpt>')
    
    gpx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <trkseg>
      {"".join(points)}
    </trkseg>
  </trk>
</gpx>"""
    
    result = parse_gpx_to_timeseries(gpx_content.encode())
    assert not result.empty
    assert len(result) >= 100
    assert "lat" in result.columns
    assert "lon" in result.columns
    assert "elevationM" in result.columns
    assert result["lat"].notna().all()
    assert result["lon"].notna().all()


def test_parse_gpx_without_timestamps():
    """Test parsing route-only GPX without timestamps."""
    points = []
    for i in range(150):
        lat = 45.0 + i * 0.01
        lon = 5.0 + i * 0.01
        ele = 100 + i * 0.5
        points.append(f'<trkpt lat="{lat}" lon="{lon}"><ele>{ele}</ele></trkpt>')
    
    gpx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <trkseg>
      {"".join(points)}
    </trkseg>
  </trk>
</gpx>"""
    
    result = parse_gpx_to_timeseries(gpx_content.encode())
    assert not result.empty
    # Timestamp column should not exist for route-only GPX
    assert "timestamp" not in result.columns or result["timestamp"].isna().all()


def test_parse_gpx_with_missing_elevation():
    """Test parsing GPX with some missing elevation values."""
    points = []
    for i in range(150):
        lat = 45.0 + i * 0.01
        lon = 5.0 + i * 0.01
        if i % 10 == 0:
            # Some points without elevation
            points.append(f'<trkpt lat="{lat}" lon="{lon}"></trkpt>')
        else:
            ele = 100 + i * 0.5
            points.append(f'<trkpt lat="{lat}" lon="{lon}"><ele>{ele}</ele></trkpt>')
    
    gpx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <trkseg>
      {"".join(points)}
    </trkseg>
  </trk>
</gpx>"""
    
    result = parse_gpx_to_timeseries(gpx_content.encode())
    assert not result.empty
    # Elevation should be forward-filled
    assert result["elevationM"].notna().all()


def test_parse_gpx_bad_input():
    """Test parsing invalid GPX returns empty DataFrame."""
    bad_content = b"<invalid>xml</invalid>"
    result = parse_gpx_to_timeseries(bad_content)
    assert result.empty


def test_parse_gpx_drops_duplicates():
    """Test that duplicate lat/lon points are dropped."""
    points = []
    for i in range(150):
        lat = 45.0 + (i % 10) * 0.01  # Only 10 unique positions
        lon = 5.0 + (i % 10) * 0.01
        ele = 100 + i * 0.5
        points.append(f'<trkpt lat="{lat}" lon="{lon}"><ele>{ele}</ele></trkpt>')
    
    gpx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <trkseg>
      {"".join(points)}
    </trkseg>
  </trk>
</gpx>"""
    
    result = parse_gpx_to_timeseries(gpx_content.encode())
    assert not result.empty
    # Should have at most 10 unique points (after duplicate removal)
    assert len(result) <= 10

