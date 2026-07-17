"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Merge helpers for Strava archive import and API sync enrichment.
Fill empty/missing values only; never overwrite non-empty with empty.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, MutableMapping, Optional


def is_empty_value(value: Any) -> bool:
    """Return True when ``value`` should be treated as missing for merge fills."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
        return True
    try:
        # pandas / numpy NA
        if value != value:  # noqa: PLR0124
            return True
    except Exception:
        pass
    return False


def merge_fill_empty(existing: Any, incoming: Any) -> Any:
    """Recursively fill empty slots in ``existing`` from ``incoming``.

    Non-empty existing values are kept. Incoming empty values never clear data.
    """
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged: Dict[str, Any] = dict(existing)
        for key, incoming_value in incoming.items():
            if key not in merged or is_empty_value(merged.get(key)):
                if not is_empty_value(incoming_value):
                    merged[key] = incoming_value
            elif isinstance(merged.get(key), dict) and isinstance(incoming_value, dict):
                merged[key] = merge_fill_empty(merged[key], incoming_value)
        return merged
    if is_empty_value(existing) and not is_empty_value(incoming):
        return incoming
    return existing


def merge_activity_row_updates(
    existing_row: Mapping[str, Any],
    incoming_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return column updates for empty fields only (never touches activityId)."""
    updates: Dict[str, Any] = {}
    for key, incoming_value in incoming_row.items():
        if key == "activityId":
            continue
        existing_value = existing_row.get(key)
        if key == "hasTimeseries":
            # Allow False → True upgrade when streams appear after archive import
            if not bool(existing_value) and bool(incoming_value):
                updates[key] = True
            continue
        if is_empty_value(existing_value) and not is_empty_value(incoming_value):
            updates[key] = incoming_value
    return updates


def raw_needs_enrichment(
    detail: Optional[Mapping[str, Any]],
    *,
    has_timeseries: bool,
) -> bool:
    """True when cached raw/detail should be refreshed from the Strava API."""
    if not has_timeseries:
        return True
    if detail is None:
        return True
    laps = detail.get("laps")
    if is_empty_value(laps):
        return True
    map_obj = detail.get("map")
    polyline = None
    if isinstance(map_obj, Mapping):
        polyline = map_obj.get("summary_polyline") or map_obj.get("polyline")
    if is_empty_value(polyline):
        return True
    return False


def apply_row_updates_inplace(
    target: MutableMapping[str, Any],
    updates: Mapping[str, Any],
) -> None:
    """Apply ``updates`` onto ``target`` in place."""
    for key, value in updates.items():
        target[key] = value
