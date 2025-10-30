"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Generic helper functions for common operations.

Consolidates helper functions used across multiple modules.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


def default_template_title(session: Dict[str, Any]) -> str:
    """Generate a default template title from session data.

    Creates a title by combining the session type (formatted) and date.

    Args:
        session: Dictionary containing 'type' and 'date' keys

    Returns:
        str: Formatted title like "Fundamental Endurance 2024-01-15"
    """
    raw_type = str(session.get("type") or "Session").replace("_", " ")
    title_part = raw_type.title()
    date_part = str(session.get("date") or "")
    return f"{title_part} {date_part}".strip()


def clean_optional(value: Any) -> str:
    """Clean optional values for display/storage.

    Converts None, empty strings, and NaN to empty string.
    Other values are converted to string.

    Args:
        value: Value to clean (can be None, str, float, etc.)

    Returns:
        str: Cleaned string value (empty string for None/empty/NaN)
    """
    if value in (None, ""):
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def clean_notes(value: Any) -> str:
    """Clean notes field value.

    Similar to clean_optional but specifically for notes fields.

    Args:
        value: Notes value to clean

    Returns:
        str: Cleaned notes string
    """
    if value in (None, ""):
        return ""
    return str(value).strip()

