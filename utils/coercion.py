"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Type coercion utilities for converting values safely to float, int, etc.

Consolidates duplicate coercion functions from across the codebase.
"""

from __future__ import annotations

import math
from typing import Any, Optional


def coerce_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float with a default fallback.

    Handles None, empty strings, and conversion errors by returning the default.

    Args:
        value: Value to coerce (can be None, str, int, float, etc.)
        default: Default value to return if coercion fails (default: 0.0)

    Returns:
        float: Coerced value or default if coercion fails
    """
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def coerce_int(value: Any, default: int = 0) -> int:
    """Coerce a value to int with a default fallback.

    Handles None, empty strings, and conversion errors by returning the default.
    Converts via float first to handle string representations of floats.

    Args:
        value: Value to coerce (can be None, str, int, float, etc.)
        default: Default value to return if coercion fails (default: 0)

    Returns:
        int: Coerced value or default if coercion fails
    """
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    """Safely convert a value to float, handling NaN and None.

    Similar to coerce_float but also handles "NaN" string and returns default for None.

    Args:
        value: Value to convert
        default: Default value to return if conversion fails (default: 0.0)

    Returns:
        float: Converted value or default if conversion fails
    """
    try:
        if value in (None, "", "NaN"):
            return default
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    """Safely convert a value to int, handling NaN and None.

    Similar to coerce_int but also handles "NaN" string and returns default for None.

    Args:
        value: Value to convert
        default: Default value to return if conversion fails (default: 0)

    Returns:
        int: Converted value or default if conversion fails
    """
    try:
        if value in (None, "", "NaN"):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float_optional(value: object) -> Optional[float]:
    """Safely convert a value to float, returning None on failure.

    Handles None, empty strings, "NaN", and math.nan by returning None.

    Args:
        value: Value to convert

    Returns:
        Optional[float]: Converted value or None if conversion fails
    """
    try:
        if value in (None, "", "NaN"):
            return None
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except Exception:
        return None


def safe_int_optional(value: object) -> Optional[int]:
    """Safely convert a value to int, returning None on failure.

    Handles None, empty strings, "NaN" by returning None.

    Args:
        value: Value to convert

    Returns:
        Optional[int]: Converted value or None if conversion fails
    """
    try:
        if value in (None, "", "NaN"):
            return None
        return int(float(value))
    except Exception:
        return None

