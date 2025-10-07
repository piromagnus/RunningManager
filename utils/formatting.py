"""
FR locale display helpers for decimals and units.

Note: CSV storage must keep '.' as decimal separator. These helpers
are for UI rendering only.
"""

from __future__ import annotations

from typing import Optional

from babel import numbers

LOCALE = "fr_FR"


def set_locale(locale_str: str = "fr_FR") -> None:
    global LOCALE
    try:
        # Validate by formatting a simple number
        numbers.format_decimal(1.0, locale=locale_str)
        LOCALE = locale_str
    except Exception:
        LOCALE = "fr_FR"


def _nbsp() -> str:
    return "\u00A0"


def fmt_decimal(value: Optional[float], digits: Optional[int] = None) -> str:
    if value is None:
        return ""
    fmt = None
    if digits is not None:
        fmt = "#" if digits == 0 else "#." + ("0" * digits)
    formatted = numbers.format_decimal(value, format=fmt, locale=LOCALE)
    if abs(value) >= 1000:
        separators = {" ", "\u00A0", "\u202F"}
        if not any(sep in formatted for sep in separators):
            integer, sep, fractional = formatted.partition(",")
            try:
                clean_integer = integer.replace(" ", "").replace("\u00A0", "").replace("\u202F", "")
                grouped = f"{int(clean_integer):,}".replace(",", " ")
                formatted = grouped + (sep + fractional if sep else "")
            except ValueError:
                pass
    if digits is None and "," in formatted:
        head, sep, tail = formatted.partition(",")
        trimmed = tail.rstrip("0")
        if trimmed == "":
            formatted = head
        elif trimmed != tail:
            formatted = head + sep + trimmed
    return formatted


def fmt_km(km: Optional[float]) -> str:
    if km is None:
        return ""
    return f"{fmt_decimal(km, 1)}{_nbsp()}km"


def fmt_m(meters: Optional[float]) -> str:
    if meters is None:
        return ""
    # integers preferred
    return f"{numbers.format_decimal(int(meters), locale=LOCALE)}{_nbsp()}m"


def fmt_speed_kmh(speed_kmh: Optional[float]) -> str:
    if speed_kmh is None:
        return ""
    return f"{fmt_decimal(speed_kmh, 1)}{_nbsp()}km/h"


def to_str_storage(value: Optional[float], ndigits: Optional[int] = None) -> str:
    if value is None:
        return ""
    if ndigits is None:
        return f"{value}"
    return f"{value:.{ndigits}f}"
