"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.pacer_service import PacerService as PacerService


def __getattr__(name: str) -> object:
    if name == "PacerService":
        from services.pacer_service import PacerService

        return PacerService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PacerService"]
