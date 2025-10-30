"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

ID helpers.
"""

from __future__ import annotations

import uuid


def new_id() -> str:
    return str(uuid.uuid4())
