"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def serialize_steps(steps: Optional[Dict[str, Any]]) -> str:
    if not steps:
        return ""
    return json.dumps(steps, ensure_ascii=False, separators=(",", ":"))


def deserialize_steps(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    return json.loads(s)
