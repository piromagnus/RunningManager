"""Pure helpers to build Planner view models for unit testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from utils.formatting import fmt_decimal


@dataclass(frozen=True)
class CardAction:
    icon: str
    label: str
    action: str
    session_id: str


def _coerce_int(value: object) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _coerce_float(value: object) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def build_card_view_model(session: Dict[str, object], *, estimated_distance_km: Optional[float] = None) -> Dict[str, object]:
    session_id = str(session.get("plannedSessionId") or "")
    session_type = str(session.get("type") or "").upper()
    duration = _coerce_int(session.get("plannedDurationSec")) or 0
    header = f"{session_type} | dur={duration}s"

    badges: List[str] = []
    planned_distance = _coerce_float(session.get("plannedDistanceKm"))
    if planned_distance is not None:
        badges.append(f"dist={fmt_decimal(planned_distance, 1)} km")
    elif estimated_distance_km is not None:
        badges.append(f"est≈{fmt_decimal(estimated_distance_km, 1)} km")

    target_type = session.get("targetType")
    target_label = session.get("targetLabel")
    if target_type or target_label:
        badges.append(f"target={target_type or ''} {target_label or ''}".strip())

    end_mode = session.get("stepEndMode")
    if isinstance(end_mode, str) and end_mode:
        badges.append(f"mode={end_mode}")

    actions = [
        CardAction("⚙️", "Edit session", "edit", session_id),
        CardAction("🗑️", "Delete session", "delete", session_id),
        CardAction("🔎", "View details", "view", session_id),
    ]

    return {
        "header": header,
        "badges": badges,
        "actions": actions,
    }


def build_empty_state_placeholder(day_label: str) -> Dict[str, str]:
    return {
        "message": f"No sessions planned for {day_label}.",
        "cta": "Click ＋ to add one.",
    }
