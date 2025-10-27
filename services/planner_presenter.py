"""Pure helpers to build Planner view models for unit testing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from services.interval_utils import describe_action, format_duration_label, normalize_steps
from utils.formatting import fmt_decimal, fmt_m


@dataclass(frozen=True)
class CardAction:
    icon: str
    label: str
    action: str
    session_id: str


@dataclass(frozen=True)
class CardSection:
    title: str
    lines: List[str]


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


def build_card_view_model(
    session: Dict[str, object],
    *,
    estimated_distance_km: Optional[float] = None,
    distance_eq_km: Optional[float] = None,
) -> Dict[str, object]:
    session_id = str(session.get("plannedSessionId") or "")
    session_type = str(session.get("type") or "").upper()
    duration = _coerce_int(session.get("plannedDurationSec")) or 0
    duration_label = format_duration_label(duration)
    header = session_type.replace("_", " ").title()
    if duration:
        header = f"{header} • {duration_label}"

    meta: List[str] = []
    planned_distance = _coerce_float(session.get("plannedDistanceKm"))
    if planned_distance is not None:
        meta.append(f"{fmt_decimal(planned_distance, 1)} km")
    elif estimated_distance_km is not None:
        meta.append(f"≈{fmt_decimal(estimated_distance_km, 1)} km")
    if distance_eq_km is not None:
        meta.append(f"DEQ {fmt_decimal(distance_eq_km, 1)} km")

    ascent = _coerce_int(session.get("plannedAscentM"))
    if ascent:
        meta.append(fmt_m(ascent))

    target_type = session.get("targetType")
    target_label = session.get("targetLabel")
    if target_type or target_label:
        target_bits = [str(target_type or "").strip()]
        if target_label:
            target_bits.append(str(target_label))
        meta.append(" ".join(bit for bit in target_bits if bit))

    end_mode = session.get("stepEndMode")
    if isinstance(end_mode, str) and end_mode:
        meta.append(f"mode {end_mode}")

    sections: List[CardSection] = []
    if session_type == "INTERVAL_SIMPLE":
        raw_steps = session.get("stepsJson")
        steps_payload: Optional[Dict[str, object]] = None
        if isinstance(raw_steps, dict):
            steps_payload = raw_steps
        elif isinstance(raw_steps, str) and raw_steps:
            try:
                steps_payload = json.loads(raw_steps)
            except Exception:
                steps_payload = None
        if steps_payload:
            normalised = normalize_steps(steps_payload)
            if normalised["preBlocks"]:
                sections.append(
                    CardSection(
                        "Avant",
                        [describe_action(block) for block in normalised["preBlocks"]],
                    )
                )
            for index, loop in enumerate(normalised["loops"], start=1):
                repeats = max(1, _coerce_int(loop.get("repeats")) or 1)
                lines = [describe_action(action) for action in loop.get("actions") or []]
                sections.append(CardSection(f"Boucle {index} ×{repeats}", lines))
            between = normalised.get("betweenBlock")
            if between and int(between.get("sec") or 0) > 0:
                sections.append(CardSection("Entre boucles", [describe_action(between)]))
            if normalised["postBlocks"]:
                sections.append(
                    CardSection(
                        "Après",
                        [describe_action(block) for block in normalised["postBlocks"]],
                    )
                )

    actions = [
        CardAction("⚙️", "Edit session", "edit", session_id),
        CardAction("📄", "Save as template", "save-template", session_id),
        CardAction("🗑️", "Delete session", "delete", session_id),
        CardAction("🔎", "View details", "view", session_id),
    ]

    return {
        "header": header,
        "meta": meta,
        "actions": actions,
        "sections": sections,
    }


def build_empty_state_placeholder(day_label: str) -> Dict[str, str]:
    return {
        "message": f"No sessions planned for {day_label}.",
        "cta": "Click ＋ to add one.",
    }
