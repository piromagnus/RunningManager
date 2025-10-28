"""Utilities for working with interval session step structures."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


Action = Dict[str, Any]
Loop = Dict[str, Any]
Steps = Dict[str, Any]


def _ensure_int(value: Any, default: int = 0, minimum: int = 0) -> int:
    try:
        num = int(float(value))
    except Exception:
        num = default
    return max(num, minimum)


def _clean_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def new_run_action(
    *,
    sec: int = 60,
    target_type: Optional[str] = None,
    target_label: Optional[str] = None,
) -> Action:
    return {
        "kind": "run",
        "sec": max(int(sec), 5),
        "targetType": target_type,
        "targetLabel": target_label,
        "ascendM": 0,
        "descendM": 0,
    }


def new_recovery_action(*, sec: int = 60) -> Action:
    return {
        "kind": "recovery",
        "sec": max(int(sec), 5),
        "targetType": None,
        "targetLabel": None,
        "ascendM": 0,
        "descendM": 0,
    }


def new_loop() -> Loop:
    return {
        "repeats": 1,
        "actions": [
            new_run_action(),
            new_recovery_action(),
        ],
    }


def _normalise_action(action: Optional[Dict[str, Any]], fallback_kind: str = "run") -> Action:
    data = dict(action or {})
    kind = str(data.get("kind") or fallback_kind).lower()
    if kind not in {"run", "recovery"}:
        kind = fallback_kind
    sec = _ensure_int(data.get("sec"), default=60, minimum=5)
    target_type = _clean_str(data.get("targetType"))
    target_label = _clean_str(data.get("targetLabel"))
    ascend = _ensure_int(data.get("ascendM"), default=0, minimum=0)
    descend = _ensure_int(data.get("descendM"), default=0, minimum=0)
    if kind == "recovery":
        target_type = None
        target_label = None
        ascend = 0
        descend = 0
    return {
        "kind": kind,
        "sec": sec,
        "targetType": target_type,
        "targetLabel": target_label,
        "ascendM": ascend,
        "descendM": descend,
    }


def _sum_seconds(items: List[Action]) -> int:
    return sum(_ensure_int(item.get("sec"), default=0, minimum=0) for item in items)


def normalize_steps(raw: Optional[Dict[str, Any]]) -> Steps:
    """Return a normalised steps structure with defaults for editor usage."""
    source = deepcopy(raw) if raw else {}

    loops: List[Loop] = []
    loops_source = source.get("loops", []) or []
    for loop in loops_source:
        actions_raw = loop.get("actions") or []
        actions = [_normalise_action(item, fallback_kind="run") for item in actions_raw]
        if not actions:
            actions = [new_run_action(), new_recovery_action()]
        repeats = _ensure_int(loop.get("repeats"), default=1, minimum=1)
        loops.append({"repeats": repeats, "actions": actions})
    if not loops:
        legacy_repeats = source.get("repeats") or []
        for repeat in legacy_repeats:
            work_action = _normalise_action(
                {
                    "kind": "run",
                    "sec": repeat.get("workSec"),
                    "targetType": repeat.get("targetType"),
                    "targetLabel": repeat.get("targetLabel"),
                    "ascendM": repeat.get("ascendM"),
                    "descendM": repeat.get("descendM"),
                },
                fallback_kind="run",
            )
            recovery_action = _normalise_action(
                {
                    "kind": "recovery",
                    "sec": repeat.get("recoverSec"),
                },
                fallback_kind="recovery",
            )
            loops.append({"repeats": 1, "actions": [work_action, recovery_action]})
    if not loops:
        loops = [new_loop()]

    warmup_sec = _ensure_int(source.get("warmupSec"), default=0, minimum=0)
    pre_blocks_raw = source.get("preBlocks")
    if pre_blocks_raw is not None:
        pre_blocks = [_normalise_action(block, fallback_kind="run") for block in pre_blocks_raw]
    elif warmup_sec > 0:
        pre_blocks = [
            _normalise_action(
                {
                    "kind": "run",
                    "sec": warmup_sec,
                    "targetType": "sensation",
                    "targetLabel": "Fundamental",
                },
                fallback_kind="run",
            )
        ]
    else:
        pre_blocks = [new_run_action(sec=600, target_type="sensation", target_label="Fundamental")]

    cooldown_sec = _ensure_int(source.get("cooldownSec"), default=0, minimum=0)
    post_blocks_raw = source.get("postBlocks")
    if post_blocks_raw is not None:
        post_blocks = [
            _normalise_action(block, fallback_kind="recovery") for block in post_blocks_raw
        ]
    elif cooldown_sec > 0:
        post_blocks = [
            _normalise_action(
                {
                    "kind": "recovery",
                    "sec": cooldown_sec,
                },
                fallback_kind="recovery",
            )
        ]
    else:
        post_blocks = [new_recovery_action(sec=300)]

    between_block_raw = source.get("betweenBlock")
    if between_block_raw:
        between_block = _normalise_action(between_block_raw, fallback_kind="recovery")
    else:
        between_sec = _ensure_int(source.get("betweenLoopRecoverSec"), default=0, minimum=0)
        between_block = (
            _normalise_action({"kind": "recovery", "sec": between_sec}, fallback_kind="recovery")
            if between_sec > 0
            else None
        )

    return {
        "preBlocks": pre_blocks,
        "loops": loops,
        "betweenBlock": between_block,
        "postBlocks": post_blocks,
    }


def serialize_steps(steps: Steps) -> Dict[str, Any]:
    """Convert a normalised structure to a JSON-serialisable dictionary."""
    pre_blocks = [
        _normalise_action(block, fallback_kind="run") for block in steps.get("preBlocks") or []
    ]
    loops_out: List[Dict[str, Any]] = []
    for loop in steps.get("loops") or []:
        actions = [_normalise_action(act, fallback_kind="run") for act in loop.get("actions") or []]
        if not actions:
            actions = [new_run_action()]
        repeats = _ensure_int(loop.get("repeats"), default=1, minimum=1)
        loops_out.append({"repeats": repeats, "actions": actions})

    between_block = steps.get("betweenBlock")
    between_serialised = (
        _normalise_action(between_block, fallback_kind="recovery") if between_block else None
    )

    post_blocks = [
        _normalise_action(block, fallback_kind="recovery")
        for block in steps.get("postBlocks") or []
    ]

    payload: Dict[str, Any] = {
        "preBlocks": pre_blocks,
        "loops": loops_out,
        "postBlocks": post_blocks,
        "warmupSec": _sum_seconds(pre_blocks),
        "cooldownSec": _sum_seconds(post_blocks),
    }
    if between_serialised:
        payload["betweenBlock"] = between_serialised
        payload["betweenLoopRecoverSec"] = between_serialised["sec"]
    else:
        payload["betweenLoopRecoverSec"] = 0
    return payload


def format_duration_label(seconds: int) -> str:
    total = max(int(seconds or 0), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}"
    if minutes:
        return f"{minutes}m{secs:02d}"
    return f"{secs}s"


def describe_action(action: Action) -> str:
    kind = (action.get("kind") or "run").lower()
    duration = format_duration_label(action.get("sec"))
    if kind == "recovery":
        return f"Recovery {duration}"

    target_type = action.get("targetType")
    target_label = action.get("targetLabel")
    target_bits: List[str] = []
    if target_type in {"hr", "pace", "speed", "distance", "threshold", "denivele"}:
        label = target_label or ""
        target_bits.append(f"{target_type} {label}".strip())
    elif target_type == "sensation" and target_label:
        target_bits.append(f"Sensation {target_label}")
    ascend = _ensure_int(action.get("ascendM"), default=0, minimum=0)
    descend = _ensure_int(action.get("descendM"), default=0, minimum=0)
    if ascend:
        target_bits.append(f"+{ascend} m")
    if descend:
        target_bits.append(f"-{descend} m")
    target_info = f" – {' • '.join(target_bits)}" if target_bits else ""
    return f"Run {duration}{target_info}"


def clone_steps(steps: Steps) -> Steps:
    return deepcopy(steps)
