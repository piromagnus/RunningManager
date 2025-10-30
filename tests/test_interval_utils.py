"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import json

from services.interval_utils import (
    describe_action,
    normalize_steps,
    serialize_steps,
)


def test_normalize_steps_from_legacy_repeats():
    raw = {
        "warmupSec": 300,
        "cooldownSec": 180,
        "betweenLoopRecoverSec": 45,
        "repeats": [
            {
                "workSec": 120,
                "recoverSec": 60,
                "targetType": "pace",
                "targetLabel": "Threshold 60",
            }
        ],
    }
    steps = normalize_steps(raw)
    assert steps["loops"]
    assert steps["preBlocks"][0]["sec"] == 300
    assert steps["postBlocks"][0]["sec"] == 180
    payload = serialize_steps(steps)
    assert payload["warmupSec"] == 300
    assert payload["cooldownSec"] == 180
    assert payload["betweenLoopRecoverSec"] == 45
    assert "preBlocks" in payload
    assert "postBlocks" in payload


def test_describe_action_denivele():
    action = {
        "kind": "run",
        "sec": 90,
        "targetType": "denivele",
        "targetLabel": "+200",
        "ascendM": 20,
        "descendM": 5,
    }
    text = describe_action(action)
    assert "Run" in text
    assert "+20 m" in text
    assert "-5 m" in text
