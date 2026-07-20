"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "synthesize_trail_digital_twin_sessions.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "synthesize_trail_digital_twin_sessions",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_session_frame_from_boundary_best_profile() -> None:
    synthesize = _load_module()
    sessions = synthesize.build_session_frame(
        exp_dir=REPO_ROOT / "data" / "exp_perf_predictions",
        session_source="trail_digital_twin_boundary_best_profile",
        activities_path=REPO_ROOT / "data" / "activities.csv",
    )
    assert not sessions.empty
    assert {"activityId", "cohort", "actualMin", "predictedMin", "errorMin", "absErrorMin"}.issubset(
        sessions.columns
    )
    assert sessions["stage"].eq(synthesize.STAGE3_LOO).all()
    summary = synthesize._cohort_summary(sessions)
    assert set(summary["cohort"]) >= {"hardTrailRun", "hardRunOrTrailRun"}


def test_render_markdown_includes_useful_elements() -> None:
    synthesize = _load_module()
    sessions = pd.DataFrame(
        {
            "activityId": ["1", "2"],
            "cohort": ["hardTrailRun", "hardTrailRun"],
            "actualMin": [100.0, 120.0],
            "predictedMin": [110.0, 115.0],
            "errorMin": [10.0, -5.0],
            "absErrorMin": [10.0, 5.0],
            "errorPct": [10.0, -4.0],
            "alpha": [0.9, 0.9],
            "fatigueCoef": [0.3, 0.3],
            "name": ["A", "B"],
        }
    )
    useful = {
        "experiment": "trail_digital_twin_hypothesis_refined",
        "winner": {
            "runId": "000",
            "meanStage3MaeMin": 10.84,
            "meanStage3MapePct": 7.91,
            "meanStage3R2": 0.982,
            "hrrReference": 0.88,
            "hrrMinFactor": 0.3,
            "hrrMaxFactor": 1.0,
            "decayLambda": 0.2,
            "minFatigueFactor": 0.6,
        },
        "cohortMae": [{"cohort": "hardTrailRun", "maeMin": 19.0, "mapePct": 12.0, "r2": 0.97, "biasMin": 1.0}],
        "fittedParams": [
            {
                "cohort": "hardTrailRun",
                "alpha": 0.9,
                "fatigueCoef": 0.6,
                "fatigueModel": "exponential",
                "fatigueState": "decayed",
                "secondaryFatigueCoef": 0.0,
            }
        ],
        "hardTrailStrata": [],
        "hardTrailTerrain": [],
    }
    markdown = synthesize.render_markdown(
        sessions=sessions,
        cohort_summary=synthesize._cohort_summary(sessions),
        useful=useful,
        session_source="trail_digital_twin_boundary_best_profile",
    )
    assert "Most Useful Elements" in markdown
    assert "0.88" in markdown
    assert "Segment stationary exclusion" in markdown
