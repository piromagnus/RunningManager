"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Tests for intensity-conditioned claim validation helpers.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import validate_intensity_conditioned_claim as validate


def test_paired_mape_bootstrap_positive_when_b_better() -> None:
    actual = np.array([3600.0, 4000.0, 5000.0, 4200.0])
    pred_a = actual * 1.20
    pred_b = actual * 1.05
    out = validate._paired_mape_bootstrap(actual, pred_a, pred_b, n_boot=500, seed=1)
    assert out["deltaMapePct"] > 0
    assert out["ciLow"] > 0
    assert out["relativeGainVsA"] > 0


def test_select_history_hrr_rules() -> None:
    history = pd.DataFrame(
        {
            "hrReserveRatio": [0.70, 0.75, 0.80, 0.60],
            "actualTimeSec": [3600, 7200, 5400, 1800],
        }
    )
    fixed = validate._select_history_hrr("fixed_hrr_reference", history, hrr_reference=0.88)
    assert fixed == 0.88
    median = validate._select_history_hrr("median_prior_hard_hrr", history, hrr_reference=0.88)
    assert 0.60 <= median <= 0.80
    nearest = validate._select_history_hrr(
        "nearest_prior_by_duration",
        history,
        hrr_reference=0.88,
        target_duration_h=1.9,
    )
    assert abs(nearest - 0.75) < 1e-9


def test_route_for_prescribed_prediction_strips_leakage() -> None:
    segments = pd.DataFrame(
        {
            "segmentIndex": [0, 1],
            "startKm": [0.0, 1.0],
            "meanHrReserve": [0.8, 0.7],
            "actualTimeSec": [400.0, 500.0],
            "cumTrimpBefore": [0.0, 1.0],
            "progress": [0.25, 0.75],
        }
    )
    route = validate._route_for_prescribed_prediction(segments)
    assert route["meanHrReserve"].isna().all()
    assert route["actualTimeSec"].isna().all()
    assert route["cumTrimpBefore"].isna().all()


def test_e2_uses_fixed_baseline_with_lowest_mape(tmp_path) -> None:
    summary = pd.DataFrame(
        {
            "cohort": ["hardTrailRun"] * 5,
            "modelId": ["B0", "B1", "B3", "B3_vs_B0", "B3_vs_B1"],
            "mapePct": [20, 40, 15, 5, 25],
            "relativeGainVsBaseline": [np.nan, np.nan, np.nan, 0.25, 0.625],
            "bootstrapCiLow": [np.nan, np.nan, np.nan, 1, 20],
            "bootstrapCiHigh": [np.nan, np.nan, np.nan, 9, 30],
        }
    )
    gates = validate.evaluate_claim_gates(
        SimpleNamespace(config={}, output_dir=tmp_path),
        e2_summary=summary,
        e4_summary=pd.DataFrame(),
        e5_summary=pd.DataFrame(columns=["cohort", "variant"]),
        e1_summary=pd.DataFrame(),
        e3_summary=pd.DataFrame(),
    )
    gate = gates.set_index("claimId").loc["prescribed_hrr_beats_fixed"]
    assert gate["status"] == "pass"
    assert gate["value"] == pytest.approx(0.25)
    assert gate["ciLow"] == pytest.approx(1)


def test_e3_concordance_excludes_small_hrr_differences(tmp_path) -> None:
    support = pd.DataFrame(
        {
            "activityId": ["a", "b", "c"],
            "cohort_hardTrailRun": [True] * 3,
            "distanceKm": [10.0] * 3,
            "ascentM": [400] * 3,
            "movingSec": [1000, 999, 900],
            "hrReserveRatio": [0.60, 0.61, 0.70],
        }
    )
    validate.run_e3_route_pairs(SimpleNamespace(output_dir=tmp_path), support)
    summary = pd.read_csv(tmp_path / "table_route_pair_validation_summary.csv").iloc[0]
    assert summary["nPairs"] == 3
    assert summary["nMaterialHrrDiff"] == 2
    assert summary["concordanceAmongMaterialHrrDiff"] == pytest.approx(1.0)


def test_e4_readiness_reference_uses_strictly_earlier_activities(tmp_path, monkeypatch) -> None:
    activities = pd.DataFrame(
        {
            "activityId": ["first", "second", "race", "same_day", "future"],
            "startDate": pd.date_range("2025-01-01", periods=5),
            "category": ["TRAIL_RUN"] * 5,
            "actualTimeSec": [3600.0] * 5,
            "hrReserveRatio": [0.70] * 5,
            "trimpRediSlow": [10, 20, 30, 5_000, 10_000],
            "trimpRediBalance": [0] * 5,
            "rediReadinessFactor": [1.0] * 5,
        }
    )
    activities.loc[3, "startDate"] = activities.loc[2, "startDate"]
    segments = {
        activity: pd.DataFrame({"activityId": [activity], "segmentIndex": [0], "startKm": [0]})
        for activity in activities["activityId"]
    }
    config = {
        "physiology": {"hrr_reference": 0.88},
        "readiness": {
            "ctl_weight": 0.10,
            "tsb_weight": 0.10,
            "ctl_factor_min": 0.50,
            "ctl_factor_max": 2.00,
        },
        "claim_validation": {"bootstrap_iterations": 10},
    }
    calls = []

    def fit(train, observed, config):
        assert list(train["activityId"]) == ["first", "second"]
        assert set(observed) == {"first", "second"}
        assert list(train["rediReadinessFactor"]) == pytest.approx([29 / 30, 31 / 30])
        return {"alpha": 1.0, "fatigueCoef": 0.0, "fatigueModel": "exponential"}

    def simulate(route, *, load_factor, **kwargs):
        assert list(route["activityId"]) == ["race"]
        calls.append(load_factor)
        return 3600.0

    monkeypatch.setattr(validate, "_fit_stage3_on_train", fit)
    monkeypatch.setattr(validate, "_simulate_prescribed", simulate)
    monkeypatch.setattr(validate, "_select_history_hrr", lambda *args, **kwargs: 0.70)
    bundle = SimpleNamespace(
        config=config,
        activity_df=activities,
        segments_by_activity=segments,
        event_labels=pd.DataFrame({"activityId": ["race"], "eventLabel": ["race"]}),
        output_dir=tmp_path,
    )
    validate.run_e4_rolling_origin(bundle)
    assert calls == pytest.approx([1.10] * 5)
