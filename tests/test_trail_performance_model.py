"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from services import trail_performance_model as model


def test_minetti_cost_is_clamped_and_gap_flat_is_one() -> None:
    assert model.minetti_running_cost(0.8) == pytest.approx(model.minetti_running_cost(0.75))
    assert model.minetti_running_cost(-0.8) == pytest.approx(model.minetti_running_cost(-0.75))
    assert model.minetti_running_cost(0.6) > model.minetti_running_cost(0.45)
    assert model.gap_factor(-0.75) == pytest.approx(0.1)
    assert model.gap_factor(0.0) == pytest.approx(1.0)
    assert model.gap_factor(0.2) > 1.0


def test_trail_gap_multiplier_is_asymmetric_on_steep_grades() -> None:
    assert model.trail_gap_multiplier(0.20, climb_scale=0.85, descent_scale=1.60) == pytest.approx(0.85)
    assert model.trail_gap_multiplier(-0.20, climb_scale=0.85, descent_scale=1.60) == pytest.approx(1.60)
    assert model.trail_gap_multiplier(0.0, climb_scale=0.85, descent_scale=1.60) == pytest.approx(1.0)
    # Soft ramp: midway between soft_start=0.04 and steep=0.15
    mid = model.trail_gap_multiplier(
        0.095,
        steep_threshold=0.15,
        soft_start=0.04,
        climb_scale=0.85,
        descent_scale=1.60,
    )
    assert mid == pytest.approx(0.925, abs=1e-3)
    scaled = model.apply_trail_gap_multipliers(
        np.array([2.0, 0.6, 1.0]),
        np.array([0.20, -0.20, 0.0]),
        climb_scale=0.85,
        descent_scale=1.60,
    )
    assert scaled[0] == pytest.approx(1.7)
    assert scaled[1] == pytest.approx(0.96)
    assert scaled[2] == pytest.approx(1.0)


def test_altitude_factor_decreases_with_altitude() -> None:
    sea_level = model.altitude_factor(0.0)
    mid_altitude = model.altitude_factor(1_000.0)
    high_altitude = model.altitude_factor(2_000.0)

    assert sea_level == pytest.approx(1.0)
    assert sea_level > mid_altitude > high_altitude


def test_fatigue_modifiers_reduce_speed_for_negative_decay() -> None:
    assert model.linear_decay_factor(0.0, -0.2) == pytest.approx(1.0)
    assert model.linear_decay_factor(1.0, -0.2) == pytest.approx(0.8)
    assert model.exponential_decay_factor(1.0, -0.2) == pytest.approx(math.exp(-0.2))


def test_negative_mu_increases_late_segment_time() -> None:
    segments = pd.DataFrame(
        {
            "distanceKm": [1.0, 1.0],
            "avgGrade": [0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0],
            "progress": [0.0, 1.0],
        }
    )

    neutral = model.predict_segment_times(segments, v_vt2_kmh=10.0, alpha=1.0, mu=0.0)
    decayed = model.predict_segment_times(segments, v_vt2_kmh=10.0, alpha=1.0, mu=-0.2)

    assert decayed.iloc[0] == pytest.approx(neutral.iloc[0])
    assert decayed.iloc[1] > neutral.iloc[1]


def test_speed_form_matches_paper_pace_equation() -> None:
    segments = pd.DataFrame(
        {
            "distanceKm": [2.0],
            "avgGrade": [0.10],
            "meanAltitudeM": [1_000.0],
            "progress": [0.5],
        }
    )
    alpha = 0.8
    mu = -0.1
    v_vt2 = 12.0
    ctl = 0.95
    predicted = model.predict_segment_times(
        segments,
        v_vt2_kmh=v_vt2,
        alpha=alpha,
        mu=mu,
        ctl_factor=ctl,
    ).iloc[0]
    gap = model.gap_factor(0.10)
    altitude = model.altitude_factor(1_000.0)
    fatigue = model.linear_decay_factor(0.5, mu)
    expected = 2.0 * 3600.0 * gap / (v_vt2 * alpha * altitude * ctl * fatigue)

    assert predicted == pytest.approx(expected)


def test_ctl_readiness_factor_is_bounded_and_directional() -> None:
    neutral = model.ctl_readiness_factor(ctl=1.0, tsb=0.0, ctl_reference=1.0)
    fresh_fit = model.ctl_readiness_factor(ctl=1.2, tsb=0.2, ctl_reference=1.0)
    stale = model.ctl_readiness_factor(ctl=0.6, tsb=-0.4, ctl_reference=1.0)
    clipped = model.ctl_readiness_factor(
        ctl=5.0,
        tsb=5.0,
        ctl_reference=1.0,
        max_factor=1.05,
    )

    assert neutral == pytest.approx(1.0)
    assert fresh_fit > neutral
    assert stale < neutral
    assert clipped == pytest.approx(1.05)


def test_segment_timeseries_preserves_distance_and_time() -> None:
    df = pd.DataFrame(
        {
            "cumulated_distance": np.linspace(0.0, 2.4, 25),
            "cumulated_duration_seconds": np.linspace(0.0, 720.0, 25),
            "elevationM_ma_5": np.linspace(100.0, 220.0, 25),
            "grade_ma_10": [0.05] * 25,
            "hr": [150.0] * 25,
            "speed_km_h": [12.0] * 25,
            "lat": np.linspace(45.0, 45.01, 25),
            "lon": np.linspace(5.0, 5.01, 25),
        }
    )

    segments = model.segment_timeseries(df, hr_rest=50.0, hr_max=200.0)

    assert segments["distanceKm"].sum() == pytest.approx(2.4)
    assert segments["actualTimeSec"].sum() == pytest.approx(720.0)
    assert segments["elevGainM"].sum() == pytest.approx(120.0)
    assert segments["meanHrReserve"].dropna().iloc[0] == pytest.approx((150.0 - 50.0) / 150.0)


def test_segment_timeseries_integrates_gap_for_mixed_climb_descent() -> None:
    df = pd.DataFrame(
        {
            "cumulated_distance": np.arange(1, 11, dtype=float) / 10.0,
            "cumulated_duration_seconds": np.arange(1, 11, dtype=float) * 60.0,
            "grade_ma_10": [0.10] * 5 + [-0.10] * 5,
        }
    )

    segments = model.segment_timeseries(df)

    assert len(segments) == 1
    segment = segments.iloc[0]
    expected_gap = 0.5 * (model.gap_factor(0.10) + model.gap_factor(-0.10))
    assert segment["avgGrade"] == pytest.approx(0.0)
    assert segment["gapFactorAvgGrade"] == pytest.approx(1.0)
    assert segment["gapFactorIntegrated"] == pytest.approx(expected_gap)
    assert segment["gapFactorIntegrated"] > segment["gapFactorAvgGrade"]
    assert segment["climbShare"] == pytest.approx(0.5)
    assert segment["descentShare"] == pytest.approx(0.5)
    assert segment["gradeSwitchCount"] == 1
    assert bool(segment["isMixedClimbDescent"]) is True
    assert segment["terrainFamily"] == "mixed_climb_descent"


def test_segment_timeseries_filters_precomputed_grade_outlier() -> None:
    df = pd.DataFrame(
        {
            "cumulated_distance": np.arange(1, 6, dtype=float) / 10.0,
            "cumulated_duration_seconds": np.arange(1, 6, dtype=float) * 60.0,
            "grade_ma_10": [0.10, 0.10, 2.00, 0.10, 0.10],
        }
    )

    segment = model.segment_timeseries(df).iloc[0]

    assert segment["avgGrade"] == pytest.approx(0.10)
    assert segment["absGradeMean"] == pytest.approx(0.10)
    assert segment["gapFactorIntegrated"] == pytest.approx(model.gap_factor(0.10))


def test_prediction_prefers_integrated_gap_over_average_grade() -> None:
    segments = pd.DataFrame(
        {
            "distanceKm": [1.0],
            "avgGrade": [0.0],
            "gapFactorIntegrated": [1.25],
            "meanAltitudeM": [0.0],
            "progress": [0.0],
        }
    )

    integrated = model.predict_segment_times(segments, v_vt2_kmh=12.0, alpha=1.0).iloc[0]
    average_grade = model.predict_segment_times(
        segments.drop(columns=["gapFactorIntegrated"]),
        v_vt2_kmh=12.0,
        alpha=1.0,
    ).iloc[0]

    assert integrated == pytest.approx(average_grade * 1.25)
    assert integrated > average_grade


def test_prepare_raw_timeseries_for_segments_builds_segment_inputs() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="1s", tz="UTC"),
            "lat": [45.0, 45.00001, 45.00002, 45.00003, 45.00004],
            "lon": [5.0, 5.00001, 5.00002, 5.00003, 5.00004],
            "elevationM": [100.0, 101.0, 102.0, 103.0, 104.0],
            "hr": [120, 122, 124, 126, 128],
        }
    )

    prepared = model.prepare_raw_timeseries_for_segments(raw)

    assert not prepared.empty
    assert {"cumulated_distance", "cumulated_duration_seconds", "elevationM_ma_5"}.issubset(
        prepared.columns
    )
    assert prepared["cumulated_distance"].is_monotonic_increasing
    assert prepared["cumulated_duration_seconds"].iloc[-1] == pytest.approx(4.0)


def test_prepare_raw_timeseries_interpolates_extreme_grade_outlier() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="2s", tz="UTC"),
            "lat": [45.0, 45.0001, 45.0002, 45.0003, 45.0004],
            "lon": [5.0] * 5,
            "elevationM": [100.0, 101.0, 200.0, 103.0, 104.0],
        }
    )

    prepared = model.prepare_raw_timeseries_for_segments(
        raw,
        elevation_window=1,
        grade_window=1,
    )

    assert not prepared.empty
    assert prepared["grade_ma_10"].abs().max() < 0.2


def test_hard_trailrun_filter_includes_known_races_and_excludes_short_spike() -> None:
    df = pd.DataFrame(
        {
            "activityId": ["16325125849", "17481444994", "short"],
            "category": ["TRAIL_RUN", "TRAIL_RUN", "TRAIL_RUN"],
            "hasTimeseries": [True, True, True],
            "movingSec": [10_638, 12_779, 600],
            "distanceKm": [28.9, 30.9, 1.5],
            "ascentM": [1008.0, 1616.0, 20.0],
            "hrReserveRatio": [0.79, 0.73, 0.95],
        }
    )

    mask = model.hard_trailrun_mask(df)

    assert mask.tolist() == [True, True, False]


def test_top_hrr_hard_trailrun_ids_uses_hard_trailruns() -> None:
    df = pd.DataFrame(
        {
            "activityId": ["easy", "hard1", "run", "hard2"],
            "category": ["TRAIL_RUN", "TRAIL_RUN", "RUN", "TRAIL_RUN"],
            "hardTrailRun": [False, True, True, True],
            "usableTrailRun": [True, True, True, True],
            "hrReserveRatio": [0.95, 0.82, 0.99, 0.88],
            "distanceKm": [5.0, 20.0, 10.0, 12.0],
        }
    )

    assert model.top_hrr_hard_trailrun_ids(df, n=2) == ["hard2", "hard1"]


def test_select_best_activity_by_dates_prefers_trail_then_run() -> None:
    df = pd.DataFrame(
        {
            "activityId": ["run_long", "trail_short", "ride"],
            "startDate": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]).date,
            "category": ["RUN", "TRAIL_RUN", "RIDE"],
            "hasTimeseries": [True, True, True],
            "distanceKm": [30.0, 5.0, 100.0],
            "ascentM": [100.0, 100.0, 1_000.0],
            "movingSec": [10_000.0, 2_000.0, 20_000.0],
        }
    )

    selected = model.select_best_activity_by_dates(df, ["2026-01-01", "2026-01-02"])

    assert selected["activityId"].tolist() == ["trail_short", "ride"]


def test_grid_search_recovers_synthetic_alpha_and_mu() -> None:
    base_segments = pd.DataFrame(
        {
            "distanceKm": [1.0, 1.0, 1.0],
            "avgGrade": [0.0, 0.08, -0.05],
            "meanAltitudeM": [200.0, 400.0, 300.0],
            "progress": [0.15, 0.5, 0.85],
        }
    )
    segments_by_activity = {
        "a": base_segments,
        "b": base_segments.assign(distanceKm=[1.2, 0.8, 1.5], avgGrade=[0.03, 0.12, -0.02]),
        "c": base_segments.assign(distanceKm=[0.6, 1.4, 2.0], meanAltitudeM=[800.0, 900.0, 700.0]),
    }
    observed = {
        activity_id: model.predict_activity_time(
            segments, v_vt2_kmh=12.0, alpha=0.90, mu=-0.10
        )
        for activity_id, segments in segments_by_activity.items()
    }

    best, _grid = model.grid_search_model(
        segments_by_activity,
        observed,
        v_vt2_kmh=12.0,
        alpha_grid=[0.86, 0.88, 0.90, 0.92],
        mu_grid=[-0.16, -0.12, -0.10, -0.08, 0.0],
    )

    assert best["alpha"] == pytest.approx(0.90)
    assert best["mu"] == pytest.approx(-0.10)
    assert best["maeSec"] == pytest.approx(0.0)


def test_previous_daily_feature_join_does_not_leak_same_day() -> None:
    activities = pd.DataFrame(
        {
            "activityId": ["a", "b"],
            "startDate": pd.to_datetime(["2026-01-02", "2026-01-03"]).date,
        }
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "loadFeature": [1.0, 2.0, 3.0],
        }
    )

    joined = model.attach_previous_daily_features(activities, daily)

    assert joined.set_index("activityId").loc["a", "loadFeature"] == pytest.approx(1.0)
    assert joined.set_index("activityId").loc["b", "loadFeature"] == pytest.approx(2.0)


def test_redi_load_features_use_slow_fast_balance() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="D"),
            "trimp": [0.0, 10.0, 0.0, 0.0],
        }
    )

    features = model.compute_redi_load_features(daily, slow_lam=0.01, fast_lam=0.50)

    assert {"trimpRediSlow", "trimpRediFast", "trimpRediBalance"}.issubset(features.columns)
    assert features["trimpRediFast"].iloc[1] > features["trimpRediSlow"].iloc[1]
    assert features["trimpRediSlow"].iloc[-1] > features["trimpRediFast"].iloc[-1]


def test_in_activity_trimp_features_are_cumulative_and_lagged() -> None:
    segments = pd.DataFrame(
        {
            "activityId": ["a", "a"],
            "segmentIndex": [0, 1],
            "actualTimeSec": [600.0, 600.0],
            "meanHrReserve": [0.5, 0.6],
        }
    )

    enriched = model.add_in_activity_trimp_features(segments, decay_lambda=0.2)

    assert enriched.loc[enriched["segmentIndex"].eq(0), "cumTrimpBefore"].iloc[0] == 0.0
    assert enriched.loc[enriched["segmentIndex"].eq(1), "cumTrimpBefore"].iloc[0] > 0.0
    assert enriched["cumTrimp"].iloc[-1] > enriched["cumTrimpBefore"].iloc[-1]


def test_segment_grid_search_recovers_synthetic_parameters_and_race_metrics() -> None:
    segments = pd.DataFrame(
        {
            "activityId": ["a", "a", "b", "b"],
            "segmentIndex": [0, 1, 0, 1],
            "distanceKm": [1.0, 1.0, 1.0, 1.0],
            "avgGrade": [0.0, 0.08, 0.0, 0.08],
            "meanAltitudeM": [0.0, 0.0, 0.0, 0.0],
            "progress": [0.25, 0.75, 0.25, 0.75],
            "terrainFamily": ["flat", "climb", "flat", "climb"],
            "technicalityGps": [0.0, 0.2, 0.0, 0.2],
            "meanHrReserve": [0.7, 0.8, 0.7, 0.8],
            "decayedTrimpBefore": [0.0, 0.2, 0.0, 0.2],
        }
    )
    true_terrain = {"flat": 1.0, "climb": 0.8}
    segments["actualTimeSec"] = model.predict_extension_segment_times(
        segments,
        v_anchor_kmh=12.0,
        alpha=0.9,
        mu=-0.1,
        terrain_multipliers=true_terrain,
        technicality_coef=0.2,
        hr_coef=0.3,
        acute_trimp_coef=0.1,
    )

    best, grid, predicted = model.segment_grid_search_model(
        segments,
        v_anchor_kmh=12.0,
        alpha_grid=[0.85, 0.9],
        mu_grid=[-0.1, 0.0],
        terrain_multiplier_grid=[{"flat": 1.0, "climb": 1.0}, true_terrain],
        technicality_coef_grid=[0.0, 0.2],
        hr_coef_grid=[0.0, 0.3],
        acute_trimp_coef_grid=[0.0, 0.1],
    )

    assert best["alpha"] == pytest.approx(0.9)
    assert best["mu"] == pytest.approx(-0.1)
    assert best["terrainMultipliers"] == true_terrain
    assert best["raceMaeSec"] == pytest.approx(0.0)
    assert "raceR2" in grid.columns
    assert predicted["predictedTimeSec"].equals(segments["actualTimeSec"])


def test_hrr_trimp_prediction_uses_linear_hrr_and_trimp_fatigue() -> None:
    segments = pd.DataFrame(
        {
            "distanceKm": [1.0, 1.0, 1.0],
            "avgGrade": [0.0, 0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0, 0.0],
            "meanHrReserve": [0.70, 0.84, 0.70],
            "decayedTrimpBefore": [0.0, 0.0, 10.0],
        }
    )

    predicted = model.predict_hrr_trimp_segment_times(
        segments,
        v_anchor_kmh=12.0,
        alpha=1.0,
        fatigue_coef=0.2,
        fatigue_model="linear",
        trimp_scale=10.0,
    )

    assert predicted.iloc[1] < predicted.iloc[0]
    assert predicted.iloc[2] > predicted.iloc[0]


def test_hrr_trimp_prediction_ignores_deprecated_trimp_scale() -> None:
    segments = pd.DataFrame(
        {
            "distanceKm": [1.0, 1.0],
            "avgGrade": [0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0],
            "meanHrReserve": [0.70, 0.70],
            "decayedTrimpBefore": [0.0, 2.0],
        }
    )

    scale_one = model.predict_hrr_trimp_segment_times(
        segments,
        v_anchor_kmh=12.0,
        alpha=1.0,
        fatigue_coef=0.2,
        fatigue_model="linear",
        trimp_scale=1.0,
    )
    scale_large = model.predict_hrr_trimp_segment_times(
        segments,
        v_anchor_kmh=12.0,
        alpha=1.0,
        fatigue_coef=0.2,
        fatigue_model="linear",
        trimp_scale=100.0,
    )

    pd.testing.assert_series_equal(scale_one, scale_large)


def test_hrr_trimp_prediction_can_use_linear_progress_fatigue() -> None:
    segments = pd.DataFrame(
        {
            "distanceKm": [1.0, 1.0],
            "avgGrade": [0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0],
            "meanHrReserve": [0.70, 0.70],
            "progress": [0.0, 1.0],
            "decayedTrimpBefore": [0.0, 0.0],
        }
    )

    predicted = model.predict_hrr_trimp_segment_times(
        segments,
        v_anchor_kmh=12.0,
        alpha=1.0,
        fatigue_coef=0.2,
        fatigue_model="linear",
        acute_trimp_col="progress",
    )

    assert predicted.iloc[1] > predicted.iloc[0]


def test_hrr_trimp_prediction_can_combine_short_and_muscular_fatigue() -> None:
    segments = pd.DataFrame(
        {
            "distanceKm": [1.0, 1.0],
            "avgGrade": [0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0],
            "meanHrReserve": [0.70, 0.70],
            "decayedTrimpBefore": [0.0, 1.0],
            "cumTrimpBefore": [0.0, 2.0],
        }
    )

    short_only = model.predict_hrr_trimp_segment_times(
        segments,
        v_anchor_kmh=12.0,
        alpha=1.0,
        fatigue_coef=0.2,
        fatigue_model="exponential",
    )
    combined = model.predict_hrr_trimp_segment_times(
        segments,
        v_anchor_kmh=12.0,
        alpha=1.0,
        fatigue_coef=0.2,
        fatigue_model="exponential",
        secondary_fatigue_coef=0.3,
        secondary_acute_trimp_col="cumTrimpBefore",
        secondary_fatigue_model="exponential",
    )

    assert combined.iloc[0] == pytest.approx(short_only.iloc[0])
    assert combined.iloc[1] > short_only.iloc[1]


def test_hrr_trimp_prediction_can_disable_hrr_effort() -> None:
    segments = pd.DataFrame(
        {
            "distanceKm": [1.0, 1.0],
            "avgGrade": [0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0],
            "meanHrReserve": [0.50, 0.90],
            "decayedTrimpBefore": [0.0, 0.0],
        }
    )

    predicted = model.predict_hrr_trimp_segment_times(
        segments,
        v_anchor_kmh=12.0,
        alpha=1.0,
        use_hrr_effort=False,
    )

    assert predicted.iloc[0] == pytest.approx(predicted.iloc[1])


def test_hrr_duration_envelope_reports_max_time_by_range() -> None:
    activities = pd.DataFrame(
        {
            "category": ["RUN", "TRAIL_RUN", "RIDE", "RUN"],
            "avgHr": [145.0, 160.0, 170.0, 175.0],
            "timeSec": [1800.0, 7200.0, 10_000.0, 3600.0],
        }
    )

    envelope = model.estimate_hrr_duration_envelope(
        activities,
        hr_rest=55.0,
        hr_max=205.0,
        hrr_min=0.55,
        hrr_max=0.85,
        bin_width=0.10,
    )

    assert model.max_duration_for_hrr(0.60, envelope) == pytest.approx(1800.0)
    assert model.max_duration_for_hrr(0.70, envelope) == pytest.approx(7200.0)
    assert math.isnan(model.max_duration_for_hrr(0.90, envelope))


def test_hrr_duration_power_law_fits_decreasing_sustainable_hrr() -> None:
    activities = pd.DataFrame(
        {
            "category": ["RUN"] * 5,
            "timeSec": [600.0, 1800.0, 3600.0, 7200.0, 14_400.0],
            "hrReserveRatio": [0.92, 0.86, 0.80, 0.72, 0.64],
        }
    )

    params, windows = model.estimate_hrr_duration_power_law(
        activities,
        duration_windows_min=[5, 10, 30, 60, 120, 240],
        min_activity_count=1,
    )

    assert params["fitWindowCount"] == 6
    assert float(params["exponent"]) < 0.0
    assert windows["targetHrr"].dropna().is_monotonic_decreasing
    assert model.hrr_for_duration_power_law(600.0, params) > model.hrr_for_duration_power_law(
        14_400.0,
        params,
    )
    assert "fitWeight" in windows.columns


def test_hrr_duration_power_law_performance_weights_best_frontier_more() -> None:
    activities = pd.DataFrame(
        {
            "category": ["RUN"] * 5,
            "timeSec": [600.0, 1800.0, 3600.0, 7200.0, 14_400.0],
            "hrReserveRatio": [0.95, 0.80, 0.72, 0.64, 0.58],
        }
    )

    _, uniform = model.estimate_hrr_duration_power_law(
        activities,
        duration_windows_min=[10, 30, 60, 120, 240],
        min_activity_count=1,
        fit_weight_mode="uniform",
    )
    params, weighted = model.estimate_hrr_duration_power_law(
        activities,
        duration_windows_min=[10, 30, 60, 120, 240],
        min_activity_count=1,
        fit_weight_mode="performance",
        fit_weight_power=6.0,
    )

    uniform_first_error = abs(uniform["residualHrr"].iloc[0])
    weighted_first_error = abs(weighted["residualHrr"].iloc[0])
    assert weighted_first_error < uniform_first_error
    assert weighted["fitWeight"].iloc[0] > weighted["fitWeight"].iloc[-1]
    assert params["fitWeightMode"] == "performance"


def test_hrr_duration_power_law_inverse_shortens_high_hrr() -> None:
    params = {
        "coefficient": 0.80,
        "exponent": -0.10,
        "minWindowSec": 300.0,
        "maxWindowSec": 86_400.0,
        "hrrMin": 0.30,
        "hrrMax": 0.98,
    }

    assert model.max_duration_for_hrr_power_law(
        0.90,
        params,
    ) < model.max_duration_for_hrr_power_law(0.70, params)


def test_route_segments_from_points_removes_pseudo_observed_time() -> None:
    points = pd.DataFrame(
        {
            "lat": np.linspace(45.0, 45.05, 150),
            "lon": np.linspace(5.0, 5.05, 150),
            "elevationM": np.linspace(100.0, 200.0, 150),
            "timestamp": pd.date_range("2026-01-01", periods=150, freq="1s", tz="UTC"),
        }
    )

    segments = model.route_segments_from_points(points, segment_km=0.5)
    raw_distance_km = model.prepare_raw_timeseries_for_segments(
        points.drop(columns=["timestamp"])
    )["distance"].sum()

    assert not segments.empty
    assert segments["distanceKm"].sum() == pytest.approx(raw_distance_km)
    assert segments["actualTimeSec"].isna().all()
    assert {"avgGrade", "meanAltitudeM", "progress"}.issubset(segments.columns)


def test_constant_hrr_route_simulation_accumulates_predicted_trimp() -> None:
    segments = pd.DataFrame(
        {
            "segmentIndex": [0, 1],
            "startKm": [0.0, 1.0],
            "distanceKm": [1.0, 1.0],
            "avgGrade": [0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0],
        }
    )

    predicted = model.simulate_constant_hrr_route(
        segments,
        hrr=0.70,
        v_anchor_kmh=10.0,
        alpha=1.0,
        fatigue_coef=0.30,
        fatigue_model="linear",
        trimp_scale=1.0,
    )

    assert predicted["cumTrimpBefore"].iloc[0] == pytest.approx(0.0)
    assert predicted["cumTrimpBefore"].iloc[1] > 0.0
    assert predicted["predictedFatigueStateBefore"].iloc[0] == pytest.approx(1.0)
    assert predicted["predictedFatigueStateBefore"].iloc[1] < 1.0
    assert predicted["predictedFatigueState"].iloc[0] <= predicted["predictedFatigueStateBefore"].iloc[0]
    assert predicted["predictedTimeSec"].iloc[1] > predicted["predictedTimeSec"].iloc[0]


def test_constant_hrr_route_uses_cumulative_fatigue_state_for_fixed_effort() -> None:
    segments = pd.DataFrame(
        {
            "segmentIndex": [0, 1, 2],
            "startKm": [0.0, 1.0, 2.0],
            "distanceKm": [1.0, 1.0, 1.0],
            "avgGrade": [0.0, 0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0, 0.0],
        }
    )

    predicted = model.simulate_constant_hrr_route(
        segments,
        hrr=0.70,
        v_anchor_kmh=10.0,
        alpha=1.0,
        fatigue_coef=0.20,
        trimp_scale=1.0,
        decay_lambda=100.0,
    )
    third_segment = predicted.iloc[[2]].copy()
    cumulative_time = model.predict_hrr_trimp_segment_times(
        third_segment,
        v_anchor_kmh=10.0,
        alpha=1.0,
        fatigue_coef=0.20,
        trimp_scale=1.0,
        acute_trimp_col="cumTrimpBefore",
        load_factor_col="_preRaceLoadFactor",
    ).iloc[0]
    decayed_time = model.predict_hrr_trimp_segment_times(
        third_segment,
        v_anchor_kmh=10.0,
        alpha=1.0,
        fatigue_coef=0.20,
        trimp_scale=1.0,
        acute_trimp_col="decayedTrimpBefore",
        load_factor_col="_preRaceLoadFactor",
    ).iloc[0]

    assert predicted["predictedTimeSec"].is_monotonic_increasing
    assert predicted["predictedTimeSec"].iloc[2] == pytest.approx(cumulative_time)
    assert cumulative_time > decayed_time


def test_observed_hrr_segments_use_true_hrr_without_actual_time_trimp_leak() -> None:
    segments = pd.DataFrame(
        {
            "segmentIndex": [0, 1],
            "startKm": [0.0, 1.0],
            "distanceKm": [1.0, 1.0],
            "avgGrade": [0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0],
            "meanHrReserve": [0.70, 0.84],
            "actualTimeSec": [9999.0, 9999.0],
        }
    )

    predicted = model.simulate_observed_hrr_segments(
        segments,
        v_anchor_kmh=10.0,
        alpha=1.0,
        fatigue_coef=0.0,
    )

    assert predicted["cumTrimpBefore"].iloc[0] == pytest.approx(0.0)
    assert predicted["cumTrimpBefore"].iloc[1] < 1.0
    assert predicted["predictedTimeSec"].iloc[1] < predicted["predictedTimeSec"].iloc[0]
    assert predicted["errorSec"].notna().all()


def test_observed_hrr_segments_use_cumulative_fatigue_state_for_fixed_effort() -> None:
    segments = pd.DataFrame(
        {
            "segmentIndex": [0, 1, 2],
            "startKm": [0.0, 1.0, 2.0],
            "distanceKm": [1.0, 1.0, 1.0],
            "avgGrade": [0.0, 0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0, 0.0],
            "meanHrReserve": [0.70, 0.70, 0.70],
            "actualTimeSec": [300.0, 300.0, 300.0],
        }
    )

    predicted = model.simulate_observed_hrr_segments(
        segments,
        v_anchor_kmh=10.0,
        alpha=1.0,
        fatigue_coef=0.20,
        trimp_scale=1.0,
        decay_lambda=100.0,
    )
    third_segment = predicted.iloc[[2]].copy()
    cumulative_time = model.predict_hrr_trimp_segment_times(
        third_segment,
        v_anchor_kmh=10.0,
        alpha=1.0,
        fatigue_coef=0.20,
        trimp_scale=1.0,
        acute_trimp_col="cumTrimpBefore",
        load_factor_col="_activityLoadFactor",
    ).iloc[0]
    decayed_time = model.predict_hrr_trimp_segment_times(
        third_segment,
        v_anchor_kmh=10.0,
        alpha=1.0,
        fatigue_coef=0.20,
        trimp_scale=1.0,
        acute_trimp_col="decayedTrimpBefore",
        load_factor_col="_activityLoadFactor",
    ).iloc[0]

    assert predicted["predictedTimeSec"].is_monotonic_increasing
    assert predicted["predictedTimeSec"].iloc[2] == pytest.approx(cumulative_time)
    assert cumulative_time > decayed_time


def test_observed_hrr_fatigue_state_starts_full_and_decreases() -> None:
    segments = pd.DataFrame(
        {
            "segmentIndex": [0, 1, 2],
            "startKm": [0.0, 1.0, 2.0],
            "distanceKm": [1.0, 1.0, 1.0],
            "avgGrade": [0.0, 0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0, 0.0],
            "meanHrReserve": [0.70, 0.60, 0.80],
            "actualTimeSec": [600.0, 600.0, 600.0],
        }
    )

    predicted = model.simulate_observed_hrr_segments(
        segments,
        v_anchor_kmh=10.0,
        alpha=1.0,
        fatigue_coef=0.30,
        trimp_scale=1.0,
    )
    state_before = predicted["predictedFatigueStateBefore"]

    assert state_before.iloc[0] == pytest.approx(1.0)
    assert state_before.between(0.50, 1.0).all()
    assert (state_before.diff().dropna() <= 0.0).all()
    assert predicted["predictedFatigueState"].iloc[-1] <= state_before.iloc[-1]


def test_constant_hrr_sweep_selects_fastest_feasible_ratio() -> None:
    segments = pd.DataFrame(
        {
            "segmentIndex": [0],
            "startKm": [0.0],
            "distanceKm": [10.0],
            "avgGrade": [0.0],
            "meanAltitudeM": [0.0],
        }
    )
    envelope = pd.DataFrame(
        {
            "hrrLower": [0.65, 0.75],
            "hrrUpper": [0.75, 0.85],
            "maxDurationSec": [4000.0, 3000.0],
        }
    )

    sweep = model.sweep_constant_hrr_route(
        segments,
        hrr_values=[0.70, 0.80],
        v_anchor_kmh=10.0,
        alpha=1.0,
        envelope_df=envelope,
    )
    best = model.select_best_constant_hrr(sweep)

    assert not bool(sweep.set_index("hrr").loc[0.80, "feasible"])
    assert best["hrr"] == pytest.approx(0.70)


def test_hrr_trimp_grid_search_recovers_synthetic_parameters_and_loo() -> None:
    segments = pd.DataFrame(
        {
            "activityId": ["a", "a", "b", "b", "c", "c"],
            "distanceKm": [1.0, 1.0, 1.2, 0.8, 0.9, 1.4],
            "avgGrade": [0.0, 0.08, 0.0, 0.08, -0.02, 0.12],
            "meanAltitudeM": [0.0, 200.0, 0.0, 200.0, 50.0, 250.0],
            "meanHrReserve": [0.70, 0.78, 0.68, 0.82, 0.72, 0.80],
            "decayedTrimpBefore": [0.0, 4.0, 0.0, 6.0, 0.0, 5.0],
        }
    )
    segments["actualTimeSec"] = model.predict_hrr_trimp_segment_times(
        segments,
        v_anchor_kmh=12.0,
        alpha=0.80,
        fatigue_coef=0.30,
        fatigue_model="linear",
        trimp_scale=10.0,
    )

    best, grid, prediction = model.hrr_trimp_grid_search_model(
        segments,
        v_anchor_kmh=12.0,
        alpha_grid=[0.70, 0.80, 0.90],
        fatigue_coef_grid=[0.0, 0.30],
        fatigue_models=("linear",),
        trimp_scale=10.0,
    )
    loo = model.leave_one_out_hrr_trimp_grid_search(
        segments,
        v_anchor_kmh=12.0,
        alpha_grid=[0.70, 0.80, 0.90],
        fatigue_coef_grid=[0.0, 0.30],
        fatigue_models=("linear",),
        trimp_scale=10.0,
    )

    assert best["alpha"] == pytest.approx(0.80)
    assert best["fatigueCoef"] == pytest.approx(0.30)
    assert best["raceMaeSec"] == pytest.approx(0.0)
    assert {"raceR2", "segmentR2", "fatigueModel"}.issubset(grid.columns)
    assert prediction["predictedTimeSec"].to_numpy() == pytest.approx(
        segments["actualTimeSec"].to_numpy()
    )
    assert len(loo) == 3
    assert {"alpha", "fatigueCoef", "predictedTimeSec", "errorSec"}.issubset(loo.columns)

    observed = segments.groupby("activityId")["actualTimeSec"].sum().add(60.0).to_dict()
    observed_best, _observed_grid, _observed_prediction = model.hrr_trimp_grid_search_model(
        segments,
        v_anchor_kmh=12.0,
        alpha_grid=[0.80],
        fatigue_coef_grid=[0.30],
        fatigue_models=("linear",),
        trimp_scale=10.0,
        observed_activity_times_sec=observed,
    )
    assert observed_best["raceMaeSec"] == pytest.approx(60.0)


def test_hrr_trimp_grid_search_recovers_secondary_fatigue_coef() -> None:
    segments = pd.DataFrame(
        {
            "activityId": ["a", "a", "b", "b"],
            "distanceKm": [1.0, 1.0, 1.0, 1.0],
            "avgGrade": [0.0, 0.0, 0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0, 0.0, 0.0],
            "meanHrReserve": [0.70, 0.70, 0.70, 0.70],
            "decayedTrimpBefore": [0.0, 1.0, 0.0, 1.0],
            "cumTrimpBefore": [0.0, 1.0, 0.0, 2.0],
        }
    )
    segments["actualTimeSec"] = model.predict_hrr_trimp_segment_times(
        segments,
        v_anchor_kmh=12.0,
        alpha=0.80,
        fatigue_coef=0.20,
        fatigue_model="exponential",
        secondary_fatigue_coef=0.30,
        secondary_acute_trimp_col="cumTrimpBefore",
        secondary_fatigue_model="exponential",
    )

    best, grid, prediction = model.hrr_trimp_grid_search_model(
        segments,
        v_anchor_kmh=12.0,
        alpha_grid=[0.80],
        fatigue_coef_grid=[0.20],
        secondary_fatigue_coef_grid=[0.0, 0.30],
        fatigue_models=("exponential",),
        secondary_acute_trimp_col="cumTrimpBefore",
        secondary_fatigue_model="exponential",
    )

    assert best["secondaryFatigueCoef"] == pytest.approx(0.30)
    assert best["raceMaeSec"] == pytest.approx(0.0)
    assert {"secondaryFatigueCoef", "secondaryAcuteTrimpCol"}.issubset(grid.columns)
    assert prediction["predictedTimeSec"].to_numpy() == pytest.approx(
        segments["actualTimeSec"].to_numpy()
    )


def test_anonymized_export_column_guard_flags_direct_identifiers() -> None:
    forbidden = model.forbidden_anonymized_columns(
        ["activityIndex", "activityId", "startTime", "distanceKm", "lat", "meanAltitudeM"]
    )

    assert forbidden == ["activityId", "startTime", "lat"]


def test_predict_many_applies_activity_ctl_factors() -> None:
    segments = pd.DataFrame(
        {
            "distanceKm": [1.0],
            "avgGrade": [0.0],
            "meanAltitudeM": [0.0],
            "progress": [0.0],
        }
    )

    predictions = model.predict_many(
        {"a": segments, "b": segments},
        v_vt2_kmh=10.0,
        alpha=1.0,
        ctl_factors={"a": 1.0, "b": 0.5},
    )

    assert predictions["b"] == pytest.approx(predictions["a"] * 2.0)


def test_hr_regression_keeps_low_submaximal_effort_rows() -> None:
    df = pd.DataFrame(
        {
            "distanceEqKm": [5.0, 8.0, 12.0, 15.0],
            "hrReserveRatio": [0.32, 0.45, 0.68, 0.82],
            "actualTimeSec": [2400.0, 3300.0, 4300.0, 5000.0],
        }
    )
    df["logDistanceEqKm"] = np.log(df["distanceEqKm"])
    df["logActualTimeSec"] = np.log(df["actualTimeSec"])

    fitted = model.fit_linear_regression(
        df,
        feature_cols=["logDistanceEqKm", "hrReserveRatio"],
        target_col="logActualTimeSec",
    )
    predicted = model.predict_linear_regression(df, fitted)

    assert len(predicted) == 4
    assert np.isfinite(predicted).all()


def test_segment_timeseries_records_stationary_share_for_idle_block() -> None:
    # Moving first half, then device-open idle with tiny GPS jitter.
    distances = [0.1, 0.2, 0.3, 0.4, 0.401, 0.402, 0.403, 0.404]
    durations = [60.0, 120.0, 180.0, 240.0, 540.0, 840.0, 1140.0, 1440.0]
    df = pd.DataFrame(
        {
            "cumulated_distance": distances,
            "cumulated_duration_seconds": durations,
            "grade_ma_10": [0.0] * len(distances),
            "hr": [140.0] * len(distances),
        }
    )

    segments = model.segment_timeseries(df, segment_km=1.0)
    assert len(segments) == 1
    assert segments.iloc[0]["stationaryTimeShare"] > 0.5
    assert segments.iloc[0]["meanSpeedKmh"] < 2.0


def test_segment_timeseries_keeps_zero_distance_dwell_time() -> None:
    # 0.5 km move, then 10 minutes stopped at the same distance, then finish the km.
    df = pd.DataFrame(
        {
            "cumulated_distance": [0.1, 0.3, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0],
            "cumulated_duration_seconds": [60.0, 120.0, 180.0, 330.0, 480.0, 780.0, 900.0, 1020.0],
            "grade_ma_10": [0.0] * 8,
            "hr": [150.0] * 8,
        }
    )
    segments = model.segment_timeseries(df, segment_km=1.0)
    assert len(segments) == 1
    assert segments.iloc[0]["distanceKm"] == pytest.approx(1.0)
    assert segments.iloc[0]["actualTimeSec"] == pytest.approx(1020.0)
    assert segments.iloc[0]["stationaryTimeSec"] >= 600.0
    assert segments.iloc[0]["stationaryTimeShare"] > 0.5


def test_apply_segment_exclusion_flags_low_speed_eq_and_high_stationary_share() -> None:
    segments = pd.DataFrame(
        {
            "distanceKm": [1.0, 1.0, 1.0, 1.0],
            "actualTimeSec": [360.0, 1800.0, 600.0, 1200.0],
            "meanSpeedKmh": [10.0, 1.5, 6.0, 2.0],
            "meanSpeedEqKmh": [10.0, 1.2, 6.0, 7.5],
            "stationaryTimeShare": [0.05, 0.20, 0.95, 0.10],
            # Flat on altitude-over-time (m of |Δelev| per clock hour).
            "absAltitudeRateMph": [40.0, 30.0, 10.0, 20.0],
        }
    )
    annotated = model.apply_segment_exclusion(
        segments,
        enabled=True,
        min_mean_speed_eq_kmh=3.0,
        max_stationary_time_share=0.80,
        max_abs_altitude_rate_mph=120.0,
    )
    assert annotated["isFitEligible"].tolist() == [True, False, False, True]
    assert "near_flat_altitude_time" in annotated.loc[1, "exclusionReason"]
    assert "low_mean_speed_eq" in annotated.loc[1, "exclusionReason"]
    assert "high_stationary_share" in annotated.loc[2, "exclusionReason"]
    assert annotated.loc[3, "exclusionReason"] == ""


def test_apply_segment_exclusion_keeps_climbing_altitude_time_profile() -> None:
    """Slow climbs must stay in the fit set when altitude rises over time."""
    segments = pd.DataFrame(
        {
            "distanceKm": [1.0, 1.0],
            "actualTimeSec": [1800.0, 1800.0],
            "meanSpeedKmh": [2.0, 1.5],
            "meanSpeedEqKmh": [8.0, 1.2],
            "stationaryTimeShare": [0.05, 0.50],
            # ~400–600 m/h vertical activity → rising altitude–time profile.
            "absAltitudeRateMph": [400.0, 600.0],
            "avgGrade": [0.02, 0.02],  # mild distance-grade must not matter
        }
    )
    annotated = model.apply_segment_exclusion(
        segments,
        enabled=True,
        min_mean_speed_eq_kmh=3.0,
        max_stationary_time_share=0.40,
        max_abs_altitude_rate_mph=120.0,
    )
    assert annotated["isFitEligible"].tolist() == [True, True]
    assert (annotated["exclusionReason"] == "").all()


def test_segment_timeseries_reports_altitude_rate_over_time() -> None:
    """absAltitudeRateMph is gross |Δelev| / clock hour, not distance grade."""
    # 1 km in 600 s with +50 m elev → 50 / 600 * 3600 = 300 m/h
    n = 11
    df = pd.DataFrame(
        {
            "cumulated_distance": np.linspace(0.0, 1.0, n),
            "cumulated_duration_seconds": np.linspace(0.0, 600.0, n),
            "elevationM_ma_5": np.linspace(0.0, 50.0, n),
            "grade_ma_10": np.full(n, 0.05),
            "speed_km_h": np.full(n, 6.0),
            "lat": np.linspace(45.0, 45.01, n),
            "lon": np.linspace(5.0, 5.01, n),
        }
    )
    segments = model.segment_timeseries(df, segment_km=1.0)
    assert not segments.empty
    assert segments.iloc[0]["absAltitudeRateMph"] == pytest.approx(300.0, rel=0.05)
    assert "netAltitudeRateMph" in segments.columns


def test_hrr_trimp_grid_search_optimizes_on_fit_mask_and_scores_full_race() -> None:
    segments = pd.DataFrame(
        {
            "activityId": ["a", "a", "a", "b", "b", "b"],
            "distanceKm": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "avgGrade": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "meanHrReserve": [0.70, 0.72, 0.70, 0.70, 0.74, 0.70],
            "decayedTrimpBefore": [0.0, 4.0, 8.0, 0.0, 5.0, 9.0],
            "isFitEligible": [True, True, False, True, True, False],
        }
    )
    clean = segments[segments["isFitEligible"]].copy()
    clean["actualTimeSec"] = model.predict_hrr_trimp_segment_times(
        clean,
        v_anchor_kmh=12.0,
        alpha=0.80,
        fatigue_coef=0.30,
        fatigue_model="linear",
        trimp_scale=10.0,
    )
    segments = segments.merge(
        clean[["activityId", "decayedTrimpBefore", "actualTimeSec"]],
        on=["activityId", "decayedTrimpBefore"],
        how="left",
        suffixes=("", "_clean"),
    )
    # Idle segments are much slower than the model would expect.
    segments.loc[~segments["isFitEligible"], "actualTimeSec"] = 1800.0

    best, grid, prediction = model.hrr_trimp_grid_search_model(
        segments,
        v_anchor_kmh=12.0,
        alpha_grid=[0.70, 0.80, 0.90],
        fatigue_coef_grid=[0.0, 0.30],
        fatigue_models=("linear",),
        trimp_scale=10.0,
        fit_mask_col="isFitEligible",
    )
    loo = model.leave_one_out_hrr_trimp_grid_search(
        segments,
        v_anchor_kmh=12.0,
        alpha_grid=[0.70, 0.80, 0.90],
        fatigue_coef_grid=[0.0, 0.30],
        fatigue_models=("linear",),
        trimp_scale=10.0,
        fit_mask_col="isFitEligible",
    )

    assert best["alpha"] == pytest.approx(0.80)
    assert best["fatigueCoef"] == pytest.approx(0.30)
    assert best["fitSegmentCount"] == 4
    assert best["fullSegmentCount"] == 6
    assert best["raceMaeSecFull"] > best["raceMaeSec"]
    assert prediction["isFitEligible"].tolist() == [True, True, False, True, True, False]
    assert len(loo) == 2
    assert {"excludedSegmentCount", "excludedTimeSec", "predictedFitEligibleSec"}.issubset(loo.columns)
    assert int(loo["excludedSegmentCount"].sum()) == 2
    assert float(grid["segmentMaeSec"].min()) <= float(grid["segmentMaeSecFull"].min()) + 1e-9


def test_hrr_trimp_grid_search_fits_on_moving_time_and_scores_full_clock() -> None:
    """Stationary dwell is stripped from the fit target; errorSec stays on full clock."""
    segments = pd.DataFrame(
        {
            "activityId": ["a", "a", "b", "b"],
            "distanceKm": [1.0, 1.0, 1.0, 1.0],
            "avgGrade": [0.0, 0.0, 0.0, 0.0],
            "meanAltitudeM": [0.0, 0.0, 0.0, 0.0],
            "meanHrReserve": [0.70, 0.72, 0.70, 0.74],
            "decayedTrimpBefore": [0.0, 4.0, 0.0, 5.0],
            "stationaryTimeSec": [0.0, 600.0, 0.0, 900.0],
        }
    )
    moving = model.predict_hrr_trimp_segment_times(
        segments,
        v_anchor_kmh=12.0,
        alpha=0.80,
        fatigue_coef=0.30,
        fatigue_model="linear",
        trimp_scale=10.0,
    )
    segments["actualMovingTimeSec"] = moving
    segments["actualTimeSec"] = moving + segments["stationaryTimeSec"]

    best, grid, prediction = model.hrr_trimp_grid_search_model(
        segments,
        v_anchor_kmh=12.0,
        alpha_grid=[0.70, 0.80, 0.90],
        fatigue_coef_grid=[0.0, 0.30],
        fatigue_models=("linear",),
        trimp_scale=10.0,
        actual_time_col="actualMovingTimeSec",
        observed_activity_times_sec={
            "a": float(segments.loc[segments["activityId"].eq("a"), "actualTimeSec"].sum()),
            "b": float(segments.loc[segments["activityId"].eq("b"), "actualTimeSec"].sum()),
        },
    )
    assert best["alpha"] == pytest.approx(0.80)
    assert best["fatigueCoef"] == pytest.approx(0.30)
    assert best["actualTimeCol"] == "actualMovingTimeSec"
    assert best["raceMaeSecFull"] > best["raceMaeSec"]
    assert "fitErrorSec" in prediction.columns
    assert prediction["fitErrorSec"].abs().max() < prediction["errorSec"].abs().max()


def test_segment_timeseries_reports_hr_valid_share() -> None:
    n = 11
    hr = np.full(n, 150.0)
    hr[0] = np.nan
    hr[1] = np.nan
    df = pd.DataFrame(
        {
            "cumulated_distance": np.linspace(0.0, 1.0, n),
            "cumulated_duration_seconds": np.linspace(0.0, 360.0, n),
            "elevationM": np.zeros(n),
            "hr": hr,
        }
    )
    segments = model.segment_timeseries(df, segment_km=1.0, hr_rest=50.0, hr_max=200.0)
    assert not segments.empty
    assert segments.iloc[0]["hrSampleCount"] == n
    assert segments.iloc[0]["hrValidSampleCount"] == n - 2
    assert segments.iloc[0]["hrValidShare"] == pytest.approx((n - 2) / n)


def test_speed_vs_hrr_curve_increases_until_effort_ceiling() -> None:
    curve = model.speed_vs_hrr_curve(
        hrr_values=[0.60, 0.70, 0.80, 0.88, 0.95],
        v_anchor_kmh=18.0,
        alpha=0.95,
        distance_km=1.0,
        avg_grade=0.0,
        fatigue_coef=0.0,
        hrr_reference=0.88,
        hrr_min_factor=0.30,
        hrr_max_factor=1.0,
    )
    assert len(curve) == 5
    assert curve.loc[curve["hrr"] == 0.80, "speedKmh"].iloc[0] > curve.loc[curve["hrr"] == 0.60, "speedKmh"].iloc[0]
    assert curve.loc[curve["hrr"] == 0.95, "speedKmh"].iloc[0] == pytest.approx(
        curve.loc[curve["hrr"] == 0.88, "speedKmh"].iloc[0],
        rel=1e-6,
    )
