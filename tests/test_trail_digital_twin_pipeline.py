"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import pandas as pd
import pytest

from services import trail_digital_twin_pipeline as pipeline
from services import trail_performance_model as model


def _valid_minimal_config() -> dict[str, object]:
    return {
        "paths": {},
        "cohorts": {},
        "physiology": {},
        "readiness": {},
        "fitting": {},
        "segment_grid": {},
        "robustness": {},
        "outputs": {},
    }


def test_config_requires_top_level_sections() -> None:
    raw = _valid_minimal_config()
    raw.pop("fitting")

    with pytest.raises(ValueError, match="missing required config sections"):
        pipeline.normalise_config(raw)


def test_config_rejects_invalid_objective() -> None:
    raw = _valid_minimal_config()
    raw["fitting"] = {"enabled_objectives": ["activity", "pace"]}

    with pytest.raises(ValueError, match="invalid fitting objectives"):
        pipeline.normalise_config(raw)


def test_config_rejects_invalid_validation_mode() -> None:
    raw = _valid_minimal_config()
    raw["fitting"] = {"validation_modes": ["in_sample", "bootstrap"]}

    with pytest.raises(ValueError, match="invalid validation modes"):
        pipeline.normalise_config(raw)


def test_config_rejects_invalid_jobs() -> None:
    raw = _valid_minimal_config()
    raw["execution"] = {"jobs": 0}

    with pytest.raises(ValueError, match="execution.jobs must be a positive integer"):
        pipeline.normalise_config(raw)


def test_config_rejects_invalid_fatigue_state() -> None:
    raw = _valid_minimal_config()
    raw["fitting"] = {"stage3_fatigue_states": [{"fatigue_state": "rolling", "acute_trimp_col": "decayedTrimpBefore"}]}

    with pytest.raises(ValueError, match="invalid fatigue state"):
        pipeline.normalise_config(raw)


def test_config_merges_defaults_and_normalises_stage3_states() -> None:
    config = pipeline.normalise_config(_valid_minimal_config())

    assert config["physiology"]["vma_flat_kmh"] == pytest.approx(18.0)
    assert config["execution"]["jobs"] == 1
    assert config["fitting"]["enabled_objectives"] == ["activity", "segment"]
    assert config["segment_exclusion"]["enabled"] is False
    assert config["segment_exclusion"]["min_mean_speed_eq_kmh"] == pytest.approx(3.0)
    assert config["segment_exclusion"]["max_stationary_time_share"] == pytest.approx(0.40)
    assert config["segment_exclusion"]["max_abs_altitude_rate_mph"] == pytest.approx(120.0)
    assert config["segment_exclusion"]["use_moving_time_for_fit"] is False
    assert pipeline._fit_actual_time_col(config) == "actualTimeSec"
    moving = pipeline._deep_merge(config, {"segment_exclusion": {"use_moving_time_for_fit": True}})
    assert pipeline._fit_actual_time_col(moving) == "actualMovingTimeSec"
    assert pipeline._fit_mask_col(config) is None
    enabled = pipeline._deep_merge(config, {"segment_exclusion": {"enabled": True}})
    assert pipeline._fit_mask_col(enabled) == "isFitEligible"
    assert {state["fatigue_state"] for state in config["fitting"]["stage3_fatigue_states"]} == {
        "decayed",
        "cumulative",
        "progress",
        "decayed_cumulative",
        "decayed_progress",
    }
    progress_state = [
        state for state in config["fitting"]["stage3_fatigue_states"] if state["fatigue_state"] == "progress"
    ][0]
    assert progress_state["fatigue_models"] == ["linear"]
    combined_state = [
        state
        for state in config["fitting"]["stage3_fatigue_states"]
        if state["fatigue_state"] == "decayed_cumulative"
    ][0]
    assert combined_state["secondary_acute_trimp_col"] == "cumTrimpBefore"
    assert config["fitting"]["hrr_trimp_secondary_kappa_grid"] == [0.0, 0.1, 0.2, 0.3, 0.4]


def test_stage_models_include_activity_and_segment_objectives() -> None:
    config = pipeline.normalise_config(_valid_minimal_config())
    config["fitting"]["stage0_alpha_grid"] = [1.0]
    config["fitting"]["stage0_mu_grid"] = [0.0]
    config["fitting"]["hrr_trimp_alpha_grid"] = [1.0]
    config["fitting"]["hrr_trimp_kappa_grid"] = [0.0, 0.2]
    config["fitting"]["fatigue_models"] = ["linear", "exponential"]
    config["execution"]["jobs"] = 2
    config["physiology"]["vma_flat_kmh"] = 10.0
    config["physiology"]["hrr_reference"] = 0.70

    activities = pd.DataFrame(
        {
            "activityId": ["a", "b", "c"],
            "actualTimeSec": [720.0, 780.0, 840.0],
            "ctlReadinessFactor": [1.0, 1.0, 1.0],
            "rediReadinessFactor": [1.0, 1.0, 1.0],
            "startDate": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]).date,
            "name": ["A", "B", "C"],
            "distanceKm": [2.0, 2.0, 2.0],
            "ascentM": [0.0, 0.0, 0.0],
            "hrReserveRatio": [0.70, 0.75, 0.80],
        }
    )
    segments_by_activity = {}
    segment_rows = []
    for idx, activity_id in enumerate(["a", "b", "c"]):
        base = pd.DataFrame(
            {
                "activityId": [activity_id, activity_id],
                "segmentIndex": [0, 1],
                "startKm": [0.0, 1.0],
                "endKm": [1.0, 2.0],
                "distanceKm": [1.0, 1.0],
                "avgGrade": [0.0, 0.04],
                "meanAltitudeM": [0.0, 0.0],
                "progress": [0.25, 0.75],
                "actualTimeSec": [340.0 + idx * 20.0, 380.0 + idx * 20.0],
                "meanHrReserve": [0.70 + idx * 0.02, 0.74 + idx * 0.02],
                "cumTrimpBefore": [0.0, 0.4 + idx * 0.1],
                "decayedTrimpBefore": [0.0, 0.2 + idx * 0.05],
                "ctlReadinessFactor": [1.0, 1.0],
                "rediReadinessFactor": [1.0, 1.0],
                "terrainFamily": ["flat", "climb"],
            }
        )
        segments_by_activity[activity_id] = base.drop(columns=["activityId"])
        segment_rows.append(base)
    segment_features = pd.concat(segment_rows, ignore_index=True)

    tables, best, _segments = pipeline.run_paper_stage_models(
        {"synthetic": activities},
        segments_by_activity,
        segment_features,
        config,
        v_vt2_kmh=10.0,
    )

    assert set(tables["table_stage_metrics"]["fitObjective"]) == {"activity", "segment"}
    comparison = tables["table_stage3_fatigue_state_comparison"]
    assert set(comparison["fatigueState"]) == {
        "decayed",
        "cumulative",
        "progress",
        "decayed_cumulative",
        "decayed_progress",
    }
    assert set(comparison["fatigueModel"]) == {"linear", "exponential"}
    progress_comparison = comparison[comparison["fatigueState"].astype(str).eq("progress")]
    assert set(progress_comparison["fatigueModel"]) == {"linear"}
    combined = comparison[comparison["fatigueState"].astype(str).str.startswith("decayed_")]
    assert set(combined["secondaryFatigueModel"]) == {"exponential"}
    assert set(combined["secondaryAcuteTrimpCol"]) == {"cumTrimpBefore", "progress"}
    grid = tables["table_hrr_trimp_grid_search"]
    assert not grid.empty
    assert {
        "cohort",
        "stage",
        "fitObjective",
        "alpha",
        "fatigueCoef",
        "raceMaeMin",
        "segmentMaeMin",
        "hrrReference",
        "minFatigueFactor",
        "secondaryFatigueCoef",
    }.issubset(grid.columns)
    assert "Stage 3 HRR speed ratio" in set(grid["stage"].astype(str))
    assert {objective for _cohort, objective in best.keys()} == {"activity", "segment"}


def test_stage_models_can_skip_loo_validation() -> None:
    config = pipeline.normalise_config(_valid_minimal_config())
    config["fitting"]["enabled_objectives"] = ["activity"]
    config["fitting"]["validation_modes"] = ["in_sample"]
    config["fitting"]["stage0_alpha_grid"] = [1.0]
    config["fitting"]["stage0_mu_grid"] = [0.0]
    config["fitting"]["hrr_trimp_alpha_grid"] = [1.0]
    config["fitting"]["hrr_trimp_kappa_grid"] = [0.0]
    config["fitting"]["fatigue_models"] = ["linear"]
    config["fitting"]["stage3_fatigue_states"] = [
        {"fatigue_state": "decayed", "acute_trimp_col": "decayedTrimpBefore", "label": "decayed TRIMP"}
    ]
    config["physiology"]["vma_flat_kmh"] = 10.0

    activities = pd.DataFrame(
        {
            "activityId": ["a", "b"],
            "actualTimeSec": [720.0, 780.0],
            "ctlReadinessFactor": [1.0, 1.0],
            "rediReadinessFactor": [1.0, 1.0],
            "startDate": pd.to_datetime(["2026-01-01", "2026-01-02"]).date,
            "name": ["A", "B"],
            "distanceKm": [2.0, 2.0],
            "ascentM": [0.0, 0.0],
            "hrReserveRatio": [0.70, 0.75],
        }
    )
    segments_by_activity = {}
    segment_rows = []
    for idx, activity_id in enumerate(["a", "b"]):
        base = pd.DataFrame(
            {
                "activityId": [activity_id],
                "segmentIndex": [0],
                "distanceKm": [1.0],
                "avgGrade": [0.0],
                "meanAltitudeM": [0.0],
                "progress": [0.5],
                "actualTimeSec": [720.0 + idx * 60.0],
                "meanHrReserve": [0.70 + idx * 0.02],
                "cumTrimpBefore": [0.0],
                "decayedTrimpBefore": [0.0],
                "ctlReadinessFactor": [1.0],
                "rediReadinessFactor": [1.0],
                "terrainFamily": ["flat"],
            }
        )
        segments_by_activity[activity_id] = base.drop(columns=["activityId"])
        segment_rows.append(base)

    tables, _best, _segments = pipeline.run_paper_stage_models(
        {"synthetic": activities},
        segments_by_activity,
        pd.concat(segment_rows, ignore_index=True),
        config,
        v_vt2_kmh=10.0,
    )

    assert tables["activity_loo_predictions"].empty
    assert not tables["table_stage_metrics"]["stage"].astype(str).str.endswith("LOO").any()


def test_build_cohorts_includes_hard_run_or_trail_run() -> None:
    config = pipeline.normalise_config(_valid_minimal_config())
    config["cohorts"]["include"] = ["hardTrailRun", "hardRunOrTrailRun"]
    activity_df = pd.DataFrame(
        {
            "activityId": ["trail", "run", "easy"],
            "category": ["TRAIL_RUN", "RUN", "RUN"],
            "movingSec": [3600.0, 3600.0, 3600.0],
            "distanceKm": [12.0, 12.0, 5.0],
            "ascentM": [600.0, 100.0, 0.0],
            "hrReserveRatio": [0.72, 0.72, 0.50],
            "hasTimeseries": [True, True, True],
        }
    )
    segments_by_activity = {
        "trail": pd.DataFrame({"distanceKm": [1.0], "actualTimeSec": [300.0]}),
        "run": pd.DataFrame({"distanceKm": [1.0], "actualTimeSec": [300.0]}),
        "easy": pd.DataFrame({"distanceKm": [1.0], "actualTimeSec": [300.0]}),
    }

    cohorts = pipeline._build_cohorts(activity_df, segments_by_activity, config)

    assert cohorts["hardTrailRun"]["activityId"].tolist() == ["trail"]
    assert cohorts["hardRunOrTrailRun"]["activityId"].tolist() == ["trail", "run"]


def test_build_cohorts_includes_run_trail_over_20_min() -> None:
    config = pipeline.normalise_config(_valid_minimal_config())
    config["cohorts"]["include"] = ["runTrailOver20Min", "hardRunOrTrailRun"]
    activity_df = pd.DataFrame(
        {
            "activityId": ["long_easy", "short", "hard"],
            "category": ["RUN", "RUN", "TRAIL_RUN"],
            "movingSec": [25 * 60.0, 15 * 60.0, 40 * 60.0],
            "distanceKm": [5.0, 3.0, 12.0],
            "ascentM": [50.0, 20.0, 600.0],
            "hrReserveRatio": [0.55, 0.50, 0.75],
            "hasTimeseries": [True, True, True],
        }
    )
    segments_by_activity = {
        "long_easy": pd.DataFrame({"distanceKm": [1.0], "actualTimeSec": [300.0]}),
        "short": pd.DataFrame({"distanceKm": [1.0], "actualTimeSec": [300.0]}),
        "hard": pd.DataFrame({"distanceKm": [1.0], "actualTimeSec": [300.0]}),
    }

    cohorts = pipeline._build_cohorts(activity_df, segments_by_activity, config)

    assert set(cohorts["runTrailOver20Min"]["activityId"]) == {"long_easy", "hard"}
    assert cohorts["hardRunOrTrailRun"]["activityId"].tolist() == ["hard"]


def test_html_renderer_is_self_contained_and_has_expected_sections() -> None:
    result = pipeline.PipelineResult(
        tables={
            "table_cohort_descriptives": pd.DataFrame(
                [
                    {
                        "cohort": "synthetic",
                        "activityCount": 1,
                        "distanceKmTotal": 10.0,
                        "ascentMTotal": 500.0,
                        "hrrMean": 0.7,
                    }
                ]
            ),
            "table_stage_metrics": pd.DataFrame(
                [
                    {
                        "cohort": "synthetic",
                        "stage": "Stage 3 HRR speed ratio LOO",
                        "fitObjective": "activity",
                        "maeMin": 5.0,
                    }
                ]
            ),
            "activity_predictions": pd.DataFrame(
                [
                    {
                        "activityId": "a",
                        "name": "Run",
                        "cohort": "synthetic",
                        "model": "Stage 3 HRR speed ratio",
                        "fitObjective": "activity",
                        "actualMin": 60.0,
                        "predictedMin": 62.0,
                        "errorMin": 2.0,
                    }
                ]
            ),
            "table_stage3_fatigue_state_comparison": pd.DataFrame(
                [
                    {
                        "cohort": "synthetic",
                        "fitObjective": "activity",
                        "validation": "loo",
                        "fatigueState": "cumulative",
                        "fatigueModel": "linear",
                        "maeMin": 5.0,
                    }
                ]
            ),
            "table_fitted_parameters": pd.DataFrame(
                [
                    {
                        "cohort": "synthetic",
                        "stage": "Stage 3 HRR speed ratio",
                        "fitObjective": "activity",
                        "alpha": 0.8,
                        "fatigueCoef": 0.2,
                        "fatigueModel": "linear",
                        "fatigueState": "cumulative",
                        "acuteTrimpCol": "cumTrimpBefore",
                        "raceMaeSec": 300.0,
                        "segmentMaeSec": 20.0,
                    }
                ]
            ),
            "table_hrr_trimp_grid_search": pd.DataFrame(
                [
                    {
                        "cohort": "synthetic",
                        "stage": "Stage 3 HRR speed ratio",
                        "fitObjective": "activity",
                        "validation": "in_sample",
                        "alpha": 0.8,
                        "fatigueCoef": 0.2,
                        "fatigueModel": "linear",
                        "fatigueState": "cumulative",
                        "acuteTrimpCol": "cumTrimpBefore",
                        "raceMaeMin": 5.0,
                        "segmentMaeMin": 0.3,
                        "raceR2": 0.9,
                        "segmentR2": 0.8,
                        "hrrReference": 0.7,
                        "hrrMinFactor": 0.55,
                        "hrrMaxFactor": 1.3,
                        "minFatigueFactor": 0.5,
                        "decayLambda": 0.25,
                    }
                ]
            ),
            "segment_predictions": pd.DataFrame(
                [
                    {
                        "cohort": "synthetic",
                        "fitObjective": "activity",
                        "activityId": "a",
                        "terrainFamily": "climb",
                        "distanceKm": 1.0,
                        "actualTimeSec": 300.0,
                        "stage3PredictedTimeSec": 330.0,
                        "stage3ResidualSec": 30.0,
                        "progress": 0.5,
                        "meanHrReserve": 0.7,
                    }
                ]
            ),
            "table_stage3_ablation": pd.DataFrame(),
            "table_robustness_checks": pd.DataFrame(),
            "table_objective_comparison": pd.DataFrame(),
        },
        metadata={"config": pipeline.DEFAULT_CONFIG, "projectRoot": "/tmp/project", "configPath": "/tmp/config.yaml"},
    )

    html = pipeline.render_html_report(result)

    assert "Trail Digital Twin Evaluation Report" in html
    assert "Metric definitions" in html
    assert "Model formulas" in html
    assert "Optimized physiological parameters" in html
    assert "Selected alpha-kappa coordinates" in html
    assert "Alpha-kappa search surface" in html
    assert "Top alpha-kappa search cells" in html
    assert "Model response functions" in html
    assert "Stage 3 fatigue variants" in html
    assert "Activity-level results table" in html
    assert "Segment-type evaluation" in html
    assert "Segment-type evaluation table" in html
    assert "Ascent" in html
    assert "No acute fatigue" in html
    assert "Plotly.newPlot" in html
    assert 'src="https://cdn.plot.ly' not in html


def test_segment_type_metrics_group_stage3_errors_by_terrain() -> None:
    segments = pd.DataFrame(
        [
            {
                "cohort": "synthetic",
                "fitObjective": "activity",
                "activityId": "a",
                "terrainFamily": "climb",
                "distanceKm": 1.0,
                "actualTimeSec": 300.0,
                "stage3PredictedTimeSec": 330.0,
            },
            {
                "cohort": "synthetic",
                "fitObjective": "activity",
                "activityId": "b",
                "terrainFamily": "climb",
                "distanceKm": 2.0,
                "actualTimeSec": 600.0,
                "stage3PredictedTimeSec": 540.0,
            },
            {
                "cohort": "synthetic",
                "fitObjective": "activity",
                "activityId": "b",
                "terrainFamily": "descent",
                "distanceKm": 1.5,
                "actualTimeSec": 240.0,
                "stage3PredictedTimeSec": 300.0,
            },
        ]
    )

    metrics = pipeline._segment_type_metrics(segments)

    climb = metrics[metrics["terrainFamily"].eq("climb")].iloc[0]
    descent = metrics[metrics["terrainFamily"].eq("descent")].iloc[0]
    assert climb["terrainLabel"] == "Ascent"
    assert climb["segmentCount"] == 2
    assert climb["activityCount"] == 2
    assert climb["distanceKm"] == pytest.approx(3.0)
    assert climb["maeMin"] == pytest.approx(0.75)
    assert climb["biasMin"] == pytest.approx(-0.25)
    assert descent["terrainLabel"] == "Descent"
    assert descent["maeMin"] == pytest.approx(1.0)


def test_hrr_trimp_loo_accepts_segment_objective() -> None:
    segments = pd.DataFrame(
        {
            "activityId": ["a", "a", "b", "b", "c", "c"],
            "distanceKm": [1.0] * 6,
            "avgGrade": [0.0, 0.05] * 3,
            "meanAltitudeM": [0.0] * 6,
            "actualTimeSec": [360.0, 390.0, 365.0, 395.0, 370.0, 400.0],
            "meanHrReserve": [0.70, 0.75, 0.70, 0.75, 0.70, 0.75],
            "decayedTrimpBefore": [0.0, 0.2, 0.0, 0.2, 0.0, 0.2],
        }
    )

    loo = model.leave_one_out_hrr_trimp_grid_search(
        segments,
        v_anchor_kmh=10.0,
        alpha_grid=[1.0],
        fatigue_coef_grid=[0.0],
        fatigue_models=("linear",),
        objective="segment",
    )

    assert {"alpha", "fatigueCoef", "predictedTimeSec", "errorSec"}.issubset(loo.columns)
    assert len(loo) == 3
