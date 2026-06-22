"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from services import trail_digital_twin_benchmark as benchmark
from services import trail_digital_twin_pipeline as pipeline


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


def _base_pipeline_config() -> dict[str, object]:
    return pipeline.normalise_config(_valid_minimal_config())


def _benchmark_run(run_id: str, mae: float) -> tuple[benchmark.BenchmarkRun, dict[str, pd.DataFrame]]:
    config = _base_pipeline_config()
    config["readiness"]["ctl_weight"] = 0.03 if mae < 10 else 0.08
    run = benchmark.BenchmarkRun(
        run_id=run_id,
        group="readiness_weights",
        name=f"mae_{mae}",
        description="synthetic",
        overrides={"readiness.ctl_weight": config["readiness"]["ctl_weight"]},
        output_dir=Path("/tmp") / run_id,
        config=config,
    )
    result = pipeline.PipelineResult(
        tables={
            "table_stage_metrics": pd.DataFrame(
                [
                    {
                        "cohort": "hardTrailRun",
                        "stage": "Stage 3 HRR speed ratio LOO",
                        "fitObjective": "activity",
                        "r2": 0.90,
                        "maeMin": mae,
                        "mapePct": mae / 2.0,
                        "biasMin": -1.0,
                    },
                    {
                        "cohort": "top10HardTrailByHRR",
                        "stage": "Stage 3 HRR speed ratio LOO",
                        "fitObjective": "segment",
                        "r2": 0.95,
                        "maeMin": mae + 2.0,
                        "mapePct": mae / 2.0 + 1.0,
                        "biasMin": 1.0,
                    },
                ]
            ),
            "table_stage3_fatigue_state_comparison": pd.DataFrame(
                [
                    {
                        "cohort": "hardTrailRun",
                        "validation": "loo",
                        "fitObjective": "activity",
                        "fatigueState": "decayed",
                        "fatigueModel": "linear",
                        "maeMin": mae,
                    }
                ]
            ),
            "table_fitted_parameters": pd.DataFrame(
                [
                    {
                        "cohort": "hardTrailRun",
                        "stage": "Stage 3 HRR speed ratio",
                        "fitObjective": "activity",
                        "alpha": 0.6,
                    }
                ]
            ),
            "table_segment_type_metrics": pd.DataFrame(
                [
                    {
                        "cohort": "hardTrailRun",
                        "fitObjective": "activity",
                        "terrainFamily": "climb",
                        "terrainLabel": "Ascent",
                        "segmentCount": 10,
                        "activityCount": 3,
                        "distanceKm": 10.0,
                        "maeMin": mae / 10.0,
                        "biasMin": 0.2,
                        "mapePct": 4.0,
                    }
                ]
            ),
        },
        metadata={"config": config},
    )
    return run, benchmark.summarize_benchmark_result(run, result, benchmark.DEFAULT_BENCHMARK_CONFIG["selection"])


def test_benchmark_config_rejects_invalid_group_mode() -> None:
    with pytest.raises(ValueError, match="invalid benchmark group mode"):
        benchmark.normalise_benchmark_config({"groups": [{"id": "bad", "mode": "random", "runs": []}]})


def test_apply_dotted_override_updates_nested_value() -> None:
    config = {"readiness": {"ctl_weight": 0.05}}

    benchmark.apply_dotted_override(config, "readiness.tsb_weight", 0.2)

    assert config["readiness"]["ctl_weight"] == pytest.approx(0.05)
    assert config["readiness"]["tsb_weight"] == pytest.approx(0.2)


def test_expand_benchmark_runs_caps_grid_and_sets_run_defaults(tmp_path: Path) -> None:
    sweep = benchmark.normalise_benchmark_config(
        {
            "execution": {"per_run_html": False, "disable_inner_robustness": True},
            "groups": [
                {
                    "id": "readiness",
                    "mode": "grid",
                    "overrides": {
                        "readiness.ctl_weight": [0.0, 0.05],
                        "readiness.tsb_weight": [0.0, 0.10],
                    },
                }
            ],
        }
    )

    runs = benchmark.expand_benchmark_runs(_base_pipeline_config(), sweep, tmp_path, max_runs=3)

    assert len(runs) == 3
    assert runs[0].config["outputs"]["write_html"] is False
    assert runs[0].config["robustness"]["enabled"] is False
    assert runs[0].config["segment_grid"]["enabled"] is False
    assert runs[1].config["readiness"]["tsb_weight"] == pytest.approx(0.10)
    assert all(str(run.output_dir).startswith(str(tmp_path / "runs")) for run in runs)


def test_expand_benchmark_runs_applies_common_overrides(tmp_path: Path) -> None:
    sweep = benchmark.normalise_benchmark_config(
        {
            "execution": {
                "common_overrides": {
                    "fitting.validation_modes": ["in_sample"],
                    "fitting.enabled_objectives": ["activity"],
                }
            },
            "groups": [
                {
                    "id": "baseline",
                    "mode": "anchor",
                    "runs": [{"name": "current", "overrides": {}}],
                }
            ],
        }
    )

    runs = benchmark.expand_benchmark_runs(_base_pipeline_config(), sweep, tmp_path)

    assert runs[0].config["fitting"]["validation_modes"] == ["in_sample"]
    assert runs[0].config["fitting"]["enabled_objectives"] == ["activity"]
    assert runs[0].overrides["fitting.validation_modes"] == ["in_sample"]


def test_summarize_result_and_leaderboard_order() -> None:
    run_a, tables_a = _benchmark_run("000_a", 8.0)
    run_b, tables_b = _benchmark_run("001_b", 12.0)
    plan = benchmark.benchmark_plan_frame([run_a, run_b])
    manifest = pd.DataFrame(
        [
            {"runId": "000_a", "status": "success"},
            {"runId": "001_b", "status": "success"},
        ]
    )

    tables = benchmark.combine_benchmark_tables([tables_b, tables_a], manifest, plan)

    assert tables["benchmark_runs"].shape[0] == 2
    assert tables["benchmark_stage_metrics"]["runId"].nunique() == 2
    assert tables["benchmark_segment_type_metrics"]["runId"].nunique() == 2
    assert tables["benchmark_leaderboard"].iloc[0]["runId"] == "000_a"
    assert tables["benchmark_leaderboard"].iloc[0]["meanStage3MaeMin"] == pytest.approx(9.0)


def test_execute_benchmark_run_returns_summary_tables(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _base_pipeline_config()
    run = benchmark.BenchmarkRun(
        run_id="000_run",
        group="group",
        name="name",
        description="desc",
        overrides={},
        output_dir=tmp_path / "run",
        config=config,
    )

    def fake_run_pipeline(
        _config: dict[str, object],
        *,
        project_root: Path | None = None,
        config_path: Path | None = None,
    ) -> pipeline.PipelineResult:
        return pipeline.PipelineResult(
            tables={
                "table_stage_metrics": pd.DataFrame(
                    [
                        {
                            "cohort": "hardTrailRun",
                            "stage": "Stage 3 HRR speed ratio LOO",
                            "fitObjective": "activity",
                            "r2": 0.90,
                            "maeMin": 8.0,
                            "mapePct": 5.0,
                            "biasMin": 1.0,
                        }
                    ]
                ),
                "table_stage3_fatigue_state_comparison": pd.DataFrame(),
                "table_fitted_parameters": pd.DataFrame(),
            },
            metadata={"projectRoot": str(project_root), "configPath": str(config_path)},
        )

    def fake_write_outputs(_result: pipeline.PipelineResult, output_dir: Path) -> dict[str, Path]:
        return {"run_manifest": output_dir / "run_manifest.csv"}

    monkeypatch.setattr(benchmark, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(benchmark, "write_outputs", fake_write_outputs)

    result = benchmark.execute_benchmark_run(
        run,
        benchmark.DEFAULT_BENCHMARK_CONFIG["selection"],
        project_root=tmp_path,
        config_path=tmp_path / "base.yaml",
    )

    assert result.manifest_row["status"] == "success"
    assert result.manifest_row["runId"] == "000_run"
    assert result.tables["benchmark_runs"].iloc[0]["meanStage3MaeMin"] == pytest.approx(8.0)


def test_execute_benchmark_run_records_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = benchmark.BenchmarkRun(
        run_id="000_run",
        group="group",
        name="name",
        description="desc",
        overrides={},
        output_dir=tmp_path / "run",
        config=_base_pipeline_config(),
    )

    def fake_run_pipeline(
        _config: dict[str, object],
        *,
        project_root: Path | None = None,
        config_path: Path | None = None,
    ) -> pipeline.PipelineResult:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(benchmark, "run_pipeline", fake_run_pipeline)

    result = benchmark.execute_benchmark_run(
        run,
        benchmark.DEFAULT_BENCHMARK_CONFIG["selection"],
        project_root=tmp_path,
        config_path=tmp_path / "base.yaml",
    )

    assert result.tables == {}
    assert result.manifest_row["status"] == "failed"
    assert "synthetic failure" in str(result.manifest_row["error"])


def test_benchmark_html_renderer_is_self_contained() -> None:
    run, tables_for_run = _benchmark_run("000_a", 8.0)
    plan = benchmark.benchmark_plan_frame([run])
    manifest = pd.DataFrame([{"runId": "000_a", "status": "success"}])
    tables = benchmark.combine_benchmark_tables([tables_for_run], manifest, plan)

    html = benchmark.render_benchmark_html(tables, {"benchmarkConfigPath": "/tmp/benchmark.yaml"})

    assert "Trail Digital Twin Benchmark Report" in html
    assert "Leaderboard table" in html
    assert "Segment-type evaluation" in html
    assert "Segment-type metrics" in html
    assert "Ascent" in html
    assert "Plotly.newPlot" in html
    assert 'src="https://cdn.plot.ly' not in html


def test_benchmark_html_renderer_adds_swept_parameter_sections() -> None:
    runs = pd.DataFrame(
        [
            {
                "runId": "000_floor_05",
                "experimentGroup": "min_fatigue_factor_sweep",
                "experimentName": "floor_05",
                "status": "success",
                "meanStage3MaeMin": 13.2,
                "medianStage3MaeMin": 12.0,
                "maxStage3MaeMin": 20.0,
                "meanStage3MapePct": 8.0,
                "decayLambda": 0.25,
                "minFatigueFactor": 0.5,
            },
            {
                "runId": "001_floor_06",
                "experimentGroup": "min_fatigue_factor_sweep",
                "experimentName": "floor_06",
                "status": "success",
                "meanStage3MaeMin": 11.4,
                "medianStage3MaeMin": 10.0,
                "maxStage3MaeMin": 18.0,
                "meanStage3MapePct": 7.0,
                "decayLambda": 0.25,
                "minFatigueFactor": 0.6,
            },
            {
                "runId": "002_decay_015",
                "experimentGroup": "decay_lambda_sweep_hrr085",
                "experimentName": "decay_015",
                "status": "success",
                "meanStage3MaeMin": 12.7,
                "medianStage3MaeMin": 11.0,
                "maxStage3MaeMin": 19.0,
                "meanStage3MapePct": 7.5,
                "decayLambda": 0.15,
                "minFatigueFactor": 0.5,
            },
            {
                "runId": "003_decay_025",
                "experimentGroup": "decay_lambda_sweep_hrr085",
                "experimentName": "decay_025",
                "status": "success",
                "meanStage3MaeMin": 13.2,
                "medianStage3MaeMin": 12.0,
                "maxStage3MaeMin": 20.0,
                "meanStage3MapePct": 8.0,
                "decayLambda": 0.25,
                "minFatigueFactor": 0.5,
            },
        ]
    )
    stage_metrics = pd.DataFrame(
        [
            {
                "runId": row["runId"],
                "experimentGroup": row["experimentGroup"],
                "cohort": "hardTrailRun",
                "stage": "Stage 3 HRR speed ratio LOO",
                "fitObjective": "activity",
                "maeMin": row["meanStage3MaeMin"],
                "decayLambda": row["decayLambda"],
                "minFatigueFactor": row["minFatigueFactor"],
            }
            for row in runs.to_dict("records")
        ]
    )
    tables = {
        "benchmark_runs": runs,
        "benchmark_leaderboard": benchmark.leaderboard_frame(runs),
        "benchmark_manifest": pd.DataFrame(
            [{"runId": run_id, "status": "success"} for run_id in runs["runId"]]
        ),
        "benchmark_plan": runs[["runId", "experimentGroup", "experimentName"]],
        "benchmark_stage_metrics": stage_metrics,
        "benchmark_stage3_fatigue": pd.DataFrame(),
        "benchmark_fitted_parameters": pd.DataFrame(),
    }

    html = benchmark.render_benchmark_html(tables, {"selection": benchmark.DEFAULT_BENCHMARK_CONFIG["selection"]})

    assert "Sweep: Decay lambda" in html
    assert "Sweep: Minimum fatigue factor" in html
    assert "Performance by Decay lambda" in html
    assert "Performance by Minimum fatigue factor" in html


def test_read_benchmark_output_tables_and_write_html(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "runId": "000_a",
                "experimentGroup": "group",
                "experimentName": "run",
                "status": "success",
                "meanStage3MaeMin": 8.0,
            }
        ]
    ).to_csv(tmp_path / "benchmark_runs.csv", index=False)
    pd.DataFrame([{"runId": "000_a", "status": "success"}]).to_csv(
        tmp_path / "benchmark_manifest.csv",
        index=False,
    )
    run_dir = tmp_path / "runs" / "000_a"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "cohort": "hardTrailRun",
                "fitObjective": "activity",
                "activityId": "a",
                "terrainFamily": "descent",
                "distanceKm": 1.0,
                "actualTimeSec": 300.0,
                "stage3PredictedTimeSec": 360.0,
            }
        ]
    ).to_csv(run_dir / "segment_predictions.csv", index=False)

    tables = benchmark.read_benchmark_output_tables(tmp_path)
    html_path = benchmark.write_benchmark_html(tables, tmp_path, {"benchmarkConfigPath": "benchmark.yaml"})

    assert tables["benchmark_runs"].iloc[0]["runId"] == "000_a"
    assert tables["benchmark_segment_type_metrics"].iloc[0]["terrainLabel"] == "Descent"
    assert html_path.exists()
    assert "Trail Digital Twin Benchmark Report" in html_path.read_text()
