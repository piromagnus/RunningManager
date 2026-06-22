"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Benchmark orchestration helpers for the trail digital-twin extension pipeline.
"""

from __future__ import annotations

import copy
import hashlib
import html
import itertools
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from services.trail_digital_twin_pipeline import (
    TERRAIN_FAMILY_LABELS,
    TERRAIN_FAMILY_ORDER,
    PipelineResult,
    _segment_type_metrics,
    normalise_config,
    run_pipeline,
    write_outputs,
)

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by CLI environment checks.
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


DEFAULT_BENCHMARK_CONFIG: dict[str, Any] = {
    "execution": {
        "output_dir": "docs/science/pipeline_outputs/trail_digital_twin_benchmark",
        "per_run_csv": True,
        "per_run_html": False,
        "disable_inner_robustness": True,
        "disable_inner_segment_grid": True,
        "write_html": True,
        "fail_fast": False,
        "jobs": 1,
        "common_overrides": {},
    },
    "selection": {
        "primary_stage": "Stage 3 HRR speed ratio LOO",
        "fallback_stage": "Stage 3 HRR speed ratio",
        "aggregate_objectives": ["activity", "segment"],
        "aggregate_cohorts": [],
    },
    "groups": [],
}

CONFIG_HIGHLIGHTS: tuple[tuple[str, str], ...] = (
    ("readiness.ctl_weight", "ctlWeight"),
    ("readiness.tsb_weight", "tsbWeight"),
    ("readiness.ctl_factor_min", "ctlFactorMin"),
    ("readiness.ctl_factor_max", "ctlFactorMax"),
    ("physiology.hrr_reference", "hrrReference"),
    ("physiology.hrr_min_factor", "hrrMinFactor"),
    ("physiology.hrr_max_factor", "hrrMaxFactor"),
    ("physiology.decay_lambda", "decayLambda"),
    ("physiology.min_fatigue_factor", "minFatigueFactor"),
    ("fitting.enabled_objectives", "enabledObjectives"),
    ("fitting.hrr_trimp_alpha_grid", "hrrTrimpAlphaGrid"),
    ("fitting.hrr_trimp_kappa_grid", "hrrTrimpKappaGrid"),
    ("fitting.fatigue_models", "fatigueModels"),
    ("fitting.stage3_fatigue_states", "stage3FatigueStates"),
)

BENCHMARK_TABLE_FILES: dict[str, str] = {
    "benchmark_runs": "benchmark_runs.csv",
    "benchmark_stage_metrics": "benchmark_stage_metrics.csv",
    "benchmark_stage3_fatigue": "benchmark_stage3_fatigue.csv",
    "benchmark_fitted_parameters": "benchmark_fitted_parameters.csv",
    "benchmark_segment_type_metrics": "benchmark_segment_type_metrics.csv",
    "benchmark_leaderboard": "benchmark_leaderboard.csv",
    "benchmark_manifest": "benchmark_manifest.csv",
    "benchmark_plan": "benchmark_plan.csv",
}

SWEEP_PARAMETER_LABELS: tuple[tuple[str, str], ...] = (
    ("ctlWeight", "CTL weight"),
    ("tsbWeight", "TSB weight"),
    ("ctlFactorMin", "Readiness minimum factor"),
    ("ctlFactorMax", "Readiness maximum factor"),
    ("hrrReference", "HRR reference"),
    ("hrrMinFactor", "HRR minimum factor"),
    ("hrrMaxFactor", "HRR maximum factor"),
    ("decayLambda", "Decay lambda"),
    ("minFatigueFactor", "Minimum fatigue factor"),
)


@dataclass(frozen=True)
class BenchmarkRun:
    """One expanded experiment with a complete pipeline config."""

    run_id: str
    group: str
    name: str
    description: str
    overrides: dict[str, Any]
    output_dir: Path
    config: dict[str, Any]


@dataclass
class BenchmarkRunResult:
    """One completed benchmark run, including failure metadata."""

    run_id: str
    tables: dict[str, pd.DataFrame]
    manifest_row: dict[str, object]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _slugify(value: object, *, max_length: int = 80) -> str:
    if isinstance(value, float):
        text = f"{value:.4g}"
    elif isinstance(value, (list, tuple, dict)):
        digest = hashlib.sha1(_compact_json(value).encode("utf-8")).hexdigest()[:8]
        text = f"set_{digest}"
    else:
        text = str(value)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return (slug or "run")[:max_length]


def _grid_run_name(overrides: Mapping[str, Any]) -> str:
    parts = []
    for dotted_path, value in overrides.items():
        key = dotted_path.split(".")[-1]
        parts.append(f"{key}_{_slugify(value, max_length=20)}")
    return "_".join(parts)[:96]


def _dedupe_run_id(run_id: str, used: set[str]) -> str:
    candidate = run_id
    index = 2
    while candidate in used:
        suffix = f"_{index}"
        candidate = f"{run_id[: 120 - len(suffix)]}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


def _get_dotted(config: Mapping[str, Any], dotted_path: str) -> object:
    current: object = config
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def apply_dotted_override(config: dict[str, Any], dotted_path: str, value: object) -> None:
    """Apply one dotted-path override to a nested config in place."""
    parts = [part for part in dotted_path.split(".") if part]
    if not parts or len(parts) != len(dotted_path.split(".")):
        raise ValueError(f"invalid dotted override path: {dotted_path!r}")
    current: dict[str, Any] = config
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            current[part] = {}
            existing = current[part]
        if not isinstance(existing, dict):
            raise ValueError(f"cannot override through non-mapping config path: {dotted_path}")
        current = existing
    current[parts[-1]] = copy.deepcopy(value)


def load_benchmark_config(path: Path) -> dict[str, Any]:
    """Load and validate a benchmark sweep YAML file."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to read trail digital-twin benchmark configs") from YAML_IMPORT_ERROR
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark config root must be a mapping")
    return normalise_benchmark_config(raw)


def normalise_benchmark_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Merge benchmark defaults and validate experiment group shape."""
    config = _deep_merge(DEFAULT_BENCHMARK_CONFIG, raw_config)
    groups = config.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("benchmark config requires a non-empty groups list")
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("each benchmark group must be a mapping")
        mode = str(group.get("mode", "variants"))
        if mode not in {"anchor", "variants", "grid"}:
            raise ValueError(f"invalid benchmark group mode: {mode}")
        if mode == "grid":
            overrides = group.get("overrides")
            if not isinstance(overrides, Mapping) or not overrides:
                raise ValueError("grid benchmark groups require non-empty overrides")
            for key, values in overrides.items():
                if not isinstance(key, str):
                    raise ValueError("grid override paths must be strings")
                if not isinstance(values, list) or not values:
                    raise ValueError(f"grid override {key} must be a non-empty list")
        else:
            runs = group.get("runs")
            if not isinstance(runs, list) or not runs:
                raise ValueError(f"{mode} benchmark groups require a non-empty runs list")
    common_overrides = config["execution"].get("common_overrides", {})
    if not isinstance(common_overrides, Mapping):
        raise ValueError("execution.common_overrides must be a mapping")
    return config


def _candidate_runs(benchmark_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for group in benchmark_config["groups"]:
        group_id = str(group.get("id", group.get("name", "group")))
        description = str(group.get("description", ""))
        mode = str(group.get("mode", "variants"))
        if mode == "grid":
            override_grid = group["overrides"]
            keys = list(override_grid.keys())
            for values in itertools.product(*(override_grid[key] for key in keys)):
                overrides = dict(zip(keys, values))
                candidates.append(
                    {
                        "group": group_id,
                        "name": _grid_run_name(overrides),
                        "description": description,
                        "overrides": overrides,
                    }
                )
        else:
            for run in group["runs"]:
                if not isinstance(run, Mapping):
                    raise ValueError("benchmark run entries must be mappings")
                candidates.append(
                    {
                        "group": group_id,
                        "name": str(run.get("name", "run")),
                        "description": str(run.get("description", description)),
                        "overrides": dict(run.get("overrides", {})),
                    }
                )
    return candidates


def expand_benchmark_runs(
    base_config: Mapping[str, Any],
    benchmark_config: Mapping[str, Any],
    output_dir: Path,
    *,
    max_runs: int | None = None,
) -> list[BenchmarkRun]:
    """Expand benchmark groups into complete pipeline configs."""
    execution = benchmark_config["execution"]
    common_overrides = dict(execution.get("common_overrides", {}))
    candidates = _candidate_runs(benchmark_config)
    if max_runs is not None:
        candidates = candidates[:max_runs]

    runs: list[BenchmarkRun] = []
    used_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        group = str(candidate["group"])
        name = str(candidate["name"])
        run_id = _dedupe_run_id(f"{index:03d}_{_slugify(group)}_{_slugify(name)}", used_ids)
        run_output_dir = output_dir / "runs" / run_id
        config = copy.deepcopy(dict(base_config))
        combined_overrides = {**common_overrides, **candidate["overrides"]}
        for dotted_path, value in combined_overrides.items():
            apply_dotted_override(config, dotted_path, value)
        config.setdefault("paths", {})["output_dir"] = str(run_output_dir)
        config.setdefault("outputs", {})["write_csv"] = bool(execution.get("per_run_csv", True))
        config.setdefault("outputs", {})["write_html"] = bool(execution.get("per_run_html", False))
        if bool(execution.get("disable_inner_robustness", True)):
            config.setdefault("robustness", {})["enabled"] = False
        if bool(execution.get("disable_inner_segment_grid", True)):
            config.setdefault("segment_grid", {})["enabled"] = False
        config = normalise_config(config, require_sections=False)
        runs.append(
            BenchmarkRun(
                run_id=run_id,
                group=group,
                name=name,
                description=str(candidate["description"]),
                overrides=copy.deepcopy(combined_overrides),
                output_dir=run_output_dir,
                config=config,
            )
        )
    return runs


def _highlight_values(config: Mapping[str, Any]) -> dict[str, object]:
    values: dict[str, object] = {}
    for dotted_path, column in CONFIG_HIGHLIGHTS:
        value = _get_dotted(config, dotted_path)
        values[column] = _compact_json(value) if isinstance(value, (list, tuple, dict)) else value
    return values


def benchmark_plan_frame(runs: Sequence[BenchmarkRun]) -> pd.DataFrame:
    """Return the expanded run plan as a table."""
    rows = []
    for run in runs:
        rows.append(
            {
                "runId": run.run_id,
                "experimentGroup": run.group,
                "experimentName": run.name,
                "description": run.description,
                "runOutputDir": str(run.output_dir),
                "overrideJson": _compact_json(run.overrides),
                **_highlight_values(run.config),
            }
        )
    return pd.DataFrame(rows)


def _with_run_columns(df: pd.DataFrame, run: BenchmarkRun) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    enriched = df.copy()
    enriched.insert(0, "runId", run.run_id)
    enriched.insert(1, "experimentGroup", run.group)
    enriched.insert(2, "experimentName", run.name)
    enriched["overrideJson"] = _compact_json(run.overrides)
    for column, value in _highlight_values(run.config).items():
        enriched[column] = value
    return enriched


def select_primary_stage_metrics(stage_metrics: pd.DataFrame, selection: Mapping[str, Any]) -> pd.DataFrame:
    """Select the stage rows used for benchmark ranking."""
    if stage_metrics.empty or "stage" not in stage_metrics.columns:
        return pd.DataFrame()
    primary_stage = str(selection.get("primary_stage", "Stage 3 HRR speed ratio LOO"))
    fallback_stage = str(selection.get("fallback_stage", "Stage 3 HRR speed ratio"))
    selected = stage_metrics[stage_metrics["stage"].astype(str).eq(primary_stage)].copy()
    selected_stage = primary_stage
    if selected.empty:
        selected = stage_metrics[stage_metrics["stage"].astype(str).eq(fallback_stage)].copy()
        selected_stage = fallback_stage
    if selected.empty:
        return selected

    objectives = [str(value) for value in selection.get("aggregate_objectives", [])]
    cohorts = [str(value) for value in selection.get("aggregate_cohorts", [])]
    if objectives and "fitObjective" in selected.columns:
        selected = selected[selected["fitObjective"].astype(str).isin(objectives)]
    if cohorts and "cohort" in selected.columns:
        selected = selected[selected["cohort"].astype(str).isin(cohorts)]
    selected["selectedStage"] = selected_stage
    return selected


def summarize_benchmark_result(
    run: BenchmarkRun,
    result: PipelineResult,
    selection: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """Extract benchmark-level tables from one completed pipeline run."""
    stage_metrics = result.tables.get("table_stage_metrics", pd.DataFrame())
    selected = select_primary_stage_metrics(stage_metrics, selection)
    summary = {
        "runId": run.run_id,
        "experimentGroup": run.group,
        "experimentName": run.name,
        "status": "success",
        "description": run.description,
        "selectedStage": str(selected["selectedStage"].iloc[0]) if not selected.empty else "",
        "selectedRowCount": int(len(selected)),
        "meanStage3MaeMin": _safe_mean(selected.get("maeMin", pd.Series(dtype=float))),
        "medianStage3MaeMin": _safe_median(selected.get("maeMin", pd.Series(dtype=float))),
        "maxStage3MaeMin": _safe_max(selected.get("maeMin", pd.Series(dtype=float))),
        "meanStage3MapePct": _safe_mean(selected.get("mapePct", pd.Series(dtype=float))),
        "meanStage3R2": _safe_mean(selected.get("r2", pd.Series(dtype=float))),
        "meanAbsBiasMin": _safe_abs_mean(selected.get("biasMin", pd.Series(dtype=float))),
        "overrideJson": _compact_json(run.overrides),
        **_highlight_values(run.config),
    }
    return {
        "benchmark_runs": pd.DataFrame([summary]),
        "benchmark_stage_metrics": _with_run_columns(stage_metrics, run),
        "benchmark_stage3_fatigue": _with_run_columns(
            result.tables.get("table_stage3_fatigue_state_comparison", pd.DataFrame()),
            run,
        ),
        "benchmark_fitted_parameters": _with_run_columns(
            result.tables.get("table_fitted_parameters", pd.DataFrame()),
            run,
        ),
        "benchmark_segment_type_metrics": _with_run_columns(
            result.tables.get("table_segment_type_metrics", pd.DataFrame()),
            run,
        ),
    }


def execute_benchmark_run(
    run: BenchmarkRun,
    selection: Mapping[str, Any],
    *,
    project_root: Path,
    config_path: Path,
) -> BenchmarkRunResult:
    """Run one benchmark experiment and return aggregate-ready tables."""
    started = time.perf_counter()
    status = "success"
    error = ""
    output_files: dict[str, Path] = {}
    tables: dict[str, pd.DataFrame] = {}
    try:
        run_config = copy.deepcopy(run.config)
        run_config.setdefault("execution", {})["jobs"] = 1
        result = run_pipeline(run_config, project_root=project_root, config_path=config_path)
        output_files = write_outputs(result, run.output_dir)
        tables = summarize_benchmark_result(run, result, selection)
    except Exception as exc:  # noqa: BLE001 - benchmark runs should be recorded independently.
        status = "failed"
        error = str(exc)
    elapsed_sec = time.perf_counter() - started
    return BenchmarkRunResult(
        run_id=run.run_id,
        tables=tables,
        manifest_row={
            "runId": run.run_id,
            "experimentGroup": run.group,
            "experimentName": run.name,
            "status": status,
            "elapsedSec": elapsed_sec,
            "runOutputDir": str(run.output_dir),
            "htmlReport": str(output_files.get("html_report", "")),
            "error": error,
        },
    )


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else math.nan


def _safe_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else math.nan


def _safe_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else math.nan


def _safe_abs_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().abs()
    return float(numeric.mean()) if not numeric.empty else math.nan


def leaderboard_frame(benchmark_runs: pd.DataFrame) -> pd.DataFrame:
    """Sort successful runs by the primary Stage 3 score."""
    if benchmark_runs.empty:
        return pd.DataFrame()
    sortable = benchmark_runs.copy()
    sortable["meanStage3MaeMin"] = pd.to_numeric(sortable["meanStage3MaeMin"], errors="coerce")
    sortable["meanStage3MapePct"] = pd.to_numeric(sortable["meanStage3MapePct"], errors="coerce")
    sortable = sortable[sortable["meanStage3MaeMin"].notna()].sort_values(
        ["meanStage3MaeMin", "meanStage3MapePct", "maxStage3MaeMin", "runId"],
        na_position="last",
    )
    sortable.insert(0, "rank", range(1, len(sortable) + 1))
    return sortable


def combine_benchmark_tables(
    per_run_tables: Sequence[Mapping[str, pd.DataFrame]],
    manifest: pd.DataFrame,
    plan: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Combine per-run summary tables and attach leaderboard/manifest outputs."""
    names = [
        "benchmark_runs",
        "benchmark_stage_metrics",
        "benchmark_stage3_fatigue",
        "benchmark_fitted_parameters",
        "benchmark_segment_type_metrics",
    ]
    tables = {}
    for name in names:
        frames = [tables_for_run[name] for tables_for_run in per_run_tables if not tables_for_run[name].empty]
        tables[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    tables["benchmark_leaderboard"] = leaderboard_frame(tables["benchmark_runs"])
    tables["benchmark_manifest"] = manifest
    tables["benchmark_plan"] = plan
    return tables


def _plot_html(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False, "responsive": True})


def _html_table(df: pd.DataFrame, table_id: str, max_rows: int = 200) -> str:
    if df.empty:
        return "<p class='empty'>No rows available.</p>"
    display_df = df.head(max_rows).copy()
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in display_df.columns)
    body_rows = []
    for _, row in display_df.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.fillna("").tolist())
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        f"<div class='table-wrap'><table id='{html.escape(table_id)}' class='sortable'>"
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _leaderboard_figure(leaderboard: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if leaderboard.empty:
        return fig
    data = leaderboard.head(30).copy()
    fig.add_trace(
        go.Bar(
            x=data["runId"],
            y=data["meanStage3MaeMin"],
            marker_color="#2563eb",
            customdata=data[["experimentGroup", "experimentName", "meanStage3MapePct"]],
            hovertemplate=(
                "Run=%{x}<br>Group=%{customdata[0]}<br>Name=%{customdata[1]}<br>"
                "Mean MAE=%{y:.2f} min<br>Mean MAPE=%{customdata[2]:.2f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Top benchmark runs by mean Stage 3 LOO MAE",
        xaxis_title="Run",
        yaxis_title="Mean MAE (min)",
        margin={"l": 50, "r": 30, "t": 60, "b": 140},
    )
    return fig


def _group_distribution_figure(benchmark_runs: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if benchmark_runs.empty or "experimentGroup" not in benchmark_runs:
        return fig
    data = benchmark_runs.copy()
    data["meanStage3MaeMin"] = pd.to_numeric(data["meanStage3MaeMin"], errors="coerce")
    data = data[data["meanStage3MaeMin"].notna()]
    for group, frame in data.groupby("experimentGroup", sort=True):
        fig.add_trace(go.Box(y=frame["meanStage3MaeMin"], name=str(group), boxpoints="all", jitter=0.35))
    fig.update_layout(
        title="Score distribution by experiment group",
        yaxis_title="Mean Stage 3 MAE (min)",
        margin={"l": 50, "r": 30, "t": 60, "b": 100},
    )
    return fig


def _readiness_heatmap(benchmark_runs: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    required = {"ctlWeight", "tsbWeight", "meanStage3MaeMin"}
    if benchmark_runs.empty or not required.issubset(benchmark_runs.columns):
        return fig
    data = benchmark_runs.copy()
    data["ctlWeight"] = pd.to_numeric(data["ctlWeight"], errors="coerce")
    data["tsbWeight"] = pd.to_numeric(data["tsbWeight"], errors="coerce")
    data["meanStage3MaeMin"] = pd.to_numeric(data["meanStage3MaeMin"], errors="coerce")
    data = data.dropna(subset=["ctlWeight", "tsbWeight", "meanStage3MaeMin"])
    if data.empty:
        return fig
    pivot = data.pivot_table(
        index="tsbWeight",
        columns="ctlWeight",
        values="meanStage3MaeMin",
        aggfunc="mean",
    ).sort_index(ascending=True)
    fig.add_trace(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            colorscale="Viridis",
            colorbar={"title": "MAE min"},
            hovertemplate="CTL=%{x}<br>TSB=%{y}<br>Mean MAE=%{z:.2f} min<extra></extra>",
        )
    )
    fig.update_layout(
        title="Readiness weight sweep",
        xaxis_title="CTL weight",
        yaxis_title="TSB weight",
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
    )
    return fig


def _stage_metric_figure(stage_metrics: pd.DataFrame, leaderboard: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if stage_metrics.empty or leaderboard.empty:
        return fig
    top_runs = leaderboard.head(10)["runId"].astype(str).tolist()
    data = stage_metrics[
        stage_metrics["runId"].astype(str).isin(top_runs)
        & stage_metrics["stage"].astype(str).eq("Stage 3 HRR speed ratio LOO")
    ].copy()
    if data.empty:
        data = stage_metrics[
            stage_metrics["runId"].astype(str).isin(top_runs)
            & stage_metrics["stage"].astype(str).eq("Stage 3 HRR speed ratio")
        ].copy()
    if data.empty:
        return fig
    for objective, frame in data.groupby("fitObjective", sort=True):
        fig.add_trace(
            go.Bar(
                x=[frame["runId"], frame["cohort"]],
                y=frame["maeMin"],
                name=str(objective),
                hovertemplate="Run=%{x[0]}<br>Cohort=%{x[1]}<br>MAE=%{y:.2f} min<extra></extra>",
            )
        )
    fig.update_layout(
        title="Top-run Stage 3 MAE by cohort and fitting objective",
        barmode="group",
        yaxis_title="MAE (min)",
        margin={"l": 50, "r": 30, "t": 60, "b": 140},
    )
    return fig


def _fatigue_variant_figure(stage3_fatigue: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    required = {"validation", "fatigueState", "fatigueModel", "maeMin"}
    if stage3_fatigue.empty or not required.issubset(stage3_fatigue.columns):
        return fig
    data = stage3_fatigue[stage3_fatigue["validation"].astype(str).eq("loo")].copy()
    if data.empty:
        data = stage3_fatigue.copy()
    data["variant"] = data["fatigueState"].astype(str) + " / " + data["fatigueModel"].astype(str)
    for variant, frame in data.groupby("variant", sort=True):
        fig.add_trace(go.Box(y=frame["maeMin"], name=str(variant), boxpoints=False))
    fig.update_layout(
        title="Stage 3 fatigue variants across benchmark runs",
        yaxis_title="MAE (min)",
        margin={"l": 50, "r": 30, "t": 60, "b": 90},
    )
    return fig


def _terrain_label(terrain_family: object) -> str:
    family = str(terrain_family) if pd.notna(terrain_family) else "unknown"
    return TERRAIN_FAMILY_LABELS.get(family, family.replace("_", " ").title())


def _segment_type_figure(segment_metrics: pd.DataFrame, leaderboard: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    required = {
        "runId",
        "cohort",
        "fitObjective",
        "terrainFamily",
        "segmentCount",
        "maeMin",
        "biasMin",
    }
    if segment_metrics.empty or not required.issubset(segment_metrics.columns):
        return fig

    data = segment_metrics.copy()
    if not leaderboard.empty and "runId" in leaderboard.columns:
        best_run_id = str(leaderboard.iloc[0]["runId"])
        best_run = data[data["runId"].astype(str).eq(best_run_id)].copy()
        if not best_run.empty:
            data = best_run
    if "terrainLabel" not in data.columns:
        data["terrainLabel"] = data["terrainFamily"].map(_terrain_label)
    data["terrainOrder"] = data["terrainFamily"].map(TERRAIN_FAMILY_ORDER).fillna(99).astype(int)
    data["maeMin"] = pd.to_numeric(data["maeMin"], errors="coerce")
    data["biasMin"] = pd.to_numeric(data["biasMin"], errors="coerce")
    data = data.dropna(subset=["maeMin"]).sort_values(
        ["terrainOrder", "terrainFamily", "cohort", "fitObjective"]
    )
    if data.empty:
        return fig

    for (cohort, objective), frame in data.groupby(["cohort", "fitObjective"], sort=True):
        context = f"{cohort} / {objective}"
        fig.add_trace(
            go.Bar(
                x=frame["terrainLabel"],
                y=frame["maeMin"],
                name=context,
                customdata=list(zip(frame["segmentCount"], frame["biasMin"], strict=False)),
                hovertemplate=(
                    "Segment type=%{x}<br>Context="
                    f"{html.escape(context)}<br>MAE=%{{y:.2f}} min<br>"
                    "Bias=%{customdata[1]:.2f} min<br>Segments=%{customdata[0]:.0f}"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Best run Stage 3 error by segment type",
        barmode="group",
        yaxis_title="MAE (min)",
        margin={"l": 55, "r": 30, "t": 60, "b": 100},
        legend={"orientation": "h", "y": -0.30},
    )
    return fig


def _has_values(values: object) -> bool:
    try:
        return len(values) > 0  # type: ignore[arg-type]
    except TypeError:
        return False


def _figure_has_data(fig: go.Figure) -> bool:
    return any(_has_values(getattr(trace, "x", [])) or _has_values(getattr(trace, "y", [])) for trace in fig.data)


def _swept_parameter_columns(benchmark_runs: pd.DataFrame) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    if benchmark_runs.empty:
        return columns
    for column, label in SWEEP_PARAMETER_LABELS:
        if column not in benchmark_runs.columns:
            continue
        values = pd.to_numeric(benchmark_runs[column], errors="coerce").dropna()
        if values.nunique() > 1:
            columns.append((column, label))
    return columns


def _varying_groups(data: pd.DataFrame, parameter_col: str) -> list[str]:
    if data.empty or "experimentGroup" not in data.columns or parameter_col not in data.columns:
        return []
    groups: list[str] = []
    for group, frame in data.groupby("experimentGroup", sort=True):
        values = pd.to_numeric(frame[parameter_col], errors="coerce").dropna()
        if values.nunique() > 1:
            groups.append(str(group))
    return groups


def _successful_runs_for_parameter(benchmark_runs: pd.DataFrame, parameter_col: str) -> pd.DataFrame:
    if benchmark_runs.empty or parameter_col not in benchmark_runs.columns:
        return pd.DataFrame()
    data = benchmark_runs.copy()
    if "experimentGroup" not in data.columns:
        data["experimentGroup"] = "benchmark"
    if "runId" not in data.columns:
        data["runId"] = data.index.astype(str)
    if "medianStage3MaeMin" not in data.columns:
        data["medianStage3MaeMin"] = data["meanStage3MaeMin"]
    if "status" in data.columns:
        data = data[data["status"].astype(str).eq("success")]
    data[parameter_col] = pd.to_numeric(data[parameter_col], errors="coerce")
    data["meanStage3MaeMin"] = pd.to_numeric(data["meanStage3MaeMin"], errors="coerce")
    data = data.dropna(subset=[parameter_col, "meanStage3MaeMin"])
    varying = _varying_groups(data, parameter_col)
    if varying:
        data = data[data["experimentGroup"].astype(str).isin(varying)]
    return data


def _selected_stage_metrics(
    stage_metrics: pd.DataFrame,
    metadata: Mapping[str, object],
) -> pd.DataFrame:
    if stage_metrics.empty or "stage" not in stage_metrics.columns:
        return pd.DataFrame()
    selection = metadata.get("selection", DEFAULT_BENCHMARK_CONFIG["selection"])
    if isinstance(selection, Mapping):
        selected = select_primary_stage_metrics(stage_metrics, selection)
        if not selected.empty:
            return selected
    data = stage_metrics[stage_metrics["stage"].astype(str).eq("Stage 3 HRR speed ratio LOO")].copy()
    if data.empty:
        data = stage_metrics[stage_metrics["stage"].astype(str).eq("Stage 3 HRR speed ratio")].copy()
    return data


def _stage_metrics_for_parameter(
    stage_metrics: pd.DataFrame,
    metadata: Mapping[str, object],
    parameter_col: str,
    groups: Sequence[str],
) -> pd.DataFrame:
    if stage_metrics.empty or parameter_col not in stage_metrics.columns:
        return pd.DataFrame()
    data = _selected_stage_metrics(stage_metrics, metadata)
    if data.empty or parameter_col not in data.columns or "maeMin" not in data.columns:
        return pd.DataFrame()
    data = data.copy()
    if "runId" not in data.columns:
        data["runId"] = data.index.astype(str)
    data[parameter_col] = pd.to_numeric(data[parameter_col], errors="coerce")
    data["maeMin"] = pd.to_numeric(data["maeMin"], errors="coerce")
    data = data.dropna(subset=[parameter_col, "maeMin"])
    if groups and "experimentGroup" in data.columns:
        data = data[data["experimentGroup"].astype(str).isin(groups)]
    return data


def _sweep_parameter_figure(
    benchmark_runs: pd.DataFrame,
    stage_metrics: pd.DataFrame,
    metadata: Mapping[str, object],
    parameter_col: str,
    label: str,
) -> go.Figure:
    fig = go.Figure()
    run_data = _successful_runs_for_parameter(benchmark_runs, parameter_col)
    if run_data.empty:
        return fig

    groups = _varying_groups(run_data, parameter_col)
    aggregate = (
        run_data.groupby(["experimentGroup", parameter_col], as_index=False)
        .agg(
            meanStage3MaeMin=("meanStage3MaeMin", "mean"),
            medianStage3MaeMin=("medianStage3MaeMin", "mean"),
            runCount=("runId", "nunique"),
        )
        .sort_values(["experimentGroup", parameter_col])
    )
    for group, frame in aggregate.groupby("experimentGroup", sort=True):
        fig.add_trace(
            go.Scatter(
                x=frame[parameter_col],
                y=frame["meanStage3MaeMin"],
                mode="lines+markers",
                name=f"{group} mean",
                line={"width": 3},
                marker={"size": 9},
                customdata=frame[["runCount", "medianStage3MaeMin"]],
                hovertemplate=(
                    f"{html.escape(label)}=%{{x:.4g}}<br>Group=%{{fullData.name}}<br>"
                    "Mean MAE=%{y:.2f} min<br>Runs=%{customdata[0]:.0f}<br>"
                    "Median MAE=%{customdata[1]:.2f} min<extra></extra>"
                ),
            )
        )

    stage_data = _stage_metrics_for_parameter(stage_metrics, metadata, parameter_col, groups)
    if not stage_data.empty and {"cohort", "fitObjective"}.issubset(stage_data.columns):
        cohort_data = (
            stage_data.groupby(["cohort", "fitObjective", parameter_col], as_index=False)
            .agg(maeMin=("maeMin", "mean"), rowCount=("runId", "nunique"))
            .sort_values(["cohort", "fitObjective", parameter_col])
        )
        for (cohort, objective), frame in cohort_data.groupby(["cohort", "fitObjective"], sort=True):
            fig.add_trace(
                go.Scatter(
                    x=frame[parameter_col],
                    y=frame["maeMin"],
                    mode="lines+markers",
                    name=f"{cohort} / {objective}",
                    line={"width": 1.6, "dash": "dot"},
                    marker={"size": 6, "symbol": "circle-open"},
                    opacity=0.72,
                    customdata=frame[["rowCount"]],
                    hovertemplate=(
                        f"{html.escape(label)}=%{{x:.4g}}<br>Cohort=%{{fullData.name}}<br>"
                        "MAE=%{y:.2f} min<br>Runs=%{customdata[0]:.0f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=f"Performance by {label}",
        xaxis_title=label,
        yaxis_title="Stage 3 LOO MAE (min, lower is better)",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.42, "xanchor": "left", "x": 0},
        margin={"l": 60, "r": 30, "t": 60, "b": 160},
    )
    return fig


def _sweep_parameter_sections(
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, object],
) -> list[tuple[str, str]]:
    runs = tables.get("benchmark_runs", pd.DataFrame())
    stage_metrics = tables.get("benchmark_stage_metrics", pd.DataFrame())
    sections: list[tuple[str, str]] = []
    for column, label in _swept_parameter_columns(runs):
        fig = _sweep_parameter_figure(runs, stage_metrics, metadata, column, label)
        if _figure_has_data(fig):
            sections.append((f"Sweep: {label}", _plot_html(fig)))
    return sections


def render_benchmark_html(tables: Mapping[str, pd.DataFrame], metadata: Mapping[str, object]) -> str:
    """Render a self-contained benchmark report."""
    runs = tables.get("benchmark_runs", pd.DataFrame())
    leaderboard = tables.get("benchmark_leaderboard", pd.DataFrame())
    manifest = tables.get("benchmark_manifest", pd.DataFrame())
    plan = tables.get("benchmark_plan", pd.DataFrame())
    completed = int(manifest["status"].eq("success").sum()) if "status" in manifest else 0
    failed = int(manifest["status"].eq("failed").sum()) if "status" in manifest else 0
    cards = [
        ("Planned runs", len(plan)),
        ("Completed", completed),
        ("Failed", failed),
        ("Ranked", len(leaderboard)),
    ]
    card_html = "\n".join(
        f"<div class='card'><h3>{html.escape(label)}</h3><p>{html.escape(str(value))}</p></div>"
        for label, value in cards
    )
    best_text = "No successful runs were ranked."
    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        best_text = (
            f"Best run: <strong>{html.escape(str(best['runId']))}</strong> "
            f"({html.escape(str(best['experimentGroup']))}) with mean Stage 3 MAE "
            f"{float(best['meanStage3MaeMin']):.2f} min."
        )
    metadata_json = json.dumps(metadata, indent=2, default=str)
    sections = [
        ("Leaderboard", _plot_html(_leaderboard_figure(leaderboard))),
        ("Group distributions", _plot_html(_group_distribution_figure(runs))),
        *_sweep_parameter_sections(tables, metadata),
        ("Readiness sweep", _plot_html(_readiness_heatmap(runs))),
        (
            "Stage 3 by cohort",
            _plot_html(_stage_metric_figure(tables.get("benchmark_stage_metrics", pd.DataFrame()), leaderboard)),
        ),
        (
            "Segment-type evaluation",
            _plot_html(_segment_type_figure(tables.get("benchmark_segment_type_metrics", pd.DataFrame()), leaderboard)),
        ),
        (
            "Fatigue variants",
            _plot_html(_fatigue_variant_figure(tables.get("benchmark_stage3_fatigue", pd.DataFrame()))),
        ),
    ]
    figure_html = "\n".join(
        f"<section><h2>{html.escape(title)}</h2><div class='chart'>{body}</div></section>" for title, body in sections
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trail Digital Twin Benchmark Report</title>
<style>
:root {{ --ink:#1f2937; --muted:#667085; --line:#d0d5dd; --bg:#f8fafc; --panel:#ffffff; }}
body {{ margin:0; font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:var(--ink); background:var(--bg); }}
header {{ padding:28px 36px 18px; background:#fff; border-bottom:1px solid var(--line); }}
main {{ max-width:1320px; margin:0 auto; padding:24px 24px 48px; }}
h1 {{ margin:0 0 8px; font-size:30px; }}
h2 {{ margin:0 0 14px; font-size:20px; }}
h3 {{ margin:0 0 8px; font-size:16px; }}
p {{ color:var(--muted); }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; margin:0 0 18px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin:16px 0 0; }}
.card {{ border:1px solid var(--line); border-radius:8px; padding:14px; background:#fff; }}
.card p {{ margin:4px 0; font-size:24px; color:var(--ink); }}
.chart {{ min-height:360px; }}
.table-wrap {{ overflow:auto; max-height:620px; border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; background:#fff; }}
th,td {{ padding:8px 10px; border-bottom:1px solid #eef2f7; text-align:left; white-space:nowrap; }}
th {{ position:sticky; top:0; background:#f3f4f6; cursor:pointer; }}
pre {{ overflow:auto; background:#111827; color:#e5e7eb; padding:14px; border-radius:8px; font-size:12px; }}
.empty {{ color:var(--muted); font-style:italic; }}
</style>
<script>{get_plotlyjs()}</script>
</head>
<body>
<header>
<h1>Trail Digital Twin Benchmark Report</h1>
<p>Large sweep comparison for readiness, HRR/TRIMP, fatigue state, fatigue shape, and fit objective settings.</p>
</header>
<main>
<section>
<h2>Benchmark summary</h2>
<p>{best_text}</p>
<div class="grid">{card_html}</div>
</section>
{figure_html}
<section>
<h2>Leaderboard table</h2>
{_html_table(leaderboard, "leaderboard")}
</section>
<section>
<h2>Segment-type metrics</h2>
<p>Per-run Stage 3 segment errors grouped by terrain family. Positive bias means predicted time is slower
than observed time.</p>
{_html_table(tables.get("benchmark_segment_type_metrics", pd.DataFrame()), "segment-type-metrics")}
</section>
<section>
<h2>Run manifest</h2>
{_html_table(manifest, "manifest")}
</section>
<section>
<h2>Expanded plan</h2>
{_html_table(plan, "plan")}
</section>
<section>
<h2>Provenance</h2>
<pre>{html.escape(metadata_json)}</pre>
</section>
</main>
<script>
document.querySelectorAll('table.sortable th').forEach((th) => {{
  th.addEventListener('click', () => {{
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const index = Array.from(th.parentElement.children).indexOf(th);
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = th.dataset.sort !== 'asc';
    rows.sort((a, b) => {{
      const av = a.children[index].textContent.trim();
      const bv = b.children[index].textContent.trim();
      const an = Number(av), bn = Number(bv);
      const cmp = Number.isFinite(an) && Number.isFinite(bn) ? an - bn : av.localeCompare(bv);
      return asc ? cmp : -cmp;
    }});
    th.dataset.sort = asc ? 'asc' : 'desc';
    rows.forEach((row) => tbody.appendChild(row));
  }});
}});
</script>
</body>
</html>
"""


def write_benchmark_outputs(
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    metadata: Mapping[str, object],
    *,
    write_html: bool = True,
) -> dict[str, Path]:
    """Write aggregate benchmark CSVs and optional HTML report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    if write_html:
        html_path = output_dir / "trail_digital_twin_benchmark_report.html"
        html_path.write_text(render_benchmark_html(tables, metadata))
        written["html_report"] = html_path
    return written


def _backfill_segment_type_metrics_from_run_outputs(
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    manifest = tables.get("benchmark_manifest", pd.DataFrame())
    if manifest.empty or "runId" not in manifest.columns:
        return pd.DataFrame()
    plan = tables.get("benchmark_plan", pd.DataFrame())
    plan_by_run = (
        {str(row["runId"]): row for row in plan.to_dict("records")}
        if not plan.empty and "runId" in plan.columns
        else {}
    )
    frames: list[pd.DataFrame] = []
    for row in manifest.to_dict("records"):
        run_id = str(row.get("runId", ""))
        if not run_id:
            continue
        run_dir_value = row.get("runOutputDir", "")
        run_dir = Path(str(run_dir_value)) if run_dir_value else output_dir / "runs" / run_id
        if not run_dir.is_absolute():
            run_dir = output_dir / run_dir
        metrics_path = run_dir / "table_segment_type_metrics.csv"
        predictions_path = run_dir / "segment_predictions.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
        elif predictions_path.exists():
            metrics = _segment_type_metrics(pd.read_csv(predictions_path))
        else:
            continue
        if metrics.empty:
            continue

        plan_row = plan_by_run.get(run_id, {})
        enriched = metrics.copy()
        enriched.insert(0, "runId", run_id)
        enriched.insert(1, "experimentGroup", row.get("experimentGroup", plan_row.get("experimentGroup", "")))
        enriched.insert(2, "experimentName", row.get("experimentName", plan_row.get("experimentName", "")))
        for column, value in plan_row.items():
            if column not in enriched.columns and column != "runId":
                enriched[column] = value
        frames.append(enriched)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_benchmark_output_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Read saved aggregate benchmark CSV outputs from a previous run."""
    tables: dict[str, pd.DataFrame] = {}
    found = False
    for name, filename in BENCHMARK_TABLE_FILES.items():
        path = output_dir / filename
        if path.exists():
            tables[name] = pd.read_csv(path)
            found = True
        else:
            tables[name] = pd.DataFrame()
    if not found:
        raise FileNotFoundError(f"no benchmark CSV outputs found in {output_dir}")
    if tables.get("benchmark_segment_type_metrics", pd.DataFrame()).empty:
        tables["benchmark_segment_type_metrics"] = _backfill_segment_type_metrics_from_run_outputs(output_dir, tables)
    return tables


def write_benchmark_html(
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    metadata: Mapping[str, object],
) -> Path:
    """Write only the aggregate benchmark HTML report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "trail_digital_twin_benchmark_report.html"
    html_path.write_text(render_benchmark_html(tables, metadata))
    return html_path
