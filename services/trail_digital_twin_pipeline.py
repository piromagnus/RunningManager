"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Configurable trail digital-twin extension pipeline.
"""

from __future__ import annotations

import html
import hashlib
import json
import logging
import math
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from services import trail_performance_model as tpm

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by CLI environment checks.
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


REQUIRED_CONFIG_SECTIONS = (
    "paths",
    "cohorts",
    "physiology",
    "readiness",
    "fitting",
    "segment_grid",
    "robustness",
    "outputs",
)
VALID_OBJECTIVES = {"activity", "segment"}
VALID_FATIGUE_MODELS = {"linear", "exponential"}
VALID_FATIGUE_STATES = {"decayed", "cumulative", "progress", "decayed_cumulative", "decayed_progress"}
VALID_FATIGUE_LOAD_COLUMNS = {"decayedTrimpBefore", "cumTrimpBefore", "progress"}
VALID_VALIDATION_MODES = {"in_sample", "loo"}
STAGE_ORDER = [
    "Stage 0 reproduction Stage 3",
    "Stage 1 TRIMP fatigue CTL",
    "Stage 2 TRIMP fatigue REDI",
    "Stage 3 HRR speed ratio",
]
TERRAIN_FAMILY_ORDER = {
    "steep_descent": 0,
    "descent": 1,
    "flat": 2,
    "mixed_climb_descent": 3,
    "climb": 4,
    "steep_climb": 5,
}
TERRAIN_FAMILY_LABELS = {
    "steep_descent": "Steep descent",
    "descent": "Descent",
    "flat": "Flat",
    "mixed_climb_descent": "Mixed climb/descent",
    "climb": "Ascent",
    "steep_climb": "Steep ascent",
}


DEFAULT_CONFIG: dict[str, Any] = {
    "execution": {
        "jobs": 1,
    },
    "paths": {
        "data_dir": "data",
        "timeseries_dir": "data/timeseries",
        "metrics_ts_dir": "data/metrics_ts",
        "raw_strava_dir": "data/raw/strava",
        "output_dir": "docs/science/pipeline_outputs/trail_digital_twin_extensions",
    },
    "cohorts": {
        "include": ["hardTrailRun", "hardRunOrTrailRun", "top10HardTrailByHRR", "selectedDateRaces"],
        # Empty = LOO on every included cohort. Non-empty restricts LOO to these names.
        "loo_include": [],
        # Optional cap for large cohorts (deterministic subsample); 0 = no cap.
        "loo_activity_cap": 0,
        "loo_activity_cap_seed": 20260721,
        "run_trail_over_20min_sec": 1200.0,
        "top_hrr_count": 10,
        "selected_race_dates": [],
    },
    "physiology": {
        "segment_km": 1.0,
        "vma_flat_kmh": 18.0,
        "vt2_threshold_name": "Threshold 30",
        "vt2_fallback_kmh": 15.0,
        "hrr_reference": 0.70,
        "hrr_min_factor": 0.55,
        "hrr_max_factor": 1.30,
        "decay_lambda": 0.30,
        "min_fatigue_factor": 0.50,
        # Asymmetric trail GAP on steep grades (1.0 = pure Minetti running).
        "gap_steep_threshold": 0.15,
        "gap_soft_start": 0.04,
        "gap_climb_scale": 0.85,
        "gap_descent_scale": 1.60,
    },
    "readiness": {
        "ctl_weight": 0.05,
        "tsb_weight": 0.10,
        "ctl_factor_min": 0.90,
        "ctl_factor_max": 1.08,
    },
    "fitting": {
        "enabled_objectives": ["activity", "segment"],
        "validation_modes": ["in_sample", "loo"],
        "stage0_alpha_grid": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
        "stage0_mu_grid": [-0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.00],
        "hrr_trimp_alpha_grid": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
        "hrr_trimp_kappa_grid": [0.0, 0.10, 0.20, 0.30, 0.40],
        "hrr_trimp_secondary_kappa_grid": [0.0, 0.10, 0.20, 0.30, 0.40],
        # reoptimize: re-fit (α, κ) after each component removal (preferred).
        # frozen: fit full model once, then disable components at prediction time.
        "ablation_protocol": "reoptimize",
        # If set, restrict component ablation to these fit objectives (e.g. ["activity"]).
        "ablation_fit_objectives": [],
        "fatigue_models": ["linear", "exponential"],
        "stage3_fatigue_states": [
            {"fatigue_state": "decayed", "acute_trimp_col": "decayedTrimpBefore", "label": "decayed TRIMP"},
            {"fatigue_state": "cumulative", "acute_trimp_col": "cumTrimpBefore", "label": "cumulative TRIMP"},
            {
                "fatigue_state": "progress",
                "acute_trimp_col": "progress",
                "label": "linear progress",
                "fatigue_models": ["linear"],
            },
            {
                "fatigue_state": "decayed_cumulative",
                "acute_trimp_col": "decayedTrimpBefore",
                "label": "decayed exp + cumulative TRIMP",
                "fatigue_models": ["exponential"],
                "secondary_acute_trimp_col": "cumTrimpBefore",
                "secondary_fatigue_model": "exponential",
            },
            {
                "fatigue_state": "decayed_progress",
                "acute_trimp_col": "decayedTrimpBefore",
                "label": "decayed exp + progress",
                "fatigue_models": ["exponential"],
                "secondary_acute_trimp_col": "progress",
                "secondary_fatigue_model": "exponential",
            },
        ],
    },
    "segment_grid": {
        "enabled": True,
        "alpha_grid": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "mu_grid": [-0.30, -0.15, 0.00],
        "terrain_multiplier_grid": [
            {"flat": 1.0, "climb": 1.0, "steep_climb": 1.0, "descent": 1.0, "steep_descent": 1.0}
        ],
        "technicality_coef_grid": [0.0, 0.20],
        "hr_coef_grid": [0.0, 0.35],
        "acute_trimp_coef_grid": [0.0, 0.04],
    },
    "robustness": {
        "enabled": False,
        "hrr_references": [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85],
        "decay_lambdas": [0.15, 0.30, 0.60],
        "elapsed_time_sensitivity": True,
        "segment_length_sensitivity": False,
        "segment_length_km": 0.50,
    },
    "segment_exclusion": {
        "enabled": False,
        "min_mean_speed_eq_kmh": 3.0,
        "max_stationary_time_share": 0.40,
        "stationary_speed_kmh": 1.0,
        "max_abs_altitude_rate_mph": 120.0,
        # Strip stationary dwell from the Stage 3 fit target (keep full race eval).
        "use_moving_time_for_fit": False,
        "exclude_from_fit": True,
        "report_full_race_eval": True,
    },
    "outputs": {
        "write_csv": True,
        "write_html": True,
        "write_png": False,
        "self_contained_html": True,
        "privacy_mode": "local",
        "html_filename": "trail_digital_twin_report.html",
    },
}


@dataclass
class PipelineResult:
    """Container for pipeline outputs before they are written."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_float_list(values: object, field_name: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    return [float(value) for value in values]


def _as_str_list(values: object, field_name: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    return [str(value) for value in values]


def _positive_int(value: object, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _normalise_stage3_states(states: object) -> list[dict[str, object]]:
    if not isinstance(states, list) or not states:
        raise ValueError("fitting.stage3_fatigue_states must be a non-empty list")
    normalised: list[dict[str, object]] = []
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("each Stage 3 fatigue state must be a mapping")
        fatigue_state = str(state.get("fatigue_state", state.get("fatigueState", "")))
        acute_col = str(state.get("acute_trimp_col", state.get("acuteTrimpCol", "")))
        label = str(state.get("label", fatigue_state or acute_col))
        if fatigue_state not in VALID_FATIGUE_STATES:
            raise ValueError(f"invalid fatigue state: {fatigue_state}")
        if acute_col not in VALID_FATIGUE_LOAD_COLUMNS:
            raise ValueError(f"invalid fatigue load column: {acute_col}")
        normalised_state: dict[str, object] = {
            "fatigue_state": fatigue_state,
            "acute_trimp_col": acute_col,
            "label": label,
        }
        secondary_col = state.get("secondary_acute_trimp_col", state.get("secondaryAcuteTrimpCol"))
        if secondary_col is not None:
            secondary_col = str(secondary_col)
            if secondary_col not in VALID_FATIGUE_LOAD_COLUMNS:
                raise ValueError(f"invalid secondary fatigue load column: {secondary_col}")
            secondary_model = str(
                state.get("secondary_fatigue_model", state.get("secondaryFatigueModel", "exponential"))
            )
            if secondary_model not in VALID_FATIGUE_MODELS:
                raise ValueError(f"invalid secondary fatigue model: {secondary_model}")
            normalised_state["secondary_acute_trimp_col"] = secondary_col
            normalised_state["secondary_fatigue_model"] = secondary_model
        state_models = state.get("fatigue_models", state.get("fatigueModels"))
        if state_models is not None:
            models = _as_str_list(state_models, "fitting.stage3_fatigue_states[].fatigue_models")
            invalid_models = sorted(set(models) - VALID_FATIGUE_MODELS)
            if invalid_models:
                raise ValueError(f"invalid fatigue models: {', '.join(invalid_models)}")
            normalised_state["fatigue_models"] = models
        normalised.append(normalised_state)
    return normalised


def normalise_config(raw_config: Mapping[str, Any], *, require_sections: bool = True) -> dict[str, Any]:
    """Merge config with defaults and validate user-controlled choices."""
    if require_sections:
        missing = [section for section in REQUIRED_CONFIG_SECTIONS if section not in raw_config]
        if missing:
            raise ValueError(f"missing required config sections: {', '.join(missing)}")

    config = _deep_merge(DEFAULT_CONFIG, raw_config)
    config.setdefault("execution", {})
    config["execution"]["jobs"] = _positive_int(config["execution"].get("jobs", 1), "execution.jobs")
    fitting = config["fitting"]
    objectives = _as_str_list(fitting.get("enabled_objectives"), "fitting.enabled_objectives")
    invalid_objectives = sorted(set(objectives) - VALID_OBJECTIVES)
    if invalid_objectives:
        raise ValueError(f"invalid fitting objectives: {', '.join(invalid_objectives)}")
    fitting["enabled_objectives"] = objectives

    validation_modes = _as_str_list(fitting.get("validation_modes"), "fitting.validation_modes")
    invalid_validation_modes = sorted(set(validation_modes) - VALID_VALIDATION_MODES)
    if invalid_validation_modes:
        raise ValueError(f"invalid validation modes: {', '.join(invalid_validation_modes)}")
    fitting["validation_modes"] = validation_modes

    ablation_protocol = str(fitting.get("ablation_protocol", "reoptimize")).strip().lower()
    if ablation_protocol not in {"reoptimize", "frozen"}:
        raise ValueError("fitting.ablation_protocol must be 'reoptimize' or 'frozen'")
    fitting["ablation_protocol"] = ablation_protocol
    ablation_objectives = fitting.get("ablation_fit_objectives") or []
    if ablation_objectives:
        ablation_objectives = _as_str_list(ablation_objectives, "fitting.ablation_fit_objectives")
        invalid_ablation_objectives = sorted(set(ablation_objectives) - VALID_OBJECTIVES)
        if invalid_ablation_objectives:
            raise ValueError(
                f"invalid ablation fit objectives: {', '.join(invalid_ablation_objectives)}"
            )
    fitting["ablation_fit_objectives"] = list(ablation_objectives)

    fatigue_models = _as_str_list(fitting.get("fatigue_models"), "fitting.fatigue_models")
    invalid_models = sorted(set(fatigue_models) - VALID_FATIGUE_MODELS)
    if invalid_models:
        raise ValueError(f"invalid fatigue models: {', '.join(invalid_models)}")
    fitting["fatigue_models"] = fatigue_models
    fitting["stage3_fatigue_states"] = _normalise_stage3_states(fitting.get("stage3_fatigue_states"))

    for key in [
        "stage0_alpha_grid",
        "stage0_mu_grid",
        "hrr_trimp_alpha_grid",
        "hrr_trimp_kappa_grid",
        "hrr_trimp_secondary_kappa_grid",
    ]:
        fitting[key] = _as_float_list(fitting.get(key), f"fitting.{key}")
    for key in ["hrr_references", "decay_lambdas"]:
        config["robustness"][key] = _as_float_list(config["robustness"].get(key), f"robustness.{key}")
    for key in ["alpha_grid", "mu_grid", "technicality_coef_grid", "hr_coef_grid", "acute_trimp_coef_grid"]:
        config["segment_grid"][key] = _as_float_list(config["segment_grid"].get(key), f"segment_grid.{key}")

    privacy_mode = str(config["outputs"].get("privacy_mode", "local"))
    if privacy_mode not in {"local", "anonymized"}:
        raise ValueError("outputs.privacy_mode must be 'local' or 'anonymized'")
    config["outputs"]["privacy_mode"] = privacy_mode
    return config


def load_config(path: Path, *, require_sections: bool = True) -> dict[str, Any]:
    """Load and validate a trail digital-twin YAML config."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to read trail digital-twin configs") from YAML_IMPORT_ERROR
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("config root must be a mapping")
    return normalise_config(raw, require_sections=require_sections)


def _project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
    return current


def _resolve_path(project_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def _metrics_row(cohort: str, stage: str, actual: Sequence[float], predicted: Sequence[float]) -> dict[str, object]:
    metrics = tpm.regression_metrics(actual, predicted)
    return {
        "cohort": cohort,
        "stage": stage,
        "r2": metrics["r2"],
        "maeMin": metrics["maeSec"] / 60.0,
        "mapePct": metrics["mapePct"],
        "biasMin": metrics["biasSec"] / 60.0,
    }


def _add_fit_objective(row: dict[str, object], objective: str) -> dict[str, object]:
    row["fitObjective"] = objective
    return row


def _model_objective(objective: str) -> str:
    return "segment" if objective == "segment" else "race"


def _segment_exclusion_kwargs(config: Mapping[str, Any]) -> dict[str, object]:
    exclusion = config.get("segment_exclusion", {}) or {}
    # Prefer grade-adjusted speed; accept deprecated raw-speed key for old YAMLs.
    min_speed_eq = exclusion.get("min_mean_speed_eq_kmh")
    if min_speed_eq is None and "min_mean_speed_kmh" in exclusion:
        logger.warning(
            "segment_exclusion.min_mean_speed_kmh is deprecated; "
            "map it to min_mean_speed_eq_kmh"
        )
        min_speed_eq = exclusion.get("min_mean_speed_kmh")
    if "max_abs_grade" in exclusion and "max_abs_altitude_rate_mph" not in exclusion:
        logger.warning(
            "segment_exclusion.max_abs_grade is deprecated; "
            "use max_abs_altitude_rate_mph (altitude over time). "
            "Falling back to default max_abs_altitude_rate_mph=120"
        )
    return {
        "enabled": bool(exclusion.get("enabled", False)),
        "min_mean_speed_eq_kmh": float(min_speed_eq if min_speed_eq is not None else 3.0),
        "max_stationary_time_share": float(exclusion.get("max_stationary_time_share", 0.40)),
        "stationary_speed_kmh": float(exclusion.get("stationary_speed_kmh", 1.0)),
        "max_abs_altitude_rate_mph": float(exclusion.get("max_abs_altitude_rate_mph", 120.0)),
    }


def _fit_mask_col(config: Mapping[str, Any]) -> str | None:
    exclusion = config.get("segment_exclusion", {}) or {}
    if bool(exclusion.get("enabled", False)) and bool(exclusion.get("exclude_from_fit", True)):
        return "isFitEligible"
    return None


def _fit_actual_time_col(config: Mapping[str, Any]) -> str:
    """Segment clock used for Stage 3 fit metrics (full race eval unchanged)."""
    exclusion = config.get("segment_exclusion", {}) or {}
    if bool(exclusion.get("use_moving_time_for_fit", False)):
        return "actualMovingTimeSec"
    return "actualTimeSec"


def _rename_load_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df.rename(
        columns={
            "load": f"{prefix}Load",
            "ctl": f"{prefix}Ctl",
            "atl": f"{prefix}Atl",
            "tsb": f"{prefix}Tsb",
        }
    )


def _add_readiness_factors(
    df: pd.DataFrame,
    value_col: str,
    balance_col: str,
    output_col: str,
    readiness_config: Mapping[str, Any],
) -> pd.DataFrame:
    result = df.copy()
    reference = float(pd.to_numeric(result[value_col], errors="coerce").median())
    if not np.isfinite(reference) or reference <= 0:
        reference = 1.0
    result[output_col] = result.apply(
        lambda row: tpm.ctl_readiness_factor(
            row.get(value_col, np.nan),
            row.get(balance_col, 0.0),
            ctl_reference=reference,
            ctl_weight=float(readiness_config["ctl_weight"]),
            tsb_weight=float(readiness_config["tsb_weight"]),
            min_factor=float(readiness_config["ctl_factor_min"]),
            max_factor=float(readiness_config["ctl_factor_max"]),
        ),
        axis=1,
    )
    return result


def _threshold_speed(thresholds: pd.DataFrame, physiology_config: Mapping[str, Any]) -> float:
    threshold_name = str(physiology_config["vt2_threshold_name"])
    match = thresholds[thresholds["name"].astype(str).eq(threshold_name)]
    if match.empty:
        return float(physiology_config["vt2_fallback_kmh"])
    threshold_row = match.iloc[0]
    value = float(
        np.nanmean(
            [
                pd.to_numeric(threshold_row.get("paceFlatKmhMin"), errors="coerce"),
                pd.to_numeric(threshold_row.get("paceFlatKmhMax"), errors="coerce"),
            ]
        )
    )
    return value if np.isfinite(value) and value > 0 else float(physiology_config["vt2_fallback_kmh"])


def _load_activity_inputs(
    config: Mapping[str, Any], project_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    paths = config["paths"]
    data_dir = _resolve_path(project_root, paths["data_dir"])
    raw_strava_dir = _resolve_path(project_root, paths["raw_strava_dir"])
    activities = pd.read_csv(data_dir / "activities.csv", dtype={"activityId": str})
    activity_metrics = pd.read_csv(data_dir / "activities_metrics.csv", dtype={"activityId": str})
    athletes = pd.read_csv(data_dir / "athlete.csv")
    thresholds = pd.read_csv(data_dir / "thresholds.csv")
    daily_metrics = pd.read_csv(data_dir / "daily_metrics.csv")

    metric_cols = [
        "activityId",
        "category",
        "distanceEqKm",
        "trimp",
        "hrSpeedShift",
        "hrZone_z1_upper",
        "hrZone_z2_upper",
        "hrZone_z3_upper",
        "hrZone_z4_upper",
    ]
    metric_cols = [col for col in metric_cols if col in activity_metrics.columns]
    activity_df = activities.merge(activity_metrics[metric_cols], on="activityId", how="left")

    athlete = athletes.iloc[0]
    hr_rest = float(athlete.get("hrRest", 60.0))
    hr_max = float(athlete.get("hrMax", 190.0))
    v_vt2_kmh = _threshold_speed(thresholds, config["physiology"])

    activity_df = tpm.add_hr_reserve(activity_df, hr_rest=hr_rest, hr_max=hr_max)
    activity_df["actualTimeSec"] = pd.to_numeric(activity_df["movingSec"], errors="coerce")
    activity_df["actualTimeSec"] = activity_df["actualTimeSec"].where(
        activity_df["actualTimeSec"] > 0,
        pd.to_numeric(activity_df["elapsedSec"], errors="coerce"),
    )
    activity_df["startDate"] = pd.to_datetime(activity_df["startTime"], errors="coerce").dt.date
    activity_df["category"] = activity_df["category"].astype(str)

    trimp_load = _rename_load_features(tpm.compute_ctl_atl_tsb(daily_metrics, load_col="trimp"), "trimp")
    distance_eq_load = _rename_load_features(
        tpm.compute_ctl_atl_tsb(daily_metrics, load_col="distanceEqKm"),
        "distanceEq",
    )
    vertical_load = _rename_load_features(tpm.compute_ctl_atl_tsb(daily_metrics, load_col="ascentM"), "vertical")
    redi_load = tpm.compute_redi_load_features(daily_metrics, load_col="trimp", prefix="trimp")
    for feature_df in [trimp_load, distance_eq_load, vertical_load, redi_load]:
        cols = [col for col in feature_df.columns if col != "date"]
        activity_df = tpm.attach_previous_daily_features(activity_df, feature_df, feature_cols=cols)
    activity_df = activity_df.rename(columns={"trimpCtl": "ctl", "trimpAtl": "atl", "trimpTsb": "tsb"})

    weather_records = []
    for activity_id in activity_df["activityId"].astype(str):
        raw_path = raw_strava_dir / f"{activity_id}.json"
        parsed = {"activityId": activity_id, "temperatureC": np.nan, "weatherSource": ""}
        if raw_path.exists():
            try:
                detail = json.loads(raw_path.read_text())
                temperature = pd.to_numeric(detail.get("average_temp"), errors="coerce")
                if pd.notna(temperature):
                    parsed["temperatureC"] = float(temperature)
                    parsed["weatherSource"] = "strava_average_temp"
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        weather_records.append(parsed)
    activity_df = activity_df.merge(pd.DataFrame(weather_records), on="activityId", how="left")
    return activity_df, daily_metrics, hr_rest, hr_max, v_vt2_kmh


def _load_processed_timeseries(activity_id: str, paths: Mapping[str, Path]) -> pd.DataFrame:
    raw_path = paths["timeseries_dir"] / f"{activity_id}.csv"
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        processed = tpm.prepare_raw_timeseries_for_segments(raw_df)
        if not processed.empty and "cumulated_distance" in processed.columns:
            if "speed_km_h" in processed.columns and "grade_ma_10" in processed.columns:
                processed["speed_eq_km_h"] = processed["speed_km_h"] * processed["grade_ma_10"].map(tpm.gap_factor)
            return processed

    metrics_path = paths["metrics_ts_dir"] / f"{activity_id}.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return pd.DataFrame()


def _build_segments(
    activity_df: pd.DataFrame,
    config: Mapping[str, Any],
    project_root: Path,
    hr_rest: float,
    hr_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    paths = {
        "timeseries_dir": _resolve_path(project_root, config["paths"]["timeseries_dir"]),
        "metrics_ts_dir": _resolve_path(project_root, config["paths"]["metrics_ts_dir"]),
    }
    candidates = activity_df[
        activity_df["category"].astype(str).str.upper().isin(["TRAIL_RUN", "RUN"])
        & activity_df["hasTimeseries"].astype(str).str.lower().isin(["true", "1", "yes"])
    ].copy()
    segment_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    segments_by_activity: dict[str, pd.DataFrame] = {}
    for _, row in candidates.iterrows():
        activity_id = str(row["activityId"])
        ts_df = _load_processed_timeseries(activity_id, paths)
        segments = tpm.segment_timeseries(
            ts_df,
            segment_km=float(config["physiology"]["segment_km"]),
            hr_rest=hr_rest,
            hr_max=hr_max,
        )
        if not segments.empty:
            segments = tpm.apply_segment_exclusion(segments, **_segment_exclusion_kwargs(config))
        usable = not segments.empty and segments["distanceKm"].sum() > 0 and segments["actualTimeSec"].sum() > 0
        fit_eligible_count = (
            int(segments["isFitEligible"].fillna(False).astype(bool).sum())
            if usable and "isFitEligible" in segments.columns
            else (len(segments) if usable else 0)
        )
        excluded_count = int(len(segments) - fit_eligible_count) if usable else 0
        excluded_time = 0.0
        if usable and "isFitEligible" in segments.columns:
            excluded_time = float(
                segments.loc[~segments["isFitEligible"].fillna(False).astype(bool), "actualTimeSec"].sum()
            )
        hr_sample_count = (
            int(pd.to_numeric(segments.get("hrSampleCount"), errors="coerce").fillna(0).sum()) if usable else 0
        )
        hr_valid_sample_count = (
            int(pd.to_numeric(segments.get("hrValidSampleCount"), errors="coerce").fillna(0).sum())
            if usable
            else 0
        )
        hr_valid_share = (
            float(hr_valid_sample_count) / float(hr_sample_count) if hr_sample_count > 0 else float("nan")
        )
        qc_rows.append(
            {
                "activityId": activity_id,
                "name": row.get("name"),
                "category": row.get("category"),
                "usableSegments": usable,
                "segmentCount": len(segments),
                "fitEligibleSegmentCount": fit_eligible_count,
                "excludedSegmentCount": excluded_count,
                "excludedTimeSec": excluded_time,
                "segmentDistanceKm": float(segments["distanceKm"].sum()) if usable else 0.0,
                "segmentTimeSec": float(segments["actualTimeSec"].sum()) if usable else 0.0,
                "hrSampleCount": hr_sample_count,
                "hrValidSampleCount": hr_valid_sample_count,
                "hrValidShare": hr_valid_share,
            }
        )
        if usable:
            segments = segments.copy()
            segments["activityId"] = activity_id
            if "terrainFamily" not in segments.columns:
                segments["terrainFamily"] = segments["avgGrade"].map(tpm.terrain_family)
            else:
                segments["terrainFamily"] = segments["terrainFamily"].fillna(
                    segments["avgGrade"].map(tpm.terrain_family)
                )
            segments_by_activity[activity_id] = segments
            segment_frames.append(segments)
    all_segments_df = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    return all_segments_df, pd.DataFrame(qc_rows), segments_by_activity


def add_segment_model_features(segments: pd.DataFrame, activity_features: pd.DataFrame) -> pd.DataFrame:
    """Add diagnostic and constrained-model features to segment rows."""
    df = segments.merge(activity_features, on="activityId", how="left", suffixes=("", "_activity"))
    df["logActualTimeSec"] = np.log(pd.to_numeric(df["actualTimeSec"], errors="coerce").clip(lower=1.0))
    df["logDistanceKm"] = np.log(pd.to_numeric(df["distanceKm"], errors="coerce").clip(lower=1e-3))
    fallback_gap = pd.to_numeric(df["avgGrade"], errors="coerce").fillna(0.0).map(tpm.gap_factor)
    integrated_gap = df.get("gapFactorIntegrated", pd.Series(np.nan, index=df.index))
    df["gapFactor"] = pd.to_numeric(integrated_gap, errors="coerce").fillna(fallback_gap)
    df["logGapFactor"] = np.log(df["gapFactor"].clip(lower=1e-6))
    df["altitudePenalty"] = 1.0 - pd.to_numeric(df["meanAltitudeM"], errors="coerce").fillna(0.0).map(
        tpm.altitude_factor
    )
    df["ascentPerKm"] = pd.to_numeric(df["elevGainM"], errors="coerce").fillna(0.0) / df["distanceKm"].clip(lower=1e-3)
    df["descentPerKm"] = pd.to_numeric(df["elevLossM"], errors="coerce").fillna(0.0) / df["distanceKm"].clip(lower=1e-3)
    df["meanHrReserve"] = pd.to_numeric(df["meanHrReserve"], errors="coerce")
    df["technicalityCombined"] = pd.to_numeric(df.get("technicalityGps"), errors="coerce").fillna(0.0)
    df["temperatureC"] = pd.to_numeric(df.get("temperatureC", pd.Series(np.nan, index=df.index)), errors="coerce")
    for col in [
        "ctl",
        "tsb",
        "trimpRediSlow",
        "trimpRediBalance",
        "distanceEqCtl",
        "distanceEqTsb",
        "verticalCtl",
        "verticalTsb",
        "cumTrimpBefore",
        "decayedTrimpBefore",
        "progress",
    ]:
        df[col] = pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")
    dummies = pd.get_dummies(df["terrainFamily"], prefix="terrain", dtype=float)
    return pd.concat([df, dummies], axis=1)


def _build_cohorts(
    activity_df: pd.DataFrame,
    segments_by_activity: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    activity_df = activity_df.copy()
    activity_df["usableSegmentActivity"] = activity_df["activityId"].astype(str).isin(segments_by_activity)
    activity_df["usableTrailRun"] = (
        activity_df["category"].astype(str).str.upper().eq("TRAIL_RUN") & activity_df["usableSegmentActivity"]
    )
    activity_df["hardTrailRun"] = tpm.hard_trailrun_mask(activity_df) & activity_df["usableSegmentActivity"]
    category = activity_df["category"].astype(str).str.upper()
    moving = pd.to_numeric(activity_df.get("movingSec"), errors="coerce").fillna(0.0)
    distance = pd.to_numeric(activity_df.get("distanceKm"), errors="coerce").fillna(0.0)
    ascent = pd.to_numeric(activity_df.get("ascentM"), errors="coerce").fillna(0.0)
    reserve = pd.to_numeric(activity_df.get("hrReserveRatio"), errors="coerce").fillna(0.0)
    hard_activity = (distance >= 10.0) | (ascent >= 500.0) | (reserve >= 0.70)
    activity_df["hardRunOrTrailRun"] = (
        category.isin(["RUN", "TRAIL_RUN"]) & activity_df["usableSegmentActivity"] & (moving >= 1800.0) & hard_activity
    )
    over20_sec = float(config["cohorts"].get("run_trail_over_20min_sec", 1200.0))
    activity_df["runTrailOver20Min"] = (
        category.isin(["RUN", "TRAIL_RUN"]) & activity_df["usableSegmentActivity"] & (moving >= over20_sec)
    )

    hard_activity_df = activity_df[activity_df["hardTrailRun"]].copy()
    hard_run_or_trail_df = activity_df[activity_df["hardRunOrTrailRun"]].copy()
    run_trail_over20_df = activity_df[activity_df["runTrailOver20Min"]].copy()
    top_ids = tpm.top_hrr_hard_trailrun_ids(activity_df, n=int(config["cohorts"]["top_hrr_count"]))
    top_activity_df = activity_df[activity_df["activityId"].astype(str).isin(top_ids)].copy()
    selected_activity_df = tpm.select_best_activity_by_dates(activity_df, config["cohorts"]["selected_race_dates"])
    selected_activity_df = selected_activity_df[
        selected_activity_df["activityId"].astype(str).isin(segments_by_activity)
    ].copy()
    all_cohorts = {
        "hardTrailRun": hard_activity_df,
        "hardRunOrTrailRun": hard_run_or_trail_df,
        "runTrailOver20Min": run_trail_over20_df,
        "top10HardTrailByHRR": top_activity_df,
        "selectedDateRaces": selected_activity_df,
    }
    included = set(config["cohorts"]["include"])
    return {name: df for name, df in all_cohorts.items() if name in included}


def _cohort_descriptives(cohorts: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort_name, cohort_df in cohorts.items():
        rows.append(
            {
                "cohort": cohort_name,
                "activityCount": int(len(cohort_df)),
                "distanceKmTotal": pd.to_numeric(cohort_df.get("distanceKm"), errors="coerce").sum(),
                "distanceKmMedian": pd.to_numeric(cohort_df.get("distanceKm"), errors="coerce").median(),
                "ascentMTotal": pd.to_numeric(cohort_df.get("ascentM"), errors="coerce").sum(),
                "ascentMMedian": pd.to_numeric(cohort_df.get("ascentM"), errors="coerce").median(),
                "durationMinMedian": pd.to_numeric(cohort_df.get("actualTimeSec"), errors="coerce").median() / 60.0,
                "durationMinTotal": pd.to_numeric(cohort_df.get("actualTimeSec"), errors="coerce").sum() / 60.0,
                "hrrMean": pd.to_numeric(cohort_df.get("hrReserveRatio"), errors="coerce").mean(),
                "hrrMedian": pd.to_numeric(cohort_df.get("hrReserveRatio"), errors="coerce").median(),
                "ctlMedian": pd.to_numeric(cohort_df.get("ctl"), errors="coerce").median(),
                "tsbMedian": pd.to_numeric(cohort_df.get("tsb"), errors="coerce").median(),
                "rediSlowMedian": pd.to_numeric(cohort_df.get("trimpRediSlow"), errors="coerce").median(),
                "rediBalanceMedian": pd.to_numeric(cohort_df.get("trimpRediBalance"), errors="coerce").median(),
            }
        )
    return pd.DataFrame(rows)


def _prediction_frame(
    cohort_name: str,
    model_name: str,
    objective: str,
    cohort_df: pd.DataFrame,
    actual: pd.Series,
    predicted: pd.Series,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "activityId": actual.index.astype(str),
            "actualTimeSec": actual.to_numpy(dtype=float),
            "predictedTimeSec": predicted.reindex(actual.index).to_numpy(dtype=float),
        }
    )
    meta_cols = [
        col
        for col in ["activityId", "startDate", "name", "distanceKm", "ascentM", "hrReserveRatio"]
        if col in cohort_df.columns
    ]
    frame = frame.merge(cohort_df[meta_cols].astype({"activityId": str}), on="activityId", how="left")
    frame["cohort"] = cohort_name
    frame["model"] = model_name
    frame["fitObjective"] = objective
    frame["actualMin"] = frame["actualTimeSec"] / 60.0
    frame["predictedMin"] = frame["predictedTimeSec"] / 60.0
    frame["errorMin"] = (frame["predictedTimeSec"] - frame["actualTimeSec"]) / 60.0
    return frame


def _cohort_inputs(
    cohort_df: pd.DataFrame,
    segments_by_activity: Mapping[str, pd.DataFrame],
) -> tuple[list[str], dict[str, pd.DataFrame], pd.Series, dict[str, float]]:
    ids = cohort_df["activityId"].astype(str).tolist()
    segments = {
        activity_id: segments_by_activity[activity_id] for activity_id in ids if activity_id in segments_by_activity
    }
    ids = [activity_id for activity_id in ids if activity_id in segments]
    observed = cohort_df.set_index("activityId").loc[ids, "actualTimeSec"].astype(float)
    ctl_factors = cohort_df.set_index("activityId").loc[ids, "ctlReadinessFactor"].astype(float).to_dict()
    return ids, segments, observed, ctl_factors


def _race_prediction_from_segments(prediction: pd.DataFrame, ids: list[str]) -> pd.Series:
    return prediction.groupby("activityId")["predictedTimeSec"].sum().reindex(ids)


def _linear_coefficient_frame(
    cohort_name: str,
    stage_name: str,
    df: pd.DataFrame,
    fitted: tpm.RegressionModel,
    target_col: str = "logActualTimeSec",
) -> pd.DataFrame:
    target = pd.to_numeric(df[target_col], errors="coerce")
    target_std = float(target.std(ddof=0)) if target.notna().any() else np.nan
    rows = [
        {
            "cohort": cohort_name,
            "stage": stage_name,
            "feature": "intercept",
            "coefficient": float(fitted.coefficients[0]),
            "standardizedCoefficient": np.nan,
            "absStandardizedCoefficient": np.nan,
        }
    ]
    for feature, coefficient in zip(fitted.feature_cols, fitted.coefficients[1:]):
        values = pd.to_numeric(df[feature], errors="coerce")
        feature_std = float(values.std(ddof=0)) if values.notna().any() else np.nan
        standardized = coefficient * feature_std / target_std if target_std and np.isfinite(target_std) else np.nan
        rows.append(
            {
                "cohort": cohort_name,
                "stage": stage_name,
                "feature": feature,
                "coefficient": float(coefficient),
                "standardizedCoefficient": float(standardized) if np.isfinite(standardized) else np.nan,
                "absStandardizedCoefficient": abs(float(standardized)) if np.isfinite(standardized) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _progressive_feature_sets(segment_features: pd.DataFrame) -> dict[str, list[str]]:
    terrain_cols = [col for col in segment_features.columns if col.startswith("terrain_")]
    return {
        "E1 distance": ["logDistanceKm"],
        "E2 terrain physics + progress": [
            "logDistanceKm",
            "logGapFactor",
            "altitudePenalty",
            "progress",
            *terrain_cols,
        ],
        "E3 CTL readiness": [
            "logDistanceKm",
            "logGapFactor",
            "altitudePenalty",
            "progress",
            "ctl",
            "tsb",
            *terrain_cols,
        ],
        "E4 REDI readiness": [
            "logDistanceKm",
            "logGapFactor",
            "altitudePenalty",
            "progress",
            "trimpRediSlow",
            "trimpRediBalance",
            *terrain_cols,
        ],
        "E5 HR effort": [
            "logDistanceKm",
            "logGapFactor",
            "altitudePenalty",
            "progress",
            "meanHrReserve",
            "ctl",
            "tsb",
            *terrain_cols,
        ],
        "E6 decayed acute TRIMP": [
            "logDistanceKm",
            "logGapFactor",
            "altitudePenalty",
            "meanHrReserve",
            "ctl",
            "tsb",
            "decayedTrimpBefore",
            *terrain_cols,
        ],
        "E6bis cumulative acute TRIMP": [
            "logDistanceKm",
            "logGapFactor",
            "altitudePenalty",
            "meanHrReserve",
            "ctl",
            "tsb",
            "cumTrimpBefore",
            *terrain_cols,
        ],
    }


def run_linear_diagnostics(
    cohorts: Mapping[str, pd.DataFrame],
    segment_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run E1-E6/E6bis linear diagnostic fits."""
    feature_sets = _progressive_feature_sets(segment_features)
    loo_stages = ["E6 decayed acute TRIMP", "E6bis cumulative acute TRIMP"]
    rows: list[dict[str, object]] = []
    coefficient_frames: list[pd.DataFrame] = []
    for cohort_name, cohort_df in cohorts.items():
        ids = cohort_df["activityId"].astype(str).tolist()
        cohort_segments = segment_features[segment_features["activityId"].isin(ids)].copy()
        if cohort_segments.empty:
            continue
        actual = cohort_segments.groupby("activityId")["actualTimeSec"].sum()
        for stage_name, features in feature_sets.items():
            try:
                fitted = tpm.fit_linear_regression(cohort_segments, features, "logActualTimeSec")
            except ValueError:
                continue
            predicted_segment = np.exp(tpm.predict_linear_regression(cohort_segments, fitted))
            prediction = cohort_segments.assign(predictedTimeSec=predicted_segment)
            predicted = prediction.groupby("activityId")["predictedTimeSec"].sum().reindex(actual.index)
            rows.append(_metrics_row(cohort_name, f"{stage_name} in-sample", actual, predicted))
            coefficient_frames.append(_linear_coefficient_frame(cohort_name, stage_name, cohort_segments, fitted))

            if stage_name in loo_stages:
                loo_rows = []
                for held_out in actual.index.astype(str):
                    train = cohort_segments[cohort_segments["activityId"].astype(str).ne(held_out)]
                    test = cohort_segments[cohort_segments["activityId"].astype(str).eq(held_out)]
                    if train.empty or test.empty:
                        continue
                    try:
                        loo_fitted = tpm.fit_linear_regression(train, features, "logActualTimeSec")
                    except ValueError:
                        continue
                    predicted = float(np.exp(tpm.predict_linear_regression(test, loo_fitted)).sum())
                    loo_rows.append(
                        {
                            "activityId": held_out,
                            "actualTimeSec": float(actual.loc[held_out]),
                            "predictedTimeSec": predicted,
                        }
                    )
                loo = pd.DataFrame(loo_rows)
                if not loo.empty:
                    rows.append(
                        _metrics_row(cohort_name, f"{stage_name} LOO", loo["actualTimeSec"], loo["predictedTimeSec"])
                    )
    variable_importance = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()
    return pd.DataFrame(rows), variable_importance


def run_hr_baselines(
    activity_df: pd.DataFrame,
    segment_features: pd.DataFrame,
    cohorts: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Run global activity HR and segment HR E6 LOO baselines."""
    all_trail_ids = activity_df[activity_df["usableTrailRun"]]["activityId"].astype(str).tolist()
    all_activity_features = activity_df[activity_df["activityId"].astype(str).isin(all_trail_ids)].copy()
    all_activity_features["distanceEqKm"] = pd.to_numeric(all_activity_features["distanceEqKm"], errors="coerce")
    all_activity_features["distanceEqKm"] = all_activity_features["distanceEqKm"].fillna(
        pd.to_numeric(all_activity_features["distanceKm"], errors="coerce")
        + pd.to_numeric(all_activity_features["ascentM"], errors="coerce").fillna(0.0) * 0.01
    )
    all_activity_features["logActualTimeSec"] = np.log(all_activity_features["actualTimeSec"].clip(lower=1.0))
    all_activity_features["logDistanceEqKm"] = np.log(all_activity_features["distanceEqKm"].clip(lower=1e-3))
    all_activity_features["ascentPerKm"] = pd.to_numeric(
        all_activity_features["ascentM"],
        errors="coerce",
    ).fillna(0.0) / pd.to_numeric(all_activity_features["distanceKm"], errors="coerce").clip(lower=1e-3)
    all_activity_features["technicalityCombined"] = pd.to_numeric(
        all_activity_features.get("technicalityGps"), errors="coerce"
    )
    hr_global_features = [
        "logDistanceEqKm",
        "ascentPerKm",
        "hrReserveRatio",
        "ctl",
        "tsb",
        "trimpRediSlow",
        "trimpRediBalance",
        "technicalityCombined",
        "meanAltitudeM",
        "temperatureC",
    ]
    hr_global_rows = []
    for held_out in all_trail_ids:
        train = all_activity_features[all_activity_features["activityId"].ne(held_out)]
        test = all_activity_features[all_activity_features["activityId"].eq(held_out)]
        if train.empty or test.empty:
            continue
        try:
            fitted = tpm.fit_linear_regression(train, hr_global_features, "logActualTimeSec")
        except ValueError:
            continue
        predicted = float(np.exp(tpm.predict_linear_regression(test, fitted))[0])
        actual = float(test["actualTimeSec"].iloc[0])
        hr_global_rows.append({"activityId": held_out, "actualTimeSec": actual, "predictedTimeSec": predicted})
    hr_global_loo = pd.DataFrame(hr_global_rows)

    feature_sets = _progressive_feature_sets(segment_features)
    all_trail_segments = segment_features[segment_features["activityId"].isin(all_trail_ids)].copy()
    hr_segment_features = feature_sets["E6 decayed acute TRIMP"]
    hr_segment_rows = []
    for held_out in all_trail_ids:
        train = all_trail_segments[all_trail_segments["activityId"].ne(held_out)]
        test = all_trail_segments[all_trail_segments["activityId"].eq(held_out)]
        if train.empty or test.empty:
            continue
        try:
            fitted = tpm.fit_linear_regression(train, hr_segment_features, "logActualTimeSec")
        except ValueError:
            continue
        predicted = float(np.exp(tpm.predict_linear_regression(test, fitted)).sum())
        actual = float(all_activity_features.set_index("activityId").loc[held_out, "actualTimeSec"])
        hr_segment_rows.append({"activityId": held_out, "actualTimeSec": actual, "predictedTimeSec": predicted})
    hr_segment_loo = pd.DataFrame(hr_segment_rows)

    rows: list[dict[str, object]] = []
    for cohort_name, cohort_df in cohorts.items():
        ids = cohort_df["activityId"].astype(str).tolist()
        global_eval = (
            hr_global_loo[hr_global_loo["activityId"].isin(ids)] if not hr_global_loo.empty else pd.DataFrame()
        )
        segment_eval = (
            hr_segment_loo[hr_segment_loo["activityId"].isin(ids)] if not hr_segment_loo.empty else pd.DataFrame()
        )
        if not global_eval.empty:
            rows.append(
                _metrics_row(
                    cohort_name, "HR global LOO", global_eval["actualTimeSec"], global_eval["predictedTimeSec"]
                )
            )
        if not segment_eval.empty:
            rows.append(
                _metrics_row(
                    cohort_name, "HR E6 segment LOO", segment_eval["actualTimeSec"], segment_eval["predictedTimeSec"]
                )
            )
    return pd.DataFrame(rows)


def _stage3_metadata(
    row: dict[str, object],
    *,
    validation: str,
    objective: str,
    fatigue_state: str,
    acute_trimp_col: str,
    fatigue_model: str,
    alpha: float,
    fatigue_coef: float,
    secondary_acute_trimp_col: str = "",
    secondary_fatigue_model: str = "",
    secondary_fatigue_coef: float = 0.0,
) -> dict[str, object]:
    row.update(
        {
            "validation": validation,
            "fitObjective": objective,
            "fatigueState": fatigue_state,
            "acuteTrimpCol": acute_trimp_col,
            "fatigueModel": fatigue_model,
            "alpha": alpha,
            "fatigueCoef": fatigue_coef,
            "secondaryAcuteTrimpCol": secondary_acute_trimp_col,
            "secondaryFatigueModel": secondary_fatigue_model,
            "secondaryFatigueCoef": secondary_fatigue_coef,
        }
    )
    return row


def _paper_stage_specs() -> list[dict[str, object]]:
    return [
        {
            "stage": "Stage 1 TRIMP fatigue CTL",
            "loadFactorCol": "ctlReadinessFactor",
            "useHrrEffort": False,
            "fatigueState": "decayed",
            "acuteTrimpCol": "decayedTrimpBefore",
        },
        {
            "stage": "Stage 2 TRIMP fatigue REDI",
            "loadFactorCol": "rediReadinessFactor",
            "useHrrEffort": False,
            "fatigueState": "decayed",
            "acuteTrimpCol": "decayedTrimpBefore",
        },
    ]


def _empty_stage_worker_result(order: int, cohort_name: str, objective: str) -> dict[str, object]:
    return {
        "order": order,
        "cohort": cohort_name,
        "objective": objective,
        "rows": [],
        "params": [],
        "predictions": [],
        "loo_frames": [],
        "stage3_fatigue_rows": [],
        "grid_search_frames": [],
        "stage3_best": None,
        "stage3_segments": pd.DataFrame(),
    }


def _hrr_trimp_grid_frame(
    grid: pd.DataFrame,
    *,
    cohort_name: str,
    stage: str,
    objective: str,
    physiology: Mapping[str, Any],
    load_factor_col: str | None,
    use_hrr_effort: bool,
    fatigue_state: str,
    acute_trimp_col: str,
    secondary_acute_trimp_col: str = "",
    secondary_fatigue_model: str = "",
) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    out = grid.copy()
    out.insert(0, "cohort", cohort_name)
    out.insert(1, "stage", stage)
    out.insert(2, "fitObjective", objective)
    out.insert(3, "validation", "in_sample")
    out["loadFactorCol"] = load_factor_col or ""
    out["useHrrEffort"] = bool(use_hrr_effort)
    out["fatigueState"] = fatigue_state
    out["acuteTrimpCol"] = acute_trimp_col
    out["secondaryAcuteTrimpCol"] = secondary_acute_trimp_col
    out["secondaryFatigueModel"] = secondary_fatigue_model
    out["vmaFlatKmh"] = float(physiology.get("vma_flat_kmh", np.nan))
    out["hrrReference"] = float(physiology.get("hrr_reference", np.nan))
    out["hrrMinFactor"] = float(physiology.get("hrr_min_factor", np.nan))
    out["hrrMaxFactor"] = float(physiology.get("hrr_max_factor", np.nan))
    out["minFatigueFactor"] = float(physiology.get("min_fatigue_factor", np.nan))
    out["decayLambda"] = float(physiology.get("decay_lambda", np.nan))
    if "raceMaeSec" in out.columns:
        out["raceMaeMin"] = pd.to_numeric(out["raceMaeSec"], errors="coerce") / 60.0
    if "segmentMaeSec" in out.columns:
        out["segmentMaeMin"] = pd.to_numeric(out["segmentMaeSec"], errors="coerce") / 60.0
    return out


def _run_paper_stage_models_for_objective(task: tuple[object, ...]) -> dict[str, object]:
    (
        order,
        cohort_name,
        objective,
        cohort_df,
        segments_by_activity,
        segment_features,
        config,
        v_vt2_kmh,
    ) = task
    order = int(order)
    cohort_name = str(cohort_name)
    objective = str(objective)
    fitting = config["fitting"]
    physiology = config["physiology"]
    run_loo = "loo" in set(fitting.get("validation_modes", ["in_sample", "loo"]))
    loo_include = [str(name) for name in config.get("cohorts", {}).get("loo_include", []) or []]
    if loo_include and cohort_name not in set(loo_include):
        logger.info(
            "Skipping LOO for cohort=%s (not in cohorts.loo_include=%s)",
            cohort_name,
            loo_include,
        )
        run_loo = False
    fit_mask_col = _fit_mask_col(config)
    fit_actual_time_col = _fit_actual_time_col(config)
    rows: list[dict[str, object]] = []
    params: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    loo_frames: list[pd.DataFrame] = []
    stage3_fatigue_rows: list[dict[str, object]] = []
    grid_search_frames: list[pd.DataFrame] = []

    ids, segments, actual, ctl_factors = _cohort_inputs(cohort_df, segments_by_activity)
    if not ids:
        return _empty_stage_worker_result(order, cohort_name, objective)
    loo_ids = list(ids)
    loo_cap = int(config.get("cohorts", {}).get("loo_activity_cap", 0) or 0)
    if run_loo and loo_cap > 0 and len(loo_ids) > loo_cap:
        seed = int(config.get("cohorts", {}).get("loo_activity_cap_seed", 20260721))
        # Stable per-cohort seed so caps are reproducible across runs/processes.
        cohort_seed = seed + _stable_cohort_seed_offset(cohort_name)
        rng = np.random.default_rng(cohort_seed)
        loo_ids = sorted(rng.choice(np.array(loo_ids, dtype=object), size=loo_cap, replace=False).tolist())
        logger.warning(
            "LOO activity cap applied for cohort=%s: using %d/%d activities (seed=%d)",
            cohort_name,
            len(loo_ids),
            len(ids),
            cohort_seed,
        )
    observed = actual.to_dict()
    cohort_segments = segment_features[segment_features["activityId"].isin(ids)].copy()
    if cohort_segments.empty:
        return _empty_stage_worker_result(order, cohort_name, objective)
    loo_segments = {
        activity_id: segments[activity_id] for activity_id in loo_ids if activity_id in segments
    }
    loo_observed = {activity_id: observed[activity_id] for activity_id in loo_ids if activity_id in observed}
    loo_ctl = {activity_id: ctl_factors[activity_id] for activity_id in loo_ids if activity_id in ctl_factors}
    loo_cohort_segments = cohort_segments[cohort_segments["activityId"].astype(str).isin(loo_ids)].copy()

    best_stage0, _grid_stage0 = tpm.grid_search_model(
        segments,
        observed,
        v_vt2_kmh=float(v_vt2_kmh),
        alpha_grid=fitting["stage0_alpha_grid"],
        mu_grid=fitting["stage0_mu_grid"],
        fatigue_model="linear",
        ctl_factors=ctl_factors,
    )
    pred_stage0 = tpm.predict_many(
        segments,
        v_vt2_kmh=float(v_vt2_kmh),
        alpha=best_stage0["alpha"],
        mu=best_stage0["mu"],
        fatigue_model="linear",
        ctl_factors=ctl_factors,
    ).reindex(ids)
    loo_stage0 = (
        tpm.leave_one_out_grid_search(
            loo_segments,
            loo_observed,
            v_vt2_kmh=float(v_vt2_kmh),
            alpha_grid=fitting["stage0_alpha_grid"],
            mu_grid=fitting["stage0_mu_grid"],
            fatigue_model="linear",
            ctl_factors=loo_ctl,
        )
        if run_loo
        else pd.DataFrame()
    )

    stage0_name = "Stage 0 reproduction Stage 3"
    rows.append(_add_fit_objective(_metrics_row(cohort_name, stage0_name, actual, pred_stage0), objective))
    params.append({"cohort": cohort_name, "stage": stage0_name, "fitObjective": objective, **best_stage0})
    predictions.append(_prediction_frame(cohort_name, stage0_name, objective, cohort_df, actual, pred_stage0))
    if not loo_stage0.empty:
        loo = loo_stage0.copy()
        loo["cohort"] = cohort_name
        loo["stage"] = f"{stage0_name} LOO"
        loo["fitObjective"] = objective
        loo_frames.append(loo)
        rows.append(
            _add_fit_objective(
                _metrics_row(cohort_name, f"{stage0_name} LOO", loo["actualTimeSec"], loo["predictedTimeSec"]),
                objective,
            )
        )

    for spec in _paper_stage_specs():
        best, grid, prediction = tpm.hrr_trimp_grid_search_model(
            cohort_segments,
            v_anchor_kmh=float(physiology["vma_flat_kmh"]),
            alpha_grid=fitting["hrr_trimp_alpha_grid"],
            fatigue_coef_grid=fitting["hrr_trimp_kappa_grid"],
            fatigue_models=fitting["fatigue_models"],
            hrr_reference=float(physiology["hrr_reference"]),
            hrr_min_factor=float(physiology["hrr_min_factor"]),
            hrr_max_factor=float(physiology["hrr_max_factor"]),
            min_fatigue_factor=float(physiology["min_fatigue_factor"]),
            gap_steep_threshold=float(physiology.get("gap_steep_threshold", 0.15)),
            gap_soft_start=float(physiology.get("gap_soft_start", 0.04)),
            gap_climb_scale=float(physiology.get("gap_climb_scale", 1.0)),
            gap_descent_scale=float(physiology.get("gap_descent_scale", 1.0)),
            load_factor_col=spec["loadFactorCol"],
            use_hrr_effort=bool(spec["useHrrEffort"]),
            acute_trimp_col=spec["acuteTrimpCol"],
            objective=_model_objective(objective),
            observed_activity_times_sec=observed,
            fit_mask_col=fit_mask_col,
            actual_time_col=fit_actual_time_col,
        )
        grid_search_frames.append(
            _hrr_trimp_grid_frame(
                grid,
                cohort_name=cohort_name,
                stage=spec["stage"],
                objective=objective,
                physiology=physiology,
                load_factor_col=spec["loadFactorCol"],
                use_hrr_effort=bool(spec["useHrrEffort"]),
                fatigue_state=str(spec["fatigueState"]),
                acute_trimp_col=str(spec["acuteTrimpCol"]),
            )
        )
        predicted_activity = _race_prediction_from_segments(prediction, ids)
        rows.append(_add_fit_objective(_metrics_row(cohort_name, spec["stage"], actual, predicted_activity), objective))
        params.append({"cohort": cohort_name, "fitObjective": objective, **spec, **best})
        predictions.append(
            _prediction_frame(cohort_name, spec["stage"], objective, cohort_df, actual, predicted_activity)
        )

        loo = (
            tpm.leave_one_out_hrr_trimp_grid_search(
                loo_cohort_segments,
                v_anchor_kmh=float(physiology["vma_flat_kmh"]),
                alpha_grid=fitting["hrr_trimp_alpha_grid"],
                fatigue_coef_grid=fitting["hrr_trimp_kappa_grid"],
                fatigue_models=fitting["fatigue_models"],
                hrr_reference=float(physiology["hrr_reference"]),
                hrr_min_factor=float(physiology["hrr_min_factor"]),
                hrr_max_factor=float(physiology["hrr_max_factor"]),
                min_fatigue_factor=float(physiology["min_fatigue_factor"]),
            gap_steep_threshold=float(physiology.get("gap_steep_threshold", 0.15)),
            gap_soft_start=float(physiology.get("gap_soft_start", 0.04)),
            gap_climb_scale=float(physiology.get("gap_climb_scale", 1.0)),
            gap_descent_scale=float(physiology.get("gap_descent_scale", 1.0)),
                load_factor_col=spec["loadFactorCol"],
                use_hrr_effort=bool(spec["useHrrEffort"]),
                acute_trimp_col=spec["acuteTrimpCol"],
                observed_activity_times_sec=loo_observed,
                objective=_model_objective(objective),
                fit_mask_col=fit_mask_col,
            actual_time_col=fit_actual_time_col,
            )
            if run_loo
            else pd.DataFrame()
        )
        if not loo.empty:
            loo["cohort"] = cohort_name
            loo["stage"] = f"{spec['stage']} LOO"
            loo["fitObjective"] = objective
            loo["fatigueState"] = spec["fatigueState"]
            loo["acuteTrimpCol"] = spec["acuteTrimpCol"]
            loo_frames.append(loo)
            rows.append(
                _add_fit_objective(
                    _metrics_row(cohort_name, f"{spec['stage']} LOO", loo["actualTimeSec"], loo["predictedTimeSec"]),
                    objective,
                )
            )

    stage3_variant_results: dict[tuple[str, str], dict[str, object]] = {}
    stage3_variant_grids: dict[tuple[str, str], pd.DataFrame] = {}
    local_stage3_rows: list[dict[str, object]] = []
    for state_spec in fitting["stage3_fatigue_states"]:
        state_models = state_spec.get("fatigue_models") or fitting["fatigue_models"]
        secondary_col = str(state_spec.get("secondary_acute_trimp_col", ""))
        secondary_model = str(state_spec.get("secondary_fatigue_model", ""))
        secondary_grid = fitting["hrr_trimp_secondary_kappa_grid"] if secondary_col else [0.0]
        for fatigue_model in state_models:
            variant_stage = f"Stage 3 HRR speed ratio {state_spec['label']} {fatigue_model}"
            best, grid, prediction = tpm.hrr_trimp_grid_search_model(
                cohort_segments,
                v_anchor_kmh=float(physiology["vma_flat_kmh"]),
                alpha_grid=fitting["hrr_trimp_alpha_grid"],
                fatigue_coef_grid=fitting["hrr_trimp_kappa_grid"],
                secondary_fatigue_coef_grid=secondary_grid,
                fatigue_models=(fatigue_model,),
                hrr_reference=float(physiology["hrr_reference"]),
                hrr_min_factor=float(physiology["hrr_min_factor"]),
                hrr_max_factor=float(physiology["hrr_max_factor"]),
                min_fatigue_factor=float(physiology["min_fatigue_factor"]),
            gap_steep_threshold=float(physiology.get("gap_steep_threshold", 0.15)),
            gap_soft_start=float(physiology.get("gap_soft_start", 0.04)),
            gap_climb_scale=float(physiology.get("gap_climb_scale", 1.0)),
            gap_descent_scale=float(physiology.get("gap_descent_scale", 1.0)),
                load_factor_col="rediReadinessFactor",
                use_hrr_effort=True,
                acute_trimp_col=state_spec["acute_trimp_col"],
                secondary_acute_trimp_col=secondary_col or None,
                secondary_fatigue_model=secondary_model or None,
                objective=_model_objective(objective),
                observed_activity_times_sec=observed,
                fit_mask_col=fit_mask_col,
            actual_time_col=fit_actual_time_col,
            )
            grid_frame = _hrr_trimp_grid_frame(
                grid,
                cohort_name=cohort_name,
                stage=variant_stage,
                objective=objective,
                physiology=physiology,
                load_factor_col="rediReadinessFactor",
                use_hrr_effort=True,
                fatigue_state=str(state_spec["fatigue_state"]),
                acute_trimp_col=str(state_spec["acute_trimp_col"]),
                secondary_acute_trimp_col=secondary_col,
                secondary_fatigue_model=secondary_model,
            )
            grid_search_frames.append(grid_frame)
            stage3_variant_grids[(str(state_spec["fatigue_state"]), str(fatigue_model))] = grid_frame
            predicted_activity = _race_prediction_from_segments(prediction, ids)
            in_sample_row = _stage3_metadata(
                _metrics_row(cohort_name, variant_stage, actual, predicted_activity),
                validation="in_sample",
                objective=objective,
                fatigue_state=state_spec["fatigue_state"],
                acute_trimp_col=state_spec["acute_trimp_col"],
                fatigue_model=fatigue_model,
                alpha=float(best.get("alpha", np.nan)),
                fatigue_coef=float(best.get("fatigueCoef", np.nan)),
                secondary_acute_trimp_col=secondary_col,
                secondary_fatigue_model=secondary_model,
                secondary_fatigue_coef=float(best.get("secondaryFatigueCoef", 0.0)),
            )
            stage3_fatigue_rows.append(in_sample_row)
            local_stage3_rows.append(in_sample_row)

            loo = (
                tpm.leave_one_out_hrr_trimp_grid_search(
                    loo_cohort_segments,
                    v_anchor_kmh=float(physiology["vma_flat_kmh"]),
                    alpha_grid=fitting["hrr_trimp_alpha_grid"],
                    fatigue_coef_grid=fitting["hrr_trimp_kappa_grid"],
                    secondary_fatigue_coef_grid=secondary_grid,
                    fatigue_models=(fatigue_model,),
                    hrr_reference=float(physiology["hrr_reference"]),
                    hrr_min_factor=float(physiology["hrr_min_factor"]),
                    hrr_max_factor=float(physiology["hrr_max_factor"]),
                    min_fatigue_factor=float(physiology["min_fatigue_factor"]),
            gap_steep_threshold=float(physiology.get("gap_steep_threshold", 0.15)),
            gap_soft_start=float(physiology.get("gap_soft_start", 0.04)),
            gap_climb_scale=float(physiology.get("gap_climb_scale", 1.0)),
            gap_descent_scale=float(physiology.get("gap_descent_scale", 1.0)),
                    load_factor_col="rediReadinessFactor",
                    use_hrr_effort=True,
                    acute_trimp_col=state_spec["acute_trimp_col"],
                    secondary_acute_trimp_col=secondary_col or None,
                    secondary_fatigue_model=secondary_model or None,
                    observed_activity_times_sec=loo_observed,
                    objective=_model_objective(objective),
                    fit_mask_col=fit_mask_col,
            actual_time_col=fit_actual_time_col,
                )
                if run_loo
                else pd.DataFrame()
            )
            if not loo.empty:
                loo["cohort"] = cohort_name
                loo["stage"] = f"{variant_stage} LOO"
                loo["fitObjective"] = objective
                loo["fatigueState"] = state_spec["fatigue_state"]
                loo["acuteTrimpCol"] = state_spec["acute_trimp_col"]
                loo["secondaryAcuteTrimpCol"] = secondary_col
                loo["secondaryFatigueModel"] = secondary_model
                loo_frames.append(loo)
                loo_row = _stage3_metadata(
                    _metrics_row(cohort_name, f"{variant_stage} LOO", loo["actualTimeSec"], loo["predictedTimeSec"]),
                    validation="loo",
                    objective=objective,
                    fatigue_state=state_spec["fatigue_state"],
                    acute_trimp_col=state_spec["acute_trimp_col"],
                    fatigue_model=fatigue_model,
                    alpha=float(loo["alpha"].median()) if "alpha" in loo else float(best.get("alpha", np.nan)),
                    fatigue_coef=float(loo["fatigueCoef"].median())
                    if "fatigueCoef" in loo
                    else float(best.get("fatigueCoef", np.nan)),
                    secondary_acute_trimp_col=secondary_col,
                    secondary_fatigue_model=secondary_model,
                    secondary_fatigue_coef=float(loo["secondaryFatigueCoef"].median())
                    if "secondaryFatigueCoef" in loo
                    else float(best.get("secondaryFatigueCoef", 0.0)),
                )
                stage3_fatigue_rows.append(loo_row)
                local_stage3_rows.append(loo_row)

            stage3_variant_results[(state_spec["fatigue_state"], fatigue_model)] = {
                "best": {
                    **best,
                    "fatigueState": state_spec["fatigue_state"],
                    "acuteTrimpCol": state_spec["acute_trimp_col"],
                    "secondaryAcuteTrimpCol": secondary_col,
                    "secondaryFatigueModel": secondary_model,
                },
                "prediction": prediction,
                "predicted_activity": predicted_activity,
            }

    local_metrics = pd.DataFrame(local_stage3_rows)
    selection_pool = local_metrics[local_metrics["validation"].eq("loo")]
    if selection_pool.empty:
        selection_pool = local_metrics[local_metrics["validation"].eq("in_sample")]
    chosen_row = selection_pool.sort_values(["maeMin", "mapePct", "fatigueState", "fatigueModel"]).iloc[0]
    chosen_key = (str(chosen_row["fatigueState"]), str(chosen_row["fatigueModel"]))
    chosen_result = stage3_variant_results[chosen_key]
    chosen_best = chosen_result["best"]

    stage3_name = "Stage 3 HRR speed ratio"
    chosen_grid = stage3_variant_grids.get(chosen_key, pd.DataFrame())
    if not chosen_grid.empty:
        selected_grid = chosen_grid.copy()
        selected_grid["stage"] = stage3_name
        selected_grid["selectedFatigueState"] = chosen_key[0]
        selected_grid["selectedFatigueModel"] = chosen_key[1]
        grid_search_frames.append(selected_grid)
    predicted_activity = chosen_result["predicted_activity"]
    rows.append(_add_fit_objective(_metrics_row(cohort_name, stage3_name, actual, predicted_activity), objective))
    params.append(
        {
            "cohort": cohort_name,
            "stage": stage3_name,
            "fitObjective": objective,
            "loadFactorCol": "rediReadinessFactor",
            "useHrrEffort": True,
            "fatigueState": chosen_best["fatigueState"],
            "acuteTrimpCol": chosen_best["acuteTrimpCol"],
            **chosen_best,
        }
    )
    predictions.append(_prediction_frame(cohort_name, stage3_name, objective, cohort_df, actual, predicted_activity))

    chosen_loo = local_metrics[
        local_metrics["validation"].eq("loo")
        & local_metrics["fatigueState"].eq(chosen_key[0])
        & local_metrics["fatigueModel"].eq(chosen_key[1])
    ]
    if not chosen_loo.empty:
        loo_match = [
            frame
            for frame in loo_frames
            if not frame.empty
            and frame["cohort"].eq(cohort_name).all()
            and frame["fitObjective"].eq(objective).all()
            and frame.get("fatigueState", pd.Series(dtype=str)).eq(chosen_key[0]).all()
            and frame.get("stage", pd.Series(dtype=str)).astype(str).str.contains(chosen_key[1], regex=False).all()
        ]
        if loo_match:
            loo = loo_match[-1].copy()
            loo["stage"] = f"{stage3_name} LOO"
            loo_frames.append(loo)
            rows.append(
                _add_fit_objective(
                    _metrics_row(cohort_name, f"{stage3_name} LOO", loo["actualTimeSec"], loo["predictedTimeSec"]),
                    objective,
                )
            )

    return {
        "order": order,
        "cohort": cohort_name,
        "objective": objective,
        "rows": rows,
        "params": params,
        "predictions": predictions,
        "loo_frames": loo_frames,
        "stage3_fatigue_rows": stage3_fatigue_rows,
        "grid_search_frames": grid_search_frames,
        "stage3_best": chosen_best,
        "stage3_segments": cohort_segments,
    }


def run_paper_stage_models(
    cohorts: Mapping[str, pd.DataFrame],
    segments_by_activity: Mapping[str, pd.DataFrame],
    segment_features: pd.DataFrame,
    config: Mapping[str, Any],
    v_vt2_kmh: float,
) -> tuple[dict[str, pd.DataFrame], dict[tuple[str, str], dict[str, object]], dict[tuple[str, str], pd.DataFrame]]:
    """Run paper-style Stage 0-3 models for all configured objectives."""
    tasks: list[tuple[object, ...]] = []
    for cohort_name, cohort_df in cohorts.items():
        ids = cohort_df["activityId"].astype(str).tolist()
        cohort_segments_by_activity = {
            activity_id: segments_by_activity[activity_id]
            for activity_id in ids
            if activity_id in segments_by_activity
        }
        cohort_segment_features = segment_features[segment_features["activityId"].astype(str).isin(ids)].copy()
        for objective in config["fitting"]["enabled_objectives"]:
            tasks.append(
                (
                    len(tasks),
                    cohort_name,
                    objective,
                    cohort_df,
                    cohort_segments_by_activity,
                    cohort_segment_features,
                    config,
                    v_vt2_kmh,
                )
            )

    jobs = int(config.get("execution", {}).get("jobs", 1))
    if jobs > 1 and len(tasks) > 1:
        max_workers = min(jobs, len(tasks))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_run_paper_stage_models_for_objective, task): task[0] for task in tasks}
            results = [future.result() for future in as_completed(future_map)]
    else:
        results = [_run_paper_stage_models_for_objective(task) for task in tasks]
    results = sorted(results, key=lambda result: int(result["order"]))

    rows: list[dict[str, object]] = []
    params: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    loo_frames: list[pd.DataFrame] = []
    stage3_fatigue_rows: list[dict[str, object]] = []
    grid_search_frames: list[pd.DataFrame] = []
    stage3_best: dict[tuple[str, str], dict[str, object]] = {}
    stage3_segments: dict[tuple[str, str], pd.DataFrame] = {}
    for result in results:
        rows.extend(result["rows"])
        params.extend(result["params"])
        predictions.extend(result["predictions"])
        loo_frames.extend(result["loo_frames"])
        stage3_fatigue_rows.extend(result["stage3_fatigue_rows"])
        grid_search_frames.extend(result["grid_search_frames"])
        best = result["stage3_best"]
        if best is not None:
            key = (str(result["cohort"]), str(result["objective"]))
            stage3_best[key] = best
            stage3_segments[key] = result["stage3_segments"]

    tables = {
        "table_stage_metrics": pd.DataFrame(rows),
        "table_fitted_parameters": pd.DataFrame(params),
        "activity_predictions": pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        "activity_loo_predictions": pd.concat(loo_frames, ignore_index=True) if loo_frames else pd.DataFrame(),
        "table_stage3_fatigue_state_comparison": pd.DataFrame(stage3_fatigue_rows),
        "table_hrr_trimp_grid_search": pd.concat(grid_search_frames, ignore_index=True)
        if grid_search_frames
        else pd.DataFrame(),
    }
    return tables, stage3_best, stage3_segments


def _stable_cohort_seed_offset(cohort_name: str) -> int:
    """Stable per-cohort offset (avoid Python's randomized ``hash()``)."""
    digest = hashlib.sha256(str(cohort_name).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000


def _select_loo_activity_ids(activity_ids: Sequence[str], cohort_name: str, config: Mapping[str, Any]) -> list[str]:
    """Apply the same LOO activity cap used by Stage 0–3 paper models."""
    loo_ids = [str(activity_id) for activity_id in activity_ids]
    loo_cap = int(config.get("cohorts", {}).get("loo_activity_cap", 0) or 0)
    if loo_cap > 0 and len(loo_ids) > loo_cap:
        seed = int(config.get("cohorts", {}).get("loo_activity_cap_seed", 20260721))
        cohort_seed = seed + _stable_cohort_seed_offset(cohort_name)
        rng = np.random.default_rng(cohort_seed)
        loo_ids = sorted(rng.choice(np.array(loo_ids, dtype=object), size=loo_cap, replace=False).tolist())
        logger.warning(
            "LOO activity cap applied for cohort=%s: using %d/%d activities (seed=%d)",
            cohort_name,
            len(loo_ids),
            len(activity_ids),
            cohort_seed,
        )
    return loo_ids


def _ablation_variant_definitions(
    cohort_segments: pd.DataFrame,
) -> list[dict[str, object]]:
    """Build leave-one-component Stage-3 ablation variants."""
    no_gap = cohort_segments.copy()
    no_gap["avgGrade"] = 0.0
    if "gapFactorIntegrated" in no_gap.columns:
        no_gap["gapFactorIntegrated"] = 1.0
    if "gapFactor" in no_gap.columns:
        no_gap["gapFactor"] = 1.0
    return [
        {"label": "full", "segments": cohort_segments},
        {"label": "no GAP", "segments": no_gap},
        {"label": "no altitude", "segments": cohort_segments.assign(meanAltitudeM=0.0)},
        {"label": "no REDI readiness", "segments": cohort_segments.assign(rediReadinessFactor=1.0)},
        {"label": "no HRR speed ratio", "segments": cohort_segments, "use_hrr_effort": False},
        {"label": "no acute fatigue", "segments": cohort_segments, "zero_fatigue": True},
        {
            "label": "no trail GAP scales",
            "segments": cohort_segments,
            "gap_climb_scale": 1.0,
            "gap_descent_scale": 1.0,
        },
    ]


def _run_stage3_ablation_frozen(
    stage3_best: Mapping[tuple[str, str], Mapping[str, object]],
    stage3_segments: Mapping[tuple[str, str], pd.DataFrame],
    cohorts: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Legacy ablation: freeze Stage-3 (α, κ) and disable components at prediction time."""
    physiology = config["physiology"]
    rows: list[dict[str, object]] = []
    for (cohort_name, objective), best in stage3_best.items():
        cohort_segments = stage3_segments[(cohort_name, objective)].copy()
        cohort_df = cohorts[cohort_name]
        ids = cohort_df["activityId"].astype(str).tolist()
        observed = cohort_df.set_index("activityId").loc[ids, "actualTimeSec"].astype(float)
        acute_col = str(best.get("acuteTrimpCol", "decayedTrimpBefore"))
        secondary_col = str(best.get("secondaryAcuteTrimpCol", ""))
        secondary_model = str(best.get("secondaryFatigueModel", ""))

        def predict_variant(
            label: str,
            variant_segments: pd.DataFrame,
            *,
            fatigue_coef: float | None = None,
            load_factor_col: str | None = "rediReadinessFactor",
            use_hrr_effort: bool = True,
            gap_climb_scale: float | None = None,
            gap_descent_scale: float | None = None,
        ) -> dict[str, object]:
            predicted_segments = tpm.predict_hrr_trimp_segment_times(
                variant_segments,
                v_anchor_kmh=float(physiology["vma_flat_kmh"]),
                alpha=float(best["alpha"]),
                fatigue_coef=float(best["fatigueCoef"] if fatigue_coef is None else fatigue_coef),
                fatigue_model=str(best["fatigueModel"]),
                hrr_reference=float(physiology["hrr_reference"]),
                hrr_min_factor=float(physiology["hrr_min_factor"]),
                hrr_max_factor=float(physiology["hrr_max_factor"]),
                min_fatigue_factor=float(physiology["min_fatigue_factor"]),
                gap_steep_threshold=float(physiology.get("gap_steep_threshold", 0.15)),
                gap_soft_start=float(physiology.get("gap_soft_start", 0.04)),
                gap_climb_scale=float(
                    physiology.get("gap_climb_scale", 1.0) if gap_climb_scale is None else gap_climb_scale
                ),
                gap_descent_scale=float(
                    physiology.get("gap_descent_scale", 1.0) if gap_descent_scale is None else gap_descent_scale
                ),
                load_factor_col=load_factor_col,
                use_hrr_effort=use_hrr_effort,
                acute_trimp_col=acute_col,
                secondary_fatigue_coef=float(best.get("secondaryFatigueCoef", 0.0)),
                secondary_acute_trimp_col=secondary_col or None,
                secondary_fatigue_model=secondary_model or None,
            )
            predicted = (
                pd.DataFrame(
                    {
                        "activityId": variant_segments["activityId"].astype(str),
                        "predictedTimeSec": predicted_segments,
                    }
                )
                .groupby("activityId")["predictedTimeSec"]
                .sum()
            )
            row = _metrics_row(cohort_name, label, observed, predicted.reindex(observed.index))
            row["fitObjective"] = objective
            row["fatigueState"] = best.get("fatigueState", "")
            row["acuteTrimpCol"] = acute_col
            row["fatigueModel"] = best.get("fatigueModel", "")
            row["secondaryAcuteTrimpCol"] = secondary_col
            row["secondaryFatigueModel"] = secondary_model
            row["secondaryFatigueCoef"] = (
                0.0 if fatigue_coef == 0.0 else best.get("secondaryFatigueCoef", 0.0)
            )
            row["alpha"] = float(best["alpha"])
            row["fatigueCoef"] = float(best["fatigueCoef"] if fatigue_coef is None else fatigue_coef)
            row["ablationProtocol"] = "frozen"
            row["validation"] = "in_sample"
            return row

        full = predict_variant("full", cohort_segments)
        full_mae = float(full["maeMin"])
        rows.append({**full, "deltaMaeMinVsFull": 0.0})
        for variant in _ablation_variant_definitions(cohort_segments):
            label = str(variant["label"])
            if label == "full":
                continue
            row = predict_variant(
                label,
                variant["segments"],  # type: ignore[arg-type]
                use_hrr_effort=bool(variant.get("use_hrr_effort", True)),
                fatigue_coef=0.0 if bool(variant.get("zero_fatigue", False)) else None,
                gap_climb_scale=variant.get("gap_climb_scale"),  # type: ignore[arg-type]
                gap_descent_scale=variant.get("gap_descent_scale"),  # type: ignore[arg-type]
            )
            row["deltaMaeMinVsFull"] = float(row["maeMin"]) - full_mae
            rows.append(row)
    return pd.DataFrame(rows)


def _run_stage3_ablation_reoptimize(
    stage3_best: Mapping[tuple[str, str], Mapping[str, object]],
    stage3_segments: Mapping[tuple[str, str], pd.DataFrame],
    cohorts: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Preferred ablation: re-optimize (α, κ) for each leave-one-component variant."""
    physiology = config["physiology"]
    fitting = config["fitting"]
    fit_mask_col = _fit_mask_col(config)
    fit_actual_time_col = _fit_actual_time_col(config)
    loo_include = set(config.get("cohorts", {}).get("loo_include") or [])
    loo_enabled = "loo" in set(fitting.get("validation_modes", ["in_sample", "loo"]))
    rows: list[dict[str, object]] = []

    for (cohort_name, objective), best in stage3_best.items():
        cohort_segments = stage3_segments[(cohort_name, objective)].copy()
        cohort_df = cohorts[cohort_name]
        ids = cohort_df["activityId"].astype(str).tolist()
        observed = cohort_df.set_index("activityId")["actualTimeSec"].astype(float)
        observed = observed.reindex(ids).astype(float)
        observed_map = observed.to_dict()
        acute_col = str(best.get("acuteTrimpCol", "decayedTrimpBefore"))
        secondary_col = str(best.get("secondaryAcuteTrimpCol", ""))
        secondary_model = str(best.get("secondaryFatigueModel", ""))
        fatigue_model = str(best.get("fatigueModel", "linear"))
        run_loo = loo_enabled and (not loo_include or cohort_name in loo_include)
        loo_ids = _select_loo_activity_ids(ids, cohort_name, config) if run_loo else ids
        loo_observed = {
            activity_id: observed_map[activity_id] for activity_id in loo_ids if activity_id in observed_map
        }

        full_mae: float | None = None
        for variant in _ablation_variant_definitions(cohort_segments):
            label = str(variant["label"])
            variant_segments = variant["segments"]  # type: ignore[assignment]
            assert isinstance(variant_segments, pd.DataFrame)
            use_hrr_effort = bool(variant.get("use_hrr_effort", True))
            zero_fatigue = bool(variant.get("zero_fatigue", False))
            gap_climb_scale = float(
                physiology.get("gap_climb_scale", 1.0)
                if variant.get("gap_climb_scale") is None
                else variant["gap_climb_scale"]
            )
            gap_descent_scale = float(
                physiology.get("gap_descent_scale", 1.0)
                if variant.get("gap_descent_scale") is None
                else variant["gap_descent_scale"]
            )
            alpha_grid = fitting["hrr_trimp_alpha_grid"]
            fatigue_coef_grid = [0.0] if zero_fatigue else fitting["hrr_trimp_kappa_grid"]
            secondary_grid = (
                [0.0]
                if zero_fatigue or not secondary_col
                else fitting["hrr_trimp_secondary_kappa_grid"]
            )
            prediction_kwargs: dict[str, object] = {
                "hrr_reference": float(physiology["hrr_reference"]),
                "hrr_min_factor": float(physiology["hrr_min_factor"]),
                "hrr_max_factor": float(physiology["hrr_max_factor"]),
                "min_fatigue_factor": float(physiology["min_fatigue_factor"]),
                "gap_steep_threshold": float(physiology.get("gap_steep_threshold", 0.15)),
                "gap_soft_start": float(physiology.get("gap_soft_start", 0.04)),
                "gap_climb_scale": gap_climb_scale,
                "gap_descent_scale": gap_descent_scale,
                "load_factor_col": "rediReadinessFactor",
                "use_hrr_effort": use_hrr_effort,
                "acute_trimp_col": acute_col,
                "secondary_acute_trimp_col": secondary_col or None,
                "secondary_fatigue_model": secondary_model or None,
            }

            if run_loo:
                eval_segments = variant_segments[
                    variant_segments["activityId"].astype(str).isin(loo_ids)
                ].copy()
                loo = tpm.leave_one_out_hrr_trimp_grid_search(
                    eval_segments,
                    v_anchor_kmh=float(physiology["vma_flat_kmh"]),
                    alpha_grid=alpha_grid,
                    fatigue_coef_grid=fatigue_coef_grid,
                    secondary_fatigue_coef_grid=secondary_grid,
                    fatigue_models=(fatigue_model,),
                    observed_activity_times_sec=loo_observed,
                    objective=_model_objective(objective),
                    fit_mask_col=fit_mask_col,
                    actual_time_col=fit_actual_time_col,
                    **prediction_kwargs,
                )
                if loo.empty:
                    logger.warning(
                        "Reoptimized ablation LOO empty for cohort=%s variant=%s; falling back to in-sample",
                        cohort_name,
                        label,
                    )
                    run_loo_variant = False
                else:
                    row = _metrics_row(
                        cohort_name,
                        label,
                        loo["actualTimeSec"],
                        loo["predictedTimeSec"],
                    )
                    row["alpha"] = float(loo["alpha"].median())
                    row["fatigueCoef"] = float(loo["fatigueCoef"].median())
                    row["secondaryFatigueCoef"] = (
                        float(loo["secondaryFatigueCoef"].median())
                        if "secondaryFatigueCoef" in loo
                        else 0.0
                    )
                    row["ablationProtocol"] = "reoptimize_loo"
                    row["validation"] = "loo"
                    run_loo_variant = True
            else:
                run_loo_variant = False

            if not run_loo_variant:
                fit_best, _grid, prediction = tpm.hrr_trimp_grid_search_model(
                    variant_segments,
                    v_anchor_kmh=float(physiology["vma_flat_kmh"]),
                    alpha_grid=alpha_grid,
                    fatigue_coef_grid=fatigue_coef_grid,
                    secondary_fatigue_coef_grid=secondary_grid,
                    fatigue_models=(fatigue_model,),
                    observed_activity_times_sec=observed_map,
                    objective=_model_objective(objective),
                    fit_mask_col=fit_mask_col,
                    actual_time_col=fit_actual_time_col,
                    **prediction_kwargs,
                )
                predicted_activity = _race_prediction_from_segments(prediction, ids)
                row = _metrics_row(cohort_name, label, observed, predicted_activity)
                row["alpha"] = float(fit_best.get("alpha", np.nan))
                row["fatigueCoef"] = float(fit_best.get("fatigueCoef", np.nan))
                row["secondaryFatigueCoef"] = float(fit_best.get("secondaryFatigueCoef", 0.0))
                row["ablationProtocol"] = "reoptimize_in_sample"
                row["validation"] = "in_sample"

            row["fitObjective"] = objective
            row["fatigueState"] = best.get("fatigueState", "")
            row["acuteTrimpCol"] = acute_col
            row["fatigueModel"] = fatigue_model
            row["secondaryAcuteTrimpCol"] = secondary_col
            row["secondaryFatigueModel"] = secondary_model
            if full_mae is None:
                full_mae = float(row["maeMin"])
                row["deltaMaeMinVsFull"] = 0.0
            else:
                row["deltaMaeMinVsFull"] = float(row["maeMin"]) - float(full_mae)
            rows.append(row)
            logger.info(
                "Ablation cohort=%s objective=%s variant=%s protocol=%s maeMin=%.2f delta=%.2f",
                cohort_name,
                objective,
                label,
                row["ablationProtocol"],
                float(row["maeMin"]),
                float(row["deltaMaeMinVsFull"]),
            )
    return pd.DataFrame(rows)


def run_stage3_ablation(
    stage3_best: Mapping[tuple[str, str], Mapping[str, object]],
    stage3_segments: Mapping[tuple[str, str], pd.DataFrame],
    cohorts: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Ablate selected Stage 3 components for each cohort/objective pair.

    Default protocol ``reoptimize`` re-fits (α, κ) after each component removal and
    evaluates with LOO when enabled (gold-standard contribution estimate). Legacy
    ``frozen`` keeps Stage-3 parameters fixed and only disables components at
    prediction time (inference-time dependency).
    """
    fitting = config.get("fitting", {})
    protocol = str(fitting.get("ablation_protocol", "reoptimize")).strip().lower()
    ablation_objectives = [str(value) for value in (fitting.get("ablation_fit_objectives") or [])]
    filtered_best = dict(stage3_best)
    if ablation_objectives:
        allowed = set(ablation_objectives)
        filtered_best = {
            key: value for key, value in stage3_best.items() if str(key[1]) in allowed
        }
        logger.info(
            "Restricting Stage-3 ablation to fit objectives=%s (%d/%d cohort-objective pairs)",
            sorted(allowed),
            len(filtered_best),
            len(stage3_best),
        )
    if not filtered_best:
        logger.warning("Stage-3 ablation skipped: no cohort/objective pairs after filtering")
        return pd.DataFrame()

    if protocol == "frozen":
        logger.warning(
            "Using frozen Stage-3 ablation protocol (fit once, then remove). "
            "Prefer fitting.ablation_protocol=reoptimize for recoverable contribution estimates."
        )
        return _run_stage3_ablation_frozen(filtered_best, stage3_segments, cohorts, config)
    if protocol != "reoptimize":
        raise ValueError(f"unsupported fitting.ablation_protocol: {protocol}")
    return _run_stage3_ablation_reoptimize(filtered_best, stage3_segments, cohorts, config)


def run_segment_grid(
    cohorts: Mapping[str, pd.DataFrame],
    segment_features: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the exploratory segment-level grid search."""
    if not bool(config["segment_grid"].get("enabled", True)):
        return pd.DataFrame(), pd.DataFrame()
    grid_config = config["segment_grid"]
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for cohort_name, cohort_df in cohorts.items():
        ids = cohort_df["activityId"].astype(str).tolist()
        cohort_segments = segment_features[segment_features["activityId"].isin(ids)].copy()
        if cohort_segments.empty:
            continue
        best, _grid, predicted = tpm.segment_grid_search_model(
            cohort_segments,
            v_anchor_kmh=float(config["physiology"]["vma_flat_kmh"]),
            alpha_grid=grid_config["alpha_grid"],
            mu_grid=grid_config["mu_grid"],
            terrain_multiplier_grid=grid_config["terrain_multiplier_grid"],
            technicality_coef_grid=grid_config["technicality_coef_grid"],
            hr_coef_grid=grid_config["hr_coef_grid"],
            acute_trimp_coef_grid=grid_config["acute_trimp_coef_grid"],
        )
        rows.append({"cohort": cohort_name, **best})
        predicted = predicted.copy()
        predicted["cohort"] = cohort_name
        predictions.append(predicted)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def run_robustness_checks(
    cohorts: Mapping[str, pd.DataFrame],
    all_segments_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Run configured Stage 3 robustness checks."""
    if not bool(config["robustness"].get("enabled", True)):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    fitting = config["fitting"]
    physiology = config["physiology"]
    fit_mask_col = _fit_mask_col(config)
    fit_actual_time_col = _fit_actual_time_col(config)
    objective = "activity"
    for decay_lambda in config["robustness"]["decay_lambdas"]:
        decayed_segments = tpm.add_in_activity_trimp_features(all_segments_df, decay_lambda=decay_lambda)
        features = add_segment_model_features(decayed_segments, activity_df)
        for hrr_reference in config["robustness"]["hrr_references"]:
            for cohort_name, cohort_df in cohorts.items():
                ids = cohort_df["activityId"].astype(str).tolist()
                cohort_segments = features[features["activityId"].isin(ids)].copy()
                if cohort_segments.empty:
                    continue
                observed = cohort_df.set_index("activityId").loc[ids, "actualTimeSec"].astype(float).to_dict()
                best, _grid, _prediction = tpm.hrr_trimp_grid_search_model(
                    cohort_segments,
                    v_anchor_kmh=float(physiology["vma_flat_kmh"]),
                    alpha_grid=fitting["hrr_trimp_alpha_grid"],
                    fatigue_coef_grid=fitting["hrr_trimp_kappa_grid"],
                    fatigue_models=fitting["fatigue_models"],
                    hrr_reference=float(hrr_reference),
                    load_factor_col="rediReadinessFactor",
                    use_hrr_effort=True,
                    acute_trimp_col="decayedTrimpBefore",
                    objective=_model_objective(objective),
                    observed_activity_times_sec=observed,
                    fit_mask_col=fit_mask_col,
            actual_time_col=fit_actual_time_col,
                )
                rows.append(
                    {
                        "checkType": "stage3_grid_sensitivity",
                        "validation": "in_sample",
                        "cohort": cohort_name,
                        "fitObjective": objective,
                        "model": "Stage 3 HRR speed ratio",
                        "hrrReference": hrr_reference,
                        "decayLambda": decay_lambda,
                        "segmentKm": physiology["segment_km"],
                        "target": "moving_time",
                        "status": "computed",
                        "alpha": best.get("alpha", np.nan),
                        "fatigueModel": best.get("fatigueModel", ""),
                        "fatigueCoef": best.get("fatigueCoef", np.nan),
                        "r2": best.get("raceR2", np.nan),
                        "maeMin": best.get("raceMaeSec", np.nan) / 60.0,
                        "mapePct": best.get("raceMapePct", np.nan),
                        "biasMin": best.get("raceBiasSec", np.nan) / 60.0,
                    }
                )

    if bool(config["robustness"].get("elapsed_time_sensitivity", True)):
        features = add_segment_model_features(tpm.add_in_activity_trimp_features(all_segments_df), activity_df)
        for cohort_name, cohort_df in cohorts.items():
            ids = cohort_df["activityId"].astype(str).tolist()
            cohort_segments = features[features["activityId"].isin(ids)].copy()
            if cohort_segments.empty:
                continue
            elapsed = pd.to_numeric(cohort_df.set_index("activityId").loc[ids].get("elapsedSec"), errors="coerce")
            moving = pd.to_numeric(cohort_df.set_index("activityId").loc[ids].get("actualTimeSec"), errors="coerce")
            observed_elapsed = elapsed.where(elapsed > 0, moving).astype(float).to_dict()
            best, _grid, _prediction = tpm.hrr_trimp_grid_search_model(
                cohort_segments,
                v_anchor_kmh=float(physiology["vma_flat_kmh"]),
                alpha_grid=fitting["hrr_trimp_alpha_grid"],
                fatigue_coef_grid=fitting["hrr_trimp_kappa_grid"],
                fatigue_models=fitting["fatigue_models"],
                hrr_reference=float(physiology["hrr_reference"]),
                load_factor_col="rediReadinessFactor",
                use_hrr_effort=True,
                acute_trimp_col="decayedTrimpBefore",
                objective=_model_objective(objective),
                observed_activity_times_sec=observed_elapsed,
                fit_mask_col=fit_mask_col,
            actual_time_col=fit_actual_time_col,
            )
            rows.append(
                {
                    "checkType": "target_sensitivity",
                    "validation": "in_sample",
                    "cohort": cohort_name,
                    "fitObjective": objective,
                    "model": "Stage 3 HRR speed ratio",
                    "hrrReference": physiology["hrr_reference"],
                    "decayLambda": physiology["decay_lambda"],
                    "segmentKm": physiology["segment_km"],
                    "target": "elapsed_time",
                    "status": "computed",
                    "alpha": best.get("alpha", np.nan),
                    "fatigueModel": best.get("fatigueModel", ""),
                    "fatigueCoef": best.get("fatigueCoef", np.nan),
                    "r2": best.get("raceR2", np.nan),
                    "maeMin": best.get("raceMaeSec", np.nan) / 60.0,
                    "mapePct": best.get("raceMapePct", np.nan),
                    "biasMin": best.get("raceBiasSec", np.nan) / 60.0,
                }
            )
    return pd.DataFrame(rows)


def _stage3_segment_predictions(
    stage3_best: Mapping[tuple[str, str], Mapping[str, object]],
    stage3_segments: Mapping[tuple[str, str], pd.DataFrame],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    physiology = config["physiology"]
    frames: list[pd.DataFrame] = []
    for (cohort_name, objective), best in stage3_best.items():
        cohort_segments = stage3_segments[(cohort_name, objective)].copy()
        predicted = tpm.predict_hrr_trimp_segment_times(
            cohort_segments,
            v_anchor_kmh=float(physiology["vma_flat_kmh"]),
            alpha=float(best["alpha"]),
            fatigue_coef=float(best["fatigueCoef"]),
            fatigue_model=str(best["fatigueModel"]),
            hrr_reference=float(physiology["hrr_reference"]),
            hrr_min_factor=float(physiology["hrr_min_factor"]),
            hrr_max_factor=float(physiology["hrr_max_factor"]),
            min_fatigue_factor=float(physiology["min_fatigue_factor"]),
            gap_steep_threshold=float(physiology.get("gap_steep_threshold", 0.15)),
            gap_soft_start=float(physiology.get("gap_soft_start", 0.04)),
            gap_climb_scale=float(physiology.get("gap_climb_scale", 1.0)),
            gap_descent_scale=float(physiology.get("gap_descent_scale", 1.0)),
            load_factor_col="rediReadinessFactor",
            use_hrr_effort=True,
            acute_trimp_col=str(best.get("acuteTrimpCol", "decayedTrimpBefore")),
            secondary_fatigue_coef=float(best.get("secondaryFatigueCoef", 0.0)),
            secondary_acute_trimp_col=str(best.get("secondaryAcuteTrimpCol", "")) or None,
            secondary_fatigue_model=str(best.get("secondaryFatigueModel", "")) or None,
        )
        cohort_segments["cohort"] = cohort_name
        cohort_segments["fitObjective"] = objective
        cohort_segments["stage3PredictedTimeSec"] = predicted
        actual_full = pd.to_numeric(cohort_segments["actualTimeSec"], errors="coerce")
        cohort_segments["stage3ResidualSec"] = predicted - actual_full
        if "actualMovingTimeSec" in cohort_segments.columns:
            actual_moving = pd.to_numeric(cohort_segments["actualMovingTimeSec"], errors="coerce")
            cohort_segments["stage3ResidualMovingSec"] = predicted - actual_moving
        else:
            cohort_segments["stage3ResidualMovingSec"] = cohort_segments["stage3ResidualSec"]
        cohort_segments["stage3FatigueState"] = best.get("fatigueState", "")
        cohort_segments["stage3AcuteTrimpCol"] = best.get("acuteTrimpCol", "")
        cohort_segments["stage3FatigueModel"] = best.get("fatigueModel", "")
        cohort_segments["stage3SecondaryAcuteTrimpCol"] = best.get("secondaryAcuteTrimpCol", "")
        cohort_segments["stage3SecondaryFatigueModel"] = best.get("secondaryFatigueModel", "")
        cohort_segments["stage3SecondaryFatigueCoef"] = best.get("secondaryFatigueCoef", 0.0)
        frames.append(cohort_segments)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _anonymized_exports(
    activity_df: pd.DataFrame,
    segment_features: pd.DataFrame,
    cohorts: Mapping[str, pd.DataFrame],
    stage3_segment_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    release_ids = sorted(segment_features["activityId"].astype(str).dropna().unique().tolist())
    activity_index_map = {activity_id: f"A{idx + 1:04d}" for idx, activity_id in enumerate(release_ids)}
    release_activity_df = activity_df[activity_df["activityId"].astype(str).isin(release_ids)].copy()
    release_activity_df["activityIndex"] = release_activity_df["activityId"].astype(str).map(activity_index_map)
    for cohort_name, cohort_df in cohorts.items():
        cohort_ids = set(cohort_df["activityId"].astype(str))
        release_activity_df[f"cohort_{cohort_name}"] = release_activity_df["activityId"].astype(str).isin(cohort_ids)

    activity_export_cols = [
        "activityIndex",
        *[f"cohort_{name}" for name in cohorts.keys()],
        "distanceKm",
        "ascentM",
        "actualTimeSec",
        "elapsedSec",
        "hrReserveRatio",
        "ctl",
        "tsb",
        "trimpRediSlow",
        "trimpRediBalance",
        "ctlReadinessFactor",
        "rediReadinessFactor",
        "meanAltitudeM",
        "technicalityGps",
        "temperatureC",
    ]
    anonymized_activity = release_activity_df[
        [col for col in activity_export_cols if col in release_activity_df.columns]
    ].sort_values("activityIndex")

    source_segments = (
        stage3_segment_predictions.copy() if not stage3_segment_predictions.empty else segment_features.copy()
    )
    if "stage3PredictedTimeSec" not in source_segments.columns:
        source_segments["stage3PredictedTimeSec"] = np.nan
        source_segments["stage3ResidualSec"] = np.nan
    source_segments["activityIndex"] = source_segments["activityId"].astype(str).map(activity_index_map)
    segment_export_cols = [
        "cohort",
        "fitObjective",
        "activityIndex",
        "segmentIndex",
        "startKm",
        "endKm",
        "distanceKm",
        "avgGrade",
        "meanAltitudeM",
        "elevGainM",
        "elevLossM",
        "progress",
        "gapFactor",
        "altitudePenalty",
        "meanHrReserve",
        "segmentTrimp",
        "cumTrimpBefore",
        "decayedTrimpBefore",
        "terrainFamily",
        "technicalityCombined",
        "actualTimeSec",
        "stage3PredictedTimeSec",
        "stage3ResidualSec",
        "stage3FatigueState",
        "stage3AcuteTrimpCol",
        "stage3FatigueModel",
    ]
    anonymized_segments = source_segments[
        [col for col in segment_export_cols if col in source_segments.columns]
    ].sort_values(
        [col for col in ["cohort", "fitObjective", "activityIndex", "segmentIndex"] if col in source_segments.columns]
    )
    feature_dictionary = pd.DataFrame(
        [
            {"feature": "fitObjective", "description": "Fitting objective used to select model hyperparameters."},
            {"feature": "stage3FatigueState", "description": "Selected Stage 3 acute fatigue state for this cohort."},
            {"feature": "stage3AcuteTrimpCol", "description": "Selected Stage 3 acute fatigue input column."},
            {"feature": "stage3FatigueModel", "description": "Selected Stage 3 fatigue shape."},
        ]
    )
    forbidden_activity = tpm.forbidden_anonymized_columns(anonymized_activity.columns)
    forbidden_segment = tpm.forbidden_anonymized_columns(anonymized_segments.columns)
    if forbidden_activity or forbidden_segment:
        raise ValueError(f"forbidden anonymized columns: {forbidden_activity + forbidden_segment}")
    return anonymized_activity, anonymized_segments, feature_dictionary


def run_pipeline(
    config: Mapping[str, Any], *, project_root: Path | None = None, config_path: Path | None = None
) -> PipelineResult:
    """Run the configured trail digital-twin extension pipeline."""
    root = _project_root(project_root)
    activity_df, _daily_metrics, hr_rest, hr_max, v_vt2_kmh = _load_activity_inputs(config, root)
    all_segments_df, qc_df, segments_by_activity = _build_segments(activity_df, config, root, hr_rest, hr_max)
    if all_segments_df.empty:
        raise ValueError("no usable activity segments were built")

    segment_summary = (
        all_segments_df.groupby("activityId")
        .agg(
            technicalityGps=("technicalityGps", "mean"),
            meanAltitudeM=("meanAltitudeM", "mean"),
            segmentDistanceKm=("distanceKm", "sum"),
            segmentActualTimeSec=("actualTimeSec", "sum"),
            meanSegmentHrReserve=("meanHrReserve", "mean"),
            meanSpeedEqKmh=("meanSpeedEqKmh", "mean"),
        )
        .reset_index()
    )
    activity_df = activity_df.merge(segment_summary, on="activityId", how="left")
    activity_df["usableSegmentActivity"] = activity_df["activityId"].astype(str).isin(segments_by_activity)
    activity_df["usableTrailRun"] = (
        activity_df["category"].astype(str).str.upper().eq("TRAIL_RUN") & activity_df["usableSegmentActivity"]
    )
    activity_df["hardTrailRun"] = tpm.hard_trailrun_mask(activity_df) & activity_df["usableSegmentActivity"]
    activity_df = _add_readiness_factors(activity_df, "ctl", "tsb", "ctlReadinessFactor", config["readiness"])
    activity_df = _add_readiness_factors(
        activity_df,
        "trimpRediSlow",
        "trimpRediBalance",
        "rediReadinessFactor",
        config["readiness"],
    )
    cohorts = _build_cohorts(activity_df, segments_by_activity, config)
    all_segments_with_trimp = tpm.add_in_activity_trimp_features(
        all_segments_df,
        decay_lambda=float(config["physiology"]["decay_lambda"]),
    )
    segment_features = add_segment_model_features(all_segments_with_trimp, activity_df)

    linear_metrics, variable_importance = run_linear_diagnostics(cohorts, segment_features)
    hr_comparison = run_hr_baselines(activity_df, segment_features, cohorts)
    stage_tables, stage3_best, stage3_segments = run_paper_stage_models(
        cohorts,
        segments_by_activity,
        segment_features,
        config,
        v_vt2_kmh,
    )
    stage3_ablation = run_stage3_ablation(stage3_best, stage3_segments, cohorts, config)
    segment_grid_metrics, segment_grid_predictions = run_segment_grid(cohorts, segment_features, config)
    robustness_checks = run_robustness_checks(cohorts, all_segments_df, activity_df, config)
    stage3_segment_predictions = _stage3_segment_predictions(stage3_best, stage3_segments, config)
    anonymized_activity, anonymized_segments, feature_dictionary = _anonymized_exports(
        activity_df,
        segment_features,
        cohorts,
        stage3_segment_predictions,
    )

    tables = {
        "table_cohort_descriptives": _cohort_descriptives(cohorts),
        "table_linear_diagnostic_metrics": linear_metrics,
        "table_full_regression_coefficients": variable_importance,
        "table_hr_regression_metrics": hr_comparison,
        "table_stage3_ablation": stage3_ablation,
        "table_robustness_checks": robustness_checks,
        "table_objective_comparison": _objective_comparison(stage_tables["table_stage_metrics"]),
        "table_segment_type_metrics": _segment_type_metrics(stage3_segment_predictions),
        "segment_predictions": stage3_segment_predictions,
        "exploratory_segment_grid_metrics": segment_grid_metrics,
        "exploratory_segment_grid_predictions": segment_grid_predictions,
        "anonymized_activity_features": anonymized_activity,
        "anonymized_segment_features": anonymized_segments,
        "table_anonymized_feature_dictionary": feature_dictionary,
        "segment_qc": qc_df,
        **stage_tables,
    }
    metadata = {
        "projectRoot": str(root),
        "configPath": str(config_path) if config_path else "",
        "hrRest": hr_rest,
        "hrMax": hr_max,
        "vVt2Kmh": v_vt2_kmh,
        "config": config,
    }
    return PipelineResult(tables=tables, metadata=metadata)


def _objective_comparison(stage_metrics: pd.DataFrame) -> pd.DataFrame:
    if stage_metrics.empty:
        return pd.DataFrame()
    comparison = stage_metrics[
        stage_metrics["stage"].astype(str).eq("Stage 3 HRR speed ratio LOO")
        | stage_metrics["stage"].astype(str).eq("Stage 3 HRR speed ratio")
    ].copy()
    return comparison.sort_values(["cohort", "stage", "fitObjective"]) if not comparison.empty else comparison


def _terrain_label(terrain_family: object) -> str:
    family = str(terrain_family) if pd.notna(terrain_family) else "unknown"
    return TERRAIN_FAMILY_LABELS.get(family, family.replace("_", " ").title())


def _segment_type_metrics(segments: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cohort",
        "fitObjective",
        "terrainFamily",
        "activityId",
        "actualTimeSec",
        "stage3PredictedTimeSec",
    }
    if segments.empty or not required.issubset(segments.columns):
        return pd.DataFrame()

    data = segments.copy()
    data["actualTimeSec"] = pd.to_numeric(data["actualTimeSec"], errors="coerce")
    data["predictedTimeSec"] = pd.to_numeric(data["stage3PredictedTimeSec"], errors="coerce")
    data = data[np.isfinite(data["actualTimeSec"]) & np.isfinite(data["predictedTimeSec"])]
    data = data[data["actualTimeSec"] > 0]
    if data.empty:
        return pd.DataFrame()

    data["terrainFamily"] = data["terrainFamily"].fillna("unknown").astype(str)
    data["residualSec"] = data["predictedTimeSec"] - data["actualTimeSec"]
    has_moving = "actualMovingTimeSec" in data.columns
    if has_moving:
        data["actualMovingTimeSec"] = pd.to_numeric(data["actualMovingTimeSec"], errors="coerce")
        data["residualMovingSec"] = data["predictedTimeSec"] - data["actualMovingTimeSec"]
    if "distanceKm" in data.columns:
        data["distanceKm"] = pd.to_numeric(data["distanceKm"], errors="coerce").fillna(0.0)
    else:
        data["distanceKm"] = 0.0

    rows = []
    for (cohort, objective, terrain), group in data.groupby(
        ["cohort", "fitObjective", "terrainFamily"],
        sort=False,
    ):
        metrics = tpm.regression_metrics(group["actualTimeSec"], group["predictedTimeSec"])
        residual = group["residualSec"].to_numpy(dtype=float)
        row = {
            "cohort": cohort,
            "fitObjective": objective,
            "terrainFamily": terrain,
            "terrainLabel": _terrain_label(terrain),
            "segmentCount": int(len(group)),
            "activityCount": int(group["activityId"].astype(str).nunique()),
            "distanceKm": float(group["distanceKm"].sum()),
            "actualMin": float(group["actualTimeSec"].sum() / 60.0),
            "predictedMin": float(group["predictedTimeSec"].sum() / 60.0),
            "r2": metrics["r2"],
            "maeMin": metrics["maeSec"] / 60.0,
            "rmseMin": float(np.sqrt(np.mean(residual**2)) / 60.0),
            "mapePct": metrics["mapePct"],
            "biasMin": metrics["biasSec"] / 60.0,
            "residualStdMin": float(np.std(residual, ddof=0) / 60.0),
        }
        if has_moving:
            moving_group = group[group["actualMovingTimeSec"].gt(1.0)]
            if not moving_group.empty:
                moving_metrics = tpm.regression_metrics(
                    moving_group["actualMovingTimeSec"],
                    moving_group["predictedTimeSec"],
                )
                moving_residual = moving_group["residualMovingSec"].to_numpy(dtype=float)
                row.update(
                    {
                        "maeMinMoving": moving_metrics["maeSec"] / 60.0,
                        "mapePctMoving": moving_metrics["mapePct"],
                        "biasMinMoving": moving_metrics["biasSec"] / 60.0,
                        "rmseMinMoving": float(np.sqrt(np.mean(moving_residual**2)) / 60.0),
                    }
                )
            else:
                row.update(
                    {
                        "maeMinMoving": np.nan,
                        "mapePctMoving": np.nan,
                        "biasMinMoving": np.nan,
                        "rmseMinMoving": np.nan,
                    }
                )
        rows.append(row)

    result = pd.DataFrame(rows)
    result["terrainOrder"] = result["terrainFamily"].map(TERRAIN_FAMILY_ORDER).fillna(99).astype(int)
    return result.sort_values(["cohort", "fitObjective", "terrainOrder", "terrainFamily"]).drop(
        columns=["terrainOrder"]
    )


def _format_number(value: object, digits: int = 1) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def _html_table(df: pd.DataFrame, table_id: str, max_rows: int = 200) -> str:
    if df.empty:
        return "<p class='empty'>No rows available.</p>"
    display_df = df.head(max_rows).copy()
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in display_df.columns)
    rows = []
    for _, row in display_df.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.fillna("").tolist())
        rows.append(f"<tr>{cells}</tr>")
    return (
        f"<div class='table-wrap'><table id='{html.escape(table_id)}' class='sortable'>"
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _plot_html(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False, "responsive": True})


def _finite_median(values: pd.Series, fallback: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return fallback
    median = float(numeric.median())
    return median if median >= 0.0 else fallback


def _stage3_kappa(tables: Mapping[str, pd.DataFrame], fatigue_model: str, fallback: float) -> float:
    params = tables.get("table_fitted_parameters", pd.DataFrame())
    if params.empty or not {"stage", "fatigueModel", "fatigueCoef"}.issubset(params.columns):
        return fallback
    stage3 = params[
        params["stage"].astype(str).eq("Stage 3 HRR speed ratio")
        & params["fatigueModel"].astype(str).eq(fatigue_model)
    ]
    return _finite_median(stage3["fatigueCoef"], fallback) if not stage3.empty else fallback


def _model_response_figure(result: PipelineResult) -> go.Figure:
    config = result.metadata.get("config", DEFAULT_CONFIG)
    physiology = config.get("physiology", DEFAULT_CONFIG["physiology"])
    readiness = config.get("readiness", DEFAULT_CONFIG["readiness"])
    fitting = config.get("fitting", DEFAULT_CONFIG["fitting"])
    min_fatigue = float(physiology.get("min_fatigue_factor", 0.50))
    kappa_grid = [float(value) for value in fitting.get("hrr_trimp_kappa_grid", [0.20])]
    fallback_kappa = kappa_grid[min(len(kappa_grid) - 1, max(0, len(kappa_grid) // 2))]
    linear_kappa = _stage3_kappa(result.tables, "linear", fallback_kappa)
    exponential_kappa = _stage3_kappa(result.tables, "exponential", linear_kappa)

    segments = result.tables.get("segment_predictions", pd.DataFrame())
    trimp_candidates = []
    for col in ["decayedTrimpBefore", "cumTrimpBefore"]:
        if col in segments.columns:
            trimp_candidates.append(pd.to_numeric(segments[col], errors="coerce"))
    if trimp_candidates:
        trimp_values = pd.concat(trimp_candidates).dropna()
        trimp_max = float(trimp_values.quantile(0.95)) if not trimp_values.empty else 5.0
    else:
        trimp_max = 5.0
    trimp_max = max(1.0, min(25.0, trimp_max * 1.10))

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "HRR speed multiplier",
            "Acute fatigue multiplier",
            "Readiness multiplier",
            "Grade-adjusted pace factor",
        ),
    )

    hrr_reference = float(physiology.get("hrr_reference", 0.70))
    hrr_x = np.linspace(0.30, max(1.05, hrr_reference * 1.6), 160)
    hrr_y = np.clip(
        hrr_x / max(hrr_reference, 1e-6),
        float(physiology.get("hrr_min_factor", 0.55)),
        float(physiology.get("hrr_max_factor", 1.30)),
    )
    fig.add_trace(
        go.Scatter(x=hrr_x, y=hrr_y, mode="lines", name="f_HRR(HRR)"),
        row=1,
        col=1,
    )
    fig.add_vline(x=hrr_reference, line_dash="dot", line_color="#64748b", row=1, col=1)

    trimp_x = np.linspace(0.0, trimp_max, 160)
    progress_x = np.linspace(0.0, 1.0, 160)
    fig.add_trace(
        go.Scatter(
            x=trimp_x,
            y=tpm._trimp_fatigue_from_load(
                trimp_x,
                fatigue_coef=linear_kappa,
                fatigue_model="linear",
                min_factor=min_fatigue,
            ),
            mode="lines",
            name=f"linear TRIMP kappa={linear_kappa:.3g}",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=trimp_x,
            y=tpm._trimp_fatigue_from_load(
                trimp_x,
                fatigue_coef=exponential_kappa,
                fatigue_model="exponential",
                min_factor=min_fatigue,
            ),
            mode="lines",
            name=f"exponential TRIMP kappa={exponential_kappa:.3g}",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=progress_x,
            y=tpm._trimp_fatigue_from_load(
                progress_x,
                fatigue_coef=linear_kappa,
                fatigue_model="linear",
                min_factor=min_fatigue,
            ),
            mode="lines",
            name="linear progress",
            line={"dash": "dash"},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=trimp_x,
            y=np.ones_like(trimp_x),
            mode="lines",
            name="no acute fatigue",
            line={"dash": "dot", "color": "#475569"},
        ),
        row=1,
        col=2,
    )

    ctl_x = np.linspace(0.50, 1.50, 120)
    ctl_factor = [
        tpm.ctl_readiness_factor(
            ctl,
            0.0,
            ctl_reference=1.0,
            ctl_weight=float(readiness.get("ctl_weight", 0.05)),
            tsb_weight=float(readiness.get("tsb_weight", 0.10)),
            min_factor=float(readiness.get("ctl_factor_min", 0.90)),
            max_factor=float(readiness.get("ctl_factor_max", 1.08)),
        )
        for ctl in ctl_x
    ]
    tsb_x = np.linspace(-0.50, 0.50, 120)
    tsb_factor = [
        tpm.ctl_readiness_factor(
            1.0,
            tsb,
            ctl_reference=1.0,
            ctl_weight=float(readiness.get("ctl_weight", 0.05)),
            tsb_weight=float(readiness.get("tsb_weight", 0.10)),
            min_factor=float(readiness.get("ctl_factor_min", 0.90)),
            max_factor=float(readiness.get("ctl_factor_max", 1.08)),
        )
        for tsb in tsb_x
    ]
    fig.add_trace(go.Scatter(x=ctl_x, y=ctl_factor, mode="lines", name="CTL/ref, TSB=0"), row=2, col=1)
    fig.add_trace(go.Scatter(x=tsb_x, y=tsb_factor, mode="lines", name="TSB/ref, CTL=ref"), row=2, col=1)

    grade_x = np.linspace(-1.0, 1.0, 240)
    gap_y = [tpm.gap_factor(float(grade)) for grade in grade_x]
    fig.add_trace(go.Scatter(x=grade_x, y=gap_y, mode="lines", name="f_GAP(grade)"), row=2, col=2)
    fig.add_hline(y=1.0, line_dash="dot", line_color="#64748b", row=2, col=2)
    fig.add_vline(x=-tpm.MINETTI_GRADE_CLAMP, line_dash="dot", line_color="#64748b", row=2, col=2)
    fig.add_vline(x=tpm.MINETTI_GRADE_CLAMP, line_dash="dot", line_color="#64748b", row=2, col=2)

    fig.update_xaxes(title_text="HR reserve ratio", row=1, col=1)
    fig.update_xaxes(title_text="raw TRIMP load or progress", row=1, col=2)
    fig.update_xaxes(title_text="relative load balance", row=2, col=1)
    fig.update_xaxes(title_text="grade", tickformat=".0%", row=2, col=2)
    fig.update_yaxes(title_text="speed factor", row=1, col=1)
    fig.update_yaxes(title_text="speed factor", row=1, col=2)
    fig.update_yaxes(title_text="speed factor", row=2, col=1)
    fig.update_yaxes(title_text="pace factor", row=2, col=2)
    fig.update_layout(
        title="Model response functions used by Stage 3",
        height=760,
        margin={"l": 55, "r": 30, "t": 85, "b": 55},
        legend={"orientation": "h", "y": -0.16},
    )
    return fig


def _metric_definitions_html() -> str:
    metrics = pd.DataFrame(
        [
            {
                "metric": "MAE min",
                "meaning": "Mean absolute prediction error in minutes. Lower is better.",
            },
            {
                "metric": "MAPE pct",
                "meaning": "Mean absolute percentage error against observed activity or segment time.",
            },
            {
                "metric": "R2",
                "meaning": "Variance explained by the predictions. Higher is better, but unstable on tiny cohorts.",
            },
            {
                "metric": "Bias min",
                "meaning": "Mean signed error in minutes. Positive means predicted too slow; negative means too fast.",
            },
            {
                "metric": "Fit objective",
                "meaning": (
                    "Activity fits choose parameters from race-summed errors; "
                    "segment fits choose from segment errors."
                ),
            },
            {
                "metric": "LOO",
                "meaning": "Leave-one-activity-out validation: fit on all other activities, predict the held-out one.",
            },
            {
                "metric": "No acute fatigue",
                "meaning": (
                    "The acute fatigue multiplier is fixed to 1 by setting kappa=0. "
                    "It is not a linear fatigue model."
                ),
            },
        ]
    )
    return _html_table(metrics, "metric-definitions")


def _model_formulas_html() -> str:
    rows = pd.DataFrame(
        [
            {
                "component": "Stage 3 time",
                "formula": (
                    "t = d * 3600 * f_GAP(grade) / "
                    "(VMA * alpha * f_alt(alt) * f_readiness * f_HRR(HRR) * f_fatigue(U))"
                ),
                "notes": "All multipliers except f_GAP are speed multipliers; f_GAP is a pace cost factor.",
            },
            {
                "component": "HRR speed",
                "formula": "f_HRR(HRR) = clip(HRR / HRR_ref, hrr_min_factor, hrr_max_factor)",
                "notes": "Fixed effort-to-speed ratio, with bounds learned from config/benchmark choices.",
            },
            {
                "component": "Linear fatigue",
                "formula": "f_fatigue(U) = clip(1 - kappa * U, min_fatigue_factor, 1)",
                "notes": "U can be raw decayed TRIMP, raw cumulative TRIMP, or progress in [0, 1].",
            },
            {
                "component": "Exponential fatigue",
                "formula": "f_fatigue(U) = clip(exp(-kappa * U), min_fatigue_factor, 1)",
                "notes": "Same load choices as linear fatigue, but with a curved decay.",
            },
            {
                "component": "Combined fatigue",
                "formula": "f_fatigue(D,M) = clip(exp(-kappa * D) * exp(-gamma * M), min_fatigue_factor, 1)",
                "notes": "D is decayed TRIMP; M is cumulative TRIMP or progress, with fitted gamma.",
            },
            {
                "component": "No acute fatigue",
                "formula": "kappa = 0, therefore f_fatigue(U) = 1",
                "notes": "This disables acute in-race fatigue modeling. It is not the linear alternative.",
            },
            {
                "component": "Readiness",
                "formula": "f_readiness = clip(1 + ctl_weight*(CTL-ref)/ref + tsb_weight*TSB/ref, min, max)",
                "notes": (
                    "Local bounded heuristic. The reference is the median CTL or REDI slow-load value "
                    "over the current activity dataset, with fallback ref=1."
                ),
            },
            {
                "component": "GAP",
                "formula": "f_GAP(grade) = MinettiCost(grade) / MinettiCost(0)",
                "notes": (
                    "Integrated segment GAP is preferred when available; mean grade is the fallback. "
                    f"The plotted grade range is -100% to +100%, while Minetti cost is clamped to "
                    f"+/-{tpm.MINETTI_GRADE_CLAMP:.0%}."
                ),
            },
            {
                "component": "Altitude",
                "formula": "f_alt(alt) = max(0.1, 1 - 11.7e-9*alt^2 - 4.01e-6*alt)",
                "notes": "Altitude VO2max correction from the trail digital-twin formulation.",
            },
        ]
    )
    return _html_table(rows, "model-formulas")


def _score_min_for_grid(data: pd.DataFrame) -> pd.Series:
    race = pd.to_numeric(data.get("raceMaeMin", pd.Series(np.nan, index=data.index)), errors="coerce")
    segment = pd.to_numeric(data.get("segmentMaeMin", pd.Series(np.nan, index=data.index)), errors="coerce")
    if "fitObjective" not in data.columns:
        return race.fillna(segment)
    objective = data["fitObjective"].astype(str)
    return segment.where(objective.eq("segment"), race).fillna(race).fillna(segment)


def _stage3_grid_search(search: pd.DataFrame) -> pd.DataFrame:
    required = {"stage", "alpha", "fatigueCoef"}
    if search.empty or not required.issubset(search.columns):
        return pd.DataFrame()
    data = search[search["stage"].astype(str).eq("Stage 3 HRR speed ratio")].copy()
    if data.empty:
        data = search[search["stage"].astype(str).str.startswith("Stage 3 HRR speed ratio")].copy()
    if data.empty:
        return data
    data["alpha"] = pd.to_numeric(data["alpha"], errors="coerce")
    data["fatigueCoef"] = pd.to_numeric(data["fatigueCoef"], errors="coerce")
    data["scoreMin"] = _score_min_for_grid(data)
    return data.dropna(subset=["alpha", "fatigueCoef", "scoreMin"])


def _optimized_parameter_table(result: PipelineResult) -> pd.DataFrame:
    params = result.tables.get("table_fitted_parameters", pd.DataFrame())
    if params.empty:
        return pd.DataFrame()
    data = params.copy()
    config = result.metadata.get("config", {})
    physiology = config.get("physiology", {}) if isinstance(config, Mapping) else {}
    for column, key in [
        ("vmaFlatKmh", "vma_flat_kmh"),
        ("hrrReference", "hrr_reference"),
        ("hrrMinFactor", "hrr_min_factor"),
        ("hrrMaxFactor", "hrr_max_factor"),
        ("minFatigueFactor", "min_fatigue_factor"),
        ("decayLambda", "decay_lambda"),
    ]:
        if column not in data.columns and key in physiology:
            data[column] = physiology[key]
    if "raceMaeSec" in data.columns and "raceMaeMin" not in data.columns:
        data["raceMaeMin"] = pd.to_numeric(data["raceMaeSec"], errors="coerce") / 60.0
    if "segmentMaeSec" in data.columns and "segmentMaeMin" not in data.columns:
        data["segmentMaeMin"] = pd.to_numeric(data["segmentMaeSec"], errors="coerce") / 60.0
    columns = [
        "cohort",
        "fitObjective",
        "stage",
        "alpha",
        "mu",
        "fatigueCoef",
        "fatigueModel",
        "fatigueState",
        "acuteTrimpCol",
        "secondaryFatigueCoef",
        "secondaryFatigueModel",
        "secondaryAcuteTrimpCol",
        "raceMaeMin",
        "segmentMaeMin",
        "hrrReference",
        "hrrMinFactor",
        "hrrMaxFactor",
        "minFatigueFactor",
        "decayLambda",
        "vmaFlatKmh",
    ]
    selected = data[[column for column in columns if column in data.columns]].copy()
    sort_cols = [column for column in ["cohort", "fitObjective", "stage"] if column in selected.columns]
    return selected.sort_values(sort_cols) if sort_cols else selected


def _top_parameter_search_table(search: pd.DataFrame, max_rows: int = 40) -> pd.DataFrame:
    data = _stage3_grid_search(search)
    if data.empty:
        return pd.DataFrame()
    columns = [
        "cohort",
        "fitObjective",
        "stage",
        "alpha",
        "fatigueCoef",
        "fatigueModel",
        "fatigueState",
        "acuteTrimpCol",
        "secondaryFatigueCoef",
        "secondaryFatigueModel",
        "secondaryAcuteTrimpCol",
        "scoreMin",
        "raceMaeMin",
        "segmentMaeMin",
        "raceR2",
        "segmentR2",
        "hrrReference",
        "hrrMinFactor",
        "hrrMaxFactor",
        "minFatigueFactor",
        "decayLambda",
    ]
    selected = data[[column for column in columns if column in data.columns]].copy()
    return selected.sort_values(["scoreMin", "cohort", "fitObjective"]).head(max_rows)


def _parameter_selection_figure(params: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    required = {"stage", "cohort", "fitObjective", "alpha", "fatigueCoef"}
    if params.empty or not required.issubset(params.columns):
        return fig
    data = params[params["stage"].astype(str).eq("Stage 3 HRR speed ratio")].copy()
    if data.empty:
        return fig
    data["alpha"] = pd.to_numeric(data["alpha"], errors="coerce")
    data["fatigueCoef"] = pd.to_numeric(data["fatigueCoef"], errors="coerce")
    data = data.dropna(subset=["alpha", "fatigueCoef"])
    for (cohort, objective), group in data.groupby(["cohort", "fitObjective"], sort=True):
        fig.add_trace(
            go.Scatter(
                x=group["alpha"],
                y=group["fatigueCoef"],
                mode="markers+text",
                name=f"{cohort} / {objective}",
                text=group["cohort"],
                textposition="top center",
                marker={"size": 13},
                customdata=np.stack(
                    [
                        group.get("fatigueModel", pd.Series("", index=group.index)),
                        group.get("fatigueState", pd.Series("", index=group.index)),
                        group.get("raceMaeSec", pd.Series(np.nan, index=group.index)),
                        group.get("segmentMaeSec", pd.Series(np.nan, index=group.index)),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Cohort=%{text}<br>Objective="
                    f"{html.escape(str(objective))}<br>alpha=%{{x:.3g}}<br>kappa=%{{y:.3g}}<br>"
                    "Fatigue=%{customdata[1]} / %{customdata[0]}<br>"
                    "Race MAE=%{customdata[2]:.1f} sec<br>Segment MAE=%{customdata[3]:.1f} sec"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Selected Stage 3 alpha and kappa by cohort",
        xaxis_title="alpha: base fraction of VMA",
        yaxis_title="kappa: acute fatigue coefficient",
        margin={"l": 55, "r": 30, "t": 60, "b": 65},
        legend={"orientation": "h", "y": -0.28},
    )
    return fig


def _alpha_kappa_search_figure(search: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    data = _stage3_grid_search(search)
    if data.empty:
        return fig
    best_per_context = (
        data.sort_values("scoreMin")
        .groupby(["cohort", "fitObjective", "alpha", "fatigueCoef"], as_index=False)
        .first()
    )
    surface = (
        best_per_context.groupby(["fatigueCoef", "alpha"], as_index=False)
        .agg(scoreMin=("scoreMin", "mean"), contextCount=("cohort", "size"))
        .sort_values(["fatigueCoef", "alpha"])
    )
    pivot = surface.pivot_table(index="fatigueCoef", columns="alpha", values="scoreMin", aggfunc="mean")
    if pivot.empty:
        return fig
    fig.add_trace(
        go.Heatmap(
            x=pivot.columns.astype(float),
            y=pivot.index.astype(float),
            z=pivot.to_numpy(dtype=float),
            colorscale="Viridis",
            colorbar={"title": "MAE min"},
            hovertemplate="alpha=%{x:.3g}<br>kappa=%{y:.3g}<br>Mean best MAE=%{z:.2f} min<extra></extra>",
        )
    )
    fig.update_layout(
        title="Stage 3 alpha-kappa search surface",
        xaxis_title="alpha: base fraction of VMA",
        yaxis_title="kappa: acute fatigue coefficient",
        margin={"l": 55, "r": 30, "t": 60, "b": 55},
    )
    return fig


def _parameter_search_html(result: PipelineResult) -> str:
    params = result.tables.get("table_fitted_parameters", pd.DataFrame())
    search = result.tables.get("table_hrr_trimp_grid_search", pd.DataFrame())
    optimized = _optimized_parameter_table(result)
    top_search = _top_parameter_search_table(search)
    return f"""
<section>
<h2>Optimized physiological parameters</h2>
<p>The fitted table exposes the selected model parameters for each cohort and fit objective. Stage 3 alpha is the
base VMA fraction, kappa is stored as fatigueCoef, and combined variants store the extra muscular coefficient
as secondaryFatigueCoef.</p>
{_html_table(optimized, "optimized-parameters")}
</section>
<section>
<h2>Selected alpha-kappa coordinates</h2>
<p>Each marker is the final Stage 3 parameter choice after searching alpha, kappa, optional secondary kappa,
fatigue state, and fatigue shape. The same physical config bounds, HRR reference, and fatigue floor are shown
in the optimized-parameter table.</p>
<div class='chart'>{_plot_html(_parameter_selection_figure(params))}</div>
</section>
<section>
<h2>Alpha-kappa search surface</h2>
<p>The heatmap summarizes the Stage 3 grid search. For each alpha-kappa pair, it takes the best fatigue state/shape
within each cohort/objective context, then averages the resulting MAE in minutes; lower cells are better.</p>
<div class='chart'>{_plot_html(_alpha_kappa_search_figure(search))}</div>
</section>
<section>
<h2>Top alpha-kappa search cells</h2>
<p>This table keeps the best individual Stage 3 grid cells for audit, including primary and secondary fatigue
settings, HRR bounds, fatigue floor, and decay lambda used during the search.</p>
{_html_table(top_search, "top-alpha-kappa-search")}
</section>
"""


def _stage_metrics_figure(stage_metrics: pd.DataFrame) -> go.Figure:
    if stage_metrics.empty or not {"stage", "cohort", "fitObjective", "maeMin"}.issubset(stage_metrics.columns):
        return go.Figure()
    data = stage_metrics[stage_metrics["stage"].astype(str).str.endswith("LOO")].copy()
    if data.empty:
        data = stage_metrics.copy()
    fig = go.Figure()
    for stage in STAGE_ORDER:
        stage_rows = data[data["stage"].astype(str).eq(f"{stage} LOO") | data["stage"].astype(str).eq(stage)]
        if stage_rows.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[stage_rows["cohort"], stage_rows["fitObjective"]],
                y=stage_rows["maeMin"],
                name=stage.replace("Stage ", "S"),
                hovertemplate="Cohort=%{x[0]}<br>Objective=%{x[1]}<br>MAE=%{y:.1f} min<extra></extra>",
            )
        )
    fig.update_layout(
        title="Stage ladder LOO MAE by cohort and fitting objective",
        barmode="group",
        yaxis_title="MAE (min)",
        legend_title="Stage",
        margin={"l": 50, "r": 30, "t": 60, "b": 90},
    )
    return fig


def _prediction_scatter(predictions: pd.DataFrame) -> go.Figure:
    required = {"model", "actualMin", "predictedMin", "cohort", "fitObjective", "errorMin", "activityId"}
    if predictions.empty or not required.issubset(predictions.columns):
        return go.Figure()
    data = predictions[predictions["model"].astype(str).eq("Stage 3 HRR speed ratio")].copy()
    if data.empty:
        data = predictions.copy()
    limit = float(max(data["actualMin"].max(), data["predictedMin"].max()) * 1.05) if not data.empty else 1.0
    fig = go.Figure()
    for (cohort, objective), group in data.groupby(["cohort", "fitObjective"], sort=True):
        fig.add_trace(
            go.Scatter(
                x=group["actualMin"],
                y=group["predictedMin"],
                mode="markers",
                name=f"{cohort} / {objective}",
                text=group.get("name", group["activityId"]),
                customdata=np.stack([group["errorMin"], group["activityId"]], axis=-1),
                hovertemplate="%{text}<br>Activity=%{customdata[1]}<br>Actual=%{x:.1f} min<br>"
                "Predicted=%{y:.1f} min<br>Error=%{customdata[0]:.1f} min<extra></extra>",
            )
        )
    fig.add_trace(go.Scatter(x=[0, limit], y=[0, limit], mode="lines", name="identity", line={"color": "#333"}))
    fig.update_layout(
        title="Stage 3 activity-level predictions",
        xaxis_title="Actual time (min)",
        yaxis_title="Predicted time (min)",
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
    )
    return fig


def _stage3_fatigue_figure(comparison: pd.DataFrame) -> go.Figure:
    required = {"validation", "fatigueState", "fatigueModel", "cohort", "fitObjective", "maeMin"}
    if comparison.empty or not required.issubset(comparison.columns):
        return go.Figure()
    data = comparison[comparison["validation"].astype(str).eq("loo")].copy()
    if data.empty:
        data = comparison.copy()
    data["variant"] = data["fatigueState"].astype(str) + " / " + data["fatigueModel"].astype(str)
    fig = go.Figure()
    for objective, group in data.groupby("fitObjective", sort=True):
        fig.add_trace(
            go.Bar(
                x=[group["cohort"], group["variant"]],
                y=group["maeMin"],
                name=objective,
                hovertemplate="Cohort=%{x[0]}<br>Variant=%{x[1]}<br>MAE=%{y:.1f} min<extra></extra>",
            )
        )
    fig.update_layout(
        title="Stage 3 fatigue state and shape comparison",
        barmode="group",
        yaxis_title="MAE (min)",
        margin={"l": 50, "r": 30, "t": 60, "b": 100},
    )
    return fig


def _segment_residual_figure(segments: pd.DataFrame) -> go.Figure:
    required = {
        "cohort",
        "fitObjective",
        "stage3ResidualSec",
        "progress",
        "meanHrReserve",
        "activityId",
    }
    if segments.empty or not required.issubset(segments.columns):
        return go.Figure()
    data = segments.copy()
    data["residualMin"] = pd.to_numeric(data["stage3ResidualSec"], errors="coerce") / 60.0
    fig = go.Figure()
    for (cohort, objective), group in data.groupby(["cohort", "fitObjective"], sort=True):
        fig.add_trace(
            go.Scatter(
                x=group["progress"],
                y=group["residualMin"],
                mode="markers",
                name=f"{cohort} / {objective}",
                text=group.get("terrainFamily", ""),
                customdata=np.stack([group["meanHrReserve"], group["activityId"]], axis=-1),
                hovertemplate="Activity=%{customdata[1]}<br>Terrain=%{text}<br>Progress=%{x:.2f}<br>"
                "HRR=%{customdata[0]:.2f}<br>Residual=%{y:.1f} min<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color="#333", line_width=1)
    fig.update_layout(
        title="Stage 3 segment residuals by progress",
        xaxis_title="Activity progress",
        yaxis_title="Residual (min)",
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
    )
    return fig


def _segment_type_figure(segment_type_metrics: pd.DataFrame) -> go.Figure:
    required = {
        "cohort",
        "fitObjective",
        "terrainFamily",
        "terrainLabel",
        "segmentCount",
        "activityCount",
        "distanceKm",
        "maeMin",
        "biasMin",
        "mapePct",
    }
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("MAE by segment type", "Signed bias by segment type"),
    )
    if segment_type_metrics.empty or not required.issubset(segment_type_metrics.columns):
        return fig

    data = segment_type_metrics.copy()
    data["terrainOrder"] = data["terrainFamily"].map(TERRAIN_FAMILY_ORDER).fillna(99).astype(int)
    data = data.sort_values(["terrainOrder", "terrainFamily", "cohort", "fitObjective"])
    for (cohort, objective), group in data.groupby(["cohort", "fitObjective"], sort=True):
        context = f"{cohort} / {objective}"
        customdata = np.stack(
            [
                group["segmentCount"],
                group["activityCount"],
                group["distanceKm"],
                group["mapePct"],
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Bar(
                x=group["terrainLabel"],
                y=group["maeMin"],
                name=context,
                customdata=customdata,
                hovertemplate=(
                    "Segment type=%{x}<br>Context="
                    f"{html.escape(context)}<br>MAE=%{{y:.2f}} min<br>"
                    "Segments=%{customdata[0]:.0f}<br>Activities=%{customdata[1]:.0f}<br>"
                    "Distance=%{customdata[2]:.1f} km<br>MAPE=%{customdata[3]:.1f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=group["terrainLabel"],
                y=group["biasMin"],
                name=context,
                customdata=customdata,
                showlegend=False,
                hovertemplate=(
                    "Segment type=%{x}<br>Context="
                    f"{html.escape(context)}<br>Bias=%{{y:.2f}} min<br>"
                    "Positive means predicted slower than observed.<br>"
                    "Segments=%{customdata[0]:.0f}<br>Activities=%{customdata[1]:.0f}<br>"
                    "Distance=%{customdata[2]:.1f} km<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    fig.add_hline(y=0, line_color="#333", line_width=1, row=1, col=2)
    fig.update_layout(
        title="Stage 3 error by terrain segment type",
        barmode="group",
        yaxis_title="MAE (min)",
        yaxis2_title="Bias (min)",
        margin={"l": 55, "r": 30, "t": 70, "b": 100},
        legend={"orientation": "h", "y": -0.30},
    )
    return fig


def _ablation_figure(ablation: pd.DataFrame) -> go.Figure:
    required = {"stage", "cohort", "fitObjective", "deltaMaeMinVsFull"}
    if ablation.empty or not required.issubset(ablation.columns):
        return go.Figure()
    data = ablation[~ablation["stage"].astype(str).eq("full")].copy() if not ablation.empty else ablation
    fig = go.Figure()
    for objective, group in data.groupby("fitObjective", sort=True):
        fig.add_trace(
            go.Bar(
                x=[group["cohort"], group["stage"]],
                y=group["deltaMaeMinVsFull"],
                name=objective,
                hovertemplate="Cohort=%{x[0]}<br>Ablation=%{x[1]}<br>Delta=%{y:.1f} min<extra></extra>",
            )
        )
    fig.update_layout(
        title="Stage 3 one-variable ablation",
        barmode="group",
        yaxis_title="Delta MAE versus full Stage 3 (min)",
        margin={"l": 50, "r": 30, "t": 60, "b": 110},
    )
    return fig


def _robustness_figure(robustness: pd.DataFrame) -> go.Figure:
    if robustness.empty or "checkType" not in robustness.columns:
        return go.Figure()
    data = robustness[robustness["checkType"].astype(str).eq("stage3_grid_sensitivity")].copy()
    fig = go.Figure()
    if data.empty:
        return fig
    for cohort, group in data.groupby("cohort", sort=True):
        best = group.sort_values("maeMin").head(12)
        labels = (
            "ref "
            + best["hrrReference"].astype(str)
            + " / decay "
            + best["decayLambda"].astype(str)
        )
        fig.add_trace(go.Bar(x=labels, y=best["maeMin"], name=cohort))
    fig.update_layout(
        title="Best robustness-grid cells by cohort",
        barmode="group",
        yaxis_title="In-sample MAE (min)",
        margin={"l": 50, "r": 30, "t": 60, "b": 120},
    )
    return fig


def render_html_report(result: PipelineResult, output_files: Mapping[str, Path] | None = None) -> str:
    """Render a self-contained Plotly HTML evaluation report."""
    tables = result.tables
    metadata = result.metadata
    cohort_table = tables.get("table_cohort_descriptives", pd.DataFrame())
    cards = []
    for _, row in cohort_table.iterrows():
        cards.append(
            "<div class='card'>"
            f"<h3>{html.escape(str(row['cohort']))}</h3>"
            f"<p><strong>{int(row['activityCount'])}</strong> activities</p>"
            f"<p>{_format_number(row['distanceKmTotal'])} km, D+ {_format_number(row['ascentMTotal'], 0)} m</p>"
            f"<p>Mean HRR {_format_number(row['hrrMean'], 3)}</p>"
            "</div>"
        )
    config_snapshot = json.dumps(metadata.get("config", {}), indent=2, default=str)
    output_rows = pd.DataFrame([{"asset": name, "path": str(path)} for name, path in (output_files or {}).items()])
    activity_table = tables.get("activity_predictions", pd.DataFrame()).copy()
    privacy_mode = str(metadata.get("config", {}).get("outputs", {}).get("privacy_mode", "local"))
    if privacy_mode == "anonymized" and not activity_table.empty:
        ids = sorted(activity_table["activityId"].astype(str).dropna().unique())
        id_map = {activity_id: f"A{idx + 1:04d}" for idx, activity_id in enumerate(ids)}
        activity_table["activityId"] = activity_table["activityId"].astype(str).map(id_map)
        if "name" in activity_table.columns:
            activity_table = activity_table.drop(columns=["name"])
    segment_type_metrics = tables.get("table_segment_type_metrics", pd.DataFrame())
    if segment_type_metrics.empty:
        segment_type_metrics = _segment_type_metrics(tables.get("segment_predictions", pd.DataFrame()))

    sections = [
        ("Model response functions", _plot_html(_model_response_figure(result))),
        ("Stage ladder", _plot_html(_stage_metrics_figure(tables.get("table_stage_metrics", pd.DataFrame())))),
        ("Activity predictions", _plot_html(_prediction_scatter(tables.get("activity_predictions", pd.DataFrame())))),
        (
            "Stage 3 fatigue variants",
            _plot_html(_stage3_fatigue_figure(tables.get("table_stage3_fatigue_state_comparison", pd.DataFrame()))),
        ),
        ("Segment residuals", _plot_html(_segment_residual_figure(tables.get("segment_predictions", pd.DataFrame())))),
        ("Segment-type evaluation", _plot_html(_segment_type_figure(segment_type_metrics))),
        ("Ablation", _plot_html(_ablation_figure(tables.get("table_stage3_ablation", pd.DataFrame())))),
        ("Robustness", _plot_html(_robustness_figure(tables.get("table_robustness_checks", pd.DataFrame())))),
    ]
    figure_html = "\n".join(
        f"<section><h2>{html.escape(title)}</h2><div class='chart'>{body}</div></section>" for title, body in sections
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trail Digital Twin Evaluation Report</title>
<style>
:root {{ --ink:#1f2937; --muted:#667085; --line:#d0d5dd; --bg:#f8fafc; --panel:#ffffff; --accent:#2563eb; }}
body {{ margin:0; font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:var(--ink); background:var(--bg); }}
header {{ padding:28px 36px 18px; background:#fff; border-bottom:1px solid var(--line); }}
main {{ max-width:1280px; margin:0 auto; padding:24px 24px 48px; }}
h1 {{ margin:0 0 8px; font-size:30px; }}
h2 {{ margin:0 0 14px; font-size:20px; }}
h3 {{ margin:0 0 8px; font-size:16px; }}
p {{ color:var(--muted); }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; margin:0 0 18px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin:16px 0 0; }}
.card {{ border:1px solid var(--line); border-radius:8px; padding:14px; background:#fff; }}
.card p {{ margin:4px 0; }}
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
<h1>Trail Digital Twin Evaluation Report</h1>
<p>Configurable Stage 0-3, HRR/TRIMP, activity-level and segment-level fitting comparison.</p>
</header>
<main>
<section>
<h2>Cohort summary</h2>
<div class="grid">{"".join(cards)}</div>
</section>
<section>
<h2>Metric definitions</h2>
{_metric_definitions_html()}
</section>
<section>
<h2>Model formulas</h2>
{_model_formulas_html()}
</section>
{_parameter_search_html(result)}
{figure_html}
<section>
	<h2>Activity-level results table</h2>
	{_html_table(activity_table, "activity-results")}
	</section>
	<section>
	<h2>Segment-type evaluation table</h2>
	<p>Stage 3 segment errors grouped by terrain family. Positive bias means the model predicts slower
	segments than observed; negative bias means it predicts faster segments than observed.</p>
	{_html_table(segment_type_metrics, "segment-type-metrics")}
	</section>
	<section>
	<h2>Objective comparison</h2>
	{_html_table(tables.get("table_objective_comparison", pd.DataFrame()), "objective-comparison")}
	</section>
<section>
<h2>Config snapshot</h2>
<pre>{html.escape(config_snapshot)}</pre>
</section>
<section>
<h2>Provenance</h2>
<p>Project root: {html.escape(str(metadata.get("projectRoot", "")))}</p>
<p>Config path: {html.escape(str(metadata.get("configPath", "")))}</p>
{_html_table(output_rows, "outputs")}
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


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("_No rows available._\n")
        return
    text_df = df.fillna("").astype(str)
    headers = text_df.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in text_df.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_outputs(result: PipelineResult, output_dir: Path) -> dict[str, Path]:
    """Write configured pipeline outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    config = result.metadata.get("config", {})
    if bool(config.get("outputs", {}).get("write_csv", True)):
        for name, table in result.tables.items():
            if table.empty and name not in {"table_robustness_checks", "exploratory_segment_grid_metrics"}:
                continue
            filename = f"{name}.csv"
            path = output_dir / filename
            table.to_csv(path, index=False)
            written[name] = path
        model_equations = pd.DataFrame(
            [
                {
                    "stage": "Stage 0",
                    "description": "Sensors-style reproduction Stage 3 with CTL readiness and progress fatigue",
                    "equation": r"t = d 3600 f_GAP / (v_VT2 alpha f_alt f_CTL f_pad(p))",
                },
                {
                    "stage": "Stage 1",
                    "description": "Replace progress fatigue with raw decayed acute TRIMP fatigue",
                    "equation": r"t = d 3600 f_GAP / (VMA alpha f_alt f_CTL F(D_raw))",
                },
                {
                    "stage": "Stage 2",
                    "description": "Replace CTL readiness with REDI readiness",
                    "equation": r"t = d 3600 f_GAP / (VMA alpha f_alt f_REDI F(D))",
                },
                {
                    "stage": "Stage 3",
                    "description": (
                        "Add fixed linear HRR speed ratio and select raw decayed TRIMP, "
                        "raw cumulative TRIMP, progress fatigue, or decayed-plus-muscular fatigue"
                    ),
                    "equation": (
                        r"t = d 3600 f_GAP / (VMA alpha f_alt f_REDI E(HRR) F(U)), "
                        r"U in {D_raw, C_raw, progress}; combined uses F(D_raw,M)"
                    ),
                },
                {
                    "stage": "Stage 3 fatigue",
                    "description": (
                        "Linear/exponential fatigue use the selected raw load directly; "
                        "combined fatigue adds a fitted muscular coefficient gamma; no acute fatigue is kappa=0"
                    ),
                    "equation": (
                        r"F_linear(U)=clip(1-kappa U), F_exp(U)=clip(exp(-kappa U)), "
                        r"F_combined(D,M)=clip(exp(-kappa D) exp(-gamma M)), F_none(U)=1"
                    ),
                },
            ]
        )
        model_path = output_dir / "table_model_equations.md"
        _write_markdown_table(model_equations, model_path)
        written["table_model_equations"] = model_path

    manifest = pd.DataFrame(
        [{"asset": name, "path": str(path), "exists": path.exists()} for name, path in sorted(written.items())]
    )
    manifest_path = output_dir / "run_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    written["run_manifest"] = manifest_path

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        logger.warning("Could not resolve git SHA for software_versions.json; leaving blank")
        git_sha = ""
    versions = {
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "numpyVersion": getattr(np, "__version__", ""),
        "pandasVersion": getattr(pd, "__version__", ""),
        "gitSha": git_sha,
        "bootstrapSeedDefault": 20260623,
        "looActivityCapSeed": int(config.get("cohorts", {}).get("loo_activity_cap_seed", 20260721) or 20260721),
        "configPath": str(result.metadata.get("configPath", "")),
    }
    versions_path = output_dir / "software_versions.json"
    versions_path.write_text(json.dumps(versions, indent=2) + "\n")
    written["software_versions"] = versions_path

    if bool(config.get("outputs", {}).get("write_html", True)):
        html_path = output_dir / str(config.get("outputs", {}).get("html_filename", "trail_digital_twin_report.html"))
        html_path.write_text(render_html_report(result, written))
        written["html_report"] = html_path
    return written
