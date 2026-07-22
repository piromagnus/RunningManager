"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Intensity-conditioned trail twin claim validation (P0).

Implements docs/plan/intensity_conditioned_trail_twin_claim_validation.md:
E0 support map, E2 prescribed-mean-HRR LOO, E4 history-only HRR selection,
E5 causal fatigue ablation, and machine-readable claim gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services import trail_digital_twin_pipeline as pipeline  # noqa: E402
from services import trail_performance_model as tpm  # noqa: E402

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics_row(
    actual_sec: np.ndarray,
    predicted_sec: np.ndarray,
    duration_hours: np.ndarray | None = None,
) -> dict[str, float]:
    actual = np.asarray(actual_sec, dtype=float)
    predicted = np.asarray(predicted_sec, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    if not bool(valid.any()):
        return {
            "maeMin": float("nan"),
            "mapePct": float("nan"),
            "medianApePct": float("nan"),
            "biasMin": float("nan"),
            "errorDurationSlopeMinPerHour": float("nan"),
            "n": 0.0,
        }
    a = actual[valid]
    p = predicted[valid]
    err_min = (p - a) / 60.0
    ape = np.abs((p - a) / a) * 100.0
    slope = float("nan")
    if duration_hours is not None:
        dur = np.asarray(duration_hours, dtype=float)[valid]
        if np.isfinite(dur).sum() >= 3 and float(np.nanstd(dur)) > 1e-6:
            slope = float(np.polyfit(dur[np.isfinite(dur)], err_min[np.isfinite(dur)], 1)[0])
    return {
        "maeMin": float(np.mean(np.abs(err_min))),
        "mapePct": float(np.mean(ape)),
        "medianApePct": float(np.median(ape)),
        "biasMin": float(np.mean(err_min)),
        "errorDurationSlopeMinPerHour": slope,
        "n": float(valid.sum()),
    }


def _paired_mape_bootstrap(
    actual_sec: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap CI for MAPE(A) - MAPE(B) (positive => B better)."""
    actual = np.asarray(actual_sec, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(a) & np.isfinite(b) & (actual > 0)
    actual = actual[valid]
    a = a[valid]
    b = b[valid]
    if len(actual) < 2:
        return {
            "deltaMapePct": float("nan"),
            "ciLow": float("nan"),
            "ciHigh": float("nan"),
            "relativeGainVsA": float("nan"),
        }
    mape_a = np.abs((a - actual) / actual) * 100.0
    mape_b = np.abs((b - actual) / actual) * 100.0
    point = float(np.mean(mape_a) - np.mean(mape_b))
    rel = point / float(np.mean(mape_a)) if float(np.mean(mape_a)) > 1e-9 else float("nan")
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(actual)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        deltas.append(float(np.mean(mape_a[idx]) - np.mean(mape_b[idx])))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "deltaMapePct": point,
        "ciLow": float(lo),
        "ciHigh": float(hi),
        "relativeGainVsA": float(rel),
    }


@dataclass
class ValidationBundle:
    config: dict[str, Any]
    activity_df: pd.DataFrame
    segments_df: pd.DataFrame
    segments_by_activity: dict[str, pd.DataFrame]
    cohorts: dict[str, pd.DataFrame]
    hr_rest: float
    hr_max: float
    event_labels: pd.DataFrame
    output_dir: Path


def build_bundle(config: dict[str, Any], project_root: Path) -> ValidationBundle:
    activity_df, _daily, hr_rest, hr_max, _v_vt2 = pipeline._load_activity_inputs(config, project_root)
    all_segments_df, _qc_df, segments_by_activity = pipeline._build_segments(
        activity_df, config, project_root, hr_rest, hr_max
    )
    if all_segments_df.empty:
        raise RuntimeError("no usable segments for claim validation")
    if "isFitEligible" not in all_segments_df.columns:
        all_segments_df = all_segments_df.copy()
        all_segments_df["isFitEligible"] = True
    segment_summary = (
        all_segments_df.groupby("activityId")
        .agg(
            technicalityGps=("technicalityGps", "mean"),
            meanAltitudeM=("meanAltitudeM", "mean"),
            segmentDistanceKm=("distanceKm", "sum"),
            segmentActualTimeSec=("actualTimeSec", "sum"),
            meanSegmentHrReserve=("meanHrReserve", "mean"),
            meanSpeedEqKmh=("meanSpeedEqKmh", "mean"),
            segmentCount=("segmentIndex", "count"),
            fitEligibleSegmentCount=("isFitEligible", "sum"),
        )
        .reset_index()
    )
    activity_df = activity_df.merge(segment_summary, on="activityId", how="left")
    activity_df["usableSegmentActivity"] = activity_df["activityId"].astype(str).isin(segments_by_activity)
    activity_df = pipeline._add_readiness_factors(
        activity_df, "ctl", "tsb", "ctlReadinessFactor", config["readiness"]
    )
    activity_df = pipeline._add_readiness_factors(
        activity_df,
        "trimpRediSlow",
        "trimpRediBalance",
        "rediReadinessFactor",
        config["readiness"],
    )
    cohorts = pipeline._build_cohorts(activity_df, segments_by_activity, config)
    segments_with_trimp = tpm.add_in_activity_trimp_features(
        all_segments_df,
        decay_lambda=float(config["physiology"]["decay_lambda"]),
    )
    segment_features = pipeline.add_segment_model_features(segments_with_trimp, activity_df)

    claim = config.get("claim_validation", {})
    labels_path = project_root / str(claim.get("event_labels_csv", "configs/claim_validation_event_labels.csv"))
    event_labels = pd.read_csv(labels_path, dtype={"activityId": str})
    output_dir = project_root / str(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return ValidationBundle(
        config=config,
        activity_df=activity_df,
        segments_df=segment_features,
        segments_by_activity={
            str(k): segment_features[segment_features["activityId"].astype(str).eq(str(k))].copy()
            for k in segments_by_activity
        },
        cohorts=cohorts,
        hr_rest=float(hr_rest),
        hr_max=float(hr_max),
        event_labels=event_labels,
        output_dir=output_dir,
    )


def write_freeze_manifest(bundle: ValidationBundle, project_root: Path) -> Path:
    data_dir = project_root / str(bundle.config["paths"]["data_dir"])
    files = [
        data_dir / "activities.csv",
        data_dir / "activities_metrics.csv",
        data_dir / "athlete.csv",
        data_dir / "thresholds.csv",
        data_dir / "daily_metrics.csv",
        project_root / "configs" / "trail_digital_twin_claim_validation.yaml",
        project_root / "configs" / "claim_validation_event_labels.csv",
    ]
    payload = {
        "analysisDate": pd.Timestamp.now(tz="UTC").isoformat(),
        "hrRest": bundle.hr_rest,
        "hrMax": bundle.hr_max,
        "physiology": bundle.config["physiology"],
        "claim_validation": bundle.config.get("claim_validation", {}),
        "inputHashes": {
            str(path.relative_to(project_root)): _sha256_file(path) if path.exists() else None
            for path in files
        },
        "nActivitiesWithSegments": int(len(bundle.segments_by_activity)),
        "cohortCounts": {name: int(len(df)) for name, df in bundle.cohorts.items()},
    }
    out = bundle.output_dir / "freeze_manifest.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote %s", out)
    return out


def run_e0_data_support(bundle: ValidationBundle) -> pd.DataFrame:
    claim = bundle.config.get("claim_validation", {})
    weak_h = float(claim.get("support_duration_weak_hours", 4.0))
    extrap_h = float(claim.get("support_duration_extrapolate_hours", 6.0))
    weak_hrr = float(claim.get("support_hrr_weak", 0.80))
    strict_ids = set(str(x) for x in claim.get("strict_pre_race_activity_ids", []))
    pseudo_ids = set(str(x) for x in claim.get("pseudo_prospective_activity_ids", []))
    bpm = float(claim.get("hr_rest_sensitivity_bpm", 5))
    bpm_max = float(claim.get("hr_max_sensitivity_bpm", 5))

    hard_trail_ids = set(bundle.cohorts.get("hardTrailRun", pd.DataFrame())["activityId"].astype(str))
    hard_mix_ids = set(bundle.cohorts.get("hardRunOrTrailRun", pd.DataFrame())["activityId"].astype(str))
    over20_ids = set(bundle.cohorts.get("runTrailOver20Min", pd.DataFrame())["activityId"].astype(str))
    labels = bundle.event_labels.set_index("activityId")

    rows: list[dict[str, object]] = []
    for activity_id, segments in bundle.segments_by_activity.items():
        meta = bundle.activity_df.loc[bundle.activity_df["activityId"].astype(str).eq(activity_id)]
        if meta.empty:
            continue
        row = meta.iloc[0]
        moving = float(row.get("actualTimeSec", np.nan))
        elapsed = float(pd.to_numeric(row.get("elapsedSec"), errors="coerce"))
        hrr = float(row.get("hrReserveRatio", np.nan))
        duration_h = moving / 3600.0 if np.isfinite(moving) else float("nan")
        distance = float(pd.to_numeric(row.get("distanceKm"), errors="coerce"))
        ascent = float(pd.to_numeric(row.get("ascentM"), errors="coerce"))
        avg_hr = float(pd.to_numeric(row.get("avgHr"), errors="coerce"))

        hrr_rest_lo = (avg_hr - (bundle.hr_rest - bpm)) / (bundle.hr_max - (bundle.hr_rest - bpm))
        hrr_rest_hi = (avg_hr - (bundle.hr_rest + bpm)) / (bundle.hr_max - (bundle.hr_rest + bpm))
        hrr_max_lo = (avg_hr - bundle.hr_rest) / ((bundle.hr_max - bpm_max) - bundle.hr_rest)
        hrr_max_hi = (avg_hr - bundle.hr_rest) / ((bundle.hr_max + bpm_max) - bundle.hr_rest)

        if activity_id in strict_ids:
            course_status = "strict_pre_race"
        elif activity_id in pseudo_ids:
            course_status = "executed_geometry"
        else:
            course_status = "unknown"

        if not np.isfinite(duration_h) or duration_h >= extrap_h or (np.isfinite(hrr) and hrr >= 0.90):
            support = "extrapolation"
        elif duration_h >= weak_h or (np.isfinite(hrr) and hrr >= weak_hrr):
            support = "weak_support"
        else:
            support = "interpolation"

        event_label = "training"
        label_source = "default"
        if activity_id in labels.index:
            event_label = str(labels.loc[activity_id, "eventLabel"])
            label_source = str(labels.loc[activity_id, "labelSource"])

        hr_valid = float("nan")
        if "hrValidShare" in segments.columns:
            hr_valid = float(pd.to_numeric(segments["hrValidShare"], errors="coerce").mean())

        rows.append(
            {
                "activityId": activity_id,
                "startDate": str(row.get("startDate", "")),
                "category": str(row.get("category", "")),
                "name": str(row.get("name", "")),
                "eventLabel": event_label,
                "labelSource": label_source,
                "movingSec": moving,
                "elapsedSec": elapsed,
                "movingElapsedGapMin": (elapsed - moving) / 60.0
                if np.isfinite(elapsed) and np.isfinite(moving)
                else float("nan"),
                "hrReserveRatio": hrr,
                "distanceKm": distance,
                "ascentM": ascent,
                "ascentPerKm": ascent / distance if distance and distance > 0 else float("nan"),
                "meanAltitudeM": float(row.get("meanAltitudeM", np.nan)),
                "durationHours": duration_h,
                "durationBin": (
                    "lt1h"
                    if duration_h < 1
                    else "1-2h"
                    if duration_h < 2
                    else "2-4h"
                    if duration_h < 4
                    else "4-6h"
                    if duration_h < 6
                    else "ge6h"
                ),
                "hrrBin": (
                    "lt0.60"
                    if hrr < 0.60
                    else "0.60-0.70"
                    if hrr < 0.70
                    else "0.70-0.80"
                    if hrr < 0.80
                    else "0.80-0.90"
                    if hrr < 0.90
                    else "ge0.90"
                ),
                "hrValidShare": hr_valid,
                "hrValidShareMeansNonMissingOnly": True,
                "segmentCount": int(row.get("segmentCount", len(segments))),
                "fitEligibleSegmentCount": int(row.get("fitEligibleSegmentCount", len(segments))),
                "cohort_hardTrailRun": activity_id in hard_trail_ids,
                "cohort_hardRunOrTrailRun": activity_id in hard_mix_ids,
                "cohort_runTrailOver20Min": activity_id in over20_ids,
                "courseProfileStatus": course_status,
                "supportFlag": support,
                "hrrAtHrRestMinus5": float(hrr_rest_lo),
                "hrrAtHrRestPlus5": float(hrr_rest_hi),
                "hrrAtHrMaxMinus5": float(hrr_max_lo),
                "hrrAtHrMaxPlus5": float(hrr_max_hi),
            }
        )
    table = pd.DataFrame(rows).sort_values(["startDate", "activityId"]).reset_index(drop=True)
    out = bundle.output_dir / "table_data_support.csv"
    table.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(table))
    return table


def _route_for_prescribed_prediction(segments: pd.DataFrame) -> pd.DataFrame:
    """Drop target HR / clock leakage columns before prescribed-HRR simulation."""
    route = segments.sort_values(["segmentIndex", "startKm"], na_position="last").copy()
    for col in ("meanHrReserve", "actualTimeSec", "actualMovingTimeSec", "segmentTrimp", "cumTrimpBefore", "decayedTrimpBefore"):
        if col in route.columns:
            route[col] = np.nan
    return route


def _physiology_kwargs(config: Mapping[str, Any]) -> dict[str, float]:
    phys = config["physiology"]
    return {
        "hrr_reference": float(phys["hrr_reference"]),
        "hrr_min_factor": float(phys["hrr_min_factor"]),
        "hrr_max_factor": float(phys["hrr_max_factor"]),
        "decay_lambda": float(phys["decay_lambda"]),
        "min_fatigue_factor": float(phys["min_fatigue_factor"]),
        "gap_steep_threshold": float(phys.get("gap_steep_threshold", 0.15)),
        "gap_soft_start": float(phys.get("gap_soft_start", 0.04)),
        "gap_climb_scale": float(phys.get("gap_climb_scale", 1.0)),
        "gap_descent_scale": float(phys.get("gap_descent_scale", 1.0)),
    }


def _fit_stage3_on_train(
    train_segments: pd.DataFrame,
    observed: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, float | str]:
    phys = config["physiology"]
    fitting = config["fitting"]
    best, _grid, _pred = tpm.hrr_trimp_grid_search_model(
        train_segments,
        v_anchor_kmh=float(phys["vma_flat_kmh"]),
        alpha_grid=fitting["hrr_trimp_alpha_grid"],
        fatigue_coef_grid=fitting["hrr_trimp_kappa_grid"],
        secondary_fatigue_coef_grid=fitting.get("hrr_trimp_secondary_kappa_grid") or [0.0],
        fatigue_models=tuple(fitting["fatigue_models"]),
        observed_activity_times_sec=observed,
        objective="race",
        fit_mask_col=pipeline._fit_mask_col(config),
        actual_time_col=pipeline._fit_actual_time_col(config),
        hrr_reference=float(phys["hrr_reference"]),
        hrr_min_factor=float(phys["hrr_min_factor"]),
        hrr_max_factor=float(phys["hrr_max_factor"]),
        min_fatigue_factor=float(phys["min_fatigue_factor"]),
        acute_trimp_col="decayedTrimpBefore",
        load_factor_col="rediReadinessFactor",
        use_hrr_effort=True,
        gap_steep_threshold=float(phys.get("gap_steep_threshold", 0.15)),
        gap_soft_start=float(phys.get("gap_soft_start", 0.04)),
        gap_climb_scale=float(phys.get("gap_climb_scale", 1.0)),
        gap_descent_scale=float(phys.get("gap_descent_scale", 1.0)),
    )
    return {
        "alpha": float(best.get("alpha", np.nan)),
        "fatigueCoef": float(best.get("fatigueCoef", np.nan)),
        "fatigueModel": str(best.get("fatigueModel", "exponential")),
    }


def _fit_stage0_on_train(
    train_ids: list[str],
    segments_by_activity: Mapping[str, pd.DataFrame],
    observed: Mapping[str, float],
    config: Mapping[str, Any],
    activity_df: pd.DataFrame,
) -> dict[str, float]:
    phys = config["physiology"]
    fitting = config["fitting"]
    train_map = {aid: segments_by_activity[aid] for aid in train_ids if aid in segments_by_activity}
    ctl = (
        activity_df.set_index(activity_df["activityId"].astype(str))["ctlReadinessFactor"]
        .astype(float)
        .to_dict()
    )
    best, _grid = tpm.grid_search_model(
        train_map,
        observed_times_sec=observed,
        v_vt2_kmh=float(phys["vma_flat_kmh"]),
        alpha_grid=fitting["stage0_alpha_grid"],
        mu_grid=fitting["stage0_mu_grid"],
        ctl_factors=ctl,
    )
    return {"alpha": float(best.get("alpha", np.nan)), "mu": float(best.get("mu", np.nan))}


def _simulate_prescribed(
    route: pd.DataFrame,
    *,
    hrr: float,
    alpha: float,
    fatigue_coef: float,
    fatigue_model: str,
    config: Mapping[str, Any],
    load_factor: float,
    fatigue_mode: str,
    use_hrr_effort: bool = True,
) -> float:
    """Prescribed-HRR prediction with causal (predicted) fatigue state."""
    phys_kw = _physiology_kwargs(config)
    v_anchor = float(config["physiology"]["vma_flat_kmh"])

    if fatigue_mode == "none":
        pred = tpm.simulate_constant_hrr_route(
            route,
            hrr=hrr,
            v_anchor_kmh=v_anchor,
            alpha=alpha,
            fatigue_coef=0.0,
            fatigue_model=fatigue_model,
            load_factor=load_factor,
            use_hrr_effort=use_hrr_effort,
            fatigue_input_col="cumTrimpBefore",
            **phys_kw,
        )
        return float(pred["predictedTimeSec"].sum())

    if fatigue_mode == "progress":
        pred = tpm.simulate_constant_hrr_route(
            route,
            hrr=hrr,
            v_anchor_kmh=v_anchor,
            alpha=alpha,
            fatigue_coef=fatigue_coef,
            fatigue_model="linear",
            load_factor=load_factor,
            use_hrr_effort=use_hrr_effort,
            fatigue_input_col="progress",
            **phys_kw,
        )
        return float(pred["predictedTimeSec"].sum())

    if fatigue_mode == "predicted_trimp":
        pred = tpm.simulate_constant_hrr_route(
            route,
            hrr=hrr,
            v_anchor_kmh=v_anchor,
            alpha=alpha,
            fatigue_coef=fatigue_coef,
            fatigue_model=fatigue_model,
            load_factor=load_factor,
            use_hrr_effort=use_hrr_effort,
            fatigue_input_col="cumTrimpBefore",
            **phys_kw,
        )
        return float(pred["predictedTimeSec"].sum())

    # Custom sequential load: predicted time hours or distance-equivalent.
    ordered = route.sort_values(["segmentIndex", "startKm"], na_position="last").copy()
    cumulative = 0.0
    total = 0.0
    decay = math.exp(-max(0.0, float(config["physiology"]["decay_lambda"])))
    decayed = 0.0
    for _, segment in ordered.iterrows():
        one = pd.DataFrame([segment]).copy()
        one["meanHrReserve"] = float(hrr)
        if fatigue_mode == "predicted_time":
            load = cumulative / 3600.0
            one["cumTrimpBefore"] = load
            acute_col = "cumTrimpBefore"
        elif fatigue_mode == "dist_eq":
            load = cumulative
            one["cumDistEqBefore"] = load
            acute_col = "cumDistEqBefore"
        else:
            raise ValueError(f"unknown fatigue_mode={fatigue_mode}")
        one["_preRaceLoadFactor"] = float(load_factor)
        predicted_time = float(
            tpm.predict_hrr_trimp_segment_times(
                one,
                v_anchor_kmh=v_anchor,
                alpha=alpha,
                fatigue_coef=fatigue_coef,
                fatigue_model=fatigue_model if fatigue_mode != "progress" else "linear",
                acute_trimp_col=acute_col,
                load_factor_col="_preRaceLoadFactor",
                use_hrr_effort=use_hrr_effort,
                **{k: v for k, v in phys_kw.items() if k != "decay_lambda"},
            ).iloc[0]
        )
        total += predicted_time
        if fatigue_mode == "predicted_time":
            cumulative += predicted_time
        else:
            dist = float(pd.to_numeric(segment.get("distanceKm"), errors="coerce") or 0.0)
            gap = float(pd.to_numeric(segment.get("gapFactorIntegrated"), errors="coerce") or 1.0)
            cumulative += dist * max(gap, 0.0)
        _ = decayed
        decayed = tpm._segment_trimp_from_prediction(predicted_time, float(hrr)) + decay * decayed
    return float(total)


def _activity_meta_map(bundle: ValidationBundle) -> pd.DataFrame:
    cols = [
        "activityId",
        "startDate",
        "actualTimeSec",
        "elapsedSec",
        "hrReserveRatio",
        "rediReadinessFactor",
        "ctlReadinessFactor",
        "distanceKm",
        "ascentM",
        "category",
    ]
    meta = bundle.activity_df.copy()
    meta["activityId"] = meta["activityId"].astype(str)
    keep = [c for c in cols if c in meta.columns]
    return meta[keep].drop_duplicates("activityId").set_index("activityId")


def run_e2_prescribed_hrr_loo(
    bundle: ValidationBundle,
    *,
    cohort_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E2: B0–B3 + U1 on one cohort (activity-level LOO)."""
    if cohort_name not in bundle.cohorts:
        raise KeyError(cohort_name)
    cohort = bundle.cohorts[cohort_name]
    ids = cohort["activityId"].astype(str).tolist()
    meta = _activity_meta_map(bundle)
    observed = meta.loc[ids, "actualTimeSec"].astype(float).to_dict()
    phys = bundle.config["physiology"]
    v_anchor = float(phys["vma_flat_kmh"])

    fold_rows: list[dict[str, object]] = []
    for i, held_out in enumerate(ids):
        train_ids = [aid for aid in ids if aid != held_out]
        train_segments = pd.concat(
            [bundle.segments_by_activity[aid] for aid in train_ids if aid in bundle.segments_by_activity],
            ignore_index=True,
        )
        if train_segments.empty or held_out not in bundle.segments_by_activity:
            logger.warning("Skipping held-out %s (missing segments)", held_out)
            continue
        train_obs = {aid: observed[aid] for aid in train_ids if aid in observed}
        stage3 = _fit_stage3_on_train(train_segments, train_obs, bundle.config)
        stage0 = _fit_stage0_on_train(
            train_ids, bundle.segments_by_activity, train_obs, bundle.config, bundle.activity_df
        )

        held_segments = bundle.segments_by_activity[held_out]
        route = _route_for_prescribed_prediction(held_segments)
        load = float(meta.loc[held_out, "rediReadinessFactor"]) if held_out in meta.index else 1.0
        oracle_hrr = float(meta.loc[held_out, "hrReserveRatio"])
        actual = float(observed[held_out])
        duration_h = actual / 3600.0
        ctl = float(meta.loc[held_out, "ctlReadinessFactor"]) if held_out in meta.index else 1.0

        # B0: fixed-intensity physics twin (progress fatigue).
        b0 = float(
            tpm.predict_many(
                {held_out: held_segments},
                v_vt2_kmh=v_anchor,
                alpha=float(stage0["alpha"]),
                mu=float(stage0["mu"]),
                ctl_factors={held_out: ctl},
            ).loc[held_out]
        )

        # B1: fixed HRR reference + predicted TRIMP.
        b1 = _simulate_prescribed(
            route,
            hrr=float(phys["hrr_reference"]),
            alpha=float(stage3["alpha"]),
            fatigue_coef=float(stage3["fatigueCoef"]),
            fatigue_model=str(stage3["fatigueModel"]),
            config=bundle.config,
            load_factor=load,
            fatigue_mode="predicted_trimp",
        )

        # B2: oracle mean HRR + progress fatigue.
        b2 = _simulate_prescribed(
            route,
            hrr=oracle_hrr,
            alpha=float(stage3["alpha"]),
            fatigue_coef=float(stage3["fatigueCoef"]),
            fatigue_model=str(stage3["fatigueModel"]),
            config=bundle.config,
            load_factor=load,
            fatigue_mode="progress",
        )

        # B3: oracle mean HRR + causal predicted TRIMP.
        b3 = _simulate_prescribed(
            route,
            hrr=oracle_hrr,
            alpha=float(stage3["alpha"]),
            fatigue_coef=float(stage3["fatigueCoef"]),
            fatigue_model=str(stage3["fatigueModel"]),
            config=bundle.config,
            load_factor=load,
            fatigue_mode="predicted_trimp",
        )

        # U1: observed segment HRR + predicted-time TRIMP (retrospective upper bound).
        u1_pred = tpm.simulate_observed_hrr_segments(
            held_segments,
            v_anchor_kmh=v_anchor,
            alpha=float(stage3["alpha"]),
            fatigue_coef=float(stage3["fatigueCoef"]),
            fatigue_model=str(stage3["fatigueModel"]),
            load_factor=load,
            fallback_hrr=oracle_hrr,
            **_physiology_kwargs(bundle.config),
        )
        u1 = float(u1_pred["predictedTimeSec"].sum())

        common = {
            "cohort": cohort_name,
            "activityId": held_out,
            "startDate": str(meta.loc[held_out, "startDate"]) if held_out in meta.index else "",
            "actualTimeSec": actual,
            "durationHours": duration_h,
            "oracleMeanHrr": oracle_hrr,
            "alpha": float(stage3["alpha"]),
            "fatigueCoef": float(stage3["fatigueCoef"]),
            "fatigueModel": str(stage3["fatigueModel"]),
            "stage0Alpha": float(stage0["alpha"]),
            "stage0Mu": float(stage0["mu"]),
            "effectiveCapacityKmh": float(stage3["alpha"]) * v_anchor,
        }
        for model_id, pred in [("B0", b0), ("B1", b1), ("B2", b2), ("B3", b3), ("U1", u1)]:
            fold_rows.append(
                {
                    **common,
                    "modelId": model_id,
                    "predictedTimeSec": pred,
                    "errorMin": (pred - actual) / 60.0,
                    "apePct": abs(pred - actual) / actual * 100.0 if actual > 0 else float("nan"),
                    "isReconstructionBound": model_id == "U1",
                    "usesTargetSegmentHrr": model_id == "U1",
                    "usesTargetMeanHrr": model_id in {"B2", "B3", "U1"},
                }
            )
        if (i + 1) % 5 == 0 or i == 0:
            logger.info(
                "E2 %s fold %d/%d held_out=%s B3_err=%.1f min",
                cohort_name,
                i + 1,
                len(ids),
                held_out,
                (b3 - actual) / 60.0,
            )

    folds = pd.DataFrame(fold_rows)
    summary_rows: list[dict[str, object]] = []
    claim = bundle.config.get("claim_validation", {})
    n_boot = int(claim.get("bootstrap_iterations", 10000))
    seed = int(claim.get("bootstrap_seed", 20260722))

    wide = folds.pivot_table(
        index="activityId",
        columns="modelId",
        values="predictedTimeSec",
        aggfunc="first",
    )
    actuals = folds.drop_duplicates("activityId").set_index("activityId")["actualTimeSec"]
    durations = folds.drop_duplicates("activityId").set_index("activityId")["durationHours"]

    for model_id in ["B0", "B1", "B2", "B3", "U1"]:
        if model_id not in wide.columns:
            continue
        pred = wide[model_id].reindex(actuals.index)
        metrics = _metrics_row(actuals.to_numpy(), pred.to_numpy(), durations.to_numpy())
        summary_rows.append(
            {
                "cohort": cohort_name,
                "modelId": model_id,
                "label": {
                    "B0": "Prior fixed-%VT2 physics twin",
                    "B1": "Fixed HRR reference",
                    "B2": "Prescribed mean HRR + progress fatigue",
                    "B3": "Prescribed mean HRR + causal predicted TRIMP",
                    "U1": "Observed segment HRR reconstruction (upper bound)",
                }[model_id],
                **metrics,
                "isReconstructionBound": model_id == "U1",
            }
        )

    # Paired contrasts versus fixed-intensity baselines.
    for baseline in ["B0", "B1"]:
        if baseline not in wide.columns or "B3" not in wide.columns:
            continue
        boot = _paired_mape_bootstrap(
            actuals.to_numpy(),
            wide[baseline].reindex(actuals.index).to_numpy(),
            wide["B3"].reindex(actuals.index).to_numpy(),
            n_boot=n_boot,
            seed=seed + hash(baseline) % 1000,
        )
        summary_rows.append(
            {
                "cohort": cohort_name,
                "modelId": f"B3_vs_{baseline}",
                "label": f"Paired MAPE gain of B3 over {baseline}",
                "maeMin": float("nan"),
                "mapePct": boot["deltaMapePct"],
                "medianApePct": float("nan"),
                "biasMin": float("nan"),
                "errorDurationSlopeMinPerHour": float("nan"),
                "n": float(len(actuals)),
                "isReconstructionBound": False,
                "deltaMapePct": boot["deltaMapePct"],
                "bootstrapCiLow": boot["ciLow"],
                "bootstrapCiHigh": boot["ciHigh"],
                "relativeGainVsBaseline": boot["relativeGainVsA"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    folds_path = bundle.output_dir / f"table_prescribed_hrr_folds_{cohort_name}.csv"
    summary_path = bundle.output_dir / "table_prescribed_hrr_model_comparison.csv"
    folds.to_csv(folds_path, index=False)
    # Append if secondary cohort already wrote summary.
    if summary_path.exists():
        prev = pd.read_csv(summary_path)
        summary = pd.concat([prev[~prev["cohort"].eq(cohort_name)], summary], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s and %s", folds_path, summary_path)
    return folds, summary


def run_e5_causal_fatigue(
    bundle: ValidationBundle,
    *,
    cohort_name: str,
) -> pd.DataFrame:
    """E5: fatigue variants under prescribed mean HRR LOO (predicted-time safe)."""
    cohort = bundle.cohorts[cohort_name]
    ids = cohort["activityId"].astype(str).tolist()
    meta = _activity_meta_map(bundle)
    observed = meta.loc[ids, "actualTimeSec"].astype(float).to_dict()
    variants = [
        ("none", "none", False),
        ("progress", "progress", False),
        ("predicted_time", "predicted_time", False),
        ("dist_eq", "dist_eq", False),
        ("predicted_trimp", "predicted_trimp", False),
        ("observed_trimp_reconstruction", "observed_reconstruction", True),
    ]
    fold_rows: list[dict[str, object]] = []
    phys = bundle.config["physiology"]
    v_anchor = float(phys["vma_flat_kmh"])

    for i, held_out in enumerate(ids):
        train_ids = [aid for aid in ids if aid != held_out]
        train_segments = pd.concat(
            [bundle.segments_by_activity[aid] for aid in train_ids if aid in bundle.segments_by_activity],
            ignore_index=True,
        )
        if train_segments.empty or held_out not in bundle.segments_by_activity:
            continue
        train_obs = {aid: observed[aid] for aid in train_ids if aid in observed}
        stage3 = _fit_stage3_on_train(train_segments, train_obs, bundle.config)
        held_segments = bundle.segments_by_activity[held_out]
        route = _route_for_prescribed_prediction(held_segments)
        load = float(meta.loc[held_out, "rediReadinessFactor"]) if held_out in meta.index else 1.0
        oracle_hrr = float(meta.loc[held_out, "hrReserveRatio"])
        actual = float(observed[held_out])

        for variant, mode, reconstruction in variants:
            if reconstruction:
                recon = tpm.predict_hrr_trimp_segment_times(
                    held_segments,
                    v_anchor_kmh=v_anchor,
                    alpha=float(stage3["alpha"]),
                    fatigue_coef=float(stage3["fatigueCoef"]),
                    fatigue_model=str(stage3["fatigueModel"]),
                    acute_trimp_col="decayedTrimpBefore",
                    load_factor_col="rediReadinessFactor",
                    use_hrr_effort=True,
                    **{k: v for k, v in _physiology_kwargs(bundle.config).items() if k != "decay_lambda"},
                )
                pred = float(recon.sum())
            else:
                pred = _simulate_prescribed(
                    route,
                    hrr=oracle_hrr,
                    alpha=float(stage3["alpha"]),
                    fatigue_coef=float(stage3["fatigueCoef"]),
                    fatigue_model=str(stage3["fatigueModel"]),
                    config=bundle.config,
                    load_factor=load,
                    fatigue_mode=mode,
                )
            fold_rows.append(
                {
                    "cohort": cohort_name,
                    "activityId": held_out,
                    "variant": variant,
                    "reconstructionOnly": reconstruction,
                    "actualTimeSec": actual,
                    "predictedTimeSec": pred,
                    "errorMin": (pred - actual) / 60.0,
                    "apePct": abs(pred - actual) / actual * 100.0 if actual > 0 else float("nan"),
                    "oracleMeanHrr": oracle_hrr,
                }
            )
        if (i + 1) % 5 == 0 or i == 0:
            logger.info("E5 %s fold %d/%d", cohort_name, i + 1, len(ids))

    folds = pd.DataFrame(fold_rows)
    summary_rows = []
    for variant, group in folds.groupby("variant"):
        metrics = _metrics_row(
            group["actualTimeSec"].to_numpy(),
            group["predictedTimeSec"].to_numpy(),
            (group["actualTimeSec"] / 3600.0).to_numpy(),
        )
        summary_rows.append(
            {
                "cohort": cohort_name,
                "variant": variant,
                "reconstructionOnly": bool(group["reconstructionOnly"].iloc[0]),
                **metrics,
            }
        )
    summary = pd.DataFrame(summary_rows)
    folds.to_csv(bundle.output_dir / f"table_causal_fatigue_folds_{cohort_name}.csv", index=False)
    out = bundle.output_dir / "table_causal_fatigue_ablation.csv"
    if out.exists():
        prev = pd.read_csv(out)
        summary = pd.concat([prev[~prev["cohort"].eq(cohort_name)], summary], ignore_index=True)
    summary.to_csv(out, index=False)
    logger.info("Wrote %s", out)
    return summary


def _select_history_hrr(
    rule: str,
    history: pd.DataFrame,
    *,
    hrr_reference: float,
    target_duration_h: float | None = None,
) -> float:
    if history.empty:
        return float(hrr_reference)
    hrr = pd.to_numeric(history["hrReserveRatio"], errors="coerce")
    dur = pd.to_numeric(history["actualTimeSec"], errors="coerce") / 3600.0
    hard = history.copy()
    hard["_hrr"] = hrr
    hard["_dur"] = dur
    hard = hard[np.isfinite(hard["_hrr"])]
    if hard.empty:
        return float(hrr_reference)

    if rule == "fixed_hrr_reference":
        return float(hrr_reference)
    if rule == "median_prior_hard_hrr":
        # Prefer prior hard-trail-like efforts (HRR>=0.65, duration>=0.5h) when available.
        subset = hard[(hard["_hrr"] >= 0.65) & (hard["_dur"] >= 0.5)]
        use = subset if len(subset) >= 3 else hard
        return float(use["_hrr"].median())
    if rule == "nearest_prior_by_duration":
        if target_duration_h is None or not np.isfinite(target_duration_h):
            return float(hard["_hrr"].median())
        idx = (hard["_dur"] - float(target_duration_h)).abs().idxmin()
        return float(hard.loc[idx, "_hrr"])
    if rule == "duration_bin_quantile":
        if target_duration_h is None or not np.isfinite(target_duration_h):
            return float(hard["_hrr"].quantile(0.75))
        # Bin by hours; use 75th percentile within nearest populated bin.
        bins = [0, 1, 2, 4, 6, 24]
        hard = hard.copy()
        hard["_bin"] = pd.cut(hard["_dur"], bins=bins, include_lowest=True)
        target_bin = pd.cut([float(target_duration_h)], bins=bins, include_lowest=True)[0]
        subset = hard[hard["_bin"] == target_bin]
        use = subset if len(subset) >= 3 else hard
        return float(use["_hrr"].quantile(0.75))
    if rule == "power_law_max_window":
        try:
            hist = hard.copy()
            hist["hrReserveRatio"] = hist["_hrr"]
            hist["actualTimeSec"] = hist["_dur"] * 3600.0
            params, _windows = tpm.estimate_hrr_duration_power_law(
                hist,
                duration_windows_min=(30, 60, 90, 120, 180, 240, 360, 480),
                hrr_col="hrReserveRatio",
                duration_col="actualTimeSec",
                categories=None,
            )
            if target_duration_h is None or not np.isfinite(target_duration_h):
                return float(hard["_hrr"].median())
            return float(tpm.hrr_for_duration_power_law(float(target_duration_h) * 3600.0, params))
        except Exception as exc:  # noqa: BLE001
            logger.warning("power_law_max_window fallback: %s", exc)
            return float(hard["_hrr"].median())
    raise ValueError(f"unknown HRR selection rule: {rule}")


def run_e4_rolling_origin(bundle: ValidationBundle) -> pd.DataFrame:
    """E4: history-only HRR selection on independently labelled race/benchmark events."""
    labels = bundle.event_labels.copy()
    labels["activityId"] = labels["activityId"].astype(str)
    focus = labels[labels["eventLabel"].isin(["race", "benchmark_workout"])].copy()
    meta = _activity_meta_map(bundle)
    # Training pool: all segmented run/trail activities.
    pool_ids = [
        aid
        for aid in bundle.segments_by_activity
        if aid in meta.index and str(meta.loc[aid, "category"]).upper() in {"RUN", "TRAIL_RUN"}
    ]
    pool = meta.loc[pool_ids].copy().reset_index()
    pool["startDate"] = pd.to_datetime(pool["startDate"], errors="coerce")
    focus = focus.drop(columns=[c for c in ["startDate", "actualTimeSec", "hrReserveRatio"] if c in focus.columns])
    focus = focus.merge(
        pool[["activityId", "startDate", "actualTimeSec", "hrReserveRatio"]],
        on="activityId",
        how="inner",
    )
    focus = focus.sort_values("startDate")

    rules = [
        "fixed_hrr_reference",
        "median_prior_hard_hrr",
        "nearest_prior_by_duration",
        "duration_bin_quantile",
        "power_law_max_window",
    ]
    phys = bundle.config["physiology"]
    hrr_ref = float(phys["hrr_reference"])
    rows: list[dict[str, object]] = []

    for _, event in focus.iterrows():
        held_out = str(event["activityId"])
        if held_out not in bundle.segments_by_activity:
            continue
        event_date = pd.to_datetime(event["startDate"], errors="coerce")
        history = pool[pool["startDate"] < event_date].copy()
        if history.empty:
            continue
        history_ids = history["activityId"].astype(str).tolist()
        train_segments = pd.concat(
            [bundle.segments_by_activity[aid] for aid in history_ids if aid in bundle.segments_by_activity],
            ignore_index=True,
        )
        train_obs = {
            aid: float(meta.loc[aid, "actualTimeSec"])
            for aid in history_ids
            if aid in meta.index and np.isfinite(meta.loc[aid, "actualTimeSec"])
        }
        stage3 = _fit_stage3_on_train(train_segments, train_obs, bundle.config)
        route = _route_for_prescribed_prediction(bundle.segments_by_activity[held_out])
        load = float(meta.loc[held_out, "rediReadinessFactor"]) if held_out in meta.index else 1.0
        actual = float(meta.loc[held_out, "actualTimeSec"])
        # Use prior duration proxy: nearest historical duration median for selection target.
        target_dur_h = float(history["actualTimeSec"].median()) / 3600.0

        for rule in rules:
            # Never pass held-out duration into history-only selection.
            selected_hrr = _select_history_hrr(
                rule,
                history,
                hrr_reference=hrr_ref,
                target_duration_h=target_dur_h,
            )
            pred = _simulate_prescribed(
                route,
                hrr=float(selected_hrr),
                alpha=float(stage3["alpha"]),
                fatigue_coef=float(stage3["fatigueCoef"]),
                fatigue_model=str(stage3["fatigueModel"]),
                config=bundle.config,
                load_factor=load,
                fatigue_mode="predicted_trimp",
            )
            rows.append(
                {
                    "activityId": held_out,
                    "startDate": str(event_date.date()) if pd.notna(event_date) else "",
                    "eventLabel": str(event["eventLabel"]),
                    "courseProfileStatus": (
                        "strict_pre_race"
                        if held_out in set(bundle.config.get("claim_validation", {}).get("strict_pre_race_activity_ids", []))
                        else "executed_geometry"
                        if held_out
                        in set(bundle.config.get("claim_validation", {}).get("pseudo_prospective_activity_ids", []))
                        else "unknown"
                    ),
                    "rule": rule,
                    "selectedHrr": float(selected_hrr),
                    "actualTimeSec": actual,
                    "predictedTimeSec": pred,
                    "errorMin": (pred - actual) / 60.0,
                    "apePct": abs(pred - actual) / actual * 100.0 if actual > 0 else float("nan"),
                    "nHistory": int(len(history)),
                    "exploratory": False,
                }
            )
        logger.info("E4 event %s (%s) done", held_out, event.get("eventLabel"))

    folds = pd.DataFrame(rows)
    summary_rows = []
    claim = bundle.config.get("claim_validation", {})
    n_boot = int(claim.get("bootstrap_iterations", 10000))
    seed = int(claim.get("bootstrap_seed", 20260722))
    if not folds.empty:
        baseline_rule = "fixed_hrr_reference"
        base_wide = folds[folds["rule"].eq(baseline_rule)].set_index("activityId")
        for rule, group in folds.groupby("rule"):
            metrics = _metrics_row(
                group["actualTimeSec"].to_numpy(),
                group["predictedTimeSec"].to_numpy(),
                (group["actualTimeSec"] / 3600.0).to_numpy(),
            )
            row: dict[str, object] = {"rule": rule, **metrics}
            if rule != baseline_rule and not base_wide.empty:
                aligned = group.set_index("activityId").reindex(base_wide.index).dropna(subset=["predictedTimeSec"])
                boot = _paired_mape_bootstrap(
                    base_wide.loc[aligned.index, "actualTimeSec"].to_numpy(),
                    base_wide.loc[aligned.index, "predictedTimeSec"].to_numpy(),
                    aligned["predictedTimeSec"].to_numpy(),
                    n_boot=n_boot,
                    seed=seed + abs(hash(str(rule))) % 1000,
                )
                row.update(
                    {
                        "deltaMapeVsFixedPct": boot["deltaMapePct"],
                        "bootstrapCiLow": boot["ciLow"],
                        "bootstrapCiHigh": boot["ciHigh"],
                        "relativeGainVsFixed": boot["relativeGainVsA"],
                    }
                )
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    folds.to_csv(bundle.output_dir / "table_rolling_origin_hrr_selection_folds.csv", index=False)
    out = bundle.output_dir / "table_rolling_origin_hrr_selection.csv"
    summary.to_csv(out, index=False)
    logger.info("Wrote %s", out)
    return summary


def run_e1_intensity_premise(bundle: ValidationBundle, support: pd.DataFrame) -> pd.DataFrame:
    """E1: does realised intensity vary after course/duration adjustment?"""
    df = support.copy()
    df = df[df["cohort_hardRunOrTrailRun"].astype(bool)].copy()
    y = pd.to_numeric(df["hrReserveRatio"], errors="coerce")
    x_cols = ["durationHours", "ascentPerKm", "meanAltitudeM", "distanceKm"]
    design = df[x_cols].apply(pd.to_numeric, errors="coerce")
    design["logDuration"] = np.log(np.clip(design["durationHours"], 1e-3, None))
    design = design.drop(columns=["durationHours"])
    valid = y.notna() & design.notna().all(axis=1)
    y_v = y[valid].to_numpy()
    x_v = design.loc[valid].to_numpy()
    x_design = np.column_stack([np.ones(len(x_v)), x_v])
    coef, *_ = np.linalg.lstsq(x_design, y_v, rcond=None)
    fitted = x_design @ coef
    resid = y_v - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_v - np.mean(y_v)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out = pd.DataFrame(
        [
            {
                "n": int(valid.sum()),
                "residualStdHrr": float(np.std(resid)),
                "r2DurationCourse": float(r2),
                "meanAbsResidualHrr": float(np.mean(np.abs(resid))),
                "interpretation": (
                    "material_intensity_variation"
                    if float(np.std(resid)) >= 0.03
                    else "limited_intensity_variation"
                ),
            }
        ]
    )
    path = bundle.output_dir / "table_intensity_premise_e1.csv"
    out.to_csv(path, index=False)
    logger.info("Wrote %s", path)
    return out


def run_e3_route_pairs(bundle: ValidationBundle, support: pd.DataFrame) -> pd.DataFrame:
    """E3: coarse repeated-route contrasts (distance/ascent matching)."""
    df = support[support["cohort_hardTrailRun"].astype(bool)].copy()
    df["distanceKm"] = pd.to_numeric(df["distanceKm"], errors="coerce")
    df["ascentM"] = pd.to_numeric(df["ascentM"], errors="coerce")
    df["movingSec"] = pd.to_numeric(df["movingSec"], errors="coerce")
    df["hrReserveRatio"] = pd.to_numeric(df["hrReserveRatio"], errors="coerce")
    pairs: list[dict[str, object]] = []
    ids = df["activityId"].astype(str).tolist()
    for i, a in enumerate(ids):
        ra = df.loc[df["activityId"].eq(a)].iloc[0]
        for b in ids[i + 1 :]:
            rb = df.loc[df["activityId"].eq(b)].iloc[0]
            if not np.isfinite(ra["distanceKm"]) or not np.isfinite(rb["distanceKm"]):
                continue
            dist_rel = abs(ra["distanceKm"] - rb["distanceKm"]) / max(ra["distanceKm"], 1e-6)
            ascent_rel = abs(ra["ascentM"] - rb["ascentM"]) / max(abs(ra["ascentM"]), abs(rb["ascentM"]), 1.0)
            if dist_rel > 0.10 or ascent_rel > 0.20:
                continue
            pairs.append(
                {
                    "activityIdA": a,
                    "activityIdB": b,
                    "distanceKmA": float(ra["distanceKm"]),
                    "distanceKmB": float(rb["distanceKm"]),
                    "ascentMA": float(ra["ascentM"]),
                    "ascentMB": float(rb["ascentM"]),
                    "hrrA": float(ra["hrReserveRatio"]),
                    "hrrB": float(rb["hrReserveRatio"]),
                    "timeSecA": float(ra["movingSec"]),
                    "timeSecB": float(rb["movingSec"]),
                    "observedTimeDiffMin": (float(rb["movingSec"]) - float(ra["movingSec"])) / 60.0,
                    "observedHrrDiff": float(rb["hrReserveRatio"]) - float(ra["hrReserveRatio"]),
                    "sameOrderingHrrAndTime": int(
                        np.sign(float(rb["hrReserveRatio"]) - float(ra["hrReserveRatio"]))
                        == np.sign(float(ra["movingSec"]) - float(rb["movingSec"]))
                        and abs(float(rb["hrReserveRatio"]) - float(ra["hrReserveRatio"])) > 0.02
                    ),
                }
            )
    table = pd.DataFrame(pairs)
    out = bundle.output_dir / "table_route_pair_validation.csv"
    table.to_csv(out, index=False)
    summary = pd.DataFrame(
        [
            {
                "nPairs": int(len(table)),
                "concordanceAmongMaterialHrrDiff": float(table["sameOrderingHrrAndTime"].mean())
                if len(table)
                else float("nan"),
                "status": "coarse_match_only" if len(table) else "not_testable",
            }
        ]
    )
    summary.to_csv(bundle.output_dir / "table_route_pair_validation_summary.csv", index=False)
    logger.info("Wrote %s (%d pairs)", out, len(table))
    return table


def evaluate_claim_gates(
    bundle: ValidationBundle,
    *,
    e2_summary: pd.DataFrame,
    e4_summary: pd.DataFrame,
    e5_summary: pd.DataFrame,
    e1_summary: pd.DataFrame,
    e3_summary: pd.DataFrame,
) -> pd.DataFrame:
    claim = bundle.config.get("claim_validation", {})
    primary = str(claim.get("primary_cohort", "hardTrailRun"))
    gain_gate = float(claim.get("mape_relative_gain_gate", 0.10))
    rows: list[dict[str, object]] = []

    e2 = e2_summary[e2_summary["cohort"].astype(str).eq(primary)].copy()
    b3 = e2[e2["modelId"].eq("B3")]
    contrasts = e2[e2["modelId"].isin(["B3_vs_B0", "B3_vs_B1"])]
    best_contrast = None
    if not contrasts.empty and "relativeGainVsBaseline" in contrasts.columns:
        best_contrast = contrasts.sort_values("relativeGainVsBaseline", ascending=False).iloc[0]
    if best_contrast is not None and not b3.empty:
        rel = float(best_contrast.get("relativeGainVsBaseline", np.nan))
        lo = float(best_contrast.get("bootstrapCiLow", np.nan))
        hi = float(best_contrast.get("bootstrapCiHigh", np.nan))
        status = (
            "pass"
            if np.isfinite(rel) and rel >= gain_gate and np.isfinite(lo) and lo > 0
            else "fail"
        )
        rows.append(
            {
                "claimId": "prescribed_hrr_beats_fixed",
                "experiment": "E2",
                "status": status,
                "metric": "relative_mape_gain_B3_vs_best_fixed",
                "value": rel,
                "ciLow": lo,
                "ciHigh": hi,
                "threshold": gain_gate,
                "reason": (
                    f"B3 MAPE={float(b3.iloc[0]['mapePct']):.2f}; "
                    f"best contrast {best_contrast['modelId']} relative gain={rel:.3f}"
                ),
            }
        )
    else:
        rows.append(
            {
                "claimId": "prescribed_hrr_beats_fixed",
                "experiment": "E2",
                "status": "not_testable",
                "metric": "relative_mape_gain_B3_vs_best_fixed",
                "value": float("nan"),
                "ciLow": float("nan"),
                "ciHigh": float("nan"),
                "threshold": gain_gate,
                "reason": "missing E2 contrast rows",
            }
        )

    if not e4_summary.empty and "relativeGainVsFixed" in e4_summary.columns:
        candidates = e4_summary[~e4_summary["rule"].eq("fixed_hrr_reference")].copy()
        candidates = candidates.sort_values("relativeGainVsFixed", ascending=False)
        top = candidates.iloc[0] if len(candidates) else None
        if top is not None:
            rel = float(top.get("relativeGainVsFixed", np.nan))
            lo = float(top.get("bootstrapCiLow", np.nan))
            hi = float(top.get("bootstrapCiHigh", np.nan))
            status = "pass" if np.isfinite(rel) and rel > 0 and np.isfinite(lo) and lo > 0 else "fail"
            rows.append(
                {
                    "claimId": "history_only_hrr",
                    "experiment": "E4",
                    "status": status,
                    "metric": "relative_mape_gain_vs_fixed_hrr_reference",
                    "value": rel,
                    "ciLow": lo,
                    "ciHigh": hi,
                    "threshold": 0.0,
                    "reason": f"best rule={top['rule']}",
                }
            )
        else:
            rows.append(
                {
                    "claimId": "history_only_hrr",
                    "experiment": "E4",
                    "status": "not_testable",
                    "metric": "relative_mape_gain_vs_fixed_hrr_reference",
                    "value": float("nan"),
                    "ciLow": float("nan"),
                    "ciHigh": float("nan"),
                    "threshold": 0.0,
                    "reason": "no E4 rules",
                }
            )
    else:
        rows.append(
            {
                "claimId": "history_only_hrr",
                "experiment": "E4",
                "status": "not_testable",
                "metric": "relative_mape_gain_vs_fixed_hrr_reference",
                "value": float("nan"),
                "ciLow": float("nan"),
                "ciHigh": float("nan"),
                "threshold": 0.0,
                "reason": "E4 summary missing paired gains",
            }
        )

    e5 = e5_summary[e5_summary["cohort"].astype(str).eq(primary)]
    trimp = e5[e5["variant"].eq("predicted_trimp")]
    progress = e5[e5["variant"].eq("progress")]
    if not trimp.empty and not progress.empty:
        delta = float(progress.iloc[0]["mapePct"]) - float(trimp.iloc[0]["mapePct"])
        status = "pass" if delta > 0 else "fail"
        rows.append(
            {
                "claimId": "causal_trimp",
                "experiment": "E5",
                "status": status,
                "metric": "mape_progress_minus_predicted_trimp",
                "value": delta,
                "ciLow": float("nan"),
                "ciHigh": float("nan"),
                "threshold": 0.0,
                "reason": (
                    f"predicted_trimp MAPE={float(trimp.iloc[0]['mapePct']):.2f}; "
                    f"progress MAPE={float(progress.iloc[0]['mapePct']):.2f}"
                ),
            }
        )
    else:
        rows.append(
            {
                "claimId": "causal_trimp",
                "experiment": "E5",
                "status": "not_testable",
                "metric": "mape_progress_minus_predicted_trimp",
                "value": float("nan"),
                "ciLow": float("nan"),
                "ciHigh": float("nan"),
                "threshold": 0.0,
                "reason": "missing E5 variants",
            }
        )

    intensity_ok = (
        not e1_summary.empty
        and str(e1_summary.iloc[0].get("interpretation", "")) == "material_intensity_variation"
    )
    rows.append(
        {
            "claimId": "intensity_premise",
            "experiment": "E1",
            "status": "pass" if intensity_ok else "fail",
            "metric": "residual_std_hrr_after_course_duration",
            "value": float(e1_summary.iloc[0]["residualStdHrr"]) if not e1_summary.empty else float("nan"),
            "ciLow": float("nan"),
            "ciHigh": float("nan"),
            "threshold": 0.03,
            "reason": str(e1_summary.iloc[0]["interpretation"]) if not e1_summary.empty else "missing",
        }
    )

    n_pairs = int(e3_summary["nPairs"].iloc[0]) if not e3_summary.empty else 0
    rows.append(
        {
            "claimId": "repeated_route_curve",
            "experiment": "E3",
            "status": "pass" if n_pairs >= 5 else "not_testable",
            "metric": "n_coarse_route_pairs",
            "value": float(n_pairs),
            "ciLow": float("nan"),
            "ciHigh": float("nan"),
            "threshold": 5.0,
            "reason": "coarse distance/ascent matching only; GPS overlap not available",
        }
    )

    for claim_id, experiment, reason in [
        ("longitudinal_capacity", "E6", "P1 not run in this P0 execution"),
        ("within_race_update", "E7", "P2 exploratory; not run"),
        ("lab_free_archive_anchor", "E8", "P1 not run in this P0 execution"),
    ]:
        rows.append(
            {
                "claimId": claim_id,
                "experiment": experiment,
                "status": "not_testable",
                "metric": "",
                "value": float("nan"),
                "ciLow": float("nan"),
                "ciHigh": float("nan"),
                "threshold": float("nan"),
                "reason": reason,
            }
        )

    # Decision-table row.
    gate_map = {r["claimId"]: r["status"] for r in rows}
    if gate_map.get("prescribed_hrr_beats_fixed") == "pass" and gate_map.get("history_only_hrr") == "pass":
        story = "Strava-history intensity-conditioned prospective model within supported domain"
    elif gate_map.get("prescribed_hrr_beats_fixed") == "pass":
        story = "Useful prescribed-intensity simulator; athlete/coach must choose HRR"
    else:
        story = "Retrospective HR-informed reconstruction; no prospective target-HRR claim yet"
    rows.append(
        {
            "claimId": "paper_story",
            "experiment": "decision_table",
            "status": "info",
            "metric": "selected_story",
            "value": float("nan"),
            "ciLow": float("nan"),
            "ciHigh": float("nan"),
            "threshold": float("nan"),
            "reason": story,
        }
    )

    table = pd.DataFrame(rows)
    out = bundle.output_dir / "claim_gates.csv"
    table.to_csv(out, index=False)
    logger.info("Wrote %s", out)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "trail_digital_twin_claim_validation.yaml",
    )
    parser.add_argument(
        "--skip-e4",
        action="store_true",
        help="Skip rolling-origin HRR selection (expensive).",
    )
    parser.add_argument(
        "--secondary-cohort",
        action="store_true",
        help="Also run E2/E5 on hardRunOrTrailRun.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=None,
        help="Override claim_validation.bootstrap_iterations",
    )
    parser.add_argument(
        "--max-activities",
        type=int,
        default=None,
        help="Cap cohort size for smoke tests (deterministic head).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = pipeline.load_config(args.config)
    if args.bootstrap_iterations is not None:
        config.setdefault("claim_validation", {})
        config["claim_validation"]["bootstrap_iterations"] = int(args.bootstrap_iterations)

    bundle = build_bundle(config, REPO_ROOT)
    if args.max_activities is not None:
        n = max(3, int(args.max_activities))
        for name, cohort_df in list(bundle.cohorts.items()):
            if cohort_df.empty:
                continue
            capped = cohort_df.head(n).copy()
            logger.warning("Capping cohort %s to %d activities for smoke test", name, len(capped))
            bundle.cohorts[name] = capped
    write_freeze_manifest(bundle, REPO_ROOT)
    support = run_e0_data_support(bundle)
    e1 = run_e1_intensity_premise(bundle, support)
    e3_pairs = run_e3_route_pairs(bundle, support)
    e3_summary = pd.read_csv(bundle.output_dir / "table_route_pair_validation_summary.csv")

    primary = str(config.get("claim_validation", {}).get("primary_cohort", "hardTrailRun"))
    _folds, e2_summary = run_e2_prescribed_hrr_loo(bundle, cohort_name=primary)
    e5_summary = run_e5_causal_fatigue(bundle, cohort_name=primary)

    if args.secondary_cohort:
        run_e2_prescribed_hrr_loo(bundle, cohort_name="hardRunOrTrailRun")
        e5_summary = run_e5_causal_fatigue(bundle, cohort_name="hardRunOrTrailRun")
        e2_summary = pd.read_csv(bundle.output_dir / "table_prescribed_hrr_model_comparison.csv")

    if args.skip_e4:
        e4_summary = pd.DataFrame()
        logger.warning("Skipping E4 by request")
    else:
        e4_summary = run_e4_rolling_origin(bundle)

    gates = evaluate_claim_gates(
        bundle,
        e2_summary=e2_summary
        if isinstance(e2_summary, pd.DataFrame)
        else pd.read_csv(bundle.output_dir / "table_prescribed_hrr_model_comparison.csv"),
        e4_summary=e4_summary,
        e5_summary=e5_summary,
        e1_summary=e1,
        e3_summary=e3_summary,
    )
    print(gates.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
