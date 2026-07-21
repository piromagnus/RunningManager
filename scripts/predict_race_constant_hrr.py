"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Prospective constant-HRR race prediction from a planned profile.

Fits Stage 3 alpha/fatigue on other activities only (hold-out races excluded),
then predicts courses at a constant HRR. Default mode selects the fastest HRR
that is historically sustainable for the predicted duration (power-law
HRR–duration envelope). Optional ``--hrr-mode reference`` holds HRR =
hrr_reference (E=1 when fresh; not HRR at VMA).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services import trail_performance_model as tpm  # noqa: E402
from utils.gpx_parser import parse_gpx_to_timeseries  # noqa: E402

logger = logging.getLogger(__name__)

# Representative windows for the sustainable-HRR power law (minutes).
DEFAULT_DURATION_WINDOWS_MIN = (
    5,
    10,
    15,
    20,
    30,
    45,
    60,
    90,
    120,
    180,
    240,
    360,
    480,
    720,
    960,
    1440,
)

HOLDOUT_ACTIVITY_IDS = (
    "17481444994",  # Grésivaudan
    "16325125849",  # LUT By Night
    "15563904138",  # Echappée Belle 2025
    "15087396899",  # Trail des Passerelles / Côte Rouge 2025
)

RACES = {
    "lut_30k": {
        "label": "Lyon Urban Trail By Night 2025",
        "activityId": "16325125849",
        "gpx": "divers/30km - LUT By Night .gpx",
        "race_pacing_id": "d8b4117c-ef61-488c-9317-812622d44863",
    },
    "gresivaudan": {
        "label": "Trail du Grésivaudan 2026 : Le Grand V",
        "activityId": "17481444994",
        "gpx": "divers/trail-du-gresivaudan-2026.gpx",
        "race_pacing_id": "c67065ad-ec3c-41ce-9388-d62acd4d1531",
    },
    "echappee_belle": {
        "label": "Echappée Belle 2025 : Parcours des crêtes",
        "activityId": "15563904138",
        "profile_source": "activity_timeseries",
    },
    "trail_passerelles": {
        "label": "Trail de côte rouge 2025 (Passerelle de Monteynard)",
        "activityId": "15087396899",
        "profile_source": "activity_timeseries",
    },
}


def _fmt_hms(seconds: float) -> str:
    total = int(round(float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def segments_from_gpx(gpx_path: Path, segment_km: float = 1.0) -> pd.DataFrame:
    route = parse_gpx_to_timeseries(gpx_path.read_bytes())
    return tpm.route_segments_from_points(route, segment_km=segment_km)


def segments_from_race_pacing(csv_path: Path) -> pd.DataFrame:
    """Convert planned pacer segments into Stage 3 route rows (no observed HR/time)."""
    raw = pd.read_csv(csv_path)
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df["segmentIndex"] = np.arange(len(df), dtype=int)
    df["startKm"] = pd.to_numeric(df.get("startKm"), errors="coerce")
    df["endKm"] = pd.to_numeric(df.get("endKm"), errors="coerce")
    df["distanceKm"] = pd.to_numeric(df["distanceKm"], errors="coerce")
    df["elevGainM"] = pd.to_numeric(df.get("elevGainM"), errors="coerce").fillna(0.0)
    df["elevLossM"] = pd.to_numeric(df.get("elevLossM"), errors="coerce").fillna(0.0)
    df["avgGrade"] = pd.to_numeric(df.get("avgGrade"), errors="coerce")
    missing_grade = df["avgGrade"].isna() & df["distanceKm"].gt(0)
    df.loc[missing_grade, "avgGrade"] = (
        (df.loc[missing_grade, "elevGainM"] - df.loc[missing_grade, "elevLossM"])
        / (df.loc[missing_grade, "distanceKm"] * 1000.0)
    )
    df["avgGrade"] = df["avgGrade"].fillna(0.0)
    # Planned profiles rarely carry absolute altitude; use 0 so altitude factor = 1.
    df["meanAltitudeM"] = 0.0
    df["gapFactorIntegrated"] = df["avgGrade"].map(tpm.gap_factor)
    df["gapFactorAvgGrade"] = df["gapFactorIntegrated"]
    total = float(df["distanceKm"].sum())
    df["progress"] = ((df["startKm"] + df["endKm"]) / 2.0 / max(total, 1e-9)).clip(0.0, 1.0)
    df["activityId"] = "route"
    df["actualTimeSec"] = np.nan
    df["meanHrReserve"] = np.nan
    df["terrainFamily"] = df["avgGrade"].map(tpm.terrain_family)
    return df[
        [
            "segmentIndex",
            "startKm",
            "endKm",
            "distanceKm",
            "elevGainM",
            "elevLossM",
            "meanAltitudeM",
            "avgGrade",
            "gapFactorIntegrated",
            "gapFactorAvgGrade",
            "terrainFamily",
            "progress",
            "activityId",
            "actualTimeSec",
            "meanHrReserve",
        ]
    ]


def segments_from_activity_timeseries(
    timeseries_csv: Path,
    *,
    segment_km: float = 1.0,
) -> pd.DataFrame:
    """Build a prospective route from a real executed activity GPS profile.

    Uses geometry only (distance, grade, altitude). Observed HR/times are not
    passed into the constant-HRR simulator; callers compare against activity
    moving time separately after hold-out fitting.
    """
    raw = pd.read_csv(timeseries_csv)
    if raw.empty:
        return pd.DataFrame()
    prepared = tpm.prepare_raw_timeseries_for_segments(raw)
    segs = tpm.segment_timeseries(prepared, segment_km=segment_km)
    if segs.empty:
        return pd.DataFrame()
    out = segs.copy()
    out["activityId"] = "route"
    # Prospective profile: strip observed effort/time from the simulator inputs.
    out["actualTimeSec"] = np.nan
    out["meanHrReserve"] = np.nan
    if "gapFactorIntegrated" not in out.columns:
        out["gapFactorIntegrated"] = out["avgGrade"].map(tpm.gap_factor)
    if "gapFactorAvgGrade" not in out.columns:
        out["gapFactorAvgGrade"] = out["gapFactorIntegrated"]
    if "terrainFamily" not in out.columns:
        out["terrainFamily"] = out["avgGrade"].map(tpm.terrain_family)
    keep = [
        "segmentIndex",
        "startKm",
        "endKm",
        "distanceKm",
        "elevGainM",
        "elevLossM",
        "meanAltitudeM",
        "avgGrade",
        "gapFactorIntegrated",
        "gapFactorAvgGrade",
        "terrainFamily",
        "progress",
        "activityId",
        "actualTimeSec",
        "meanHrReserve",
    ]
    return out[[c for c in keep if c in out.columns]].reset_index(drop=True)


def load_train_segments(
    predictions_csv: Path,
    holdout_ids: set[str],
    *,
    cohort: str = "hardRunOrTrailRun",
) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv, dtype={"activityId": str})
    sub = df[(df["cohort"] == cohort) & (df["fitObjective"] == "segment")].copy()
    sub = sub.drop_duplicates(["activityId", "segmentIndex"], keep="first")
    sub = sub[~sub["activityId"].isin(holdout_ids)].copy()
    return sub.reset_index(drop=True)


def attach_altitude_from_gpx(pacing: pd.DataFrame, gpx_path: Path) -> pd.DataFrame:
    """Fill meanAltitudeM on planned segments from a GPX elevation profile."""
    route = parse_gpx_to_timeseries(gpx_path.read_bytes())
    prepared = tpm.prepare_raw_timeseries_for_segments(route.drop(columns=["timestamp"], errors="ignore"))
    elev_col = "elevationM_ma_5" if "elevationM_ma_5" in prepared.columns else "elevationM"
    dist = prepared["cumulated_distance"].to_numpy(dtype=float)
    elev = pd.to_numeric(prepared[elev_col], errors="coerce").to_numpy(dtype=float)
    out = pacing.copy()
    means: list[float] = []
    for _, row in out.iterrows():
        mask = (dist >= float(row["startKm"]) - 1e-9) & (dist <= float(row["endKm"]) + 1e-9)
        if mask.any() and np.isfinite(elev[mask]).any():
            means.append(float(np.nanmean(elev[mask])))
        else:
            means.append(0.0)
    out["meanAltitudeM"] = means
    return out


def fit_stage3(
    train: pd.DataFrame,
    physiology: dict[str, float],
    *,
    objective: str = "race",
) -> dict[str, object]:
    actual_col = "actualMovingTimeSec" if "actualMovingTimeSec" in train.columns else "actualTimeSec"
    observed: dict[str, float] | None = None
    if objective == "race" and "actualTimeSec_activity" in train.columns:
        observed = {
            str(activity_id): float(group["actualTimeSec_activity"].iloc[0])
            for activity_id, group in train.groupby("activityId")
        }
    best, _grid, _pred = tpm.hrr_trimp_grid_search_model(
        train,
        v_anchor_kmh=float(physiology["vma_flat_kmh"]),
        alpha_grid=[0.85, 0.90, 0.95, 1.00, 1.05],
        fatigue_coef_grid=[0.2, 0.3, 0.4, 0.6],
        secondary_fatigue_coef_grid=[0.0],
        fatigue_models=("linear", "exponential"),
        objective=objective,
        actual_time_col=actual_col,
        observed_activity_times_sec=observed,
        hrr_reference=float(physiology["hrr_reference"]),
        hrr_min_factor=float(physiology["hrr_min_factor"]),
        hrr_max_factor=float(physiology["hrr_max_factor"]),
        min_fatigue_factor=float(physiology["min_fatigue_factor"]),
        load_factor_col="rediReadinessFactor" if "rediReadinessFactor" in train.columns else None,
        use_hrr_effort=True,
        acute_trimp_col="decayedTrimpBefore",
        gap_steep_threshold=float(physiology["gap_steep_threshold"]),
        gap_soft_start=float(physiology["gap_soft_start"]),
        gap_climb_scale=float(physiology["gap_climb_scale"]),
        gap_descent_scale=float(physiology["gap_descent_scale"]),
    )
    return {
        "alpha": float(best["alpha"]),
        "fatigueCoef": float(best["fatigueCoef"]),
        "fatigueModel": str(best["fatigueModel"]),
        "segmentMaeSec": float(best.get("segmentMaeSec", float("nan"))),
        "raceMaeSec": float(best.get("raceMaeSec", float("nan"))),
        "fitObjective": objective,
        "actualTimeCol": actual_col,
        "nTrainSegments": int(len(train)),
        "nTrainActivities": int(train["activityId"].nunique()),
    }


def predict_route(
    segments: pd.DataFrame,
    *,
    hrr: float,
    fit: dict[str, object],
    physiology: dict[str, float],
    fatigue_input_col: str = "cumTrimpBefore",
) -> pd.DataFrame:
    return tpm.simulate_constant_hrr_route(
        segments,
        hrr=hrr,
        v_anchor_kmh=float(physiology["vma_flat_kmh"]),
        alpha=float(fit["alpha"]),
        fatigue_coef=float(fit["fatigueCoef"]),
        fatigue_model=str(fit["fatigueModel"]),
        hrr_reference=float(physiology["hrr_reference"]),
        hrr_min_factor=float(physiology["hrr_min_factor"]),
        hrr_max_factor=float(physiology["hrr_max_factor"]),
        decay_lambda=float(physiology["decay_lambda"]),
        min_fatigue_factor=float(physiology["min_fatigue_factor"]),
        load_factor=1.0,
        fatigue_input_col=fatigue_input_col,
        gap_steep_threshold=float(physiology["gap_steep_threshold"]),
        gap_soft_start=float(physiology["gap_soft_start"]),
        gap_climb_scale=float(physiology["gap_climb_scale"]),
        gap_descent_scale=float(physiology["gap_descent_scale"]),
    )


def fit_hrr_duration_power_law(
    *,
    data_dir: Path,
    holdout_ids: set[str],
    hr_rest: float,
    hr_max: float,
    duration_windows_min: tuple[float, ...] = DEFAULT_DURATION_WINDOWS_MIN,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit sustainable HRR vs duration, excluding prospective hold-out activities."""
    metrics = pd.read_csv(data_dir / "activities_metrics.csv", dtype={"activityId": str})
    if "activityId" in metrics.columns and holdout_ids:
        before = len(metrics)
        metrics = metrics[~metrics["activityId"].astype(str).isin(holdout_ids)].copy()
        dropped = before - len(metrics)
        if dropped:
            logger.info(
                "Excluded %d hold-out activities from HRR–duration envelope fit",
                dropped,
            )
    params, windows = tpm.estimate_hrr_duration_power_law(
        metrics,
        duration_windows_min=list(duration_windows_min),
        hr_rest=hr_rest,
        hr_max=hr_max,
        hrr_min=0.30,
        hrr_max=0.98,
        target_stat="max",
        target_quantile=0.90,
        min_activity_count=3,
        fit_weight_mode="performance",
        fit_weight_power=10.0,
        categories=("RUN", "TRAIL_RUN"),
    )
    return dict(params), windows


def select_duration_feasible_constant_hrr(
    segments: pd.DataFrame,
    *,
    fit: dict[str, object],
    physiology: dict[str, float],
    power_law_params: dict[str, Any],
    hr_rest: float | None = None,
    hr_max: float | None = None,
    hrr_min: float = 0.50,
    hrr_step: float = 0.01,
    fatigue_input_col: str = "cumTrimpBefore",
) -> tuple[pd.Series, pd.DataFrame]:
    """Sweep constant HRR and pick the fastest historically sustainable value.

    For each candidate HRR ``x``, predict finish time ``T(x)`` and compare to the
    power-law max duration maintainable at ``x``. Selection is the fastest
    feasible row (``select_best_constant_hrr``).
    """
    hrr_ref = float(physiology["hrr_reference"])
    # Above HRR_ref, E cannot rise under hrr_max_factor=1.0; only TRIMP grows.
    hrr_max_grid = min(hrr_ref, 0.98)
    hrr_values = np.round(np.arange(hrr_min, hrr_max_grid + 0.5 * hrr_step, hrr_step), 4)
    sweep = tpm.sweep_constant_hrr_route(
        segments,
        hrr_values=hrr_values,
        v_anchor_kmh=float(physiology["vma_flat_kmh"]),
        alpha=float(fit["alpha"]),
        fatigue_coef=float(fit["fatigueCoef"]),
        fatigue_model=str(fit["fatigueModel"]),
        hrr_reference=hrr_ref,
        hrr_min_factor=float(physiology["hrr_min_factor"]),
        hrr_max_factor=float(physiology["hrr_max_factor"]),
        decay_lambda=float(physiology["decay_lambda"]),
        min_fatigue_factor=float(physiology["min_fatigue_factor"]),
        load_factor=1.0,
        hr_rest=hr_rest,
        hr_max=hr_max,
        fatigue_input_col=fatigue_input_col,
        gap_steep_threshold=float(physiology["gap_steep_threshold"]),
        gap_soft_start=float(physiology["gap_soft_start"]),
        gap_climb_scale=float(physiology["gap_climb_scale"]),
        gap_descent_scale=float(physiology["gap_descent_scale"]),
    )
    sweep = sweep.copy()
    sweep["maxSustainableSec"] = sweep["hrr"].map(
        lambda hrr: tpm.max_duration_for_hrr_power_law(float(hrr), power_law_params)
    )
    sweep["maxSustainableHours"] = sweep["maxSustainableSec"] / 3600.0
    sweep["sustainabilityMarginMin"] = (
        sweep["maxSustainableSec"] - sweep["predictedTimeSec"]
    ) / 60.0
    sweep["feasible"] = sweep["predictedTimeSec"].le(sweep["maxSustainableSec"])
    sweep["durationModel"] = "power_law"
    best = tpm.select_best_constant_hrr(sweep)
    if best.empty:
        logger.warning("Empty HRR sweep; falling back to hrr_reference=%.2f", hrr_ref)
        return pd.Series({"hrr": hrr_ref, "feasible": False}), sweep
    if not bool(best.get("feasible", False)):
        logger.warning(
            "No feasible constant HRR on envelope; using least-infeasible hrr=%.3f "
            "(pred=%.0fs, maxSustainable=%.0fs)",
            float(best["hrr"]),
            float(best.get("predictedTimeSec", float("nan"))),
            float(best.get("maxSustainableSec", float("nan"))),
        )
    return best, sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path(
            "data/exp_perf_predictions/trail_digital_twin_moving_time_fit/"
            "runs/001_moving_time_fit_ab_fit_moving_time_only/segment_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/exp_perf_predictions/trail_digital_twin_race_prediction"),
    )
    parser.add_argument(
        "--profile-source",
        choices=("race_pacing", "gpx", "race_pacing_gpxalt", "both"),
        default="race_pacing_gpxalt",
    )
    parser.add_argument("--fit-objective", choices=("race", "segment"), default="race")
    parser.add_argument(
        "--hrr-mode",
        choices=("duration-feasible", "reference"),
        default="duration-feasible",
        help=(
            "duration-feasible: fastest constant HRR sustainable for predicted duration "
            "(power-law envelope). reference: hold HRR=hrr_reference (E=1)."
        ),
    )
    parser.add_argument(
        "--hard-hrr",
        type=float,
        default=None,
        help="Force a fixed constant HRR (overrides --hrr-mode)",
    )
    args = parser.parse_args()

    physiology = {
        "vma_flat_kmh": 18.0,
        "hrr_reference": 0.88,
        "hrr_min_factor": 0.30,
        "hrr_max_factor": 1.00,
        "decay_lambda": 0.20,
        "min_fatigue_factor": 0.60,
        "gap_steep_threshold": 0.15,
        "gap_soft_start": 0.04,
        "gap_climb_scale": 0.85,
        "gap_descent_scale": 1.60,
    }
    holdout = set(HOLDOUT_ACTIVITY_IDS)

    train = load_train_segments(args.predictions_csv, holdout, cohort="hardRunOrTrailRun")
    if train.empty:
        raise SystemExit("no training segments after hold-out")
    leaked = set(train["activityId"]) & holdout
    if leaked:
        raise SystemExit(f"hold-out leak in train: {leaked}")

    fit = fit_stage3(train, physiology, objective=args.fit_objective)
    print("Fit (hold-outs excluded):", json.dumps(fit, indent=2))

    activities = pd.read_csv(REPO_ROOT / "data" / "activities.csv", dtype={"activityId": str})
    athlete = pd.read_csv(REPO_ROOT / "data" / "athlete.csv").iloc[0]
    hr_rest, hr_max = float(athlete["hrRest"]), float(athlete["hrMax"])

    power_law_params: dict[str, Any] | None = None
    if args.hard_hrr is None and args.hrr_mode == "duration-feasible":
        power_law_params, _windows = fit_hrr_duration_power_law(
            data_dir=REPO_ROOT / "data",
            holdout_ids=holdout,
            hr_rest=hr_rest,
            hr_max=hr_max,
        )
        print(
            "HRR–duration power law:",
            json.dumps(
                {
                    "coefficient": power_law_params.get("coefficient"),
                    "exponent": power_law_params.get("exponent"),
                    "maeHrr": power_law_params.get("maeHrr"),
                    "fitWindowCount": power_law_params.get("fitWindowCount"),
                },
                indent=2,
            ),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for race_key, meta in RACES.items():
        act = activities[activities["activityId"].eq(meta["activityId"])].iloc[0]
        actual_moving = float(act["movingSec"])
        actual_elapsed = float(act["elapsedSec"])
        # Observed HRR for journal comparison only — never used in prediction.
        observed_hrr = (float(act["avgHr"]) - hr_rest) / (hr_max - hr_rest)

        sources: list[tuple[str, pd.DataFrame]] = []
        if meta.get("profile_source") == "activity_timeseries":
            ts_path = REPO_ROOT / "data" / "timeseries" / f"{meta['activityId']}.csv"
            sources.append(("activity_timeseries", segments_from_activity_timeseries(ts_path)))
        else:
            pacing_path = REPO_ROOT / "data" / "race_pacing" / f"{meta['race_pacing_id']}_segments.csv"
            gpx_path = REPO_ROOT / meta["gpx"]
            if args.profile_source in ("race_pacing", "both"):
                sources.append(("race_pacing", segments_from_race_pacing(pacing_path)))
            if args.profile_source in ("race_pacing_gpxalt", "both"):
                pacing = segments_from_race_pacing(pacing_path)
                sources.append(("race_pacing_gpxalt", attach_altitude_from_gpx(pacing, gpx_path)))
            if args.profile_source in ("gpx", "both"):
                sources.append(("gpx", segments_from_gpx(gpx_path)))

        for source_name, segments in sources:
            if segments.empty:
                print(f"skip empty profile {race_key}/{source_name}")
                continue
            if args.hard_hrr is not None:
                chosen_hrr = float(args.hard_hrr)
                hrr_mode = "fixed_override"
                feasible = np.nan
                max_sustainable = np.nan
                margin_min = np.nan
                sweep = pd.DataFrame()
            elif power_law_params is not None:
                best, sweep = select_duration_feasible_constant_hrr(
                    segments,
                    fit=fit,
                    physiology=physiology,
                    power_law_params=power_law_params,
                    hr_rest=hr_rest,
                    hr_max=hr_max,
                )
                chosen_hrr = float(best["hrr"])
                hrr_mode = "duration_feasible"
                feasible = bool(best.get("feasible", False))
                max_sustainable = float(best.get("maxSustainableSec", float("nan")))
                margin_min = float(best.get("sustainabilityMarginMin", float("nan")))
            else:
                chosen_hrr = float(physiology["hrr_reference"])
                hrr_mode = "reference"
                feasible = np.nan
                max_sustainable = np.nan
                margin_min = np.nan
                sweep = pd.DataFrame()

            pred = predict_route(segments, hrr=chosen_hrr, fit=fit, physiology=physiology)
            pred_sec = float(pred["predictedTimeSec"].sum())
            ref_pred = predict_route(
                segments,
                hrr=float(physiology["hrr_reference"]),
                fit=fit,
                physiology=physiology,
            )
            ref_sec = float(ref_pred["predictedTimeSec"].sum())
            row = {
                "raceKey": race_key,
                "label": meta["label"],
                "profileSource": source_name,
                "activityId": meta["activityId"],
                "hrrMode": hrr_mode,
                "hardHrr": chosen_hrr,
                "hrrFeasible": feasible,
                "maxSustainableSec": max_sustainable,
                "sustainabilityMarginMin": margin_min,
                "referenceHrr": float(physiology["hrr_reference"]),
                "referencePredictedSec": ref_sec,
                "fitObjective": fit["fitObjective"],
                "alpha": fit["alpha"],
                "fatigueCoef": fit["fatigueCoef"],
                "fatigueModel": fit["fatigueModel"],
                "profileDistanceKm": float(segments["distanceKm"].sum()),
                "profileElevGainM": float(segments["elevGainM"].sum()),
                "profileMeanAltitudeM": float(
                    pd.to_numeric(segments["meanAltitudeM"], errors="coerce").mean()
                ),
                "predictedTimeSec": pred_sec,
                "predictedTimeMin": pred_sec / 60.0,
                "predictedHms": _fmt_hms(pred_sec),
                # Evaluation-only fields (not used for estimation):
                "actualMovingSec": actual_moving,
                "actualMovingMin": actual_moving / 60.0,
                "actualMovingHms": _fmt_hms(actual_moving),
                "actualElapsedSec": actual_elapsed,
                "deltaPredMinusActualMin": (pred_sec - actual_moving) / 60.0,
                "observedAvgHrr": observed_hrr,
                "nSegments": int(len(segments)),
            }
            summary_rows.append(row)
            pred.to_csv(args.output_dir / f"{race_key}_{source_name}_segments.csv", index=False)
            if not sweep.empty:
                sweep.to_csv(args.output_dir / f"{race_key}_{source_name}_hrr_sweep.csv", index=False)
            print(
                f"{meta['label']} [{source_name}] hrr={chosen_hrr:.3f} ({hrr_mode}) "
                f"pred={row['predictedHms']} "
                f"actual_moving={row['actualMovingHms']} "
                f"delta={row['deltaPredMinusActualMin']:+.1f} min "
                f"(D+ profile={row['profileElevGainM']:.0f}m)"
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "race_prediction_summary.csv", index=False)
    (args.output_dir / "fit_manifest.json").write_text(
        json.dumps(
            {
                "holdoutActivityIds": sorted(holdout),
                "fit": fit,
                "physiology": physiology,
                "hrrMode": args.hrr_mode if args.hard_hrr is None else "fixed_override",
                "powerLawParams": {
                    k: power_law_params.get(k)
                    for k in ("coefficient", "exponent", "maeHrr", "fitWindowCount")
                }
                if power_law_params
                else None,
                "note": (
                    "Actual race times/HR are evaluation-only. "
                    "duration-feasible uses HRR–duration power law excluding hold-outs."
                ),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {args.output_dir / 'race_prediction_summary.csv'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
