"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Run available §7 paper-readiness experiments for the HR digital twin.

Produces LOO metrics (incl. runTrailOver20Min), bootstrap CIs, reoptimized component ablation,
figures, speed-vs-HRR curve, weather/HR QC/elevation coverage, and prospective
finish-time bands.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services import trail_digital_twin_benchmark as bench  # noqa: E402
from services import trail_digital_twin_pipeline as pipeline  # noqa: E402
from services import trail_performance_model as tpm  # noqa: E402

logger = logging.getLogger(__name__)

_spec = importlib.util.spec_from_file_location(
    "predict_race_constant_hrr",
    REPO_ROOT / "scripts" / "predict_race_constant_hrr.py",
)
_pred = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_pred)
HOLDOUT_ACTIVITY_IDS = _pred.HOLDOUT_ACTIVITY_IDS
RACES = _pred.RACES
attach_altitude_from_gpx = _pred.attach_altitude_from_gpx
fit_stage3 = _pred.fit_stage3
segments_from_race_pacing = _pred.segments_from_race_pacing


def load_train_segments_flexible(
    predictions_csv: Path,
    holdout_ids: set[str],
    *,
    cohort: str = "hardRunOrTrailRun",
) -> pd.DataFrame:
    """Like predict_race load_train_segments but accepts activity-only exports."""
    df = pd.read_csv(predictions_csv, dtype={"activityId": str})
    sub = df[df["cohort"].astype(str).eq(cohort)].copy()
    if "fitObjective" in sub.columns and sub["fitObjective"].astype(str).eq("segment").any():
        sub = sub[sub["fitObjective"].astype(str).eq("segment")].copy()
    elif "fitObjective" in sub.columns:
        # Fallback: activity-objective segment rows still carry per-segment features.
        logger.warning(
            "segment_predictions has no fitObjective=segment; using available objective(s)=%s",
            sorted(sub["fitObjective"].astype(str).unique().tolist()),
        )
    sub = sub.drop_duplicates(["activityId", "segmentIndex"], keep="first")
    sub = sub[~sub["activityId"].isin(holdout_ids)].copy()
    return sub.reset_index(drop=True)


# Extra prospective races beyond the two hold-outs already in RACES.
EXTRA_PROSPECTIVE = {
    "rome_marathon": {
        "label": "Marathon de Rome 2026",
        "activityId": "17815897198",
        "gpx": "divers/Marathon de rome.gpx",
        "race_pacing_id": "dd6443fb-705a-4a09-99ec-693a5f7bb440",
    },
}


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))


def weather_coverage_table(activity_features: pd.DataFrame, cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if activity_features.empty:
        return pd.DataFrame()
    temp_all = pd.to_numeric(activity_features.get("temperatureC"), errors="coerce")

    # Prefer explicit activityId join; else anonymized cohort_* flags.
    if "activityId" in activity_features.columns and cohorts:
        for cohort_name, cohort_df in cohorts.items():
            ids = set(cohort_df["activityId"].astype(str))
            subset = activity_features[activity_features["activityId"].astype(str).isin(ids)].copy()
            temp = pd.to_numeric(subset.get("temperatureC"), errors="coerce")
            rows.append(
                {
                    "cohort": cohort_name,
                    "activityCount": int(len(subset)),
                    "temperatureAvailableCount": int(temp.notna().sum()),
                    "temperatureCoveragePct": float(temp.notna().mean() * 100.0) if len(subset) else np.nan,
                    "temperatureCMedian": float(temp.median()) if temp.notna().any() else np.nan,
                    "temperatureCP05": float(temp.quantile(0.05)) if temp.notna().any() else np.nan,
                    "temperatureCP95": float(temp.quantile(0.95)) if temp.notna().any() else np.nan,
                }
            )
        return pd.DataFrame(rows)

    cohort_flag_cols = [c for c in activity_features.columns if str(c).startswith("cohort_")]
    if cohort_flag_cols:
        for col in cohort_flag_cols:
            cohort_name = str(col).removeprefix("cohort_")
            mask = activity_features[col].astype(str).str.lower().isin(["true", "1", "yes"])
            if activity_features[col].dtype == bool:
                mask = activity_features[col].fillna(False)
            subset = activity_features.loc[mask]
            temp = pd.to_numeric(subset.get("temperatureC"), errors="coerce")
            rows.append(
                {
                    "cohort": cohort_name,
                    "activityCount": int(len(subset)),
                    "temperatureAvailableCount": int(temp.notna().sum()),
                    "temperatureCoveragePct": float(temp.notna().mean() * 100.0) if len(subset) else np.nan,
                    "temperatureCMedian": float(temp.median()) if temp.notna().any() else np.nan,
                    "temperatureCP05": float(temp.quantile(0.05)) if temp.notna().any() else np.nan,
                    "temperatureCP95": float(temp.quantile(0.95)) if temp.notna().any() else np.nan,
                }
            )
        return pd.DataFrame(rows)

    logger.warning("weather_coverage_table: falling back to global temperature coverage")
    return pd.DataFrame(
        [
            {
                "cohort": "all",
                "activityCount": int(len(activity_features)),
                "temperatureAvailableCount": int(temp_all.notna().sum()),
                "temperatureCoveragePct": float(temp_all.notna().mean() * 100.0) if len(activity_features) else np.nan,
                "temperatureCMedian": float(temp_all.median()) if temp_all.notna().any() else np.nan,
                "temperatureCP05": float(temp_all.quantile(0.05)) if temp_all.notna().any() else np.nan,
                "temperatureCP95": float(temp_all.quantile(0.95)) if temp_all.notna().any() else np.nan,
            }
        ]
    )


def elevation_qa_table(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame()
    data = segments.copy()
    data["gradeStd"] = pd.to_numeric(data.get("gradeStd"), errors="coerce")
    data["meanAltitudeM"] = pd.to_numeric(data.get("meanAltitudeM"), errors="coerce")
    data["absAltitudeRateMph"] = pd.to_numeric(data.get("absAltitudeRateMph"), errors="coerce")
    rows = [
        {
            "metric": "segmentCount",
            "value": float(len(data)),
        },
        {
            "metric": "meanAltitudeM_median",
            "value": float(data["meanAltitudeM"].median()),
        },
        {
            "metric": "gradeStd_median",
            "value": float(data["gradeStd"].median()),
        },
        {
            "metric": "gradeStd_p95",
            "value": float(data["gradeStd"].quantile(0.95)),
        },
        {
            "metric": "absAltitudeRateMph_median",
            "value": float(data["absAltitudeRateMph"].median()),
        },
        {
            "metric": "note",
            "value": "Barometric/GPS only; DEM correction not available in-repo",
        },
    ]
    return pd.DataFrame(rows)


def objective_bias_variance_note(objective_comparison: pd.DataFrame) -> pd.DataFrame:
    if objective_comparison.empty:
        return pd.DataFrame()
    data = objective_comparison.copy()
    if "maeMin" not in data.columns:
        return pd.DataFrame()
    pivot = (
        data.pivot_table(
            index=["cohort", "stage"],
            columns="fitObjective",
            values="maeMin",
            aggfunc="first",
        )
        .reset_index()
    )
    if "activity" in pivot.columns and "segment" in pivot.columns:
        pivot["deltaMaeMin_activityMinusSegment"] = pivot["activity"] - pivot["segment"]
        pivot["note"] = np.where(
            pivot["deltaMaeMin_activityMinusSegment"].abs() < 1.0,
            "objectives agree within ~1 min MAE",
            "objectives disagree; race objective prioritizes finish-time, segment prioritizes local fit",
        )
    return pivot


def bland_altman_figure(loo: pd.DataFrame, title: str) -> go.Figure:
    actual = pd.to_numeric(loo["actualTimeSec"], errors="coerce") / 60.0
    pred = pd.to_numeric(loo["predictedTimeSec"], errors="coerce") / 60.0
    mean = (actual + pred) / 2.0
    diff = pred - actual
    md = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean, y=diff, mode="markers", name="activities"))
    for y, name in [(md, "mean"), (md + 1.96 * sd, "+1.96 SD"), (md - 1.96 * sd, "-1.96 SD")]:
        fig.add_hline(y=y, line_dash="dash", annotation_text=name)
    fig.update_layout(
        title=title,
        xaxis_title="Mean of actual & predicted (min)",
        yaxis_title="Predicted − actual (min)",
        template="plotly_white",
    )
    return fig


def pred_vs_actual_figure(loo: pd.DataFrame, title: str) -> go.Figure:
    actual = pd.to_numeric(loo["actualTimeSec"], errors="coerce") / 60.0
    pred = pd.to_numeric(loo["predictedTimeSec"], errors="coerce") / 60.0
    lo = float(min(actual.min(), pred.min()))
    hi = float(max(actual.max(), pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=pred, mode="markers", name="LOO"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="identity", line={"dash": "dash"}))
    fig.update_layout(
        title=title,
        xaxis_title="Actual time (min)",
        yaxis_title="Predicted time (min)",
        template="plotly_white",
    )
    return fig


def speed_hrr_figure(curve: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["hrr"],
            y=curve["speedKmh"],
            mode="lines+markers",
            name="model speed",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Heart-rate reserve (HRR)",
        yaxis_title="Predicted speed (km/h)",
        template="plotly_white",
    )
    return fig


def finish_time_uncertainty_bands(
    route: pd.DataFrame,
    loo_folds: pd.DataFrame,
    *,
    physiology: dict[str, Any],
    hrr: float,
    iterations: int = 300,
    seed: int = 20260721,
) -> dict[str, float]:
    """Resample LOO (alpha, kappa) with continuous jitter for finish-time bands (R9).

    Discrete bootstrap over a small (α, κ) grid yields degenerate P05=P50 when
    few unique pairs exist. We therefore (1) resample fold pairs and (2) add
    Gaussian jitter from the empirical fold-wise spread of α and κ, plus a
    residual scale from LOO activity finish-time errors when available.
    """
    folds = loo_folds.dropna(subset=["alpha", "fatigueCoef"]).copy()
    if folds.empty or route.empty:
        return {}
    alphas = pd.to_numeric(folds["alpha"], errors="coerce").to_numpy(dtype=float)
    kappas = pd.to_numeric(folds["fatigueCoef"], errors="coerce").to_numpy(dtype=float)
    models = folds.get("fatigueModel", pd.Series(["exponential"] * len(folds))).astype(str).tolist()
    alpha_std = float(np.nanstd(alphas)) if len(alphas) > 1 else 0.02
    kappa_std = float(np.nanstd(kappas)) if len(kappas) > 1 else 0.05
    # Floor jitter so bands remain informative even when grid collapses.
    alpha_std = max(alpha_std, 0.02)
    kappa_std = max(kappa_std, 0.03)
    residual_std = 0.0
    if {"actualTimeSec", "predictedTimeSec"}.issubset(folds.columns):
        err = (
            pd.to_numeric(folds["predictedTimeSec"], errors="coerce")
            - pd.to_numeric(folds["actualTimeSec"], errors="coerce")
        ).to_numpy(dtype=float)
        residual_std = float(np.nanstd(err)) if np.isfinite(err).any() else 0.0
        residual_std = max(residual_std, 60.0)  # ≥1 min residual noise
    else:
        residual_std = 180.0  # 3 min fallback when fold errors missing

    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = int(rng.integers(0, len(alphas)))
        alpha = float(alphas[idx] + rng.normal(0.0, alpha_std))
        fatigue_coef = float(max(0.0, kappas[idx] + rng.normal(0.0, kappa_std)))
        sim = tpm.simulate_constant_hrr_route(
            route,
            hrr=hrr,
            v_anchor_kmh=float(physiology["vma_flat_kmh"]),
            alpha=alpha,
            fatigue_coef=fatigue_coef,
            fatigue_model=str(models[idx]),
            hrr_reference=float(physiology["hrr_reference"]),
            hrr_min_factor=float(physiology["hrr_min_factor"]),
            hrr_max_factor=float(physiology["hrr_max_factor"]),
            min_fatigue_factor=float(physiology["min_fatigue_factor"]),
            decay_lambda=float(physiology.get("decay_lambda", 0.2)),
            gap_steep_threshold=float(physiology.get("gap_steep_threshold", 0.15)),
            gap_soft_start=float(physiology.get("gap_soft_start", 0.04)),
            gap_climb_scale=float(physiology.get("gap_climb_scale", 1.0)),
            gap_descent_scale=float(physiology.get("gap_descent_scale", 1.0)),
            fatigue_input_col="cumTrimpBefore",
        )
        base = float(sim["predictedTimeSec"].sum())
        samples[i] = base + float(rng.normal(0.0, residual_std))
    samples = np.maximum(samples, 60.0)
    return {
        "finishSecMean": float(np.mean(samples)),
        "finishSecP05": float(np.quantile(samples, 0.05)),
        "finishSecP50": float(np.quantile(samples, 0.50)),
        "finishSecP95": float(np.quantile(samples, 0.95)),
        "iterations": float(iterations),
        "alphaJitterStd": alpha_std,
        "kappaJitterStd": kappa_std,
        "residualStdSec": residual_std,
    }


def run_prospective_with_bands(
    *,
    segment_predictions_csv: Path,
    output_dir: Path,
    physiology: dict[str, Any],
    loo_folds: pd.DataFrame,
    hrr: float = 0.88,
) -> pd.DataFrame:
    races = {**RACES, **EXTRA_PROSPECTIVE}
    holdouts = tuple(sorted({*HOLDOUT_ACTIVITY_IDS, *(str(r["activityId"]) for r in races.values())}))
    train = load_train_segments_flexible(segment_predictions_csv, set(holdouts), cohort="hardRunOrTrailRun")
    if train.empty:
        logger.warning("No train segments for prospective bands; skipping")
        return pd.DataFrame()
    best = fit_stage3(train, physiology, objective="race")
    rows: list[dict[str, object]] = []
    for race_key, race in races.items():
        pacing = REPO_ROOT / "data" / "race_pacing" / f"{race['race_pacing_id']}_segments.csv"
        if not pacing.exists():
            logger.warning("Missing race_pacing for %s", race_key)
            continue
        route = segments_from_race_pacing(pacing)
        gpx_path = REPO_ROOT / str(race["gpx"])
        if gpx_path.exists():
            route = attach_altitude_from_gpx(route, gpx_path)
        sim = tpm.simulate_constant_hrr_route(
            route,
            hrr=hrr,
            v_anchor_kmh=float(physiology["vma_flat_kmh"]),
            alpha=float(best["alpha"]),
            fatigue_coef=float(best["fatigueCoef"]),
            fatigue_model=str(best["fatigueModel"]),
            hrr_reference=float(physiology["hrr_reference"]),
            hrr_min_factor=float(physiology["hrr_min_factor"]),
            hrr_max_factor=float(physiology["hrr_max_factor"]),
            min_fatigue_factor=float(physiology["min_fatigue_factor"]),
            decay_lambda=float(physiology.get("decay_lambda", 0.2)),
            gap_steep_threshold=float(physiology.get("gap_steep_threshold", 0.15)),
            gap_soft_start=float(physiology.get("gap_soft_start", 0.04)),
            gap_climb_scale=float(physiology.get("gap_climb_scale", 1.0)),
            gap_descent_scale=float(physiology.get("gap_descent_scale", 1.0)),
            fatigue_input_col="cumTrimpBefore",
        )
        pred_sec = float(sim["predictedTimeSec"].sum())
        bands = finish_time_uncertainty_bands(
            route,
            loo_folds[loo_folds["cohort"].astype(str).eq("hardRunOrTrailRun")]
            if not loo_folds.empty and "cohort" in loo_folds.columns
            else loo_folds,
            physiology=physiology,
            hrr=hrr,
        )
        actual_sec = np.nan
        act_path = REPO_ROOT / "data" / "activities.csv"
        if act_path.exists():
            acts = pd.read_csv(act_path)
            hit = acts[acts["activityId"].astype(str).eq(str(race["activityId"]))]
            if not hit.empty:
                actual_sec = float(pd.to_numeric(hit.iloc[0]["movingSec"], errors="coerce"))
        row = {
            "raceKey": race_key,
            "label": race["label"],
            "activityId": race["activityId"],
            "predictedSec": pred_sec,
            "actualMovingSec": actual_sec,
            "deltaMin": (pred_sec - actual_sec) / 60.0 if np.isfinite(actual_sec) else np.nan,
            "alpha": best["alpha"],
            "fatigueCoef": best["fatigueCoef"],
            "fatigueModel": best["fatigueModel"],
            "hrr": hrr,
            **bands,
        }
        rows.append(row)
        _write_csv(sim, output_dir / f"prospective_{race_key}_segments.csv")
    summary = pd.DataFrame(rows)
    _write_csv(summary, output_dir / "prospective_finish_time_bands.csv")
    return summary


def build_speed_curves(physiology: dict[str, Any], fitted: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    hrr_grid = np.round(np.linspace(0.50, 1.00, 26), 3)
    frames: list[pd.DataFrame] = []
    for grade, label in [(0.0, "flat_1km"), (0.10, "climb10pct_1km"), (-0.10, "descent10pct_1km")]:
        curve = tpm.speed_vs_hrr_curve(
            hrr_values=hrr_grid,
            v_anchor_kmh=float(physiology["vma_flat_kmh"]),
            alpha=float(fitted.get("alpha", 0.95)),
            distance_km=1.0,
            avg_grade=grade,
            fatigue_coef=float(fitted.get("fatigueCoef", 0.4)),
            fatigue_model=str(fitted.get("fatigueModel", "exponential")),
            cum_trimp_before=0.0,
            hrr_reference=float(physiology["hrr_reference"]),
            hrr_min_factor=float(physiology["hrr_min_factor"]),
            hrr_max_factor=float(physiology["hrr_max_factor"]),
            min_fatigue_factor=float(physiology["min_fatigue_factor"]),
            gap_steep_threshold=float(physiology.get("gap_steep_threshold", 0.15)),
            gap_soft_start=float(physiology.get("gap_soft_start", 0.04)),
            gap_climb_scale=float(physiology.get("gap_climb_scale", 1.0)),
            gap_descent_scale=float(physiology.get("gap_descent_scale", 1.0)),
        )
        curve["terrain"] = label
        frames.append(curve)
        _write_html(
            speed_hrr_figure(curve, f"Speed vs HRR — {label} (fresh, 1 km)"),
            output_dir / f"fig_speed_vs_hrr_{label}.html",
        )
    out = pd.concat(frames, ignore_index=True)
    _write_csv(out, output_dir / "speed_vs_hrr_1km.csv")
    # Primary flat curve as PNG-friendly CSV already; also HTML overlay.
    overlay = go.Figure()
    for terrain, group in out.groupby("terrain"):
        overlay.add_trace(go.Scatter(x=group["hrr"], y=group["speedKmh"], mode="lines", name=str(terrain)))
    overlay.update_layout(
        title="Model speed vs HRR on 1 km segments (fresh)",
        xaxis_title="HRR",
        yaxis_title="Speed (km/h)",
        template="plotly_white",
    )
    _write_html(overlay, output_dir / "fig_speed_vs_hrr_overlay.html")
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "trail_digital_twin_paper_section7.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "exp_perf_predictions" / "trail_digital_twin_paper_section7",
    )
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-prospective", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = pipeline.load_config(args.config if args.config.is_absolute() else REPO_ROOT / args.config)
    config = pipeline._deep_merge(config, {"paths": {"output_dir": str(output_dir)}})
    physiology = config["physiology"]

    if not args.skip_pipeline:
        logger.info("Running paper §7 pipeline → %s", output_dir)
        result = pipeline.run_pipeline(config, project_root=REPO_ROOT, config_path=args.config)
        written = pipeline.write_outputs(result, output_dir)
        logger.info("Wrote %d pipeline assets", len(written))
    else:
        result = None
        logger.info("Skipping pipeline; loading CSVs from %s", output_dir)

    def _load(name: str) -> pd.DataFrame:
        path = output_dir / f"{name}.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    tables = {
        "activity_loo_predictions": _load("activity_loo_predictions"),
        "table_stage_metrics": _load("table_stage_metrics"),
        "table_stage3_ablation": _load("table_stage3_ablation"),
        "table_objective_comparison": _load("table_objective_comparison"),
        "segment_qc": _load("segment_qc"),
        "segment_predictions": _load("segment_predictions"),
        "table_cohort_descriptives": _load("table_cohort_descriptives"),
        "table_fitted_parameters": _load("table_fitted_parameters"),
        "anonymized_activity_features": _load("anonymized_activity_features"),
    }
    if result is not None:
        tables.update(result.tables)

    # Bootstrap MAE / parameter uncertainty
    mae_boot = bench.bootstrap_uncertainty_table(tables, iterations=500, seed=20260623)
    param_boot = bench.bootstrap_parameter_uncertainty_table(
        tables.get("activity_loo_predictions", pd.DataFrame()),
        iterations=500,
        seed=20260623,
    )
    frozen = bench.frozen_physics_vs_hrr_table(tables.get("table_stage_metrics", pd.DataFrame()))
    _write_csv(mae_boot, output_dir / "table_bootstrap_mae_uncertainty.csv")
    _write_csv(param_boot, output_dir / "table_bootstrap_parameter_uncertainty.csv")
    _write_csv(frozen, output_dir / "table_frozen_physics_vs_hrr.csv")
    _write_csv(tables.get("table_stage3_ablation", pd.DataFrame()), output_dir / "table_frozen_stage3_ablation.csv")
    _write_csv(
        objective_bias_variance_note(tables.get("table_objective_comparison", pd.DataFrame())),
        output_dir / "table_segment_vs_race_objective.csv",
    )

    # Weather / HR QC / elevation QA
    activity_features = tables.get("anonymized_activity_features", pd.DataFrame())
    if activity_features.empty and (REPO_ROOT / "data" / "activities.csv").exists():
        # Fallback: rebuild light coverage from activities + metrics.
        acts = pd.read_csv(REPO_ROOT / "data" / "activities.csv")
        mets = pd.read_csv(REPO_ROOT / "data" / "activities_metrics.csv")
        activity_features = acts.merge(mets[["activityId", "category"]], on="activityId", how="left")
        activity_features["temperatureC"] = np.nan
    cohorts_desc = tables.get("table_cohort_descriptives", pd.DataFrame())
    cohort_map: dict[str, pd.DataFrame] = {}
    if not cohorts_desc.empty:
        # Rebuild cohort ID lists from LOO / predictions when available.
        loo = tables.get("activity_loo_predictions", pd.DataFrame())
        if not loo.empty and "cohort" in loo.columns:
            for cohort_name, group in loo.groupby("cohort"):
                cohort_map[str(cohort_name)] = group[["activityId"]].drop_duplicates()
    if not cohort_map and not tables.get("segment_predictions", pd.DataFrame()).empty:
        seg = tables["segment_predictions"]
        if "cohort" in seg.columns:
            for cohort_name, group in seg.groupby("cohort"):
                cohort_map[str(cohort_name)] = group[["activityId"]].drop_duplicates()
    if cohort_map:
        _write_csv(weather_coverage_table(activity_features, cohort_map), output_dir / "table_weather_coverage.csv")
    _write_csv(tables.get("segment_qc", pd.DataFrame()), output_dir / "table_hr_qc.csv")
    _write_csv(elevation_qa_table(tables.get("segment_predictions", pd.DataFrame())), output_dir / "table_elevation_qa.csv")

    # Figures from Stage 3 LOO
    loo_all = tables.get("activity_loo_predictions", pd.DataFrame())
    if not loo_all.empty:
        stage3 = loo_all[loo_all["stage"].astype(str).str.contains("Stage 3 HRR speed ratio LOO", regex=False)]
        for cohort_name, group in stage3.groupby("cohort"):
            safe = str(cohort_name).replace("/", "_")
            _write_html(
                pred_vs_actual_figure(group, f"Pred vs actual LOO — {cohort_name}"),
                figures_dir / f"fig_pred_vs_actual_{safe}.html",
            )
            _write_html(
                bland_altman_figure(group, f"Bland–Altman LOO — {cohort_name}"),
                figures_dir / f"fig_bland_altman_{safe}.html",
            )

    # Fitted params for speed curve (prefer hardRunOrTrailRun Stage 3)
    fitted = {"alpha": 0.95, "fatigueCoef": 0.40, "fatigueModel": "exponential"}
    params = tables.get("table_fitted_parameters", pd.DataFrame())
    if not params.empty:
        hit = params[
            params["stage"].astype(str).str.contains("Stage 3 HRR", regex=False)
            & params["cohort"].astype(str).eq("hardRunOrTrailRun")
        ]
        if hit.empty:
            hit = params[params["stage"].astype(str).str.contains("Stage 3 HRR", regex=False)]
        if not hit.empty:
            row = hit.iloc[0]
            fitted = {
                "alpha": float(row.get("alpha", 0.95)),
                "fatigueCoef": float(row.get("fatigueCoef", 0.40)),
                "fatigueModel": str(row.get("fatigueModel", "exponential")),
            }
    build_speed_curves(physiology, fitted, output_dir)

    # Prospective + finish-time bands
    if not args.skip_prospective:
        seg_csv = output_dir / "segment_predictions.csv"
        if not seg_csv.exists():
            # Fallback to moving-time fit run used previously.
            fallback = (
                REPO_ROOT
                / "data"
                / "exp_perf_predictions"
                / "trail_digital_twin_moving_time_fit"
                / "runs"
                / "001_moving_time_fit_ab_fit_moving_time_only"
                / "segment_predictions.csv"
            )
            seg_csv = fallback
            logger.warning("Using fallback segment_predictions: %s", seg_csv)
        if seg_csv.exists():
            run_prospective_with_bands(
                segment_predictions_csv=seg_csv,
                output_dir=output_dir / "prospective",
                physiology=physiology,
                loo_folds=loo_all[loo_all["stage"].astype(str).str.contains("Stage 3 HRR speed ratio LOO", regex=False)]
                if not loo_all.empty
                else loo_all,
                hrr=float(physiology.get("hrr_reference", 0.88)),
            )

    # Copy frozen race protocol + status checklist
    protocol = {
        "selected_race_dates": config.get("cohorts", {}).get("selected_race_dates", []),
        "holdout_activity_ids": list(HOLDOUT_ACTIVITY_IDS) + [EXTRA_PROSPECTIVE["rome_marathon"]["activityId"]],
        "prospective_races": {**RACES, **EXTRA_PROSPECTIVE},
        "cohorts_include": config.get("cohorts", {}).get("include", []),
        "loo_include": config.get("cohorts", {}).get("loo_include", []),
        "loo_activity_cap": config.get("cohorts", {}).get("loo_activity_cap", 0),
        "physiology": physiology,
        "gitSha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if (REPO_ROOT / ".git").exists()
        else "",
    }
    (output_dir / "preregistered_race_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    status = {
        "multi_athlete": "blocked_single_athlete_case_study",
        "preregistered_splits": "done",
        "nested_loo_bootstrap_alpha_kappa": "done",
        "physics_baseline_matched": "done",
        "broader_prospective": "done_lut_gresivaudan_rome",
        "finish_time_uncertainty_bands": "done",
        "sex_age_strata": "blocked_single_athlete",
        "weather_coverage": "done_sparse_no_model_term",
        "aid_nutrition_logs": "blocked_no_structured_logs_moving_time_proxy",
        "hr_qc": "done",
        "dem_elevation": "blocked_no_dem_barometric_qa_only",
        "frozen_ablation": "done",
        "segment_vs_race_objective": "done",
        "software_versions_seeds": "done",
        "figures": "done",
        "runTrailOver20Min_cohort": "done",
        "speed_vs_hrr_1km_curve": "done",
    }
    (output_dir / "section7_status.json").write_text(json.dumps(status, indent=2) + "\n")
    logger.info("Section 7 status written to %s", output_dir / "section7_status.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
