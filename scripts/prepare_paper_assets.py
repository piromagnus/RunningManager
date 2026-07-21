"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Assemble publication-ready figures (PNG) and scientific tables for the HR
digital-twin paper from section-7 experiment outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

SECTION7 = REPO_ROOT / "data" / "exp_perf_predictions" / "trail_digital_twin_paper_section7"
PAPER_ROOT = REPO_ROOT / "docs" / "science" / "paper"
FIGURES_DIR = PAPER_ROOT / "figures"
TABLES_DIR = PAPER_ROOT / "tables"

COHORT_LABELS = {
    "hardTrailRun": "Hard trail runs",
    "hardRunOrTrailRun": "Hard run or trail runs",
    "runTrailOver20Min": "Run/trail > 20 min",
    "top10HardTrailByHRR": "Top-10 hard trail by HRR",
    "selectedDateRaces": "Selected race dates",
}

STAGE_LOO_MAP = {
    "Stage 0 reproduction Stage 3 LOO": {
        "model_id": "M0",
        "label": "Baseline physics digital twin (GAP, altitude, CTL, progress fatigue)",
        "components": "GAP + altitude + CTL readiness + progress decay",
    },
    "Stage 1 TRIMP fatigue CTL LOO": {
        "model_id": "M1",
        "label": "M0 with acute TRIMP fatigue (CTL readiness retained)",
        "components": "M0 − progress + acute TRIMP",
    },
    "Stage 2 TRIMP fatigue REDI LOO": {
        "model_id": "M2",
        "label": "M1 with REDI readiness replacing CTL",
        "components": "M1 − CTL + REDI",
    },
    "Stage 3 HRR speed ratio LOO": {
        "model_id": "M3",
        "label": "Full HRR + TRIMP digital twin",
        "components": "M2 + continuous HRR effort",
    },
}

ABLATION_LABELS = {
    "full": "Full model (M3 + trail GAP scales)",
    "no HRR speed ratio": "Without HRR effort term",
    "no acute fatigue": "Without acute TRIMP fatigue",
    "no GAP": "Without grade-adjusted pace",
    "no altitude": "Without altitude correction",
    "no REDI readiness": "Without REDI readiness",
    "no trail GAP scales": "Without asymmetric trail GAP scales",
}


def _write_png(fig: go.Figure, path: Path, *, width: int = 900, height: int = 650) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(
        template="plotly_white",
        font={"family": "Times New Roman, Times, serif", "size": 14},
        margin={"l": 60, "r": 30, "t": 60, "b": 55},
    )
    fig.write_image(str(path), format="png", scale=2, width=width, height=height)
    logger.info("Wrote %s", path)


def _fmt_hms(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "—"
    total = int(round(float(seconds)))
    h, rem = divmod(abs(total), 3600)
    m, s = divmod(rem, 60)
    sign = "−" if total < 0 else ""
    return f"{sign}{h:d}:{m:02d}:{s:02d}"


def table_cohort_descriptives(src: Path) -> pd.DataFrame:
    raw = pd.read_csv(src / "table_cohort_descriptives.csv")
    rows = []
    for _, row in raw.iterrows():
        rows.append(
            {
                "Cohort": COHORT_LABELS.get(str(row["cohort"]), row["cohort"]),
                "n activities": int(row["activityCount"]),
                "Median distance (km)": round(float(row["distanceKmMedian"]), 1),
                "Median ascent (m)": round(float(row["ascentMMedian"]), 0),
                "Median duration (min)": round(float(row["durationMinMedian"]), 1),
                "Mean HRR": round(float(row["hrrMean"]), 3),
                "Median HRR": round(float(row["hrrMedian"]), 3),
            }
        )
    return pd.DataFrame(rows)


def table_incremental_models(src: Path) -> pd.DataFrame:
    """Baseline physics twin and successive model additions (LOO, activity objective)."""
    metrics = pd.read_csv(src / "table_stage_metrics.csv")
    loo = metrics[
        metrics["fitObjective"].astype(str).eq("activity")
        & metrics["stage"].astype(str).isin(STAGE_LOO_MAP)
    ].copy()
    rows: list[dict[str, object]] = []
    for cohort, group in loo.groupby("cohort", sort=False):
        baseline = group[group["stage"].eq("Stage 0 reproduction Stage 3 LOO")]
        baseline_mae = float(baseline["maeMin"].iloc[0]) if not baseline.empty else np.nan
        for stage, meta in STAGE_LOO_MAP.items():
            hit = group[group["stage"].eq(stage)]
            if hit.empty:
                continue
            mae = float(hit["maeMin"].iloc[0])
            rows.append(
                {
                    "Cohort": COHORT_LABELS.get(str(cohort), cohort),
                    "Model": meta["model_id"],
                    "Specification": meta["label"],
                    "Components": meta["components"],
                    "MAE (min)": round(mae, 2),
                    "MAPE (%)": round(float(hit["mapePct"].iloc[0]), 2),
                    "Bias (min)": round(float(hit["biasMin"].iloc[0]), 2),
                    "R²": round(float(hit["r2"].iloc[0]), 3),
                    "ΔMAE vs M0 (min)": round(mae - baseline_mae, 2)
                    if np.isfinite(baseline_mae)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def table_component_ablation(src: Path) -> pd.DataFrame:
    abl = pd.read_csv(src / "table_frozen_stage3_ablation.csv")
    abl = abl[abl["fitObjective"].astype(str).eq("activity")].copy()
    rows = []
    for _, row in abl.iterrows():
        stage = str(row["stage"])
        rows.append(
            {
                "Cohort": COHORT_LABELS.get(str(row["cohort"]), row["cohort"]),
                "Variant": ABLATION_LABELS.get(stage, stage),
                "MAE (min)": round(float(row["maeMin"]), 2),
                "MAPE (%)": round(float(row["mapePct"]), 2),
                "Bias (min)": round(float(row["biasMin"]), 2),
                "R²": round(float(row["r2"]), 3),
                "ΔMAE vs full (min)": round(float(row["deltaMaeMinVsFull"]), 2),
                "Protocol": str(row["ablationProtocol"]) if "ablationProtocol" in abl.columns else "frozen",
            }
        )
    return pd.DataFrame(rows)


def table_bootstrap_uncertainty(src: Path) -> pd.DataFrame:
    boot = pd.read_csv(src / "table_bootstrap_mae_uncertainty.csv")
    boot = boot[boot["fitObjective"].astype(str).eq("activity")].copy()
    params = pd.read_csv(src / "table_bootstrap_parameter_uncertainty.csv")
    params = params[params["fitObjective"].astype(str).eq("activity")].copy()
    merged = boot.merge(params, on=["cohort", "fitObjective"], how="left", suffixes=("", "_param"))
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "Cohort": COHORT_LABELS.get(str(row["cohort"]), row["cohort"]),
                "n (LOO folds)": int(row["activityCount"]),
                "MAE (min)": round(float(row["maeMin"]), 2),
                "MAE 90% CI (min)": f"{row['maeMinP05']:.2f}–{row['maeMinP95']:.2f}",
                "Bias (min)": round(float(row["biasMin"]), 2),
                "Bias 90% CI (min)": f"{row['biasMinP05']:.2f}–{row['biasMinP95']:.2f}",
                "α mean": round(float(row.get("alphaMean", np.nan)), 3)
                if pd.notna(row.get("alphaMean", np.nan))
                else "—",
                "α 90% CI": (
                    f"{row['alphaP05']:.3f}–{row['alphaP95']:.3f}"
                    if pd.notna(row.get("alphaP05", np.nan))
                    else "—"
                ),
                "κ mean": round(float(row.get("fatigueCoefMean", np.nan)), 3)
                if pd.notna(row.get("fatigueCoefMean", np.nan))
                else "—",
                "κ 90% CI": (
                    f"{row['fatigueCoefP05']:.3f}–{row['fatigueCoefP95']:.3f}"
                    if pd.notna(row.get("fatigueCoefP05", np.nan))
                    else "—"
                ),
            }
        )
    return pd.DataFrame(rows)


def table_prospective(src: Path) -> pd.DataFrame:
    path = src / "prospective" / "prospective_finish_time_bands.csv"
    raw = pd.read_csv(path)
    rows = []
    for _, row in raw.iterrows():
        profile = str(row["profileSource"]) if "profileSource" in raw.columns else ""
        obs_hrr = (
            round(float(row["observedMeanHrr"]), 2)
            if "observedMeanHrr" in raw.columns and pd.notna(row["observedMeanHrr"])
            else np.nan
        )
        rows.append(
            {
                "Race": row["label"],
                "Profile": profile,
                "Predicted moving time": _fmt_hms(float(row["predictedSec"])),
                "Observed moving time": _fmt_hms(float(row["actualMovingSec"])),
                "Δ (min)": round(float(row["deltaMin"]), 1),
                "Observed mean HRR": obs_hrr,
                "P05 finish": _fmt_hms(float(row["finishSecP05"])),
                "P50 finish": _fmt_hms(float(row["finishSecP50"])),
                "P95 finish": _fmt_hms(float(row["finishSecP95"])),
                "Constant HRR": round(float(row["hrr"]), 2),
                "α": round(float(row["alpha"]), 2),
                "κ": round(float(row["fatigueCoef"]), 2),
            }
        )
    return pd.DataFrame(rows)


def table_speed_hrr_excerpt(src: Path) -> pd.DataFrame:
    curve = pd.read_csv(src / "speed_vs_hrr_1km.csv")
    flat = curve[curve["terrain"].eq("flat_1km")].copy()
    keep_hrr = {0.50, 0.60, 0.70, 0.80, 0.88, 0.95, 1.00}
    flat = flat[flat["hrr"].round(2).isin(keep_hrr)]
    return pd.DataFrame(
        {
            "HRR": flat["hrr"].round(2),
            "Predicted speed (km·h⁻¹)": flat["speedKmh"].round(2),
            "Predicted pace (min·km⁻¹)": flat["paceMinPerKm"].round(2),
            "Segment": "1 km flat, fresh (TRIMP = 0)",
        }
    )


def _qc_exclusion_summary(qc_path: Path, label: str, thresholds: str) -> dict[str, object]:
    qc = pd.read_csv(qc_path)
    n_seg = int(pd.to_numeric(qc["segmentCount"], errors="coerce").fillna(0).sum())
    n_ex = int(pd.to_numeric(qc["excludedSegmentCount"], errors="coerce").fillna(0).sum())
    t_ex = float(pd.to_numeric(qc["excludedTimeSec"], errors="coerce").fillna(0).sum())
    return {
        "Policy": label,
        "Thresholds": thresholds,
        "Segments total": n_seg,
        "Segments rejected": n_ex,
        "Rejected share (%)": round(100.0 * n_ex / n_seg, 2) if n_seg else np.nan,
        "Rejected time (min)": round(t_ex / 60.0, 1),
    }


def table_segment_rejection_summary(src: Path) -> pd.DataFrame:
    """Slight near-flat immobile rejection policies and LOO impact."""
    excl_root = REPO_ROOT / "data" / "exp_perf_predictions" / "trail_digital_twin_segment_exclusion"
    rows: list[dict[str, object]] = []
    run_map = [
        (
            "None (baseline)",
            "exclusion disabled",
            excl_root / "runs" / "000_segment_exclusion_ab_baseline_no_exclusion" / "segment_qc.csv",
            "000_segment_exclusion_ab_baseline_no_exclusion",
        ),
        (
            "Slight (paper default)",
            "speedEq < 3 km·h⁻¹ or stationary share > 0.40; |Δelev|/h ≤ 120 m·h⁻¹",
            excl_root / "runs" / "001_segment_exclusion_ab_exclude_speed_eq_3kmh_share_040" / "segment_qc.csv",
            "001_segment_exclusion_ab_exclude_speed_eq_3kmh_share_040",
        ),
        (
            "Moderate",
            "speedEq < 4 km·h⁻¹ or stationary share > 0.30; |Δelev|/h ≤ 120 m·h⁻¹",
            excl_root / "runs" / "002_segment_exclusion_ab_exclude_speed_eq_4kmh_share_030" / "segment_qc.csv",
            "002_segment_exclusion_ab_exclude_speed_eq_4kmh_share_030",
        ),
    ]
    # Prefer paper §7 QC for the slight policy actually used in the manuscript pipeline
    # (moving-time fit + slight exclusion → very few residual rejects).
    paper_qc = src / "segment_qc.csv"
    leaderboard = (
        pd.read_csv(excl_root / "benchmark_leaderboard.csv")
        if (excl_root / "benchmark_leaderboard.csv").exists()
        else pd.DataFrame()
    )
    mae_by_run = {}
    if not leaderboard.empty and "runId" in leaderboard.columns:
        for _, row in leaderboard.iterrows():
            mae_by_run[str(row["runId"])] = float(row["meanStage3MaeMin"])

    for label, thresholds, qc_path, run_id in run_map:
        if not qc_path.exists():
            continue
        row = _qc_exclusion_summary(qc_path, label, thresholds)
        row["Mean Stage-3 LOO MAE (min)"] = (
            round(mae_by_run[run_id], 2) if run_id in mae_by_run else np.nan
        )
        rows.append(row)

    if paper_qc.exists():
        paper_row = _qc_exclusion_summary(
            paper_qc,
            "Slight + moving-time fit (paper §7)",
            "speedEq < 3 km·h⁻¹ or share > 0.40; fit on moving time",
        )
        # Paper §7 LOO MAE on hardRunOrTrailRun from stage metrics if available.
        stage = src / "table_stage_metrics.csv"
        if stage.exists():
            m = pd.read_csv(stage)
            hit = m[
                m["stage"].astype(str).eq("Stage 3 HRR speed ratio LOO")
                & m["cohort"].astype(str).eq("hardRunOrTrailRun")
                & m["fitObjective"].astype(str).eq("activity")
            ]
            if not hit.empty:
                paper_row["Mean Stage-3 LOO MAE (min)"] = round(float(hit.iloc[0]["maeMin"]), 2)
        rows.append(paper_row)
    return pd.DataFrame(rows)


def table_segment_rejection_examples(src: Path) -> pd.DataFrame:
    path = (
        REPO_ROOT
        / "data"
        / "exp_perf_predictions"
        / "trail_digital_twin_segment_exclusion"
        / "rejected_near_flat_immobile_segments.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    raw = raw.sort_values("actualTimeSec", ascending=False).head(8)
    return pd.DataFrame(
        {
            "Activity": raw["name"].astype(str),
            "km": raw["startKm"].round(1),
            "SpeedEq (km·h⁻¹)": pd.to_numeric(raw["meanSpeedEqKmh"], errors="coerce").round(2),
            "|Δelev|/h (m·h⁻¹)": pd.to_numeric(raw["absAltitudeRateMph"], errors="coerce").round(1),
            "Stationary share": pd.to_numeric(raw["stationaryTimeShare"], errors="coerce").round(2),
            "Duration (min)": (pd.to_numeric(raw["actualTimeSec"], errors="coerce") / 60.0).round(1),
            "Reason": raw["exclusionReason"]
            .astype(str)
            .str.replace("near_flat_altitude_time|", "", regex=False)
            .str.replace("|", " + ", regex=False),
        }
    )


def table_segment_gap_optimisation() -> pd.DataFrame:
    path = (
        REPO_ROOT
        / "data"
        / "exp_perf_predictions"
        / "trail_digital_twin_steep_gap"
        / "steep_gap_baseline_vs_winner_terrain.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    order = ["flat", "climb", "steep_climb", "descent", "steep_descent", "mixed_climb_descent"]
    raw["terrainFamily"] = pd.Categorical(raw["terrainFamily"], categories=order, ordered=True)
    raw = raw.sort_values("terrainFamily")
    labels = {
        "flat": "Flat",
        "climb": "Climb",
        "steep_climb": "Steep climb",
        "descent": "Descent",
        "steep_descent": "Steep descent",
        "mixed_climb_descent": "Mixed",
    }
    return pd.DataFrame(
        {
            "Terrain": raw["terrainFamily"].astype(str).map(labels).fillna(raw["terrainFamily"]),
            "n segments": raw["segmentCount_base"].astype(int),
            "MAE before (min)": raw["maeMin_base"].round(2),
            "Bias before (min)": raw["biasMin_base"].round(2),
            "MAE after (min)": raw["maeMin_win"].round(2),
            "Bias after (min)": raw["biasMin_win"].round(2),
            "ΔMAE (min)": (raw["maeMin_win"] - raw["maeMin_base"]).round(2),
        }
    )


def table_segment_vs_race_objective(src: Path) -> pd.DataFrame:
    path = src / "table_segment_vs_race_objective.csv"
    if not path.exists():
        path = src / "table_objective_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    if "activity" in raw.columns and "segment" in raw.columns:
        data = raw[raw["stage"].astype(str).str.contains("LOO", regex=False)].copy()
        rows = []
        for _, row in data.iterrows():
            rows.append(
                {
                    "Cohort": COHORT_LABELS.get(str(row["cohort"]), row["cohort"]),
                    "Stage": "Stage 3 LOO",
                    "MAE activity objective (min)": round(float(row["activity"]), 2),
                    "MAE segment objective (min)": round(float(row["segment"]), 2),
                    "Δ (activity − segment) (min)": round(float(row["deltaMaeMin_activityMinusSegment"]), 2),
                    "Interpretation": str(row.get("note", "")),
                }
            )
        return pd.DataFrame(rows)
    # Fallback from objective comparison
    data = raw[
        raw["stage"].astype(str).eq("Stage 3 HRR speed ratio LOO")
        & raw["fitObjective"].isin(["activity", "segment"])
    ].copy()
    pivot = data.pivot_table(index="cohort", columns="fitObjective", values="maeMin", aggfunc="first")
    rows = []
    for cohort, row in pivot.iterrows():
        act = float(row.get("activity", np.nan))
        seg = float(row.get("segment", np.nan))
        rows.append(
            {
                "Cohort": COHORT_LABELS.get(str(cohort), cohort),
                "Stage": "Stage 3 LOO",
                "MAE activity objective (min)": round(act, 2) if np.isfinite(act) else np.nan,
                "MAE segment objective (min)": round(seg, 2) if np.isfinite(seg) else np.nan,
                "Δ (activity − segment) (min)": round(act - seg, 2)
                if np.isfinite(act) and np.isfinite(seg)
                else np.nan,
                "Interpretation": "",
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, caption: str, notes: str = "") -> str:
    lines = [f"**{caption}**", ""]
    if df.empty:
        lines.append("_No data._")
        return "\n".join(lines) + "\n"
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    if notes:
        lines.extend(["", f"*{notes}*"])
    return "\n".join(lines) + "\n"


def write_tables(src: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "table01_cohort_descriptives": (
            table_cohort_descriptives(src),
            "Table 1. Descriptive statistics of evaluation cohorts.",
            "HRR denotes mean activity heart-rate reserve. Durations use activity moving time.",
        ),
        "table02_incremental_model_loo": (
            table_incremental_models(src),
            "Table 2. Leave-one-out errors for the baseline physics twin and successive additions.",
            "M0 is the physics-informed baseline (grade-adjusted pace, altitude, CTL readiness, "
            "progress fatigue). M1 replaces progress fatigue with acute TRIMP; M2 replaces CTL with "
            "REDI; M3 adds continuous HRR effort. Metrics use the activity fit objective.",
        ),
        "table03_component_ablation": (
            table_component_ablation(src),
            "Table 3. Component ablation with re-optimized (α, κ) per removal (LOO, activity objective).",
            "Positive ΔMAE indicates degraded accuracy after removing the component and re-fitting. "
            "Protocol column reports reoptimize_loo (preferred) vs legacy frozen. "
            "Trail GAP scales refer to asymmetric soft-ramped climb/descent corrections.",
        ),
        "table04_bootstrap_uncertainty": (
            table_bootstrap_uncertainty(src),
            "Table 4. Bootstrap uncertainty for Stage-3 LOO MAE/bias and fold-wise (α, κ).",
            "Percentile intervals are empirical 5th–95th percentiles over 500 resamples of LOO folds.",
        ),
        "table05_prospective_predictions": (
            table_prospective(src),
            "Table 5. Prospective constant-HRR race predictions with finish-time uncertainty bands.",
            "Hold-out races were excluded from parameter estimation. Profiles are planned "
            "race_pacing+GPX altitude or executed activity GPS geometry (activity_timeseries). "
            "Predictions use constant HRR = 0.88. Δ is predicted − observed moving time; "
            "large negative Δ with submaximal observed mean HRR is an upper-bound hard-effort envelope.",
        ),
        "table06_speed_vs_hrr_flat": (
            table_speed_hrr_excerpt(src),
            "Table 6. Model-implied speed–HRR response on a synthetic 1 km flat segment.",
            "Fresh condition (cumulative TRIMP = 0). Effort saturates at HRR_ref under hrr_max_factor = 1.0.",
        ),
        "table07_segment_rejection_summary": (
            table_segment_rejection_summary(src),
            "Table 7. Slight near-flat immobile segment rejection policies and LOO impact.",
            "Rejection requires altitude–time flatness (|Δelev|/h ≤ 120 m·h⁻¹) and immobility "
            "(low grade-adjusted speed or high stationary share). Rejected segments are withheld "
            "from parameter fitting but retained for full-race evaluation. The paper §7 pipeline "
            "combines slight exclusion with moving-time fitting, leaving only a few residual rejects.",
        ),
        "table07b_segment_rejection_examples": (
            table_segment_rejection_examples(src),
            "Table 7b. Illustrative rejected near-flat immobile segments.",
            "Durations are segment clock times dominated by dwell (aid stations, traffic, device open).",
        ),
        "table08_segment_gap_optimisation": (
            table_segment_gap_optimisation(),
            "Table 8. Segment-level terrain optimisation via asymmetric trail GAP scales.",
            "Before/after soft-ramped climb scale 0.85 and descent scale 1.60 (hardTrailRun, "
            "segment objective, moving-time residuals). Positive bias = model too slow.",
        ),
        "table09_segment_vs_race_objective": (
            table_segment_vs_race_objective(src),
            "Table 9. Segment versus activity (race) fit-objective optimisation of (α, κ).",
            "Both objectives use the same Stage-3 family; only the LOO scoring target differs. "
            "Large positive Δ indicates finish-time calibration is worse than local segment fit.",
        ),
    }
    written: dict[str, Path] = {}
    index_lines = [
        "# Paper tables",
        "",
        "Publication-formatted extracts from the section-7 digital-twin experiments.",
        "",
    ]
    for stem, (df, caption, notes) in tables.items():
        csv_path = out_dir / f"{stem}.csv"
        md_path = out_dir / f"{stem}.md"
        df.to_csv(csv_path, index=False)
        md_path.write_text(_markdown_table(df, caption, notes))
        written[stem] = csv_path
        index_lines.append(f"- [{caption}]({stem}.md) ([CSV]({stem}.csv))")
        logger.info("Wrote table %s (%d rows)", stem, len(df))
    (out_dir / "README.md").write_text("\n".join(index_lines) + "\n")
    return written


def fig_pred_vs_actual(loo: pd.DataFrame, cohort: str) -> go.Figure:
    group = loo[loo["cohort"].astype(str).eq(cohort)].copy()
    actual = pd.to_numeric(group["actualTimeSec"], errors="coerce") / 60.0
    pred = pd.to_numeric(group["predictedTimeSec"], errors="coerce") / 60.0
    lo = float(min(actual.min(), pred.min()))
    hi = float(max(actual.max(), pred.max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=pred,
            mode="markers",
            name="Leave-one-out folds",
            marker={"size": 8, "opacity": 0.75},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Identity",
            line={"dash": "dash", "color": "black"},
        )
    )
    fig.update_layout(
        title=f"Predicted versus observed finish time — {COHORT_LABELS.get(cohort, cohort)}",
        xaxis_title="Observed time (min)",
        yaxis_title="Predicted time (min)",
        legend={"orientation": "h", "y": 1.08},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def fig_bland_altman(loo: pd.DataFrame, cohort: str) -> go.Figure:
    group = loo[loo["cohort"].astype(str).eq(cohort)].copy()
    actual = pd.to_numeric(group["actualTimeSec"], errors="coerce") / 60.0
    pred = pd.to_numeric(group["predictedTimeSec"], errors="coerce") / 60.0
    mean = (actual + pred) / 2.0
    diff = pred - actual
    md = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean, y=diff, mode="markers", name="Activities", marker={"size": 8}))
    for y, name, dash in [
        (md, f"Mean bias ({md:.1f} min)", "solid"),
        (md + 1.96 * sd, "+1.96 SD", "dash"),
        (md - 1.96 * sd, "−1.96 SD", "dash"),
    ]:
        fig.add_hline(y=y, line_dash=dash, annotation_text=name, annotation_position="top left")
    fig.update_layout(
        title=f"Bland–Altman agreement — {COHORT_LABELS.get(cohort, cohort)}",
        xaxis_title="Mean of predicted and observed time (min)",
        yaxis_title="Predicted − observed (min)",
    )
    return fig


def fig_speed_vs_hrr(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    labels = {
        "flat_1km": "Flat (0%)",
        "climb10pct_1km": "Climb (+10%)",
        "descent10pct_1km": "Descent (−10%)",
    }
    for terrain, group in curve.groupby("terrain"):
        fig.add_trace(
            go.Scatter(
                x=group["hrr"],
                y=group["speedKmh"],
                mode="lines",
                name=labels.get(str(terrain), str(terrain)),
                line={"width": 2.5},
            )
        )
    fig.update_layout(
        title="Model-implied speed as a function of heart-rate reserve (1 km segments, fresh)",
        xaxis_title="Heart-rate reserve (HRR)",
        yaxis_title="Predicted ground speed (km·h⁻¹)",
        legend={"orientation": "h", "y": 1.1},
    )
    return fig


def fig_incremental_mae(incremental: pd.DataFrame) -> go.Figure:
    focus = incremental[
        incremental["Cohort"].isin(
            [
                COHORT_LABELS["hardRunOrTrailRun"],
                COHORT_LABELS["hardTrailRun"],
                COHORT_LABELS["runTrailOver20Min"],
                COHORT_LABELS["selectedDateRaces"],
            ]
        )
    ].copy()
    fig = go.Figure()
    for cohort, group in focus.groupby("Cohort"):
        fig.add_trace(
            go.Scatter(
                x=group["Model"],
                y=group["MAE (min)"],
                mode="lines+markers",
                name=str(cohort),
                line={"width": 2.5},
                marker={"size": 9},
            )
        )
    fig.update_layout(
        title="Leave-one-out MAE across successive model additions",
        xaxis_title="Model specification",
        yaxis_title="MAE (min)",
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def fig_ablation_delta(ablation: pd.DataFrame) -> go.Figure:
    focus_cohorts = [
        COHORT_LABELS["hardRunOrTrailRun"],
        COHORT_LABELS["hardTrailRun"],
        COHORT_LABELS["runTrailOver20Min"],
    ]
    data = ablation[
        ablation["Cohort"].isin(focus_cohorts) & ~ablation["Variant"].str.startswith("Full")
    ].copy()
    order = [
        "Without HRR effort term",
        "Without acute TRIMP fatigue",
        "Without grade-adjusted pace",
        "Without asymmetric trail GAP scales",
        "Without altitude correction",
        "Without REDI readiness",
    ]
    fig = go.Figure()
    for cohort in focus_cohorts:
        group = data[data["Cohort"].eq(cohort)].set_index("Variant").reindex(order).dropna(subset=["ΔMAE vs full (min)"])
        fig.add_trace(
            go.Bar(
                x=group.index.tolist(),
                y=group["ΔMAE vs full (min)"].tolist(),
                name=cohort,
            )
        )
    fig.update_layout(
        barmode="group",
        title="Increase in MAE after removing individual model components",
        xaxis_title="Removed component",
        yaxis_title="ΔMAE versus full model (min)",
        legend={"orientation": "h", "y": 1.12},
    )
    fig.update_xaxes(tickangle=-25)
    return fig


def fig_segment_rejection_policies(summary: pd.DataFrame) -> go.Figure:
    data = summary.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=data["Policy"],
            y=data["Rejected share (%)"],
            name="Rejected segment share (%)",
            marker_color="#4C78A8",
        )
    )
    if "Mean Stage-3 LOO MAE (min)" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data["Policy"],
                y=data["Mean Stage-3 LOO MAE (min)"],
                mode="lines+markers",
                name="Mean Stage-3 LOO MAE (min)",
                yaxis="y2",
                line={"width": 2.5, "color": "#F58518"},
                marker={"size": 9},
            )
        )
    fig.update_layout(
        title="Near-flat immobile segment rejection: extent versus LOO error",
        xaxis_title="Rejection policy",
        yaxis_title="Rejected segments (%)",
        yaxis2={
            "title": "Mean Stage-3 LOO MAE (min)",
            "overlaying": "y",
            "side": "right",
        },
        legend={"orientation": "h", "y": 1.14},
        barmode="group",
    )
    fig.update_xaxes(tickangle=-20)
    return fig


def fig_steep_gap_optimisation(gap_table: pd.DataFrame) -> go.Figure:
    if gap_table.empty:
        return go.Figure()
    terrains = gap_table["Terrain"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="MAE before", x=terrains, y=gap_table["MAE before (min)"]))
    fig.add_trace(go.Bar(name="MAE after", x=terrains, y=gap_table["MAE after (min)"]))
    fig.update_layout(
        barmode="group",
        title="Segment terrain residuals before and after trail GAP scale optimisation",
        xaxis_title="Terrain family",
        yaxis_title="Segment MAE (min)",
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def fig_steep_gap_bias(gap_table: pd.DataFrame) -> go.Figure:
    if gap_table.empty:
        return go.Figure()
    focus = gap_table[gap_table["Terrain"].isin(["Steep climb", "Steep descent", "Flat"])]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Bias before", x=focus["Terrain"], y=focus["Bias before (min)"])
    )
    fig.add_trace(
        go.Bar(name="Bias after", x=focus["Terrain"], y=focus["Bias after (min)"])
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="black")
    fig.update_layout(
        barmode="group",
        title="Segment bias correction on steep terrain after GAP scale optimisation",
        xaxis_title="Terrain family",
        yaxis_title="Bias (predicted − observed, min)",
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def fig_segment_vs_race_objective(obj_table: pd.DataFrame) -> go.Figure:
    if obj_table.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Activity (race) objective",
            x=obj_table["Cohort"],
            y=obj_table["MAE activity objective (min)"],
        )
    )
    fig.add_trace(
        go.Bar(
            name="Segment objective",
            x=obj_table["Cohort"],
            y=obj_table["MAE segment objective (min)"],
        )
    )
    fig.update_layout(
        barmode="group",
        title="Leave-one-out MAE under segment versus activity fit objectives",
        xaxis_title="Cohort",
        yaxis_title="MAE (min)",
        legend={"orientation": "h", "y": 1.12},
    )
    fig.update_xaxes(tickangle=-15)
    return fig


def write_figures(src: Path, out_dir: Path, incremental: pd.DataFrame, ablation: pd.DataFrame) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    loo = pd.read_csv(src / "activity_loo_predictions.csv")
    loo = loo[loo["stage"].astype(str).str.contains("Stage 3 HRR speed ratio LOO", regex=False)]
    cohorts = [
        "hardRunOrTrailRun",
        "hardTrailRun",
        "runTrailOver20Min",
        "selectedDateRaces",
    ]
    for cohort in cohorts:
        if cohort not in set(loo["cohort"].astype(str)):
            continue
        path = out_dir / f"fig_pred_vs_actual_{cohort}.png"
        _write_png(fig_pred_vs_actual(loo, cohort), path)
        written[path.name] = path
        path_ba = out_dir / f"fig_bland_altman_{cohort}.png"
        _write_png(fig_bland_altman(loo, cohort), path_ba)
        written[path_ba.name] = path_ba

    curve = pd.read_csv(src / "speed_vs_hrr_1km.csv")
    path_speed = out_dir / "fig_speed_vs_hrr.png"
    _write_png(fig_speed_vs_hrr(curve), path_speed, width=950, height=620)
    written[path_speed.name] = path_speed

    # Per-terrain companion panels (same data, scientific titles).
    for terrain, label in [
        ("flat_1km", "flat"),
        ("climb10pct_1km", "climb10pct"),
        ("descent10pct_1km", "descent10pct"),
    ]:
        subset = curve[curve["terrain"].eq(terrain)]
        fig = go.Figure(
            go.Scatter(x=subset["hrr"], y=subset["speedKmh"], mode="lines+markers", line={"width": 2.5})
        )
        fig.update_layout(
            title=f"Speed–HRR response on a 1 km {label} segment",
            xaxis_title="Heart-rate reserve (HRR)",
            yaxis_title="Predicted ground speed (km·h⁻¹)",
        )
        path = out_dir / f"fig_speed_vs_hrr_{label}.png"
        _write_png(fig, path, width=800, height=560)
        written[path.name] = path

    path_inc = out_dir / "fig_incremental_model_mae.png"
    _write_png(fig_incremental_mae(incremental), path_inc, width=950, height=620)
    written[path_inc.name] = path_inc

    path_abl = out_dir / "fig_component_ablation_delta_mae.png"
    _write_png(fig_ablation_delta(ablation), path_abl, width=1050, height=640)
    written[path_abl.name] = path_abl

    rejection = table_segment_rejection_summary(src)
    if not rejection.empty:
        path_rej = out_dir / "fig_segment_rejection_policies.png"
        _write_png(fig_segment_rejection_policies(rejection), path_rej, width=1000, height=620)
        written[path_rej.name] = path_rej

    gap_table = table_segment_gap_optimisation()
    if not gap_table.empty:
        path_gap = out_dir / "fig_segment_gap_optimisation_mae.png"
        _write_png(fig_steep_gap_optimisation(gap_table), path_gap, width=1000, height=620)
        written[path_gap.name] = path_gap
        path_bias = out_dir / "fig_segment_gap_optimisation_bias.png"
        _write_png(fig_steep_gap_bias(gap_table), path_bias, width=900, height=600)
        written[path_bias.name] = path_bias

    obj_table = table_segment_vs_race_objective(src)
    if not obj_table.empty:
        path_obj = out_dir / "fig_segment_vs_race_objective.png"
        _write_png(fig_segment_vs_race_objective(obj_table), path_obj, width=1000, height=620)
        written[path_obj.name] = path_obj

    # Copy prior paper_assets PNGs that remain relevant.
    legacy = REPO_ROOT / "docs" / "science" / "paper_assets"
    for name in [
        "fig_stage_predicted_vs_actual.png",
        "fig_stage3_ablation.png",
        "fig_stage3_fatigue_state_comparison.png",
        "fig_model_stage_flow.png",
        "fig_loo_residuals_drivers.png",
        "fig_hrr_speed_residual.png",
        "fig_robustness_heatmaps.png",
        "fig_regression_coefficients.png",
    ]:
        src_png = legacy / name
        if src_png.exists():
            dest = out_dir / f"legacy_{name}"
            shutil.copy2(src_png, dest)
            written[dest.name] = dest
    return written


def write_captions(figures: dict[str, Path], out_dir: Path) -> Path:
    captions = {
        "fig_pred_vs_actual_hardRunOrTrailRun.png": (
            "Figure. Leave-one-out predicted versus observed activity times for the hard run/trail "
            "cohort under the full HRR+TRIMP model. The dashed line denotes perfect agreement."
        ),
        "fig_pred_vs_actual_hardTrailRun.png": (
            "Figure. Leave-one-out predicted versus observed times for hard trail runs."
        ),
        "fig_pred_vs_actual_runTrailOver20Min.png": (
            "Figure. Leave-one-out predicted versus observed times for all run/trail activities "
            "longer than 20 min (LOO subsampled to 80 activities)."
        ),
        "fig_pred_vs_actual_selectedDateRaces.png": (
            "Figure. Leave-one-out predicted versus observed times for the selected race-date cohort."
        ),
        "fig_bland_altman_hardRunOrTrailRun.png": (
            "Figure. Bland–Altman plot of Stage-3 leave-one-out residuals for hard run/trail activities. "
            "Horizontal lines show mean bias and approximate 95% limits of agreement."
        ),
        "fig_bland_altman_hardTrailRun.png": (
            "Figure. Bland–Altman agreement for hard trail runs."
        ),
        "fig_bland_altman_runTrailOver20Min.png": (
            "Figure. Bland–Altman agreement for the run/trail > 20 min cohort."
        ),
        "fig_bland_altman_selectedDateRaces.png": (
            "Figure. Bland–Altman agreement for selected race dates."
        ),
        "fig_speed_vs_hrr.png": (
            "Figure. Model-implied ground speed as a function of heart-rate reserve on synthetic "
            "1 km segments (flat, +10% climb, −10% descent) under fresh conditions (TRIMP = 0)."
        ),
        "fig_incremental_model_mae.png": (
            "Figure. Leave-one-out mean absolute error across successive model additions "
            "(M0 physics baseline → M1 acute TRIMP → M2 REDI → M3 HRR effort)."
        ),
        "fig_component_ablation_delta_mae.png": (
            "Figure. Increase in leave-one-out mean absolute error after removing individual "
            "components and re-optimizing (α, κ) (positive values indicate loss of accuracy)."
        ),
        "fig_segment_rejection_policies.png": (
            "Figure. Extent of near-flat immobile segment rejection (bars) and associated mean "
            "Stage-3 leave-one-out MAE (line) across exclusion policies. The paper pipeline uses "
            "a slight policy combined with moving-time fitting."
        ),
        "fig_segment_gap_optimisation_mae.png": (
            "Figure. Segment-level MAE by terrain family before and after soft-ramped asymmetric "
            "trail GAP scale optimisation (climb 0.85, descent 1.60)."
        ),
        "fig_segment_gap_optimisation_bias.png": (
            "Figure. Segment bias on flat and steep terrain before and after trail GAP scale "
            "optimisation. Soft-ramped scales remove the opposing climb/descent bias pattern."
        ),
        "fig_segment_vs_race_objective.png": (
            "Figure. Leave-one-out MAE when (α, κ) are optimised under a segment residual objective "
            "versus an activity finish-time objective."
        ),
    }
    path = out_dir / "figure_captions.md"
    lines = ["# Figure captions", ""]
    for name, caption in captions.items():
        if name in figures:
            lines.append(f"## `{name}`")
            lines.append("")
            lines.append(caption)
            lines.append("")
    path.write_text("\n".join(lines))
    return path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section7-dir", type=Path, default=SECTION7)
    parser.add_argument("--paper-root", type=Path, default=PAPER_ROOT)
    args = parser.parse_args()

    src = args.section7_dir if args.section7_dir.is_absolute() else REPO_ROOT / args.section7_dir
    paper_root = args.paper_root if args.paper_root.is_absolute() else REPO_ROOT / args.paper_root
    figures_dir = paper_root / "figures"
    tables_dir = paper_root / "tables"

    if not src.exists():
        raise SystemExit(f"section7 output dir not found: {src}")

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    tables = write_tables(src, tables_dir)
    incremental = pd.read_csv(tables_dir / "table02_incremental_model_loo.csv")
    ablation = pd.read_csv(tables_dir / "table03_component_ablation.csv")
    figures = write_figures(src, figures_dir, incremental, ablation)
    write_captions(figures, figures_dir)

    # README needs paper_root context; write inline here.
    readme_path = paper_root / "README.md"
    lines = [
        "# Paper assets — HR digital twin for trail-running performance",
        "",
        "Publication-oriented figures and tables derived from the section-7 experiments.",
        "Tone and labelling follow the working manuscript "
        "`docs/science/trail_digital_twin_hr_performance_paper_draft.md`.",
        "",
        "## Regenerating",
        "",
        "```bash",
        "uv run python scripts/prepare_paper_assets.py",
        "```",
        "",
        "## Figures",
        "",
        "PNG exports (2× scale) are stored under `figures/`. Scientific captions: "
        "`figures/figure_captions.md`.",
        "",
    ]
    for name in sorted(figures):
        lines.append(f"- `{name}`")
    lines.extend(["", "## Tables", "", "CSV + Markdown under `tables/`.", ""])
    for stem in sorted(tables):
        lines.append(f"- `{stem}.csv` / `{stem}.md`")
    lines.extend(
        [
            "",
            "## Model ladder (for Table 2)",
            "",
            "| ID | Addition relative to previous |",
            "|----|-------------------------------|",
            "| M0 | Baseline physics twin (GAP, altitude, CTL, progress fatigue) |",
            "| M1 | + acute TRIMP fatigue (replaces progress decay) |",
            "| M2 | + REDI readiness (replaces CTL) |",
            "| M3 | + continuous HRR effort term |",
            "| Full | M3 + asymmetric trail GAP soft-ramp scales |",
            "",
        ]
    )
    readme_path.write_text("\n".join(lines) + "\n")

    html_dir = figures_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["fig_*.html", "figures/*.html"]:
        for html in src.glob(pattern):
            shutil.copy2(html, html_dir / html.name)

    manifest = {
        "section7Dir": str(src),
        "figures": sorted(figures.keys()),
        "tables": sorted(tables.keys()),
    }
    (paper_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Paper assets ready under %s", paper_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
