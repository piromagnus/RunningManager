"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Focused steep GAP scale grid on hardTrailRun moving-time segments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from services import trail_performance_model as tpm


def _load_hard_trail_segments(predictions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv, dtype={"activityId": str})
    sub = df[(df["cohort"] == "hardTrailRun") & (df["fitObjective"] == "segment")].copy()
    sub = sub.drop_duplicates(["activityId", "segmentIndex"], keep="first")
    if "terrainFamily" not in sub.columns:
        sub["terrainFamily"] = pd.to_numeric(sub["avgGrade"], errors="coerce").map(tpm.terrain_family)
    return sub.reset_index(drop=True)


def _terrain_mae(segments: pd.DataFrame, predicted: pd.Series, actual_col: str) -> pd.DataFrame:
    work = segments.copy()
    work["predictedTimeSec"] = predicted.to_numpy(dtype=float)
    work["actualFitSec"] = pd.to_numeric(work[actual_col], errors="coerce")
    work = work[work["actualFitSec"].gt(1.0) & np.isfinite(work["predictedTimeSec"])]
    rows = []
    for terrain, group in work.groupby("terrainFamily", sort=False):
        metrics = tpm.regression_metrics(group["actualFitSec"], group["predictedTimeSec"])
        rows.append(
            {
                "terrainFamily": terrain,
                "segmentCount": int(len(group)),
                "maeMin": metrics["maeSec"] / 60.0,
                "biasMin": metrics["biasSec"] / 60.0,
                "mapePct": metrics["mapePct"],
                "r2": metrics["r2"],
            }
        )
    return pd.DataFrame(rows).sort_values("maeMin", ascending=False)


def _score_row(terrain_metrics: pd.DataFrame) -> dict[str, float]:
    by = terrain_metrics.set_index("terrainFamily")
    def get(name: str, col: str) -> float:
        return float(by.loc[name, col]) if name in by.index else float("nan")

    steep_mae = np.nanmean([get("steep_climb", "maeMin"), get("steep_descent", "maeMin")])
    return {
        "steepMaeMin": float(steep_mae),
        "steepClimbMaeMin": get("steep_climb", "maeMin"),
        "steepDescentMaeMin": get("steep_descent", "maeMin"),
        "flatMaeMin": get("flat", "maeMin"),
        "climbMaeMin": get("climb", "maeMin"),
        "descentMaeMin": get("descent", "maeMin"),
        "mixedMaeMin": get("mixed_climb_descent", "maeMin"),
        "steepClimbBiasMin": get("steep_climb", "biasMin"),
        "steepDescentBiasMin": get("steep_descent", "biasMin"),
        "flatBiasMin": get("flat", "biasMin"),
    }


def run_grid(
    segments: pd.DataFrame,
    *,
    climb_scales: list[float],
    descent_scales: list[float],
    steep_threshold: float,
    actual_col: str,
    flat_guardrail_mae_min: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    for climb_scale in climb_scales:
        for descent_scale in descent_scales:
            best, _grid, prediction = tpm.hrr_trimp_grid_search_model(
                segments,
                v_anchor_kmh=18.0,
                alpha_grid=[0.85, 0.90, 0.95, 1.00, 1.05],
                fatigue_coef_grid=[0.2, 0.3, 0.4, 0.6],
                secondary_fatigue_coef_grid=[0.0, 0.1, 0.2],
                fatigue_models=("linear", "exponential"),
                objective="segment",
                actual_time_col=actual_col,
                hrr_reference=0.88,
                hrr_min_factor=0.30,
                hrr_max_factor=1.0,
                min_fatigue_factor=0.60,
                load_factor_col="rediReadinessFactor",
                use_hrr_effort=True,
                acute_trimp_col="decayedTrimpBefore",
                secondary_acute_trimp_col="progress",
                secondary_fatigue_model="exponential",
                gap_steep_threshold=steep_threshold,
                gap_soft_start=0.04,
                gap_climb_scale=climb_scale,
                gap_descent_scale=descent_scale,
            )
            terrain = _terrain_mae(segments, prediction["predictedTimeSec"], actual_col)
            score = _score_row(terrain)
            passes_flat = score["flatMaeMin"] <= flat_guardrail_mae_min
            summary_rows.append(
                {
                    "gapClimbScale": climb_scale,
                    "gapDescentScale": descent_scale,
                    "gapSteepThreshold": steep_threshold,
                    "alpha": best.get("alpha"),
                    "fatigueCoef": best.get("fatigueCoef"),
                    "secondaryFatigueCoef": best.get("secondaryFatigueCoef"),
                    "fatigueModel": best.get("fatigueModel"),
                    "segmentMaeSec": best.get("segmentMaeSec"),
                    "passesFlatGuardrail": passes_flat,
                    **score,
                }
            )
            terrain = terrain.assign(
                gapClimbScale=climb_scale,
                gapDescentScale=descent_scale,
            )
            detail_frames.append(terrain)
            print(
                f"climb={climb_scale:.2f} descent={descent_scale:.2f} "
                f"steepMAE={score['steepMaeMin']:.3f} "
                f"sc={score['steepClimbMaeMin']:.3f} sd={score['steepDescentMaeMin']:.3f} "
                f"flat={score['flatMaeMin']:.3f} pass={passes_flat}"
            )
    return pd.DataFrame(summary_rows), pd.concat(detail_frames, ignore_index=True)


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
        default=Path("data/exp_perf_predictions/trail_digital_twin_steep_gap"),
    )
    parser.add_argument("--steep-threshold", type=float, default=0.15)
    parser.add_argument("--flat-guardrail", type=float, default=0.97)
    args = parser.parse_args()

    segments = _load_hard_trail_segments(args.predictions_csv)
    actual_col = "actualMovingTimeSec" if "actualMovingTimeSec" in segments.columns else "actualTimeSec"
    print(f"Loaded {len(segments)} hardTrailRun segments; fit clock={actual_col}")

    # Exp 1: empirical scales around implied corrections (~0.87 climb, ~1.58 descent)
    climb_scales = [1.0, 0.95, 0.90, 0.85, 0.80]
    descent_scales = [1.0, 1.30, 1.45, 1.60, 1.80]
    summary, details = run_grid(
        segments,
        climb_scales=climb_scales,
        descent_scales=descent_scales,
        steep_threshold=args.steep_threshold,
        actual_col=actual_col,
        flat_guardrail_mae_min=args.flat_guardrail,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "steep_gap_grid_summary.csv"
    details_path = args.output_dir / "steep_gap_grid_terrain.csv"
    summary.sort_values(["passesFlatGuardrail", "steepMaeMin"], ascending=[False, True]).to_csv(
        summary_path, index=False
    )
    details.to_csv(details_path, index=False)

    eligible = summary[summary["passesFlatGuardrail"]].sort_values("steepMaeMin")
    print("\nTop eligible (flat guardrail):")
    print(
        eligible[
            [
                "gapClimbScale",
                "gapDescentScale",
                "steepMaeMin",
                "steepClimbMaeMin",
                "steepDescentMaeMin",
                "flatMaeMin",
                "descentMaeMin",
                "climbMaeMin",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
