"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Pre-race constant-HRR estimator for GPX routes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services import trail_performance_model as tpm  # noqa: E402
from utils.gpx_parser import parse_gpx_to_timeseries  # noqa: E402

DEFAULT_ALPHA = 0.45
DEFAULT_FATIGUE_COEF = 0.40
DEFAULT_FATIGUE_MODEL = "linear"
DEFAULT_VMA_KMH = 18.0


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "n/a"
    total_minutes = int(round(float(seconds) / 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:d}h{minutes:02d}"


def _load_athlete(data_dir: Path) -> tuple[float, float]:
    athlete = pd.read_csv(data_dir / "athlete.csv").iloc[0]
    return float(athlete["hrRest"]), float(athlete["hrMax"])


def _load_stage3_defaults(asset_dir: Path, cohort: str) -> tuple[float, float, str]:
    parameter_path = asset_dir / "table_fitted_parameters.csv"
    if not parameter_path.exists():
        return DEFAULT_ALPHA, DEFAULT_FATIGUE_COEF, DEFAULT_FATIGUE_MODEL
    params = pd.read_csv(parameter_path)
    match = params[
        params["cohort"].astype(str).eq(cohort)
        & params["stage"].astype(str).eq("Stage 3 HRR speed ratio")
    ]
    if match.empty:
        return DEFAULT_ALPHA, DEFAULT_FATIGUE_COEF, DEFAULT_FATIGUE_MODEL
    row = match.iloc[0]
    alpha = float(row.get("alpha", DEFAULT_ALPHA))
    fatigue_coef = float(row.get("fatigueCoef", DEFAULT_FATIGUE_COEF))
    fatigue_model = str(row.get("fatigueModel", DEFAULT_FATIGUE_MODEL) or DEFAULT_FATIGUE_MODEL)
    return alpha, fatigue_coef, fatigue_model


def _load_envelope(data_dir: Path, hr_rest: float, hr_max: float, bin_width: float) -> pd.DataFrame:
    metrics = pd.read_csv(data_dir / "activities_metrics.csv")
    return tpm.estimate_hrr_duration_envelope(
        metrics,
        hr_rest=hr_rest,
        hr_max=hr_max,
        bin_width=bin_width,
        hrr_min=0.30,
        hrr_max=0.95,
        categories=("RUN", "TRAIL_RUN"),
    )


def _route_segments(gpx_path: Path, segment_km: float) -> pd.DataFrame:
    route = parse_gpx_to_timeseries(gpx_path.read_bytes())
    if route.empty:
        return pd.DataFrame()
    return tpm.route_segments_from_points(route, segment_km=segment_km)


def _summarize_route(gpx_path: Path, segments: pd.DataFrame) -> str:
    distance = float(pd.to_numeric(segments["distanceKm"], errors="coerce").sum())
    ascent = float(pd.to_numeric(segments["elevGainM"], errors="coerce").sum())
    descent = float(pd.to_numeric(segments["elevLossM"], errors="coerce").sum())
    return f"{gpx_path.name}: {distance:.1f} km, D+ {ascent:.0f} m, D- {descent:.0f} m"


def _print_envelope(envelope: pd.DataFrame) -> None:
    visible = envelope[envelope["activityCount"].gt(0)].copy()
    visible["maxDuration"] = visible["maxDurationSec"].map(_format_duration)
    print("\nObserved max duration by average HRR range")
    print(visible[["hrrRange", "activityCount", "maxDuration"]].to_string(index=False))


def _print_route_result(gpx_path: Path, segments: pd.DataFrame, sweep: pd.DataFrame) -> None:
    best = tpm.select_best_constant_hrr(sweep)
    print(f"\n{_summarize_route(gpx_path, segments)}")
    if best.empty:
        print("No candidate HRR could be evaluated.")
        return
    status = "feasible" if bool(best["feasible"]) else "least infeasible"
    print(
        "Best constant HRR "
        f"({status}): HRR {best['hrr']:.2f}, HR {best['heartRateBpm']:.0f} bpm, "
        f"time {_format_duration(best['predictedTimeSec'])}, "
        f"pace {best['paceMinPerKm']:.2f} min/km, "
        f"margin {best['sustainabilityMarginMin']:.1f} min"
    )
    display = sweep.copy()
    display["time"] = display["predictedTimeSec"].map(_format_duration)
    display["maxSustainable"] = display["maxSustainableSec"].map(_format_duration)
    display["heartRateBpm"] = display["heartRateBpm"].round(0).astype("Int64")
    print(
        display[
            [
                "hrr",
                "heartRateBpm",
                "time",
                "paceMinPerKm",
                "maxSustainable",
                "sustainabilityMarginMin",
                "feasible",
            ]
        ]
        .round({"hrr": 2, "paceMinPerKm": 2, "sustainabilityMarginMin": 1})
        .to_string(index=False)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate pre-race time over GPX routes by sweeping constant HRR."
    )
    parser.add_argument("gpx", nargs="+", type=Path, help="GPX route files to evaluate.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--asset-dir", type=Path, default=Path("docs/science/paper_assets"))
    parser.add_argument("--cohort", default="selectedDateRaces")
    parser.add_argument("--segment-km", type=float, default=1.0)
    parser.add_argument("--vma-kmh", type=float, default=DEFAULT_VMA_KMH)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--fatigue-coef", type=float, default=None)
    parser.add_argument("--fatigue-model", choices=["linear", "exponential"], default=None)
    parser.add_argument("--hrr-min", type=float, default=0.55)
    parser.add_argument("--hrr-max", type=float, default=0.88)
    parser.add_argument("--hrr-step", type=float, default=0.01)
    parser.add_argument("--hrr-reference", type=float, default=0.70)
    parser.add_argument("--trimp-scale", type=float, default=10.0)
    parser.add_argument("--decay-lambda", type=float, default=0.30)
    parser.add_argument("--load-factor", type=float, default=1.0)
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hr_rest, hr_max = _load_athlete(args.data_dir)
    default_alpha, default_fatigue_coef, default_fatigue_model = _load_stage3_defaults(
        args.asset_dir,
        args.cohort,
    )
    alpha = args.alpha if args.alpha is not None else default_alpha
    fatigue_coef = args.fatigue_coef if args.fatigue_coef is not None else default_fatigue_coef
    fatigue_model = args.fatigue_model if args.fatigue_model is not None else default_fatigue_model
    envelope = _load_envelope(args.data_dir, hr_rest, hr_max, args.bin_width)
    _print_envelope(envelope)

    hrr_values = np.round(
        np.arange(args.hrr_min, args.hrr_max + args.hrr_step * 0.5, args.hrr_step),
        4,
    )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        envelope.to_csv(args.output_dir / "hrr_duration_envelope.csv", index=False)

    print(
        "\nModel defaults: "
        f"cohort={args.cohort}, VMA={args.vma_kmh:.1f} km/h, alpha={alpha:.2f}, "
        f"fatigue={fatigue_model} {fatigue_coef:.2f}, HRR reference={args.hrr_reference:.2f}"
    )

    for gpx_path in args.gpx:
        segments = _route_segments(gpx_path, args.segment_km)
        if segments.empty:
            print(f"\n{gpx_path}: no usable GPX segments")
            continue
        sweep = tpm.sweep_constant_hrr_route(
            segments,
            hrr_values=hrr_values,
            v_anchor_kmh=args.vma_kmh,
            alpha=alpha,
            fatigue_coef=fatigue_coef,
            fatigue_model=fatigue_model,
            hrr_reference=args.hrr_reference,
            trimp_scale=args.trimp_scale,
            decay_lambda=args.decay_lambda,
            load_factor=args.load_factor,
            envelope_df=envelope,
            hr_rest=hr_rest,
            hr_max=hr_max,
        )
        _print_route_result(gpx_path, segments, sweep)
        if args.output_dir:
            safe_name = gpx_path.stem.replace(" ", "_").replace("/", "_")
            segments.to_csv(args.output_dir / f"{safe_name}_segments.csv", index=False)
            sweep.to_csv(args.output_dir / f"{safe_name}_hrr_sweep.csv", index=False)


if __name__ == "__main__":
    main()
