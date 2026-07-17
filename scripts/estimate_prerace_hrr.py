"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Pre-race constant-HRR estimator for GPX routes.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services import trail_performance_model as tpm  # noqa: E402
from utils.gpx_parser import parse_gpx_to_timeseries  # noqa: E402

DEFAULT_ALPHA = 0.8
DEFAULT_FATIGUE_COEF = 0.20
DEFAULT_FATIGUE_MODEL = "exponential"
DEFAULT_VMA_KMH = 18.0
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


def _to_float(value: object, default: float = np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


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
    params = _load_parameter_table(asset_dir)
    if params.empty:
        return DEFAULT_ALPHA, DEFAULT_FATIGUE_COEF, DEFAULT_FATIGUE_MODEL
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


def _load_parameter_table(asset_dir: Path) -> pd.DataFrame:
    candidates = [asset_dir] if asset_dir.is_file() else [
        asset_dir / "table_fitted_parameters.csv",
        asset_dir / "benchmark_fitted_parameters.csv",
    ]
    for parameter_path in candidates:
        if parameter_path.exists():
            return pd.read_csv(parameter_path)
    return pd.DataFrame()


def _load_stage3_validation_metrics(asset_dir: Path, cohort: str) -> dict[str, float]:
    params = _load_parameter_table(asset_dir)
    if params.empty:
        return {}
    match = params[
        params["cohort"].astype(str).eq(cohort)
        & params["stage"].astype(str).eq("Stage 3 HRR speed ratio")
    ]
    if match.empty:
        return {}
    row = match.iloc[0]
    return {
        "raceMaeSec": _to_float(row.get("raceMaeSec")),
        "raceMapePct": _to_float(row.get("raceMapePct")),
        "raceR2": _to_float(row.get("raceR2")),
    }


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


def _parse_duration_windows(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _load_power_law_envelope(
    data_dir: Path,
    hr_rest: float,
    hr_max: float,
    duration_windows_min: list[float],
    hrr_min: float,
    hrr_max: float,
    target_stat: str,
    target_quantile: float,
    min_activity_count: int,
    fit_weight_mode: str,
    fit_weight_power: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    metrics = pd.read_csv(data_dir / "activities_metrics.csv")
    params, windows = tpm.estimate_hrr_duration_power_law(
        metrics,
        duration_windows_min=duration_windows_min,
        hr_rest=hr_rest,
        hr_max=hr_max,
        hrr_min=hrr_min,
        hrr_max=hrr_max,
        target_stat=target_stat,
        target_quantile=target_quantile,
        min_activity_count=min_activity_count,
        fit_weight_mode=fit_weight_mode,
        fit_weight_power=fit_weight_power,
        categories=("RUN", "TRAIL_RUN"),
    )
    return dict(params), windows


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


def _print_power_law(params: dict[str, object], windows: pd.DataFrame) -> None:
    coefficient = float(params.get("coefficient", np.nan))
    exponent = float(params.get("exponent", np.nan))
    print("\nObserved best maintained HRR by representative duration window")
    has_fit = bool(np.isfinite(coefficient) and np.isfinite(exponent))
    if has_fit:
        print(f"Power law: HRR = {coefficient:.3f} * duration_hours^{exponent:.3f}")
        print(
            "Fit quality: "
            f"windows={int(params.get('fitWindowCount', 0))}, "
            f"log-R2={float(params.get('r2Log', np.nan)):.3f}, "
            f"MAE={float(params.get('maeHrr', np.nan)):.3f} HRR, "
            f"weights={params.get('fitWeightMode', 'uniform')}"
            f"^{float(params.get('fitWeightPower', 0.0)):.1f}"
        )
    else:
        print("Power-law fit unavailable: not enough observed duration/HRR support.")
    visible = windows.copy()
    visible["duration"] = visible["durationMin"].map(lambda value: _format_duration(value * 60.0))
    visible["fittedHrr"] = visible["fittedHrr"].round(3)
    visible["targetHrr"] = visible["targetHrr"].round(3)
    visible["maxObservedHrr"] = visible["maxObservedHrr"].round(3)
    visible["fitWeight"] = visible["fitWeight"].round(3)
    print(
        visible[
            [
                "duration",
                "activityCount",
                "maxObservedHrr",
                "targetHrr",
                "fittedHrr",
                "fitWeight",
                "usedForFit",
                "isExtrapolated",
            ]
        ].to_string(index=False)
    )
    if not has_fit:
        return


def _apply_power_law_feasibility(
    sweep: pd.DataFrame,
    params: dict[str, object],
) -> pd.DataFrame:
    result = sweep.copy()
    result["maxSustainableSec"] = result["hrr"].map(
        lambda hrr: tpm.max_duration_for_hrr_power_law(float(hrr), params)
    )
    result["maxSustainableHours"] = result["maxSustainableSec"] / 3600.0
    result["sustainabilityMarginMin"] = (
        result["maxSustainableSec"] - result["predictedTimeSec"]
    ) / 60.0
    result["feasible"] = result["predictedTimeSec"].le(result["maxSustainableSec"])
    result["durationModel"] = "power_law"
    return result


def _local_time_sensitivity_sec_per_hrr(sweep: pd.DataFrame, hrr: float) -> float:
    working = sweep[["hrr", "predictedTimeSec"]].copy()
    working["hrr"] = pd.to_numeric(working["hrr"], errors="coerce")
    working["predictedTimeSec"] = pd.to_numeric(working["predictedTimeSec"], errors="coerce")
    working = working.dropna().sort_values("hrr").reset_index(drop=True)
    if len(working) < 2:
        return np.nan

    values = working["hrr"].to_numpy(dtype=float)
    index = int(np.argmin(np.abs(values - float(hrr))))
    if 0 < index < len(working) - 1:
        low = working.iloc[index - 1]
        high = working.iloc[index + 1]
    elif index == 0:
        low = working.iloc[0]
        high = working.iloc[1]
    else:
        low = working.iloc[-2]
        high = working.iloc[-1]

    delta_hrr = float(high["hrr"] - low["hrr"])
    if not np.isfinite(delta_hrr) or abs(delta_hrr) < 1e-12:
        return np.nan
    return float((high["predictedTimeSec"] - low["predictedTimeSec"]) / delta_hrr)


def _uncertainty_risk_label(hrr_margin_to_mae: float) -> str:
    if not np.isfinite(hrr_margin_to_mae):
        return "unknown"
    if hrr_margin_to_mae < 1.0:
        return "high"
    if hrr_margin_to_mae < 2.0:
        return "medium"
    return "low"


def _route_uncertainty_metrics(
    best: pd.Series,
    sweep: pd.DataFrame,
    power_law_params: dict[str, object],
    validation_metrics: dict[str, float],
    route_model_mae_sec: float | None,
) -> dict[str, object]:
    predicted_sec = _to_float(best.get("predictedTimeSec"))
    best_hrr = _to_float(best.get("hrr"))
    model_mae_sec = (
        _to_float(route_model_mae_sec)
        if route_model_mae_sec is not None
        else _to_float(validation_metrics.get("raceMaeSec"))
    )
    model_mape_pct = _to_float(validation_metrics.get("raceMapePct"))
    model_r2 = _to_float(validation_metrics.get("raceR2"))

    slope = _local_time_sensitivity_sec_per_hrr(sweep, best_hrr)
    abs_slope = abs(slope) if np.isfinite(slope) else np.nan
    power_law_mae_hrr = _to_float(power_law_params.get("maeHrr"))
    power_law_time_sec = (
        abs_slope * power_law_mae_hrr
        if np.isfinite(abs_slope) and np.isfinite(power_law_mae_hrr)
        else np.nan
    )
    sustainable_hrr = (
        tpm.hrr_for_duration_power_law(predicted_sec, power_law_params)
        if np.isfinite(predicted_sec) and power_law_params
        else np.nan
    )
    hrr_margin = sustainable_hrr - best_hrr if np.isfinite(sustainable_hrr) else np.nan
    hrr_margin_to_mae = (
        hrr_margin / power_law_mae_hrr
        if np.isfinite(hrr_margin) and np.isfinite(power_law_mae_hrr) and power_law_mae_hrr > 0
        else np.nan
    )

    components = [value for value in (model_mae_sec, power_law_time_sec) if np.isfinite(value)]
    total_sec = float(np.sqrt(np.sum(np.square(components)))) if components else np.nan
    lower_sec = predicted_sec - total_sec if np.isfinite(total_sec) else np.nan
    upper_sec = predicted_sec + total_sec if np.isfinite(total_sec) else np.nan
    return {
        "routeModelMaeSec": model_mae_sec,
        "routeModelMapePct": model_mape_pct,
        "routeModelR2": model_r2,
        "powerLawMaeHrr": power_law_mae_hrr,
        "timeSensitivitySecPerHrr": slope,
        "powerLawTimeUncertaintySec": power_law_time_sec,
        "sustainableHrrAtPredictedTime": sustainable_hrr,
        "hrrSafetyMargin": hrr_margin,
        "hrrSafetyMarginToMae": hrr_margin_to_mae,
        "timeUncertaintySec": total_sec,
        "timeUncertaintyMin": total_sec / 60.0 if np.isfinite(total_sec) else np.nan,
        "timeUncertaintyPct": total_sec / predicted_sec * 100.0
        if np.isfinite(total_sec) and predicted_sec > 0
        else np.nan,
        "timeLowerSec": lower_sec,
        "timeUpperSec": upper_sec,
        "timeLowerHours": lower_sec / 3600.0 if np.isfinite(lower_sec) else np.nan,
        "timeUpperHours": upper_sec / 3600.0 if np.isfinite(upper_sec) else np.nan,
        "sustainabilityRisk": _uncertainty_risk_label(hrr_margin_to_mae),
        "uncertaintyMethod": "sqrt(route_model_mae_sec^2 + power_law_hrr_mae_time_sensitivity^2)",
    }


def _print_uncertainty(metrics: dict[str, object]) -> None:
    uncertainty_min = _to_float(metrics.get("timeUncertaintyMin"))
    if not np.isfinite(uncertainty_min):
        print("Uncertainty: n/a")
        return
    print(
        "Uncertainty: "
        f"+/- {uncertainty_min:.1f} min "
        f"(route MAE {_to_float(metrics.get('routeModelMaeSec')) / 60.0:.1f} min, "
        f"power-law {_to_float(metrics.get('powerLawTimeUncertaintySec')) / 60.0:.1f} min); "
        f"HRR safety margin {_to_float(metrics.get('hrrSafetyMargin')):.3f} "
        f"({metrics.get('sustainabilityRisk')} sustainability risk)"
    )


def _write_power_law_curve_html(
    output_path: Path,
    params: dict[str, object],
    windows: pd.DataFrame,
    route_estimates: list[dict[str, object]],
) -> None:
    coefficient = float(params.get("coefficient", np.nan))
    exponent = float(params.get("exponent", np.nan))
    if not np.isfinite(coefficient) or not np.isfinite(exponent):
        return

    min_window_sec = float(params.get("minWindowSec", 300.0))
    max_window_sec = float(params.get("maxWindowSec", 86_400.0))
    hrr_min = float(params.get("hrrMin", 0.0))
    hrr_max = float(params.get("hrrMax", 1.2))
    duration_sec = np.geomspace(max(min_window_sec, 1.0), max_window_sec, 240)
    duration_hours = duration_sec / 3600.0
    fitted_hrr = np.clip(coefficient * np.power(duration_hours, exponent), hrr_min, hrr_max)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=duration_hours,
            y=fitted_hrr,
            mode="lines",
            name="Fitted power law",
            line={"color": "#1f77b4", "width": 3},
            hovertemplate="Duration %{x:.2f} h<br>HRR %{y:.3f}<extra></extra>",
        )
    )

    visible_windows = windows[windows["targetHrr"].notna()].copy()
    if not visible_windows.empty:
        visible_windows["durationLabel"] = visible_windows["durationMin"].map(
            lambda value: _format_duration(float(value) * 60.0)
        )
        fig.add_trace(
            go.Scatter(
                x=visible_windows["durationHours"],
                y=visible_windows["targetHrr"],
                mode="markers",
                name="Observed frontier windows",
                marker={
                    "color": "#2ca02c",
                    "size": 10,
                    "line": {"color": "white", "width": 1},
                },
                customdata=np.stack(
                    [
                        visible_windows["durationLabel"],
                        visible_windows["activityCount"],
                        visible_windows["maxObservedHrr"],
                        visible_windows["fittedHrr"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Window %{customdata[0]}<br>"
                    "Activities %{customdata[1]}<br>"
                    "Observed HRR %{customdata[2]:.3f}<br>"
                    "Fitted HRR %{customdata[3]:.3f}<extra></extra>"
                ),
            )
        )

    if route_estimates:
        route_df = pd.DataFrame(route_estimates)
        route_df["timeLabel"] = route_df["predictedTimeSec"].map(_format_duration)
        route_df["uncertaintyLabel"] = route_df["timeUncertaintySec"].map(_format_duration)
        time_uncertainty_hours = (
            pd.to_numeric(route_df["timeUncertaintySec"], errors="coerce").fillna(0.0) / 3600.0
        )
        fig.add_trace(
            go.Scatter(
                x=route_df["predictedTimeHours"],
                y=route_df["bestHrr"],
                mode="markers+text",
                name="Route estimates",
                marker={
                    "color": "#d62728",
                    "size": 12,
                    "symbol": "diamond",
                    "line": {"color": "white", "width": 1},
                },
                text=route_df["route"].str.replace(".gpx", "", regex=False),
                textposition="top center",
                customdata=np.stack(
                    [
                        route_df["route"],
                        route_df["timeLabel"],
                        route_df["heartRateBpm"],
                        route_df["sustainabilityMarginMin"],
                        route_df["uncertaintyLabel"],
                        route_df["hrrSafetyMargin"],
                        route_df["sustainabilityRisk"],
                    ],
                    axis=-1,
                ),
                error_x={
                    "type": "data",
                    "array": time_uncertainty_hours,
                    "visible": bool(time_uncertainty_hours.gt(0).any()),
                    "color": "#d62728",
                    "thickness": 1.4,
                },
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "Time %{customdata[1]}<br>"
                    "Uncertainty +/- %{customdata[4]}<br>"
                    "HRR %{y:.2f}<br>"
                    "HR %{customdata[2]:.0f} bpm<br>"
                    "Margin %{customdata[3]:.1f} min<br>"
                    "HRR safety %{customdata[5]:.3f}<br>"
                    "Risk %{customdata[6]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Best Maintained HRR Over Time",
        template="plotly_white",
        width=1100,
        height=640,
        xaxis={
            "title": "Duration",
            "type": "log",
            "tickmode": "array",
            "tickvals": [5 / 60, 10 / 60, 0.5, 1, 2, 4, 8, 12, 24],
            "ticktext": ["5m", "10m", "30m", "1h", "2h", "4h", "8h", "12h", "24h"],
        },
        yaxis={"title": "HRR", "range": [max(0.0, hrr_min - 0.02), min(1.05, hrr_max + 0.03)]},
        legend={"orientation": "h", "y": -0.22},
        margin={"l": 70, "r": 30, "t": 80, "b": 110},
    )

    chart_html = pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )
    fit_text = (
        f"HRR(T_hours) = {coefficient:.3f} * T_hours^{exponent:.3f}; "
        f"log-R2 = {float(params.get('r2Log', np.nan)):.3f}; "
        f"MAE = {float(params.get('maeHrr', np.nan)):.3f} HRR; "
        f"fit weights = {params.get('fitWeightMode', 'uniform')}"
        f"^{float(params.get('fitWeightPower', 0.0)):.1f}."
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HRR Duration Power Law</title>
  <style>
    body {{
      margin: 0;
      color: #1f2937;
      background: #f8fafc;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      font-weight: 700;
    }}
    p {{
      margin: 0 0 18px;
      color: #4b5563;
      line-height: 1.5;
    }}
    .panel {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .formula {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      color: #111827;
    }}
  </style>
  <script>{get_plotlyjs()}</script>
</head>
<body>
  <main>
    <h1>HRR Duration Power Law</h1>
    <p class="formula">{html.escape(fit_text)}</p>
    <p>Green points are the observed best maintained HRR at representative windows.
    The blue curve is the fitted monotone power law from 5 minutes to 24 hours.
    Red diamonds are the selected constant-HRR route estimates; their horizontal
    bars show the combined time uncertainty from route validation error and
    HRR-duration fit residual sensitivity.</p>
    <section class="panel">
      {chart_html}
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


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
    parser.add_argument("--fatigue-model", choices=["exponential"], default="exponential")
    parser.add_argument("--hrr-min", type=float, default=0.2)
    parser.add_argument("--hrr-max", type=float, default=0.95)
    parser.add_argument("--hrr-step", type=float, default=0.01)
    parser.add_argument("--hrr-reference", type=float, default=0.85)
    parser.add_argument("--hrr-min-factor", type=float, default=0.3)
    parser.add_argument("--hrr-max-factor", type=float, default=1.30)
    parser.add_argument("--trimp-scale", type=float, default=1.0)
    parser.add_argument("--decay-lambda", type=float, default=0.25)
    parser.add_argument("--min-fatigue-factor", type=float, default=0.50)
    parser.add_argument(
        "--fatigue-input-col",
        choices=["cumTrimpBefore", "decayedTrimpBefore"],
        default="decayedTrimpBefore",
    )
    parser.add_argument("--load-factor", type=float, default=1.0)
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--duration-model", choices=["bin", "power-law"], default="bin")
    parser.add_argument(
        "--duration-windows-min",
        default=",".join(str(value) for value in DEFAULT_DURATION_WINDOWS_MIN),
        help="Comma-separated representative windows in minutes for the power-law envelope.",
    )
    parser.add_argument("--power-law-target", choices=["max", "quantile"], default="max")
    parser.add_argument("--power-law-quantile", type=float, default=0.90)
    parser.add_argument("--power-law-min-count", type=int, default=2)
    parser.add_argument("--power-law-hrr-min", type=float, default=0.30)
    parser.add_argument("--power-law-hrr-max", type=float, default=0.98)
    parser.add_argument(
        "--power-law-weight-mode",
        choices=["uniform", "performance"],
        default="performance",
    )
    parser.add_argument("--power-law-weight-power", type=float, default=10.0)
    parser.add_argument(
        "--route-model-mae-sec",
        type=float,
        default=None,
        help="Override the validation MAE used in uncertainty bands.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hr_rest, hr_max = _load_athlete(args.data_dir)
    default_alpha, default_fatigue_coef, default_fatigue_model = _load_stage3_defaults(
        args.asset_dir,
        args.cohort,
    )
    validation_metrics = _load_stage3_validation_metrics(args.asset_dir, args.cohort)
    alpha = args.alpha if args.alpha is not None else default_alpha
    fatigue_coef = args.fatigue_coef if args.fatigue_coef is not None else default_fatigue_coef
    fatigue_model = args.fatigue_model if args.fatigue_model is not None else default_fatigue_model
    envelope = pd.DataFrame()
    power_law_params: dict[str, object] = {}
    power_law_windows = pd.DataFrame()
    if args.duration_model == "bin":
        envelope = _load_envelope(args.data_dir, hr_rest, hr_max, args.bin_width)
        _print_envelope(envelope)
    else:
        duration_windows = _parse_duration_windows(args.duration_windows_min)
        power_law_params, power_law_windows = _load_power_law_envelope(
            args.data_dir,
            hr_rest,
            hr_max,
            duration_windows,
            hrr_min=args.power_law_hrr_min,
            hrr_max=args.power_law_hrr_max,
            target_stat=args.power_law_target,
            target_quantile=args.power_law_quantile,
            min_activity_count=args.power_law_min_count,
            fit_weight_mode=args.power_law_weight_mode,
            fit_weight_power=args.power_law_weight_power,
        )
        _print_power_law(power_law_params, power_law_windows)

    hrr_values = np.round(
        np.arange(args.hrr_min, args.hrr_max + args.hrr_step * 0.5, args.hrr_step),
        4,
    )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.duration_model == "bin":
            envelope.to_csv(args.output_dir / "hrr_duration_envelope.csv", index=False)
        else:
            power_law_windows.to_csv(
                args.output_dir / "hrr_duration_power_law_windows.csv",
                index=False,
            )
            pd.DataFrame([power_law_params]).to_csv(
                args.output_dir / "hrr_duration_power_law_params.csv",
                index=False,
            )

    print(
        "\nModel defaults: "
        f"cohort={args.cohort}, VMA={args.vma_kmh:.1f} km/h, alpha={alpha:.2f}, "
        f"fatigue={fatigue_model} {fatigue_coef:.2f}, HRR reference={args.hrr_reference:.2f}, "
        f"HRR factor clip=[{args.hrr_min_factor:.2f}, {args.hrr_max_factor:.2f}], "
        f"min fatigue={args.min_fatigue_factor:.2f}, fatigue input={args.fatigue_input_col}, "
        f"duration model={args.duration_model}"
    )

    route_estimates: list[dict[str, object]] = []
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
            hrr_min_factor=args.hrr_min_factor,
            hrr_max_factor=args.hrr_max_factor,
            trimp_scale=args.trimp_scale,
            decay_lambda=args.decay_lambda,
            min_fatigue_factor=args.min_fatigue_factor,
            load_factor=args.load_factor,
            fatigue_input_col=args.fatigue_input_col,
            envelope_df=envelope if args.duration_model == "bin" else None,
            hr_rest=hr_rest,
            hr_max=hr_max,
        )
        if args.duration_model == "power-law":
            sweep = _apply_power_law_feasibility(sweep, power_law_params)
        _print_route_result(gpx_path, segments, sweep)
        best = tpm.select_best_constant_hrr(sweep)
        if not best.empty:
            uncertainty = _route_uncertainty_metrics(
                best,
                sweep,
                power_law_params,
                validation_metrics,
                args.route_model_mae_sec,
            )
            _print_uncertainty(uncertainty)
            route_estimates.append(
                {
                    "route": gpx_path.name,
                    "distanceKm": float(
                        pd.to_numeric(segments["distanceKm"], errors="coerce").sum()
                    ),
                    "elevGainM": float(pd.to_numeric(segments["elevGainM"], errors="coerce").sum()),
                    "elevLossM": float(pd.to_numeric(segments["elevLossM"], errors="coerce").sum()),
                    "bestHrr": float(best["hrr"]),
                    "heartRateBpm": float(best["heartRateBpm"]),
                    "predictedTimeSec": float(best["predictedTimeSec"]),
                    "predictedTimeHours": float(best["predictedTimeHours"]),
                    "paceMinPerKm": float(best["paceMinPerKm"]),
                    "maxSustainableSec": float(best["maxSustainableSec"]),
                    "sustainabilityMarginMin": float(best["sustainabilityMarginMin"]),
                    "feasible": bool(best["feasible"]),
                    "durationModel": args.duration_model,
                    "alpha": float(alpha),
                    "fatigueCoef": float(fatigue_coef),
                    "fatigueModel": fatigue_model,
                    "fatigueInputCol": args.fatigue_input_col,
                    "hrrReference": float(args.hrr_reference),
                    "hrrMinFactor": float(args.hrr_min_factor),
                    "hrrMaxFactor": float(args.hrr_max_factor),
                    "minFatigueFactor": float(args.min_fatigue_factor),
                    "powerLawCoefficient": power_law_params.get("coefficient", np.nan),
                    "powerLawExponent": power_law_params.get("exponent", np.nan),
                    "powerLawFitWeightMode": power_law_params.get("fitWeightMode", ""),
                    "powerLawFitWeightPower": power_law_params.get("fitWeightPower", np.nan),
                    **uncertainty,
                }
            )
        if args.output_dir:
            safe_name = gpx_path.stem.replace(" ", "_").replace("/", "_")
            segments.to_csv(args.output_dir / f"{safe_name}_segments.csv", index=False)
            sweep.to_csv(args.output_dir / f"{safe_name}_hrr_sweep.csv", index=False)
    if args.output_dir and route_estimates:
        pd.DataFrame(route_estimates).to_csv(args.output_dir / "route_estimates.csv", index=False)
    if args.output_dir and args.duration_model == "power-law":
        _write_power_law_curve_html(
            args.output_dir / "hrr_duration_power_law_curve.html",
            power_law_params,
            power_law_windows,
            route_estimates,
        )


if __name__ == "__main__":
    main()
