"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Prospective race prediction with terrain-family–adaptive constant HRR.

Estimates HRR distributions on the five grade families from other activities,
picks a duration-feasible target mean HRR, modulates per segment from real-data
family means, and simulates with sequential TRIMP.

Journal: docs/science/journal_segment_hrr_adaptive.md
"""

from __future__ import annotations

import argparse
import importlib.util
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
fit_hrr_duration_power_law = _pred.fit_hrr_duration_power_law
select_duration_feasible_constant_hrr = _pred.select_duration_feasible_constant_hrr
segments_from_race_pacing = _pred.segments_from_race_pacing
segments_from_activity_timeseries = _pred.segments_from_activity_timeseries
load_train_segments = _pred.load_train_segments

FIVE_FAMILIES = ("flat", "climb", "steep_climb", "descent", "steep_descent")

DEFAULT_PHYSIOLOGY = {
    "vma_flat_kmh": 18.0,
    "hrr_reference": 0.88,
    "hrr_min_factor": 0.30,
    "hrr_max_factor": 1.20,
    "decay_lambda": 0.20,
    "min_fatigue_factor": 0.60,
    "gap_steep_threshold": 0.15,
    "gap_soft_start": 0.04,
    "gap_climb_scale": 0.85,
    "gap_descent_scale": 1.60,
}


def _fmt_hms(seconds: float) -> str:
    total = int(round(float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def resolve_terrain_family(row: pd.Series) -> str:
    """Map a segment to one of the five grade families."""
    fam = str(row.get("terrainFamily", "") or "")
    if fam in FIVE_FAMILIES:
        return fam
    grade = float(pd.to_numeric(row.get("avgGrade"), errors="coerce") or 0.0)
    return tpm.terrain_family(grade)


def estimate_hrr_by_terrain(
    segment_predictions_csv: Path,
    holdout_ids: set[str],
    *,
    cohort: str = "hardTrailRun",
    fit_objective: str = "activity",
    statistic: str = "mean",
) -> pd.DataFrame:
    """Empirical HRR distribution by terrain family (hold-outs excluded)."""
    seg = pd.read_csv(segment_predictions_csv, dtype={"activityId": str})
    if "fitObjective" in seg.columns and seg["fitObjective"].astype(str).eq(fit_objective).any():
        seg = seg[seg["fitObjective"].astype(str).eq(fit_objective)].copy()
    if "cohort" in seg.columns and seg["cohort"].astype(str).eq(cohort).any():
        seg = seg[seg["cohort"].astype(str).eq(cohort)].copy()
    seg = seg.drop_duplicates(["activityId", "segmentIndex"], keep="first")
    seg = seg[~seg["activityId"].isin(holdout_ids)].copy()
    if "isFitEligible" in seg.columns:
        seg = seg[seg["isFitEligible"].astype(bool)].copy()
    seg["meanHrReserve"] = pd.to_numeric(seg["meanHrReserve"], errors="coerce")
    seg["actualMovingTimeSec"] = pd.to_numeric(
        seg.get("actualMovingTimeSec", seg.get("actualTimeSec")), errors="coerce"
    )
    seg["terrainFamilyResolved"] = seg.apply(resolve_terrain_family, axis=1)
    overall = float(seg["meanHrReserve"].mean())
    rows: list[dict[str, Any]] = []
    for fam in FIVE_FAMILIES:
        g = seg[seg["terrainFamilyResolved"].eq(fam)]
        hrr = g["meanHrReserve"].dropna()
        if hrr.empty:
            rows.append(
                {
                    "terrainFamily": fam,
                    "n": 0,
                    "statHrr": overall,
                    "meanHrr": overall,
                    "p50": overall,
                    "p75": overall,
                    "p90": overall,
                    "modVsOverall": 1.0,
                }
            )
            continue
        mean_h = float(hrr.mean())
        p50 = float(hrr.quantile(0.50))
        p75 = float(hrr.quantile(0.75))
        p90 = float(hrr.quantile(0.90))
        if statistic == "p75":
            stat = p75
        elif statistic == "p90":
            stat = p90
        elif statistic == "p50":
            stat = p50
        else:
            stat = mean_h
        rows.append(
            {
                "terrainFamily": fam,
                "n": int(len(g)),
                "statHrr": stat,
                "meanHrr": mean_h,
                "p50": p50,
                "p75": p75,
                "p90": p90,
                "modVsOverall": float(stat / overall) if overall > 1e-9 else 1.0,
                "totalMovingMin": float(g["actualMovingTimeSec"].fillna(0).sum() / 60.0),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["overallMeanHrr"] = overall
    out.attrs["cohort"] = cohort
    out.attrs["statistic"] = statistic
    return out


def empirical_hrr_for_duration(duration_sec: float, windows: pd.DataFrame) -> float:
    """Stepwise monotone envelope: max observed HRR among activities lasting ≥ T.

    Uses the precomputed ``targetHrr`` column from ``estimate_hrr_duration_power_law``
    windows (already monotone non-increasing). Hold-outs must already be excluded
    from the window fit.
    """
    if windows is None or windows.empty or "targetHrr" not in windows.columns:
        return float("nan")
    m = windows[windows["targetHrr"].notna()].copy()
    if "activityCount" in m.columns:
        m = m[pd.to_numeric(m["activityCount"], errors="coerce").fillna(0) >= 3]
    if m.empty:
        return float("nan")
    dmin = float(duration_sec) / 60.0
    le = m[pd.to_numeric(m["durationMin"], errors="coerce") <= dmin]
    if le.empty:
        return float(pd.to_numeric(m.iloc[0]["targetHrr"], errors="coerce"))
    return float(pd.to_numeric(le.iloc[-1]["targetHrr"], errors="coerce"))


def blend_powerlaw_empirical_hrr(
    *,
    powerlaw_hrr: float,
    empirical_hrr: float,
    predicted_sec: float,
    t0_hours: float = 4.0,
    t1_hours: float = 10.0,
    w_max: float = 0.70,
) -> tuple[float, float]:
    """Duration-weighted blend of power-law feasible HRR and empirical window HRR.

    Short races (≤ ``t0_hours``) keep the power-law target. Beyond that, weight
    toward the empirical envelope up to ``w_max`` at ``t1_hours``. Returns
    ``(target_hrr, blend_weight)``.
    """
    pl = float(powerlaw_hrr)
    emp = float(empirical_hrr)
    if not np.isfinite(emp):
        return pl, 0.0
    hours = float(predicted_sec) / 3600.0
    w = float(np.clip((hours - t0_hours) / max(t1_hours - t0_hours, 1e-9), 0.0, 1.0) * w_max)
    return float((1.0 - w) * pl + w * emp), w


def assign_adaptive_hrr(
    route: pd.DataFrame,
    *,
    target_mean_hrr: float,
    modulators: pd.DataFrame,
    hrr_min: float = 0.40,
    hrr_max: float = 0.98,
    recenter: bool = True,
) -> pd.DataFrame:
    """Assign per-segment HRR = target * m_family, optionally re-centered."""
    out = route.copy()
    mod_map = {
        str(r["terrainFamily"]): float(r["modVsOverall"]) for _, r in modulators.iterrows()
    }
    families = [resolve_terrain_family(row) for _, row in out.iterrows()]
    out["terrainFamilyResolved"] = families
    mods = np.array([mod_map.get(f, 1.0) for f in families], dtype=float)
    raw = float(target_mean_hrr) * mods
    if recenter and len(raw) > 0:
        # Distance-weighted re-center to preserve target mean effort.
        dist = pd.to_numeric(out["distanceKm"], errors="coerce").fillna(0.0).to_numpy()
        w = np.clip(dist, 1e-6, None)
        cur = float(np.average(raw, weights=w))
        if cur > 1e-9:
            raw = raw * (float(target_mean_hrr) / cur)
    out["plannedHrr"] = np.clip(raw, hrr_min, hrr_max)
    out["meanHrReserve"] = out["plannedHrr"]
    return out


def apply_family_duration_caps(
    route: pd.DataFrame,
    *,
    power_law_params: dict[str, Any],
    fit: dict[str, object],
    physiology: dict[str, float],
) -> pd.DataFrame:
    """H3: cap each family's HRR by power-law max for that family's predicted time."""
    out = route.copy()
    # First-pass prediction to get time shares.
    first = tpm.simulate_observed_hrr_segments(
        out,
        v_anchor_kmh=float(physiology["vma_flat_kmh"]),
        alpha=float(fit["alpha"]),
        fatigue_coef=float(fit["fatigueCoef"]),
        fatigue_model=str(fit["fatigueModel"]),
        hrr_reference=float(physiology["hrr_reference"]),
        hrr_min_factor=float(physiology["hrr_min_factor"]),
        hrr_max_factor=float(physiology["hrr_max_factor"]),
        decay_lambda=float(physiology["decay_lambda"]),
        min_fatigue_factor=float(physiology["min_fatigue_factor"]),
        hrr_col="meanHrReserve",
        fatigue_input_col="cumTrimpBefore",
        gap_steep_threshold=float(physiology["gap_steep_threshold"]),
        gap_soft_start=float(physiology["gap_soft_start"]),
        gap_climb_scale=float(physiology["gap_climb_scale"]),
        gap_descent_scale=float(physiology["gap_descent_scale"]),
    )
    caps: dict[str, float] = {}
    for fam in FIVE_FAMILIES:
        mask = first["terrainFamilyResolved"].astype(str).eq(fam) if "terrainFamilyResolved" in first.columns else (
            first.apply(resolve_terrain_family, axis=1).eq(fam)
        )
        t_f = float(pd.to_numeric(first.loc[mask, "predictedTimeSec"], errors="coerce").sum())
        if t_f <= 0:
            caps[fam] = 0.98
            continue
        # Max HRR sustainable for duration spent on this family.
        caps[fam] = float(tpm.hrr_for_duration_power_law(t_f, power_law_params))
    out["familyHrrCap"] = out["terrainFamilyResolved"].map(caps)
    out["plannedHrr"] = np.minimum(
        pd.to_numeric(out["plannedHrr"], errors="coerce"),
        pd.to_numeric(out["familyHrrCap"], errors="coerce"),
    )
    out["meanHrReserve"] = out["plannedHrr"]
    return out


def load_race_route(race_key: str, meta: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    if meta.get("profile_source") == "activity_timeseries":
        ts = REPO_ROOT / "data" / "timeseries" / f"{meta['activityId']}.csv"
        return segments_from_activity_timeseries(ts), "activity_timeseries"
    pacing = REPO_ROOT / "data" / "race_pacing" / f"{meta['race_pacing_id']}_segments.csv"
    route = segments_from_race_pacing(pacing)
    gpx = REPO_ROOT / str(meta["gpx"])
    if gpx.exists():
        route = attach_altitude_from_gpx(route, gpx)
    return route, "race_pacing_gpxalt"


def simulate_adaptive(
    route: pd.DataFrame,
    *,
    fit: dict[str, object],
    physiology: dict[str, float],
) -> pd.DataFrame:
    return tpm.simulate_observed_hrr_segments(
        route,
        v_anchor_kmh=float(physiology["vma_flat_kmh"]),
        alpha=float(fit["alpha"]),
        fatigue_coef=float(fit["fatigueCoef"]),
        fatigue_model=str(fit["fatigueModel"]),
        hrr_reference=float(physiology["hrr_reference"]),
        hrr_min_factor=float(physiology["hrr_min_factor"]),
        hrr_max_factor=float(physiology["hrr_max_factor"]),
        decay_lambda=float(physiology["decay_lambda"]),
        min_fatigue_factor=float(physiology["min_fatigue_factor"]),
        fallback_hrr=float(physiology["hrr_reference"]),
        hrr_col="meanHrReserve",
        fatigue_input_col="cumTrimpBefore",
        gap_steep_threshold=float(physiology["gap_steep_threshold"]),
        gap_soft_start=float(physiology["gap_soft_start"]),
        gap_climb_scale=float(physiology["gap_climb_scale"]),
        gap_descent_scale=float(physiology["gap_descent_scale"]),
    )


def run_hypothesis(
    *,
    hypothesis: str,
    races: dict[str, Any],
    fit: dict[str, object],
    physiology: dict[str, float],
    power_law_params: dict[str, Any],
    modulators: pd.DataFrame,
    output_dir: Path,
    hr_rest: float,
    hr_max: float,
    windows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    activities = pd.read_csv(REPO_ROOT / "data" / "activities.csv", dtype={"activityId": str})
    rows: list[dict[str, Any]] = []
    for race_key, meta in races.items():
        if "Rome" in str(meta.get("label", "")):
            continue  # out of scope for trail adaptive
        route, profile_source = load_race_route(race_key, meta)
        if route.empty:
            logger.warning("Empty route %s", race_key)
            continue
        best, _sweep = select_duration_feasible_constant_hrr(
            route,
            fit=fit,
            physiology=physiology,
            power_law_params=power_law_params,
            hr_rest=hr_rest,
            hr_max=hr_max,
        )
        pl_hrr = float(best["hrr"])
        pl_pred = float(best["predictedTimeSec"])
        emp_hrr = empirical_hrr_for_duration(pl_pred, windows) if windows is not None else float("nan")
        blend_w = 0.0
        target = pl_hrr
        if hypothesis.startswith("H10_blend") or hypothesis.startswith("H12_blend"):
            # Default H10: w_max=0.70 over 4–10 h predicted duration.
            w_max = 0.70
            if "w50" in hypothesis:
                w_max = 0.50
            elif "w85" in hypothesis:
                w_max = 0.85
            elif "w100" in hypothesis:
                w_max = 1.00
            target, blend_w = blend_powerlaw_empirical_hrr(
                powerlaw_hrr=pl_hrr,
                empirical_hrr=emp_hrr,
                predicted_sec=pl_pred,
                w_max=w_max,
            )
            if hypothesis.startswith("H12_blend"):
                # One re-evaluation of empirical envelope at the blended prediction.
                planned0 = assign_adaptive_hrr(route, target_mean_hrr=target, modulators=modulators)
                sim0 = simulate_adaptive(planned0, fit=fit, physiology=physiology)
                pred0 = float(sim0["predictedTimeSec"].sum())
                emp1 = empirical_hrr_for_duration(pred0, windows) if windows is not None else emp_hrr
                target, blend_w = blend_powerlaw_empirical_hrr(
                    powerlaw_hrr=pl_hrr,
                    empirical_hrr=emp1,
                    predicted_sec=pred0,
                    w_max=w_max,
                )
        elif hypothesis.startswith("H9_min") or hypothesis.startswith("H9c_"):
            target = float(min(pl_hrr, emp_hrr)) if np.isfinite(emp_hrr) else pl_hrr
            blend_w = 1.0 if np.isfinite(emp_hrr) and emp_hrr < pl_hrr else 0.0
        elif hypothesis.startswith("H11_switch"):
            hours = pl_pred / 3600.0
            if hours >= 6.0 and np.isfinite(emp_hrr):
                target, blend_w = float(emp_hrr), 1.0
            else:
                target, blend_w = pl_hrr, 0.0

        planned = assign_adaptive_hrr(route, target_mean_hrr=target, modulators=modulators)
        if hypothesis == "H3_family_caps":
            planned = apply_family_duration_caps(
                planned,
                power_law_params=power_law_params,
                fit=fit,
                physiology=physiology,
            )
        if hypothesis == "H5_supra_vma_flat_only":
            ref = float(physiology["hrr_reference"])
            mask_flat = planned["terrainFamilyResolved"].astype(str).eq("flat")
            planned.loc[~mask_flat, "plannedHrr"] = np.minimum(
                planned.loc[~mask_flat, "plannedHrr"], ref
            )
            planned["meanHrReserve"] = planned["plannedHrr"]

        sim = simulate_adaptive(planned, fit=fit, physiology=physiology)
        pred_sec = float(sim["predictedTimeSec"].sum())
        # Constant baselines (power-law feasible constant HRR)
        const_sim = tpm.simulate_constant_hrr_route(
            route,
            hrr=pl_hrr,
            v_anchor_kmh=float(physiology["vma_flat_kmh"]),
            alpha=float(fit["alpha"]),
            fatigue_coef=float(fit["fatigueCoef"]),
            fatigue_model=str(fit["fatigueModel"]),
            hrr_reference=float(physiology["hrr_reference"]),
            hrr_min_factor=float(physiology["hrr_min_factor"]),
            hrr_max_factor=float(physiology["hrr_max_factor"]),
            decay_lambda=float(physiology["decay_lambda"]),
            min_fatigue_factor=float(physiology["min_fatigue_factor"]),
            fatigue_input_col="cumTrimpBefore",
            gap_steep_threshold=float(physiology["gap_steep_threshold"]),
            gap_soft_start=float(physiology["gap_soft_start"]),
            gap_climb_scale=float(physiology["gap_climb_scale"]),
            gap_descent_scale=float(physiology["gap_descent_scale"]),
        )
        const_sec = float(const_sim["predictedTimeSec"].sum())
        act = activities[activities["activityId"].eq(str(meta["activityId"]))]
        actual = float(act.iloc[0]["movingSec"]) if not act.empty else np.nan
        avg_hr = float(pd.to_numeric(act.iloc[0]["avgHr"], errors="coerce")) if not act.empty else np.nan
        obs_hrr = (avg_hr - hr_rest) / (hr_max - hr_rest) if np.isfinite(avg_hr) else np.nan
        row = {
            "hypothesis": hypothesis,
            "raceKey": race_key,
            "label": meta["label"],
            "profileSource": profile_source,
            "targetMeanHrr": target,
            "plannedMeanHrr": float(planned["plannedHrr"].mean()),
            "predictedSec": pred_sec,
            "constantFeasibleSec": const_sec,
            "actualMovingSec": actual,
            "deltaMin": (pred_sec - actual) / 60.0 if np.isfinite(actual) else np.nan,
            "constantFeasibleDeltaMin": (const_sec - actual) / 60.0 if np.isfinite(actual) else np.nan,
            "observedMeanHrr": obs_hrr,
            "predictedHms": _fmt_hms(pred_sec),
            "actualHms": _fmt_hms(actual) if np.isfinite(actual) else "",
            "feasibleHrr": pl_hrr,
            "empiricalHrr": emp_hrr if np.isfinite(emp_hrr) else np.nan,
            "blendWeight": blend_w,
        }
        rows.append(row)
        sim.to_csv(output_dir / f"{hypothesis}_{race_key}_segments.csv", index=False)
        logger.info(
            "%s %s: target=%.3f pred=%s actual=%s Δ=%+.1f (const Δ=%+.1f)",
            hypothesis,
            race_key,
            target,
            row["predictedHms"],
            row["actualHms"],
            row["deltaMin"],
            row["constantFeasibleDeltaMin"],
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path(
            "data/exp_perf_predictions/trail_digital_twin_paper_section7/segment_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/exp_perf_predictions/trail_digital_twin_segment_hrr_adaptive"),
    )
    parser.add_argument(
        "--hypotheses",
        default=(
            "H1_mean_mod,H2_p75_mod,H3_family_caps,H4_hardtrail_mod,"
            "H10_blend_w70_mean,H10_blend_w70_p75"
        ),
        help="Comma-separated hypothesis ids",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    physiology = dict(DEFAULT_PHYSIOLOGY)
    holdout = set(HOLDOUT_ACTIVITY_IDS) | {"17815897198"}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_train_segments(args.predictions_csv, holdout, cohort="hardRunOrTrailRun")
    fit = fit_stage3(train, physiology, objective="race")
    athlete = pd.read_csv(REPO_ROOT / "data" / "athlete.csv").iloc[0]
    hr_rest, hr_max = float(athlete["hrRest"]), float(athlete["hrMax"])
    power_law_params, windows = fit_hrr_duration_power_law(
        data_dir=REPO_ROOT / "data",
        holdout_ids=holdout,
        hr_rest=hr_rest,
        hr_max=hr_max,
    )
    windows.to_csv(args.output_dir / "hrr_duration_power_law_windows.csv", index=False)

    # Modulator tables
    mod_hard = estimate_hrr_by_terrain(
        args.predictions_csv, holdout, cohort="hardTrailRun", statistic="mean"
    )
    mod_hard.to_csv(args.output_dir / "hrr_by_terrain_hardTrailRun_mean.csv", index=False)
    mod_p75 = estimate_hrr_by_terrain(
        args.predictions_csv, holdout, cohort="hardTrailRun", statistic="p75"
    )
    mod_p75.to_csv(args.output_dir / "hrr_by_terrain_hardTrailRun_p75.csv", index=False)
    mod_all = estimate_hrr_by_terrain(
        args.predictions_csv, holdout, cohort="runTrailOver20Min", statistic="mean"
    )
    mod_all.to_csv(args.output_dir / "hrr_by_terrain_runTrailOver20Min_mean.csv", index=False)

    hyp_map = {
        "H1_mean_mod": ("H1_mean_mod", mod_hard),
        "H2_p75_mod": ("H2_p75_mod", mod_p75),
        "H3_family_caps": ("H3_family_caps", mod_hard),
        "H4_hardtrail_mod": ("H4_hardtrail_mod", mod_hard),  # same as H1; contrast vs H4b
        "H4b_all_usable_mod": ("H4b_all_usable_mod", mod_all),
        "H5_supra_vma_flat_only": ("H5_supra_vma_flat_only", mod_hard),
        "H9_min_pl_emp_p75": ("H9_min_pl_emp_p75", mod_p75),
        "H10_blend_w70_mean": ("H10_blend_w70_mean", mod_hard),
        "H10_blend_w70_p75": ("H10_blend_w70_p75", mod_p75),
        "H10_blend_w50_p75": ("H10_blend_w50_p75", mod_p75),
        "H11_switch6h_p75": ("H11_switch6h_p75", mod_p75),
        "H12_blend_iter_p75": ("H12_blend_iter_p75", mod_p75),
    }
    requested = [h.strip() for h in args.hypotheses.split(",") if h.strip()]
    # Always include H4b when H4 requested for comparison
    if "H4_hardtrail_mod" in requested and "H4b_all_usable_mod" not in requested:
        requested.append("H4b_all_usable_mod")

    frames: list[pd.DataFrame] = []
    for hid in requested:
        if hid not in hyp_map:
            logger.warning("Unknown hypothesis %s", hid)
            continue
        name, mods = hyp_map[hid]
        frame = run_hypothesis(
            hypothesis=name,
            races=RACES,
            fit=fit,
            physiology=physiology,
            power_law_params=power_law_params,
            modulators=mods,
            output_dir=args.output_dir,
            hr_rest=hr_rest,
            hr_max=hr_max,
            windows=windows,
        )
        frames.append(frame)

    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary.to_csv(args.output_dir / "adaptive_hrr_summary.csv", index=False)
    (args.output_dir / "fit_manifest.json").write_text(
        json.dumps(
            {
                "fit": fit,
                "physiology": physiology,
                "powerLaw": {
                    k: power_law_params.get(k)
                    for k in ("coefficient", "exponent", "maeHrr", "fitWindowCount")
                },
                "holdouts": sorted(holdout),
                "hypotheses": requested,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    if not summary.empty:
        print("\n=== Summary (Δ min) ===")
        pivot = summary.pivot_table(
            index="raceKey", columns="hypothesis", values="deltaMin", aggfunc="first"
        )
        print(pivot.round(1).to_string())
        print("\n|Δ| MAE by hypothesis:")
        for hyp, g in summary.groupby("hypothesis"):
            print(f"  {hyp}: {g['deltaMin'].abs().mean():.2f} min")
        print(
            "  constant feasible baseline: "
            f"{summary.groupby('raceKey')['constantFeasibleDeltaMin'].first().abs().mean():.2f} min"
        )
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
