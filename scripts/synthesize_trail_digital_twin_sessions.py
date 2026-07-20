"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Consolidate trail digital-twin benchmark results at the session (activity) level.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EXP_DIR = REPO_ROOT / "data" / "exp_perf_predictions"
DEFAULT_ACTIVITIES = REPO_ROOT / "data" / "activities.csv"
DEFAULT_OUTPUT_MD = (
    REPO_ROOT / "docs" / "science" / "trail_digital_twin_session_benchmark_review.md"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "data"
    / "exp_perf_predictions"
    / "session_benchmark_review"
    / "session_benchmark_review.csv"
)
DEFAULT_SESSION_SOURCE = "trail_digital_twin_boundary_best_profile"
DEFAULT_HYPER_SOURCE = "trail_digital_twin_hypothesis_refined"
STAGE3_LOO = "Stage 3 HRR speed ratio LOO"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 2) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "—"
    return f"{parsed:.{digits}f}"


def _load_leaderboard(exp_dir: Path, folder: str) -> pd.DataFrame:
    path = exp_dir / folder / "benchmark_leaderboard.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _find_activity_loo(exp_dir: Path, folder: str) -> Path | None:
    direct = exp_dir / folder / "activity_loo_predictions.csv"
    if direct.exists():
        return direct
    runs = exp_dir / folder / "runs"
    if not runs.is_dir():
        return None
    candidates = sorted(runs.glob("*/activity_loo_predictions.csv"))
    return candidates[0] if candidates else None


def build_session_frame(
    *,
    exp_dir: Path,
    session_source: str,
    activities_path: Path,
) -> pd.DataFrame:
    loo_path = _find_activity_loo(exp_dir, session_source)
    if loo_path is None:
        raise FileNotFoundError(
            f"No activity_loo_predictions.csv under {exp_dir / session_source}"
        )
    loo = pd.read_csv(loo_path)
    stage3 = loo[
        loo["stage"].astype(str).eq(STAGE3_LOO)
        & loo["fitObjective"].astype(str).eq("activity")
    ].copy()
    if stage3.empty:
        raise ValueError(f"No Stage 3 LOO activity rows in {loo_path}")

    stage3["activityId"] = stage3["activityId"].astype(str)
    stage3["actualMin"] = pd.to_numeric(stage3["actualTimeSec"], errors="coerce") / 60.0
    stage3["predictedMin"] = pd.to_numeric(stage3["predictedTimeSec"], errors="coerce") / 60.0
    stage3["errorMin"] = pd.to_numeric(stage3["errorSec"], errors="coerce") / 60.0
    stage3["absErrorMin"] = stage3["errorMin"].abs()
    stage3["errorPct"] = pd.to_numeric(stage3["errorPct"], errors="coerce")
    stage3["sourceExperiment"] = session_source

    if activities_path.exists():
        activities = pd.read_csv(activities_path)
        activities["activityId"] = activities["activityId"].astype(str)
        keep = [
            col
            for col in [
                "activityId",
                "name",
                "category",
                "distanceKm",
                "ascentM",
                "movingSec",
                "elapsedSec",
            ]
            if col in activities.columns
        ]
        stage3 = stage3.merge(activities[keep], on="activityId", how="left")
    return stage3.sort_values(["cohort", "absErrorMin"], ascending=[True, False])


def _cohort_summary(sessions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in sessions.groupby("cohort", sort=True):
        rows.append(
            {
                "cohort": cohort,
                "sessionCount": int(len(group)),
                "maeMin": float(group["absErrorMin"].mean()),
                "medianAbsErrorMin": float(group["absErrorMin"].median()),
                "mapePct": float(group["errorPct"].abs().mean()),
                "biasMin": float(group["errorMin"].mean()),
                "worstAbsErrorMin": float(group["absErrorMin"].max()),
            }
        )
    return pd.DataFrame(rows)


def _useful_elements_from_hyper(exp_dir: Path, hyper_source: str) -> dict[str, Any]:
    leaderboard = _load_leaderboard(exp_dir, hyper_source)
    stage = exp_dir / hyper_source / "benchmark_stage_metrics.csv"
    fitted = exp_dir / hyper_source / "benchmark_fitted_parameters.csv"
    strata = exp_dir / hyper_source / "benchmark_activity_error_strata.csv"
    segment_types = exp_dir / hyper_source / "benchmark_segment_type_metrics.csv"
    out: dict[str, Any] = {"experiment": hyper_source}
    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        out["winner"] = {
            "runId": best.get("runId"),
            "meanStage3MaeMin": _safe_float(best.get("meanStage3MaeMin")),
            "meanStage3MapePct": _safe_float(best.get("meanStage3MapePct")),
            "meanStage3R2": _safe_float(best.get("meanStage3R2")),
            "hrrReference": _safe_float(best.get("hrrReference")),
            "hrrMinFactor": _safe_float(best.get("hrrMinFactor")),
            "hrrMaxFactor": _safe_float(best.get("hrrMaxFactor")),
            "decayLambda": _safe_float(best.get("decayLambda")),
            "minFatigueFactor": _safe_float(best.get("minFatigueFactor")),
            "overrideJson": best.get("overrideJson"),
        }
    if stage.exists():
        metrics = pd.read_csv(stage)
        winner_id = out.get("winner", {}).get("runId")
        if winner_id is not None:
            metrics = metrics[metrics["runId"].astype(str).eq(str(winner_id))]
        stage3 = metrics[
            metrics["stage"].astype(str).eq(STAGE3_LOO)
            & metrics["fitObjective"].astype(str).eq("activity")
        ]
        out["cohortMae"] = (
            stage3[["cohort", "maeMin", "mapePct", "r2", "biasMin"]]
            .sort_values("maeMin")
            .to_dict(orient="records")
            if not stage3.empty
            else []
        )
    if fitted.exists():
        params = pd.read_csv(fitted)
        winner_id = out.get("winner", {}).get("runId")
        if winner_id is not None:
            params = params[params["runId"].astype(str).eq(str(winner_id))]
        stage3_params = params[params["stage"].astype(str).eq("Stage 3 HRR speed ratio")]
        keep_cols = [
            col
            for col in [
                "cohort",
                "alpha",
                "fatigueCoef",
                "fatigueModel",
                "fatigueState",
                "secondaryFatigueCoef",
                "raceMaeSec",
                "segmentMaeSec",
            ]
            if col in stage3_params.columns
        ]
        out["fittedParams"] = (
            stage3_params[keep_cols].to_dict(orient="records") if keep_cols else []
        )
    if strata.exists():
        strata_df = pd.read_csv(strata)
        winner_id = out.get("winner", {}).get("runId")
        if winner_id is not None:
            strata_df = strata_df[strata_df["runId"].astype(str).eq(str(winner_id))]
        hard = strata_df[strata_df["cohort"].astype(str).eq("hardTrailRun")]
        out["hardTrailStrata"] = (
            hard.sort_values("maeMin", ascending=False)
            .head(12)[
                [c for c in ["strataType", "strataValue", "activityCount", "maeMin", "biasMin"] if c in hard.columns]
            ]
            .to_dict(orient="records")
            if not hard.empty
            else []
        )
    if segment_types.exists():
        seg = pd.read_csv(segment_types)
        winner_id = out.get("winner", {}).get("runId")
        if winner_id is not None:
            seg = seg[seg["runId"].astype(str).eq(str(winner_id))]
        hard = seg[seg["cohort"].astype(str).eq("hardTrailRun")]
        out["hardTrailTerrain"] = (
            hard.sort_values("maeMin", ascending=False)[
                [
                    c
                    for c in [
                        "terrainFamily",
                        "terrainLabel",
                        "segmentCount",
                        "maeMin",
                        "biasMin",
                        "mapePct",
                    ]
                    if c in hard.columns
                ]
            ]
            .to_dict(orient="records")
            if not hard.empty
            else []
        )
    return out


def render_markdown(
    *,
    sessions: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    useful: dict[str, Any],
    session_source: str,
) -> str:
    winner = useful.get("winner") or {}
    lines = [
        "# Trail Digital Twin Session-Level Benchmark Review",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "Consolidates hyperparameter-tuning winners with **session (activity/race)** LOO",
        "errors, highlighting the most useful elements for continued modeling.",
        "",
        "## Most Useful Elements",
        "",
        "1. **Operational physiology defaults** (current-code hypothesis winner):",
        f"   `hrr_reference={_fmt(winner.get('hrrReference'), 2)}`,",
        f"   `hrr_min_factor={_fmt(winner.get('hrrMinFactor'), 2)}`,",
        f"   `hrr_max_factor={_fmt(winner.get('hrrMaxFactor'), 2)}`,",
        f"   `decay_lambda={_fmt(winner.get('decayLambda'), 2)}`,",
        f"   `min_fatigue_factor={_fmt(winner.get('minFatigueFactor'), 2)}`.",
        f"   Aggregate Stage 3 LOO MAE ≈ **{_fmt(winner.get('meanStage3MaeMin'))} min**",
        f"   (MAPE {_fmt(winner.get('meanStage3MapePct'))}%, R² {_fmt(winner.get('meanStage3R2'), 3)}).",
        "2. **Stage 3 HRR speed-ratio + decayed TRIMP** dominates earlier stages;",
        "   muscular secondary fatigue usually stays at 0 but should remain in the grid.",
        "3. **Session residuals** are the actionable unit: large positive errors often mean",
        "   the athlete was slower than the model (stops, nutrition, pacing issues,",
        "   device-open idle time); large negatives can mean unusually strong execution.",
        "4. **Hard-trail high-duration / high-TRIMP strata** remain the main residual risk;",
        "   terrain families (steep descent / steep climb) still drive segment MAE.",
        "5. **Segment stationary exclusion** (new): fit on moving segments, then score the",
        "   full race so excluded idle time still appears as informative residual.",
        "",
        "## Hyperparameter Winner Snapshot",
        "",
        f"- Source experiment: `{useful.get('experiment')}`",
        f"- Winner run: `{winner.get('runId', '—')}`",
        "",
        "### Cohort Stage 3 LOO (winner run)",
        "",
        "| cohort | MAE min | MAPE % | R2 | bias min |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in useful.get("cohortMae") or []:
        lines.append(
            f"| {row.get('cohort')} | {_fmt(row.get('maeMin'))} | {_fmt(row.get('mapePct'))} | "
            f"{_fmt(row.get('r2'), 3)} | {_fmt(row.get('biasMin'))} |"
        )
    lines.extend(
        [
            "",
            "### Fitted Stage 3 parameters (winner run)",
            "",
            "| cohort | alpha | fatigueCoef | model | state | secondary |",
            "| --- | ---: | ---: | --- | --- | ---: |",
        ]
    )
    for row in useful.get("fittedParams") or []:
        lines.append(
            f"| {row.get('cohort')} | {_fmt(row.get('alpha'), 2)} | {_fmt(row.get('fatigueCoef'), 2)} | "
            f"{row.get('fatigueModel')} | {row.get('fatigueState')} | {_fmt(row.get('secondaryFatigueCoef'), 2)} |"
        )

    lines.extend(
        [
            "",
            "## Session-Level LOO Review",
            "",
            f"Source LOO table: `{session_source}` (Stage 3 HRR speed ratio LOO, activity objective).",
            "Note: archived session LOO may use an earlier physiology profile than the",
            "hypothesis winner; use it for residual triage, then re-run with the winner",
            "config when timeseries are available.",
            "",
            "### Cohort session summary",
            "",
            "| cohort | n | MAE min | median |abs| | MAPE % | bias min | worst |abs| |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in cohort_summary.iterrows():
        lines.append(
            f"| {row['cohort']} | {int(row['sessionCount'])} | {_fmt(row['maeMin'])} | "
            f"{_fmt(row['medianAbsErrorMin'])} | {_fmt(row['mapePct'])} | {_fmt(row['biasMin'])} | "
            f"{_fmt(row['worstAbsErrorMin'])} |"
        )

    worst = sessions.nlargest(12, "absErrorMin")
    lines.extend(
        [
            "",
            "### Largest absolute session residuals",
            "",
            "| activity | name | cohort | actual min | pred min | error min | error % | α | κ |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in worst.iterrows():
        name = str(row.get("name") or "")[:48]
        lines.append(
            f"| {row['activityId']} | {name} | {row['cohort']} | {_fmt(row['actualMin'])} | "
            f"{_fmt(row['predictedMin'])} | {_fmt(row['errorMin'])} | {_fmt(row['errorPct'])} | "
            f"{_fmt(row.get('alpha'), 2)} | {_fmt(row.get('fatigueCoef'), 2)} |"
        )

    lines.extend(
        [
            "",
            "### Hard-trail error strata (hypothesis winner)",
            "",
            "| strata | value | n | MAE min | bias min |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in useful.get("hardTrailStrata") or []:
        lines.append(
            f"| {row.get('strataType')} | {row.get('strataValue')} | {row.get('activityCount')} | "
            f"{_fmt(row.get('maeMin'))} | {_fmt(row.get('biasMin'))} |"
        )

    lines.extend(
        [
            "",
            "### Hard-trail terrain segment MAE (hypothesis winner)",
            "",
            "| terrain | segments | MAE min | bias min | MAPE % |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in useful.get("hardTrailTerrain") or []:
        label = row.get("terrainLabel") or row.get("terrainFamily")
        lines.append(
            f"| {label} | {row.get('segmentCount')} | {_fmt(row.get('maeMin'))} | "
            f"{_fmt(row.get('biasMin'))} | {_fmt(row.get('mapePct'))} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- **Positive error** (pred > actual): model too slow / athlete faster than twin.",
            "- **Negative error** (pred < actual): model too fast / athlete slower — often",
            "  includes aid stops, walking, or device-open idle segments.",
            "- Prefer diagnosing large residuals with segment exclusion enabled: if fit-eligible",
            "  MAE improves while full-race MAE stays high, the gap is non-model time.",
            "",
            "## Next Experiments",
            "",
            "1. Run `configs/trail_digital_twin_benchmark_segment_exclusion.yaml` when",
            "   `data/timeseries` is available.",
            "2. Compare baseline vs exclusion using full-race Stage 3 LOO MAE and per-session",
            "   `excludedTimeSec` from `segment_qc` / LOO columns.",
            "3. Re-generate this report after the exclusion sweep with",
            "   `--session-source trail_digital_twin_segment_exclusion`.",
            "",
            "## Regenerator",
            "",
            "```bash",
            "uv run python scripts/synthesize_trail_digital_twin_sessions.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument("--session-source", default=DEFAULT_SESSION_SOURCE)
    parser.add_argument("--hyper-source", default=DEFAULT_HYPER_SOURCE)
    parser.add_argument("--activities", type=Path, default=DEFAULT_ACTIVITIES)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args(argv)

    sessions = build_session_frame(
        exp_dir=args.exp_dir,
        session_source=args.session_source,
        activities_path=args.activities,
    )
    cohort_summary = _cohort_summary(sessions)
    useful = _useful_elements_from_hyper(args.exp_dir, args.hyper_source)
    markdown = render_markdown(
        sessions=sessions,
        cohort_summary=cohort_summary,
        useful=useful,
        session_source=args.session_source,
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(args.output_csv, index=False)
    meta_path = args.output_csv.with_name("session_benchmark_useful_elements.json")
    meta_path.write_text(json.dumps(useful, indent=2, default=str))
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
