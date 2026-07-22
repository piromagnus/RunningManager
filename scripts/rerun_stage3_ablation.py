"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Re-run Stage-3 component ablation without re-fitting the M0–M3 ladder.

Rebuilds segments/cohorts from the paper §7 config, restores Stage-3 fatigue
choices from ``table_fitted_parameters.csv``, then writes a fresh
``table_stage3_ablation.csv`` / ``table_frozen_stage3_ablation.csv`` using the
configured ablation protocol (default: reoptimize + LOO).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services import trail_digital_twin_pipeline as pipeline  # noqa: E402
from services import trail_performance_model as tpm  # noqa: E402

logger = logging.getLogger(__name__)


def _restore_stage3_best(
    fitted_params: pd.DataFrame,
    cohorts: dict[str, pd.DataFrame],
    segment_features: pd.DataFrame,
    objectives: list[str],
) -> tuple[dict[tuple[str, str], dict[str, object]], dict[tuple[str, str], pd.DataFrame]]:
    stage3_best: dict[tuple[str, str], dict[str, object]] = {}
    stage3_segments: dict[tuple[str, str], pd.DataFrame] = {}
    stage_rows = fitted_params[fitted_params["stage"].astype(str).eq("Stage 3 HRR speed ratio")].copy()
    for cohort_name, cohort_df in cohorts.items():
        ids = cohort_df["activityId"].astype(str).tolist()
        cohort_segments = segment_features[segment_features["activityId"].astype(str).isin(ids)].copy()
        if cohort_segments.empty:
            continue
        for objective in objectives:
            match = stage_rows[
                stage_rows["cohort"].astype(str).eq(cohort_name)
                & stage_rows["fitObjective"].astype(str).eq(objective)
            ]
            if match.empty:
                logger.warning("No Stage-3 fitted params for cohort=%s objective=%s", cohort_name, objective)
                continue
            row = match.iloc[0]
            stage3_best[(cohort_name, objective)] = {
                "alpha": float(row["alpha"]),
                "fatigueCoef": float(row["fatigueCoef"]),
                "fatigueModel": str(row.get("fatigueModel", "linear")),
                "fatigueState": str(row.get("fatigueState", "")),
                "acuteTrimpCol": str(row.get("acuteTrimpCol", "decayedTrimpBefore")),
                "secondaryAcuteTrimpCol": str(row.get("secondaryAcuteTrimpCol", "") or ""),
                "secondaryFatigueModel": str(row.get("secondaryFatigueModel", "") or ""),
                "secondaryFatigueCoef": float(row.get("secondaryFatigueCoef", 0.0) or 0.0),
            }
            stage3_segments[(cohort_name, objective)] = cohort_segments.copy()
    return stage3_best, stage3_segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "trail_digital_twin_paper_section7.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to paths.output_dir from the config",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = pipeline.load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / str(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted_path = output_dir / "table_fitted_parameters.csv"
    if not fitted_path.exists():
        raise FileNotFoundError(f"missing fitted parameters: {fitted_path}")

    root = REPO_ROOT
    activity_df, _daily, hr_rest, hr_max, _v_vt2 = pipeline._load_activity_inputs(config, root)
    all_segments_df, _qc_df, segments_by_activity = pipeline._build_segments(
        activity_df, config, root, hr_rest, hr_max
    )
    if all_segments_df.empty:
        raise RuntimeError("no usable activity segments were built")
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
    all_segments_with_trimp = tpm.add_in_activity_trimp_features(
        all_segments_df,
        decay_lambda=float(config["physiology"]["decay_lambda"]),
    )
    segment_features = pipeline.add_segment_model_features(all_segments_with_trimp, activity_df)

    fitted_params = pd.read_csv(fitted_path)
    objectives = list(config["fitting"].get("ablation_fit_objectives") or config["fitting"]["enabled_objectives"])
    stage3_best, stage3_segments = _restore_stage3_best(
        fitted_params, cohorts, segment_features, objectives
    )
    if not stage3_best:
        raise RuntimeError("could not restore any Stage-3 cohort/objective pairs for ablation")

    stage_metrics_path = output_dir / "table_stage_metrics.csv"
    stage3_loo_baseline = {}
    if stage_metrics_path.exists():
        stage3_loo_baseline = pipeline.stage3_loo_baseline_from_metrics(pd.read_csv(stage_metrics_path))
        logger.info("Loaded Stage-3 LOO baseline for %d cohort/objective pairs (R1)", len(stage3_loo_baseline))
    else:
        logger.warning("Missing %s; ablation full row will re-run LOO independently", stage_metrics_path)

    logger.info(
        "Running Stage-3 ablation protocol=%s on %d cohort/objective pairs",
        config["fitting"]["ablation_protocol"],
        len(stage3_best),
    )
    ablation = pipeline.run_stage3_ablation(
        stage3_best,
        stage3_segments,
        cohorts,
        config,
        stage3_loo_baseline=stage3_loo_baseline,
    )
    ablation_path = output_dir / "table_stage3_ablation.csv"
    frozen_path = output_dir / "table_frozen_stage3_ablation.csv"
    ablation.to_csv(ablation_path, index=False)
    ablation.to_csv(frozen_path, index=False)
    logger.info("Wrote %s (%d rows)", ablation_path, len(ablation))
    logger.info("Wrote %s", frozen_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
