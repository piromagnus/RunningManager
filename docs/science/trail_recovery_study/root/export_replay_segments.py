"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Export the archived PR #5 E5 cohort without Strava IDs, dates or GPS."""

import sys
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))

    import pandas as pd

    from services import trail_digital_twin_pipeline as pipeline
    from services import trail_performance_model as model

    result_dir = repo / "data/exp_perf_predictions/trail_digital_twin_claim_validation"
    archived = pd.read_csv(
        result_dir / "table_causal_fatigue_folds_hardTrailRun.csv",
        dtype={"activityId": str},
    )
    config = pipeline.load_config(args.config)
    activities, _, rest, maximum, _ = pipeline._load_activity_inputs(config, repo)
    cohort = activities[activities["activityId"].isin(archived["activityId"])].copy()
    cohort["dateGroup"] = cohort["startDate"].rank(method="dense").astype(int)
    cohort = cohort.sort_values(["dateGroup", "activityId"]).reset_index(drop=True)
    cohort["activityIndex"] = cohort.index + 1
    segments, _, _ = pipeline._build_segments(cohort, config, repo, rest, maximum)
    phys = config["physiology"]
    route = segments.copy()
    route["gapFactor"] = model._gap_factors_for_segments(
        segments,
        steep_threshold=phys["gap_steep_threshold"],
        soft_start=phys["gap_soft_start"],
        climb_scale=phys["gap_climb_scale"],
        descent_scale=phys["gap_descent_scale"],
    )
    route["altitudePenalty"] = 1 - route["meanAltitudeM"].map(model.altitude_factor)
    route["cohort"] = "hardTrailRun"
    metadata = cohort[
        ["activityId", "activityIndex", "dateGroup", "actualTimeSec", "hrReserveRatio"]
    ]
    metadata = metadata.rename(
        columns={"actualTimeSec": "activityMovingTimeSec", "hrReserveRatio": "providedHrr"}
    )
    route = route.merge(metadata, on="activityId", validate="many_to_one")
    columns = [
        "cohort",
        "activityIndex",
        "dateGroup",
        "segmentIndex",
        "distanceKm",
        "gapFactor",
        "altitudePenalty",
        "meanHrReserve",
        "providedHrr",
        "actualTimeSec",
        "actualMovingTimeSec",
        "activityMovingTimeSec",
        "stationaryTimeSec",
        "isFitEligible",
    ]
    route = route[columns].sort_values(["activityIndex", "segmentIndex"])
    assert route["activityIndex"].nunique() == len(cohort) == archived["activityId"].nunique()
    assert route["dateGroup"].gt(0).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    route.to_csv(args.output, index=False)
    print(
        f"{len(route)} segments, {len(cohort)} activities, {route['dateGroup'].nunique()} date groups, "
        f"{int(route['isFitEligible'].sum())} eligible segments"
    )


if __name__ == "__main__":
    main()
