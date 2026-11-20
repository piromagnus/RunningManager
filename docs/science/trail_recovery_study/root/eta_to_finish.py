"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Replay moving-time ETA at course checkpoints using strictly earlier runs."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("forecasts_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pair = pd.read_csv(args.forecasts_dir / "segment_predictions.csv")
    fold = pd.read_csv(args.forecasts_dir / "fold_predictions.csv")
    pair = pair.query("validation == 'forward' and variant == 'cumulative'").copy()
    fold = fold.query("validation == 'forward' and variant == 'cumulative'")
    pair = pair.merge(
        fold[["scenario", "activityIndex", "dateGroup"]], on=["scenario", "activityIndex"]
    )
    pair = pair.sort_values(["scenario", "activityIndex", "segmentIndex"])
    valid = np.isfinite(pair[["actual_sec", "predicted_sec"]]).all(axis=1)
    pair["residual"] = np.log(
        (pair["actual_sec"] / pair["predicted_sec"]).where(
            valid & pair["actual_sec"].gt(0) & pair["predicted_sec"].gt(0)
        )
    )
    group = pair.groupby(["scenario", "activityIndex"])
    pair["previous_residual"] = group["residual"].shift()
    pair["previous_index"] = group["segmentIndex"].shift()
    neighbors = pair[
        pair["segmentIndex"].eq(pair["previous_index"] + 1)
        & pair["residual"].notna()
        & pair["previous_residual"].notna()
    ]
    rows = []
    clipped_slopes = 0
    for (scenario, activity), run in group:
        if (
            len(run) < 4
            or run["residual"].isna().any()
            or not np.all(np.diff(run["segmentIndex"]) == 1)
        ):
            continue
        history = neighbors[
            neighbors["scenario"].eq(scenario) & neighbors["dateGroup"].lt(run["dateGroup"].iloc[0])
        ]
        if len(history) < 20 or history["activityIndex"].nunique() < 5:
            continue
        x = history["previous_residual"].to_numpy()
        y = history["residual"].to_numpy()
        slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1]
        rho = float(np.clip(slope, 0.0, 0.99))
        clipped_slopes += int(slope != rho)
        offset = y.mean() - rho * x.mean()
        actual = run["actual_sec"].to_numpy()
        predicted = run["predicted_sec"].to_numpy()
        residual = run["residual"].to_numpy()
        for checkpoint in (0.25, 0.50, 0.75):
            seen = int(np.floor(checkpoint * len(run)))
            state = residual[seen - 1]
            rest = predicted[seen:]
            states = np.empty(len(rest))
            for i in range(len(rest)):
                state = offset + rho * state
                states[i] = state
            predictions = {
                "static": rest.sum(),
                "last_residual": (rest * np.exp(residual[seen - 1])).sum(),
                "mean_residual": (rest * np.exp(residual[:seen].mean())).sum(),
                "ar1": (rest * np.exp(states)).sum(),
            }
            for method, estimate in predictions.items():
                rows.append(
                    (scenario, checkpoint, activity, method, estimate - actual[seen:].sum())
                )
    errors = pd.DataFrame(rows, columns=["scenario", "checkpoint", "activity", "method", "error"])
    errors["absolute"] = errors["error"].abs()
    result = errors.groupby(["scenario", "checkpoint", "method"], as_index=False).agg(
        n=("activity", "size"), mae_sec=("absolute", "mean"), bias_sec=("error", "mean")
    )
    rng = np.random.default_rng(20261120)
    for (scenario, checkpoint), block in errors.groupby(["scenario", "checkpoint"]):
        matrix = block.pivot(index="activity", columns="method", values="absolute")
        take = rng.integers(len(matrix), size=(10000, len(matrix)))
        for alternative in ("static", "last_residual", "mean_residual"):
            delta = (matrix["ar1"] - matrix[alternative]).to_numpy()
            low, high = np.quantile(delta[take].mean(axis=1), [0.025, 0.975])
            mask = (
                result["scenario"].eq(scenario)
                & result["checkpoint"].eq(checkpoint)
                & result["method"].eq(alternative)
            )
            result.loc[mask, ["ar1_minus_method_sec", "ic95_low_sec", "ic95_high_sec"]] = [
                delta.mean(),
                low,
                high,
            ]
    result.to_csv(args.output, index=False)
    print(f"slopes clipped: {clipped_slopes}")
    print(result.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
