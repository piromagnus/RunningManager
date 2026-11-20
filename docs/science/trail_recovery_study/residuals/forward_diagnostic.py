"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

here = Path(__file__).parent
src = here.parent / "fatigue" / "replay"
segment = pd.read_csv(src / "segment_predictions.csv")
fold = pd.read_csv(src / "fold_predictions.csv")
segment = segment[(segment.validation == "forward") & (segment.variant == "cumulative")].copy()
fold = fold[(fold.validation == "forward") & (fold.variant == "cumulative")].copy()
segment = segment.merge(
    fold[["scenario", "activityIndex", "dateGroup"]],
    on=["scenario", "activityIndex"],
    validate="many_to_one",
)
valid = np.isfinite(segment[["actual_sec", "predicted_sec", "segmentIndex"]]).all(axis=1)
valid &= (segment.actual_sec > 0) & (segment.predicted_sec > 0)
segment = segment[valid].sort_values(["scenario", "dateGroup", "activityIndex", "segmentIndex"])
segment["resid"] = np.log(segment.actual_sec / segment.predicted_sec)
groups = segment.groupby(["scenario", "activityIndex"], sort=False)
segment["previous_index"] = groups.segmentIndex.shift()
segment["x"] = groups.resid.shift()
segment["previous_est_min"] = groups.predicted_sec.shift() / 60
segment["running_mean"] = groups.resid.transform(lambda s: s.shift().expanding().mean())
pairs = segment[
    (segment.segmentIndex == segment.previous_index + 1) & (segment.previous_est_min > 0)
].copy()


def fitted(train):
    x, y, dt = [train[c].to_numpy() for c in ["x", "resid", "previous_est_min"]]
    intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]

    def loss(theta):
        mu, log_tau = theta
        return mu + np.exp(-dt / np.exp(log_tau)) * (x - mu) - y

    fit = least_squares(
        loss, [np.mean(y), np.log(20)], bounds=([-5, np.log(0.05)], [5, np.log(100000)])
    )
    return intercept, slope, fit.x[0], np.exp(fit.x[1])


out, folds = [], []
for (scenario, activity), test in pairs.groupby(["scenario", "activityIndex"], sort=False):
    date_group = test.dateGroup.iloc[0]
    train = pairs[(pairs.scenario == scenario) & (pairs.dateGroup < date_group)]
    n_train_runs = train.activityIndex.nunique()
    if n_train_runs < 5 or len(train) < 20:
        continue
    intercept, slope, mu, tau = fitted(train)
    x, dt = test.x.to_numpy(), test.previous_est_min.to_numpy()
    estimates = {
        "static": np.zeros(len(test)),
        "persistent": x,
        "running_intercept": test.running_mean.to_numpy(),
        "ar1": intercept + slope * x,
        "ou": mu + np.exp(-dt / tau) * (x - mu),
    }
    folds.append(
        {
            "scenario": scenario,
            "activityIndex": int(activity),
            "dateGroup": int(date_group),
            "n_train_runs": int(n_train_runs),
            "n_train_pairs": len(train),
            "n_pairs": len(test),
            "ar_slope": slope,
            "ou_tau_min": tau,
        }
    )
    for model, expected in estimates.items():
        df = test[
            ["scenario", "activityIndex", "segmentIndex", "actual_sec", "predicted_sec", "resid"]
        ].copy()
        df["model"] = model
        df["corrected_sec"] = df.predicted_sec * np.exp(expected)
        df["abs_sec"] = np.abs(df.corrected_sec - df.actual_sec)
        df["abs_log"] = np.abs(df.resid - expected)
        out.append(df)
out = pd.concat(out, ignore_index=True)
per = out.groupby(["scenario", "model", "activityIndex"], as_index=False).agg(
    n=("segmentIndex", "size"), mae_sec=("abs_sec", "mean"), mae_log=("abs_log", "mean")
)
pooled = out.groupby(["scenario", "model"], as_index=False).agg(
    n=("segmentIndex", "size"), mae_sec=("abs_sec", "mean"), mae_log=("abs_log", "mean")
)
equal = per.groupby(["scenario", "model"], as_index=False).agg(
    n_runs=("activityIndex", "size"), mae_sec=("mae_sec", "mean"), mae_log=("mae_log", "mean")
)
comparisons = []
for scenario in sorted(per.scenario.unique()):
    group = per[per.scenario == scenario]
    base = group[group.model == "static"].set_index("activityIndex")
    for model in ["persistent", "running_intercept", "ar1", "ou"]:
        alt = (
            group[group.model == model]
            .set_index("activityIndex")
            .join(base[["mae_sec", "mae_log"]], rsuffix="_static")
        )
        delta = (alt.mae_sec - alt.mae_sec_static).to_numpy()
        rng = np.random.default_rng(42)
        ci = np.quantile(
            np.mean(rng.choice(delta, size=(10000, len(delta)), replace=True), axis=1),
            [0.025, 0.975],
        ).tolist()
        comparisons.append(
            {
                "scenario": scenario,
                "model": model,
                "n_runs": len(delta),
                "mean_paired_difference_sec": float(delta.mean()),
                "ci95_bootstrap_runs": ci,
                "n_improved": int((delta < 0).sum()),
                "n_worse": int((delta > 0).sum()),
            }
        )
out.to_csv(here / "forward_pair_predictions.csv", index=False)
per.to_csv(here / "forward_per_activity.csv", index=False)
pd.DataFrame(folds).to_csv(here / "forward_parameters.csv", index=False)
result = {
    "baseline": "cumulative predicted TRIMP",
    "minimum_history": "5 earlier forward-predicted activities and 20 earlier adjacent pairs",
    "pooled": pooled.to_dict(orient="records"),
    "per_activity": equal.to_dict(orient="records"),
    "paired": comparisons,
    "notes": [
        "All training pairs have strictly earlier dateGroup than target, including no same-day events.",
        "Residuals from prior segments of target activity are available to update the correction.",
        "The oracle scenario uses the full activity observed mean HRR, fixed uses assumed HRR 0.70.",
        "Predicted previous segment duration is the OU time scale.",
        "Unit of actual_sec is moving time, not elapsed time.",
        "Scenario predictions were generated by upstream rolling origin models with >=15 prior activities.",
    ],
}
(here / "forward_summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
