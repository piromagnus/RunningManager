"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

here = Path(__file__).parent
df = pd.read_csv(here / "segments.csv")
df = df[df.cohort == "hardTrailRun"].copy()
for c in ["actualTimeSec", "stage3PredictedTimeSec", "distanceKm", "segmentIndex"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
valid = np.isfinite(
    df[["actualTimeSec", "stage3PredictedTimeSec", "distanceKm", "segmentIndex"]]
).all(axis=1)
valid &= (df["actualTimeSec"] > 0) & (df["stage3PredictedTimeSec"] > 0) & (df["distanceKm"] > 0)
df = df[valid].sort_values(["activityIndex", "segmentIndex"]).copy()
df["resid"] = np.log(df["actualTimeSec"] / df["stage3PredictedTimeSec"])
group = df.groupby("activityIndex", sort=False)
df["previous_index"] = group["segmentIndex"].shift()
df["x"] = group["resid"].shift()
df["running_mean"] = group["resid"].transform(lambda s: s.shift().expanding().mean())
df["delta_min"] = group["stage3PredictedTimeSec"].shift() / 60.0
pairs = df[(df.segmentIndex == df.previous_index + 1) & (df.delta_min > 0)].copy()


def fit(training):
    x, y, dt = (training[c].to_numpy() for c in ["x", "resid", "delta_min"])
    A = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(A, y, rcond=None)[0]

    def loss(params):
        mu, log_tau = params
        return mu + np.exp(-dt / np.exp(log_tau)) * (x - mu) - y

    fit = least_squares(
        loss, [np.mean(y), np.log(20.0)], bounds=([-5, np.log(0.05)], [5, np.log(1e5)])
    )
    return intercept, slope, fit.x[0], np.exp(fit.x[1])


def expected(frame, param, model):
    x, dt = frame["x"].to_numpy(), frame["delta_min"].to_numpy()
    intercept, slope, mu, tau = param
    if model == "static":
        return np.zeros(len(x))
    if model == "persistent":
        return x
    if model == "running_intercept":
        return frame["running_mean"].to_numpy()
    if model == "ar1":
        return intercept + slope * x
    return mu + np.exp(-dt / tau) * (x - mu)


records = []
parameters = []
models = ["static", "persistent", "running_intercept", "ar1", "ou"]
for activity in pairs.activityIndex.unique():
    train, test = pairs[pairs.activityIndex != activity], pairs[pairs.activityIndex == activity]
    param = fit(train)
    parameters.append(
        dict(
            activityIndex=activity,
            ar_intercept=param[0],
            ar_slope=param[1],
            ou_mu=param[2],
            ou_tau_min=param[3],
            n_train=len(train),
            n_test=len(test),
        )
    )
    for model in models:
        yp = expected(test, param, model)
        trerr = train["resid"].to_numpy() - expected(train, param, model)
        lo, hi = np.quantile(trerr, [0.05, 0.95])
        result = test[
            ["activityIndex", "segmentIndex", "actualTimeSec", "stage3PredictedTimeSec", "resid"]
        ].copy()
        result["model"] = model
        result["predicted_log_residual"] = yp
        result["predicted_sec"] = result["stage3PredictedTimeSec"] * np.exp(yp)
        result["abs_log_error"] = np.abs(result["resid"] - yp)
        result["abs_sec_error"] = np.abs(result["actualTimeSec"] - result["predicted_sec"])
        result["covered_90"] = (result["resid"] >= yp + lo) & (result["resid"] <= yp + hi)
        result["width_90_sec"] = result["stage3PredictedTimeSec"] * (
            np.exp(yp + hi) - np.exp(yp + lo)
        )
        records.append(result)
out = pd.concat(records, ignore_index=True)
per_activity = out.groupby(["model", "activityIndex"], as_index=False).agg(
    n=("segmentIndex", "size"),
    mae_sec=("abs_sec_error", "mean"),
    mae_log=("abs_log_error", "mean"),
    coverage=("covered_90", "mean"),
    width_sec=("width_90_sec", "mean"),
)
pooled = out.groupby("model", as_index=False).agg(
    n=("segmentIndex", "size"),
    mae_sec=("abs_sec_error", "mean"),
    mae_log=("abs_log_error", "mean"),
    coverage=("covered_90", "mean"),
    width_sec=("width_90_sec", "mean"),
)
byrun = per_activity.groupby("model", as_index=False).agg(
    n_runs=("activityIndex", "size"),
    avg_run_mae_sec=("mae_sec", "mean"),
    avg_run_mae_log=("mae_log", "mean"),
    avg_run_coverage=("coverage", "mean"),
)
baseline = per_activity[per_activity.model == "static"].set_index("activityIndex")
comparisons = {}
for model in models[1:]:
    alt = (
        per_activity[per_activity.model == model]
        .set_index("activityIndex")
        .join(baseline[["mae_sec", "mae_log"]], rsuffix="_static")
    )
    rng = np.random.default_rng(42)
    d = (alt.mae_sec - alt.mae_sec_static).to_numpy()
    boot = np.mean(rng.choice(d, size=(10000, len(d)), replace=True), axis=1)
    comparisons[model] = {
        "n_runs": len(d),
        "mean_paired_sec": float(np.mean(d)),
        "median_paired_sec": float(np.median(d)),
        "runs_improved": int((d < 0).sum()),
        "runs_worse": int((d > 0).sum()),
        "bootstrap_95_ci_mean_delta_sec": np.quantile(boot, [0.025, 0.975]).tolist(),
        "mean_paired_log": float((alt.mae_log - alt.mae_log_static).mean()),
    }
out.to_csv(here / "pair_predictions.csv", index=False)
per_activity.to_csv(here / "per_activity.csv", index=False)
pd.DataFrame(parameters).to_csv(here / "fold_parameters.csv", index=False)
summary = {
    "source_n_rows": len(df),
    "n_activities": int(df.activityIndex.nunique()),
    "n_pairs": len(pairs),
    "n_pair_activities": int(pairs.activityIndex.nunique()),
    "pair_segment_lengths": pairs.groupby("activityIndex").size().describe().to_dict(),
    "pooled": pooled.to_dict(orient="records"),
    "activity_weighted": byrun.to_dict(orient="records"),
    "comparisons": comparisons,
    "parameters_describe": pd.DataFrame(parameters)[["ar_slope", "ou_tau_min"]]
    .describe()
    .to_dict(),
    "notes": [
        "held-out activity base predictions fitted beforehand and observed HRR used",
        "using predicted previous segment duration as OU time lag",
        "intervals use training fold empirical residual quantiles and are evaluated on held-out segments",
        "bootstrap resamples activities and evaluates paired mean differences only",
    ],
}
(here / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
