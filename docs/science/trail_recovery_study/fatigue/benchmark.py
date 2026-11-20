"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "segments.csv")
df = raw.loc[raw.cohort.eq("hardTrailRun")].copy()
for c in [
    "segmentIndex",
    "distanceKm",
    "gapFactor",
    "altitudePenalty",
    "meanHrReserve",
    "actualTimeSec",
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
valid = np.isfinite(
    df[
        [
            "segmentIndex",
            "distanceKm",
            "gapFactor",
            "altitudePenalty",
            "meanHrReserve",
            "actualTimeSec",
        ]
    ]
).all(axis=1)
valid &= (
    df.distanceKm.gt(0)
    & df.gapFactor.gt(0)
    & df.actualTimeSec.gt(0)
    & df.meanHrReserve.ge(0)
    & df.meanHrReserve.le(1.2)
    & df.altitudePenalty.lt(1)
)
df = df.loc[valid].sort_values(["activityIndex", "segmentIndex"])
activities = []
for key, x in df.groupby("activityIndex", sort=True):
    hrr = np.average(x.meanHrReserve.to_numpy(), weights=x.actualTimeSec.to_numpy())
    activities.append(
        dict(
            key=key,
            distance=x.distanceKm.to_numpy(),
            gap=x.gapFactor.to_numpy(),
            altitude=1 - x.altitudePenalty.to_numpy(),
            actual=x.actualTimeSec.to_numpy(),
            hrr=hrr,
            idx=x.segmentIndex.to_numpy(),
        )
    )
print(
    "activities",
    len(activities),
    "segments",
    len(df),
    "dropped",
    len(raw.loc[raw.cohort.eq("hardTrailRun")]) - len(df),
    flush=True,
)

# Parameter grid same across models, per-model memory coefficient fixed a priori.
alphas = np.arange(0.35, 1.001, 0.05)
ks = np.arange(0, 0.401, 0.05)
variants = ["none", "cumulative", "per_segment", "time_30m", "time_90m"]


# TRIMP rate in the repository is in hours, yielding raw load units/hour.
def simulate(a, alpha, k, variant, prescribed):
    h = a["hrr"] if prescribed == "oracle" else 0.70
    h = float(np.clip(h, 0, 1.2))
    effort = float(np.clip(h / 0.85, 0.30, 1.2))
    q = h * 0.64 * np.exp(1.92 * h)
    f = 0.0
    arr = []
    for d, g, alt in zip(a["distance"], a["gap"], a["altitude"]):
        fat = max(0.5, math.exp(-k * f))
        speed = max(0.1, 18 * alpha * effort * alt * fat / g)
        t = 3600 * d / speed
        arr.append(t)
        work = q * t / 3600
        if variant == "cumulative":
            f += work
        elif variant == "per_segment":
            f = math.exp(-0.2) * f + work
        elif variant.startswith("time_"):
            tau = int(variant.split("_")[1].rstrip("m")) / 60
            rho = math.exp(-(t / 3600) / tau)
            f = f * rho + q * tau * (1 - rho)
    return np.array(arr)


def metric(x, y):
    e = y - x
    return dict(
        mae_min=float(np.mean(abs(e)) / 60),
        mape_pct=float(np.mean(abs(e) / x) * 100),
        bias_min=float(np.mean(e) / 60),
        n=len(x),
    )


rows = []
segment = []
selections = []
for prescribed in ["oracle", "fixed"]:
    for variant in variants:
        params = [(a, k) for a in alphas for k in ([0.0] if variant == "none" else ks)]
        Y = np.array([a["actual"].sum() for a in activities])
        predictions = []
        total = []
        for alpha, k in params:
            p = [simulate(a, alpha, k, variant, prescribed) for a in activities]
            predictions.append(p)
            total.append([v.sum() for v in p])
        total = np.array(total)
        for hold in range(len(activities)):
            train = np.arange(len(activities)) != hold
            # MAPE objective at activity level; deterministic tie resolution grid order.
            error = np.mean(abs(total[:, train] - Y[None, train]) / Y[None, train], axis=1)
            best = int(np.argmin(error))
            p = predictions[best][hold]
            x = activities[hold]["actual"]
            rows.append(
                dict(
                    scenario=prescribed,
                    variant=variant,
                    activityIndex=activities[hold]["key"],
                    actual_sec=Y[hold],
                    predicted_sec=p.sum(),
                )
            )
            selections.append(
                dict(
                    scenario=prescribed,
                    variant=variant,
                    activityIndex=activities[hold]["key"],
                    alpha=params[best][0],
                    k=params[best][1],
                )
            )
            for idx, actual, pred in zip(activities[hold]["idx"], x, p):
                segment.append(
                    dict(
                        scenario=prescribed,
                        variant=variant,
                        activityIndex=activities[hold]["key"],
                        segmentIndex=idx,
                        actual_sec=actual,
                        predicted_sec=pred,
                    )
                )
        print(prescribed, variant, flush=True)
pred = pd.DataFrame(rows)
seg = pd.DataFrame(segment)
sel = pd.DataFrame(selections)
pred.to_csv(HERE / "fold_predictions.csv", index=False)
seg.to_csv(HERE / "segment_predictions.csv", index=False)
sel.to_csv(HERE / "fold_parameters.csv", index=False)
summary = []
for (scenario, variant), g in pred.groupby(["scenario", "variant"], sort=False):
    z = seg[(seg.scenario == scenario) & (seg.variant == variant)]
    summary.append(
        dict(
            scenario=scenario,
            variant=variant,
            **metric(g.actual_sec.to_numpy(), g.predicted_sec.to_numpy()),
            segment_mae_sec=float(np.mean(abs(z.predicted_sec - z.actual_sec))),
            segment_mape_pct=float(
                np.mean(abs(z.predicted_sec - z.actual_sec) / z.actual_sec) * 100
            ),
        )
    )
pd.DataFrame(summary).to_csv(HERE / "metrics.csv", index=False)
# Paired bootstrap by activity, deterministic seed. Positive delta favors time_30m.
rng = np.random.default_rng(20261120)
ci = []
for scenario in ["oracle", "fixed"]:
    wide = pred[pred.scenario == scenario].pivot(
        index="activityIndex", columns="variant", values="predicted_sec"
    )
    actual = (
        pred[(pred.scenario == scenario) & (pred.variant == "none")]
        .set_index("activityIndex")
        .loc[wide.index, "actual_sec"]
    )
    for variant in ["none", "cumulative", "per_segment", "time_90m"]:
        delta = (
            abs(wide[variant] - actual) / actual - abs(wide["time_30m"] - actual) / actual
        ).to_numpy() * 100
        samples = np.mean(delta[rng.integers(0, len(delta), size=(10000, len(delta)))], axis=1)
        ci.append(
            dict(
                scenario=scenario,
                comparator=variant,
                delta_mape_pp=float(delta.mean()),
                ci_low=float(np.quantile(samples, 0.025)),
                ci_high=float(np.quantile(samples, 0.975)),
            )
        )
pd.DataFrame(ci).to_csv(HERE / "paired_bootstrap.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False), flush=True)
print(pd.DataFrame(ci).to_string(index=False), flush=True)


# Analytical exact update checks F <- F exp(-dt/tau) + q*tau*(1-exp(-dt/tau)).
def update(f, q, dt, tau):
    e = math.exp(-dt / tau)
    return f * e + q * tau * (1 - e)


f, tau, q = 10.0, 30.0, 0.1
checks = {
    "half_life": abs(update(f, 0, tau * math.log(2), tau) - f / 2) < 1e-10,
    "composition": abs(update(update(f, q, 7, tau), q, 11, tau) - update(f, q, 18, tau)) < 1e-10,
    "walking_decreases": update(f, q, 5, tau) < f and f > tau * q,
    "walking_can_increase": update(1, q, 5, tau) > 1,
    "stopping_decreases": update(f, 0, 5, tau) < f,
    "nonnegative": update(0, q, 5, tau) >= 0,
}
print("checks", checks, flush=True)
pd.Series(checks).to_csv(HERE / "deterministic_checks.csv")
assert all(checks.values())
