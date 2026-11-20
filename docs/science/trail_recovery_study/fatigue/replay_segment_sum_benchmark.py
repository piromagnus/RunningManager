"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "root" / "replay_segments.csv"
OUT = HERE / "replay_segment_sum"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(SOURCE)
columns = [
    "dateGroup",
    "segmentIndex",
    "distanceKm",
    "gapFactor",
    "altitudePenalty",
    "providedHrr",
    "actualMovingTimeSec",
    "activityMovingTimeSec",
]
for c in columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
valid = (
    np.isfinite(df[columns]).all(axis=1)
    & df.distanceKm.gt(0)
    & df.gapFactor.gt(0)
    & df.altitudePenalty.lt(1)
    & df.actualMovingTimeSec.ge(0)
    & df.activityMovingTimeSec.gt(0)
    & df.providedHrr.between(0, 1.2)
)
print("rows", len(df), "valid", valid.sum(), "excluded", len(df) - valid.sum())
if not valid.all():
    raise ValueError("Invalid segment fields need explicit treatment before running.")
df = df.loc[valid].sort_values(["activityIndex", "segmentIndex"])
activities = []
for key, x in df.groupby("activityIndex", sort=True):
    for col in ("dateGroup", "providedHrr", "activityMovingTimeSec"):
        assert x[col].nunique() == 1, (key, col)
    activities.append(
        dict(
            key=key,
            date=int(x.dateGroup.iloc[0]),
            distance=x.distanceKm.to_numpy(),
            gap=x.gapFactor.to_numpy(),
            altitude=1 - x.altitudePenalty.to_numpy(),
            actual=x.actualMovingTimeSec.to_numpy(),
            target=x.actualMovingTimeSec.sum(),
            hrr=x.providedHrr.iloc[0],
            idx=x.segmentIndex.to_numpy(),
        )
    )
print(
    "activities",
    len(activities),
    "date groups",
    len(set(a["date"] for a in activities)),
    "fit eligible segments",
    df.isFitEligible.sum(),
)

alphas = np.arange(0.35, 1.001, 0.05)
ks = np.arange(0, 1.001, 0.05)
variants = ["none", "cumulative", "per_segment", "time_30m", "time_90m", "time_tuned"]


def simulate(a, alpha, k, variant, hrr):
    h = float(np.clip(hrr, 0, 1.2))
    effort = float(np.clip(h / 0.85, 0.30, 1.2))
    q = h * 0.64 * np.exp(1.92 * h)
    f = 0.0
    pred = []
    for d, g, alt in zip(a["distance"], a["gap"], a["altitude"]):
        speed = max(0.1, 18 * alpha * effort * alt * max(0.5, math.exp(-k * f)) / g)
        t = 3600 * d / speed
        pred.append(t)
        work = q * t / 3600
        if variant == "cumulative":
            f += work
        elif variant == "per_segment":
            f = math.exp(-0.2) * f + work
        elif variant.startswith("time_"):
            tau = int(variant.split("_")[1].rstrip("m")) / 60
            rho = math.exp(-(t / 3600) / tau)
            f = f * rho + q * tau * (1 - rho)
    return np.array(pred)


def metrics(x, y):
    e = y - x
    return dict(
        mae_min=float(np.mean(abs(e)) / 60),
        mape_pct=float(np.mean(abs(e) / x) * 100),
        bias_min=float(np.mean(e) / 60),
        n=len(x),
    )


rows = []
segs = []
selected = []
Y = np.array([a["target"] for a in activities])
dates = np.array([a["date"] for a in activities])
groups = np.arange(len(activities))
for scenario in ["oracle", "fixed"]:
    for variant in variants:
        kinds = ["time_30m", "time_90m"] if variant == "time_tuned" else [variant]
        params = [
            (a, k, kind)
            for kind in kinds
            for a in alphas
            for k in ([0.0] if kind == "none" else ks)
        ]
        predictions = []
        total = []
        for alpha, k, kind in params:
            p = [
                simulate(a, alpha, k, kind, a["hrr"] if scenario == "oracle" else 0.70)
                for a in activities
            ]
            predictions.append(p)
            total.append([v.sum() for v in p])
        total = np.array(total)
        for validation in ["loo", "forward"]:
            for hold in groups:
                train = groups != hold if validation == "loo" else dates < dates[hold]
                if train.sum() < 15:
                    continue
                errs = np.mean(abs(total[:, train] - Y[None, train]) / Y[None, train], axis=1)
                best = int(np.argmin(errs))
                p = predictions[best][hold]
                actual = activities[hold]["actual"]
                a = activities[hold]
                rows.append(
                    dict(
                        scenario=scenario,
                        variant=variant,
                        validation=validation,
                        activityIndex=a["key"],
                        dateGroup=a["date"],
                        n_train=int(train.sum()),
                        actual_sec=Y[hold],
                        predicted_sec=p.sum(),
                    )
                )
                selected.append(
                    dict(
                        scenario=scenario,
                        variant=variant,
                        validation=validation,
                        activityIndex=a["key"],
                        alpha=params[best][0],
                        k=params[best][1],
                        kind=params[best][2],
                    )
                )
                for idx, x, y in zip(a["idx"], actual, p):
                    segs.append(
                        dict(
                            scenario=scenario,
                            variant=variant,
                            validation=validation,
                            activityIndex=a["key"],
                            segmentIndex=idx,
                            actual_sec=x,
                            predicted_sec=y,
                        )
                    )
        print(scenario, variant, flush=True)
pred = pd.DataFrame(rows)
seg = pd.DataFrame(segs)
sel = pd.DataFrame(selected)
pred.to_csv(OUT / "fold_predictions.csv", index=False)
seg.to_csv(OUT / "segment_predictions.csv", index=False)
sel.to_csv(OUT / "fold_parameters.csv", index=False)
summary = []
for (scenario, validation, variant), g in pred.groupby(
    ["scenario", "validation", "variant"], sort=False
):
    z = seg[(seg.scenario == scenario) & (seg.validation == validation) & (seg.variant == variant)]
    summary.append(
        dict(
            scenario=scenario,
            validation=validation,
            variant=variant,
            **metrics(g.actual_sec.to_numpy(), g.predicted_sec.to_numpy()),
            segment_mae_sec=float(np.mean(abs(z.predicted_sec - z.actual_sec))),
            n_segments=len(z),
        )
    )
pd.DataFrame(summary).to_csv(OUT / "metrics.csv", index=False)
rng = np.random.default_rng(20261120)
ci = []
for scenario in ["oracle", "fixed"]:
    for validation in ["loo", "forward"]:
        d = pred[(pred.scenario == scenario) & (pred.validation == validation)]
        wide = d.pivot(index="activityIndex", columns="variant", values="predicted_sec")
        actual = d[d.variant == "none"].set_index("activityIndex").loc[wide.index, "actual_sec"]
        for other in ["none", "cumulative", "per_segment", "time_90m", "time_tuned"]:
            diff = (
                abs(wide[other] - actual) / actual - abs(wide["time_30m"] - actual) / actual
            ).to_numpy() * 100
            samples = np.mean(diff[rng.integers(0, len(diff), size=(10000, len(diff)))], axis=1)
            ci.append(
                dict(
                    scenario=scenario,
                    validation=validation,
                    comparator=other,
                    delta_mape_pp=float(diff.mean()),
                    ci_low=float(np.quantile(samples, 0.025)),
                    ci_high=float(np.quantile(samples, 0.975)),
                    n=len(diff),
                )
            )
pd.DataFrame(ci).to_csv(OUT / "paired_bootstrap.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False))
print(pd.DataFrame(ci).to_string(index=False))
