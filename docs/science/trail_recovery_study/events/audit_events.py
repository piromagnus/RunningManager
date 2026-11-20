"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Count candidate low-speed episodes in the frozen hard-trail cohort.

The procedure uses paceKmh as a speed proxy. Timestamp gaps over 5 seconds
break candidate episodes and are never interpreted as stopping time.
"""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from statistics import median
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = "https://raw.githubusercontent.com/piromagnus/RunningManager/"
HEAD = "5cc439ffbdbacfc66c86a733ec4ace942b4647f6"
DATA = "830ab045367fc91bd77fb40ccdb1316b2b2db88a"


def read_url(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def fetch_activity(activity_id):
    data = read_url(f"{ROOT}{DATA}/data/timeseries/{activity_id}.csv")
    return pd.read_csv(
        io.BytesIO(data), usecols=["timestamp", "paceKmh", "hr", "cadence", "elevationM"]
    )


def analyze(df):
    t = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    speed = pd.to_numeric(df["paceKmh"], errors="coerce")
    hr = pd.to_numeric(df["hr"], errors="coerce")
    elevation = pd.to_numeric(df["elevationM"], errors="coerce")
    delta = t.diff().dt.total_seconds()
    valid = delta.gt(0) & delta.le(5)
    durations = delta[delta.gt(0)].to_numpy()
    summary = {
        "rows": len(df),
        "valid_timestamps": int(t.notna().sum()),
        "speed_present": int(speed.notna().sum()),
        "hr_present": int(hr.notna().sum()),
        "cadence_present": int(df["cadence"].notna().sum()),
        "median_dt": float(np.median(durations)) if len(durations) else None,
        "gaps_over_5": int(delta.gt(5).sum()),
        "max_gap": float(delta.max()) if delta.notna().any() else None,
    }
    events = {}
    # A contiguous strictly low-speed block. Effective duration sums observed
    # intervals only. The proxy is subject to speed smoothing and GPS errors.
    for label, threshold in [("stationary", 1.0), ("slow", 4.0)]:
        low = valid & speed.ge(0) & speed.lt(threshold)
        block = low.ne(low.shift(fill_value=False)).cumsum()
        candidates = []
        for _, indices in df[low].groupby(block[low]).groups.items():
            first, last = int(indices[0]), int(indices[-1])
            duration = float(delta.loc[indices].sum())
            if 60 <= duration <= 600:
                before = (t.iloc[first] - t.iloc[0]).total_seconds()
                after = (t.iloc[-1] - t.iloc[last]).total_seconds()
                event = {"duration": duration, "before": before, "after": after}
                # Five-minute pre/post windows, with recorded movement and no long
                # timestamp gaps. Net grade is a coarse terrain match only.
                if label == "stationary" and before >= 300 and after >= 300:
                    windows = []
                    for lo, hi in [
                        (t.iloc[first] - pd.Timedelta(seconds=300), t.iloc[first]),
                        (t.iloc[last], t.iloc[last] + pd.Timedelta(seconds=300)),
                    ]:
                        ix = t.ge(lo) & t.le(hi)
                        ss, hh, ee, dd = speed[ix], hr[ix], elevation[ix], delta[ix]
                        observed = dd[dd.gt(0) & dd.le(5)].sum()
                        distance = (
                            ss.clip(lower=0) * dd.where(dd.le(5), 0).clip(lower=0) / 3600
                        ).sum()
                        if observed < 270 or distance < 0.05 or ee.notna().sum() < 2:
                            break
                        windows.append(
                            {
                                "speed": distance / (observed / 3600),
                                "hr": float(hh.mean()),
                                "grade": float((ee.iloc[-1] - ee.iloc[0]) / (distance * 1000)),
                            }
                        )
                    if len(windows) == 2:
                        event["pre"] = windows[0]
                        event["post"] = windows[1]
                candidates.append(event)
        events[label] = candidates
    return summary, events


def main():
    support_url = f"{ROOT}{HEAD}/data/exp_perf_predictions/trail_digital_twin_claim_validation/table_data_support.csv"
    support = pd.read_csv(io.BytesIO(read_url(support_url)))
    ids = (
        support.loc[support["cohort_hardTrailRun"].astype(str).str.lower().eq("true"), "activityId"]
        .astype(str)
        .tolist()
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(fetch_activity, ids))
    results = [analyze(frame) for frame in frames]
    report = {"cohort_size": len(ids), "fetched": len(frames)}
    report["rows"] = sum(summary["rows"] for summary, _ in results)
    for key in [
        "valid_timestamps",
        "speed_present",
        "hr_present",
        "cadence_present",
        "gaps_over_5",
    ]:
        report[key] = sum(summary[key] for summary, _ in results)
    report["median_of_activity_median_dt"] = median(summary["median_dt"] for summary, _ in results)
    report["activities_with_gaps_over_5"] = sum(
        summary["gaps_over_5"] > 0 for summary, _ in results
    )
    report["longest_timestamp_gap_s"] = max(summary["max_gap"] for summary, _ in results)
    for label in ["stationary", "slow"]:
        events = [(n, event) for n, (_, ev) in enumerate(results) for event in ev[label]]
        usable = [
            (n, event) for n, event in events if event["before"] >= 300 and event["after"] >= 300
        ]
        report[label] = {
            "events_1_to_10_min": len(events),
            "activities": len({n for n, _ in events}),
            "events_with_5_min_each_side": len(usable),
            "activities_with_5_min_each_side": len({n for n, _ in usable}),
            "median_duration_s": median(event["duration"] for _, event in events)
            if events
            else None,
        }
        if label == "stationary":
            matched = [
                (n, event)
                for n, event in usable
                if "pre" in event
                and abs(event["pre"]["grade"] - event["post"]["grade"]) <= 0.05
                and abs(event["pre"]["hr"] - event["post"]["hr"]) <= 5
                and event["pre"]["speed"] > 1
                and event["post"]["speed"] > 1
            ]
            changes = [
                100 * (event["post"]["speed"] / event["pre"]["speed"] - 1) for _, event in matched
            ]
            report[label]["matched_grade_hr_events"] = len(matched)
            report[label]["matched_grade_hr_activities"] = len({n for n, _ in matched})
            report[label]["matched_median_speed_change_pct"] = (
                float(np.median(changes)) if changes else None
            )
            report[label]["matched_speed_change_iqr_pct"] = (
                [float(x) for x in np.quantile(changes, [0.25, 0.75])] if changes else None
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
