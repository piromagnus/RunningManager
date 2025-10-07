import json

import pytest


def test_estimate_km_uses_fundamental_threshold(planner):
    km = planner.estimate_km("ath-1", duration_sec=2700)
    assert km == pytest.approx((2700 / 3600) * 10.0, rel=1e-3)


def test_estimate_km_falls_back_to_recent_easy_pace(planner):
    df = planner.thresholds.list(athleteId="ath-1")
    fund = df[df["name"] == "Fundamental"].iloc[0]
    planner.thresholds.update(
        fund["thresholdId"],
        {
            "paceFlatKmhMin": "",
            "paceFlatKmhMax": "",
        },
    )
    for dist_km, moving_sec, start in [
        (9.0, 3600, "2024-01-05T07:00:00"),
        (11.0, 3600, "2024-01-04T07:00:00"),
        (10.0, 3600, "2024-01-03T07:00:00"),
    ]:
        planner.activities.create(
            {
                "athleteId": "ath-1",
                "distanceKm": dist_km,
                "movingSec": moving_sec,
                "startTime": start,
            }
        )
    km = planner.estimate_km("ath-1", duration_sec=3600)
    # Median pace = 10 km/h
    assert km == pytest.approx(10.0, rel=1e-3)


def test_interval_duration_supports_loops_structure(planner, interval_steps_loops):
    assert planner.estimate_interval_duration_sec(interval_steps_loops) == 300 + 180 + (2 * (120 + 60) + 45)


def test_between_loop_recoveries_counted_once(planner, interval_steps_loops):
    loops = json.loads(json.dumps(interval_steps_loops))
    loops["loops"][0]["repeats"] = 3
    loops["betweenLoopRecoverSec"] = 90
    duration = planner.estimate_interval_duration_sec(loops)
    # warmup + cooldown + repeats + two gaps of 90s (repeats-1)
    expected = 300 + 180 + (3 * (120 + 60)) + (2 * 90)
    assert duration == expected


def test_interval_distance_loops_use_threshold_pace(planner, interval_steps_loops):
    km = planner.estimate_interval_distance_km("ath-1", interval_steps_loops)
    warm = (300 / 3600) * 10.0
    loops = 2 * ((120 / 3600) * 14.0 + (60 / 3600) * 10.0)
    between = (1 * ((45 / 3600) * 10.0))
    cool = (180 / 3600) * 10.0
    assert km == pytest.approx(warm + loops + between + cool, rel=1e-3)


def test_interval_distance_backcompat_repeats(planner, interval_steps_legacy):
    km = planner.estimate_interval_distance_km("ath-1", interval_steps_legacy)
    warm = (300 / 3600) * 10.0
    first = (90 / 3600) * 14.0 + (45 / 3600) * 10.0
    second = (60 / 3600) * 10.0 + (30 / 3600) * 10.0
    cool = (180 / 3600) * 10.0
    assert km == pytest.approx(warm + first + second + cool, rel=1e-3)


def test_interval_ascent_loops(planner, interval_steps_loops):
    asc = planner.estimate_interval_ascent_m(interval_steps_loops)
    # repeats=2 so (20 + 0) * 2 = 40
    assert asc == 40


def test_estimate_session_distance_prefers_planned(planner):
    row = {
        "type": "LONG_RUN",
        "plannedDistanceKm": 25.0,
        "plannedDurationSec": 7200,
    }
    assert planner.estimate_session_distance_km("ath-1", row) == 25.0


def test_estimate_session_distance_fundamental_uses_duration(planner):
    row = {
        "type": "FUNDAMENTAL_ENDURANCE",
        "plannedDurationSec": 5400,
        "plannedDistanceKm": "",
    }
    km = planner.estimate_session_distance_km("ath-1", row)
    assert km == pytest.approx(planner.estimate_km("ath-1", 5400), rel=1e-6)


def test_estimate_session_distance_interval_uses_steps(planner, interval_steps_loops):
    row = {
        "type": "INTERVAL_SIMPLE",
        "stepsJson": json.dumps(interval_steps_loops),
        "plannedDistanceKm": "",
    }
    km = planner.estimate_session_distance_km("ath-1", row)
    assert km == pytest.approx(planner.estimate_interval_distance_km("ath-1", interval_steps_loops), rel=1e-6)


def test_estimate_session_distance_returns_none_when_unknown(planner):
    row = {
        "type": "LONG_RUN",
        "plannedDurationSec": "",
        "plannedDistanceKm": "",
    }
    assert planner.estimate_session_distance_km("ath-1", row) is None


def test_estimate_session_ascent_prefers_planned(planner):
    row = {
        "type": "LONG_RUN",
        "plannedAscentM": 450,
        "stepsJson": "",
    }
    assert planner.estimate_session_ascent_m("ath-1", row) == 450


def test_estimate_session_ascent_intervals(planner, interval_steps_loops):
    row = {
        "type": "INTERVAL_SIMPLE",
        "stepsJson": json.dumps(interval_steps_loops),
    }
    assert planner.estimate_session_ascent_m("ath-1", row) == planner.estimate_interval_ascent_m(interval_steps_loops)


def test_compute_weekly_totals(planner, week_sessions):
    totals = planner.compute_weekly_totals("ath-1", week_sessions)
    fundamental_est = planner.estimate_session_distance_km("ath-1", week_sessions[0])
    interval_est = planner.estimate_session_distance_km("ath-1", week_sessions[2])
    expected_distance = 20.0 + fundamental_est + interval_est
    expected_time = 3600 + 7200 + 1800
    expected_ascent = 600 + planner.estimate_session_ascent_m("ath-1", week_sessions[2])
    assert totals["timeSec"] == expected_time
    assert totals["distanceKm"] == pytest.approx(expected_distance, rel=1e-6)
    assert totals["ascentM"] == expected_ascent


def test_compute_weekly_totals_empty(planner):
    totals = planner.compute_weekly_totals("ath-1", [])
    assert totals == {"timeSec": 0, "distanceKm": 0.0, "ascentM": 0}
