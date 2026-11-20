"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Deterministic demonstration of continuous recovery and two ARD rules.

All values and time constants are synthetic and do not estimate physiology.
"""

import json
import math


def continuous(fatigue, load_rate, duration, tau):
    """Exact propagation for constant q in dF/dt = q - F/tau."""
    return tau * load_rate + (fatigue - tau * load_rate) * math.exp(-duration / tau)


def ard_demo():
    rho = 0.5
    increments = [10.0, 4.0]
    ard_one = 0.0
    ard_infinity = 0.0
    events = []
    for increment in increments:
        before_one = ard_one + increment
        before_infinity = ard_infinity + increment
        ard_one = before_one - rho * increment
        ard_infinity = (1.0 - rho) * before_infinity
        events.append(
            {
                "increment": increment,
                "ard1_before": before_one,
                "ard1_after": ard_one,
                "ardinf_before": before_infinity,
                "ardinf_after": ard_infinity,
            }
        )
    assert events[0]["ard1_after"] == events[0]["ardinf_after"] == 5.0
    assert events[1]["ard1_after"] == 7.0
    assert events[1]["ardinf_after"] == 4.5
    return {"rho": rho, "events": events}


def main():
    tau = 30.0
    hard_rate = 1.0
    before = continuous(0.0, hard_rate, 60.0, tau)
    walk_easy = continuous(before, 0.1, 5.0, tau)
    walk_hard = continuous(before, 0.9, 5.0, tau)
    nonstop = continuous(before, hard_rate, 5.0, tau)
    stops = {str(minutes): continuous(before, 0.0, minutes, tau) for minutes in (1, 3, 5)}
    after_three = stops["3"]
    reprise = {}
    for minutes in (0, 5, 15, 30, 60):
        resumed_stop = continuous(after_three, hard_rate, minutes, tau)
        resumed_control = continuous(before, hard_rate, minutes, tau)
        reprise[str(minutes)] = {
            "stop": resumed_stop,
            "control": resumed_control,
            "difference": resumed_control - resumed_stop,
        }
        assert math.isclose(
            resumed_control - resumed_stop,
            (before - after_three) * math.exp(-minutes / tau),
            rel_tol=1e-12,
        )
    gains = {key: before - value for key, value in stops.items()}
    assert walk_easy < before < walk_hard
    assert before < walk_hard < nonstop
    assert gains["3"] - gains["1"] > gains["5"] - gains["3"]
    assert reprise["60"]["difference"] < reprise["5"]["difference"]
    out = {
        "all_values_synthetic": True,
        "model": "dF/dt = q - F/tau; exact update for constant q",
        "tau_min": tau,
        "preparation": {
            "initial_fatigue": 0,
            "load_per_min": hard_rate,
            "duration_min": 60,
            "fatigue_before_intervention": before,
        },
        "five_minute_scenarios": {
            "initial": before,
            "easy_walk_q_0.1": walk_easy,
            "sustained_walk_q_0.9": walk_hard,
            "continuous_hard_q_1": nonstop,
            "stationary_q_0": stops["5"],
        },
        "stops": {
            "after": stops,
            "gain": gains,
            "marginal_gain_minutes_1_to_3": gains["3"] - gains["1"],
            "marginal_gain_minutes_3_to_5": gains["5"] - gains["3"],
        },
        "resumption_after_three_min_stop_equal_load": reprise,
        "ard": ard_demo(),
        "assertions_passed": True,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
