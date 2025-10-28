import json

from services.planner_presenter import build_card_view_model, build_empty_state_placeholder
from utils.formatting import fmt_m


def test_build_card_view_model_with_estimated_distance():
    session = {
        "plannedSessionId": "abc",
        "type": "fundamental_endurance",
        "plannedDurationSec": 3600,
        "plannedDistanceKm": "",
        "plannedAscentM": 250,
        "targetType": "hr",
        "targetLabel": "Threshold 60",
        "stepEndMode": "auto",
    }
    model = build_card_view_model(session, estimated_distance_km=10.0, distance_eq_km=11.2)
    assert model["header"] == "Fundamental Endurance • 1h00"
    assert "≈10,0 km" in model["meta"]
    assert "DEQ 11,2 km" in model["meta"]
    assert "hr Threshold 60" in model["meta"]
    assert "mode auto" in model["meta"]
    expected_ascent_label = fmt_m(250)
    assert any(expected_ascent_label in meta for meta in model["meta"])
    assert any(action.action == "view" for action in model["actions"])
    assert any(action.action == "save-template" for action in model["actions"])
    assert len(model["actions"]) == 4


def test_build_card_view_model_prefers_planned_distance():
    session = {
        "plannedSessionId": "xyz",
        "type": "INTERVAL_SIMPLE",
        "plannedDurationSec": 1800,
        "plannedDistanceKm": 20.0,
        "stepsJson": json.dumps(
            {
                "preBlocks": [
                    {"kind": "run", "sec": 300, "targetType": "pace", "targetLabel": "Fundamental"}
                ],
                "loops": [
                    {
                        "repeats": 2,
                        "actions": [
                            {
                                "kind": "run",
                                "sec": 120,
                                "targetType": "pace",
                                "targetLabel": "Threshold 60",
                                "ascendM": 10,
                                "descendM": 0,
                            },
                            {"kind": "recovery", "sec": 60},
                        ],
                    }
                ],
                "betweenBlock": {"kind": "recovery", "sec": 45},
                "postBlocks": [{"kind": "recovery", "sec": 300}],
                "warmupSec": 300,
                "cooldownSec": 300,
                "betweenLoopRecoverSec": 45,
            }
        ),
    }
    model = build_card_view_model(session, estimated_distance_km=5.0, distance_eq_km=22.0)
    assert "20,0 km" in model["meta"]
    assert "DEQ 22,0 km" in model["meta"]
    assert all(action.session_id == "xyz" for action in model["actions"])
    assert model["sections"]
    assert model["sections"][0].title == "Avant"
    assert any("Run" in line for line in model["sections"][0].lines)
    loop_section = next(
        section for section in model["sections"] if section.title.startswith("Boucle")
    )
    assert "Run" in loop_section.lines[0]
    assert any("Recovery" in line for section in model["sections"] for line in section.lines)


def test_build_empty_state_placeholder():
    placeholder = build_empty_state_placeholder("Monday")
    assert placeholder["message"].startswith("No sessions planned for Monday")
    assert "Click ＋" in placeholder["cta"]
