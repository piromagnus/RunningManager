from services.planner_presenter import build_card_view_model, build_empty_state_placeholder


def test_build_card_view_model_with_estimated_distance():
    session = {
        "plannedSessionId": "abc",
        "type": "fundamental_endurance",
        "plannedDurationSec": 3600,
        "plannedDistanceKm": "",
        "targetType": "hr",
        "targetLabel": "Threshold 60",
        "stepEndMode": "auto",
    }
    model = build_card_view_model(session, estimated_distance_km=10.0, distance_eq_km=11.2)
    assert model["header"] == "FUNDAMENTAL_ENDURANCE | dur=3600s"
    assert any(badge == "est≈10,0 km" for badge in model["badges"])
    assert any(badge == "deq=11,2 km" for badge in model["badges"])
    assert any("target=hr Threshold 60" == badge for badge in model["badges"])
    assert any(action.action == "view" for action in model["actions"])
    assert any(action.action == "save-template" for action in model["actions"])
    assert len(model["actions"]) == 4


def test_build_card_view_model_prefers_planned_distance():
    session = {
        "plannedSessionId": "xyz",
        "type": "INTERVAL_SIMPLE",
        "plannedDurationSec": 1800,
        "plannedDistanceKm": 20.0,
    }
    model = build_card_view_model(session, estimated_distance_km=5.0, distance_eq_km=22.0)
    assert any(badge.startswith("dist=20,0 km") for badge in model["badges"])
    assert any(badge == "deq=22,0 km" for badge in model["badges"])
    assert all(action.session_id == "xyz" for action in model["actions"])


def test_build_empty_state_placeholder():
    placeholder = build_empty_state_placeholder("Monday")
    assert placeholder["message"].startswith("No sessions planned for Monday")
    assert "Click ＋" in placeholder["cta"]
