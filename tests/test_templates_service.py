import datetime as dt
import json

from persistence.csv_storage import CsvStorage
from persistence.repositories import PlannedSessionsRepo
from services.templates_service import TemplatesService


def test_save_and_apply_week_template(tmp_path):
    storage = CsvStorage(base_dir=tmp_path)
    service = TemplatesService(storage)
    sessions_repo = PlannedSessionsRepo(storage)

    week_start = dt.date(2024, 1, 1)
    steps_payload = json.dumps({"warmupSec": 300, "loops": []}, separators=(",", ":"))
    sessions = [
        {
            "date": "2024-01-01",
            "type": "FUNDAMENTAL_ENDURANCE",
            "plannedDurationSec": 3600,
            "plannedDistanceKm": "",
            "plannedAscentM": "",
            "targetType": "hr",
            "targetLabel": "Threshold 60",
            "notes": "Base endurance",
            "stepEndMode": "auto",
            "stepsJson": "",
        },
        {
            "date": "2024-01-03",
            "type": "INTERVAL_SIMPLE",
            "plannedDurationSec": 1800,
            "plannedDistanceKm": "",
            "plannedAscentM": "",
            "targetType": "pace",
            "targetLabel": "Threshold 60",
            "notes": "Intervals",
            "stepEndMode": "lap",
            "stepsJson": steps_payload,
        },
    ]

    template_id = service.save_week_template("ath-1", sessions, week_start, "Race prep")
    records = service.list("ath-1")
    assert len(records) == 1
    assert records[0]["templateId"] == template_id

    payload = json.loads(records[0]["stepsJson"])
    assert len(payload) == 2
    assert payload[0]["dateOffset"] == 0
    assert payload[1]["dateOffset"] == 2
    assert payload[1]["targetLabel"] == "Threshold 60"
    assert payload[1]["stepsJson"] == steps_payload

    # Applying should create new planned sessions with shifted dates
    target_week = dt.date(2024, 1, 8)
    service.apply_week_template("ath-1", template_id, target_week, sessions_repo)

    applied_df = sessions_repo.list(athleteId="ath-1")
    assert len(applied_df) == 2
    dates = set(applied_df["date"].tolist())
    assert dates == {"2024-01-08", "2024-01-10"}
    assert set(applied_df["targetLabel"].dropna().tolist()) == {"Threshold 60"}
    saved_steps = applied_df.set_index("date").loc["2024-01-10", "stepsJson"]
    assert saved_steps == steps_payload

    # Templates for another athlete should not be mixed in
    assert service.list("ath-2") == []
