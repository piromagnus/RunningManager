"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import datetime as dt

import pytest

from persistence.csv_storage import CsvStorage
from persistence.repositories import PlannedSessionsRepo
from services.session_templates_service import SessionTemplatesService


def test_session_templates_crud_and_apply(tmp_path):
    storage = CsvStorage(base_dir=tmp_path)
    service = SessionTemplatesService(storage)

    base_payload = {
        "type": "FUNDAMENTAL_ENDURANCE",
        "plannedDistanceKm": 12.0,
        "plannedDurationSec": 3600,
        "plannedAscentM": 400,
        "targetType": "pace",
        "targetLabel": "Fundamental",
        "notes": "easy run",
        "stepEndMode": None,
        "stepsJson": None,
    }
    template_id = service.create(
        athlete_id="athlete-1",
        title="Fundamentale",
        base_type="FUNDAMENTAL_ENDURANCE",
        payload=base_payload,
        notes="template notes",
    )
    assert template_id

    template = service.get(template_id)
    assert template is not None
    assert template["title"] == "Fundamentale"
    assert template["payload"]["plannedDistanceKm"] == 12.0

    updated_payload = dict(base_payload)
    updated_payload["plannedDistanceKm"] = 15.0
    service.update(
        template_id,
        title="Endurance fondamentale",
        base_type="FUNDAMENTAL_ENDURANCE",
        payload=updated_payload,
        notes="updated",
    )
    template = service.get(template_id)
    assert template["title"] == "Endurance fondamentale"
    assert template["payload"]["plannedDistanceKm"] == 15.0
    assert template["notes"] == "updated"

    duplicate_id = service.duplicate(template_id, title="Endurance copie")
    assert duplicate_id is not None
    assert duplicate_id != template_id

    session_row = {
        "athleteId": "athlete-1",
        "type": "LONG_RUN",
        "plannedDistanceKm": 30.0,
        "plannedDurationSec": 10800,
        "plannedAscentM": 1200,
        "targetType": "hr",
        "targetLabel": "Threshold 30",
        "notes": "weekend long run",
        "stepEndMode": None,
        "stepsJson": None,
    }
    from_session_id = service.create_from_session(session_row, title="Long run plan")
    from_session_template = service.get(from_session_id)
    assert from_session_template is not None
    assert from_session_template["payload"]["plannedDurationSec"] == 10800

    apply_date = dt.date(2025, 1, 1)
    planned_session_id = service.apply_to_calendar(
        template_id, "athlete-1", apply_date, notes="calendar note"
    )
    assert planned_session_id is not None

    sessions_repo = PlannedSessionsRepo(storage)
    planned_session = sessions_repo.get(planned_session_id)
    assert planned_session is not None
    assert planned_session["date"] == str(apply_date)
    assert planned_session["notes"] == "calendar note"

    refreshed_template = service.get(template_id)
    assert refreshed_template["lastUsedAt"] == dt.datetime.utcnow().date().isoformat()
