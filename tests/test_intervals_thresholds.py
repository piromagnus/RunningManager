"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from pathlib import Path
import json
from persistence.csv_storage import CsvStorage
from persistence.repositories import PlannedSessionsRepo


def test_interval_target_labels_roundtrip(tmp_path):
    storage = CsvStorage(base_dir=tmp_path)
    repo = PlannedSessionsRepo(storage)
    row = {
        "athleteId": "ath1",
        "date": "2025-10-07",
        "type": "INTERVAL_SIMPLE",
        "plannedDistanceKm": None,
        "plannedDurationSec": None,
        "plannedAscentM": None,
        "targetType": None,
        "targetLabel": None,
        "notes": "",
        "stepEndMode": "lap",
        "stepsJson": json.dumps(
            {
                "warmupSec": 600,
                "repeats": [
                    {
                        "workSec": 60,
                        "recoverSec": 30,
                        "targetType": "hr",
                        "targetLabel": "Threshold 60",
                    },
                    {
                        "workSec": 60,
                        "recoverSec": 30,
                        "targetType": "hr",
                        "targetLabel": "Threshold 30",
                    },
                ],
                "cooldownSec": 600,
            }
        ),
    }
    sid = repo.create(row)
    got = repo.get(sid)
    assert got["stepEndMode"] == "lap"
    sj = got.get("stepsJson")
    assert "Threshold 60" in sj and "Threshold 30" in sj
