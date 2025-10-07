import json
import sys
import types
from pathlib import Path

import pytest


class _DummyLock:
    """Minimal stand-in for portalocker.Lock during tests."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


sys.modules.setdefault(
    "portalocker",
    types.SimpleNamespace(Lock=_DummyLock, LOCK_SH=1, LOCK_EX=2),
)


if "babel" not in sys.modules:
    class _FakeBabelNumbers:
        @staticmethod
        def format_decimal(value, format=None, locale=None):
            if format == "#":
                digits = 0
            elif format and "." in format:
                digits = len(format.split(".")[-1])
            elif isinstance(value, float):
                digits = 2
            else:
                digits = 0
            if digits == 0:
                try:
                    result = str(int(round(float(value))))
                except Exception:
                    result = str(value)
            else:
                try:
                    result = f"{float(value):.{digits}f}"
                except Exception:
                    result = str(value)
            return result.replace(".", ",")


    sys.modules.setdefault(
        "babel",
        types.SimpleNamespace(numbers=_FakeBabelNumbers()),
    )


# Ensure project root is importable for tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from persistence.csv_storage import CsvStorage
from persistence.repositories import ThresholdsRepo
from services.planner_service import PlannerService


@pytest.fixture
def planner(tmp_path):
    storage = CsvStorage(base_dir=tmp_path)
    thresholds = ThresholdsRepo(storage)
    thresholds.create(
        {
            "athleteId": "ath-1",
            "name": "Fundamental",
            "paceFlatKmhMin": 9.0,
            "paceFlatKmhMax": 11.0,
        }
    )
    thresholds.create(
        {
            "athleteId": "ath-1",
            "name": "Threshold 60",
            "paceFlatKmhMin": 13.0,
            "paceFlatKmhMax": 15.0,
        }
    )
    thresholds.create(
        {
            "athleteId": "ath-1",
            "name": "Threshold 30",
            "paceFlatKmhMin": 11.0,
            "paceFlatKmhMax": 12.5,
        }
    )
    return PlannerService(storage)


@pytest.fixture
def interval_steps_legacy():
    return {
        "warmupSec": 300,
        "cooldownSec": 180,
        "repeats": [
            {
                "workSec": 90,
                "recoverSec": 45,
                "targetType": "pace",
                "targetLabel": "Threshold 60",
            },
            {
                "workSec": 60,
                "recoverSec": 30,
                "targetType": "sensation",
                "targetLabel": "tempo",
            },
        ],
    }


@pytest.fixture
def interval_steps_loops():
    return {
        "warmupSec": 300,
        "cooldownSec": 180,
        "betweenLoopRecoverSec": 45,
        "loops": [
            {
                "repeats": 2,
                "actions": [
                    {
                        "kind": "run",
                        "sec": 120,
                        "targetType": "pace",
                        "targetLabel": "Threshold 60",
                        "ascendM": 20,
                    },
                    {
                        "kind": "recovery",
                        "sec": 60,
                        "targetType": "hr",
                        "targetLabel": "Fundamental",
                        "ascendM": 0,
                    },
                ],
            }
        ],
    }


@pytest.fixture
def week_sessions(interval_steps_loops):
    return [
        {
            "plannedSessionId": "s1",
            "athleteId": "ath-1",
            "type": "FUNDAMENTAL_ENDURANCE",
            "plannedDurationSec": 3600,
            "plannedDistanceKm": "",
            "plannedAscentM": "",
        },
        {
            "plannedSessionId": "s2",
            "athleteId": "ath-1",
            "type": "LONG_RUN",
            "plannedDurationSec": 7200,
            "plannedDistanceKm": 20.0,
            "plannedAscentM": 600,
        },
        {
            "plannedSessionId": "s3",
            "athleteId": "ath-1",
            "type": "INTERVAL_SIMPLE",
            "plannedDurationSec": 1800,
            "plannedDistanceKm": "",
            "plannedAscentM": "",
            "stepsJson": json.dumps(interval_steps_loops),
        },
    ]
