"""Service managing single-session templates for the planner."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from persistence.csv_storage import CsvStorage
from persistence.repositories import PlannedSessionsRepo, SessionTemplatesRepo


def _serialize_payload(payload: Dict[str, Any]) -> str:
    """Serialize payload dictionaries to compact JSON."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _deserialize_payload(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


@dataclass
class SessionTemplatesService:
    storage: CsvStorage

    def __post_init__(self) -> None:
        self.repo = SessionTemplatesRepo(self.storage)
        self.sessions_repo = PlannedSessionsRepo(self.storage)

    # CRUD operations -------------------------------------------------
    def list(self, athlete_id: Optional[str] = None) -> List[Dict[str, Any]]:
        df = self.repo.list(athleteId=athlete_id) if athlete_id else self.repo.list()
        if df.empty:
            return []
        records = df.to_dict(orient="records")
        for rec in records:
            rec["payload"] = _deserialize_payload(rec.get("payloadJson"))
        return records

    def get(self, template_id: str) -> Optional[Dict[str, Any]]:
        row = self.repo.get(template_id)
        if not row:
            return None
        row["payload"] = _deserialize_payload(row.get("payloadJson"))
        return row

    def create(
        self,
        *,
        athlete_id: str,
        title: str,
        base_type: str,
        payload: Dict[str, Any],
        notes: Optional[str] = None,
    ) -> str:
        self._validate_base_type(base_type)
        body = {
            "athleteId": athlete_id,
            "title": title,
            "baseType": base_type,
            "payloadJson": _serialize_payload(payload),
            "notes": notes or "",
            "lastUsedAt": "",
        }
        return self.repo.create(body)

    def update(
        self,
        template_id: str,
        *,
        title: Optional[str] = None,
        base_type: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> None:
        updates: Dict[str, Any] = {}
        if title is not None:
            updates["title"] = title
        if base_type is not None:
            self._validate_base_type(base_type)
            updates["baseType"] = base_type
        if payload is not None:
            updates["payloadJson"] = _serialize_payload(payload)
        if notes is not None:
            updates["notes"] = notes
        if not updates:
            return
        self.repo.update(template_id, updates)

    def delete(self, template_id: str) -> None:
        self.repo.delete(template_id)

    # Template helpers ------------------------------------------------
    def create_from_session(
        self,
        planned_session: Dict[str, Any],
        *,
        title: str,
        notes: Optional[str] = None,
    ) -> str:
        payload = self._session_payload(planned_session)
        base_type = str(planned_session.get("type") or "").upper()
        athlete_id = str(planned_session.get("athleteId") or "")
        return self.create(
            athlete_id=athlete_id,
            title=title,
            base_type=base_type,
            payload=payload,
            notes=notes,
        )

    def duplicate(
        self,
        template_id: str,
        *,
        title: str,
        notes: Optional[str] = None,
    ) -> Optional[str]:
        template = self.get(template_id)
        if not template:
            return None
        return self.create(
            athlete_id=str(template.get("athleteId") or ""),
            title=title,
            base_type=str(template.get("baseType") or ""),
            payload=template.get("payload") or {},
            notes=notes if notes is not None else template.get("notes") or "",
        )

    def apply_to_calendar(
        self,
        template_id: str,
        athlete_id: str,
        target_date: dt.date,
        *,
        notes: Optional[str] = None,
    ) -> Optional[str]:
        template = self.get(template_id)
        if not template:
            return None
        payload = dict(template.get("payload") or {})
        payload.update(
            {
                "athleteId": athlete_id,
                "date": str(target_date),
                "notes": notes if notes is not None else payload.get("notes") or template.get("notes") or "",
            }
        )
        payload["templateTitle"] = template.get("title") or payload.get("templateTitle") or ""
        planned_session_id = self.sessions_repo.create(payload)
        self.repo.update(
            template_id,
            {
                "lastUsedAt": dt.datetime.utcnow().date().isoformat(),
            },
        )
        return planned_session_id

    # Internal utilities ----------------------------------------------
    @staticmethod
    def _validate_base_type(base_type: str) -> None:
        allowed = {"FUNDAMENTAL_ENDURANCE", "LONG_RUN", "INTERVAL_SIMPLE", "RACE"}
        norm = (base_type or "").upper()
        if norm not in allowed:
            raise ValueError(f"Unsupported base type '{base_type}'. Expected one of {sorted(allowed)}.")

    @staticmethod
    def _session_payload(session: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "type",
            "plannedDistanceKm",
            "plannedDurationSec",
            "plannedAscentM",
            "targetType",
            "targetLabel",
            "notes",
            "templateTitle",
            "raceName",
            "stepEndMode",
            "stepsJson",
        ]
        payload = {key: session.get(key) for key in keys}
        payload["athleteId"] = session.get("athleteId")
        return payload
