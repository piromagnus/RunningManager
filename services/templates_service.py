"""Templates service to save/apply week templates as JSON payloads."""

from __future__ import annotations

import json
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from persistence.csv_storage import CsvStorage
from utils.ids import new_id


@dataclass
class TemplatesService:
    storage: CsvStorage

    @property
    def _path(self) -> Path:
        return self.storage.base_dir / "templates.csv"

    def list(self) -> List[Dict[str, Any]]:
        df = self.storage.read_csv("templates.csv")
        return df.to_dict(orient="records") if not df.empty else []

    def save_template(self, name: str, steps: List[Dict[str, Any]]) -> str:
        """Save a week template where steps is a list of session dicts."""
        tid = new_id()
        row = {"templateId": tid, "name": name, "stepsJson": json.dumps(steps)}
        columns = ["templateId", "name", "stepsJson"]
        self.storage.append_row("templates.csv", row, columns)
        return tid

    def get(self, template_id: str) -> Optional[Dict[str, Any]]:
        df = self.storage.read_csv("templates.csv")
        if df.empty:
            return None
        hit = df[df["templateId"] == template_id]
        if hit.empty:
            return None
        row = hit.iloc[0].to_dict()
        try:
            row["steps"] = json.loads(row.get("stepsJson") or "[]")
        except Exception:
            row["steps"] = []
        return row

    # New methods for 4.3
    def save_week_template(self, athlete_id: str, sessions: List[Dict[str, Any]], week_start_date: dt.date, name: str) -> str:
        items: List[Dict[str, Any]] = []
        for s in sessions:
            try:
                s_date = dt.date.fromisoformat(str(s.get("date")))
            except Exception:
                continue
            offset = (s_date - week_start_date).days
            items.append({
                "dateOffset": offset,
                "type": s.get("type"),
                "plannedDistanceKm": s.get("plannedDistanceKm"),
                "plannedDurationSec": s.get("plannedDurationSec"),
                "plannedAscentM": s.get("plannedAscentM"),
                "targetType": s.get("targetType"),
                "targetLabel": s.get("targetLabel"),
                "notes": s.get("notes"),
                "stepEndMode": s.get("stepEndMode"),
                "stepsJson": s.get("stepsJson"),
            })
        tid = new_id()
        row = {"templateId": tid, "athleteId": athlete_id, "name": name, "stepsJson": json.dumps(items, ensure_ascii=False)}
        columns = ["templateId", "athleteId", "name", "stepsJson"]
        self.storage.append_row("templates.csv", row, columns)
        return tid

    def apply_week_template(self, athlete_id: str, template_id: str, target_week_start: dt.date, sessions_repo) -> None:
        row = self.get(template_id)
        if not row:
            return
        try:
            items = json.loads(row.get("stepsJson") or "[]")
        except Exception:
            items = []
        for it in items:
            d = target_week_start + dt.timedelta(days=int(it.get("dateOffset", 0)))
            new_row = {
                "athleteId": athlete_id,
                "date": str(d),
                "type": it.get("type"),
                "plannedDistanceKm": it.get("plannedDistanceKm"),
                "plannedDurationSec": it.get("plannedDurationSec"),
                "plannedAscentM": it.get("plannedAscentM"),
                "targetType": it.get("targetType"),
                "targetLabel": it.get("targetLabel"),
                "notes": it.get("notes"),
                "stepEndMode": it.get("stepEndMode"),
                "stepsJson": it.get("stepsJson"),
            }
            sessions_repo.create(new_row)
