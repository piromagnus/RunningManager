"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.config import Config


@dataclass
class TimeseriesService:
    config: Config

    def load(self, activity_id: str) -> Optional[pd.DataFrame]:
        """Return the activity timeseries DataFrame if available."""
        path = self.config.timeseries_dir / f"{activity_id}.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty:
            return None
        return df
