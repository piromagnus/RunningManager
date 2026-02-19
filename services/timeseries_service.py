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

    def load_metrics_ts(self, activity_id: str) -> Optional[pd.DataFrame]:
        """Return the cached metrics timeseries DataFrame if available.
        
        The metrics_ts files contain preprocessed data like speedeq_smooth,
        grade_ma_10, elevationM_ma_5, cumulated_distance, etc.
        """
        path = self.config.metrics_ts_dir / f"{activity_id}.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty:
            return None
        return df

    def has_elevation_metrics(self, activity_id: str) -> bool:
        """Check if cached elevation metrics are available for an activity.
        
        Returns True if metrics_ts file exists and contains the required
        elevation profile columns (cumulated_distance, elevationM_ma_5, grade_ma_10).
        """
        df = self.load_metrics_ts(activity_id)
        if df is None:
            return False
        
        required_cols = ["cumulated_distance", "elevationM_ma_5", "grade_ma_10"]
        return all(col in df.columns for col in required_cols)
