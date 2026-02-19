"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd
from streamlit.logger import get_logger

from persistence.csv_storage import CsvStorage
from utils.ids import new_id

logger = get_logger(__name__)


class RacePersistence:
    """Persistence helpers for pacer races and segments."""

    def __init__(
        self,
        storage: CsvStorage,
        compute_aid_station_times: Callable[[list[float], pd.DataFrame], list[float]],
        compute_aid_station_stats: Callable[[list[float], pd.DataFrame], list[dict]],
    ) -> None:
        self.storage = storage
        self.compute_aid_station_times = compute_aid_station_times
        self.compute_aid_station_stats = compute_aid_station_stats

    def save_race(
        self,
        race_name: str,
        aid_stations_km: list[float],
        segments_df: pd.DataFrame,
        race_id: Optional[str] = None,
        aid_stations_times: Optional[list[float]] = None,
    ) -> str:
        """Save race pacing data to CSV files."""
        if race_id is None:
            race_id = new_id()

        race_pacing_dir = self.storage.base_dir / "race_pacing"
        race_pacing_dir.mkdir(parents=True, exist_ok=True)

        if aid_stations_times is None:
            aid_stations_times = self.compute_aid_station_times(aid_stations_km, segments_df)

        races_file = self.storage.base_dir / "races.csv"
        races_df = self.storage.read_csv(races_file)

        segment_stats = self.compute_aid_station_stats(aid_stations_km, segments_df)

        if races_df.empty:
            races_df = pd.DataFrame(
                columns=[
                    "raceId",
                    "name",
                    "createdAt",
                    "aidStationsKm",
                    "aidStationsTimes",
                    "aidStationsDistEq",
                    "aidStationsElevGain",
                    "aidStationsElevLoss",
                ]
            )

        if "aidStationsTimes" not in races_df.columns:
            races_df["aidStationsTimes"] = ""
        if "aidStationsDistEq" not in races_df.columns:
            races_df["aidStationsDistEq"] = ""
        if "aidStationsElevGain" not in races_df.columns:
            races_df["aidStationsElevGain"] = ""
        if "aidStationsElevLoss" not in races_df.columns:
            races_df["aidStationsElevLoss"] = ""

        aid_stations_str = ",".join([str(x) for x in sorted(aid_stations_km)])
        aid_times_str = ",".join([str(int(t)) for t in aid_stations_times])

        aid_dist_eq_list = [str(round(seg.get("distanceEqKm", 0.0), 1)) for seg in segment_stats]
        aid_elev_gain_list = [str(int(seg.get("elevGainM", 0.0))) for seg in segment_stats]
        aid_elev_loss_list = [str(int(seg.get("elevLossM", 0.0))) for seg in segment_stats]
        aid_dist_eq_str = ",".join(aid_dist_eq_list)
        aid_elev_gain_str = ",".join(aid_elev_gain_list)
        aid_elev_loss_str = ",".join(aid_elev_loss_list)

        if "raceId" in races_df.columns and race_id in races_df["raceId"].values:
            races_df.loc[races_df["raceId"] == race_id, "name"] = race_name
            races_df.loc[races_df["raceId"] == race_id, "aidStationsKm"] = aid_stations_str
            races_df.loc[races_df["raceId"] == race_id, "aidStationsTimes"] = aid_times_str
            races_df.loc[races_df["raceId"] == race_id, "aidStationsDistEq"] = aid_dist_eq_str
            races_df.loc[races_df["raceId"] == race_id, "aidStationsElevGain"] = aid_elev_gain_str
            races_df.loc[races_df["raceId"] == race_id, "aidStationsElevLoss"] = aid_elev_loss_str
        else:
            import datetime

            new_row = {
                "raceId": race_id,
                "name": race_name,
                "createdAt": datetime.datetime.now().isoformat(),
                "aidStationsKm": aid_stations_str,
                "aidStationsTimes": aid_times_str,
                "aidStationsDistEq": aid_dist_eq_str,
                "aidStationsElevGain": aid_elev_gain_str,
                "aidStationsElevLoss": aid_elev_loss_str,
            }
            races_df = pd.concat([races_df, pd.DataFrame([new_row])], ignore_index=True)

        self.storage.write_csv(races_file, races_df)

        segments_file = race_pacing_dir / f"{race_id}_segments.csv"
        self.storage.write_csv(segments_file, segments_df)

        logger.info("Saved race %s: %s", race_id, race_name)
        return race_id

    def load_race(
        self, race_id: str
    ) -> Optional[tuple[str, list[float], pd.DataFrame, Optional[list[float]]]]:
        """Load race pacing data from CSV files."""
        race_pacing_dir = self.storage.base_dir / "race_pacing"
        races_file = self.storage.base_dir / "races.csv"

        if not races_file.exists():
            return None

        races_df = self.storage.read_csv(races_file)
        if races_df.empty or race_id not in races_df["raceId"].values:
            return None

        race_row = races_df[races_df["raceId"] == race_id].iloc[0]
        race_name = str(race_row["name"])

        aid_stations_str = str(race_row.get("aidStationsKm", ""))
        aid_stations_km = [float(x.strip()) for x in aid_stations_str.split(",") if x.strip()]

        aid_stations_times = None
        if "aidStationsTimes" in race_row and pd.notna(race_row.get("aidStationsTimes")):
            aid_times_str = str(race_row["aidStationsTimes"])
            if aid_times_str.strip():
                aid_stations_times = [float(x.strip()) for x in aid_times_str.split(",") if x.strip()]

        segments_file = race_pacing_dir / f"{race_id}_segments.csv"
        if not segments_file.exists():
            return None

        segments_df = self.storage.read_csv(segments_file)
        return (race_name, aid_stations_km, segments_df, aid_stations_times)

    def list_races(self) -> pd.DataFrame:
        """List all saved races."""
        races_file = self.storage.base_dir / "races.csv"

        if not races_file.exists():
            return pd.DataFrame(columns=["raceId", "name", "createdAt"])

        races_df = self.storage.read_csv(races_file)
        if races_df.empty:
            return pd.DataFrame(columns=["raceId", "name", "createdAt"])

        return races_df[["raceId", "name", "createdAt"]].copy()
