"""
CSV storage abstraction using pandas with basic file locking.

Notes:
- Always write CSV with '.' decimal; UI formatting uses FR locale separately.
- Ensure headers exist for empty file creation.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from pandas.errors import EmptyDataError
import portalocker


@dataclass
class CsvStorage:
    base_dir: Path

    def _path(self, relative: str | Path) -> Path:
        p = self.base_dir / Path(relative)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def read_csv(
        self, relative: str | Path, dtypes: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        path = self._path(relative)
        if not path.exists():
            # Return empty DataFrame with provided dtypes as columns if given
            if dtypes:
                return pd.DataFrame(columns=list(dtypes.keys())).astype(dtypes)
            return pd.DataFrame()
        with portalocker.Lock(str(path), timeout=10, flags=portalocker.LOCK_SH):
            try:
                df = pd.read_csv(path)
            except EmptyDataError:
                if dtypes:
                    return pd.DataFrame(columns=list(dtypes.keys())).astype(dtypes)
                return pd.DataFrame()
        if dtypes:
            # Coerce types; allow errors='ignore' to avoid raising on empty
            for col, typ in dtypes.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(typ)
                    except Exception:
                        # Leave as-is if coercion fails; upstream should validate
                        pass
                else:
                    df[col] = pd.Series(dtype=typ)
        return df

    def write_csv(self, relative: str | Path, df: pd.DataFrame) -> None:
        path = self._path(relative)
        # Use a temp buffer then write under exclusive lock
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        data = csv_buf.getvalue()
        with portalocker.Lock(str(path), timeout=10, flags=portalocker.LOCK_EX):
            path.write_text(data)

    def append_row(
        self, relative: str | Path, row: Dict[str, object], columns: Iterable[str]
    ) -> None:
        path = self._path(relative)
        with portalocker.Lock(str(path), timeout=10, flags=portalocker.LOCK_EX):
            exists = path.exists()
            empty = exists and path.stat().st_size == 0
            df = pd.DataFrame([row], columns=list(columns))
            if exists and not empty:
                df.to_csv(path, mode="a", index=False, header=False)
            else:
                df.to_csv(path, index=False, header=True)

    def upsert(self, relative: str | Path, key_cols: List[str], row: Dict[str, object]) -> None:
        df = self.read_csv(relative)
        if df.empty:
            self.write_csv(relative, pd.DataFrame([row]))
            return
        # Build mask for match
        mask = pd.Series([True] * len(df))
        for key in key_cols:
            mask &= df[key].astype(str) == str(row[key])
        if mask.any():
            # Update first match
            idx = df.index[mask][0]
            for k, v in row.items():
                if k in df.columns:
                    df.at[idx, k] = v
                else:
                    df[k] = None
                    df.at[idx, k] = v
            self.write_csv(relative, df)
        else:
            # Append new row with all columns present
            # Ensure all columns in union
            for k in row.keys():
                if k not in df.columns:
                    df[k] = None
            df.loc[len(df)] = row
            self.write_csv(relative, df)
