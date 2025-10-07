from pathlib import Path
from utils.config import load_config
import os

def test_load_config_creates_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    cfg = load_config()
    assert cfg.data_dir.exists()
    assert cfg.timeseries_dir.exists()
    assert cfg.raw_strava_dir.exists()
