from pathlib import Path
import pandas as pd
from persistence.csv_storage import CsvStorage

def test_append_and_read(tmp_path):
    storage = CsvStorage(base_dir=tmp_path)
    cols = ["id","name","val"]
    storage.append_row("t.csv", {"id":"1","name":"a","val":1}, cols)
    df = storage.read_csv("t.csv")
    assert list(df.columns) == cols
    assert len(df) == 1


def test_upsert(tmp_path):
    storage = CsvStorage(base_dir=tmp_path)
    storage.upsert("t.csv", ["id"], {"id":"1","name":"a","val":1})
    storage.upsert("t.csv", ["id"], {"id":"1","name":"b","val":2})
    df = storage.read_csv("t.csv")
    assert len(df) == 1
    assert df.iloc[0]["name"] == "b"
    assert df.iloc[0]["val"] == 2
