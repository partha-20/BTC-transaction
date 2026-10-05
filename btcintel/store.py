"""Thin SQLite persistence layer (single file, offline, zero services)."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
import pandas as pd


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)

    def write(self, name: str, df: pd.DataFrame, index: list[str] | None = None):
        df.to_sql(name, self.con, if_exists="replace", index=False)
        for col in index or []:
            self.con.execute(f"CREATE INDEX IF NOT EXISTS ix_{name}_{col} ON {name}({col})")
        self.con.commit()

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_sql(f"SELECT * FROM {name}", self.con)

    def query(self, sql: str, params=()) -> pd.DataFrame:
        return pd.read_sql(sql, self.con, params=params)

    def tables(self):
        return [r[0] for r in self.con.execute("SELECT name FROM sqlite_master WHERE type='table'")]

    def put_json(self, key: str, obj):
        self.con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self.con.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(obj, default=str)))
        self.con.commit()

    def get_json(self, key: str):
        try:
            r = self.con.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        except sqlite3.OperationalError:
            return None
        return json.loads(r[0]) if r else None

    def close(self):
        self.con.close()
