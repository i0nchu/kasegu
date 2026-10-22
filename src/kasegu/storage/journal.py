import json
import sqlite3
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts INTEGER NOT NULL,
    kind TEXT NOT NULL,
    symbol TEXT,
    data TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    client_id TEXT PRIMARY KEY,
    ts INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    purpose TEXT,
    status TEXT,
    data TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY,
    ts INTEGER NOT NULL,
    client_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT,
    qty REAL,
    price REAL,
    fee REAL,
    maker INTEGER
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY,
    ts INTEGER NOT NULL,
    run_id TEXT,
    symbol TEXT NOT NULL,
    data TEXT
);
CREATE TABLE IF NOT EXISTS state (
    symbol TEXT PRIMARY KEY,
    ts INTEGER NOT NULL,
    data TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_kind ON events(kind);
"""


def _dump(obj: Any) -> str:
    if is_dataclass(obj):
        obj = asdict(obj)
    return json.dumps(obj, default=str)


def _now() -> int:
    return int(time.time() * 1000)


class Journal:
    def __init__(self, path: str | Path, run_id: str = ""):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(_SCHEMA)
        self.run_id = run_id

    def event(self, kind: str, symbol: str | None = None, ts: int | None = None, **data):
        self.db.execute(
            "INSERT INTO events (ts, kind, symbol, data) VALUES (?, ?, ?, ?)",
            (ts or _now(), kind, symbol, _dump(data)),
        )
        self.db.commit()

    def order(self, o, ts: int | None = None):
        self.db.execute(
            "INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?)",
            (o.client_id, ts or o.time or _now(), o.symbol, o.purpose, o.status, _dump(o)),
        )
        self.db.commit()

    def fill(self, f):
        self.db.execute(
            "INSERT INTO fills (ts, client_id, symbol, side, qty, price, fee, maker)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (f.time, f.client_id, f.symbol, f.side, f.qty, f.price, f.fee, int(f.maker)),
        )
        self.db.commit()

    def trade(self, t, ts: int | None = None):
        self.db.execute(
            "INSERT INTO trades (ts, run_id, symbol, data) VALUES (?, ?, ?, ?)",
            (ts or _now(), self.run_id, t.symbol, _dump(t)),
        )
        self.db.commit()

    def save_state(self, symbol: str, data: dict, ts: int | None = None):
        self.db.execute(
            "INSERT OR REPLACE INTO state VALUES (?, ?, ?)",
            (symbol, ts or _now(), _dump(data)),
        )
        self.db.commit()

    def load_state(self) -> dict[str, dict]:
        rows = self.db.execute("SELECT symbol, data FROM state").fetchall()
        return {s: json.loads(d) for s, d in rows}

    def trades(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT data FROM trades WHERE run_id = ? ORDER BY id", (self.run_id,)
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def close(self):
        self.db.close()
