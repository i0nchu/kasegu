"""Aligned 1m panel for the alpha research: prices, volume, taker flow and open interest.

Every array is on the same minute grid GRID (2024-10-01 to 2026-09-30, UTC). Bar i covers
[GRID[i], GRID[i] + 60s); everything about bar i is known at its close, GRID[i] + 60s.

Timestamps (what is usable when):
  o h l c v qv n tb   1m kline of bar i, known at the close of bar i
  oi oiv              5m open interest snapshot stamped T by Binance, treated as known at
                      T + 5 min (one full metrics period of delay) and carried forward
  live                False where Binance had no kline (filled with the previous close)
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HIST = Path("data/history")
OUT = Path("data/research/alpha")
MIN = 60_000
START = int(pd.Timestamp("2024-10-01", tz="UTC").value // 10**6)
END = int(pd.Timestamp("2026-10-01", tz="UTC").value // 10**6)
GRID = np.arange(START, END, MIN, dtype=np.int64)
N = len(GRID)
Y2 = int(np.searchsorted(GRID, int(pd.Timestamp("2025-10-01", tz="UTC").value // 10**6)))
MONTHS = pd.date_range("2024-10-01", "2026-09-01", freq="MS").strftime("%Y-%m").tolist()

# 28 research coins of the earlier rounds + 12 coins added for the holdout
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT", "SUIUSDT",
    "1000PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT", "APTUSDT", "NEARUSDT", "TIAUSDT",
    "INJUSDT", "SEIUSDT", "ENAUSDT", "1000BONKUSDT", "FETUSDT", "ORDIUSDT", "WLDUSDT",
    "1000FLOKIUSDT", "AAVEUSDT", "ATOMUSDT", "CRVUSDT", "ETCUSDT", "FILUSDT", "HBARUSDT",
    "JUPUSDT", "ONDOUSDT", "PENDLEUSDT", "UNIUSDT", "XLMUSDT",
]  # fmt: skip
FIELDS = ("o", "h", "l", "c", "v", "qv", "n", "tb", "oi", "oiv", "live")


def _klines(symbol: str) -> pd.DataFrame:
    parts = []
    for m in MONTHS:
        p = HIST / "monthly/klines" / symbol / "1m" / f"{symbol}-1m-{m}.zip"
        if not p.exists():
            continue
        with zipfile.ZipFile(p) as z, z.open(z.namelist()[0]) as f:
            d = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4, 5, 7, 8, 9])
        if not str(d.iloc[0, 0]).isdigit():
            d = d.iloc[1:]
        d = d.astype(float)
        d.columns = ["t", "o", "h", "l", "c", "v", "qv", "n", "tb"]
        parts.append(d)
    d = pd.concat(parts)
    d["t"] = d.t.astype(np.int64)
    d.loc[d.t > 10**14, "t"] //= 1000
    return d.drop_duplicates("t").set_index("t").sort_index()


def _metrics(symbol: str) -> pd.DataFrame:
    rows = []
    for p in sorted((HIST / "daily/metrics" / symbol).glob("*.zip")):
        with zipfile.ZipFile(p) as z:
            raw = z.read(z.namelist()[0])
        d = pd.read_csv(io.BytesIO(raw), usecols=[0, 2, 3])
        rows.append(d)
    if not rows:
        return pd.DataFrame(columns=["oi", "oiv"])
    d = pd.concat(rows)
    d.columns = ["ts", "oi", "oiv"]
    t = pd.to_datetime(d.ts, utc=True).astype("datetime64[ms, UTC]").astype("int64")
    d = d.assign(t=t.to_numpy() + 5 * MIN).drop(columns="ts")  # known one period later
    return d.drop_duplicates("t").set_index("t").sort_index().astype(float)


def build_klines(symbol: str) -> dict:
    k = _klines(symbol).reindex(GRID)
    live = k.c.notna().to_numpy()
    c = k.c.ffill().bfill()
    out = {"live": live, "c": c.to_numpy()}
    for col in ("o", "h", "l"):
        out[col] = k[col].fillna(c).to_numpy()
    for col in ("v", "qv", "n", "tb"):
        out[col] = k[col].fillna(0).to_numpy()
    return out


def build_oi(symbol: str) -> dict:
    m = _metrics(symbol)
    out = {}
    if not len(m):
        return {"oi": np.full(N, np.nan), "oiv": np.full(N, np.nan)}
    # last value known at or before each bar close (GRID + 1 min)
    ts = m.index.to_numpy()
    idx = np.searchsorted(ts, GRID + MIN, side="right") - 1
    ok = idx >= 0
    age = np.full(N, np.inf)
    age[ok] = GRID[ok] + MIN - ts[idx[ok]]
    for col in ("oi", "oiv"):
        a = np.full(N, np.nan)
        a[ok] = m[col].to_numpy()[idx[ok]]
        a[age > 30 * MIN] = np.nan  # a gap in the metrics files, not a flat market
        out[col] = a
    return out


_CACHE: dict = {}


def panel(symbol: str) -> dict:
    """Kline fields, plus oi / oiv once `data.py oi` has cached them (NaN before that)."""
    if symbol in _CACHE:
        return _CACHE[symbol]
    p = OUT / "panel" / f"{symbol}.npz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **build_klines(symbol))
    z = np.load(p)
    d = {f: z[f] for f in z.files}
    q = OUT / "panel" / f"{symbol}_oi.npz"
    if q.exists():
        z = np.load(q)
        d.update({f: z[f] for f in z.files})
    else:
        d["oi"] = d["oiv"] = np.full(N, np.nan)
    _CACHE[symbol] = d
    return d


def matrix(field: str, symbols=UNIVERSE, dtype=np.float64) -> np.ndarray:
    """(N, len(symbols)) array of one field."""
    return np.stack([panel(s)[field].astype(dtype) for s in symbols], axis=1)


def ticks() -> dict:
    import json

    p = Path("data/research/ticks.json")
    return json.loads(p.read_text()) if p.exists() else {}


def _save_oi(symbol: str) -> str:
    d = build_oi(symbol)
    np.savez(OUT / "panel" / f"{symbol}_oi.npz", **d)
    return f"{symbol} oi bars {int(np.isfinite(d['oi']).sum())} / {N}"


if __name__ == "__main__":
    import sys
    from concurrent.futures import ProcessPoolExecutor

    # data.py klines | data.py oi   (rebuild the caches)
    what, syms = sys.argv[1], sys.argv[2:] or UNIVERSE
    (OUT / "panel").mkdir(parents=True, exist_ok=True)
    if what == "klines":
        for s in syms:
            (OUT / "panel" / f"{s}.npz").unlink(missing_ok=True)
        with ProcessPoolExecutor(6) as ex:
            list(ex.map(panel, syms))
        print("klines cached", len(syms))
    else:
        with ProcessPoolExecutor(6) as ex:
            for msg in ex.map(_save_oi, syms):
                print(msg)
