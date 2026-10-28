"""Benchmarks for the event study: BTCUSDT spot 1m and 1h klines of every USDT spot pair (for the
equal-weight alt basket), delisted pairs included.

Usage: PYTHONPATH=research/nla uv run python research/nla/n4_benchmarks.py
Needs: data/research/nla/roster.json
Output: data/research/nla/bench_btc_1m.npz, data/research/nla/spot_1h.npz
"""

import concurrent.futures as cf
import json
from pathlib import Path

import numpy as np
import pandas as pd
from archive import fetch, read_klines
from n2_events import LEVERAGED, STABLE, split_spot

OUT = Path("data/research/nla")
ROOT = "data/spot/monthly/klines"


def load(sym: str, iv: str, months: list[str]) -> pd.DataFrame:
    ps = [fetch(f"{ROOT}/{sym}/{iv}/{sym}-{iv}-{m}.zip") for m in months]
    ds = [read_klines(p) for p in ps if p is not None]
    return pd.concat(ds, ignore_index=True) if ds else pd.DataFrame()


def main():
    roster = json.loads((OUT / "roster.json").read_text())["spot"]
    btc = load("BTCUSDT", "1m", roster["BTCUSDT"])
    np.savez_compressed(
        OUT / "bench_btc_1m.npz", t=btc["open_time"].to_numpy(np.int64), c=btc["close"].to_numpy()
    )
    syms = []
    for s in roster:
        sq = split_spot(s)
        if sq and sq[1] == "USDT" and sq[0] not in STABLE and not LEVERAGED.match(sq[0]):
            syms.append(s)
    print(len(syms), "USDT pairs")
    with cf.ThreadPoolExecutor(24) as ex:
        ds = list(ex.map(lambda s: load(s, "1h", roster[s]), syms))
    hours = np.arange(
        pd.Timestamp("2017-07-01").value // 10**6,
        pd.Timestamp("2026-10-01").value // 10**6,
        3_600_000,
    )
    close = np.full((len(hours), len(syms)), np.nan, np.float32)
    qvol = np.zeros((len(hours), len(syms)), np.float32)
    for j, d in enumerate(ds):
        if d.empty:
            continue
        i = np.searchsorted(hours, d["open_time"].to_numpy(np.int64))
        ok = (i < len(hours)) & (hours[np.minimum(i, len(hours) - 1)] == d["open_time"].to_numpy())
        close[i[ok], j] = d["close"].to_numpy()[ok]
        qvol[i[ok], j] = d["quote_volume"].to_numpy()[ok]
    np.savez_compressed(OUT / "spot_1h.npz", t=hours, names=np.array(syms), close=close, qvol=qvol)


if __name__ == "__main__":
    main()
