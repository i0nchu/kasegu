"""Download daily aggTrades for a list of (symbol, date) pairs from data.binance.vision.

Usage: uv run python research/alpha/download_trades.py pairs.csv   (columns: symbol,date)
"""

import concurrent.futures as cf
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
CACHE = Path("data/history/daily/aggTrades")


def fetch(args) -> str:
    s, d = args
    path = CACHE / s / f"{s}-aggTrades-{d}.zip"
    if path.exists():
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        try:
            with urllib.request.urlopen(f"{BASE}/{s}/{path.name}", timeout=300) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            return f"{e.code} {path.name}"
        except OSError as e:
            if i == 4:
                return f"failed {path.name}: {e}"
            time.sleep(2**i)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(path)
    return f"ok {path.name} {len(data) >> 20}MB"


if __name__ == "__main__":
    pairs = pd.read_csv(sys.argv[1])
    with cf.ThreadPoolExecutor(6) as ex:
        for msg in ex.map(fetch, zip(pairs.symbol, pairs.date, strict=True)):
            if msg != "cached":
                print(msg, flush=True)
