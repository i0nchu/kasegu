"""Download monthly 1m klines and funding rates from data.binance.vision.

Usage: uv run python research/download.py --months 2025-10:2026-09 BTCUSDT ETHUSDT ...
"""

import argparse
import concurrent.futures as cf
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://data.binance.vision/data/futures/um/monthly"
CACHE = Path("data/history/monthly")

UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT", "SUIUSDT",
    "1000PEPEUSDT", "WIFUSDT", "ARBUSDT", "OPUSDT", "APTUSDT", "NEARUSDT", "TIAUSDT",
    "INJUSDT", "SEIUSDT", "ENAUSDT", "1000BONKUSDT", "FETUSDT", "ORDIUSDT", "WLDUSDT",
]  # fmt: skip


def month_range(spec: str) -> list[str]:
    a, b = spec.split(":")
    y, m = map(int, a.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= b:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch(rel: str) -> str:
    path = CACHE / rel
    if path.exists():
        return f"cached {rel}"
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        try:
            with urllib.request.urlopen(f"{BASE}/{rel}", timeout=120) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            return f"{e.code} {rel}"
        except OSError as e:  # connection reset / timeout: back off and retry
            if i == 4:
                return f"failed {rel}: {e}"
            time.sleep(2**i)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(path)
    return f"ok {rel} {len(data) // 1024}KB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("symbols", nargs="*")
    p.add_argument("--months", default="2025-10:2026-09")
    a = p.parse_args()
    jobs = []
    for s in a.symbols or UNIVERSE:
        for m in month_range(a.months):
            jobs.append(f"klines/{s}/1m/{s}-1m-{m}.zip")
            jobs.append(f"fundingRate/{s}/{s}-fundingRate-{m}.zip")
    with cf.ThreadPoolExecutor(8) as ex:
        for msg in ex.map(fetch, jobs):
            if not msg.startswith("cached"):
                print(msg, flush=True)


if __name__ == "__main__":
    main()
