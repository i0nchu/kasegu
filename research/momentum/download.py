"""Download monthly 1h klines and funding rates (2020-01 to 2026-09) from data.binance.vision.

Usage: uv run python research/momentum/download.py
       uv run python research/momentum/download.py SYMBOL_FILE 2024-01   (1h klines only)
Files land in data/history/monthly/{klines/SYM/1h, fundingRate/SYM}/; coins not yet listed 404.
"""

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
    "1000FLOKIUSDT", "AAVEUSDT", "ATOMUSDT", "CRVUSDT", "ETCUSDT", "FILUSDT", "HBARUSDT",
    "JUPUSDT", "ONDOUSDT", "PENDLEUSDT", "UNIUSDT", "XLMUSDT",
]  # fmt: skip


def months(a="2020-01", b="2026-09"):
    y, m = map(int, a.split("-"))
    while f"{y:04d}-{m:02d}" <= b:
        yield f"{y:04d}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch(rel: str) -> str:
    path = CACHE / rel
    miss = path.with_suffix(".404")
    if path.exists() or miss.exists():
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        try:
            with urllib.request.urlopen(f"{BASE}/{rel}", timeout=120) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                miss.touch()
            return f"{e.code} {rel}"
        except OSError as e:
            if i == 4:
                return f"failed {rel}: {e}"
            time.sleep(2**i)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(path)
    return "ok"


def main():
    import sys

    jobs = []
    if len(sys.argv) > 2:  # 1h klines for a symbol list (funding comes from download_all.py)
        syms = Path(sys.argv[1]).read_text().split()
        jobs = [f"klines/{s}/1h/{s}-1h-{m}.zip" for s in syms for m in months(sys.argv[2])]
    for s in UNIVERSE if not jobs else []:
        for m in months():
            jobs.append(f"klines/{s}/1h/{s}-1h-{m}.zip")
            jobs.append(f"fundingRate/{s}/{s}-fundingRate-{m}.zip")
    with cf.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(fetch, jobs))
    print({k: sum(r.startswith(k) for r in res) for k in ("ok", "cached", "404", "failed")})
    for r in res:
        if r.startswith("failed"):
            print(r)


if __name__ == "__main__":
    main()
