"""Download daily 5m futures metrics (open interest, long/short ratios) from data.binance.vision.

Usage: uv run python research/alpha/download_metrics.py --days 2024-10-01:2026-09-30 [SYMBOLS]
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
CACHE = Path("data/history/daily/metrics")


def day_range(spec: str) -> list[str]:
    a, b = (dt.date.fromisoformat(x) for x in spec.split(":"))
    return [str(a + dt.timedelta(days=i)) for i in range((b - a).days + 1)]


def fetch(args) -> str:
    s, d = args
    path = CACHE / s / f"{s}-metrics-{d}.zip"
    if path.exists():
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        try:
            with urllib.request.urlopen(f"{BASE}/{s}/{path.name}", timeout=60) as r:
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
    return "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("symbols", nargs="*")
    p.add_argument("--days", default="2024-10-01:2026-09-30")
    a = p.parse_args()
    syms = a.symbols or sorted(x.name for x in Path("data/history/monthly/klines").iterdir())
    jobs = [(s, d) for s in syms for d in day_range(a.days)]
    n = {"ok": 0, "cached": 0}
    with cf.ThreadPoolExecutor(16) as ex:
        for msg in ex.map(fetch, jobs):
            if msg in n:
                n[msg] += 1
            else:
                print(msg, flush=True)
    print(n)


if __name__ == "__main__":
    main()
