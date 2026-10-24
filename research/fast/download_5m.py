"""Download 5m klines for every coin in the point-in-time top-40 universe, only for the months it is in it.

The universe is E5's: each day, the 40 USDT perpetuals with the largest trailing 30-day quote volume
that have been listed for at least 60 days, delisted coins included. A coin's months are every month
with at least one eligible day, plus the month before (indicator warm-up).

Needs the daily panel from research/momentum/download_all.py first.
Usage: PYTHONPATH=research/momentum uv run python research/fast/download_5m.py [INTERVAL FROM_MONTH]
       (e.g. `1m 2022-01` for the EPR15 check: 1m klines of the same coin-months from 2022-01 on)
Files land in data/history/monthly/klines/SYM/<interval>/; the month list in data/research/fast/months.json.
"""

import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

import e5_survivorship as E5

BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
CACHE = Path("data/history/monthly/klines")
OUT = Path("data/research/fast")
FIRST = "2020-06"  # 30-day volume + 60-day age needs a few months of history
LAST = "2026-09"


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


def universe_months() -> dict[str, list[str]]:
    D = E5.daily_panel()
    elig = E5.eligible(D)
    mon = pd.to_datetime(D["t"], unit="ms").strftime("%Y-%m").to_numpy()
    out = {}
    for j, s in enumerate(D["names"]):
        ms = sorted(set(mon[elig[:, j]]))
        ms = [m for m in ms if FIRST <= m <= LAST]
        if not ms:
            continue
        prev = {(pd.Period(m, "M") - 1).strftime("%Y-%m") for m in ms}
        out[str(s)] = sorted(set(ms) | {p for p in prev if p >= "2020-01"})
    return out


def main():
    import sys

    iv, first = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("5m", "2020-01")
    OUT.mkdir(parents=True, exist_ok=True)
    um = universe_months()
    (OUT / "months.json").write_text(json.dumps(um))
    jobs = [f"{s}/{iv}/{s}-{iv}-{m}.zip" for s, ms in um.items() for m in ms if m >= first]
    print(len(um), "coins", len(jobs), "files")
    with cf.ThreadPoolExecutor(24) as ex:
        res = list(ex.map(fetch, jobs))
    print({k: sum(r.startswith(k) for r in res) for k in ("ok", "cached", "404", "failed")})
    for r in res:
        if r.startswith("failed"):
            print(r)


if __name__ == "__main__":
    main()
