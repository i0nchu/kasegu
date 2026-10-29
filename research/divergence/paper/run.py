"""Run the DIV-S4 paper ledgers on live Binance data (no orders are ever sent).

First start backfills from frozen paper_start (2026-10-01; basket history from 31 days earlier),
flagging those rows "backfill" (no real quote). After that it wakes up a minute after every 4h
close to record new signals with the live book ticker, and every 15 minutes to close trades.
State is in --state (default data/paper/divs4); restarting resumes where it stopped.
Usage:
  uv run --with numba python research/divergence/paper/run.py            # keep running
  uv run --with numba python research/divergence/paper/run.py --once     # catch up and exit (cron)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]

import pandas as pd  # noqa: E402

import evaluate  # noqa: E402
from engine import DAY, F, H4, Engine  # noqa: E402
from sources import BinanceSource  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="data/paper/divs4")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    d = Path(a.state)
    d.mkdir(parents=True, exist_ok=True)

    def log(m):
        line = f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M:%S} {m}"
        print(line, flush=True)
        with open(d / "run.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    start = int(pd.Timestamp(F["paper_start"]).value // 10**6)
    eng = Engine(BinanceSource(d, log), d, start - 31 * DAY, start, log)
    while True:
        now = int(time.time() * 1000)
        try:
            eng.run_until(now)
            if (d / "trades.csv").exists() and (d / "signals.csv").exists():
                evaluate.main(d, quiet=True)
        except Exception as e:  # noqa: BLE001
            log(f"error: {e!r}")
            if a.once:
                raise
        if a.once:
            return
        nxt = (now // H4 + 1) * H4 + 60_000
        wait = min(nxt - int(time.time() * 1000), 15 * 60_000)
        time.sleep(max(wait, 5_000) / 1000)


if __name__ == "__main__":
    main()
