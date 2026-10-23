"""Frozen VDA validation on completed October 2026 daily public archives.

The prior research ends at 2026-09.  This module uses 2026-10-01 through
2026-10-22 as a short forward holdout.  It accesses only public market-data
endpoints when --fetch is explicitly supplied and otherwise runs offline.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import sim  # noqa: E402
from lab import UNIVERSE  # noqa: E402
from sim import Cfg  # noqa: E402

DATA = ROOT / "data/validation"
OUT = ROOT / "research/validation/results"
START = date(2026, 10, 1)
END = date(2026, 10, 22)
DAILY = "https://data.binance.vision/data/futures/um/daily"
FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"


def days() -> list[date]:
    return [START + timedelta(days=i) for i in range((END - START).days + 1)]


def _fetch_file(args) -> str:
    symbol, day = args
    name = f"{symbol}-1m-{day}.zip"
    rel = f"klines/{symbol}/1m/{name}"
    path = DATA / "forward/daily" / rel
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{DAILY}/{rel}", timeout=120) as response:
                payload = response.read()
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    temporary = path.with_suffix(".part")
    temporary.write_bytes(payload)
    temporary.rename(path)
    return str(path)


def _fetch_funding(symbol: str) -> None:
    path = DATA / "forward/funding" / f"{symbol}.json"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    start = int(datetime.combine(START, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)
    end = int(
        datetime.combine(END + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp()
        * 1000
    )
    query = urllib.parse.urlencode(
        {"symbol": symbol, "startTime": start, "endTime": end, "limit": 1000}
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{FUNDING}?{query}", timeout=60) as response:
                payload = json.loads(response.read())
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def fetch() -> None:
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(_fetch_file, [(s, d.isoformat()) for s in UNIVERSE for d in days()]))
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(_fetch_funding, UNIVERSE))


def _funding(symbol: str) -> pd.DataFrame:
    payload = json.loads((DATA / "forward/funding" / f"{symbol}.json").read_text())
    return pd.DataFrame(
        [(int(row["fundingTime"]), float(row["fundingRate"])) for row in payload],
        columns=["t", "rate"],
    )


def _bars(symbol: str) -> pd.DataFrame:
    # September is warmup only. October daily bars are the untouched evaluation period.
    september = sim._read_zip(
        DATA / f"monthly/klines/{symbol}/1m/{symbol}-1m-2026-09.zip"
    )
    october = [
        sim._read_zip(
            DATA
            / f"forward/daily/klines/{symbol}/1m/{symbol}-1m-{day.isoformat()}.zip"
        )
        for day in days()
    ]
    return pd.concat([september, *october], ignore_index=True).drop_duplicates("t").sort_values("t")


def run() -> pd.DataFrame:
    ticks = json.loads((DATA / "ticks.json").read_text())
    base = Cfg(
        k=12,
        trend="against",
        sides="long",
        sl=3.0,
        max_hold=30,
        extra={"min_vol_ratio": 1.07, "max_ret60": -0.26, "min_atr_pct": 0.081},
    )
    scenarios = {
        "frozen": base,
        "penetration_0.2_stop_slip_0.3": replace(base, penetrate=0.2, level_slip=0.3),
    }
    start_ms = int(datetime.combine(START, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)
    rows = []
    summaries = []
    for symbol in UNIVERSE:
        bars = _bars(symbol)
        funding = _funding(symbol).to_numpy()
        for name, cfg in scenarios.items():
            tr = sim.simulate(
                bars,
                replace(cfg, tick=ticks[symbol]),
                funding,
                start_ms=start_ms,
            )
            if tr.empty:
                continue
            tr = tr[tr.reason != "end_of_data"].copy()
            if tr.empty:
                continue
            tr.insert(0, "symbol", symbol)
            tr.insert(1, "scenario", name)
            rows.append(tr)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(result):
        result.insert(2, "trade_id", result.groupby(["scenario", "symbol"]).cumcount())
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "forward_holdout_trades.csv", index=False)
    for name in scenarios:
        tr = result[result.scenario == name] if len(result) else result
        summaries.append(
            {
                "scenario": name,
                "start": START.isoformat(),
                "end": END.isoformat(),
                "days": len(days()),
                "trades": len(tr),
                "avg_net_pct": float(tr.net.mean()) if len(tr) else None,
                "net_usdt_independent_accounts": float(tr.net.sum() * 2.97) if len(tr) else 0.0,
                "positive_symbols": (
                    int((tr.groupby("symbol").net.sum() > 0).sum()) if len(tr) else 0
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "forward_holdout_summary.csv", index=False)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true")
    args = p.parse_args()
    if args.fetch:
        fetch()
    print(run().to_string(index=False))


if __name__ == "__main__":
    main()
