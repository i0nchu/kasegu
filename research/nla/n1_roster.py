"""NLA-01 step 1: every spot and USD-M futures symbol in the Binance archive, delisted included,
with the months that have 1m klines. The archive keeps delisted symbols, unlike exchangeInfo.

Usage: PYTHONPATH=research/nla uv run python research/nla/n1_roster.py
Output: data/research/nla/roster.json  {market: {symbol: [YYYY-MM, ...]}}
"""

import concurrent.futures as cf
import json
from pathlib import Path

from archive import listing

OUT = Path("data/research/nla")
MARKETS = {"spot": "data/spot/monthly/klines/", "um": "data/futures/um/monthly/klines/"}


def months(prefix: str) -> list[str]:
    keys = listing(prefix + "1m/", False)
    return sorted({k.rsplit("-1m-", 1)[1][:7] for k in keys if k.endswith(".zip")})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    roster = {}
    for mkt, base in MARKETS.items():
        syms = [p.rstrip("/").split("/")[-1] for p in listing(base, True)]
        print(mkt, len(syms), "symbols")
        with cf.ThreadPoolExecutor(24) as ex:
            ms = list(ex.map(months, [f"{base}{s}/" for s in syms]))
        roster[mkt] = {s: m for s, m in zip(syms, ms, strict=False) if m}
        print(mkt, len(roster[mkt]), "with 1m data")
    (OUT / "roster.json").write_text(json.dumps(roster))


if __name__ == "__main__":
    main()
