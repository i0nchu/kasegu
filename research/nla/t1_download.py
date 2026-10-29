"""NLA-T01 data: funding rates and 1h klines of every USDT perpetual in the archive (delisted
included). Funding is needed for every trade; 1h klines for the established-alt control shorts.

Usage: PYTHONPATH=research/nla uv run python research/nla/t1_download.py
Needs: data/research/nla/roster.json
Output: cache under data/history/vision/futures/um/monthly/{fundingRate,klines/*/1h}/
"""

import concurrent.futures as cf
import json
from pathlib import Path

from archive import fetch, listing

OUT = Path("data/research/nla")
ROOT = "data/futures/um/monthly"


def keys(sym: str) -> list[str]:
    ks = [f"{ROOT}/klines/{sym}/1h/{sym}-1h-{m}.zip" for m in ROSTER[sym]]
    ks += [k for k in listing(f"{ROOT}/fundingRate/{sym}/", False) if k.endswith(".zip")]
    return ks


ROSTER = {
    s: m
    for s, m in json.loads((OUT / "roster.json").read_text())["um"].items()
    if s.endswith("USDT") and "_" not in s
}


def main():
    print(len(ROSTER), "USDT perpetuals")
    with cf.ThreadPoolExecutor(24) as ex:
        ks = [k for x in ex.map(keys, ROSTER) for k in x]
        print(len(ks), "files")
        res = list(ex.map(fetch, ks))
    print(sum(r is None for r in res), "missing")


if __name__ == "__main__":
    main()
