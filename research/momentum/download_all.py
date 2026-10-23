"""Download daily klines and funding for every USDT perpetual Binance has ever listed, delisted ones
included, so the daily strategies can be checked on a universe without survivorship bias.

Usage: uv run python research/momentum/download_all.py
Files land in data/history/monthly/{klines/SYM/1d, fundingRate/SYM}/.
"""

import concurrent.futures as cf
import re
import time
import urllib.request
from pathlib import Path

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BASE = "https://data.binance.vision"
CACHE = Path("data/history/monthly")


def get(url: str) -> bytes:
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except OSError:
            if i == 4:
                raise
            time.sleep(2**i)


def listing(prefix: str, dirs: bool) -> list[str]:
    out, marker = [], ""
    while True:
        url = f"{S3}?prefix={prefix}&marker={marker}" + ("&delimiter=/" if dirs else "")
        x = get(url).decode()
        tag = "Prefix" if dirs else "Key"
        found = re.findall(rf"<{tag}>([^<]+)</{tag}>", x)
        found = [f for f in found if f != prefix]
        out += found
        if "<IsTruncated>true</IsTruncated>" not in x or not found:
            return out
        marker = found[-1]


def fetch(key: str) -> str:
    rel = key.split("/um/monthly/", 1)[1]
    path = CACHE / rel
    if path.exists():
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = get(f"{BASE}/{key}")
    except OSError as e:
        return f"failed {key}: {e}"
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(path)
    return "ok"


def files_for(sym: str) -> list[str]:
    keys = listing(f"data/futures/um/monthly/klines/{sym}/1d/", False)
    keys += listing(f"data/futures/um/monthly/fundingRate/{sym}/", False)
    return [k for k in keys if k.endswith(".zip")]


def main():
    syms = [p.rstrip("/").split("/")[-1] for p in listing("data/futures/um/monthly/klines/", True)]
    syms = [s for s in syms if s.endswith("USDT")]
    print(len(syms), "USDT symbols")
    with cf.ThreadPoolExecutor(16) as ex:
        keys = [k for ks in ex.map(files_for, syms) for k in ks]
        print(len(keys), "files")
        res = list(ex.map(fetch, keys))
    print({k: sum(r.startswith(k) for r in res) for k in ("ok", "cached", "failed")})
    (CACHE / "all_usdt_symbols.txt").write_text("\n".join(syms))


if __name__ == "__main__":
    main()
