import csv
import io
import logging
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

from kasegu.core.types import Kline

log = logging.getLogger(__name__)

BASE = "https://data.binance.vision/data/futures/um"


def days(start: date, end: date) -> list[date]:
    """Inclusive date range."""
    return [start + timedelta(d) for d in range((end - start).days + 1)]


def months(start: date, end: date) -> list[str]:
    out, d = [], start.replace(day=1)
    while d <= end:
        out.append(d.strftime("%Y-%m"))
        d = (d + timedelta(32)).replace(day=1)
    return out


def _ms(v: str) -> int:
    t = int(v)
    return t // 1000 if t > 10**14 else t  # some archives use microseconds


def _rows(path: Path) -> Iterator[list[str]]:
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            for row in csv.reader(io.TextIOWrapper(f, "utf-8")):
                if row and row[0][:1].isdigit():  # skip header
                    yield row


class Archive:
    """Cached download of data.binance.vision files."""

    def __init__(self, cache: str | Path = "data/history"):
        self.cache = Path(cache)

    def get(self, rel: str) -> Path | None:
        path = self.cache / rel
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}/{rel}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                log.info("not in archive (yet): %s", rel)
            else:
                log.warning("download failed %s: %s", url, e)
            return None
        except Exception as e:
            log.warning("download failed %s: %s", url, e)
            return None
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(path)
        return path

    def klines(self, symbol: str, day: date) -> list[Kline]:
        d = day.isoformat()
        path = self.get(f"daily/klines/{symbol}/1m/{symbol}-1m-{d}.zip")
        if path is None:
            return []
        return [
            Kline(
                symbol,
                _ms(r[0]),
                _ms(r[6]),
                float(r[1]),
                float(r[2]),
                float(r[3]),
                float(r[4]),
                float(r[5]),
            )
            for r in _rows(path)
        ]

    def trades(self, symbol: str, day: date) -> Iterator[tuple[int, float, float]]:
        """(time ms, price, qty) per aggregated trade."""
        d = day.isoformat()
        path = self.get(f"daily/aggTrades/{symbol}/{symbol}-aggTrades-{d}.zip")
        if path is None:
            raise FileNotFoundError(f"aggTrades {symbol} {d}")
        for r in _rows(path):
            yield _ms(r[5]), float(r[1]), float(r[2])

    def funding(self, symbol: str, start: date, end: date) -> list[tuple[int, float]]:
        """(funding time ms, rate) sorted by time."""
        out = []
        for m in months(start, end):
            path = self.get(f"monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip")
            if path is None:
                continue
            out += [(_ms(r[0]), float(r[2])) for r in _rows(path)]
        return sorted(out)
