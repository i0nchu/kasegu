"""Data sources for the DIV-S4 paper engine.

BinanceSource: public USDⓈ-M futures REST (klines with taker-buy quote volume, funding rates, book
ticker). Universe = top 40 USDT perpetuals by trailing 30-day mean daily quote volume, listed for at
least 60 days, decided at the close of the previous UTC day (same rule as the research panel).
HistSource: the research 5m npz panel, for replaying the engine against the backtest.
"""

import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

import common as C
from engine import DAY, F, H4, M5

BASE = "https://fapi.binance.com"


class HistSource:
    def __init__(self):
        self.X, self.B, self.days = {}, {}, {}
        for s in C.coins():
            X = C.load(s)
            self.X[s] = X
            self.B[s] = C.resample(X, 48)
            d = X["t"] // DAY * DAY
            first = np.r_[True, d[1:] != d[:-1]]
            for day, e in zip(d[first], X["elig"][first]):
                if e:
                    self.days.setdefault(int(day), set()).add(s)

    def universe(self, day):
        return self.days.get(day, set())

    def bars4h(self, sym, last):
        B = self.B.get(sym)
        if B is None:
            return None
        k = int(np.searchsorted(B["t"], last, side="right"))
        sl = slice(max(k - F["bars_4h_window"], 0), k)
        return {c: B[c][sl] for c in ("t", "o", "h", "l", "c", "qv", "tb")}

    def bars5m(self, sym, start, end):
        X = self.X[sym]
        a, b = np.searchsorted(X["t"], [start, end])
        return pd.DataFrame({c: X[c][a:b] for c in ("t", "o", "h", "l", "c")})

    def funding(self, sym, start, end):
        X = self.X[sym]
        a, b = np.searchsorted(X["t"], [start, end])
        f = X["fund"][a:b]
        nz = np.nonzero(f)[0]
        return list(zip(X["t"][a:b][nz].tolist(), f[nz].tolist()))

    def quote(self, sym, now):
        return None

    def delisted(self, sym, now):
        return self.X[sym]["t"][-1] + M5 <= now


class BinanceSource:
    def __init__(self, cache_dir, log=print):
        self.dir = cache_dir
        self.log = log
        self.k4, self.k1, self.k5, self.info, self.info_t, self.last = {}, {}, {}, {}, 0, 0.0
        p = self.dir / "universe_cache.json"
        self.uni = {int(k): set(v) for k, v in json.loads(p.read_text()).items()} if p.exists() else {}

    def get(self, path, **params):
        url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
        time.sleep(max(0.0, self.last + 0.3 - time.time()))  # <= 200 requests a minute, well under the weight limit
        self.last = time.time()
        for k in range(5):
            try:
                with urllib.request.urlopen(url, timeout=20) as r:
                    return json.loads(r.read())
            except Exception as e:  # noqa: BLE001
                if k == 4:
                    raise
                self.log(f"retry {path} {params.get('symbol', '')}: {e}")
                time.sleep(2 ** (k + 1))

    def symbols(self, now):
        if now - self.info_t > 3_600_000:
            info = self.get("/fapi/v1/exchangeInfo")
            self.info = {s["symbol"]: s for s in info["symbols"]
                         if s["quoteAsset"] == "USDT" and "PERPETUAL" in s["contractType"]}  # fmt: skip
            self.info_t = now
        return self.info

    def klines(self, sym, interval, start, end, limit=1500):
        """Closed klines with start in [start, end): t o h l c qv tb."""
        rows, ms = [], {"5m": M5, "4h": H4, "1d": DAY}[interval]
        while start < end:
            k = self.get("/fapi/v1/klines", symbol=sym, interval=interval, startTime=start, endTime=end - 1, limit=limit)
            if not k:
                break
            rows += k
            start = int(k[-1][0]) + ms
            if len(k) < limit:
                break
        now = int(time.time() * 1000)
        d = pd.DataFrame([[int(r[0]), *map(float, (r[1], r[2], r[3], r[4], r[7], r[10])), int(r[6])] for r in rows],
                         columns=["t", "o", "h", "l", "c", "qv", "tb", "tc"])  # fmt: skip
        return d[d.tc < now].drop(columns="tc").drop_duplicates("t").reset_index(drop=True)

    def daily(self, sym, day):
        """Daily klines up to (not including) `day`, cached per symbol and extended when needed."""
        d = self.k1.get(sym)
        if d is None:
            d = self.klines(sym, "1d", day - 100 * DAY, int(time.time() * 1000))
        elif len(d) and d.t.iloc[-1] < day - DAY:
            d = pd.concat([d, self.klines(sym, "1d", int(d.t.iloc[-1]) + DAY, int(time.time() * 1000))])
        self.k1[sym] = d
        return d[d.t < day]

    def universe(self, day):
        """Top 40 by mean quote volume over the 30 days before `day`, at least 60 daily bars."""
        if day in self.uni:
            return self.uni[day]
        vols = {}
        for s, info in self.symbols(day + DAY).items():
            if info.get("onboardDate", 0) > day - F["universe_min_age_days"] * DAY:
                continue
            d = self.daily(s, day)
            if len(d) < F["universe_min_age_days"] or d.t.iloc[-1] != day - DAY:
                continue
            d = d[d.t >= day - F["universe_volume_days"] * DAY]
            if len(d) == F["universe_volume_days"]:
                vols[s] = d.qv.mean()
        top = set(sorted(vols, key=vols.get, reverse=True)[: F["universe_top"]])
        self.uni[day] = top
        (self.dir / "universe_cache.json").write_text(json.dumps({k: sorted(v) for k, v in self.uni.items()}))
        self.log(f"universe {pd.Timestamp(day, unit='ms'):%Y-%m-%d}: {len(top)} coins from {len(vols)} candidates")
        return top

    def bars4h(self, sym, last):
        d = self.k4.get(sym)
        if d is None or d.t.iloc[-1] < last:
            start = int(last - F["bars_4h_window"] * H4) if d is None else int(d.t.iloc[-1]) + H4
            new = self.klines(sym, "4h", start, last + H4)
            d = new if d is None else pd.concat([d, new], ignore_index=True)
            self.k4[sym] = d
        if d is None or not len(d):
            return None
        d = d[d.t <= last].tail(F["bars_4h_window"])
        return {c: d[c].to_numpy() for c in ("t", "o", "h", "l", "c", "qv", "tb")}

    def bars5m(self, sym, start, end):
        d = self.k5.get(sym)
        if d is None or not len(d) or start < d.t.iloc[0]:
            d = self.klines(sym, "5m", start, end)
        elif d.t.iloc[-1] + M5 < end:
            d = pd.concat([d, self.klines(sym, "5m", int(d.t.iloc[-1]) + M5, end)], ignore_index=True)
        self.k5[sym] = d[d.t >= end - 10 * DAY].reset_index(drop=True) if len(d) else d
        return d[(d.t >= start) & (d.t < end)][["t", "o", "h", "l", "c"]]

    def funding(self, sym, start, end):
        f = self.get("/fapi/v1/fundingRate", symbol=sym, startTime=start, endTime=end, limit=1000)
        return [(int(x["fundingTime"]), float(x["fundingRate"])) for x in f]

    def quote(self, sym, now):
        q = self.get("/fapi/v1/ticker/bookTicker", symbol=sym)
        return {"bid": float(q["bidPrice"]), "ask": float(q["askPrice"]), "time": int(q.get("time") or now)}

    def delisted(self, sym, now):
        s = self.symbols(now).get(sym)
        return s is None or s.get("status") != "TRADING"
