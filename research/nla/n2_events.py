"""NLA-01 step 2: turn the archive roster into listing events and download the first 30 days of 1m
klines for each one.

One event per (base asset, market): the first time the asset traded on Binance spot (any quote) and,
separately, the first time its USD-M perpetual traded. A base that disappears from the archive for
2+ months and comes back gets a second event, flagged as a relist / possible ticker reuse.

Usage: PYTHONPATH=research/nla uv run python research/nla/n2_events.py
Needs: data/research/nla/roster.json (n1_roster.py)
Output: data/research/nla/events.csv, data/research/nla/bars/<event_id>.npz (1m, first 31 days)
"""

import concurrent.futures as cf
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from archive import fetch, read_klines

OUT = Path("data/research/nla")
DAY = 86_400_000
WINDOW = 31 * DAY
QUOTES = [
    "FDUSD",
    "USDT",
    "BUSD",
    "TUSD",
    "USDC",
    "USDP",
    "USDS",
    "DAI",
    "PAX",
    "BTC",
    "ETH",
    "BNB",
    "TRY",
    "EUR",
    "GBP",
    "AUD",
    "BRL",
    "RUB",
    "UAH",
    "NGN",
    "BIDR",
    "IDRT",
    "ZAR",
    "BKRW",
    "VAI",
    "PLN",
    "RON",
    "ARS",
    "JPY",
    "MXN",
    "COP",
    "CZK",
    "XRP",
    "TRX",
    "DOGE",
    "DOT",
    "SOL",
    "AEUR",
    "EURI",
]
PANEL_QUOTES = ["USDT", "FDUSD", "USDC", "BUSD"]  # price series used for the study, in this order
STABLE = {
    "USDC",
    "BUSD",
    "TUSD",
    "USDP",
    "PAX",
    "DAI",
    "FDUSD",
    "USDS",
    "USDSB",
    "EUR",
    "GBP",
    "AUD",
    "AEUR",
    "EURI",
    "USD1",
    "BFUSD",
    "XUSD",
    "RLUSD",
    "PAXG",
    "WBTC",
    "WBETH",
    "BETH",
    "USDE",
    "SUSDE",
    "BKRW",
    "BVND",
    "IDRT",
    "BIDR",
}
LEVERAGED = re.compile(r".+(UP|DOWN|BULL|BEAR)$")
MULT = re.compile(r"^(1000000|100000|10000|1000|1M)(.+)$")


def split_spot(sym: str) -> tuple[str, str] | None:
    if sym == "BNBIDR":
        return "BNB", "IDR"
    for q in QUOTES:
        if sym.endswith(q) and len(sym) > len(q):
            base = sym[: -len(q)]
            m = MULT.match(base)
            if m and m.group(2) not in ("INCH", "X"):
                base = m.group(2)  # spot 1000SATS etc. is the same asset as perp 1000SATS
            return base, q
    return None


def split_um(sym: str) -> tuple[str, str, int] | None:
    if "_" in sym:
        return None  # delivery contracts
    for q in ("USDT", "USDC", "BUSD"):
        if sym.endswith(q):
            base, mult = sym[: -len(q)], 1
            m = MULT.match(base)
            if m and m.group(2) not in ("INCH", "X"):  # 1INCH is a real name, 1000X is a token
                base, mult = (
                    m.group(2),
                    1_000_000 if m.group(1) in ("1M", "1000000") else int(m.group(1)),
                )
            return base, q, mult
    return None


def segments(months: list[str]) -> list[list[str]]:
    p = [pd.Period(m, "M") for m in months]
    out = [[months[0]]]
    for a, b, m in zip(p, p[1:], months[1:], strict=False):
        if (b - a).n > 2:
            out.append([m])
        else:
            out[-1].append(m)
    return out


def candidates(roster: dict) -> pd.DataFrame:
    """One row per symbol segment, tagged with the base-level segment it opens in."""
    rows = []
    for market, key, split in (("spot", "spot", split_spot), ("perp", "um", split_um)):
        for sym, ms in roster[key].items():
            sq = split(sym)
            if sq:
                for part in segments(ms):
                    rows.append(
                        dict(
                            market=market,
                            symbol=sym,
                            base=sq[0],
                            quote=sq[1],
                            mult=sq[2] if len(sq) > 2 else 1,
                            months=part,
                        )
                    )
    c = pd.DataFrame(rows)
    seg = []
    for (_, _), g in c.groupby(["market", "base"]):
        segs = segments(sorted({m for ms in g["months"] for m in ms}))
        starts = [s[0] for s in segs]
        for i, ms in g["months"].items():
            seg.append((i, sum(st <= ms[0] for st in starts) - 1, segs))
    for i, k, segs in seg:
        c.at[i, "seg"] = k
        c.at[i, "seg_start"] = segs[k][0]
        c.at[i, "last_month"] = c.at[i, "months"][-1]
    c["seg"] = c["seg"].astype(int)
    c["first_month"] = [ms[0] for ms in c["months"]]
    return c


def first_trade(market: str, sym: str, month: str) -> tuple[float, float]:
    """(first 1m open time, first non-zero-volume 1m open time) in the given month."""
    d = kline_month(market, sym, month)
    if d is None or d.empty:
        return np.nan, np.nan
    live = d[d["volume"] > 0]
    return float(d["open_time"].iat[0]), (float(live["open_time"].iat[0]) if len(live) else np.nan)


def kline_month(market: str, sym: str, month: str) -> pd.DataFrame | None:
    root = "data/spot/monthly/klines" if market == "spot" else "data/futures/um/monthly/klines"
    p = fetch(f"{root}/{sym}/1m/{sym}-1m-{month}.zip")
    return None if p is None else read_klines(p)


def window(market: str, sym: str, months: list[str], t0: float) -> pd.DataFrame:
    parts = []
    for m in months[:3]:
        d = kline_month(market, sym, m)
        if d is not None:
            parts.append(d[(d["open_time"] >= t0) & (d["open_time"] < t0 + WINDOW)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    roster = json.loads((OUT / "roster.json").read_text())
    c = candidates(roster)
    print(len(c), "symbol segments")
    # exact first trade of every segment that could be the first one of its base
    nxt = {m: str(pd.Period(m, "M") + 1) for m in c["seg_start"]}
    first = c[
        (c["first_month"] == c["seg_start"])
        | ((c["first_month"] == c["seg_start"].map(nxt)) & c["quote"].isin(PANEL_QUOTES))
    ]
    jobs = list(first[["market", "symbol", "first_month"]].itertuples(index=False, name=None))
    cache = OUT / "first_trades.json"
    ft = (
        {tuple(k.split("|")): v for k, v in json.loads(cache.read_text()).items()}
        if cache.exists()
        else {}
    )
    todo = [j for j in jobs if j not in ft]
    with cf.ThreadPoolExecutor(16) as ex:
        ft |= dict(zip(todo, ex.map(lambda j: first_trade(*j), todo), strict=True))
    cache.write_text(json.dumps({"|".join(k): v for k, v in ft.items()}))
    c["t_first_bar"] = [
        ft.get((r.market, r.symbol, r.first_month), (np.nan,))[0] for r in c.itertuples()
    ]
    c["t_first_trade"] = [
        ft.get((r.market, r.symbol, r.first_month), (np.nan, np.nan))[1] for r in c.itertuples()
    ]
    c.drop(columns="months").to_csv(OUT / "segments.csv", index=False)
    build_events(c)


def build_events(c: pd.DataFrame):
    ev = []
    for (mkt, base, seg), every in c.groupby(["market", "base", "seg"]):
        g = every.dropna(subset=["t_first_trade"])
        if g.empty:
            continue
        t0 = g["t_first_trade"].min()
        first = g[g["t_first_trade"] == t0].iloc[0]
        panel = None
        for q in PANEL_QUOTES:
            h = g[g["quote"] == q]
            if len(h) and h["t_first_trade"].min() - t0 < DAY:
                panel = h.sort_values("t_first_trade").iloc[0]
                break
        ev.append(
            dict(
                event_id=f"{mkt}_{base}_{seg}",
                asset_id=base,
                market_type=mkt,
                seg=seg,
                first_trade_time=t0,
                first_symbol=first.symbol,
                first_quote=first.quote,
                panel_symbol=None if panel is None else panel.symbol,
                panel_first_trade=np.nan if panel is None else panel.t_first_trade,
                panel_month=None if panel is None else panel.first_month,
                mult=first.mult,
                first_month=first.first_month,
                last_month=every["last_month"].max(),
                n_pairs=every["symbol"].nunique(),
                quotes=",".join(sorted(set(every["quote"]))),
            )
        )
    ev = pd.DataFrame(ev)
    spot0 = ev[ev.market_type == "spot"].groupby("asset_id")["first_trade_time"].min()
    perp0 = ev[ev.market_type == "perp"].groupby("asset_id")["first_trade_time"].min()
    ev["spot_listing_time"] = ev["asset_id"].map(spot0)
    ev["perp_listing_time"] = ev["asset_id"].map(perp0)
    ev["listing_category"] = [category(r) for r in ev.itertuples()]
    ev["prior_market_exists"] = [prior(r) for r in ev.itertuples()]
    ev["announcement_time"] = np.nan  # needs Binance announcements, not in the archive
    ev.to_csv(OUT / "events_raw.csv", index=False)
    print(ev["listing_category"].value_counts())
    download_bars(ev, c)


def category(r) -> str:
    if r.seg > 0:
        return "relist"
    other = r.perp_listing_time if r.market_type == "spot" else r.spot_listing_time
    if np.isnan(other) or other - r.first_trade_time > DAY:
        return f"{r.market_type}_first"
    if r.first_trade_time - other > DAY:
        return f"{r.market_type}_after_{'perp' if r.market_type == 'spot' else 'spot'}"
    return "spot_perp_same_day"


def prior(r) -> str:
    if r.listing_category in ("perp_after_spot", "spot_after_perp", "relist"):
        return "yes_binance"
    return "unknown"  # DEX / other-CEX history needs an external source


def to_usd(d: pd.DataFrame, quote: str, ms: list[str]) -> pd.DataFrame:
    """Price a BTC/ETH/BNB-quoted pair in USDT with the quote's own USDT pair, minute by minute."""
    parts = [kline_month("spot", f"{quote}USDT", m) for m in ms[:3]]
    parts = [x for x in parts if x is not None]
    if not parts:
        return pd.DataFrame()
    q = pd.concat(parts).set_index("open_time")["close"]
    px = q.reindex(d["open_time"]).ffill().to_numpy()
    d = d.copy()
    for col in ("open", "high", "low", "close", "quote_volume", "taker_quote"):
        d[col] = d[col] * px
    return d.dropna(subset=["close"]).reset_index(drop=True)


def download_bars(ev: pd.DataFrame, c: pd.DataFrame):
    (OUT / "bars").mkdir(exist_ok=True)
    months = {(r.market, r.symbol, r.first_month): r.months for r in c.itertuples()}
    mkt = {"spot": "spot", "perp": "perp"}

    def one(r) -> dict:
        sym = r.panel_symbol if isinstance(r.panel_symbol, str) else None
        synth = sym is None and r.market_type == "spot" and r.first_quote in ("BTC", "ETH", "BNB")
        if sym is None and not synth:
            return dict(event_id=r.event_id, n_bars=0)
        if synth:
            sym, t0 = r.first_symbol, r.first_trade_time
            ms = months[("spot", sym, r.first_month)]
        else:
            t0 = r.panel_first_trade
            ms = months[(r.market_type, sym, r.panel_month)]
        d = window(mkt[r.market_type], sym, ms, t0)
        if synth and not d.empty:
            d = to_usd(d, r.first_quote, ms)
        if d.empty:
            return dict(event_id=r.event_id, n_bars=0)
        a = d[
            [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trades",
                "taker_quote",
            ]
        ].to_numpy(np.float64)
        np.savez_compressed(OUT / "bars" / f"{r.event_id}.npz", k=a)
        return dict(
            event_id=r.event_id,
            n_bars=len(d),
            t_last_bar=d["open_time"].iat[-1],
            synthetic_usd=synth,
            bars_symbol=sym,
        )

    with cf.ThreadPoolExecutor(16) as ex:
        res = pd.DataFrame(list(ex.map(one, ev.itertuples())))
    ev.merge(res, on="event_id", how="left").to_csv(OUT / "events_raw.csv", index=False)


if __name__ == "__main__":
    main()
