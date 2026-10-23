"""Passive limit order event study: every maker order a rule would rest, and what follows its fill.

Order rule (same as the engine / earlier rounds): at the close of bar i, rest a limit at
  L = ask x (1 - side x k x dist / 100),  ask = close + tick / 2,  dist = (ATR% + Amp%) / 2
rounded away from the market to the tick. It fills in bar i+1 when the opposite quote trades
through L (low + tick/2 < L for a buy). One order per bar, no position limit: this measures
the market, not a portfolio. Episodes (first fill per coin within 30 minutes) are kept as a flag.

Markouts are signed returns in % from the fill price L:
  m0    to the close of the fill bar (the intra-bar bounce a 1m bar can overstate)
  mH    to the close H minutes after the fill bar
  x_mH  mH - m0 - side x beta x market move over the same minutes: what happens after the fill
        bar, net of the coin's normal co-movement with the market
  mfe30 / mae30: best / worst signed excursion within 30 minutes, the fill bar low included
Features are the values at the close of bar i, before the order can fill.
"""

import numpy as np
import pandas as pd
from features import coin, lag, rsum

from data import GRID, MIN, Y2, ticks

HORIZONS = (1, 5, 15, 30, 60, 240)
FEATS = (
    "dist", "atr_pct", "ema_gap", "ret5", "ret15", "ret60", "ret240", "imb1", "imb5", "imb15",
    "imb60", "vol_ratio", "vz60", "v1_ratio", "rv_exp", "rv1d", "atr_ratio", "oi15", "oi60",
    "oi240", "liq_long", "liq_short", "mret5", "mret15", "mret60", "rel60",
)  # fmt: skip
WARM = 1440 + 240


def beta1d(r: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Rolling one day beta of a coin's 1m returns to the market, known at each bar close."""
    ok = np.isfinite(r) & np.isfinite(m)
    r0, m0 = np.where(ok, r, 0), np.where(ok, m, 0)
    n = rsum(ok.astype(float), 1440)
    sxy, sx, sy, sxx = rsum(r0 * m0, 1440), rsum(m0, 1440), rsum(r0, 1440), rsum(m0 * m0, 1440)
    cov = sxy / n - sx / n * sy / n
    var = sxx / n - (sx / n) ** 2
    return np.clip(cov / np.where(var > 0, var, np.nan), 0, 3)


def events(symbol: str, mkt: np.ndarray, rules: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Fills for each rule {name, side: +1/-1, trend: down/up/any, k}. Returns fills, order counts."""  # noqa: E501
    f = coin(symbol, mkt)
    tick = ticks().get(symbol, 0.0)
    half = tick / 2
    c, h, l = f["c"], f["h"], f["l"]  # noqa: E741
    n = len(c)
    M = np.exp(np.cumsum(np.nan_to_num(mkt)))  # market index level
    beta = beta1d(f["r1"], mkt)
    live = f["live"] & np.r_[f["live"][1:], False]  # order bar and fill bar both traded
    base = np.isfinite(f["dist"]) & (f["dist"] > 0) & live
    base[:WARM] = False
    base[-max(HORIZONS) - 2 :] = False
    out, counts = [], []
    for r in rules:
        sd, k = r["side"], r["k"]
        cond = base.copy()
        if r["trend"] == "down":
            cond &= f["ema_gap"] < 0
        elif r["trend"] == "up":
            cond &= f["ema_gap"] > 0
        dist = f["dist"]
        if r.get("fixed"):
            # same depth in plain percent all the time: the median volatility unit of year 1
            dist = np.full(n, np.nanmedian(f["dist"][WARM:Y2]))
        lim = (c + half) * (1 - sd * k * dist / 100)
        if tick:
            lim = (
                np.floor(lim / tick + 1e-9) * tick if sd > 0 else np.ceil(lim / tick - 1e-9) * tick
            )
        post = (lim >= c + half) if sd > 0 else (lim <= c - half)
        placed = cond & ~post
        i = np.flatnonzero(placed)
        j = i + 1
        L = lim[i]
        hit = (l[j] + half < L) if sd > 0 else (h[j] - half > L)
        counts.append(
            {
                "symbol": symbol,
                "rule": r["name"],
                "bars_in_regime": int(cond.sum()),
                "rejected:post_only": int((cond & post).sum()),
                "placed": int(len(i)),
                "filled": int(hit.sum()),
                "expired": int(len(i) - hit.sum()),
            }
        )
        i, j, L = i[hit], j[hit], L[hit]
        if not len(i):
            continue
        e = {
            "symbol": symbol,
            "rule": r["name"],
            "side": sd,
            "k": k,
            "t": GRID[j],
            "limit": L,
            "pen": (L - l[j]) / L * 100 if sd > 0 else (h[j] - L) / L * 100,  # ex post only
            "fill_bar_ret": (c[j] / c[i] - 1) * 100,
            "mkt_fill_bar": (M[j] / M[i] - 1) * 100,
            "beta": beta[i],
        }
        e["m0"] = (c[j] - L) / L * 100 * sd
        for H in HORIZONS:
            e[f"m{H}"] = (c[j + H] - L) / L * 100 * sd
            mk = (M[j + H] / M[j] - 1) * 100
            e[f"x_m{H}"] = e[f"m{H}"] - e["m0"] - sd * np.nan_to_num(beta[i]) * mk
            e[f"mk{H}"] = mk * sd
        # excursions within 30 minutes; the fill bar extreme beyond L is surely after the fill
        hh = np.stack([h[j + q] for q in range(1, 31)], 1)
        ll = np.stack([l[j + q] for q in range(1, 31)], 1)
        if sd > 0:
            e["mfe30"] = (np.maximum(hh.max(1), c[j]) - L) / L * 100
            e["mae30"] = (np.minimum(ll.min(1), l[j]) - L) / L * 100
        else:
            e["mfe30"] = (L - np.minimum(ll.min(1), c[j])) / L * 100
            e["mae30"] = (L - np.maximum(hh.max(1), h[j])) / L * 100
        for x in FEATS:
            e[x] = f[x][i] if x in f else np.nan
        d = pd.DataFrame(e)
        # episodes: first fill of this rule on this coin after 30 quiet minutes
        gap = np.diff(np.r_[-(10**9), j])
        d["first"] = gap > 30
        out.append(d)
    fills = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if len(fills):
        fills["day"] = fills.t // (1440 * MIN)
        fills["y2"] = fills.t >= GRID[Y2]
        fills["month"] = pd.to_datetime(fills.t, unit="ms").dt.strftime("%Y-%m")
    return fills, counts


__all__ = ["events", "HORIZONS", "FEATS", "lag"]
