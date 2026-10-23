"""Ex-ante features per coin, all computed from bars up to and including bar i (known at its close).

Nothing here looks at bar i+1 or later, so any of these may filter an order placed at the close
of bar i. Units: percent for returns and distances, plain ratios otherwise.
"""

import numpy as np
import pandas as pd

from data import MIN, UNIVERSE, matrix, panel


def lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    out[k:] = x[:-k]
    return out


def rsum(x: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(np.nan_to_num(x, nan=0.0))
    out = np.full(len(x), np.nan)
    out[n - 1 :] = c[n - 1 :] - np.r_[0, c[:-n]]
    return out


def rmean(x: np.ndarray, n: int) -> np.ndarray:
    return rsum(x, n) / n


def ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def coin(symbol: str, mkt: np.ndarray | None = None) -> dict:
    """Feature arrays (length N) for one coin. mkt: equal weight market 1m log returns."""
    d = panel(symbol)
    o, h, l, c, v, tb = (d[x] for x in ("o", "h", "l", "c", "v", "tb"))  # noqa: E741
    f = {"c": c, "h": h, "l": l, "o": o, "live": d["live"]}
    prev = lag(c, 1)
    tr = np.fmax(h - l, np.fmax(np.abs(h - prev), np.abs(l - prev)))
    f["atr_pct"] = rmean(tr, 60) / c * 100
    f["amp_pct"] = rmean((h - l) / c * 100, 60)
    f["dist"] = (f["atr_pct"] + f["amp_pct"]) / 2  # baseline volatility unit, %
    e = ema(c, 240)
    f["ema_gap"] = (c - e) / e * 100
    r1 = np.log(c / prev)
    f["r1"] = r1
    for k in (1, 5, 15, 60, 240):
        f[f"ret{k}"] = (c / lag(c, k) - 1) * 100
    sv = 2 * tb - v  # signed taker volume: buys minus sells
    for k in (1, 5, 15, 60):
        vs = rsum(v, k)
        f[f"imb{k}"] = np.where(vs > 0, rsum(sv, k) / np.where(vs > 0, vs, 1), 0.0)
    f["vol_ratio"] = rmean(v, 5) / rmean(v, 240)  # same as the earlier rounds
    f["vz60"] = rmean(v, 60) / rmean(v, 1440)
    f["v1_ratio"] = v / rmean(v, 1440)
    sd15 = pd.Series(r1).rolling(15).std().to_numpy()
    sd1d = pd.Series(r1).rolling(1440).std().to_numpy()
    f["rv_exp"] = sd15 / sd1d  # short realized vol relative to the last day
    f["rv1d"] = sd1d * 100
    f["atr_ratio"] = f["atr_pct"] / rmean(f["atr_pct"], 1440)
    oi = np.where(d["oi"] > 0, d["oi"], np.nan)  # zero before a listing starts trading
    for k in (15, 60, 240):
        f[f"oi{k}"] = (oi / lag(oi, k) - 1) * 100
    # long liquidation proxy: open interest falling while price falls (forced sells close longs)
    f["liq_long"] = np.where(f["ret60"] < 0, np.fmax(-f["oi60"], 0), 0.0)
    f["liq_short"] = np.where(f["ret60"] > 0, np.fmax(-f["oi60"], 0), 0.0)
    if mkt is not None:
        cm = np.cumsum(np.nan_to_num(mkt))
        for k in (1, 5, 15, 60):
            f[f"mret{k}"] = (np.exp(cm - lag(cm, k)) - 1) * 100
        f["rel60"] = f["ret60"] - f["mret60"]
    return f


def market_returns(symbols=UNIVERSE) -> np.ndarray:
    """Equal weight 1m log return of the universe (coins without a live bar are skipped)."""
    c = matrix("c", symbols)
    live = matrix("live", symbols, bool)
    r = np.log(c[1:] / c[:-1])
    r = np.where(live[1:] & live[:-1], r, np.nan)
    m = np.nanmean(r, axis=1)
    return np.r_[np.nan, m]


def day_of(t_ms: np.ndarray) -> np.ndarray:
    return t_ms // (1440 * MIN)
