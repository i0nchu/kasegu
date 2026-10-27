"""E11 market-state hypotheses for the second-test (W / M) trades, one state at a time.

States, measured on the last completed UTC day before the signal (no look-ahead), signed by the trade side so
that "positive" always means "in the trade's favour" (long = bottom: market rising; short: market falling):
  btc_7d     BTC 7-day return x side          (BTC stopped falling / still falling hard)
  btc_30d    BTC 30-day return x side         (broader market trend)
  vol        log(BTC 7d / 60d realised vol)   (volatility contracting < 0 < expanding)
  corr       mean 30-day correlation of eligible coins' daily returns with BTC (market-wide correlation)
  rs_30d     (coin 30d return - BTC 30d return) x side   (coin's strength vs the market, in the trade direction)
Hypotheses are fixed before looking: better when BTC is no longer moving against the trade (btc_7d high), when
volatility is contracting (vol low) and when correlation is low; rs_30d has no prior direction.
Development: terciles cut on 2021-01 .. 2026-09; the pre-2021 hold-out is only read for the final check.
"""

import numpy as np
import pandas as pd

import common as C

FEATS = ["btc_7d", "btc_30d", "vol", "corr", "rs_30d"]
PRIOR = {"btc_7d": 1, "btc_30d": 1, "vol": -1, "corr": -1, "rs_30d": 0}  # expected sign of the effect


def daily_panel():
    closes, elig = {}, {}
    for sym in C.coins():
        X = C.load(sym)
        B = C.resample(X, 288)
        idx = B["t"] // C.DAY
        closes[sym] = pd.Series(B["c"], index=idx)
        elig[sym] = pd.Series(B["elig"].astype(bool), index=idx)
    P = pd.DataFrame(closes).sort_index()
    E = pd.DataFrame(elig).reindex(P.index).fillna(False).astype(bool)
    return P, E


def market_states():
    P, E = daily_panel()
    r = np.log(P).diff()
    btc = r["BTCUSDT"]
    M = pd.DataFrame(index=P.index)
    M["btc_7d"] = np.log(P["BTCUSDT"]).diff(7)
    M["btc_30d"] = np.log(P["BTCUSDT"]).diff(30)
    M["vol"] = np.log(btc.rolling(7).std() / btc.rolling(60).std())
    cor = r.rolling(30, min_periods=20).corr(btc)
    M["corr"] = cor.where(E).drop(columns="BTCUSDT").mean(axis=1)
    R30 = np.log(P).diff(30).sub(M["btc_30d"], axis=0)
    return M, R30


def attach(T, M, R30):
    """Add the state of the last completed day before each signal; side-signed where it has a direction."""
    d = T.t // C.DAY - 1
    for f in ("btc_7d", "btc_30d", "vol", "corr"):
        T[f] = M[f].reindex(d).to_numpy()
    rs = R30.stack()
    T["rs_30d"] = rs.reindex(list(zip(d, T.sym))).to_numpy()
    for f in ("btc_7d", "btc_30d", "rs_30d"):
        T[f] = T[f] * T.side
    return T


def terciles(T, f, cuts):
    return pd.cut(T[f], [-np.inf, *cuts, np.inf], labels=["low", "mid", "high"])


def table(T, cuts):
    rows = []
    for f in FEATS:
        q = terciles(T, f, cuts[f])
        for lab, g in T.groupby(q, observed=True):
            mu, tt, _ = C.day_t(g.t.to_numpy(), g.R.to_numpy())
            rows.append({"feature": f, "tercile": lab, "n": len(g), "win": (g.R > 0).mean(), "R": mu, "t": tt})
    return pd.DataFrame(rows)
