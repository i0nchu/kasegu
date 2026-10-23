"""E1: daily time-series trend following on 40 coins (Donchian ensemble and TSMOM).

Concepts from the literature survey:
  - Zarattini, Pagani & Barbon, "Catching Crypto Trends": ensemble of Donchian breakouts with
    different lookbacks, trailing exit, volatility-scaled sizing, rotation over liquid coins.
  - Time-series momentum (sign of the past L-day return), the one effect that the academic
    crypto literature finds robust (cross-sectional momentum is weak, see E2).

Rules (daily bars, 00:00 UTC, decided at the close and traded at the next open):
  donchian  for each L: enter long when close > highest high of the previous L days, exit when
            close < channel midline (trailing); short mirror for the long/short variant.
            Signal = mean over L in LOOKBACKS, in [-1, 1].
  tsmom     signal = mean over L of sign(close / close[L days ago] - 1).
  sizing    w = signal * min(TV / sigma_60d, 2) / N_COINS, so a full basket of fully-on coins runs
            at about TV volatility per coin; gross exposure floats with how many coins trend.
Costs: 0.10% per side on turnover, funding charged daily.

Usage: uv run python research/momentum/e1_trend.py
"""

import json

import numpy as np
import pandas as pd

import common as C

LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250)
TSM_LOOKBACKS = (7, 14, 28, 56, 112)
TV = 0.5  # target annual volatility per coin position before dividing by the number of coins


def rolling_max(a, n):
    return pd.DataFrame(a).rolling(n, min_periods=n).max().to_numpy()


def rolling_min(a, n):
    return pd.DataFrame(a).rolling(n, min_periods=n).min().to_numpy()


def donchian_state(B, L, short):
    c, h, l = B["c"], B["h"], B["l"]
    hi = np.vstack([np.full((1, c.shape[1]), np.nan), rolling_max(h, L)[:-1]])  # previous L days
    lo = np.vstack([np.full((1, c.shape[1]), np.nan), rolling_min(l, L)[:-1]])
    mid = (hi + lo) / 2
    s = np.zeros_like(c)
    pos = np.zeros(c.shape[1])
    for i in range(len(c)):
        ci = c[i]
        up = ci > hi[i]
        dn = ci < lo[i]
        pos = np.where((pos > 0) & (ci < mid[i]), 0, pos)
        pos = np.where((pos < 0) & (ci > mid[i]), 0, pos)
        pos = np.where(up, 1, pos)
        if short:
            pos = np.where(dn, -1, pos)
        pos = np.where(np.isnan(ci), 0, pos)
        s[i] = pos
    return s


def vol_scale(B, sig):
    r = pd.DataFrame(B["c"]).pct_change()
    sd = r.rolling(60, min_periods=30).std().to_numpy() * np.sqrt(365)
    lev = np.minimum(TV / sd, 2.0)
    return np.nan_to_num(sig * lev) / len(C.UNIVERSE)


def tsmom(B, short):
    c = pd.DataFrame(B["c"])
    s = np.mean([np.sign(c / c.shift(L) - 1).fillna(0).to_numpy() for L in TSM_LOOKBACKS], axis=0)
    return s if short else np.clip(s, 0, None)


def main():
    P = C.panel()
    D = C.resample(P, 24)
    rows, curves = [], {}
    sigs = {}
    for short in (False, True):
        tag = "LS" if short else "L"
        sigs[f"donchian_{tag}"] = np.mean([donchian_state(D, L, short) for L in LOOKBACKS], axis=0)
        sigs[f"tsmom_{tag}"] = tsmom(D, short)
    btc = np.zeros_like(D["c"])
    btc[:, 0] = 1.0
    sigs["buyhold_BTC"] = None
    for name, s in sigs.items():
        W = btc if s is None else vol_scale(D, s)
        for cost in (0.0005, C.COST, 0.002):
            r = C.run_weights(W, D, cost)
            m = C.split_metrics(r)
            row = C.flat(f"{name}@{cost * 100:.2f}%", m)
            row["avg_gross"] = round(float(np.abs(W).sum(axis=1).mean()), 2)
            rows.append(row)
            if cost == C.COST:
                curves[name] = r
    df = pd.DataFrame(rows)
    out = C.RESULTS / "E1_trend"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "daily_returns.csv.gz")
    json.dump({"lookbacks": LOOKBACKS, "tsm": TSM_LOOKBACKS, "tv": TV}, open(out / "config.json", "w"))
    cols = ["name", "avg_gross", "is_cagr", "is_sharpe", "is_mdd", "oos_cagr", "oos_sharpe", "oos_mdd"]
    print(df[cols].to_string(index=False))
    print(df[["name"] + [c for c in df.columns if c.startswith("y")]].to_string(index=False))


if __name__ == "__main__":
    main()
