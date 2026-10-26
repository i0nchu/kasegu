"""D6: what makes a divergence work? Feature mining on every 4h pivot pair.

Every 4h higher-high (short) and lower-low (long) pivot pair, divergent or not, gets the base trade
outcome (entry next 5m open, stop 1.5 ATR, take profit 2R, exit after 24 bars, each event simulated
on its own) and a set of features known at the signal: divergence strength for each indicator,
funding, relative strength vs BTC, trend, RSI, volume. Buckets are cut on IS quantiles and checked
on OOS; a linear score fitted on IS is checked on OOS.
"""

import multiprocessing as mp

import numba as nb
import numpy as np
import pandas as pd

import common as C
import d3_strategy as D3

N5 = 48
HOLD = 24


@nb.njit(cache=True)
def outcome(o, h, l, c, fund, j0, side, stop, tp, hold, cost):  # noqa: E741
    """Net return and exit bar of one trade per entry bar j0 (each simulated alone, same rules as
    C.simulate)."""
    res = np.full(len(j0), np.nan)
    xi = np.full(len(j0), -1, np.int64)
    for e in range(len(j0)):
        j = j0[e]
        if j >= len(o) or o[j] != o[j]:
            continue
        s = side[e]
        px = o[j]
        sp = px * (1 - s * stop[e])
        tq = px * (1 + s * tp[e])
        f = 0.0
        ex = np.nan
        m = j
        while m < len(o):
            if o[m] != o[m]:
                ex = c[m - 1]
                m -= 1
                break
            f += fund[m]
            op = o[m] if m > j else px
            if s > 0:
                if l[m] <= sp:
                    ex = min(sp, op)
                    break
                if h[m] >= tq:
                    ex = max(tq, op)
                    break
            else:
                if h[m] >= sp:
                    ex = max(sp, op)
                    break
                if l[m] <= tq:
                    ex = min(tq, op)
                    break
            if m - j + 1 >= hold or m == len(o) - 1:
                ex = c[m]
                break
            m += 1
        res[e] = s * (ex / px - 1) - 2 * cost - s * f
        xi[e] = m
    return res, xi


_BTC = {}


def btc():
    if not _BTC:
        B = C.resample(C.load("BTCUSDT"), N5)
        _BTC.update(B)
    return _BTC


def run_sym(sym):
    X = C.load(sym)
    B = C.resample(X, N5)
    if len(B["t"]) < 300:
        return None
    I = C.indicators(B)
    E = C.events(B, I)
    a = D3.atr(B)
    e200 = C.ema(B["c"], 200)
    fz = pd.Series(B["fund"]).rolling(6, min_periods=1).sum().to_numpy()  # funding paid over 24h
    Bb = btc()
    bi = np.searchsorted(Bb["t"], B["t"])
    bi = np.clip(bi, 0, len(Bb["t"]) - 1)
    bc = np.where(Bb["t"][bi] == B["t"], Bb["c"][bi], np.nan)
    vq = pd.Series(B["qv"]).rolling(180, min_periods=30).mean().to_numpy()  # 30-day avg volume
    hi30 = pd.Series(B["h"]).rolling(180, min_periods=30).max().to_numpy()
    lo30 = pd.Series(B["l"]).rolling(180, min_periods=30).min().to_numpy()
    rows = []
    for side, e in E.items():
        i, p, q = e["i"], e["p"], e["q"]
        ext = side * e["dp"] < 0
        i, p, q, dp = i[ext], p[ext], q[ext], e["dp"][ext]
        ok = B["elig"][i] & (B["t"][i] >= C.T_IS) & (i + 1 < len(B["t"]))
        i, p, q, dp = i[ok], p[ok], q[ok], dp[ok]
        if not len(i):
            continue
        t_end = B["t"][i] + (N5 - 1) * C.M5
        j = np.searchsorted(X["t"], t_end) + 1
        stop = 1.5 * a[i] / B["c"][i]
        r, xi = outcome(X["o"], X["h"], X["l"], X["c"], X["fund"], j.astype(np.int64),
                    np.full(len(i), side, np.int64), stop, 2 * stop, HOLD * N5, C.COST)  # fmt: skip
        tx = np.where(xi >= 0, X["t"][np.clip(xi, 0, len(X["t"]) - 1)], -1)
        d = {"sym": sym, "t": B["t"][i], "tx": tx, "side": side, "net": r, "dp": side * -dp}
        for nm in ("macd", "hist", "rsi"):
            d[nm] = side * (I[nm][p] - I[nm][q])  # >0 = divergence (indicator disagrees)
        seg = I["cvd"][p] - I["cvd"][q]
        vol = pd.Series(B["qv"]).cumsum().to_numpy()
        d["cvd"] = side * seg / (vol[p] - vol[q])
        d["obv"] = side * (I["obv"][p] - I["obv"][q]) / (vol[p] - vol[q])
        d["volr"] = np.log(I["vol"][p] / I["vol"][q])
        d["vol30"] = np.log(I["vol"][i] / vq[i])
        d["fund"] = fz[i]
        d["fundchg"] = fz[p] - fz[q]
        d["btcrel"] = side * -(np.log(B["c"][p] / B["c"][q]) - np.log(bc[p] / bc[q]))
        d["trend"] = side * (B["c"][i] / e200[i] - 1)
        d["rsi_p"] = I["rsi"][p]
        d["sep"] = p - q
        d["atr"] = a[i] / B["c"][i]
        d["pos30"] = (B["c"][i] - lo30[i]) / (hi30[i] - lo30[i])
        rows.append(pd.DataFrame(d))
    return pd.concat(rows) if rows else None


FEATS = ["dp", "macd", "hist", "rsi", "cvd", "obv", "volr", "vol30", "fund", "fundchg", "btcrel",
         "trend", "rsi_p", "sep", "atr", "pos30"]  # fmt: skip


def buckets(D):
    out = []
    for side, g in D.groupby("side"):
        isg = g[g.t < C.T_OOS]
        for f in FEATS:
            qs = isg[f].quantile([1 / 3, 2 / 3]).to_numpy()
            b = np.digitize(g[f], qs)
            for k in range(3):
                row = {"side": side, "feat": f, "bucket": k}
                for tag, pm in (("is", g.t < C.T_OOS), ("oos", g.t >= C.T_OOS)):
                    m = ((b == k) & pm).to_numpy()
                    mu, t, _ = C.day_t(g.t.to_numpy()[m], g.net.to_numpy()[m])
                    row |= {f"{tag}_n": m.sum(), f"{tag}_bp": 1e4 * mu, f"{tag}_t": t}
                out.append(row)
    return pd.DataFrame(out)


def main():
    with mp.Pool(4) as pool:
        parts = pool.map(run_sym, C.coins(), chunksize=2)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True)
    D = D[np.isfinite(D.net)]
    D.to_pickle("data/research/divergence_d6.pkl")
    S = buckets(D)
    (C.RESULTS / "D6_features").mkdir(parents=True, exist_ok=True)
    S.to_csv(C.RESULTS / "D6_features/buckets.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(S.round(1).to_string())


if __name__ == "__main__":
    main()
