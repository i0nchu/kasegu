"""D3: turn pivot divergences into trades on 5m bars with stops, take profit and time exit.

Signal: a regular divergence confirmed at the close of a TF bar (D1 definition, pivot k=3,
previous pivot 5-60 TF bars back), optionally filtered by trend (close vs EMA200 on the TF) and
RSI zone at the pivot. Entry at the next 5m open (market). Exits on 5m bars: stop at the pivot
extreme plus a buffer, or a fixed ATR multiple; take profit as R multiple; time exit.
Usage: PYTHONPATH=research/divergence python research/divergence/d3_strategy.py
"""

import itertools
import multiprocessing as mp

import numpy as np
import pandas as pd

import common as C

TFS = {"1h": 12, "4h": 48, "15m": 3}


def atr(B, n=14):
    pc = np.roll(B["c"], 1)
    pc[0] = np.nan
    tr = np.nanmax(np.vstack([B["h"] - B["l"], np.abs(B["h"] - pc), np.abs(B["l"] - pc)]), 0)
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


_CACHE = {}


def prep(X: dict, tf: str):
    key = (id(X), tf)
    if key not in _CACHE:
        _CACHE.clear() if len(_CACHE) > 8 else None
        B = C.resample(X, TFS[tf])
        I = C.indicators(B)
        _CACHE[key] = (B, I, C.events(B, I), C.ema(B["c"], 200), atr(B))
    return _CACHE[key]


def signals(X: dict, tf: str, ind: str, side: int, trend: str, zone: float | None, kind: str = "div"):
    """kind: div = regular divergence, ctl = same new extreme with the indicator confirming, all = both.
    5m-indexed arrays: sig (side or 0), stop distance from the signal close to the pivot extreme,
    and the TF ATR as a fraction of price."""
    n5 = TFS[tf]
    B, I, E, e200, a = prep(X, tf)
    e = E[side]
    i, p, q = e["i"], e["p"], e["q"]
    v = I[ind] if not (ind == "vol" and side == 1) else -I[ind]
    ext = side * e["dp"] < 0
    div = np.sign(v[p] - v[q]) * side > 0
    m = ext & {"div": div, "ctl": ~div, "all": np.ones_like(div)}[kind]
    if trend != "any":
        tr = np.sign(B["c"][i] / e200[i] - 1) * side
        m &= (tr > 0) if trend == "with" else (tr < 0)
    if zone is not None:
        rp = I["rsi"][p]
        m &= (rp < zone) if side == 1 else (rp > 100 - zone)
    i, p = i[m], p[m]
    # map TF bar i close to the last 5m bar inside it
    t_end = B["t"][i] + (n5 - 1) * C.M5
    j = np.searchsorted(X["t"], t_end)
    ok = (j < len(X["t"])) & (X["t"][np.minimum(j, len(X["t"]) - 1)] == t_end)
    i, p, j = i[ok], p[ok], j[ok]
    sig = np.zeros(len(X["t"]), np.int64)
    sig[j] = side
    ext_px = B["l"][p] if side == 1 else B["h"][p]
    sd = np.full(len(X["t"]), np.nan)
    sd[j] = side * (B["c"][i] - ext_px) / B["c"][i]  # distance to the pivot extreme
    av = np.full(len(X["t"]), np.nan)
    av[j] = a[i] / B["c"][i]
    return sig, sd, av


GRID = {
    "tf": ["1h", "4h"],
    "ind": ["cvd", "hist", "macd", "rsi", "obv", "vol"],
    "side": [1, -1],
    "trend": ["any", "with", "against"],
    "zone": [None, 30],
}
EXITS = [  # (stop mode, stop k, tp in R, hold in TF bars)
    ("atr", 1.5, 2.0, 24),
    ("atr", 2.0, None, 24),
    ("pivot", 0.5, 2.0, 24),
    ("none", 0, None, 24),
    ("none", 0, None, 6),
]


def run_sym(sym):
    X = C.load(sym)
    out = []
    for tf, ind, side, trend, zone in itertools.product(*GRID.values()):
        sig, sd, av = signals(X, tf, ind, side, trend, zone)
        if not sig.any():
            continue
        for ex in EXITS:
            mode, k, tpR, hold = ex
            if mode == "atr":
                stop = k * av
            elif mode == "pivot":
                stop = sd + k * av
            else:
                stop = np.full_like(av, np.nan)
            tp = tpR * stop if tpR else np.nan
            d = C.trades(X, sig, stop, tp, hold * TFS[tf])
            d = d[d.t >= C.T_IS]
            if len(d):
                d["key"] = f"{tf}|{ind}|{side}|{trend}|{zone}|{mode}|{k}|{tpR}|{hold}"
                d["sym"] = sym
                out.append(d[["key", "sym", "t", "tx", "net", "gross", "why"]])
    return pd.concat(out) if out else None


def main():
    with mp.Pool(4) as pool:
        parts = pool.map(run_sym, C.coins(), chunksize=2)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True)
    D.to_pickle("data/research/divergence_d3.pkl")
    rows = []
    for key, g in D.groupby("key"):
        row = dict(zip(["tf", "ind", "side", "trend", "zone", "stop", "k", "tpR", "hold"], key.split("|")))
        for tag, pm in (("is", g.t < C.T_OOS), ("oos", g.t >= C.T_OOS)):
            x = g[pm]
            mu, t, nd = C.day_t(x.t.to_numpy(), x.net.to_numpy())
            row[f"{tag}_n"] = len(x)
            row[f"{tag}_bp"] = 1e4 * mu
            row[f"{tag}_t"] = t
            row[f"{tag}_win"] = (x.net > 0).mean()
        rows.append(row)
    S = pd.DataFrame(rows)
    (C.RESULTS / "D3_strategy").mkdir(parents=True, exist_ok=True)
    S.to_csv(C.RESULTS / "D3_strategy/summary.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    print(len(S), "configs;", (S.is_t > 2).sum(), "IS t>2;", ((S.is_t > 2) & (S.oos_t > 2)).sum(), "both")
    print(S.sort_values("is_t", ascending=False).head(30).round(2).to_string())


if __name__ == "__main__":
    main()
