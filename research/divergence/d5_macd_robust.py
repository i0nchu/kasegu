"""D5: robustness of the one D4 survivor, 4h bearish MACD divergence (short).

Base: 4h bars, MACD(12,26,9) line, pivot highs k=3, previous pivot 5-60 bars back, price higher
high with a lower MACD line -> short at the next 5m open; stop 1.5 ATR(14, 4h), take profit 2R,
time exit after 24 bars (4 days). Each row changes one thing from the base; every row also runs
the control (same higher high, MACD confirming) so the excess is the divergence itself.
"""

import multiprocessing as mp

import numpy as np
import pandas as pd

import common as C
import d3_strategy as D3

BASE = dict(n5=48, fast=12, slow=26, sig=9, line="macd", k=3, minsep=5, maxsep=60,
            stop=1.5, tp=2.0, hold=24, side=-1)  # fmt: skip
VARS = [("base", {})]
VARS += [(f"tf={n // 12}h", {"n5": n}) for n in (24, 72, 96)]
VARS += [(f"macd={f},{s},{g}", {"fast": f, "slow": s, "sig": g}) for f, s, g in ((8, 21, 5), (19, 39, 9), (5, 35, 5))]
VARS += [("line=hist", {"line": "hist"})]
VARS += [(f"k={k}", {"k": k}) for k in (2, 4, 5)]
VARS += [(f"maxsep={m}", {"maxsep": m}) for m in (30, 120)]
VARS += [(f"stop={s}", {"stop": s}) for s in (1.0, 2.0, 3.0)]
VARS += [(f"tp={t}", {"tp": t}) for t in (1.0, 1.5, 3.0, None)]
VARS += [(f"hold={h}", {"hold": h}) for h in (6, 12, 48)]
VARS += [("long mirror", {"side": 1})]


def signals(X, P, kind):
    B = C.resample(X, P["n5"])
    if len(B["t"]) < 300:
        return None
    c = B["c"]
    line = C.ema(c, P["fast"]) - C.ema(c, P["slow"])
    v = line if P["line"] == "macd" else line - C.ema(line, P["sig"])
    side = P["side"]
    x = B["l"] if side == 1 else B["h"]
    piv = C.pivots(x, P["k"], side == 1)
    q = C.pair(piv, P["minsep"], P["maxsep"])
    i = np.nonzero(q >= 0)[0]
    p, qq = piv[i], q[i]
    ext = side * (x[p] / x[qq] - 1) < 0
    div = np.sign(v[p] - v[qq]) * side > 0
    i = i[ext & (div if kind == "div" else ~div)]
    a = D3.atr(B)
    t_end = B["t"][i] + (P["n5"] - 1) * C.M5
    j = np.searchsorted(X["t"], t_end)
    ok = (j < len(X["t"])) & (X["t"][np.minimum(j, len(X["t"]) - 1)] == t_end)
    i, j = i[ok], j[ok]
    sig = np.zeros(len(X["t"]), np.int64)
    sig[j] = side
    av = np.full(len(X["t"]), np.nan)
    av[j] = a[i] / c[i]
    return sig, av


def run_sym(sym):
    X = C.load(sym)
    out = []
    for name, ch in VARS:
        P = BASE | ch
        for kind in ("div", "ctl"):
            r = signals(X, P, kind)
            if r is None or not r[0].any():
                continue
            sig, av = r
            stop = P["stop"] * av
            tp = P["tp"] * stop if P["tp"] else np.nan
            d = C.trades(X, sig, stop, tp, P["hold"] * P["n5"])
            d = d[d.t >= C.T_IS]
            if len(d):
                d["var"], d["kind"], d["sym"] = name, kind, sym
                out.append(d[["var", "kind", "sym", "t", "tx", "net", "why"]])
    return pd.concat(out) if out else None


def summary(D):
    rows = []
    for (var, kind), g in D.groupby(["var", "kind"], sort=False):
        row = {"var": var, "kind": kind}
        for tag, pm in (("is", g.t < C.T_OOS), ("oos", g.t >= C.T_OOS)):
            x = g[pm]
            mu, t, _ = C.day_t(x.t.to_numpy(), x.net.to_numpy())
            row |= {f"{tag}_n": len(x), f"{tag}_bp": 1e4 * mu, f"{tag}_t": t}
        cm = g.groupby("sym").net.agg(["mean", "count"])
        row["coins_pos"] = (cm[cm["count"] >= 10]["mean"] > 0).mean()
        rows.append(row)
    S = pd.DataFrame(rows)
    d = S[S.kind == "div"].set_index("var")
    c = S[S.kind == "ctl"].set_index("var")
    d["is_ex"] = d.is_bp - c.is_bp
    d["oos_ex"] = d.oos_bp - c.oos_bp
    return d.reset_index()


def main():
    with mp.Pool(4) as pool:
        parts = pool.map(run_sym, C.coins(), chunksize=2)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True)
    D.to_pickle("data/research/divergence_d5.pkl")
    S = summary(D)
    (C.RESULTS / "D5_macd_robust").mkdir(parents=True, exist_ok=True)
    S.to_csv(C.RESULTS / "D5_macd_robust/summary.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 250)
    print(S.round(2).to_string())


if __name__ == "__main__":
    main()
