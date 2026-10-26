"""D4: is the D3 edge divergence, or just "short a coin after a local high"?

For the D3 survivors, rerun with the same pattern and exits but (a) the indicator confirming the
new high (control) and (b) every new-high pivot. Also per-year results and the day-weighted mean,
which is what an account with few concurrent positions earns.
"""

import multiprocessing as mp

import numpy as np
import pandas as pd

import common as C
import d3_strategy as D3

CANDS = [  # tf, ind, side, trend, zone, stop mode, k, tpR, hold
    ("4h", "macd", -1, "any", None, "atr", 1.5, 2.0, 24),
    ("4h", "hist", -1, "any", None, "atr", 1.5, 2.0, 24),
    ("4h", "vol", -1, "any", None, "atr", 1.5, 2.0, 24),
    ("4h", "cvd", -1, "any", None, "atr", 1.5, 2.0, 24),
    ("4h", "rsi", -1, "any", None, "atr", 1.5, 2.0, 24),
    ("1h", "cvd", -1, "any", None, "atr", 1.5, 2.0, 24),
    ("1h", "obv", -1, "any", None, "atr", 1.5, 2.0, 24),
    ("4h", "macd", 1, "any", None, "atr", 1.5, 2.0, 24),
    ("1h", "cvd", 1, "any", None, "atr", 1.5, 2.0, 24),
]


def run_sym(sym):
    X = C.load(sym)
    out = []
    for cand in CANDS:
        tf, ind, side, trend, zone, mode, k, tpR, hold = cand
        for kind in ("div", "ctl", "all"):
            sig, sd, av = D3.signals(X, tf, ind, side, trend, zone, kind)
            if not sig.any():
                continue
            stop = k * av
            d = C.trades(X, sig, stop, tpR * stop, hold * D3.TFS[tf])
            d = d[d.t >= C.T_IS]
            if len(d):
                d["key"] = f"{tf}|{ind}|{side}"
                d["kind"] = kind
                d["sym"] = sym
                out.append(d[["key", "kind", "sym", "t", "tx", "net", "why"]])
    return pd.concat(out) if out else None


def main():
    with mp.Pool(4) as pool:
        parts = pool.map(run_sym, C.coins(), chunksize=2)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True)
    D.to_pickle("data/research/divergence_d4.pkl")
    D["yr"] = pd.to_datetime(D.t, unit="ms").dt.year
    rows = []
    for (key, kind), g in D.groupby(["key", "kind"]):
        row = {"key": key, "kind": kind}
        for tag, pm in (("is", g.t < C.T_OOS), ("oos", g.t >= C.T_OOS)):
            x = g[pm]
            mu, t, nd = C.day_t(x.t.to_numpy(), x.net.to_numpy())
            dm = x.groupby(x.t // C.DAY).net.mean().mean()
            row |= {f"{tag}_n": len(x), f"{tag}_bp": 1e4 * mu, f"{tag}_daybp": 1e4 * dm, f"{tag}_t": t}
        for y, x in g.groupby("yr"):
            row[str(y)] = 1e4 * x.net.mean()
        rows.append(row)
    S = pd.DataFrame(rows)
    (C.RESULTS / "D4_controls").mkdir(parents=True, exist_ok=True)
    S.to_csv(C.RESULTS / "D4_controls/summary.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 250)
    print(S.round(1).to_string())


if __name__ == "__main__":
    main()
