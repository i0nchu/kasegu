"""D1: event study of classic pivot divergences on 15m / 1h / 4h bars.

For every confirmed pivot low (high) paired with the previous pivot 5-60 bars earlier:
  regular bullish (bearish) divergence = price lower low (higher high) while the indicator makes a
  higher low (lower high); hidden divergence = price higher low (lower high) while the indicator
  makes a lower low (higher high). Volume divergence = new extreme on lower volume.
Signal at the confirmation bar close (pivot + 3 bars), entry at the next open, exit at the close
H bars later. Control group = same price pattern without the indicator divergence.
Usage: PYTHONPATH=research/divergence python research/divergence/d1_events.py
"""

import multiprocessing as mp
import pickle

import numpy as np
import pandas as pd

import common as C

TFS = {"15m": 3, "1h": 12, "4h": 48}
HS = (1, 3, 6, 12, 24)
INDS = ("macd", "hist", "rsi", "obv", "cvd", "vol")
K = 3


def one(sym: str) -> list[pd.DataFrame]:
    X = C.load(sym)
    out = []
    for tf, n in TFS.items():
        B = C.resample(X, n)
        if len(B["t"]) < 300:
            continue
        I = C.indicators(B)
        E = C.events(B, I, k=K)
        for side, e in E.items():
            i, p, q = e["i"], e["p"], e["q"]
            ok = (i + 1 < len(B["t"])) & B["elig"][i] & (B["t"][i] >= C.T_IS)
            i, p, q, dp = i[ok], p[ok], q[ok], e["dp"][ok]
            if not len(i):
                continue
            d = {"t": B["t"][i], "side": side, "dp": dp, "sym": sym, "tf": tf}
            for nm in INDS:
                v = I[nm]
                if nm == "vol" and side == 1:
                    v = -v
                dv = v[p] - v[q]
                d[nm] = np.sign(dv) * side  # +1: indicator disagrees with a new price extreme
            for H in HS:
                d[f"r{H}"] = C.fwd(B, i, H, side)
            # context: trend (close vs 200-bar EMA) and depth of the move
            e200 = C.ema(B["c"], 200)
            d["trend"] = np.sign(B["c"][i] / e200[i] - 1) * side  # +1: signal with the trend
            d["rsi_p"] = I["rsi"][p]
            out.append(pd.DataFrame(d))
    return out


def main():
    syms = C.coins()
    with mp.Pool(4) as pool:
        parts = pool.map(one, syms, chunksize=4)
    D = pd.concat([x for p in parts for x in p], ignore_index=True)
    D.to_pickle("data/research/divergence_d1.pkl")
    print(len(D))


if __name__ == "__main__":
    main()
