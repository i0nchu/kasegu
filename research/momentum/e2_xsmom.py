"""E2: cross-sectional momentum, rank coins by past return and rotate.

Concepts: Liu & Tsyvinski style weekly winner-minus-loser, the HSE result that a 2-week
sort/hold is strongest, and the "rotation into the strongest coins" part of Zarattini et al.

Rules (daily bars, rebalance every HOLD days at the close, trade at the next open):
  rank coins with at least 120 days of history by close / close[L days ago] - 1
  long_short  long the top K, short the bottom K, each leg 0.5 gross, equal weight
  long_top    long the top K at 1.0 gross
  long_top_f  long_top only while BTC is above its 100-day mean (otherwise flat)
Costs: 0.10% per side on turnover, funding charged daily.

Usage: PYTHONPATH=research/momentum uv run python research/momentum/e2_xsmom.py
"""

import numpy as np
import pandas as pd

import common as C

K = 5


def weights(D, L, hold, mode):
    c = pd.DataFrame(D["c"])
    mom = (c / c.shift(L) - 1).to_numpy().copy()
    age = c.notna().cumsum().to_numpy()
    mom[age < 120] = np.nan
    btc_ok = (c[0] > c[0].rolling(100).mean()).to_numpy()
    W = np.zeros_like(mom)
    cur = np.zeros(mom.shape[1])
    for i in range(len(mom)):
        if i % hold == 0:
            m = mom[i]
            ok = np.where(np.isfinite(m))[0]
            cur = np.zeros(mom.shape[1])
            if len(ok) >= 2 * K:
                order = ok[np.argsort(m[ok])]
                if mode == "long_short":
                    cur[order[-K:]] = 0.5 / K
                    cur[order[:K]] = -0.5 / K
                elif mode == "long_top" or (mode == "long_top_f" and btc_ok[i]):
                    cur[order[-K:]] = 1.0 / K
        cur = np.where(np.isfinite(D["c"][i]), cur, 0)
        W[i] = cur
    return W


def main():
    D = C.resample(C.panel(), 24)
    rows = []
    for mode in ("long_short", "long_top", "long_top_f"):
        for L in (7, 14, 28, 56):
            for hold in (7, 14):
                W = weights(D, L, hold, mode)
                r = C.run_weights(W, D)
                rows.append(C.flat(f"{mode}_L{L}_H{hold}", C.split_metrics(r)))
    # equal-weight basket of the same eligible coins, the benchmark the rotation must beat
    c = pd.DataFrame(D["c"])
    elig = (c.notna().cumsum() >= 120).to_numpy() & np.isfinite(D["c"])
    W = elig / np.maximum(elig.sum(axis=1, keepdims=True), 1)
    rows.append(C.flat("equal_weight_basket", C.split_metrics(C.run_weights(W, D))))
    df = pd.DataFrame(rows)
    out = C.RESULTS / "E2_xsmom"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    cols = ["name", "is_cagr", "is_sharpe", "is_mdd", "oos_cagr", "oos_sharpe", "oos_mdd"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
