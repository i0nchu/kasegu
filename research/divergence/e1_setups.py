"""E1: second-leg setups -- will price break the prior swing high, or make a double top?

Bearish setup (short side, side=-1), on 1h / 4h bars:
  L0 = lowest low in the 60 bars before swing high H1 (pivot, k=3), impulse L0 -> H1 >= 2 ATR.
  After H1 price pulls back to L1 (running low since H1), L1 > L0 (uptrend bias intact) and the
  pullback is >= 1 ATR. Trigger t = first bar after H1 is confirmed where the MACD line turns up
  (macd[t] > macd[t-1] <= macd[t-2]) while price is still below H1 and the MACD line is still
  below its peak M1 around H1. One trigger per H1, at most 60 bars after it.
Label (first touch after t, up to 120 bars): "break" = high > H1, "fail" = low < L1, else "none";
for breaks, "div" = MACD line at the break bar < M1 (a bearish divergence actually formed).
The bullish mirror (side=+1: downtrend, bounce, second leg down) runs on the negated prices.
Each event also gets the trade outcome of shorting (side) at the next 5m open with stop at
H1 + b*ATR and take profit at L1, exit after 60 bars, simulated on 5m bars on its own.
"""

import multiprocessing as mp

import numba as nb
import numpy as np
import pandas as pd

import common as C
import d3_strategy as D3
import d6_features as D6

TFS = {"1h": 12, "4h": 48}
K = 3
LOOK = 60
WAIT = 60
HORIZON = 120


@nb.njit(cache=True)
def setups(h, l, c, macd, atr, piv, trough=False):
    """piv[i] = pivot-high bar confirmed at i (or -1). Returns rows
    (t, a, b, cidx, L0, H1, L1, M1, label, label_bar, div_at_break)."""
    n = len(h)
    out = np.full((n // 4 + 1, 11), np.nan)
    k = 0
    for i in range(n):
        b = piv[i]
        if b < LOOK:
            continue
        H1 = h[b]
        a = b - LOOK
        for j in range(b - LOOK, b):
            if l[j] < l[a]:
                a = j
        L0 = l[a]
        A = atr[b]
        if A != A or A <= 0 or (H1 - L0) < 2 * A or a >= b - 2:
            continue
        M1 = -1e18
        for j in range(b - K, b + K + 1):
            if macd[j] > M1:
                M1 = macd[j]
        L1 = 1e18
        ci = -1
        broke = False
        for j in range(b + 1, i + 1):
            if h[j] > H1:
                broke = True
            if l[j] < L1:
                L1 = l[j]
                ci = j
        if broke:
            continue
        t = -1
        for j in range(i, min(n - 1, b + WAIT)):
            if h[j] > H1:
                break
            if l[j] < L1:
                L1 = l[j]
                ci = j
            if L1 <= L0:
                break
            if (j >= 2 and macd[j] > macd[j - 1] and macd[j - 1] <= macd[j - 2]
                    and macd[j] < M1 and H1 - L1 >= atr[j] and c[j] < H1 and ci < j
                    and (not trough or (ci + K <= j and ci - b >= K))):
                t = j
                break
        if t < 0:
            continue
        lab = 0.0
        lb = -1
        dv = np.nan
        for j in range(t + 1, min(n, t + 1 + HORIZON)):
            up = h[j] > H1
            dn = l[j] < L1
            if up and dn:
                lab = 3.0
                lb = j
                break
            if up:
                lab = 1.0
                lb = j
                dv = 1.0 if macd[j] < M1 else 0.0
                break
            if dn:
                lab = 2.0
                lb = j
                break
        vals = (float(t), float(a), float(b), float(ci), L0, H1, L1, M1, lab, float(lb), dv)
        for z in range(11):
            out[k, z] = vals[z]
        k += 1
    return out[:k]


def feats(B, I, R, side, a5, n5):
    """Features known at the trigger bar t (R rows from setups, on side-transformed prices)."""
    t, a, b, ci = (R[:, j].astype(np.int64) for j in range(4))
    L0, H1, L1, M1 = R[:, 4], R[:, 5], R[:, 6], R[:, 7]
    c = B["c"] * (1 if side == -1 else -1)
    atr = a5
    qv, tb = B["qv"], B["tb"]
    cq = np.nancumsum(np.nan_to_num(qv))
    cf = np.nancumsum(np.nan_to_num(2 * tb - qv)) * (1 if side == -1 else -1)  # buying in the leg direction
    fund = np.nancumsum(np.nan_to_num(B["fund"])) * (1 if side == -1 else -1)
    macd = I["macd"] * B["c"] * (1 if side == -1 else -1)

    def per_bar(cum, i0, i1):
        return (cum[i1] - cum[i0]) / np.maximum(i1 - i0, 1)

    v1 = per_bar(cq, a, b)
    v2 = per_bar(cq, ci, t)
    f1 = (cf[b] - cf[a]) / (cq[b] - cq[a])
    f2 = (cf[t] - cf[ci]) / (cq[t] - cq[ci] + 1e-12)
    r1 = np.diff(np.log(B["c"]), prepend=np.nan)
    rv = pd.Series(r1)
    rv24 = rv.rolling(24, min_periods=12).std().to_numpy()
    rv120 = rv.rolling(120, min_periods=60).std().to_numpy()
    vq30 = pd.Series(qv).rolling(180, min_periods=30).mean().to_numpy()
    e200 = C.ema(B["c"], 200)
    return {
        "retr": (H1 - L1) / (H1 - L0),
        "impulse_atr": (H1 - L0) / atr[t],
        "dist_atr": (H1 - c[t]) / atr[t],
        "recov": (c[t] - L1) / (H1 - L1),
        "macd_gap": (M1 - macd[t]) / atr[t],
        "macd_slope": (macd[t] - macd[t - 1]) / atr[t],
        "vol_leg": np.log(v2 / v1),
        "vol_now": np.log(qv[t] / vq30[t]),
        "flow1": f1,
        "flow2": f2,
        "flow_chg": f2 - f1,
        "fund24": fund[t] - fund[np.maximum(t - 288 // n5, 0)],  # funding paid by the leg side, last 24h
        "fund_chg": fund[t] - fund[b],
        "atr_chg": np.log(atr[t] / atr[b]),
        "rv_ratio": np.log(rv24[t] / rv120[t]),
        "bars_leg1": (b - a).astype(float),
        "bars_since": (t - b).astype(float),
        "trend200": -side * (B["c"][t] / e200[t] - 1),
        "rsi_t": I["rsi"][t] if side == -1 else 100 - I["rsi"][t],
        "rsi_h1": I["rsi"][b] if side == -1 else 100 - I["rsi"][b],
    }


def run_sym(sym):
    X = C.load(sym)
    rows = []
    for tf, n5 in TFS.items():
        B = C.resample(X, n5)
        if len(B["t"]) < 400:
            continue
        I = C.indicators(B)
        atr = D3.atr(B)
        macd_px = I["macd"] * B["c"]
        for side in (-1, 1):
            if side == -1:
                h, l, c, m = B["h"], B["l"], B["c"], macd_px
            else:
                h, l, c, m = -B["l"], -B["h"], -B["c"], -macd_px
            piv = C.pivots(h, K, False)
            R = setups(h, l, c, m, atr, piv)
            if not len(R):
                continue
            t = R[:, 0].astype(np.int64)
            ok = B["elig"][t] & (B["t"][t] >= C.T_IS) & (t + 1 < len(B["t"]))
            R = R[ok]
            if not len(R):
                continue
            t = R[:, 0].astype(np.int64)
            F = feats(B, I, R, side, atr, n5)
            d = pd.DataFrame(F)
            d["sym"], d["tf"], d["side"] = sym, tf, side
            d["t"] = B["t"][t]
            d["label"] = R[:, 8]
            d["div"] = R[:, 10]
            # trade: trade in the reversal direction (side) at the next 5m open
            t_end = B["t"][t] + (n5 - 1) * C.M5
            j = (np.searchsorted(X["t"], t_end) + 1).astype(np.int64)
            px = B["c"][t]
            H1 = np.abs(R[:, 5])
            L1 = np.abs(R[:, 6])
            A = atr[t]
            for buf in (0.0, 0.5, 1.0):
                stop = np.abs(H1 + buf * A * (1 if side == -1 else -1) - px) / px
                tp = np.abs(px - L1) / px
                r, xi = D6.outcome(X["o"], X["h"], X["l"], X["c"], X["fund"], j, np.full(len(t), side, np.int64),
                                   stop, tp, 60 * n5, C.COST)  # fmt: skip
                d[f"net_b{buf}"] = r
            d["tx"] = np.where(xi >= 0, X["t"][np.clip(xi, 0, len(X["t"]) - 1)], -1)
            d["stop0"] = np.abs(H1 - px) / px
            d["tp0"] = np.abs(px - L1) / px
            rows.append(d)
    return pd.concat(rows) if rows else None


def main():
    with mp.Pool(4) as pool:
        parts = pool.map(run_sym, C.coins(), chunksize=2)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True)
    D.to_pickle("data/research/divergence_e1.pkl")
    D["per"] = np.where(D.t < C.T_OOS, "is", "oos")
    lab = D.label.map({0: "none", 1: "break", 2: "fail", 3: "both"})
    print(len(D))
    print(pd.crosstab([D.tf, D.side, D.per], lab, normalize="index").round(3))
    br = D[D.label == 1]
    print("share of breaks that are divergent highs:")
    print(br.groupby(["tf", "side", "per"])["div"].mean().round(3))
    g = D.groupby(["tf", "side", "per"])
    print((1e4 * g[["net_b0.0", "net_b0.5", "net_b1.0"]].mean()).round(1))
    print(g[["stop0", "tp0"]].median().round(4))


if __name__ == "__main__":
    main()
