"""D2: continuous divergence factors, Fama-MacBeth on the hourly top-40 panel.

Question: holding the price move fixed, does an indicator that disagrees with it predict the next
hours? Each hour, across eligible coins, regress the forward log return on the past return R_L and
one indicator factor X_L (both cross-sectional z-scores). The X coefficient is the divergence
effect: e.g. positive for flow means coins whose taker flow was stronger than their price move
outperform. t-stats use Newey-West with lag = horizon. Also reports X alone and R alone.
"""

import numpy as np
import pandas as pd

import common as C
import panel

P = panel.build(12)
T = P["t"]
c = P["c"].astype(float)
qv = P["qv"].astype(float)
tb = P["tb"].astype(float)
el = P["elig"]
lc = np.log(c)
r1 = np.diff(lc, axis=0, prepend=np.nan)


def rsum(x, L):
    s = np.nancumsum(np.nan_to_num(x), axis=0)
    o = np.full_like(s, np.nan)
    o[L:] = s[L:] - s[:-L]
    return o


def lag(x, L):
    o = np.full_like(x, np.nan)
    o[L:] = x[:-L]
    return o


def hist():
    h = np.full_like(c, np.nan)
    for j in range(c.shape[1]):
        m = ~np.isnan(c[:, j])
        if m.sum() > 50:
            h[m, j] = C.macd(c[m, j])[1]
    return h


def rsi_all():
    x = np.full_like(c, np.nan)
    for j in range(c.shape[1]):
        m = ~np.isnan(c[:, j])
        if m.sum() > 50:
            x[m, j] = C.rsi(c[m, j])
    return x


HIST = hist()
RSI = rsi_all()


def factors(L):
    R = lc - lag(lc, L)
    V = rsum(qv, L)
    F = {
        "flow": rsum(2 * tb - qv, L) / V,  # CVD change, normalized
        "obv": rsum(np.sign(r1) * qv, L) / V,  # OBV change, normalized
        "volchg": np.log(V / lag(V, L)),  # volume rising vs falling
        "hist": HIST - lag(HIST, L),
        "rsi": RSI - lag(RSI, L),
    }
    F["volxR"] = F["volchg"] * np.sign(R)  # >0: move on rising volume
    return R, F


def zcs(x, m):
    x = np.where(m, x, np.nan)
    mu = np.nanmean(x, 1, keepdims=True)
    sd = np.nanstd(x, 1, keepdims=True)
    return (x - mu) / sd


def nw_t(b, lag_):
    b = b[~np.isnan(b)]
    n = len(b)
    if n < 50:
        return np.nan
    e = b - b.mean()
    v = e @ e / n
    for k in range(1, lag_ + 1):
        v += 2 * (1 - k / (lag_ + 1)) * (e[k:] @ e[:-k]) / n
    return b.mean() / np.sqrt(v / n)


def fm(y, Xs, m):
    """Per-row OLS of y on [1, Xs...]; returns coefficient matrix rows x len(Xs)."""
    nT = y.shape[0]
    B = np.full((nT, len(Xs)), np.nan)
    for t in range(nT):
        mm = m[t] & ~np.isnan(y[t])
        for X in Xs:
            mm &= ~np.isnan(X[t])
        if mm.sum() < 15:
            continue
        A = np.column_stack([np.ones(mm.sum())] + [X[t, mm] for X in Xs])
        B[t] = np.linalg.lstsq(A, y[t, mm], rcond=None)[0][1:]
    return B


def main():
    rows = []
    for H in (1, 4, 12, 24):
        y = lag(lc, -H) - lc if False else np.vstack([lc[H:] - lc[:-H], np.full((H, lc.shape[1]), np.nan)])
        for L in (4, 12, 24, 72):
            R, F = factors(L)
            ok = el & ~np.isnan(R)
            zR = zcs(R, ok)
            bR = fm(y, [zR], ok)[:, 0]
            for nm, X in F.items():
                zX = zcs(X, ok)
                B = fm(y, [zR, zX], ok)
                bX = fm(y, [zX], ok)[:, 0]
                for tag, pm in (("is", (T >= C.T_IS) & (T < C.T_OOS)), ("oos", T >= C.T_OOS)):
                    row = {"H": H, "L": L, "factor": nm, "period": tag}
                    row["b_div_bp"] = 1e4 * np.nanmean(B[pm, 1])
                    row["t_div"] = nw_t(B[pm, 1], H)
                    row["b_alone_bp"] = 1e4 * np.nanmean(bX[pm])
                    row["t_alone"] = nw_t(bX[pm], H)
                    row["b_R_bp"] = 1e4 * np.nanmean(bR[pm])
                    row["t_R"] = nw_t(bR[pm], H)
                    rows.append(row)
            print(H, L, flush=True)
    S = pd.DataFrame(rows)
    (C.RESULTS / "D2_fm").mkdir(parents=True, exist_ok=True)
    S.to_csv(C.RESULTS / "D2_fm/summary.csv", index=False, float_format="%.3f")
    W = S.pivot_table(index=["factor", "L", "H"], columns="period", values=["b_div_bp", "t_div", "t_alone", "t_R"])
    pd.set_option("display.width", 200)
    print(W.round(2).to_string())


if __name__ == "__main__":
    main()
