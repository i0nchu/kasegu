"""E5: the manual setup with the user's refinements (2026-10-26).

Changes from E1:
- L1 must be a clear trough: a confirmed pivot low (3 bars each side) before the trigger.
- Time symmetry: pullback bars (H1 -> L1) over impulse bars (L0 -> H1) is kept as a feature and
  filtered to a band (default 0.33 - 3).
- Exits: stop above H1 with a buffer (0.5 / 1 / 1.5 ATR) against false breaks; TP1 at L1 closes
  half and moves the stop on the rest to entry (breakeven); TP2 at L0 or at the measured move
  L1 - (H1 - L1); time exit after 60 bars. "tp1 only" closes everything at L1.
Within one 5m bar the stop is checked before a target; after TP1 in the same bar the breakeven
stop counts as hit if the bar also traded back through entry (conservative).
"""

import multiprocessing as mp

import numba as nb
import numpy as np
import pandas as pd

import common as C
import d3_strategy as D3
import e1_setups as E1

TFS = {"1h": 12, "4h": 48}


@nb.njit(cache=True)
def partial(o, h, l, c, fund, j0, side, sp, t1, t2, hold, cost):
    n = len(o)
    res = np.full(len(j0), np.nan)
    why = np.full(len(j0), -1, np.int64)  # 0 time, 1 stop, 2 tp1 then breakeven, 3 tp1+tp2, 4 tp1 then time
    for e in range(len(j0)):
        j = j0[e]
        if j >= n or o[j] != o[j]:
            continue
        s = side
        px = o[j]
        stop = sp[e]
        half = 0.0
        hit1 = False
        f = 0.0
        out = np.nan
        w = 0
        m = j
        while m < n:
            if o[m] != o[m]:
                out = c[m - 1]
                w = 4 if hit1 else 0
                break
            f += fund[m]
            op = o[m] if m > j else px
            adverse = (h[m] >= stop) if s < 0 else (l[m] <= stop)
            if adverse:
                fill = max(stop, op) if s < 0 else min(stop, op)
                out = fill
                w = 2 if hit1 else 1
                break
            if not hit1:
                got1 = (l[m] <= t1[e]) if s < 0 else (h[m] >= t1[e])
                if got1:
                    f1 = min(t1[e], op) if s < 0 else max(t1[e], op)
                    half = s * (f1 / px - 1)
                    hit1 = True
                    stop = px
                    if t2[e] == t1[e]:
                        out = f1
                        w = 3
                        break
                    back = (h[m] >= px) if s < 0 else (l[m] <= px)
                    if back:
                        out = px
                        w = 2
                        break
            if hit1:
                got2 = (l[m] <= t2[e]) if s < 0 else (h[m] >= t2[e])
                if got2:
                    out = min(t2[e], op) if s < 0 else max(t2[e], op)
                    w = 3
                    break
            if m - j + 1 >= hold or m == n - 1:
                out = c[m]
                w = 4 if hit1 else 0
                break
            m += 1
        rest = s * (out / px - 1)
        g = 0.5 * half + 0.5 * rest if hit1 else rest
        res[e] = g - 2 * cost - s * f
        why[e] = w
    return res, why


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
            piv = C.pivots(h, E1.K, False)
            R = E1.setups(h, l, c, m, atr, piv, True)
            if not len(R):
                continue
            t = R[:, 0].astype(np.int64)
            ok = B["elig"][t] & (B["t"][t] >= C.T_IS) & (t + 1 < len(B["t"]))
            R = R[ok]
            if not len(R):
                continue
            t, a, b, ci = (R[:, k].astype(np.int64) for k in range(4))
            d = pd.DataFrame(E1.feats(B, I, R, side, atr, n5))
            d["sym"], d["tf"], d["side"] = sym, tf, side
            d["t"] = B["t"][t]
            d["label"] = R[:, 8]
            d["sym_ratio"] = (ci - b) / (b - a)
            d["leg2_ratio"] = (t - ci) / (b - a)
            L0, H1, L1 = np.abs(R[:, 4]), np.abs(R[:, 5]), np.abs(R[:, 6])
            A = atr[t]
            t_end = B["t"][t] + (n5 - 1) * C.M5
            j = (np.searchsorted(X["t"], t_end) + 1).astype(np.int64)
            mm = L1 + side * np.abs(H1 - L1)
            for buf in (0.5, 1.0, 1.5):
                sp = H1 - side * buf * A
                for tn, t2 in (("tp1only", L1), ("L0", L0), ("mm", mm)):
                    r, w = partial(X["o"], X["h"], X["l"], X["c"], X["fund"], j, side, sp, L1, t2, 60 * n5, C.COST)
                    d[f"net_{buf}_{tn}"] = r
                    if buf == 1.0 and tn == "L0":
                        d["why"] = w
            px = B["c"][t]
            d["risk"] = np.abs(H1 + (-side) * 1.0 * A - px) / px
            rows.append(d)
    return pd.concat(rows) if rows else None


def summarize(D, cond, name):
    rows = []
    cols = [c for c in D.columns if c.startswith("net_")]
    for (tf, side), g in D[cond].groupby(["tf", "side"]):
        for col in cols:
            _, buf, tn = col.split("_")
            row = {"set": name, "tf": tf, "side": side, "stop_buf": buf, "tp2": tn}
            for tag, pm in (("is", g.t < C.T_OOS), ("oos", g.t >= C.T_OOS)):
                x = g[col].to_numpy()[pm.to_numpy()]
                mu, t, _ = C.day_t(g.t.to_numpy()[pm.to_numpy()], x)
                row |= {f"{tag}_n": len(x), f"{tag}_bp": 1e4 * mu, f"{tag}_t": t, f"{tag}_win": np.nanmean(x > 0)}
            rows.append(row)
    return rows


def main():
    with mp.Pool(4) as pool:
        parts = pool.map(run_sym, C.coins(), chunksize=2)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True).replace([np.inf, -np.inf], np.nan)
    D.to_pickle("data/research/divergence_e5.pkl")
    symm = D.sym_ratio.between(0.33, 3.0)
    isd = D[D.t < C.T_OOS]
    med = isd.groupby(["tf", "side"])[["vol_leg", "atr_chg"]].median()
    mv = D.join(med, on=["tf", "side"], rsuffix="_med")
    filt = symm & (mv.vol_leg < mv.vol_leg_med) & (mv.atr_chg < mv.atr_chg_med)
    rows = summarize(D, pd.Series(True, index=D.index), "all") + summarize(D, symm, "symmetric")
    rows += summarize(D, filt, "symmetric+volume+atr")
    S = pd.DataFrame(rows)
    out = C.RESULTS / "E5_refined"
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "summary.csv", index=False, float_format="%.3f")
    lab = D.label.map({0: "none", 1: "break", 2: "fail", 3: "both"})
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 300)
    print(pd.crosstab([D.tf, D.side, symm], lab, normalize="index").round(3))
    # symmetry buckets: fail share and base trade
    D["sb"] = pd.cut(D.sym_ratio, [0, 0.2, 0.33, 0.6, 1.0, 1.5, 3, 100])
    print(D[D.label.isin([1, 2])].groupby(["tf", "side", "sb"], observed=True).agg(
        n=("label", "size"), fail=("label", lambda x: (x == 2).mean()),
        rw=("recov", lambda x: 1 - x.clip(0, 1).mean()), bp=("net_1.0_L0", lambda x: 1e4 * x.mean())).round(3).to_string())
    print(S.round(2).to_string())
    print(D.groupby(["tf", "side"]).why.value_counts(normalize=True).unstack().round(3))


if __name__ == "__main__":
    main()
