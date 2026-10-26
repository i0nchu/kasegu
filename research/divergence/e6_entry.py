"""E6: entry location and full vs partial take profit for the second-leg double-top trade.

Same setups as E5 (clear trough L1, trigger when the MACD line turns up below H1). Entries:
  market  next 5m open after the trigger (E5)
  limX    sell limit at L1 + X * (H1 - L1), X = 0.5 / 0.75 / 0.9, waiting up to 60 TF bars after
          the trigger; cancelled if price breaks L1 or H1 first (a gap through the limit fills at
          the open). Closer to H1 = shorter stop, longer target.
Exits: stop at H1 + 0.5 or 1 ATR; "full" closes everything at L1; "partial" closes half at L1,
moves the stop to entry, the rest targets the measured move L1 - (H1 - L1); time exit 60 bars
after entry. Long side is the mirror.
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
def sim(o, h, l, c, fund, j0, jmax, s, lim, lo_cancel, hi_cancel, sp, t1, t2, hold, cost):
    """Per setup: wait for the limit (lim NaN = market at o[j0]); then stop / tp1 / tp2 / time.
    For s=-1 lo_cancel is L1 (cancel if traded below before fill), hi_cancel is H1."""
    n = len(o)
    E = len(j0)
    net = np.full(E, np.nan)
    ji = np.full(E, -1, np.int64)
    xi = np.full(E, -1, np.int64)
    epx = np.full(E, np.nan)
    xpx = np.full(E, np.nan)
    why = np.full(E, -1, np.int64)
    for e in range(E):
        j = j0[e]
        if j >= n or o[j] != o[j]:
            continue
        # entry
        if lim[e] != lim[e]:
            px = o[j]
            m0 = j
        else:
            px = np.nan
            m0 = -1
            for m in range(j, min(n, jmax[e])):
                if o[m] != o[m]:
                    break
                if s < 0:
                    if l[m] < lo_cancel[e] and h[m] < lim[e]:
                        break
                    if h[m] > hi_cancel[e] and o[m] > hi_cancel[e]:
                        break
                    if h[m] >= lim[e]:
                        px = max(lim[e], o[m])
                        m0 = m
                        break
                else:
                    if h[m] > lo_cancel[e] and l[m] > lim[e]:
                        break
                    if l[m] < hi_cancel[e] and o[m] < hi_cancel[e]:
                        break
                    if l[m] <= lim[e]:
                        px = min(lim[e], o[m])
                        m0 = m
                        break
            if m0 < 0:
                continue
        stop = sp[e]
        hit1 = False
        half = 0.0
        f = 0.0
        out = np.nan
        w = 0
        m = m0
        while m < n:
            if o[m] != o[m]:
                out = c[m - 1]
                m -= 1
                break
            f += fund[m]
            op = o[m] if m > m0 else px
            # fill bar: order inside the bar unknown, so the stop counts if the bar reached it and a
            # target only if the bar closed beyond it (conservative both ways)
            hh = h[m] if m > m0 else (h[m] if s < 0 else max(px, c[m]))
            ll = l[m] if m > m0 else (l[m] if s > 0 else min(px, c[m]))
            if (s < 0 and hh >= stop) or (s > 0 and ll <= stop):
                out = max(stop, op) if s < 0 else min(stop, op)
                w = 2 if hit1 else 1
                break
            if not hit1 and ((s < 0 and ll <= t1[e]) or (s > 0 and hh >= t1[e])):
                f1 = min(t1[e], op) if s < 0 else max(t1[e], op)
                if t2[e] != t2[e]:
                    out = f1
                    w = 3
                    break
                half = s * (f1 / px - 1)
                hit1 = True
                stop = px
                if (s < 0 and hh >= px) or (s > 0 and ll <= px):
                    out = px
                    w = 2
                    break
            if hit1 and ((s < 0 and ll <= t2[e]) or (s > 0 and hh >= t2[e])):
                out = min(t2[e], op) if s < 0 else max(t2[e], op)
                w = 3
                break
            if m - m0 + 1 >= hold or m == n - 1:
                out = c[m]
                w = 4 if hit1 else 0
                break
            m += 1
        g = 0.5 * half + 0.5 * s * (out / px - 1) if hit1 else s * (out / px - 1)
        cst = 2 * cost if lim[e] != lim[e] else cost + 0.0002  # limit entry pays the 0.02% maker fee
        net[e] = g - cst - s * f
        ji[e], xi[e], epx[e], xpx[e], why[e] = m0, m, px, out, w
    return net, ji, xi, epx, xpx, why


def run_sym(sym, keep_rows=False):
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
            symr = (ci - b) / (b - a)
            L0, H1, L1 = np.abs(R[:, 4]), np.abs(R[:, 5]), np.abs(R[:, 6])
            A = atr[t]
            t_end = B["t"][t] + (n5 - 1) * C.M5
            j = (np.searchsorted(X["t"], t_end) + 1).astype(np.int64)
            jmax = j + 60 * n5
            rng = np.abs(H1 - L1)
            mm = L1 + side * rng
            base = {"sym": sym, "tf": tf, "side": side, "t": B["t"][t], "a": B["t"][a], "b": B["t"][b],
                    "ci": B["t"][ci], "L0": L0, "H1": H1, "L1": L1, "atr": A, "sym_ratio": symr}  # fmt: skip
            for ent in ("market", "lim0.5", "lim0.75", "lim0.9"):
                if ent == "market":
                    lim = np.full(len(t), np.nan)
                else:
                    x = float(ent[3:])
                    lim = L1 - side * x * rng  # short: L1 + x * range
                for buf in (0.5, 1.0):
                    sp = H1 - side * buf * A
                    for ex, t2 in (("full", np.full(len(t), np.nan)), ("partial", mm)):
                        r = sim(X["o"], X["h"], X["l"], X["c"], X["fund"], j, jmax, side, lim, L1, H1,
                                sp, L1, t2, 60 * n5, C.COST)  # fmt: skip
                        d = pd.DataFrame(base)
                        d["entry"], d["buf"], d["exit"] = ent, buf, ex
                        d["net"] = r[0]
                        d["t_in"] = np.where(r[1] >= 0, X["t"][np.clip(r[1], 0, None)], -1)
                        d["t_out"] = np.where(r[2] >= 0, X["t"][np.clip(r[2], 0, None)], -1)
                        d["px_in"], d["px_out"], d["why"] = r[3], r[4], r[5]
                        d["stop"] = sp
                        rows.append(d)
    return pd.concat(rows) if rows else None


def main():
    with mp.Pool(4) as pool:
        parts = pool.map(run_sym, C.coins(), chunksize=2)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True)
    D.to_pickle("data/research/divergence_e6.pkl")
    D["set"] = np.where(D.sym_ratio.between(0.33, 3.0), "symmetric", "other")
    rows = []
    for key, g in D.groupby(["tf", "side", "entry", "buf", "exit"]):
        for st, gg in (("all", g), ("symmetric", g[g.set == "symmetric"])):
            row = dict(zip(["tf", "side", "entry", "buf", "exit"], key)) | {"set": st}
            row["fill_rate"] = gg.net.notna().mean()
            for tag, pm in (("is", gg.t < C.T_OOS), ("oos", gg.t >= C.T_OOS)):
                x = gg[pm & gg.net.notna()]
                mu, tt, _ = C.day_t(x.t.to_numpy(), x.net.to_numpy())
                row |= {f"{tag}_n": len(x), f"{tag}_bp": 1e4 * mu, f"{tag}_t": tt, f"{tag}_win": (x.net > 0).mean()}
            rows.append(row)
    S = pd.DataFrame(rows)
    out = C.RESULTS / "E6_entry"
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "summary.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 300)
    print(S.round(2).to_string())


if __name__ == "__main__":
    main()
