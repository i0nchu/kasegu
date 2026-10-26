"""E8: Kuri's early double-bottom / double-top entry (策略修正.pdf), long side native, short = mirror.

Setup (long, after a down leg; the short side runs on negated prices):
  H0 -> L0   down leg: L0 = lowest low in the LOOK bars before the neckline, H0 = highest high in the
             LOOK bars before L0. MACD trough M0 near L0 is below zero (trend context), and the MACD
             crossed from above zero during the down leg ("由零軸上方跌破零軸").
  L0 -> H1   rebound to the neckline H1 (pivot high, k bars, highest high since L0). Same scale as
             the down leg: rebound / down leg = rd, rebound bars / down-leg bars = sym. MACD repairs
             toward zero but stays near or below it (max MACD in the rebound / |M0| = z).
  entry      after H1 is confirmed: buy limit at L0 + x * (H1 - L0), x in 0.5 / 0.25 / 0 / -0.15
             (mid of H1-L0 down to just below L0), waiting up to WAIT bars; cancelled if price
             breaks H1 first.
  outcomes   W bottom (holds L0, reaches H1), bullish divergence (breaks L0 with MACD above M0,
             then reaches H1), continuation (breaks L0 and keeps falling -> stop).
  exits      stop L0 - s * (H1 - L0); "divfail": also exit at the next bar open when a TF bar
             closes below L0 with MACD below M0 (no divergence = continuation); TP at the neckline
             H1 (full) or half at H1, stop to entry, rest to H1 + (H1 - L0) (partial); time exit.
Everything used for a setup is known at the bar that confirms H1. Simulated on 5m bars.
"""

import multiprocessing as mp
import sys

import numba as nb
import numpy as np
import pandas as pd

import common as C
import d3_strategy as D3

TFS = {"1h": 12, "4h": 48}
LOOK = 60
HOLD = 60
XS = (0.5, 0.25, 0.0, -0.15)
STOPS = (0.25, 0.5)


@nb.njit(cache=True)
def setups(h, l, macd, atr, piv, K):
    """piv[i] = pivot-high bar b confirmed at i. Rows: i, a0, a, b, H0, L0, H1, M0, cross, zmax."""
    n = len(h)
    out = np.full((n // 4 + 1, 10), np.nan)
    k = 0
    for i in range(n):
        b = piv[i]
        if b < 2 * LOOK:
            continue
        a = b - LOOK
        for j in range(b - LOOK, b):
            if l[j] < l[a]:
                a = j
        if b - a < 2 * K or a - K < 0:
            continue
        lowpiv = True  # L0 must be a pivot low on its left side too
        for j in range(a - K, a):
            if l[j] <= l[a]:
                lowpiv = False
        if not lowpiv:
            continue
        top = True
        for j in range(a, b):
            if h[j] > h[b]:
                top = False
        if not top:
            continue
        intact = True  # structure still intact when H1 is confirmed
        for j in range(b + 1, i + 1):
            if l[j] <= l[a]:
                intact = False
        if not intact:
            continue
        a0 = a - LOOK
        for j in range(a - LOOK, a):
            if h[j] > h[a0]:
                a0 = j
        if a - a0 < 2 * K:
            continue
        M0 = 1e18
        for j in range(a - K, min(a + 2 * K + 1, i + 1)):
            if macd[j] < M0:
                M0 = macd[j]
        cross = 0.0
        for j in range(max(a0 - K, 0), a + 1):
            if macd[j] >= 0:
                cross = 1.0
        zmax = -1e18
        for j in range(a, i + 1):
            if macd[j] > zmax:
                zmax = macd[j]
        vals = (float(i), float(a0), float(a), float(b), h[a0], l[a], h[b], M0, cross, zmax)
        for z in range(10):
            out[k, z] = vals[z]
        k += 1
    return out[:k]


@nb.njit(cache=True)
def divfail(c, macd, i, L0, M0, n_tf):
    """First TF bar after i that closes below L0 with MACD below M0 (within HOLD + LOOK bars), else -1."""
    E = len(i)
    f = np.full(E, -1, np.int64)
    for e in range(E):
        for j in range(i[e] + 1, min(n_tf, i[e] + HOLD + LOOK)):
            if c[j] < L0[e] and macd[j] < M0[e]:
                f[e] = j
                break
    return f


@nb.njit(cache=True)
def sim(o, h, l, c, fund, j0, jmax, lim, H1, L0, sp, t2, jx, hold, cost, tp1):
    """Long only (short side: mirrored prices, returns use |entry price|). Limit fill, then stop / tp1 / t2 / jx / time.
    The setup is cancelled if price breaks the neckline H1 before the fill.
    why: 0 time, 1 stop, 2 half at H1 then breakeven, 3 target, 4 half at H1 then time, 5 divfail exit."""
    n = len(o)
    E = len(j0)
    net = np.full(E, np.nan)
    ji = np.full(E, -1, np.int64)
    xi = np.full(E, -1, np.int64)
    epx = np.full(E, np.nan)
    xpx = np.full(E, np.nan)
    why = np.full(E, -1, np.int64)
    brk = np.zeros(E)
    for e in range(E):
        px = np.nan
        m0 = -1
        for m in range(j0[e], min(n, jmax[e])):
            if o[m] != o[m]:
                break
            if h[m] > H1[e] and l[m] > lim[e]:
                break  # neckline broke before the dip: setup gone
            if l[m] <= lim[e]:
                if o[m] <= sp[e]:
                    break  # gapped through the stop
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
            if m > m0 and m == jx[e] and not hit1:
                out = op
                w = 5
                break
            # fill bar: stop counts if reached, a target only if the bar closed beyond it
            hh = h[m] if m > m0 else max(px, c[m])
            ll = l[m]
            if ll < L0[e]:
                brk[e] = 1.0
            if ll <= stop:
                out = min(stop, op)
                w = 2 if hit1 else 1
                break
            if not hit1 and hh >= tp1[e]:
                f1 = max(tp1[e], op)
                if t2[e] != t2[e]:
                    out = f1
                    w = 3
                    break
                half = (f1 - px) / abs(px)
                hit1 = True
                stop = px
            if hit1 and hh >= t2[e]:
                out = max(t2[e], op)
                w = 3
                break
            if m - m0 + 1 >= hold or m == n - 1:
                out = c[m]
                w = 4 if hit1 else 0
                break
            m += 1
        g = (out - px) / abs(px)
        if hit1:
            g = 0.5 * half + 0.5 * g
        net[e] = g - (cost + 0.0002)  # maker entry, taker exit
        net[e] -= f  # funding: longs pay positive funding
        ji[e], xi[e], epx[e], xpx[e], why[e] = m0, m, px, out, w
    return net, ji, xi, epx, xpx, why, brk


def run_sym(sym, K=3):
    X = C.load(sym)
    rows = []
    for tf, n5 in TFS.items():
        B = C.resample(X, n5)
        if len(B["t"]) < 400:
            continue
        ml, _ = C.macd(B["c"])
        atr = D3.atr(B)
        for side in (1, -1):
            s = side
            h, l, c, m = (B["h"], B["l"], B["c"], ml) if s > 0 else (-B["l"], -B["h"], -B["c"], -ml)
            ho, hh, hl, hc = (X["o"], X["h"], X["l"], X["c"]) if s > 0 else (-X["o"], -X["l"], -X["h"], -X["c"])
            fund = s * X["fund"]
            piv = C.pivots(h, K, False)
            R = setups(h, l, m, atr, piv, K)
            if not len(R):
                continue
            i = R[:, 0].astype(np.int64)
            ok = B["elig"][i] & (B["t"][i] >= C.T_IS) & (i + 1 < len(B["t"]))
            R, i = R[ok], i[ok]
            if not len(R):
                continue
            a0, a, b = (R[:, k].astype(np.int64) for k in (1, 2, 3))
            H0, L0, H1, M0, cross, zmax = (R[:, k] for k in range(4, 10))
            A = atr[b]
            rng = H1 - L0
            t_end = B["t"][i] + (n5 - 1) * C.M5
            j = (np.searchsorted(X["t"], t_end) + 1).astype(np.int64)
            wait = np.minimum(np.maximum(b - a, 6), LOOK)  # wait for the dip about as long as the rebound took
            jmax = j + wait * n5
            fx = divfail(c, m, i, L0, M0, len(c))
            jx = np.where(fx >= 0, np.searchsorted(X["t"], B["t"][np.clip(fx, 0, None)] + (n5 - 1) * C.M5) + 1, -1)
            jx = jx.astype(np.int64)
            base = {
                "sym": sym, "tf": tf, "side": side, "K": K, "t": B["t"][i], "t0": B["t"][a0], "ta": B["t"][a],
                "tb": B["t"][b], "H0": s * H0, "L0": s * L0, "H1": s * H1, "M0": s * M0, "atr": A,
                "rd": rng / (H0 - L0), "sym_ratio": (b - a) / (a - a0), "z": zmax / np.abs(M0),
                "cross": cross, "leg_atr": (H0 - L0) / A, "reb_atr": rng / A, "m0_neg": M0 < 0,
            }  # fmt: skip
            for x in XS:
                lim = L0 + x * rng
                for st in STOPS:
                    sp = L0 - st * rng
                    for ex, t2 in (("full", np.full(len(i), np.nan)), ("partial", H1 + rng)):
                        for dfx in (0, 1):
                            r = sim(ho, hh, hl, hc, fund, j, jmax, lim, H1, L0, sp, t2,
                                    jx if dfx else np.full(len(i), -1, np.int64), HOLD * n5, C.COST, H1)  # fmt: skip
                            d = pd.DataFrame(base)
                            d["x"], d["stop_k"], d["exit"], d["divfail"] = x, st, ex, dfx
                            d["net"] = r[0]
                            d["t_in"] = np.where(r[1] >= 0, X["t"][np.clip(r[1], 0, None)], -1)
                            d["t_out"] = np.where(r[2] >= 0, X["t"][np.clip(r[2], 0, None)], -1)
                            d["px_in"], d["px_out"] = s * r[3], s * r[4]
                            d["why"], d["broke_L0"] = r[5], r[6]
                            d["stop"], d["lim"] = s * sp, s * lim
                            rows.append(d)
    return pd.concat(rows) if rows else None


def rule(D):
    """Kuri's context filters, fixed in advance (not tuned). Rebound/down-leg time ratio is loose (0.2-5):
    Kuri's own BCH example (2025-12-15 L0) has a 2-day rebound after an 8-day down leg (0.23)."""
    return (D.m0_neg & (D.cross == 1) & D.rd.between(0.382, 1.0) & D.sym_ratio.between(0.2, 5)
            & (D.z <= 0.25) & (D.leg_atr >= 3))  # fmt: skip


def summary(D):
    D = D[D.t_in >= 0].copy()
    D["rule"] = rule(D)
    rows = []
    for key, g in D.groupby(["tf", "side", "x", "stop_k", "exit", "divfail"]):
        for st, gg in (("all", g), ("rule", g[g.rule])):
            row = dict(zip(["tf", "side", "x", "stop_k", "exit", "divfail"], key)) | {"set": st}
            for tag, pm in (("is", gg.t < C.T_OOS), ("oos", gg.t >= C.T_OOS)):
                x = gg[pm]
                mu, tt, _ = C.day_t(x.t.to_numpy(), x.net.to_numpy())
                row |= {f"{tag}_n": len(x), f"{tag}_bp": 1e4 * mu, f"{tag}_t": tt, f"{tag}_win": (x.net > 0).mean()}
            rows.append(row)
    return pd.DataFrame(rows)


def main(K):
    with mp.Pool(4) as pool:
        parts = pool.starmap(run_sym, [(s, K) for s in C.coins()], chunksize=2)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True)
    D.to_pickle(f"data/research/divergence_e8_k{K}.pkl")
    S = summary(D)
    out = C.RESULTS / "E8_wbottom"
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / f"summary_k{K}.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(S[S.set == "rule"].round(2).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
