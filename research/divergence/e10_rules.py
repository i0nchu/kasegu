"""E10: third round of the user's manual-entry rules (long side native, short = mirrored prices).

New entry criteria on top of E9 (all known when the neckline H1 is confirmed):
  range     the rebound L0 -> H1 retraces at most 0.618 of the leg H0 -> L0; deeper means the "second
            extreme" sits near the start of the leg, i.e. a wide range on a larger cycle, not a double top/bottom
            (user's BCH 2025-07-24 short: L1 back near L0).
  trend     the leg H0 -> L0 continues a trend on this cycle: H0 is a lower high (some high in the leg-length
            window before H0 is higher). If H0 is the top of that window, the leg is the first move after an
            opposite trend, i.e. the structure belongs to a larger cycle (user's BCH 2025-07-24 short: the up leg
            started right after a decline, MACD crossing down through zero at the start; ADA 15m long: MACD had
            just poked above zero at the start and fell back). The run of MACD >= 0 before the leg ("prior") was
            also tried as the cycle test but did not separate the user's examples; it is kept as a column only.
  fresh     L0 must not already be a divergence: the MACD low of the leg is at L0 (no lower MACD low earlier in
            the leg while price made a higher low). Otherwise L0 itself was the entry (user's ADA example).
  rr        theoretical reward/risk at the entry: (TP1 - entry) / (entry - stop) >= RR. The entry price is
            set from RR instead of a fixed fraction of the neckline range, so a low-RR setup waits for a
            deeper dip toward L0 (user's ARB / APT / SOL examples); entries are only taken between L0 - 0.15
            range and the midpoint.
Exits as E9: TP1 at the neckline candle's body edge, stop L0 - 0.25 or 0.5 * range, "no divergence" exit, time exit;
"body" all at TP1, "body_half" half at TP1 then breakeven and the measured move.
Timeframes 5m (finest data; 3m is not available) up to 1d.
"""

import multiprocessing as mp
import sys

import numba as nb
import numpy as np
import pandas as pd

import common as C
import d3_strategy as D3
import e8_wbottom as E8

TFS = {"5m": 1, "15m": 3, "1h": 12, "4h": 48, "6h": 72, "8h": 96, "12h": 144, "1d": 288}
K = 3
LOOK = 60
RRS = (1.0, 1.5, 2.0, 3.0)
STOPS = (0.25, 0.5)
SYM = (0.2, 1.0)


@nb.njit(cache=True)
def setups(h, l, macd, piv, K):
    """E8 setups plus: prior positive-MACD run before the leg (bars), earliest-leg MACD low, lower-high flag.
    Rows: i, a0, a, b, H0, L0, H1, M0, cross, zmax, prior_run, early_min, lower_high."""
    n = len(h)
    out = np.full((n // 4 + 1, 13), np.nan)
    k = 0
    for i in range(n):
        b = piv[i]
        if b < 3 * LOOK:
            continue
        a = b - LOOK
        for j in range(b - LOOK, b):
            if l[j] < l[a]:
                a = j
        if b - a < 2 * K:
            continue
        ok = True
        for j in range(a - K, a):
            if l[j] <= l[a]:
                ok = False
        for j in range(a, b):
            if h[j] > h[b]:
                ok = False
        for j in range(b + 1, i + 1):
            if l[j] <= l[a]:
                ok = False
        if not ok:
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
        for j in range(a0 - K, a + 1):
            if macd[j] >= 0:
                cross = 1.0
        zmax = -1e18
        for j in range(a, i + 1):
            if macd[j] > zmax:
                zmax = macd[j]
        # last bar >= 0 at or before L0, then the length of that positive run
        jl = -1
        for j in range(a, max(a0 - LOOK, 0) - 1, -1):
            if macd[j] >= 0:
                jl = j
                break
        run = 0.0
        if jl >= 0:
            j = jl
            while j >= 0 and macd[j] >= 0:
                run += 1.0
                j -= 1
        emin = 1e18
        for j in range(a0, a - K):
            if macd[j] < emin:
                emin = macd[j]
        L = a - a0
        lh = 0.0
        for j in range(max(a0 - L, 0), a0):
            if h[j] > h[a0]:
                lh = 1.0
        vals = (float(i), float(a0), float(a), float(b), h[a0], l[a], h[b], M0, cross, zmax, run, emin, lh)
        for z in range(13):
            out[k, z] = vals[z]
        k += 1
    return out[:k]


def run_sym(sym, tfs=None, keep_all=False):
    X = C.load(sym)
    rows = []
    for tf, n5 in TFS.items():
        if tfs and tf not in tfs:
            continue
        B = C.resample(X, n5)
        if len(B["t"]) < 400:
            continue
        ml, _ = C.macd(B["c"])
        atr = D3.atr(B)
        for s in (1, -1):
            h, l, c, m = (B["h"], B["l"], B["c"], ml) if s > 0 else (-B["l"], -B["h"], -B["c"], -ml)
            bo = B["o"] if s > 0 else -B["o"]
            ho, hh, hl, hc = (X["o"], X["h"], X["l"], X["c"]) if s > 0 else (-X["o"], -X["l"], -X["h"], -X["c"])
            fund = s * X["fund"]
            R = setups(h, l, m, C.pivots(h, K, False), K)
            if not len(R):
                continue
            i = R[:, 0].astype(np.int64)
            ok = B["elig"][i] & (B["t"][i] >= C.T_IS) & (i + 1 < len(B["t"]))
            R, i = R[ok], i[ok]
            a0, a, b = (R[:, k].astype(np.int64) for k in (1, 2, 3))
            H0, L0, H1, M0, cross, zmax, run, emin, lh = (R[:, k] for k in range(4, 13))
            rng = H1 - L0
            leg = a - a0
            sr = (b - a) / leg
            rd = rng / (H0 - L0)
            # E9 context (unchanged) + time window
            keep = (M0 < 0) & (cross == 1) & (rd >= 0.382) & (zmax <= 0.25 * np.abs(M0)) & ((H0 - L0) >= 3 * atr[b])
            keep &= (sr >= SYM[0]) & (sr <= SYM[1])
            if not keep.any():
                continue
            sel = np.flatnonzero(keep)
            i, a0, a, b = i[sel], a0[sel], a[sel], b[sel]
            H0, L0, H1, M0, run, emin, lh, rng, leg, sr, rd = (
                v[sel] for v in (H0, L0, H1, M0, run, emin, lh, rng, leg, sr, rd))
            body = np.maximum(bo[b], c[b])
            t_end = B["t"][i] + (n5 - 1) * C.M5
            j = (np.searchsorted(X["t"], t_end) + 1).astype(np.int64)
            jmax0 = j + np.minimum(np.maximum(b - a, 6), LOOK) * n5
            fx = E8.divfail(c, m, i, L0, M0, len(c))
            jx = np.where(fx >= 0, np.searchsorted(X["t"], B["t"][np.clip(fx, 0, None)] + (n5 - 1) * C.M5) + 1, -1)
            jx = jx.astype(np.int64)
            hold = E8.HOLD * n5
            base = {"sym": sym, "tf": tf, "side": s, "t": B["t"][i], "t0": B["t"][a0], "ta": B["t"][a], "tb": B["t"][b],
                    "H0": s * H0, "L0": s * L0, "H1": s * H1, "body": s * body, "M0": s * M0, "sym_ratio": sr,
                    "rd": rd, "prior": run / leg, "fresh": emin >= M0, "lower_high": lh == 1, "tp1": s * body}  # fmt: skip
            nan = np.full(len(i), np.nan)
            for st in STOPS:
                sp = L0 - st * rng
                for rr in RRS:
                    lim = (body + rr * sp) / (1 + rr)  # entry where (TP1 - entry) / (entry - stop) = rr
                    x = (lim - L0) / rng
                    okx = (x >= -0.15) & (x <= 0.5) & (body > lim)
                    for ex, t2 in (("body", nan), ("body_half", H1 + rng)):
                        r = E8.sim(ho, hh, hl, hc, fund, j, np.where(okx, jmax0, j), lim, H1, L0, sp, t2, jx, hold,
                                   C.COST, body)  # fmt: skip
                        d = pd.DataFrame(base)
                        d["stop_k"], d["stop"], d["rr"], d["x"], d["exit"], d["lim"] = st, s * sp, rr, x, ex, s * lim
                        d["net"] = r[0]
                        d["t_in"] = np.where(r[1] >= 0, X["t"][np.clip(r[1], 0, None)], -1)
                        d["t_out"] = np.where(r[2] >= 0, X["t"][np.clip(r[2], 0, None)], -1)
                        d["px_in"], d["px_out"] = s * r[3], s * r[4]
                        d["why"], d["broke_L0"] = r[5], r[6]
                        rows.append(d if keep_all else d[d.t_in >= 0])
    return pd.concat(rows) if rows else None


def rules(D):
    """The user's round-3 criteria (fixed thresholds, set from the user's descriptions, not fitted)."""
    return (D.rd <= 0.618) & D.lower_high & D.fresh


def prep(D):
    D = D.copy()
    D["risk"] = (D.px_in - D.stop).abs() / D.px_in + C.COST + 0.0002
    D["R"] = D.net / D.risk
    D["rule3"] = rules(D)
    return D


def main(tfs=None):
    with mp.Pool(4) as pool:
        parts = pool.starmap(run_sym, [(s, tfs) for s in C.coins()], chunksize=2)
    D = prep(pd.concat([p for p in parts if p is not None], ignore_index=True))
    tag = "_".join(tfs) if tfs else "all"
    D.to_pickle(f"data/research/divergence_e10_{tag}.pkl")
    print(D.groupby(["tf", "rule3"]).size())


if __name__ == "__main__":
    main(sys.argv[1:] or None)
