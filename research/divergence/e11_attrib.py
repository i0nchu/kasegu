"""E11: alpha attribution of E10 plus a backtest audit, on all data (development 2021-01 .. 2026-09 and the
untouched hold-out before 2021 / from 2026-10).

Versions, all with the same entry pricing (limit at theoretical reward/risk 1.5 from TP1 = neckline candle body,
stop L0 - 0.5 * (H1 - L0)) and the same exits (TP1 all, stop, 60-bar time exit; no MACD exit):
  A  random timing: for each D trade, NRAND market entries at random 5m bars of the same coin within +-90 days,
     same side, same stop and target distances in percent -> what the distances alone give.
  B  price structure only: pivots L0 / H1, H1 highest since L0, rebound 0.382-1.0 of the leg, correction time
     0.2-1.0x the leg, leg >= 3 ATR.
  C  B + MACD context: MACD trough at L0 below zero, MACD crossed zero during the leg, MACD only repairs to
     near zero after L0.
  D  C + the user's round-3 rules (rebound <= 0.618, start is a lower high, L0 not already a divergence) = E10.
  D+ D with the "no divergence" MACD exit (E10 as run).

Audit changes versus E8-E10 simulation (sim2):
  - limit orders (entry and TP1) can need a trade-through: price must pass the limit by THR (0.05%), not just
    touch it, to count as filled (queue risk). Touch-only fills (THR = 0) are kept for comparison.
  - an entry is cancelled if the "no divergence" condition already closed before the fill.
  - unchanged and checked: pivots are used only from the bar that confirms them (k bars to the right); orders start
    on the first 5m bar after the signal bar closes; inside a 5m bar the stop is assumed first; a gap through
    the stop exits at the open; a gap through a limit fills at the open; the MACD exit acts on the next bar
    after the signal-timeframe bar closes; exits pay taker fee + slippage, entries pay the maker fee.
"""

import multiprocessing as mp
import zlib

import numba as nb
import numpy as np
import pandas as pd

import common as C
import d3_strategy as D3
import e8_wbottom as E8
import e10_rules as E10

TFS = {"15m": 3, "1h": 12, "4h": 48, "12h": 144, "1d": 288}
K = 3
RR = 1.5
STOP = 0.5
THRS = (0.0, 0.0005)
NRAND = 5
SEED = 11


@nb.njit(cache=True)
def sim2(o, h, l, c, fund, j0, jmax, lim, H1, L0, sp, jx, hold, cost, tp1, thr):
    """E8.sim (TP1 all) with trade-through fills for the entry and TP1 limits and cancel-on-divfail-before-fill.
    Returns net, entry index, exit index, entry price, exit price, why (0 time, 1 stop, 3 TP1, 5 MACD exit),
    touched (limit touched but not filled)."""
    n = len(o)
    E = len(j0)
    net = np.full(E, np.nan)
    ji = np.full(E, -1, np.int64)
    xi = np.full(E, -1, np.int64)
    epx = np.full(E, np.nan)
    xpx = np.full(E, np.nan)
    why = np.full(E, -1, np.int64)
    touched = np.zeros(E)
    for e in range(E):
        px = np.nan
        m0 = -1
        for m in range(j0[e], min(n, jmax[e])):
            if o[m] != o[m]:
                break
            if jx[e] >= 0 and m >= jx[e]:
                break  # continuation already confirmed: no entry
            if h[m] > H1[e] and l[m] > lim[e]:
                break
            if o[m] <= lim[e]:  # gap through the limit: filled at the open
                if o[m] <= sp[e]:
                    break
                px = o[m]
                m0 = m
                break
            if l[m] <= lim[e]:
                if l[m] < lim[e] - thr * abs(lim[e]):
                    px = lim[e]
                    m0 = m
                    break
                touched[e] = 1.0
        if m0 < 0:
            continue
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
            if m > m0 and m == jx[e]:
                out = op
                w = 5
                break
            if m > m0 and op >= tp1[e]:  # gap through TP1
                out = op
                w = 3
                break
            hh = h[m] if m > m0 else max(px, c[m])
            if l[m] <= sp[e]:
                out = min(sp[e], op)
                w = 1
                break
            if hh > tp1[e] + thr * abs(tp1[e]) or (thr == 0 and hh >= tp1[e]):
                out = tp1[e]
                w = 3
                break
            if m - m0 + 1 >= hold or m == n - 1:
                out = c[m]
                w = 0
                break
            m += 1
        net[e] = (out - px) / abs(px) - (cost + 0.0002) - f
        ji[e], xi[e], epx[e], xpx[e], why[e] = m0, m, px, out, w
    return net, ji, xi, epx, xpx, why, touched


def period(t):
    return np.where(t < C.T_IS, "hold_pre", np.where(t < C.T_END, "dev", "hold_post"))


def run_sym(sym):
    X = C.load(sym)
    rng_ = np.random.default_rng(zlib.crc32(sym.encode()) + SEED)
    rows, rnd, cnt = [], [], []
    elig5 = np.flatnonzero(X["elig"] & ~np.isnan(X["o"]))
    for tf, n5 in TFS.items():
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
            R = E10.setups(h, l, m, C.pivots(h, K, False), K)
            if not len(R):
                continue
            i = R[:, 0].astype(np.int64)
            ok = B["elig"][i] & (i + 1 < len(B["t"]))
            R, i = R[ok], i[ok]
            a0, a, b = (R[:, k].astype(np.int64) for k in (1, 2, 3))
            H0, L0, H1, M0, cross, zmax, run, emin, lh = (R[:, k] for k in range(4, 13))
            rng = H1 - L0
            sr = (b - a) / (a - a0)
            rd = rng / (H0 - L0)
            struct = (rd >= 0.382) & (rd < 1) & (sr >= 0.2) & (sr <= 1) & ((H0 - L0) >= 3 * atr[b])
            if not struct.any():
                continue
            sel = np.flatnonzero(struct)
            i, a0, a, b, H0, L0, H1, M0, cross, zmax, emin, lh, rng, sr, rd = (
                v[sel] for v in (i, a0, a, b, H0, L0, H1, M0, cross, zmax, emin, lh, rng, sr, rd))
            macd_ctx = (M0 < 0) & (cross == 1) & (zmax <= 0.25 * np.abs(M0))
            r3 = (rd <= 0.618) & (lh == 1) & (emin >= M0)
            body = np.maximum(bo[b], c[b])
            sp = L0 - STOP * rng
            lim = (body + RR * sp) / (1 + RR)
            x = (lim - L0) / rng
            okx = (x >= -0.15) & (x <= 0.5) & (body > lim)
            t_end = B["t"][i] + (n5 - 1) * C.M5
            j = (np.searchsorted(X["t"], t_end) + 1).astype(np.int64)
            jmax = np.where(okx, j + np.minimum(np.maximum(b - a, 6), E10.LOOK) * n5, j)
            fx = E8.divfail(c, m, i, L0, M0, len(c))
            jx = np.where(fx >= 0, np.searchsorted(X["t"], B["t"][np.clip(fx, 0, None)] + (n5 - 1) * C.M5) + 1, -1)
            jx = jx.astype(np.int64)
            nojx = np.full(len(i), -1, np.int64)
            hold = E8.HOLD * n5
            base = {"sym": sym, "tf": tf, "side": s, "t": B["t"][i], "t0": B["t"][a0], "ta": B["t"][a],
                    "tb": B["t"][b], "H0": s * H0, "L0": s * L0, "H1": s * H1, "M0": s * M0, "tp1": s * body,
                    "stop": s * sp, "lim": s * lim, "rd": rd, "sym_ratio": sr, "macd_ctx": macd_ctx, "r3": r3}  # fmt: skip
            for thr in THRS:
                for dfx in (0, 1):
                    if dfx and thr == 0:
                        continue
                    r = sim2(ho, hh, hl, hc, fund, j, jmax, lim, H1, L0, sp, jx if dfx else nojx, hold, C.COST, body,
                             thr)  # fmt: skip
                    d = pd.DataFrame(base)
                    d["thr"], d["divfail"] = thr, dfx
                    d["net"] = r[0]
                    d["t_in"] = np.where(r[1] >= 0, X["t"][np.clip(r[1], 0, None)], -1)
                    d["t_out"] = np.where(r[2] >= 0, X["t"][np.clip(r[2], 0, None)], -1)
                    d["px_in"], d["px_out"], d["why"], d["touched"] = s * r[3], s * r[4], r[5], r[6]
                    d = d[okx]
                    d["period"] = period(d.t.to_numpy())
                    d["set"] = np.where(d.macd_ctx & d.r3, "D", np.where(d.macd_ctx, "C", "B"))
                    cnt.append(d.groupby(["tf", "side", "thr", "divfail", "period", "set"]).agg(
                        setups=("net", "size"), filled=("t_in", lambda v: (v >= 0).sum()),
                        touched=("touched", "sum")).reset_index())
                    rows.append(d[d.t_in >= 0])
                    if thr == THRS[-1] and dfx == 0:
                        # A: random-timing controls for the D trades of this run
                        f = np.flatnonzero((r[1] >= 0) & macd_ctx & r3)
                        if not len(f):
                            continue
                        risk = np.abs(r[3][f] - sp[f]) / np.abs(r[3][f])
                        rew = np.abs(body[f] - r[3][f]) / np.abs(r[3][f])
                        t_in = X["t"][r[1][f]]
                        win = 90 * C.DAY
                        jr, rk, rw, src = [], [], [], []
                        for q in range(len(f)):
                            lo_, hi_ = np.searchsorted(X["t"][elig5], [t_in[q] - win, t_in[q] + win])
                            if hi_ - lo_ < 10:
                                continue
                            pick = elig5[rng_.integers(lo_, hi_, NRAND)]
                            jr += list(pick)
                            rk += [risk[q]] * NRAND
                            rw += [rew[q]] * NRAND
                            src += [q] * NRAND
                        if not jr:
                            continue
                        jr = np.array(jr, np.int64)
                        px = ho[jr]
                        rk, rw = np.array(rk), np.array(rw)
                        spr = px - rk * np.abs(px)
                        tpr = px + rw * np.abs(px)
                        big = np.full(len(jr), np.inf)
                        rr_ = sim2(ho, hh, hl, hc, fund, jr, jr + 1, px, big, spr, spr, np.full(len(jr), -1, np.int64),
                                   hold, C.COST, tpr, thr)  # fmt: skip
                        rnd.append(pd.DataFrame({"sym": sym, "tf": tf, "side": s, "t": X["t"][jr], "net": rr_[0],
                                                 "risk": rk + C.COST + 0.0002, "src_t": t_in[np.array(src)],
                                                 "why": rr_[5]}))  # fmt: skip
    D = pd.concat(rows) if rows else None
    A = pd.concat(rnd) if rnd else None
    N = pd.concat(cnt) if cnt else None
    return D, A, N


def main():
    (C.RESULTS / "E11_attrib").mkdir(parents=True, exist_ok=True)
    with mp.Pool(4) as pool:
        parts = pool.map(run_sym, C.coins(), chunksize=2)
    D = pd.concat([p[0] for p in parts if p[0] is not None], ignore_index=True)
    A = pd.concat([p[1] for p in parts if p[1] is not None], ignore_index=True)
    N = pd.concat([p[2] for p in parts if p[2] is not None]).groupby(
        ["tf", "side", "thr", "divfail", "period", "set"]).sum().reset_index()
    N.to_csv(C.RESULTS / "E11_attrib" / "fill_counts.csv", index=False)
    A["period"] = period(A.src_t.to_numpy())
    D["risk"] = (D.px_in - D.stop).abs() / D.px_in + C.COST + 0.0002
    D["R"] = D.net / D.risk
    A["R"] = A.net / A.risk
    D.to_pickle("data/research/divergence_e11.pkl")
    A.to_pickle("data/research/divergence_e11_random.pkl")
    print(D.groupby(["period", "tf"]).size().unstack())


if __name__ == "__main__":
    main()
