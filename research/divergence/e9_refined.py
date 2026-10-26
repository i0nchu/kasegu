"""E9: E8 with the user's second round of corrections.

- TP1 at the body edge of the neckline pivot candle (long: max(open, close) of the H1 bar), not its wick.
  Exits: "body" all at TP1; "body_half" half at TP1, stop to entry, rest to the measured move H1 + (H1 - L0);
  "wick" all at H1 (E8, for comparison).
- Correction time: the L0 -> H1 swing must take 0.2-1.0x the bars of the H0 -> L0 leg (E8 allowed 0.2-5x);
  a swing as long as the leg (e.g. DOGE 2024-05-20, 1.28x) rarely gives a double top or divergence.
- Timeframes 15m / 1h / 4h. Stop L0 - 0.5 * (H1 - L0), "no divergence" exit on (E8 defaults).
Usage: PYTHONPATH=research/divergence python research/divergence/e9_refined.py
"""

import multiprocessing as mp

import numpy as np
import pandas as pd

import common as C
import d3_strategy as D3
import e8_wbottom as E8

TFS = {"15m": 3, "1h": 12, "4h": 48}
K = 3
XS = (0.5, 0.25, 0.0, -0.15)
STOP = 0.5
SYM = (0.2, 1.0)


def run_sym(sym):
    X = C.load(sym)
    rows = []
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
            R = E8.setups(h, l, m, atr, C.pivots(h, K, False), K)
            if not len(R):
                continue
            i = R[:, 0].astype(np.int64)
            ok = B["elig"][i] & (B["t"][i] >= C.T_IS) & (i + 1 < len(B["t"]))
            R, i = R[ok], i[ok]
            a0, a, b = (R[:, k].astype(np.int64) for k in (1, 2, 3))
            H0, L0, H1, M0, cross, zmax = (R[:, k] for k in range(4, 10))
            rng = H1 - L0
            sr = (b - a) / (a - a0)
            keep = (M0 < 0) & (cross == 1) & (rng / (H0 - L0) >= 0.382) & (rng / (H0 - L0) < 1) & (zmax <= 0.25 * np.abs(M0))
            keep &= ((H0 - L0) >= 3 * atr[b]) & (sr >= SYM[0]) & (sr <= SYM[1])
            if not keep.any():
                continue
            R, i, a0, a, b, H0, L0, H1, M0, rng, sr = (v[keep] for v in (R, i, a0, a, b, H0, L0, H1, M0, rng, sr))
            body = np.maximum(bo[b], c[b])  # body top of the neckline candle (mirrored for shorts)
            t_end = B["t"][i] + (n5 - 1) * C.M5
            j = (np.searchsorted(X["t"], t_end) + 1).astype(np.int64)
            jmax = j + np.minimum(np.maximum(b - a, 6), E8.LOOK) * n5
            fx = E8.divfail(c, m, i, L0, M0, len(c))
            jx = np.where(fx >= 0, np.searchsorted(X["t"], B["t"][np.clip(fx, 0, None)] + (n5 - 1) * C.M5) + 1, -1)
            jx = jx.astype(np.int64)
            sp = L0 - STOP * rng
            base = {"sym": sym, "tf": tf, "side": s, "t": B["t"][i], "t0": B["t"][a0], "ta": B["t"][a], "tb": B["t"][b],
                    "H0": s * H0, "L0": s * L0, "H1": s * H1, "body": s * body, "M0": s * M0, "sym_ratio": sr,
                    "stop": s * sp}  # fmt: skip
            nan = np.full(len(i), np.nan)
            for x in XS:
                lim = L0 + x * rng
                for ex, tp1, t2 in (("body", body, nan), ("body_half", body, H1 + rng), ("wick", H1, nan)):
                    tp = np.where(tp1 > lim, tp1, np.nan)  # a body below the entry leaves no target: skip
                    r = E8.sim(ho, hh, hl, hc, fund, j, np.where(tp == tp, jmax, j), lim, H1, L0, sp, t2, jx,
                               E8.HOLD * n5, C.COST, tp)  # fmt: skip
                    d = pd.DataFrame(base)
                    d["x"], d["exit"], d["lim"], d["tp1"] = x, ex, s * lim, s * tp1
                    d["net"] = r[0]
                    d["t_in"] = np.where(r[1] >= 0, X["t"][np.clip(r[1], 0, None)], -1)
                    d["t_out"] = np.where(r[2] >= 0, X["t"][np.clip(r[2], 0, None)], -1)
                    d["px_in"], d["px_out"] = s * r[3], s * r[4]
                    d["why"], d["broke_L0"] = r[5], r[6]
                    rows.append(d[d.t_in >= 0])
    return pd.concat(rows) if rows else None


def prep(D):
    D = D.copy()
    D["risk"] = (D.px_in - D.stop).abs() / D.px_in + C.COST + 0.0002
    D["R"] = D.net / D.risk
    D["reward"] = (D.tp1 - D.px_in).abs() / D.px_in
    return D


def gap(x):
    """Distance to break even: mean R, the hit-rate change that would make it 0, and the same in bp per trade."""
    win, loss = x.R[x.R > 0].mean(), x.R[x.R <= 0].mean()
    mu = x.R.mean()
    return {"n": len(x), "win": (x.R > 0).mean(), "R": mu, "avg_win_R": win, "avg_loss_R": loss,
            "gap_win_pp": 100 * max(-mu, 0) / (win - loss), "net_bp": 1e4 * x.net.mean(),
            "gap_bp": 1e4 * max(-x.net.mean(), 0)}  # fmt: skip


def main():
    with mp.Pool(4) as pool:
        parts = pool.map(run_sym, C.coins(), chunksize=2)
    D = prep(pd.concat([p for p in parts if p is not None], ignore_index=True))
    D.to_pickle("data/research/divergence_e9.pkl")
    rows = []
    for key, g in D.groupby(["tf", "side", "x", "exit"]):
        for tag, pm in (("is", g.t < C.T_OOS), ("oos", g.t >= C.T_OOS)):
            rows.append(dict(zip(["tf", "side", "x", "exit"], key)) | {"per": tag} | gap(g[pm]))
    S = pd.DataFrame(rows)
    out = C.RESULTS / "E9_refined"
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
