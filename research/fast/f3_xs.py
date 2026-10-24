"""F3: cross-sectional short-horizon momentum and reversal on 1h bars, point-in-time universe.

Every R hours, rank the coins eligible that day by their past L-hour return. mom: long the top k,
short the bottom k; rev: the opposite; also long-only top-k (mom) and bottom-k (rev) legs. Equal
weight, gross exposure 1 (0.5 per side for long/short), held R hours. Turnover pays 0.05% fee +
0.05% slippage per unit traded; funding is paid on the weights held over each funding time.

Usage: PYTHONPATH=research/momentum;research/fast uv run --with numba python research/fast/f3_xs.py
"""

import itertools

import numpy as np
import pandas as pd

import fast as F

H = 3_600_000
T0 = int(pd.Timestamp("2020-06-01", tz="UTC").value // 10**6)
GRID = np.arange(T0, F.T_END, H, dtype=np.int64)


def panel_1h() -> dict:
    p = F.OUT / "panel_1h.npz"
    if not p.exists():
        syms = F.coins()
        o, c, e, f = (np.full((len(GRID), len(syms)), np.nan) for _ in range(4))
        for j, s in enumerate(syms):
            X = F.load(s)
            if X is None:
                continue
            B = F.resample(X, 12)
            i = (B["t"] - T0) // H
            ok = (i >= 0) & (i < len(GRID))
            o[i[ok], j], c[i[ok], j] = B["o"][ok], B["c"][ok]
            e[i[ok], j], f[i[ok], j] = B["elig"][ok], B["fund"][ok]
        np.savez(p, o=o, c=c, e=e, f=f, names=np.array(syms))
    z = np.load(p)
    return {k: z[k] for k in z.files}


def backtest(P, L, R, k, mode, side, cost=F.FEE + F.SLIP):
    c = P["c"]
    ret = np.nan_to_num(np.vstack([np.full((1, c.shape[1]), np.nan), c[1:] / c[:-1] - 1]))
    past = c / np.vstack([np.full((L, c.shape[1]), np.nan), c[:-L]]) - 1
    ok = (P["e"] == 1) & np.isfinite(past) & np.isfinite(c)
    W = np.zeros_like(c)
    cur = np.zeros(c.shape[1])
    for i in range(L, len(c)):
        if i % R == 0:
            cur = np.zeros(c.shape[1])
            idx = np.nonzero(ok[i])[0]
            if len(idx) >= 4 * k:
                order = idx[np.argsort(past[i, idx])]
                top, bot = order[-k:], order[:k]
                if mode == "rev":
                    top, bot = bot, top
                if side == "ls":
                    cur[top], cur[bot] = 0.5 / k, -0.5 / k
                else:
                    cur[top] = 1.0 / k
        cur = np.where(np.isfinite(c[i]), cur, 0.0)  # a coin whose data stops is closed at its last close
        W[i] = cur
    held = np.vstack([np.zeros((1, c.shape[1])), W[:-1]])  # decided at close i, held over bar i+1
    turn = np.abs(np.diff(np.vstack([np.zeros((1, c.shape[1])), W]), axis=0)).sum(1)
    fund = np.nan_to_num(P["f"])
    r = (held * ret).sum(1) - (held * fund).sum(1) - cost * np.concatenate([[0], turn[:-1]])
    return pd.Series(r, index=GRID), turn.mean()


def main():
    P = panel_1h()
    out = F.RESULTS / "F3_xs"
    out.mkdir(parents=True, exist_ok=True)
    rows, daily = [], {}
    for L, R, k, mode, side in itertools.product((1, 4, 24), (1, 4, 24), (5,), ("mom", "rev"), ("ls", "long")):
        if R > L * 6:
            continue
        r, to = backtest(P, L, R, k, mode, side)
        g, _ = backtest(P, L, R, k, mode, side, cost=0.0)
        d = r.groupby(r.index // F.DAY * F.DAY).sum()
        dg = g.groupby(g.index // F.DAY * F.DAY).sum()
        name = f"{mode}_{side}_L{L}_R{R}"
        daily[name] = d
        row = {"name": name, "turnover_per_h": to}
        for tag, m in (("is", (d.index >= F.T_IS) & (d.index < F.T_OOS)), ("oos", d.index >= F.T_OOS)):
            row[f"{tag}_ann"] = d[m].mean() * 365
            row[f"{tag}_gross_ann"] = dg[m].mean() * 365
            row[f"{tag}_sharpe"] = d[m].mean() / d[m].std() * np.sqrt(365)
        rows.append(row)
        print(row, flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False, float_format="%.4g")
    pd.DataFrame(daily).to_csv(out / "daily_returns.csv.gz", float_format="%.6g")
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
