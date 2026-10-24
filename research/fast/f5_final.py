"""F5: the converged candidates as accounts, and what leverage does to doubling and ruin odds.

Candidates (F4 configs, base slippage):
  429  best of all 640 F4 configs by in-sample day t-stat: rolling 4h return first above +12% with
       4h volume > 3x normal, buy next 5m open, sell 48h later.
  177  the UTC-anchored neighbour that is positive in both samples: up > 5% from 00:00 UTC at
       06:00 with 3x volume, buy, sell 48h later.
Account: each trade gets a fixed fraction w = 10% of equity (notional), at most 10 open at once
(later signals are skipped while 10 are open), so gross exposure is at most 1x at multiple 1.
P&L is marked to market daily on 5m closes. For a leverage multiple m every position is m times
larger; equity <= 0 is ruin. Bootstrap as E6: 10,000 one-year paths, stationary blocks of 20
days, from the 2024-01 to 2026-09 daily returns.

Usage: PYTHONPATH=research/momentum;research/fast uv run --with numba python research/fast/f5_final.py
"""

import concurrent.futures as cf

import numpy as np
import pandas as pd

import e6_leverage as E6
import f4_movers as F4
import fast as F

CANDS = {429: "roll4h_12pct_vol3_48h", 177: "utc6h_5pct_vol3_48h"}
W = 0.10
MAXPOS = 10


def coin_trades(sym):
    X = F.load(sym)
    if X is None:
        return []
    out, cache = [], {}
    for k in CANDS:
        cfg = F4.CONFIGS[k]
        d = F.trades(X, F4.signal(X, cfg, cache), np.nan, np.nan, cfg["X"] * 12)
        if len(d) == 0:
            continue
        # daily mark-to-market path of each trade: cumulative gross return at each UTC day close
        c, t = X["c"], X["t"]
        for r in d.itertuples():
            i, x = int(r.i), int(r.x)
            px = X["o"][i]
            days = np.unique(t[i : x + 1] // F.DAY)
            marks = []
            for dd in days:
                last = min(x, np.searchsorted(t, (dd + 1) * F.DAY) - 1)
                v = r.side * (c[last] / px - 1) if last < x else r.net  # final mark is the net result
                marks.append(v)
            out.append({"cfg": k, "sym": sym, "t": r.t, "tx": r.tx, "net": r.net, "days": days, "marks": marks})
    return out


def account(trs: list, w=W, maxpos=MAXPOS) -> pd.Series:
    """Daily account returns at multiple 1 (no compounding of a trade's size after entry)."""
    trs = sorted(trs, key=lambda r: r["t"])
    open_until, taken = [], []
    for r in trs:
        open_until = [x for x in open_until if x > r["t"]]
        if len(open_until) < maxpos:
            open_until.append(r["tx"])
            taken.append(r)
    idx = np.arange(F.T_IS // F.DAY, F.T_END // F.DAY)
    pnl = pd.Series(0.0, index=idx)
    for r in taken:
        prev = 0.0
        for dd, m in zip(r["days"], r["marks"]):
            if dd in pnl.index:
                pnl[dd] += w * (m - prev)
            prev = m
    pnl.index = pd.to_datetime(pnl.index * F.DAY, unit="ms")
    return pnl, len(taken)


def main():
    rows = []
    with cf.ProcessPoolExecutor(12) as ex:
        for r in ex.map(coin_trades, F.coins(), chunksize=4):
            rows += r
    out = F.RESULTS / "F5_final"
    out.mkdir(parents=True, exist_ok=True)
    daily, res = {}, []
    for k, name in CANDS.items():
        r, n = account([x for x in rows if x["cfg"] == k])
        daily[name] = r
        oos = r[r.index >= "2024-01-01"].to_numpy()
        ins = r[(r.index >= "2021-01-01") & (r.index < "2024-01-01")].to_numpy()
        idx = E6.bootstrap_idx(len(oos))
        mu, var = oos.mean(), oos.var()
        print(f"{name}: trades {n}, IS ann {ins.mean() * 365:.3f} sharpe {ins.mean() / ins.std() * 365**0.5:.2f}, "
              f"OOS ann {mu * 365:.3f} sharpe {mu / oos.std() * 365**0.5:.2f}, Kelly x{mu / var:.1f}", flush=True)  # fmt: skip
        for m in (1, 2, 3, 5, 10, 20):
            eq = E6.paths(oos, m, idx)
            hist = np.cumprod(np.maximum(1 + m * oos, 0))
            res.append({
                "candidate": name, "mult": m, "max_gross": m * W * MAXPOS,
                "ann_vol": round(oos.std() * 365**0.5 * m, 3),
                "p_double_90d": round(float((eq[:, :90].max(1) >= 2).mean()), 3),
                "p_double_365d": round(float((eq.max(1) >= 2).mean()), 3),
                "p_lose50": round(float((eq.min(1) <= 0.5).mean()), 3),
                "p_lose90": round(float((eq.min(1) <= 0.1).mean()), 3),
                "median_1y": round(float(np.median(eq[:, -1])), 2),
                "p5_1y": round(float(np.percentile(eq[:, -1], 5)), 2),
                "oos_hist_multiple": round(float(hist[-1]), 2),
                "oos_hist_mdd": round(float((hist / np.maximum.accumulate(hist) - 1).min()), 3),
            })  # fmt: skip
    pd.DataFrame(daily).to_csv(out / "daily_returns.csv.gz", float_format="%.6g")
    df = pd.DataFrame(res)
    df.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
