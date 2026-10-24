"""F4: big-mover continuation, the one F1 family with a positive net edge in both samples, widened.

Event: a coin's return over the last H hours first crosses above thr (long) or below -thr (short).
  anchor utc   the window is 00:00 UTC to hour H (F1 "im"), checked once a day at hour H
  anchor roll  a rolling H-hour window, checked on every 5m close (the first crossing after the
               return was below thr starts a trade; while in a trade new crossings are ignored)
Optional volume filter: the H-hour quote volume above v x its trailing 7-day mean per H hours.
Exit: time exit after X hours, optionally a trailing stop at trail x thr below the best close.

Inference is by day: trades of one day are averaged, then a t-stat over days (one pump day
lifts dozens of coins together, so per-trade t-stats overstate). Costs: 0.05% fee + slip per side
(base 0.05%, also 0.10% and 0.15% for the illiquid hours right after a pump) plus funding.

Usage: PYTHONPATH=research/momentum;research/fast uv run --with numba python research/fast/f4_movers.py
"""

import concurrent.futures as cf
import itertools

import numpy as np
import pandas as pd

import fast as F

LONG = [
    dict(side=1, anchor=a, H=H, thr=thr, vol=v, X=X, trail=tr)
    for a, H, thr, v, X, tr in itertools.product(
        ("utc", "roll"), (2, 4, 6, 12), (0.03, 0.05, 0.08, 0.12), (0, 3), (12, 24, 48), (0, 0.5, 1.0)
    )
]
SHORT = [
    dict(side=-1, anchor=a, H=H, thr=thr, vol=0, X=24, trail=tr)
    for a, H, thr, tr in itertools.product(("utc", "roll"), (2, 4, 6, 12), (0.03, 0.05, 0.08, 0.12), (0, 1.0))
]
CONFIGS = LONG + SHORT
SLIPS = (0.0005, 0.0010, 0.0015)


def signal(X, cfg, cache):
    H, thr, s = cfg["H"], cfg["thr"], cfg["side"]
    n = H * 12
    key = (cfg["anchor"], H)
    if key not in cache:
        c, t = X["c"], X["t"]
        if cfg["anchor"] == "utc":
            sid = t // F.DAY
            pos = (t % F.DAY) // F.M5
            o0 = pd.Series(X["o"]).groupby(sid).transform("first").to_numpy()
            r = np.where(pos == n - 1, c / o0 - 1, np.nan)
        else:
            r = c / pd.Series(c).shift(n).to_numpy() - 1
        qv = pd.Series(X["qv"]).rolling(n, min_periods=n).sum()
        base = qv.shift(n).rolling(7 * 24 // H, min_periods=24 // H).mean()  # sampled every bar
        cache[key] = (r, (qv / base).to_numpy())
    r, vr = cache[key]
    with np.errstate(invalid="ignore"):
        hit = s * r > thr
        if cfg["anchor"] == "roll":
            hit = hit & ~np.concatenate([[False], hit[:-1]])
        if cfg["vol"]:
            hit = hit & (vr > cfg["vol"])
    return np.where(hit, s, 0)


def run_coin(sym):
    X = F.load(sym)
    if X is None:
        return []
    out, cache = [], {}
    for k, cfg in enumerate(CONFIGS):
        sig = signal(X, cfg, cache)
        tr = cfg["trail"] * cfg["thr"] if cfg["trail"] else np.nan
        for slip in SLIPS:
            d = F.trades(X, sig, np.nan, np.nan, cfg["X"] * 12, cost=F.FEE + slip, trail=tr)
            g = d.groupby(d.t // F.DAY).net.agg(["sum", "count"])
            g["win"] = (d.net > 0).groupby(d.t // F.DAY).sum()
            g["cfg"], g["slip"] = k, slip
            out.append(g.reset_index(names="day"))
    return [pd.concat(out)] if out else []  # per config, slip and entry day: summed net, trades, wins


def day_stats(d: pd.DataFrame, tag: str) -> dict:
    d = d.groupby("day")[["sum", "count", "win"]].sum()
    g = d["sum"] / d["count"]
    n = d["count"].sum()
    return {
        f"{tag}_n": n,
        f"{tag}_days": len(g),
        f"{tag}_bp": 1e4 * d["sum"].sum() / n if n else np.nan,
        f"{tag}_win": d["win"].sum() / n if n else np.nan,
        f"{tag}_day_bp": 1e4 * g.mean() if len(g) else np.nan,
        f"{tag}_day_t": g.mean() / g.std() * np.sqrt(len(g)) if len(g) > 2 else np.nan,
    }


def main():
    parts = []
    with cf.ProcessPoolExecutor(12) as ex:
        for r in ex.map(run_coin, F.coins(), chunksize=4):
            parts += r
    T = pd.concat(parts, ignore_index=True)
    T.to_pickle(F.OUT / "F4_days.pkl")
    out = F.RESULTS / "F4_movers"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for (k, slip), d in T.groupby(["cfg", "slip"]):
        row = {**CONFIGS[k], "cfg": k, "slip": slip}
        t = d.day * F.DAY
        row.update(day_stats(d[(t >= F.T_IS) & (t < F.T_OOS)], "is"))
        row.update(day_stats(d[t >= F.T_OOS], "oos"))
        rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250, "display.max_rows", 500)
    b = S[S.slip == 0.0005].sort_values("is_day_t", ascending=False)
    print(b.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
